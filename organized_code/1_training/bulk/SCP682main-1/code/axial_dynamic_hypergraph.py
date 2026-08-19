"""Protein-anchored axial dynamic pathway graph for phosphosite residuals.

The total-protein matrix is an immutable model input.  This module never owns
or trains an RNA-to-protein predictor, so phosphosite gradients cannot alter
the parent-protein anchor.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(device=logits.device, dtype=torch.bool)
    weights = torch.softmax(logits.masked_fill(~mask, -1.0e4), dim=dim)
    weights = weights * mask.to(weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1.0e-8)


def _logit(probability: float) -> float:
    value = min(max(float(probability), 1.0e-4), 1.0 - 1.0e-4)
    return math.log(value / (1.0 - value))


def project_fixed_vocabulary_zero_median(
    values: torch.Tensor,
    centering_site_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project every row over a fixed, label-independent site vocabulary."""
    center_values = (
        values
        if centering_site_index is None
        else values.index_select(1, centering_site_index.to(values.device))
    )
    median = torch.nanquantile(center_values.float(), 0.5, dim=1, keepdim=True)
    return values - median.to(values.dtype)


@dataclass(frozen=True)
class AxialHypergraphConfig:
    n_rna: int
    n_proteins: int
    n_sites: int
    n_pathways: int
    n_kinases: int
    hidden: int = 64
    heads: int = 4
    axial_layers: int = 2
    pathway_adapter_rank: int = 8
    dropout: float = 0.10
    site_chunk_size: int = 512
    initial_site_shrinkage: float = 0.10

    def __post_init__(self) -> None:
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")
        if self.axial_layers < 1:
            raise ValueError("axial_layers must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MaskedPathwayPool(nn.Module):
    """Pool typed molecular values into pathway tokens."""

    def __init__(
        self,
        vocabulary_size: int,
        n_pathways: int,
        hidden: int,
        member_index: torch.Tensor,
        member_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.n_pathways = int(n_pathways)
        self.register_buffer("member_index", member_index.long())
        self.register_buffer("member_mask", member_mask.bool())
        self.identity = nn.Embedding(vocabulary_size, hidden)
        self.value_projection = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden, bias=False)
        )
        self.pathway_query = nn.Parameter(torch.empty(n_pathways, hidden))
        nn.init.normal_(self.identity.weight, std=0.02)
        nn.init.normal_(self.pathway_query, std=0.02)

    def forward(
        self,
        values: torch.Tensor,
        value_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected = values[:, self.member_index]
        token = self.identity(self.member_index).unsqueeze(0)
        token = token + self.value_projection(selected.unsqueeze(-1))
        mask = self.member_mask.unsqueeze(0).expand(values.shape[0], -1, -1)
        if value_mask is not None:
            mask = mask & value_mask[:, self.member_index].bool()
        query = self.pathway_query.view(1, self.n_pathways, 1, self.hidden)
        logits = (token * query).sum(dim=-1) / math.sqrt(self.hidden)
        attention = _masked_softmax(logits, mask, dim=2)
        pooled = (attention.unsqueeze(-1) * token).sum(dim=2)

        mask_f = mask.to(selected.dtype)
        count = mask_f.sum(dim=2).clamp_min(1.0)
        mean = (selected * mask_f).sum(dim=2) / count
        variance = ((selected - mean.unsqueeze(-1)).square() * mask_f).sum(dim=2) / count
        moments = torch.stack([mean, variance.clamp_min(1.0e-8).sqrt()], dim=-1)
        return pooled, moments


class MultimodalPathwayEncoder(nn.Module):
    """Combine RNA ranks and frozen predicted total proteins per pathway."""

    def __init__(
        self,
        config: AxialHypergraphConfig,
        rna_pathway_index: torch.Tensor,
        rna_pathway_mask: torch.Tensor,
        protein_pathway_index: torch.Tensor,
        protein_pathway_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.rna_pool = MaskedPathwayPool(
            config.n_rna,
            config.n_pathways,
            config.hidden,
            rna_pathway_index,
            rna_pathway_mask,
        )
        self.protein_pool = MaskedPathwayPool(
            config.n_proteins,
            config.n_pathways,
            config.hidden,
            protein_pathway_index,
            protein_pathway_mask,
        )
        self.pathway_identity = nn.Embedding(config.n_pathways, config.hidden)
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden * 2 + 4, config.hidden),
            nn.LayerNorm(config.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.hidden),
            nn.LayerNorm(config.hidden),
        )
        nn.init.normal_(self.pathway_identity.weight, std=0.02)

    def forward(
        self,
        rna_rank: torch.Tensor,
        protein_prediction: torch.Tensor,
        rna_mask: torch.Tensor | None = None,
        protein_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rna_state, rna_moments = self.rna_pool(rna_rank, rna_mask)
        protein_state, protein_moments = self.protein_pool(
            protein_prediction.detach(), protein_mask
        )
        summary = torch.cat([rna_moments, protein_moments], dim=-1)
        fusion_input = torch.cat([rna_state, protein_state, summary], dim=-1)
        state = self.fusion(fusion_input)
        pathway_ids = torch.arange(state.shape[1], device=state.device)
        return state + self.pathway_identity(pathway_ids).unsqueeze(0)


class PathwayAxisLayer(nn.Module):
    """Exchange information along the pathway axis inside each sample."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        dropout: float,
        pathway_relation_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        if pathway_relation_mask.ndim != 2 or pathway_relation_mask.shape[0] != pathway_relation_mask.shape[1]:
            raise ValueError("pathway_relation_mask must be square")
        self.register_buffer("pathway_relation_mask", pathway_relation_mask.bool())
        self.norm1 = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(state)
        # MultiheadAttention uses True to block an edge.
        blocked = ~self.pathway_relation_mask
        message, _ = self.attention(
            normalized, normalized, normalized, attn_mask=blocked, need_weights=False
        )
        state = state + self.dropout(message)
        return state + self.dropout(self.ffn(self.norm2(state)))


class DynamicPathwaySampleLayer(nn.Module):
    """Learn pathway-specific dynamic weights inside a shared RNA candidate graph."""

    def __init__(
        self,
        n_pathways: int,
        hidden: int,
        heads: int,
        adapter_rank: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_pathways = int(n_pathways)
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.head_dim = hidden // heads
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden * 3 + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, heads),
        )
        self.output = nn.Linear(hidden, hidden, bias=False)
        self.adapter_down = nn.Parameter(torch.empty(n_pathways, adapter_rank, hidden))
        self.adapter_up = nn.Parameter(torch.zeros(n_pathways, hidden, adapter_rank))
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 3, hidden),
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.adapter_down, std=0.01)

    def _gather(
        self, source_state: torch.Tensor, neighbour_index: torch.Tensor
    ) -> torch.Tensor:
        if neighbour_index.ndim != 3:
            raise ValueError("neighbour_index must have shape [pathway, query, k]")
        n_pathways, batch, k = neighbour_index.shape
        if n_pathways != self.n_pathways:
            raise ValueError("neighbour_index pathway count differs from the model")
        n_source = source_state.shape[0]
        source_flat = source_state.permute(1, 0, 2).reshape(
            self.n_pathways * n_source, self.hidden
        )
        offsets = torch.arange(self.n_pathways, device=source_state.device).view(-1, 1, 1)
        flat_index = neighbour_index.long() + offsets * n_source
        gathered = source_flat.index_select(0, flat_index.reshape(-1))
        return gathered.reshape(self.n_pathways, batch, k, self.hidden).permute(1, 0, 2, 3)

    def forward(
        self,
        query_state: torch.Tensor,
        source_state: torch.Tensor,
        neighbour_index: torch.Tensor,
        neighbour_similarity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_state.shape[1:] != (self.n_pathways, self.hidden):
            raise ValueError("query pathway-state shape differs from the model")
        if neighbour_similarity.shape != neighbour_index.shape:
            raise ValueError("neighbour similarity and index shapes differ")
        neighbours = self._gather(source_state, neighbour_index)
        normalized_query = self.norm1(query_state)
        q_full = self.query(normalized_query)
        k_full = self.key(neighbours)
        v_full = self.value(neighbours)
        q = q_full.view(*q_full.shape[:-1], self.heads, self.head_dim).unsqueeze(2)
        key = k_full.view(*k_full.shape[:-1], self.heads, self.head_dim)
        value = v_full.view(*v_full.shape[:-1], self.heads, self.head_dim)
        dot = (q * key).sum(dim=-1) / math.sqrt(self.head_dim)

        query_expanded = normalized_query.unsqueeze(2).expand_as(neighbours)
        edge_features = torch.cat(
            [
                query_expanded,
                neighbours,
                (query_expanded - neighbours).abs(),
                neighbour_similarity.permute(1, 0, 2).unsqueeze(-1),
            ],
            dim=-1,
        )
        learned_bias = self.edge_gate(edge_features)
        prior = ((neighbour_similarity.permute(1, 0, 2) + 1.0) * 0.5)
        prior = prior.clamp_min(1.0e-4).log().unsqueeze(-1)
        attention = torch.softmax(dot + learned_bias + prior, dim=2)
        attention = self.dropout(attention)
        message = (attention.unsqueeze(-1) * value).sum(dim=2).reshape(
            query_state.shape[0], self.n_pathways, self.hidden
        )
        base_message = self.output(message)
        low_rank = torch.einsum("bph,prh->bpr", message, self.adapter_down)
        adapted = torch.einsum("bpr,phr->bph", low_rank, self.adapter_up)
        state = query_state + self.dropout(base_message + adapted)
        state = state + self.dropout(self.ffn(self.norm2(state)))
        # Returned as [batch, pathway, head, k] for audit tables.
        return state, attention.permute(0, 1, 3, 2).contiguous()


class HeterogeneousSiteHypergraphDecoder(nn.Module):
    """Decode parent/site/kinase/pathway hyperedges into centered residuals."""

    def __init__(
        self,
        config: AxialHypergraphConfig,
        site_pathway_index: torch.Tensor,
        site_pathway_mask: torch.Tensor,
        site_pathway_weight: torch.Tensor,
        parent_protein_index: torch.Tensor,
        parent_protein_mask: torch.Tensor,
        site_kinase_index: torch.Tensor,
        site_kinase_mask: torch.Tensor,
        site_coverage: torch.Tensor,
        site_anchor_quality: torch.Tensor,
        site_anchor_coverage: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("site_pathway_index", site_pathway_index.long())
        self.register_buffer("site_pathway_mask", site_pathway_mask.bool())
        self.register_buffer("site_pathway_weight", site_pathway_weight.float())
        self.register_buffer("parent_protein_index", parent_protein_index.long())
        self.register_buffer("parent_protein_mask", parent_protein_mask.bool())
        self.register_buffer("site_kinase_index", site_kinase_index.long())
        self.register_buffer("site_kinase_mask", site_kinase_mask.bool())
        self.register_buffer("site_coverage", site_coverage.float())
        self.register_buffer("site_anchor_quality", site_anchor_quality.float())
        self.register_buffer("site_anchor_coverage", site_anchor_coverage.float())

        hidden = config.hidden
        self.site_embedding = nn.Embedding(config.n_sites, hidden)
        self.parent_embedding = nn.Embedding(config.n_proteins + 1, hidden, padding_idx=config.n_proteins)
        self.kinase_embedding = nn.Embedding(config.n_kinases + 1, hidden, padding_idx=0)
        self.parent_value = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.coverage_projection = nn.Sequential(nn.Linear(1, hidden), nn.GELU())
        self.memory_projection = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.pathway_key = nn.Linear(hidden, hidden, bias=False)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.type_gate = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 2 + 7, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.site_shrinkage_logit = nn.Parameter(
            torch.full((config.n_sites,), _logit(config.initial_site_shrinkage))
        )
        self.register_buffer(
            "centering_site_index",
            torch.arange(config.n_sites, dtype=torch.long),
            persistent=False,
        )

    def set_centering_site_index(self, index: torch.Tensor) -> None:
        index = torch.as_tensor(index, dtype=torch.long, device=self.site_embedding.weight.device)
        if index.ndim != 1 or index.numel() < 1:
            raise ValueError("centering site index must be a non-empty vector")
        if int(index.min()) < 0 or int(index.max()) >= self.config.n_sites:
            raise IndexError("centering site index falls outside the output vocabulary")
        if torch.unique(index).numel() != index.numel():
            raise ValueError("centering site index contains duplicates")
        self.centering_site_index = index

    def _static_query(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.site_embedding.weight.device
        site_ids = torch.arange(start, end, device=device)
        site = self.site_embedding(site_ids)
        parent_index = self.parent_protein_index[start:end]
        safe_parent = torch.where(
            self.parent_protein_mask[start:end],
            parent_index,
            torch.full_like(parent_index, self.config.n_proteins),
        )
        parent = self.parent_embedding(safe_parent)
        kinase_mask = self.site_kinase_mask[start:end]
        kinase = self.kinase_embedding(self.site_kinase_index[start:end])
        kinase_f = kinase_mask.to(kinase.dtype).unsqueeze(-1)
        kinase = (kinase * kinase_f).sum(dim=1) / kinase_f.sum(dim=1).clamp_min(1.0)
        coverage = self.coverage_projection(self.site_coverage[start:end, None])
        return site + coverage, parent, kinase

    def _residual_memory_chunk(
        self,
        start: int,
        end: int,
        sample_attention: torch.Tensor,
        neighbour_index: torch.Tensor,
        reference_residual: torch.Tensor,
        reference_residual_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate label memory with pathway-specific graph weights.

        Reference values are detached and are only introduced after graph
        attention has been computed from query RNA and predicted protein.
        """
        attention = sample_attention.detach() if not sample_attention.requires_grad else sample_attention
        attention = attention.mean(dim=2)
        residual = reference_residual.detach()
        residual_mask = reference_residual_mask.detach().bool()
        if residual.ndim != 2 or residual.shape != residual_mask.shape:
            raise ValueError("reference residual and mask must share shape [reference, site]")
        if residual.shape[1] != self.config.n_sites:
            raise ValueError("reference residual site order differs from the decoder")
        if neighbour_index.ndim != 3 or neighbour_index.shape[:2] != (
            self.config.n_pathways,
            attention.shape[0],
        ):
            raise ValueError("pathway neighbour index has an incompatible shape")
        if neighbour_index.numel() and (
            int(neighbour_index.min()) < 0
            or int(neighbour_index.max()) >= residual.shape[0]
        ):
            raise IndexError("neighbour index falls outside the ordered residual reference library")
        batch = attention.shape[0]
        width = end - start
        numerator = attention.new_zeros((batch, width))
        denominator = attention.new_zeros((batch, width))
        possible = attention.new_zeros((batch, width))
        site_ids = torch.arange(start, end, device=attention.device)
        pathway_index = self.site_pathway_index[start:end]
        pathway_mask = self.site_pathway_mask[start:end]
        pathway_weight = self.site_pathway_weight[start:end]
        for relation in range(pathway_index.shape[1]):
            relation_valid = pathway_mask[:, relation]
            if not bool(relation_valid.any()):
                continue
            pathway = pathway_index[:, relation]
            selected_neighbour = neighbour_index[pathway, :, :].permute(1, 0, 2)
            selected_attention = attention[:, pathway, :]
            selected_value = residual[selected_neighbour, site_ids.view(1, -1, 1)]
            selected_mask = residual_mask[selected_neighbour, site_ids.view(1, -1, 1)]
            relation_weight = (
                pathway_weight[:, relation] * relation_valid.to(pathway_weight.dtype)
            ).view(1, width, 1)
            weight = selected_attention * relation_weight
            numerator = numerator + (weight * selected_value * selected_mask).sum(dim=2)
            denominator = denominator + (weight * selected_mask).sum(dim=2)
            possible = possible + weight.sum(dim=2)
        memory = numerator / denominator.clamp_min(1.0e-8)
        memory = torch.where(denominator > 0, memory, torch.zeros_like(memory))
        availability = denominator / possible.clamp_min(1.0e-8)
        availability = torch.where(possible > 0, availability, torch.zeros_like(availability))
        return memory, availability.clamp(0.0, 1.0)

    def forward(
        self,
        pathway_state: torch.Tensor,
        protein_prediction: torch.Tensor,
        centered_baseline: torch.Tensor,
        sample_attention: torch.Tensor,
        neighbour_index: torch.Tensor,
        reference_residual: torch.Tensor,
        reference_residual_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        protein_prediction = protein_prediction.detach()
        centered_baseline = centered_baseline.detach()
        raw_parts: list[torch.Tensor] = []
        pathway_attention_parts: list[torch.Tensor] = []
        type_gate_parts: list[torch.Tensor] = []

        for start in range(0, self.config.n_sites, self.config.site_chunk_size):
            end = min(self.config.n_sites, start + self.config.site_chunk_size)
            width = end - start
            site, parent_static, kinase = self._static_query(start, end)
            parent_index = self.parent_protein_index[start:end].clamp_min(0)
            parent_abundance = protein_prediction[:, parent_index]
            parent_abundance = parent_abundance * self.parent_protein_mask[start:end].unsqueeze(0)
            baseline = centered_baseline[:, start:end]
            anchor_quality = self.site_anchor_quality[start:end].view(1, width).expand(
                pathway_state.shape[0], -1
            )
            anchor_coverage = self.site_anchor_coverage[start:end].view(1, width).expand(
                pathway_state.shape[0], -1
            )
            memory, memory_coverage = self._residual_memory_chunk(
                start,
                end,
                sample_attention,
                neighbour_index,
                reference_residual,
                reference_residual_mask,
            )
            parent_dynamic = parent_static.unsqueeze(0) + self.parent_value(
                torch.stack(
                    [parent_abundance, baseline, anchor_quality, anchor_coverage], dim=-1
                )
            )
            sample_kinase_context = getattr(self, "_sample_site_kinase_context", None)
            if sample_kinase_context is None:
                kinase_dynamic = kinase.unsqueeze(0).expand(pathway_state.shape[0], -1, -1)
            else:
                if sample_kinase_context.shape != (
                    pathway_state.shape[0],
                    self.config.n_sites,
                    self.config.hidden,
                ):
                    raise ValueError("sample kinase context has an incompatible shape")
                kinase_dynamic = kinase.unsqueeze(0) + sample_kinase_context[:, start:end]
            site_dynamic = site.unsqueeze(0).expand(pathway_state.shape[0], -1, -1)

            pathway_index = self.site_pathway_index[start:end]
            related = pathway_state[:, pathway_index]
            query = self.query(site_dynamic + parent_dynamic + kinase_dynamic).unsqueeze(2)
            logits = (query * self.pathway_key(related)).sum(dim=-1) / math.sqrt(self.config.hidden)
            prior = self.site_pathway_weight[start:end].clamp_min(1.0e-4).log().unsqueeze(0)
            mask = self.site_pathway_mask[start:end].unsqueeze(0)
            pathway_attention = _masked_softmax(logits + prior, mask, dim=2)
            pathway_context = (pathway_attention.unsqueeze(-1) * related).sum(dim=2)

            gate_input = torch.cat(
                [site_dynamic, parent_dynamic, kinase_dynamic, pathway_context], dim=-1
            )
            type_gate = torch.softmax(self.type_gate(gate_input), dim=-1)
            fused = (
                type_gate[..., 0:1] * parent_dynamic
                + type_gate[..., 1:2] * kinase_dynamic
                + type_gate[..., 2:3] * pathway_context
            )
            fused = fused + self.memory_projection(
                torch.stack([memory, memory_coverage], dim=-1)
            )
            decoder_input = torch.cat(
                [
                    site_dynamic,
                    fused,
                    baseline.unsqueeze(-1),
                    parent_abundance.unsqueeze(-1),
                    anchor_quality.unsqueeze(-1),
                    anchor_coverage.unsqueeze(-1),
                    memory.unsqueeze(-1),
                    memory_coverage.unsqueeze(-1),
                    self.site_coverage[start:end].view(1, width, 1).expand(pathway_state.shape[0], -1, -1),
                ],
                dim=-1,
            )
            raw_part = self.decoder(decoder_input).squeeze(-1)
            raw_parts.append(raw_part)
            if return_attention:
                pathway_attention_parts.append(pathway_attention)
                type_gate_parts.append(type_gate)

        raw_residual = torch.cat(raw_parts, dim=1)
        # The projection uses the fixed output vocabulary. It never reads the
        # phosphosite observation mask, so external predictions are invariant
        # to label availability and query-cohort composition.
        centered_residual = project_fixed_vocabulary_zero_median(
            raw_residual, self.centering_site_index
        )
        shrinkage = torch.sigmoid(self.site_shrinkage_logit)
        neural_correction = centered_residual * shrinkage.unsqueeze(0)
        prediction = project_fixed_vocabulary_zero_median(
            centered_baseline + neural_correction,
            self.centering_site_index,
        )
        correction = prediction - centered_baseline
        return {
            "prediction": prediction,
            "correction": correction,
            "neural_correction": neural_correction,
            "centered_residual": centered_residual,
            "site_shrinkage": shrinkage,
            "site_pathway_attention": (
                torch.cat(pathway_attention_parts, dim=1) if return_attention else None
            ),
            "hyperedge_type_gate": (
                torch.cat(type_gate_parts, dim=1) if return_attention else None
            ),
        }


class ProteinAnchoredAxialDynamicHypergraph(nn.Module):
    """Full phosphosite model with alternating pathway and sample axes."""

    def __init__(
        self,
        config: AxialHypergraphConfig,
        *,
        rna_pathway_index: torch.Tensor,
        rna_pathway_mask: torch.Tensor,
        protein_pathway_index: torch.Tensor,
        protein_pathway_mask: torch.Tensor,
        pathway_relation_mask: torch.Tensor,
        site_pathway_index: torch.Tensor,
        site_pathway_mask: torch.Tensor,
        site_pathway_weight: torch.Tensor,
        parent_protein_index: torch.Tensor,
        parent_protein_mask: torch.Tensor,
        site_kinase_index: torch.Tensor,
        site_kinase_mask: torch.Tensor,
        site_coverage: torch.Tensor,
        site_anchor_quality: torch.Tensor,
        site_anchor_coverage: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self.pathway_encoder = MultimodalPathwayEncoder(
            config,
            rna_pathway_index,
            rna_pathway_mask,
            protein_pathway_index,
            protein_pathway_mask,
        )
        self.pathway_axis = nn.ModuleList(
            [
                PathwayAxisLayer(
                    config.hidden,
                    config.heads,
                    config.dropout,
                    pathway_relation_mask,
                )
                for _ in range(config.axial_layers)
            ]
        )
        self.sample_axis = nn.ModuleList(
            [
                DynamicPathwaySampleLayer(
                    config.n_pathways,
                    config.hidden,
                    config.heads,
                    config.pathway_adapter_rank,
                    config.dropout,
                )
                for _ in range(config.axial_layers)
            ]
        )
        self.site_decoder = HeterogeneousSiteHypergraphDecoder(
            config,
            site_pathway_index,
            site_pathway_mask,
            site_pathway_weight,
            parent_protein_index,
            parent_protein_mask,
            site_kinase_index,
            site_kinase_mask,
            site_coverage,
            site_anchor_quality,
            site_anchor_coverage,
        )

    @staticmethod
    def _expand_candidates(
        index: torch.Tensor,
        similarity: torch.Tensor,
        n_pathways: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if index.ndim == 2:
            index = index.unsqueeze(0).expand(n_pathways, -1, -1)
        if similarity.ndim == 2:
            similarity = similarity.unsqueeze(0).expand(n_pathways, -1, -1)
        return index, similarity

    def encode_base(
        self,
        rna_rank: torch.Tensor,
        protein_prediction: torch.Tensor,
        rna_mask: torch.Tensor | None = None,
        protein_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.pathway_encoder(
            rna_rank, protein_prediction.detach(), rna_mask, protein_mask
        )

    @torch.no_grad()
    def build_reference_cache(
        self,
        reference_rna_rank: torch.Tensor,
        reference_protein_prediction: torch.Tensor,
        neighbour_index: torch.Tensor,
        neighbour_similarity: torch.Tensor,
        chunk_size: int = 64,
    ) -> tuple[torch.Tensor, ...]:
        """Build layer-specific source states using synchronous graph updates."""
        state = self.encode_base(reference_rna_rank, reference_protein_prediction)
        index, similarity = self._expand_candidates(
            neighbour_index, neighbour_similarity, self.config.n_pathways
        )
        cache: list[torch.Tensor] = []
        for pathway_layer, sample_layer in zip(self.pathway_axis, self.sample_axis):
            axial = pathway_layer(state)
            cache.append(axial.detach())
            parts = []
            for start in range(0, axial.shape[0], chunk_size):
                end = min(axial.shape[0], start + chunk_size)
                part, _ = sample_layer(
                    axial[start:end],
                    axial,
                    index[:, start:end],
                    similarity[:, start:end],
                )
                parts.append(part)
            state = torch.cat(parts, dim=0)
        return tuple(cache)

    def forward(
        self,
        query_rna_rank: torch.Tensor,
        query_protein_prediction: torch.Tensor,
        centered_parent_baseline: torch.Tensor,
        reference_cache: Sequence[torch.Tensor],
        neighbour_index: torch.Tensor,
        neighbour_similarity: torch.Tensor,
        reference_residual: torch.Tensor,
        reference_residual_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        if len(reference_cache) != self.config.axial_layers:
            raise ValueError("reference cache layer count differs from the model")
        index, similarity = self._expand_candidates(
            neighbour_index, neighbour_similarity, self.config.n_pathways
        )
        state = self.encode_base(query_rna_rank, query_protein_prediction)
        sample_attention: list[torch.Tensor] = []
        final_attention: torch.Tensor | None = None
        for layer, (pathway_layer, sample_layer) in enumerate(
            zip(self.pathway_axis, self.sample_axis)
        ):
            state = pathway_layer(state)
            state, attention = sample_layer(
                state,
                reference_cache[layer].detach(),
                index,
                similarity,
            )
            final_attention = attention
            if return_attention:
                sample_attention.append(attention)
        if final_attention is None:
            raise RuntimeError("axial model produced no sample attention")
        output = self.site_decoder(
            state,
            query_protein_prediction.detach(),
            centered_parent_baseline.detach(),
            final_attention,
            index,
            reference_residual.detach(),
            reference_residual_mask.detach(),
            return_attention=return_attention,
        )
        output["sample_attention"] = sample_attention if return_attention else None
        return output

    def shrinkage_regularization(self) -> torch.Tensor:
        return torch.sigmoid(self.site_decoder.site_shrinkage_logit).square().mean()

    def set_centering_site_index(self, index: torch.Tensor) -> None:
        self.site_decoder.set_centering_site_index(index)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "architecture": "protein_anchored_axial_dynamic_pathway_heterogeneous_hypergraph",
            "formula": "centered_phosphosite = centered_calibrated_parent_protein + site_shrinkage * zero_median_residual",
            "total_protein_is_frozen_input": True,
            "phosphosite_gradient_to_total_protein": False,
            "sample_graph_input": ["RNA rank", "predicted total protein"],
            "reference_label_memory": "training residual only, introduced after graph attention",
            "external_query_to_query_edges": False,
            "final_projection": "fixed-panel sample-row zero median",
            "centering_site_count": int(self.site_decoder.centering_site_index.numel()),
            "config": self.config.to_dict(),
        }


def masked_site_equal_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    minimum_observations: int = 3,
) -> torch.Tensor:
    """Give every sufficiently observed site equal weight in the value loss."""
    prediction = prediction.float()
    target = target.float()
    mask_f = mask.float()
    count = mask_f.sum(dim=0)
    per_site = ((prediction - target).square() * mask_f).sum(dim=0) / count.clamp_min(1.0)
    valid = count >= int(minimum_observations)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return per_site[valid].mean()


def masked_sample_intercept_site_equal_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Remove a nuisance sample offset before the site-equal value loss."""
    mask_f = mask.float()
    count = mask_f.sum(dim=1, keepdim=True)
    offset = ((prediction - target) * mask_f).sum(dim=1, keepdim=True) / count.clamp_min(1.0)
    aligned = prediction - offset
    return masked_site_equal_mse(aligned, target, mask)


def masked_site_equal_pearson_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    minimum_observations: int = 8,
) -> torch.Tensor:
    """Differentiable cross-sample correlation loss with equal site weight."""
    prediction = prediction.float()
    target = target.float()
    mask_f = mask.float()
    count = mask_f.sum(dim=0)
    pred_mean = (prediction * mask_f).sum(dim=0) / count.clamp_min(1.0)
    target_mean = (target * mask_f).sum(dim=0) / count.clamp_min(1.0)
    pred_centered = (prediction - pred_mean) * mask_f
    target_centered = (target - target_mean) * mask_f
    numerator = (pred_centered * target_centered).sum(dim=0)
    pred_square_sum = pred_centered.square().sum(dim=0)
    target_square_sum = target_centered.square().sum(dim=0)
    epsilon = 1.0e-8
    denominator = pred_square_sum.clamp_min(epsilon).sqrt()
    denominator = denominator * target_square_sum.clamp_min(epsilon).sqrt()
    correlation = numerator / denominator
    valid = (
        (count >= int(minimum_observations))
        & (pred_square_sum > epsilon)
        & (target_square_sum > epsilon)
    )
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return 1.0 - correlation[valid].mean()


def masked_site_equal_variance_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    minimum_observations: int = 8,
) -> torch.Tensor:
    """Recover cross-sample site variance with equal weight per valid site."""
    prediction = prediction.float()
    target = target.float()
    mask_f = mask.float()
    count = mask_f.sum(dim=0)
    pred_mean = (prediction * mask_f).sum(dim=0) / count.clamp_min(1.0)
    target_mean = (target * mask_f).sum(dim=0) / count.clamp_min(1.0)
    denominator = (count - 1.0).clamp_min(1.0)
    pred_variance = (((prediction - pred_mean) * mask_f).square()).sum(dim=0) / denominator
    target_variance = (((target - target_mean) * mask_f).square()).sum(dim=0) / denominator
    epsilon = 1.0e-6
    difference = torch.log(pred_variance + epsilon) - torch.log(target_variance + epsilon)
    valid = (count >= int(minimum_observations)) & (target_variance > epsilon)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return difference[valid].square().mean()


def compose_training_objective(
    value_mse: torch.Tensor,
    pearson_loss: torch.Tensor,
    variance_loss: torch.Tensor,
    shrinkage_loss: torch.Tensor,
    *,
    pearson_weight: float,
    variance_weight: float,
    shrinkage_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compose the unique training terms; no residual MSE is accepted."""
    components = {
        "site_equal_mse": value_mse,
        "site_equal_pearson": pearson_loss,
        "site_equal_variance": variance_loss,
        "site_shrinkage": shrinkage_loss,
    }
    total = (
        value_mse
        + float(pearson_weight) * pearson_loss
        + float(variance_weight) * variance_loss
        + float(shrinkage_weight) * shrinkage_loss
    )
    return total, components
