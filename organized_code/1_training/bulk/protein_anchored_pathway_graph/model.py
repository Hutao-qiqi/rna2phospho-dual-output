from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    n_rna: int
    n_proteins: int
    n_sites: int
    n_pathways: int
    n_kinases: int
    protein_hidden: int = 384
    pathway_hidden: int = 64
    pathway_adapter_rank: int = 8
    dropout: float = 0.15
    initial_site_shrinkage: float = 0.30
    site_chunk_size: int = 1024

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProteinAnchorPredictor(nn.Module):
    """Predict total-protein abundance from RNA before phosphosite fitting."""

    def __init__(self, n_rna: int, n_proteins: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_rna, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, n_proteins)

    def forward(self, rna_z: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(rna_z))


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool, device=logits.device)
    masked = logits.masked_fill(~mask, -1.0e4)
    weights = torch.softmax(masked, dim=dim) * mask.to(logits.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1.0e-8)


class PathwayTokenEncoder(nn.Module):
    """Pool sample-specific RNA and predicted-protein values within each pathway."""

    def __init__(
        self,
        n_rna: int,
        n_proteins: int,
        n_pathways: int,
        hidden: int,
        dropout: float,
        rna_pathway_index: torch.Tensor,
        rna_pathway_mask: torch.Tensor,
        protein_pathway_index: torch.Tensor,
        protein_pathway_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.n_pathways = n_pathways
        self.hidden = hidden
        self.register_buffer("rna_pathway_index", rna_pathway_index.long())
        self.register_buffer("rna_pathway_mask", rna_pathway_mask.bool())
        self.register_buffer("protein_pathway_index", protein_pathway_index.long())
        self.register_buffer("protein_pathway_mask", protein_pathway_mask.bool())

        self.rna_gene_embedding = nn.Embedding(n_rna, hidden)
        self.protein_embedding = nn.Embedding(n_proteins, hidden)
        self.pathway_query = nn.Parameter(torch.empty(n_pathways, hidden))
        self.rna_value_direction = nn.Parameter(torch.empty(hidden))
        self.protein_value_direction = nn.Parameter(torch.empty(hidden))
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2 + 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        nn.init.normal_(self.pathway_query, std=0.02)
        nn.init.normal_(self.rna_value_direction, std=0.02)
        nn.init.normal_(self.protein_value_direction, std=0.02)

    @staticmethod
    def _moments(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_f = mask.to(values.dtype).unsqueeze(0)
        count = mask_f.sum(dim=-1).clamp_min(1.0)
        mean = (values * mask_f).sum(dim=-1) / count
        var = ((values - mean.unsqueeze(-1)).square() * mask_f).sum(dim=-1) / count
        return mean, torch.sqrt(var.clamp_min(1.0e-8))

    def _pool(
        self,
        values: torch.Tensor,
        index: torch.Tensor,
        mask: torch.Tensor,
        embedding: nn.Embedding,
        direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected = values[:, index]
        gene_token = embedding(index).unsqueeze(0)
        token = gene_token + selected.unsqueeze(-1) * direction.view(1, 1, 1, -1)
        query = self.pathway_query.view(1, self.n_pathways, 1, self.hidden)
        logits = (token * query).sum(dim=-1) / math.sqrt(self.hidden)
        weights = _masked_softmax(logits, mask.view(1, *mask.shape), dim=-1)
        pooled = (weights.unsqueeze(-1) * token).sum(dim=-2)
        mean, std = self._moments(selected, mask)
        return pooled, mean, std

    def forward(self, rna_rank: torch.Tensor, protein_hat: torch.Tensor) -> torch.Tensor:
        rna_pool, rna_mean, rna_std = self._pool(
            rna_rank,
            self.rna_pathway_index,
            self.rna_pathway_mask,
            self.rna_gene_embedding,
            self.rna_value_direction,
        )
        protein_pool, protein_mean, protein_std = self._pool(
            protein_hat,
            self.protein_pathway_index,
            self.protein_pathway_mask,
            self.protein_embedding,
            self.protein_value_direction,
        )
        summary = torch.stack([rna_mean, rna_std, protein_mean, protein_std], dim=-1)
        return self.fusion(torch.cat([rna_pool, protein_pool, summary], dim=-1))


class PathwaySampleGraphLayer(nn.Module):
    """Apply attention over a separate sample-neighbour graph for every pathway."""

    def __init__(self, n_pathways: int, hidden: int, adapter_rank: int, dropout: float) -> None:
        super().__init__()
        self.n_pathways = n_pathways
        self.hidden = hidden
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.base_output = nn.Linear(hidden, hidden, bias=False)
        self.adapter_v = nn.Parameter(torch.empty(n_pathways, adapter_rank, hidden))
        self.adapter_u = nn.Parameter(torch.empty(n_pathways, hidden, adapter_rank))
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.adapter_v, std=0.01)
        nn.init.zeros_(self.adapter_u)

    def forward(
        self,
        query_state: torch.Tensor,
        source_state: torch.Tensor,
        neighbour_index: torch.Tensor,
        neighbour_similarity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_pathways, hidden = query_state.shape
        if n_pathways != self.n_pathways or hidden != self.hidden:
            raise ValueError("Pathway-state shape does not match model configuration")
        if neighbour_index.shape[:2] != (n_pathways, batch):
            raise ValueError("Neighbour index must have shape [pathway, query_sample, k]")

        n_source = source_state.shape[0]
        source_by_pathway = source_state.permute(1, 0, 2).contiguous()
        flat_source = source_by_pathway.reshape(n_pathways * n_source, hidden)
        offset = torch.arange(n_pathways, device=query_state.device).view(n_pathways, 1, 1) * n_source
        flat_index = neighbour_index.long() + offset
        neighbours = flat_source.index_select(0, flat_index.reshape(-1))
        neighbours = neighbours.reshape(n_pathways, batch, neighbour_index.shape[-1], hidden)
        neighbours = neighbours.permute(1, 0, 2, 3).contiguous()

        q = self.query(query_state).unsqueeze(2)
        k = self.key(neighbours)
        logits = (q * k).sum(dim=-1) / math.sqrt(hidden)
        graph_prior = ((neighbour_similarity.permute(1, 0, 2) + 1.0) * 0.5).clamp_min(1.0e-4)
        attention = torch.softmax(logits + graph_prior.log(), dim=-1)
        aggregated = (attention.unsqueeze(-1) * self.value(neighbours)).sum(dim=2)

        base_message = self.base_output(aggregated)
        low_rank = torch.einsum("bph,prh->bpr", aggregated, self.adapter_v)
        pathway_message = torch.einsum("bpr,phr->bph", low_rank, self.adapter_u)
        output = self.norm(query_state + self.dropout(base_message + pathway_message))
        return output, attention


class SitePathwayDecoder(nn.Module):
    """Read related pathway states with a parent/site/kinase query."""

    def __init__(
        self,
        n_sites: int,
        n_proteins: int,
        n_kinases: int,
        hidden: int,
        dropout: float,
        initial_shrinkage: float,
        site_pathway_index: torch.Tensor,
        site_pathway_mask: torch.Tensor,
        site_pathway_weight: torch.Tensor,
        parent_protein_index: torch.Tensor,
        parent_protein_mask: torch.Tensor,
        site_kinase_index: torch.Tensor,
        site_kinase_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.n_sites = n_sites
        self.hidden = hidden
        self.register_buffer("site_pathway_index", site_pathway_index.long())
        self.register_buffer("site_pathway_mask", site_pathway_mask.bool())
        self.register_buffer("site_pathway_weight", site_pathway_weight.float())
        self.register_buffer("parent_protein_index", parent_protein_index.long())
        self.register_buffer("parent_protein_mask", parent_protein_mask.bool())
        self.register_buffer("site_kinase_index", site_kinase_index.long())
        self.register_buffer("site_kinase_mask", site_kinase_mask.bool())

        self.site_embedding = nn.Embedding(n_sites, hidden)
        self.parent_embedding = nn.Embedding(n_proteins, hidden)
        self.kinase_embedding = nn.Embedding(n_kinases + 1, hidden, padding_idx=0)
        self.query_norm = nn.LayerNorm(hidden)
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        initial = min(max(float(initial_shrinkage), 1.0e-4), 1.0 - 1.0e-4)
        self.site_shrinkage_logit = nn.Parameter(
            torch.full((n_sites,), math.log(initial / (1.0 - initial)))
        )

    def _site_query(self, start: int, end: int) -> torch.Tensor:
        site_ids = torch.arange(start, end, device=self.site_embedding.weight.device)
        query = self.site_embedding(site_ids)
        parent_index = self.parent_protein_index[start:end]
        parent = self.parent_embedding(parent_index.clamp_min(0))
        query = query + parent * self.parent_protein_mask[start:end].unsqueeze(-1)

        kinase_index = self.site_kinase_index[start:end]
        kinase_mask = self.site_kinase_mask[start:end].unsqueeze(-1)
        kinase = self.kinase_embedding(kinase_index)
        kinase = (kinase * kinase_mask).sum(dim=1) / kinase_mask.sum(dim=1).clamp_min(1)
        return self.query_norm(query + kinase)

    def forward(
        self,
        pathway_state: torch.Tensor,
        chunk_size: int,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        residual_chunks: list[torch.Tensor] = []
        attention_chunks: list[torch.Tensor] = []
        for start in range(0, self.n_sites, chunk_size):
            end = min(start + chunk_size, self.n_sites)
            path_index = self.site_pathway_index[start:end]
            path_state = pathway_state[:, path_index]
            query = self._site_query(start, end)
            logits = (path_state * query.view(1, end - start, 1, self.hidden)).sum(dim=-1)
            logits = logits / math.sqrt(self.hidden)
            prior = self.site_pathway_weight[start:end].clamp_min(1.0e-4).log().unsqueeze(0)
            mask = self.site_pathway_mask[start:end].unsqueeze(0)
            attention = _masked_softmax(logits + prior, mask, dim=-1)
            context = (attention.unsqueeze(-1) * path_state).sum(dim=2)
            query_batch = query.unsqueeze(0).expand(pathway_state.shape[0], -1, -1)
            residual_chunks.append(self.decoder(torch.cat([context, query_batch], dim=-1)).squeeze(-1))
            if return_attention:
                attention_chunks.append(attention)

        raw_residual = torch.cat(residual_chunks, dim=1)
        centered_residual = raw_residual - raw_residual.median(dim=1, keepdim=True).values
        shrinkage = torch.sigmoid(self.site_shrinkage_logit)
        attention_out = torch.cat(attention_chunks, dim=1) if return_attention else None
        return centered_residual, shrinkage, attention_out


class ProteinAnchoredPathwayGraphModel(nn.Module):
    """Total-protein anchor followed by a pathway-specific sample-graph residual."""

    def __init__(
        self,
        config: ModelConfig,
        rna_pathway_index: torch.Tensor,
        rna_pathway_mask: torch.Tensor,
        protein_pathway_index: torch.Tensor,
        protein_pathway_mask: torch.Tensor,
        site_pathway_index: torch.Tensor,
        site_pathway_mask: torch.Tensor,
        site_pathway_weight: torch.Tensor,
        parent_protein_index: torch.Tensor,
        parent_protein_mask: torch.Tensor,
        site_kinase_index: torch.Tensor,
        site_kinase_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self.protein_predictor = ProteinAnchorPredictor(
            config.n_rna, config.n_proteins, config.protein_hidden, config.dropout
        )
        self.pathway_encoder = PathwayTokenEncoder(
            config.n_rna,
            config.n_proteins,
            config.n_pathways,
            config.pathway_hidden,
            config.dropout,
            rna_pathway_index,
            rna_pathway_mask,
            protein_pathway_index,
            protein_pathway_mask,
        )
        self.sample_graph = PathwaySampleGraphLayer(
            config.n_pathways,
            config.pathway_hidden,
            config.pathway_adapter_rank,
            config.dropout,
        )
        self.site_decoder = SitePathwayDecoder(
            config.n_sites,
            config.n_proteins,
            config.n_kinases,
            config.pathway_hidden,
            config.dropout,
            config.initial_site_shrinkage,
            site_pathway_index,
            site_pathway_mask,
            site_pathway_weight,
            parent_protein_index,
            parent_protein_mask,
            site_kinase_index,
            site_kinase_mask,
        )
        self.register_buffer("site_intercept", torch.zeros(config.n_sites))
        self.register_buffer("site_parent_scale", torch.zeros(config.n_sites))
        self.register_buffer("parent_protein_index", parent_protein_index.long())
        self.register_buffer("parent_protein_mask", parent_protein_mask.bool())

    def set_parent_calibration(self, intercept: torch.Tensor, scale: torch.Tensor) -> None:
        if intercept.numel() != self.config.n_sites or scale.numel() != self.config.n_sites:
            raise ValueError("Parent calibration length differs from the number of phosphosites")
        self.site_intercept.copy_(intercept.to(self.site_intercept))
        self.site_parent_scale.copy_(scale.to(self.site_parent_scale))

    def predict_protein(self, rna_z: torch.Tensor) -> torch.Tensor:
        return self.protein_predictor(rna_z)

    def parent_baseline(self, protein_hat: torch.Tensor) -> torch.Tensor:
        parent = protein_hat[:, self.parent_protein_index.clamp_min(0)]
        parent = parent * self.parent_protein_mask.unsqueeze(0)
        return self.site_intercept.unsqueeze(0) + self.site_parent_scale.unsqueeze(0) * parent

    def encode_pathways(self, rna_rank: torch.Tensor, protein_hat: torch.Tensor) -> torch.Tensor:
        return self.pathway_encoder(rna_rank, protein_hat.detach())

    def residual_forward(
        self,
        rna_rank: torch.Tensor,
        protein_hat: torch.Tensor,
        source_pathway_state: torch.Tensor,
        neighbour_index: torch.Tensor,
        neighbour_similarity: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        query_state = self.encode_pathways(rna_rank, protein_hat)
        graph_state, sample_attention = self.sample_graph(
            query_state,
            source_pathway_state.detach(),
            neighbour_index,
            neighbour_similarity,
        )
        raw_residual, shrinkage, site_attention = self.site_decoder(
            graph_state,
            chunk_size=self.config.site_chunk_size,
            return_attention=return_attention,
        )
        correction = raw_residual * shrinkage.unsqueeze(0)
        baseline = self.parent_baseline(protein_hat.detach())
        return {
            "prediction": baseline + correction,
            "baseline": baseline,
            "centered_residual": raw_residual,
            "correction": correction,
            "site_shrinkage": shrinkage,
            "sample_attention": sample_attention,
            "site_pathway_attention": site_attention,
        }

    def freeze_protein_anchor(self) -> None:
        self.protein_predictor.eval()
        for parameter in self.protein_predictor.parameters():
            parameter.requires_grad_(False)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "architecture": "protein_anchored_pathway_specific_sample_graph_residual",
            "formula": "phosphosite_hat = calibrated_parent_protein_hat + site_shrinkage * centered_pathway_graph_residual",
            "protein_gradient_from_phosphosite_loss": False,
            "external_graph_inputs": ["RNA rank", "predicted total protein"],
            "config": self.config.to_dict(),
        }
