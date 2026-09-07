"""Performer RNA-to-protein translator with a phosphosite residual decoder."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

try:
    from .performer_core import PerformerCrossAttention, PerformerEncoder
except ImportError:
    from performer_core import PerformerCrossAttention, PerformerEncoder


@dataclass(frozen=True)
class ModelConfig:
    n_genes: int
    n_proteins: int
    n_sites: int
    n_kinases: int = 1
    n_pathways: int = 1
    d_model: int = 128
    encoder_depth: int = 2
    decoder_depth: int = 2
    n_heads: int = 8
    dim_head: int = 64
    translator_depth: int = 2
    translator_hidden: int = 0
    ff_mult: int = 4
    dropout: float = 0.10
    gene_mask_probability: float = 0.10
    nb_features: int | None = None
    feature_redraw_interval: int | None = 1000
    site_query_chunk: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RNAEncoder(nn.Module):
    """Encode fixed-order RNA genes as identity-plus-expression tokens."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.gene_identity = nn.Embedding(config.n_genes, config.d_model)
        self.expression_projection = nn.Linear(1, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.encoder = PerformerEncoder(
            dim=config.d_model,
            depth=config.encoder_depth,
            heads=config.n_heads,
            dim_head=config.dim_head,
            dropout=config.dropout,
            ff_mult=config.ff_mult,
            nb_features=config.nb_features,
            feature_redraw_interval=config.feature_redraw_interval,
        )
        self.register_buffer(
            "gene_indices",
            torch.arange(config.n_genes, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        expression: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if expression.ndim != 2 or expression.shape[1] != self.config.n_genes:
            raise ValueError(
                f"expression must have shape (batch, {self.config.n_genes})"
            )
        if valid_mask is None:
            valid_mask = torch.ones_like(expression, dtype=torch.bool)
        elif valid_mask.shape != expression.shape:
            raise ValueError("valid_mask must match expression")
        else:
            valid_mask = valid_mask.bool()

        effective_mask = valid_mask
        if self.training and self.config.gene_mask_probability > 0:
            keep = torch.rand_like(expression) >= self.config.gene_mask_probability
            effective_mask = valid_mask & keep
            empty_rows = ~effective_mask.any(dim=1)
            if empty_rows.any():
                effective_mask = effective_mask.clone()
                effective_mask[empty_rows] = valid_mask[empty_rows]

        identity = self.gene_identity(self.gene_indices).unsqueeze(0)
        tokens = identity + self.expression_projection(expression.unsqueeze(-1))
        tokens = self.input_dropout(self.input_norm(tokens))
        hidden = self.encoder(tokens, mask=effective_mask)
        return hidden, effective_mask


class AxisMLPTranslator(nn.Module):
    """Translate the complete RNA position axis into the complete protein axis."""

    def __init__(
        self,
        n_genes: int,
        n_proteins: int,
        depth: int,
        hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if depth not in {1, 2}:
            raise ValueError("translator_depth must be 1 or 2")
        self.n_genes = n_genes
        self.n_proteins = n_proteins
        self.depth = depth
        if depth == 1:
            self.layers = nn.Linear(n_genes, n_proteins)
            self.hidden = 0
        else:
            if hidden <= 0:
                hidden = int(math.ceil(math.sqrt(n_genes * n_proteins)))
            self.hidden = hidden
            self.layers = nn.Sequential(
                nn.Linear(n_genes, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_proteins),
            )

    def forward(self, rna_hidden: Tensor) -> Tensor:
        if rna_hidden.ndim != 3 or rna_hidden.shape[1] != self.n_genes:
            raise ValueError(
                f"rna_hidden must have shape (batch, {self.n_genes}, dim)"
            )
        translated = self.layers(rna_hidden.transpose(1, 2).contiguous())
        return translated.transpose(1, 2).contiguous()


class ProteinDecoder(nn.Module):
    """Decode all fixed-order protein tokens and their abundances."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.protein_identity = nn.Embedding(config.n_proteins, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.decoder = PerformerEncoder(
            dim=config.d_model,
            depth=config.decoder_depth,
            heads=config.n_heads,
            dim_head=config.dim_head,
            dropout=config.dropout,
            ff_mult=config.ff_mult,
            nb_features=config.nb_features,
            feature_redraw_interval=config.feature_redraw_interval,
        )
        self.abundance_head = nn.Linear(config.d_model, 1)
        self.register_buffer(
            "protein_indices",
            torch.arange(config.n_proteins, dtype=torch.long),
            persistent=False,
        )

    def forward(self, translated: Tensor) -> tuple[Tensor, Tensor]:
        if translated.ndim != 3 or translated.shape[1] != self.config.n_proteins:
            raise ValueError(
                f"translated must have shape (batch, {self.config.n_proteins}, dim)"
            )
        identity = self.protein_identity(self.protein_indices).unsqueeze(0)
        hidden = self.decoder(self.input_norm(translated + identity))
        abundance = self.abundance_head(hidden).squeeze(-1)
        return abundance, hidden


class PriorBiasedPathwayAttention(nn.Module):
    """Attend from phosphosite queries to pathway states with a fixed prior bias."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("d_model must be divisible by n_heads for pathway attention")
        self.dim = dim
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head**-0.5
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.to_query = nn.Linear(dim, dim, bias=False)
        self.to_key = nn.Linear(dim, dim, bias=False)
        self.to_value = nn.Linear(dim, dim, bias=False)
        self.to_output = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.prior_strength_raw = nn.Parameter(torch.tensor(0.0))

    def _heads(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.heads, self.dim_head).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        pathway_state: Tensor,
        site_pathway_prior: Tensor | None,
    ) -> Tensor:
        q = self._heads(self.to_query(self.query_norm(query)))
        k = self._heads(self.to_key(self.context_norm(pathway_state)))
        v = self._heads(self.to_value(self.context_norm(pathway_state)))
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if site_pathway_prior is not None:
            if site_pathway_prior.shape != (query.shape[1], pathway_state.shape[1]):
                raise ValueError("site_pathway_prior has an incompatible shape")
            strength = torch.nn.functional.softplus(self.prior_strength_raw)
            bias = torch.log1p(site_pathway_prior.clamp_min(0) * strength)
            logits = logits + bias.unsqueeze(0).unsqueeze(0).to(logits.dtype)
        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        output = torch.matmul(weights, v).transpose(1, 2).contiguous()
        output = output.view(query.shape[0], query.shape[1], self.dim)
        return self.to_output(output)


class PhosphositeResidualDecoder(nn.Module):
    """Predict parent-anchored direct and pathway phosphosite residuals."""

    def __init__(
        self,
        config: ModelConfig,
        site_to_protein_index: Tensor,
        site_kinase_weights: Tensor | None,
        pathway_gene_weights: Tensor | None,
        site_pathway_prior: Tensor | None,
    ) -> None:
        super().__init__()
        if site_to_protein_index.numel() != config.n_sites:
            raise ValueError("site_to_protein_index length must equal n_sites")
        self.config = config
        self.site_identity = nn.Embedding(config.n_sites, config.d_model)
        self.kinase_identity = nn.Embedding(config.n_kinases, config.d_model)
        self.unknown_kinase = nn.Parameter(torch.zeros(config.d_model))
        self.query_norm = nn.LayerNorm(config.d_model)
        self.direct_attention = PerformerCrossAttention(
            dim=config.d_model,
            heads=config.n_heads,
            dim_head=config.dim_head,
            dropout=config.dropout,
            nb_features=config.nb_features,
            feature_redraw_interval=config.feature_redraw_interval,
        )
        self.direct_decoder = nn.Sequential(
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )
        self.pathway_attention = PriorBiasedPathwayAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
        )
        self.pathway_decoder = nn.Sequential(
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )
        self.site_intercept = nn.Parameter(torch.zeros(config.n_sites))
        self.parent_scale = nn.Parameter(torch.ones(config.n_sites))
        self.pathway_gate_logit = nn.Parameter(torch.zeros(config.n_sites))

        site_to_protein_index = site_to_protein_index.long()
        valid_parent = site_to_protein_index >= 0
        self.register_buffer(
            "site_to_protein_index",
            site_to_protein_index.clamp(min=0),
        )
        self.register_buffer("site_has_parent", valid_parent.float())

        if site_kinase_weights is None:
            site_kinase_weights = torch.zeros(config.n_sites, config.n_kinases)
        if tuple(site_kinase_weights.shape) != (config.n_sites, config.n_kinases):
            raise ValueError("site_kinase_weights has an incompatible shape")
        self.register_buffer("site_kinase_weights", site_kinase_weights.float())

        if pathway_gene_weights is None:
            pathway_gene_weights = torch.full(
                (config.n_pathways, config.n_genes),
                1.0 / max(config.n_genes, 1),
            )
        if tuple(pathway_gene_weights.shape) != (config.n_pathways, config.n_genes):
            raise ValueError("pathway_gene_weights has an incompatible shape")
        self.register_buffer("pathway_gene_weights", pathway_gene_weights.float())

        if site_pathway_prior is None:
            site_pathway_prior = torch.zeros(config.n_sites, config.n_pathways)
        if tuple(site_pathway_prior.shape) != (config.n_sites, config.n_pathways):
            raise ValueError("site_pathway_prior has an incompatible shape")
        self.register_buffer("site_pathway_prior", site_pathway_prior.float())
        self.register_buffer(
            "site_indices",
            torch.arange(config.n_sites, dtype=torch.long),
            persistent=False,
        )

    def _kinase_embedding(self, selected: Tensor) -> Tensor:
        weights = self.site_kinase_weights.index_select(0, selected)
        known = torch.matmul(weights, self.kinase_identity.weight)
        missing = (weights.sum(dim=-1, keepdim=True) <= 0).to(known.dtype)
        return known + missing * self.unknown_kinase.unsqueeze(0)

    def _pathway_states(self, rna_hidden: Tensor) -> Tensor:
        weights = self.pathway_gene_weights.to(dtype=rna_hidden.dtype)
        return torch.einsum("pg,bgd->bpd", weights, rna_hidden)

    def _decode_chunk(
        self,
        selected: Tensor,
        rna_hidden: Tensor,
        rna_mask: Tensor,
        protein_prediction: Tensor,
        protein_hidden: Tensor,
        protein_identity_weight: Tensor,
        pathway_state: Tensor,
    ) -> dict[str, Tensor]:
        parent_index = self.site_to_protein_index.index_select(0, selected)
        parent_mask = self.site_has_parent.index_select(0, selected)
        parent_hidden = protein_hidden.index_select(1, parent_index)
        parent_hidden = parent_hidden * parent_mask.view(1, -1, 1)
        parent_identity = protein_identity_weight.index_select(0, parent_index)
        parent_identity = parent_identity * parent_mask.unsqueeze(-1)

        static_query = (
            self.site_identity(selected)
            + parent_identity
            + self._kinase_embedding(selected)
        )
        query = self.query_norm(static_query.unsqueeze(0) + parent_hidden)

        direct_context = self.direct_attention(query, rna_hidden, rna_mask)
        direct = self.direct_decoder(torch.cat((query, direct_context), dim=-1)).squeeze(-1)

        prior = self.site_pathway_prior.index_select(0, selected)
        pathway_context = self.pathway_attention(query, pathway_state, prior)
        pathway = self.pathway_decoder(
            torch.cat((query, pathway_context), dim=-1)
        ).squeeze(-1)
        gate = torch.sigmoid(self.pathway_gate_logit.index_select(0, selected))
        gate = gate.unsqueeze(0).expand_as(pathway)

        parent_value = protein_prediction.index_select(1, parent_index)
        anchor = self.site_intercept.index_select(0, selected).unsqueeze(0)
        anchor = anchor + (
            self.parent_scale.index_select(0, selected).unsqueeze(0)
            * parent_value
            * parent_mask.unsqueeze(0)
        )
        residual = direct + gate * pathway
        return {
            "phosphosite": anchor + residual,
            "phosphosite_anchor": anchor,
            "phosphosite_direct_residual": direct,
            "phosphosite_pathway_residual": pathway,
            "phosphosite_pathway_gate": gate,
            "phosphosite_residual": residual,
        }

    def forward(
        self,
        rna_hidden: Tensor,
        rna_mask: Tensor,
        protein_prediction: Tensor,
        protein_hidden: Tensor,
        protein_identity_weight: Tensor,
        site_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if site_indices is None:
            site_indices = self.site_indices
        site_indices = site_indices.long()
        pathway_state = self._pathway_states(rna_hidden)
        chunk = self.config.site_query_chunk or int(site_indices.numel())
        parts: dict[str, list[Tensor]] = {
            "phosphosite": [],
            "phosphosite_anchor": [],
            "phosphosite_direct_residual": [],
            "phosphosite_pathway_residual": [],
            "phosphosite_pathway_gate": [],
            "phosphosite_residual": [],
        }
        for start in range(0, site_indices.numel(), chunk):
            selected = site_indices[start : start + chunk]
            decoded = self._decode_chunk(
                selected,
                rna_hidden,
                rna_mask,
                protein_prediction,
                protein_hidden,
                protein_identity_weight,
                pathway_state,
            )
            for name, value in decoded.items():
                parts[name].append(value)
        output = {name: torch.cat(values, dim=1) for name, values in parts.items()}
        output["site_indices"] = site_indices
        return output


class RNAProteinPhosphositeTranslator(nn.Module):
    """Joint Performer encoder, all-protein translator and phosphosite head."""

    def __init__(
        self,
        config: ModelConfig,
        protein_to_gene_index: Tensor,
        site_to_protein_index: Tensor,
        residue_type_index: Tensor | None = None,
        site_kinase_weights: Tensor | None = None,
        pathway_gene_weights: Tensor | None = None,
        site_pathway_prior: Tensor | None = None,
    ) -> None:
        super().__init__()
        if protein_to_gene_index.numel() != config.n_proteins:
            raise ValueError("protein_to_gene_index length must equal n_proteins")
        self.config = config
        self.rna_encoder = RNAEncoder(config)
        self.cross_modal_translator = AxisMLPTranslator(
            config.n_genes,
            config.n_proteins,
            config.translator_depth,
            config.translator_hidden,
            config.dropout,
        )
        self.protein_decoder = ProteinDecoder(config)
        self.site_decoder = PhosphositeResidualDecoder(
            config,
            site_to_protein_index,
            site_kinase_weights,
            pathway_gene_weights,
            site_pathway_prior,
        )
        self.register_buffer("protein_to_gene_index", protein_to_gene_index.long())
        if residue_type_index is not None:
            self.register_buffer("residue_type_index", residue_type_index.long())

    def encode_rna(
        self,
        expression: Tensor,
        rna_valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.rna_encoder(expression, rna_valid_mask)

    def decode_protein(self, rna_hidden: Tensor) -> tuple[Tensor, Tensor]:
        translated = self.cross_modal_translator(rna_hidden)
        return self.protein_decoder(translated)

    def forward(
        self,
        expression: Tensor,
        rna_rank: Tensor | None = None,
        rna_valid_mask: Tensor | None = None,
        site_indices: Tensor | None = None,
        protein_only: bool = False,
        return_hidden: bool = False,
    ) -> dict[str, Tensor]:
        del rna_rank
        rna_hidden, rna_mask = self.encode_rna(expression, rna_valid_mask)
        protein_prediction, protein_hidden = self.decode_protein(rna_hidden)
        output: dict[str, Tensor] = {"protein": protein_prediction}
        if return_hidden:
            output["rna_hidden"] = rna_hidden
            output["protein_hidden"] = protein_hidden
        if protein_only:
            return output
        output.update(
            self.site_decoder(
                rna_hidden,
                rna_mask,
                protein_prediction,
                protein_hidden,
                self.protein_decoder.protein_identity.weight,
                site_indices,
            )
        )
        return output

    def fix_projection_matrices_(self) -> "RNAProteinPhosphositeTranslator":
        self.rna_encoder.encoder.fix_projection_matrices_()
        self.protein_decoder.decoder.fix_projection_matrices_()
        self.site_decoder.direct_attention.fix_projection_matrices_()
        return self

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
