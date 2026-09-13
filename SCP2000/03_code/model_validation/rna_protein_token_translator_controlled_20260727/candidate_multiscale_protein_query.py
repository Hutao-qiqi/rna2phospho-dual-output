"""Multiscale protein-query translator and phosphosite residual decoder.

The protein model keeps a complete fixed protein vocabulary without a dense
RNA-position to protein-position matrix. Each protein reads two complementary
RNA contexts: a fixed local biological neighbourhood and a compact set of
sample-specific global state tokens. A split-local cognate-RNA linear anchor
provides the interpretable starting prediction.

The phosphosite decoder reuses the encoded RNA and global states. Its static
site query contains site, parent-protein, and kinase-prior identities. Parent
protein abundance enters only through the explicit anchor term.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

try:
    from .model import RNAEncoder
    from .performer_core import PerformerCrossAttention, PerformerEncoder
    from .protein_graph_prior import ProteinGraphArtifact
except ImportError:
    from model import RNAEncoder
    from performer_core import PerformerCrossAttention, PerformerEncoder
    from protein_graph_prior import ProteinGraphArtifact


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("gate initialization must lie strictly between zero and one")
    return math.log(probability / (1.0 - probability))


@dataclass(frozen=True)
class LocalGenePrior:
    """Fixed target-to-RNA neighbourhood prepared before model fitting."""

    gene_index: np.ndarray
    gene_mask: np.ndarray
    relation_index: np.ndarray
    edge_strength: np.ndarray
    relation_names: tuple[str, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        gene_index = np.asarray(self.gene_index, dtype=np.int64)
        gene_mask = np.asarray(self.gene_mask, dtype=bool)
        relation_index = np.asarray(self.relation_index, dtype=np.int64)
        edge_strength = np.asarray(self.edge_strength, dtype=np.float32)
        if gene_index.ndim != 2:
            raise ValueError("local gene arrays must be two-dimensional")
        if not (
            gene_mask.shape
            == relation_index.shape
            == edge_strength.shape
            == gene_index.shape
        ):
            raise ValueError("local gene arrays must have identical shapes")
        if not self.relation_names:
            raise ValueError("at least one relation type is required")
        if gene_mask.any():
            if gene_index[gene_mask].min() < 0:
                raise ValueError("valid local gene indices cannot be negative")
            valid_relations = relation_index[gene_mask]
            if valid_relations.min() < 0 or valid_relations.max() >= len(
                self.relation_names
            ):
                raise ValueError("valid relation indices are outside relation_names")
            if not np.isfinite(edge_strength[gene_mask]).all():
                raise ValueError("valid edge strengths must be finite")
            if (edge_strength[gene_mask] <= 0).any():
                raise ValueError("valid edge strengths must be positive")
        object.__setattr__(self, "gene_index", gene_index)
        object.__setattr__(self, "gene_mask", gene_mask)
        object.__setattr__(self, "relation_index", relation_index)
        object.__setattr__(self, "edge_strength", edge_strength)
        object.__setattr__(
            self, "relation_names", tuple(str(value) for value in self.relation_names)
        )

    @property
    def n_targets(self) -> int:
        return int(self.gene_index.shape[0])

    @property
    def max_local_genes(self) -> int:
        return int(self.gene_index.shape[1])

    def audit(self) -> dict[str, Any]:
        counts = self.gene_mask.sum(axis=1)
        relation_counts = {
            name: int(
                np.sum(self.gene_mask & (self.relation_index == relation_index))
            )
            for relation_index, name in enumerate(self.relation_names)
        }
        return {
            **self.metadata,
            "n_targets": self.n_targets,
            "max_local_genes": self.max_local_genes,
            "targets_with_any_local_gene": int(np.sum(counts > 0)),
            "targets_with_full_local_set": int(np.sum(counts == self.max_local_genes)),
            "local_gene_count_median": float(np.median(counts)),
            "local_gene_count_mean": float(np.mean(counts)),
            "relation_counts": relation_counts,
        }


def build_protein_local_gene_prior(
    gene_names: Sequence[str],
    protein_names: Sequence[str],
    parent_gene_index: Sequence[int] | np.ndarray,
    graph: ProteinGraphArtifact,
    *,
    max_local_genes: int = 64,
) -> LocalGenePrior:
    """Build cognate-plus-STRING local RNA sets without using target labels."""

    genes = tuple(str(value) for value in gene_names)
    proteins = tuple(str(value) for value in protein_names)
    parent = np.asarray(parent_gene_index, dtype=np.int64)
    if max_local_genes < 1:
        raise ValueError("max_local_genes must be positive")
    if len(set(genes)) != len(genes):
        raise ValueError("gene_names must be unique")
    if len(set(proteins)) != len(proteins):
        raise ValueError("protein_names must be unique")
    if parent.shape != (len(proteins),):
        raise ValueError("parent_gene_index must contain one entry per protein")
    if (parent < -1).any() or (parent >= len(genes)).any():
        raise ValueError("parent_gene_index contains an invalid gene index")
    if graph.protein_names != proteins:
        raise ValueError("protein graph order must match the output protein order")

    gene_lookup = {name: index for index, name in enumerate(genes)}
    adjacency: list[list[tuple[float, int]]] = [[] for _ in proteins]
    for position, (left, right) in enumerate(graph.edge_index.T):
        score = float(graph.edge_score[position])
        normalized = score / 1000.0 if score > 1.0 else score
        normalized = min(max(normalized, 1e-4), 1.0)
        adjacency[int(left)].append((normalized, int(right)))
        adjacency[int(right)].append((normalized, int(left)))

    shape = (len(proteins), max_local_genes)
    local_index = np.full(shape, -1, dtype=np.int64)
    local_mask = np.zeros(shape, dtype=bool)
    relation_index = np.full(shape, -1, dtype=np.int64)
    edge_strength = np.zeros(shape, dtype=np.float32)

    for protein_index, neighbours in enumerate(adjacency):
        candidates: list[tuple[int, float, int]] = []
        seen: set[int] = set()
        cognate = int(parent[protein_index])
        if cognate >= 0:
            candidates.append((cognate, 1.0, 0))
            seen.add(cognate)
        ordered = sorted(
            neighbours,
            key=lambda item: (-item[0], proteins[item[1]], item[1]),
        )
        for score, neighbour in ordered:
            gene_index = gene_lookup.get(proteins[neighbour])
            if gene_index is None or gene_index in seen:
                continue
            candidates.append((gene_index, score, 1))
            seen.add(gene_index)
            if len(candidates) >= max_local_genes:
                break
        for offset, (gene_index, strength, relation) in enumerate(candidates):
            local_index[protein_index, offset] = gene_index
            local_mask[protein_index, offset] = True
            relation_index[protein_index, offset] = relation
            edge_strength[protein_index, offset] = strength

    return LocalGenePrior(
        gene_index=local_index,
        gene_mask=local_mask,
        relation_index=relation_index,
        edge_strength=edge_strength,
        relation_names=("cognate_rna", "string_interactor"),
        metadata={
            "construction": "cognate RNA plus fixed STRING neighbours",
            "graph_mode": graph.graph_mode,
            "graph_edges": graph.n_edges,
            "uses_protein_abundance_labels": False,
            "uses_phosphosite_labels": False,
            "uses_validation_samples": False,
        },
    )


@dataclass(frozen=True)
class MultiscaleProteinQueryConfig:
    """Settings for the multiscale RNA-to-protein query model."""

    n_genes: int
    n_proteins: int
    n_state_tokens: int = 128
    max_local_genes: int = 64
    d_model: int = 128
    encoder_depth: int = 2
    n_heads: int = 4
    dim_head: int = 32
    ff_mult: int = 2
    dropout: float = 0.10
    gene_mask_probability: float = 0.0
    nb_features: int | None = 128
    feature_redraw_interval: int | None = None
    protein_query_chunk: int = 512
    protein_decoder_depth: int = 0
    initial_local_gate: float = 0.10
    initial_global_gate: float = 0.10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExactLocalGeneAttention(nn.Module):
    """Exact multi-head attention over a small fixed target-specific RNA set."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
        prior: LocalGenePrior,
    ) -> None:
        super().__init__()
        if heads < 1 or dim_head < 1:
            raise ValueError("heads and dim_head must be positive")
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head**-0.5
        self.context_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.relation_embedding = nn.Embedding(len(prior.relation_names), dim)
        self.strength_projection = nn.Linear(1, dim, bias=False)
        self.to_query = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_key = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_value = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_output = nn.Linear(self.inner_dim, dim, bias=False)
        self.output_dropout = nn.Dropout(dropout)
        self.prior_logit_scale = nn.Parameter(torch.tensor(0.0))
        self.register_buffer(
            "local_gene_index",
            torch.as_tensor(prior.gene_index, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "local_gene_mask",
            torch.as_tensor(prior.gene_mask, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "local_relation_index",
            torch.as_tensor(prior.relation_index, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "local_edge_strength",
            torch.as_tensor(prior.edge_strength, dtype=torch.float32),
            persistent=True,
        )

    def _split_query(self, value: Tensor) -> Tensor:
        batch, targets, _ = value.shape
        return value.reshape(batch, targets, self.heads, self.dim_head)

    def _split_context(self, value: Tensor) -> Tensor:
        batch, targets, local, _ = value.shape
        return value.reshape(batch, targets, local, self.heads, self.dim_head)

    def forward(
        self,
        query: Tensor,
        rna_hidden: Tensor,
        rna_valid_mask: Tensor,
        target_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if query.ndim != 3 or rna_hidden.ndim != 3:
            raise ValueError("query and rna_hidden must be three-dimensional")
        selected = target_indices.to(device=query.device, dtype=torch.long)
        gene_index = self.local_gene_index.index_select(0, selected)
        prior_mask = self.local_gene_mask.index_select(0, selected)
        relation_index = self.local_relation_index.index_select(0, selected)
        strength = self.local_edge_strength.index_select(0, selected)
        safe_gene_index = gene_index.clamp_min(0)
        batch = rna_hidden.shape[0]
        targets, local = safe_gene_index.shape

        gathered = rna_hidden.index_select(1, safe_gene_index.reshape(-1))
        gathered = gathered.reshape(batch, targets, local, self.dim)
        sample_mask = rna_valid_mask.index_select(1, safe_gene_index.reshape(-1))
        sample_mask = sample_mask.reshape(batch, targets, local)
        valid = sample_mask & prior_mask.unsqueeze(0)

        safe_relation = relation_index.clamp_min(0)
        relation = self.relation_embedding(safe_relation).unsqueeze(0)
        strength_hidden = self.strength_projection(
            strength.unsqueeze(-1)
        ).unsqueeze(0)
        context = self.context_norm(gathered) + relation + strength_hidden

        projected_query = self._split_query(self.to_query(self.query_norm(query)))
        projected_key = self._split_context(self.to_key(context))
        projected_value = self._split_context(self.to_value(context))
        logits = torch.einsum(
            "bthd,btlhd->bthl", projected_query, projected_key
        ) * self.scale
        log_strength = torch.log(strength.clamp_min(1e-4)).unsqueeze(0).unsqueeze(2)
        logits = logits + self.prior_logit_scale * log_strength

        attention_mask = valid.unsqueeze(2)
        has_context = attention_mask.any(dim=-1, keepdim=True)
        minimum = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~attention_mask, minimum)
        logits = torch.where(has_context, logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=-1)
        weights = weights * attention_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attended = torch.einsum("bthl,btlhd->bthd", weights, projected_value)
        attended = attended.reshape(batch, targets, self.inner_dim)
        output = self.output_dropout(self.to_output(attended))
        valid_target = valid.any(dim=-1)
        output = output * valid_target.unsqueeze(-1).to(output.dtype)
        return output, valid_target


def _residual_head(dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(dim * 2),
        nn.Linear(dim * 2, dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim, 1),
    )


def _gate_network(
    dim: int,
    dropout: float,
    initial_probabilities: Sequence[float],
) -> nn.Sequential:
    network = nn.Sequential(
        nn.LayerNorm(dim * 3),
        nn.Linear(dim * 3, dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim, len(initial_probabilities)),
    )
    final = network[-1]
    assert isinstance(final, nn.Linear)
    nn.init.zeros_(final.weight)
    with torch.no_grad():
        final.bias.copy_(
            torch.tensor([_logit(value) for value in initial_probabilities])
        )
    return network


class MultiscaleProteinQueryTranslator(nn.Module):
    """Protein identities read local RNA and global sample-state contexts."""

    def __init__(
        self,
        config: MultiscaleProteinQueryConfig,
        parent_gene_index: Tensor | Sequence[int],
        anchor_intercept: Tensor | Sequence[float],
        anchor_slope: Tensor | Sequence[float],
        local_gene_prior: LocalGenePrior,
    ) -> None:
        super().__init__()
        if config.n_genes < 1 or config.n_proteins < 1:
            raise ValueError("n_genes and n_proteins must be positive")
        if config.n_state_tokens < 1 or config.protein_query_chunk < 1:
            raise ValueError("state tokens and protein query chunk must be positive")
        if config.max_local_genes < 1:
            raise ValueError("max_local_genes must be positive")
        if config.protein_decoder_depth not in {0, 1}:
            raise ValueError("protein_decoder_depth must be zero or one")
        if local_gene_prior.n_targets != config.n_proteins:
            raise ValueError("local_gene_prior must contain one row per protein")
        if local_gene_prior.max_local_genes != config.max_local_genes:
            raise ValueError("local_gene_prior width must equal max_local_genes")
        if local_gene_prior.gene_mask.any() and (
            local_gene_prior.gene_index[local_gene_prior.gene_mask].max()
            >= config.n_genes
        ):
            raise ValueError("local gene prior contains an invalid RNA index")

        parent = torch.as_tensor(parent_gene_index, dtype=torch.long)
        intercept = torch.as_tensor(anchor_intercept, dtype=torch.float32)
        slope = torch.as_tensor(anchor_slope, dtype=torch.float32)
        expected = (config.n_proteins,)
        if parent.shape != expected:
            raise ValueError("parent_gene_index must contain one entry per protein")
        if intercept.shape != expected or slope.shape != expected:
            raise ValueError("anchor coefficients must contain one entry per protein")
        if (parent < -1).any() or (parent >= config.n_genes).any():
            raise ValueError("parent_gene_index contains an invalid RNA index")

        self.config = config
        self.local_prior_metadata = dict(local_gene_prior.metadata)
        self.local_relation_names = local_gene_prior.relation_names
        self.rna_encoder = RNAEncoder(config)
        self.state_queries = nn.Parameter(
            torch.empty(config.n_state_tokens, config.d_model)
        )
        nn.init.normal_(self.state_queries, std=config.d_model**-0.5)

        attention_kwargs = {
            "dim": config.d_model,
            "heads": config.n_heads,
            "dim_head": config.dim_head,
            "dropout": config.dropout,
            "nb_features": config.nb_features,
            "feature_redraw_interval": config.feature_redraw_interval,
        }
        self.state_reader = PerformerCrossAttention(**attention_kwargs)
        self.state_norm = nn.LayerNorm(config.d_model)
        self.state_feed_forward = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model * config.ff_mult),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * config.ff_mult, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.protein_identity = nn.Embedding(config.n_proteins, config.d_model)
        self.local_reader = ExactLocalGeneAttention(
            config.d_model,
            config.n_heads,
            config.dim_head,
            config.dropout,
            local_gene_prior,
        )
        self.global_reader = PerformerCrossAttention(**attention_kwargs)
        self.global_norm = nn.LayerNorm(config.d_model)
        self.context_gate = _gate_network(
            config.d_model,
            config.dropout,
            (config.initial_local_gate, config.initial_global_gate),
        )
        self.local_residual_head = _residual_head(config.d_model, config.dropout)
        self.global_residual_head = _residual_head(config.d_model, config.dropout)
        self.local_hidden_projection = nn.Linear(config.d_model, config.d_model)
        self.global_hidden_projection = nn.Linear(config.d_model, config.d_model)
        self.protein_hidden_norm = nn.LayerNorm(config.d_model)

        if config.protein_decoder_depth:
            self.protein_decoder = PerformerEncoder(
                dim=config.d_model,
                depth=1,
                heads=config.n_heads,
                dim_head=config.dim_head,
                dropout=config.dropout,
                ff_mult=config.ff_mult,
                nb_features=config.nb_features,
                feature_redraw_interval=config.feature_redraw_interval,
            )
            self.protein_decoder_head = nn.Linear(config.d_model, 1)
        else:
            self.protein_decoder = None
            self.protein_decoder_head = None

        self.register_buffer("parent_gene_index", parent, persistent=True)
        self.register_buffer("anchor_intercept", intercept, persistent=True)
        self.register_buffer("anchor_slope", slope, persistent=True)
        self.register_buffer(
            "protein_indices",
            torch.arange(config.n_proteins, dtype=torch.long),
            persistent=False,
        )

    def linear_anchor(self, expression: Tensor) -> Tensor:
        safe_index = self.parent_gene_index.clamp_min(0)
        cognate = expression.index_select(1, safe_index)
        mapped = (self.parent_gene_index >= 0).to(cognate.dtype)
        return (self.anchor_intercept + self.anchor_slope * cognate) * mapped

    def encode_rna(
        self,
        expression: Tensor,
        rna_valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.rna_encoder(expression, rna_valid_mask)

    def aggregate_state_tokens(
        self,
        rna_hidden: Tensor,
        rna_effective_mask: Tensor,
    ) -> Tensor:
        queries = self.state_queries.unsqueeze(0).expand(rna_hidden.shape[0], -1, -1)
        states = self.state_norm(
            queries
            + self.state_reader(
                queries,
                rna_hidden,
                context_mask=rna_effective_mask,
            )
        )
        return states + self.state_feed_forward(states)

    def _decode_chunk(
        self,
        selected: Tensor,
        state_hidden: Tensor,
        rna_hidden: Tensor,
        rna_effective_mask: Tensor,
    ) -> dict[str, Tensor]:
        identity = self.protein_identity(selected).unsqueeze(0)
        identity = identity.expand(rna_hidden.shape[0], -1, -1)
        local_context, local_valid = self.local_reader(
            identity,
            rna_hidden,
            rna_effective_mask,
            selected,
        )
        global_context = self.global_norm(
            identity + self.global_reader(identity, state_hidden)
        )
        gates = torch.sigmoid(
            self.context_gate(
                torch.cat((identity, local_context, global_context), dim=-1)
            )
        )
        local_gate = gates[..., 0] * local_valid.to(gates.dtype)
        global_gate = gates[..., 1]
        local_delta = self.local_residual_head(
            torch.cat((identity, local_context), dim=-1)
        ).squeeze(-1)
        global_delta = self.global_residual_head(
            torch.cat((identity, global_context), dim=-1)
        ).squeeze(-1)
        residual = local_gate * local_delta + global_gate * global_delta
        protein_hidden = self.protein_hidden_norm(
            identity
            + local_gate.unsqueeze(-1) * self.local_hidden_projection(local_context)
            + global_gate.unsqueeze(-1) * self.global_hidden_projection(global_context)
        )
        return {
            "residual": residual,
            "local_residual": local_delta,
            "global_residual": global_delta,
            "local_gate": local_gate,
            "global_gate": global_gate,
            "protein_hidden": protein_hidden,
        }

    def decode_proteins(
        self,
        state_hidden: Tensor,
        rna_hidden: Tensor,
        rna_effective_mask: Tensor,
    ) -> dict[str, Tensor]:
        parts: dict[str, list[Tensor]] = {
            "residual": [],
            "local_residual": [],
            "global_residual": [],
            "local_gate": [],
            "global_gate": [],
            "protein_hidden": [],
        }
        chunk = self.config.protein_query_chunk
        for start in range(0, self.config.n_proteins, chunk):
            selected = self.protein_indices[
                start : min(start + chunk, self.config.n_proteins)
            ]
            decoded = self._decode_chunk(
                selected,
                state_hidden,
                rna_hidden,
                rna_effective_mask,
            )
            for name, value in decoded.items():
                parts[name].append(value)
        output = {name: torch.cat(values, dim=1) for name, values in parts.items()}
        if self.protein_decoder is not None:
            decoded_hidden = self.protein_decoder(output["protein_hidden"])
            decoder_delta = self.protein_decoder_head(decoded_hidden).squeeze(-1)
            output["protein_hidden"] = decoded_hidden
        else:
            decoder_delta = torch.zeros_like(output["residual"])
        output["decoder_residual"] = decoder_delta
        output["residual"] = output["residual"] + decoder_delta
        return output

    def forward(
        self,
        expression: Tensor,
        rna_valid_mask: Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, Tensor]:
        rna_hidden, effective_mask = self.encode_rna(expression, rna_valid_mask)
        state_hidden = self.aggregate_state_tokens(rna_hidden, effective_mask)
        decoded = self.decode_proteins(state_hidden, rna_hidden, effective_mask)
        anchor = self.linear_anchor(expression)
        output: dict[str, Tensor] = {"protein": anchor + decoded["residual"]}
        if return_hidden:
            output.update(
                {
                    "linear_anchor": anchor,
                    "protein_residual": decoded["residual"],
                    "protein_local_residual": decoded["local_residual"],
                    "protein_global_residual": decoded["global_residual"],
                    "protein_decoder_residual": decoded["decoder_residual"],
                    "protein_local_gate": decoded["local_gate"],
                    "protein_global_gate": decoded["global_gate"],
                    "rna_hidden": rna_hidden,
                    "rna_effective_mask": effective_mask,
                    "state_hidden": state_hidden,
                    "protein_hidden": decoded["protein_hidden"],
                }
            )
        return output

    def fix_projection_matrices_(self) -> "MultiscaleProteinQueryTranslator":
        self.rna_encoder.encoder.fix_projection_matrices_()
        self.state_reader.fix_projection_matrices_()
        self.global_reader.fix_projection_matrices_()
        if self.protein_decoder is not None:
            self.protein_decoder.fix_projection_matrices_()
        return self

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (value for value in parameters if value.requires_grad)
        return sum(value.numel() for value in parameters)

    def parameter_breakdown(self) -> dict[str, int]:
        modules = {
            "rna_encoder": self.rna_encoder,
            "state_reader": self.state_reader,
            "state_feed_forward": self.state_feed_forward,
            "protein_identity": self.protein_identity,
            "local_reader": self.local_reader,
            "global_reader": self.global_reader,
            "context_gate": self.context_gate,
            "local_residual_head": self.local_residual_head,
            "global_residual_head": self.global_residual_head,
        }
        if self.protein_decoder is not None:
            modules["protein_decoder"] = self.protein_decoder
            modules["protein_decoder_head"] = self.protein_decoder_head
        counts = {
            name: sum(value.numel() for value in module.parameters())
            for name, module in modules.items()
        }
        counted = sum(counts.values())
        counts["other"] = self.parameter_count() - counted
        counts["total"] = self.parameter_count()
        return counts

    def architecture_contract(self) -> dict[str, Any]:
        return {
            "dense_rna_to_protein_axis_map": False,
            "complete_fixed_protein_axis": True,
            "split_local_cognate_linear_anchor": True,
            "global_state_tokens": self.config.n_state_tokens,
            "fixed_local_gene_prior": True,
            "max_local_genes": self.config.max_local_genes,
            "local_prior_relations": list(self.local_relation_names),
            "sample_protein_specific_scalar_gates": True,
            "protein_decoder_depth": self.config.protein_decoder_depth,
            "protein_query_chunk": self.config.protein_query_chunk,
            "protein_labels_are_model_inputs": False,
            "cross_sample_message_passing": False,
        }


@dataclass(frozen=True)
class MultiscalePhosphositeConfig:
    n_sites: int
    n_kinases: int
    d_model: int = 128
    n_heads: int = 4
    dim_head: int = 32
    dropout: float = 0.10
    nb_features: int | None = 128
    feature_redraw_interval: int | None = None
    site_query_chunk: int = 512
    initial_pathway_gate: float = 0.10
    detach_parent_prediction: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiscalePhosphositeResidualHead(nn.Module):
    """Parent-anchored site decoder with disjoint local and global contexts."""

    def __init__(
        self,
        config: MultiscalePhosphositeConfig,
        n_proteins: int,
        site_to_protein_index: Tensor | Sequence[int],
        site_kinase_weights: Tensor | np.ndarray,
        local_gene_prior: LocalGenePrior,
    ) -> None:
        super().__init__()
        if local_gene_prior.n_targets != config.n_sites:
            raise ValueError("site local prior must contain one row per site")
        parent = torch.as_tensor(site_to_protein_index, dtype=torch.long)
        if parent.shape != (config.n_sites,):
            raise ValueError("site_to_protein_index must contain one entry per site")
        if (parent < -1).any() or (parent >= n_proteins).any():
            raise ValueError("site_to_protein_index contains an invalid protein index")
        kinase_weights = torch.as_tensor(site_kinase_weights, dtype=torch.float32)
        if kinase_weights.shape != (config.n_sites, config.n_kinases):
            raise ValueError("site_kinase_weights has an incompatible shape")
        if (kinase_weights < 0).any() or not torch.isfinite(kinase_weights).all():
            raise ValueError("site_kinase_weights must be finite and non-negative")
        denominator = kinase_weights.sum(dim=1, keepdim=True)
        kinase_weights = torch.where(
            denominator > 0,
            kinase_weights / denominator.clamp_min(1e-8),
            kinase_weights,
        )

        self.config = config
        self.site_identity = nn.Embedding(config.n_sites, config.d_model)
        self.kinase_prior_identity = nn.Embedding(config.n_kinases, config.d_model)
        self.unknown_kinase_prior = nn.Parameter(torch.zeros(config.d_model))
        self.query_norm = nn.LayerNorm(config.d_model)
        self.local_reader = ExactLocalGeneAttention(
            config.d_model,
            config.n_heads,
            config.dim_head,
            config.dropout,
            local_gene_prior,
        )
        self.global_reader = PerformerCrossAttention(
            dim=config.d_model,
            heads=config.n_heads,
            dim_head=config.dim_head,
            dropout=config.dropout,
            nb_features=config.nb_features,
            feature_redraw_interval=config.feature_redraw_interval,
        )
        self.global_norm = nn.LayerNorm(config.d_model)
        self.direct_residual_head = _residual_head(config.d_model, config.dropout)
        self.pathway_residual_head = _residual_head(config.d_model, config.dropout)
        self.pathway_gate = _gate_network(
            config.d_model,
            config.dropout,
            (config.initial_pathway_gate,),
        )
        self.site_intercept = nn.Parameter(torch.zeros(config.n_sites))
        self.parent_scale = nn.Parameter(torch.ones(config.n_sites))

        self.register_buffer("site_to_protein_index", parent, persistent=True)
        self.register_buffer("site_kinase_weights", kinase_weights, persistent=True)
        self.register_buffer(
            "site_indices",
            torch.arange(config.n_sites, dtype=torch.long),
            persistent=False,
        )

    def _kinase_prior_embedding(self, selected: Tensor) -> Tensor:
        weights = self.site_kinase_weights.index_select(0, selected)
        known = torch.matmul(weights, self.kinase_prior_identity.weight)
        missing = weights.sum(dim=-1, keepdim=True) <= 0
        return known + missing.to(known.dtype) * self.unknown_kinase_prior

    def _decode_chunk(
        self,
        selected: Tensor,
        rna_hidden: Tensor,
        rna_mask: Tensor,
        state_hidden: Tensor,
        protein_prediction: Tensor,
        protein_identity_weight: Tensor,
    ) -> dict[str, Tensor]:
        parent_index = self.site_to_protein_index.index_select(0, selected)
        parent_valid = parent_index >= 0
        safe_parent = parent_index.clamp_min(0)
        parent_identity = protein_identity_weight.index_select(0, safe_parent)
        parent_identity = parent_identity * parent_valid.unsqueeze(-1)
        static_query = self.query_norm(
            self.site_identity(selected)
            + parent_identity
            + self._kinase_prior_embedding(selected)
        )
        query = static_query.unsqueeze(0).expand(rna_hidden.shape[0], -1, -1)
        direct_context, direct_valid = self.local_reader(
            query,
            rna_hidden,
            rna_mask,
            selected,
        )
        global_context = self.global_norm(
            query + self.global_reader(query, state_hidden)
        )
        direct = self.direct_residual_head(
            torch.cat((query, direct_context), dim=-1)
        ).squeeze(-1)
        direct = direct * direct_valid.to(direct.dtype)
        pathway = self.pathway_residual_head(
            torch.cat((query, global_context), dim=-1)
        ).squeeze(-1)
        gate = torch.sigmoid(
            self.pathway_gate(
                torch.cat((query, direct_context, global_context), dim=-1)
            )
        ).squeeze(-1)

        parent_source = (
            protein_prediction.detach()
            if self.config.detach_parent_prediction
            else protein_prediction
        )
        parent_value = parent_source.index_select(1, safe_parent)
        anchor = self.site_intercept.index_select(0, selected).unsqueeze(0)
        anchor = anchor + (
            self.parent_scale.index_select(0, selected).unsqueeze(0)
            * parent_value
            * parent_valid.unsqueeze(0)
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
        state_hidden: Tensor,
        protein_prediction: Tensor,
        protein_identity_weight: Tensor,
        site_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        selected_sites = self.site_indices if site_indices is None else site_indices.long()
        parts: dict[str, list[Tensor]] = {
            "phosphosite": [],
            "phosphosite_anchor": [],
            "phosphosite_direct_residual": [],
            "phosphosite_pathway_residual": [],
            "phosphosite_pathway_gate": [],
            "phosphosite_residual": [],
        }
        for start in range(0, selected_sites.numel(), self.config.site_query_chunk):
            selected = selected_sites[
                start : min(start + self.config.site_query_chunk, selected_sites.numel())
            ]
            decoded = self._decode_chunk(
                selected,
                rna_hidden,
                rna_mask,
                state_hidden,
                protein_prediction,
                protein_identity_weight,
            )
            for name, value in decoded.items():
                parts[name].append(value)
        output = {name: torch.cat(values, dim=1) for name, values in parts.items()}
        output["site_indices"] = selected_sites
        return output

    def fix_projection_matrices_(self) -> "MultiscalePhosphositeResidualHead":
        self.global_reader.fix_projection_matrices_()
        return self


class RNAProteinPhosphositeQueryTranslator(nn.Module):
    """Shared RNA encoder with protein and phosphosite query decoders."""

    def __init__(
        self,
        protein_model: MultiscaleProteinQueryTranslator,
        phosphosite_head: MultiscalePhosphositeResidualHead,
    ) -> None:
        super().__init__()
        if protein_model.config.d_model != phosphosite_head.config.d_model:
            raise ValueError("protein model and phosphosite head dimensions must match")
        self.protein_model = protein_model
        self.phosphosite_head = phosphosite_head

    def forward(
        self,
        expression: Tensor,
        rna_valid_mask: Tensor | None = None,
        site_indices: Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, Tensor]:
        protein_output = self.protein_model(
            expression,
            rna_valid_mask=rna_valid_mask,
            return_hidden=True,
        )
        site_output = self.phosphosite_head(
            protein_output["rna_hidden"],
            protein_output["rna_effective_mask"],
            protein_output["state_hidden"],
            protein_output["protein"],
            self.protein_model.protein_identity.weight,
            site_indices=site_indices,
        )
        output = {
            "protein": protein_output["protein"],
            **site_output,
        }
        if return_hidden:
            output.update(
                {
                    name: value
                    for name, value in protein_output.items()
                    if name != "protein"
                }
            )
        return output

    def fix_projection_matrices_(self) -> "RNAProteinPhosphositeQueryTranslator":
        self.protein_model.fix_projection_matrices_()
        self.phosphosite_head.fix_projection_matrices_()
        return self


__all__ = [
    "ExactLocalGeneAttention",
    "LocalGenePrior",
    "MultiscalePhosphositeConfig",
    "MultiscalePhosphositeResidualHead",
    "MultiscaleProteinQueryConfig",
    "MultiscaleProteinQueryTranslator",
    "RNAProteinPhosphositeQueryTranslator",
    "build_protein_local_gene_prior",
]
