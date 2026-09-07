"""Compact RNA-to-protein candidate based on latent state queries.

The model keeps the existing Performer RNA encoder, compresses its output into
a small learned state-token set, and lets protein identity queries read those
states with FAVOR+ cross-attention.  Protein queries never attend directly to
the complete RNA axis, so no protein-by-RNA attention matrix is materialized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

try:
    from .model import RNAEncoder
    from .performer_core import PerformerCrossAttention
except ImportError:
    from model import RNAEncoder
    from performer_core import PerformerCrossAttention


@dataclass(frozen=True)
class QueryLatentConfig:
    """Architecture settings for the query-latent protein candidate."""

    n_genes: int
    n_proteins: int
    n_state_tokens: int = 256
    d_model: int = 128
    encoder_depth: int = 2
    n_heads: int = 4
    dim_head: int = 32
    ff_mult: int = 4
    dropout: float = 0.10
    gene_mask_probability: float = 0.10
    nb_features: int | None = None
    feature_redraw_interval: int | None = 1000
    protein_query_chunk: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RNAProteinQueryLatent(nn.Module):
    """Predict all proteins through state-token and cognate-RNA readouts.

    ``parent_gene_index[j]`` gives the RNA-axis position of protein ``j``.
    A value of ``-1`` means that no cognate RNA is available; its local state
    and local gate are then exactly zero for every sample.
    """

    def __init__(
        self,
        config: QueryLatentConfig,
        parent_gene_index: Tensor | Sequence[int],
    ) -> None:
        super().__init__()
        if config.n_genes < 1 or config.n_proteins < 1:
            raise ValueError("n_genes and n_proteins must be positive")
        if config.n_state_tokens < 1:
            raise ValueError("n_state_tokens must be positive")
        if config.protein_query_chunk < 0:
            raise ValueError("protein_query_chunk cannot be negative")

        parent_index = torch.as_tensor(parent_gene_index, dtype=torch.long)
        if parent_index.ndim != 1 or parent_index.numel() != config.n_proteins:
            raise ValueError("parent_gene_index must contain one entry per protein")
        if (parent_index < -1).any() or (parent_index >= config.n_genes).any():
            raise ValueError("parent_gene_index entries must be -1 or valid gene indices")

        self.config = config
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
        self.protein_reader = PerformerCrossAttention(**attention_kwargs)
        self.protein_global_norm = nn.LayerNorm(config.d_model)
        self.cognate_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.fusion_gate = nn.Linear(config.d_model * 3, config.d_model)
        self.fusion_norm = nn.LayerNorm(config.d_model)
        self.abundance_head = nn.Linear(config.d_model, 1)

        self.register_buffer("parent_gene_index", parent_index, persistent=True)
        self.register_buffer(
            "protein_indices",
            torch.arange(config.n_proteins, dtype=torch.long),
            persistent=False,
        )

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
        batch_size = rna_hidden.shape[0]
        queries = self.state_queries.unsqueeze(0).expand(batch_size, -1, -1)
        states = self.state_norm(
            queries
            + self.state_reader(
                queries,
                rna_hidden,
                context_mask=rna_effective_mask,
            )
        )
        return states + self.state_feed_forward(states)

    def _cognate_hidden(
        self,
        rna_hidden: Tensor,
        rna_effective_mask: Tensor,
        start: int,
        end: int,
    ) -> tuple[Tensor, Tensor]:
        parent_index = self.parent_gene_index[start:end]
        mapped = parent_index >= 0
        safe_index = parent_index.clamp_min(0)
        local = rna_hidden.index_select(1, safe_index)
        sample_valid = rna_effective_mask.index_select(1, safe_index)
        valid = sample_valid & mapped.unsqueeze(0)
        local = self.cognate_projection(local)
        local = local * valid.unsqueeze(-1).to(local.dtype)
        return local, valid

    def _read_protein_chunk(
        self,
        state_hidden: Tensor,
        rna_hidden: Tensor,
        rna_effective_mask: Tensor,
        start: int,
        end: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = state_hidden.shape[0]
        identity = self.protein_identity(self.protein_indices[start:end])
        identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
        global_hidden = self.protein_global_norm(
            identity + self.protein_reader(identity, state_hidden)
        )
        cognate_hidden, cognate_valid = self._cognate_hidden(
            rna_hidden,
            rna_effective_mask,
            start,
            end,
        )
        gate = torch.sigmoid(
            self.fusion_gate(
                torch.cat((global_hidden, cognate_hidden, identity), dim=-1)
            )
        )
        gate = gate * cognate_valid.unsqueeze(-1).to(gate.dtype)
        fused = self.fusion_norm(
            (1.0 - gate) * global_hidden + gate * cognate_hidden
        )
        abundance = self.abundance_head(fused).squeeze(-1)
        return abundance, fused, gate

    def decode_proteins(
        self,
        state_hidden: Tensor,
        rna_hidden: Tensor,
        rna_effective_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        chunk = self.config.protein_query_chunk or self.config.n_proteins
        predictions: list[Tensor] = []
        hidden_chunks: list[Tensor] = []
        gate_chunks: list[Tensor] = []
        for start in range(0, self.config.n_proteins, chunk):
            end = min(start + chunk, self.config.n_proteins)
            prediction, hidden, gate = self._read_protein_chunk(
                state_hidden,
                rna_hidden,
                rna_effective_mask,
                start,
                end,
            )
            predictions.append(prediction)
            hidden_chunks.append(hidden)
            gate_chunks.append(gate)
        return (
            torch.cat(predictions, dim=1),
            torch.cat(hidden_chunks, dim=1),
            torch.cat(gate_chunks, dim=1),
        )

    def forward(
        self,
        expression: Tensor,
        rna_valid_mask: Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, Tensor]:
        rna_hidden, effective_mask = self.encode_rna(expression, rna_valid_mask)
        state_hidden = self.aggregate_state_tokens(rna_hidden, effective_mask)
        protein, protein_hidden, cognate_gate = self.decode_proteins(
            state_hidden,
            rna_hidden,
            effective_mask,
        )
        output = {"protein": protein}
        if return_hidden:
            output.update(
                {
                    "rna_hidden": rna_hidden,
                    "rna_effective_mask": effective_mask,
                    "state_hidden": state_hidden,
                    "protein_hidden": protein_hidden,
                    "cognate_gate": cognate_gate,
                }
            )
        return output

    def fix_projection_matrices_(self) -> "RNAProteinQueryLatent":
        self.rna_encoder.encoder.fix_projection_matrices_()
        self.state_reader.fix_projection_matrices_()
        self.protein_reader.fix_projection_matrices_()
        return self

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)

    def parameter_breakdown(self) -> dict[str, int]:
        """Return non-overlapping parameter counts for the major modules."""

        groups = {
            "rna_encoder": self.rna_encoder,
            "state_reader": self.state_reader,
            "state_feed_forward": self.state_feed_forward,
            "protein_identity": self.protein_identity,
            "protein_reader": self.protein_reader,
            "cognate_projection": self.cognate_projection,
            "fusion_gate": self.fusion_gate,
            "shared_abundance_head": self.abundance_head,
        }
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in groups.items()
        }
        counted = sum(counts.values())
        counts["other_norms_and_state_tokens"] = self.parameter_count() - counted
        counts["total"] = self.parameter_count()
        return counts

    def complexity(self, batch_size: int = 1) -> dict[str, int | str]:
        """Describe the two linear-attention contractions used in a forward."""

        return {
            "batch_size": batch_size,
            "rna_tokens": self.config.n_genes,
            "state_tokens": self.config.n_state_tokens,
            "protein_queries": self.config.n_proteins,
            "attention_scaling": "O(B * H * M * D_h * (G + 2L + P))",
            "largest_explicit_attention_matrix": 0,
        }
