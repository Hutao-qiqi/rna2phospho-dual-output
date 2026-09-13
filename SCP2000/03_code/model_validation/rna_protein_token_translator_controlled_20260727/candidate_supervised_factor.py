"""R3: supervised protein-state factors with a gated cognate-RNA shortcut.

The RNA encoder is shared with the other candidates. A small set of learned
queries summarizes the encoded transcriptome, after which an end-to-end
supervised factor head produces ``K`` sample-level protein-state factors.
Protein-specific loadings decode those factors into all protein abundances.
The only local shortcut is a protein-specific linear function of the cognate
RNA expression. No gene-by-protein parameter or attention matrix is created.

For sample ``i`` and protein ``j`` the model is

    H_i = RNAEncoder(x_i)
    T_i = Q + LinearAttention(Q, H_i, H_i)
    s_i = sum_q softmax(a)_q T_iq
    f_i = FactorHead(s_i),                         f_i in R^K
    p_factor_ij = b_j + <f_i, L_j>
    d_ij = a_j + beta_j x_i,parent(j)
    g_ij = m_ij sigmoid(eta_j + u_j x_i,parent(j) + v_j p_factor_ij)
    p_hat_ij = p_factor_ij + g_ij d_ij

where ``m_ij`` is zero when the protein has no mapped cognate RNA or when that
RNA token is masked. Consequently, ``parent_gene_index == -1`` always falls
back to the supervised factor decoder.
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
class SupervisedFactorConfig:
    """Architecture settings for the supervised-factor candidate."""

    n_genes: int
    n_proteins: int
    n_factors: int = 256
    n_sample_queries: int = 16
    d_model: int = 128
    encoder_depth: int = 2
    n_heads: int = 4
    dim_head: int = 32
    ff_mult: int = 4
    dropout: float = 0.10
    gene_mask_probability: float = 0.10
    nb_features: int | None = None
    feature_redraw_interval: int | None = 1000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SupervisedFactorProteinTranslator(nn.Module):
    """Decode protein abundance from supervised factors and cognate RNA.

    ``parent_gene_index[j]`` stores the RNA-axis position of protein ``j``.
    ``-1`` denotes a protein without a cognate RNA in the fixed vocabulary.
    """

    def __init__(
        self,
        config: SupervisedFactorConfig,
        parent_gene_index: Tensor | Sequence[int],
    ) -> None:
        super().__init__()
        if config.n_genes < 1 or config.n_proteins < 1:
            raise ValueError("n_genes and n_proteins must be positive")
        if config.n_factors < 1 or config.n_sample_queries < 1:
            raise ValueError("n_factors and n_sample_queries must be positive")

        parent_index = torch.as_tensor(parent_gene_index, dtype=torch.long)
        if parent_index.ndim != 1 or parent_index.numel() != config.n_proteins:
            raise ValueError("parent_gene_index must contain one entry per protein")
        if (parent_index < -1).any() or (parent_index >= config.n_genes).any():
            raise ValueError("parent_gene_index entries must be -1 or valid gene indices")

        self.config = config
        self.rna_encoder = RNAEncoder(config)

        self.sample_queries = nn.Parameter(
            torch.empty(config.n_sample_queries, config.d_model)
        )
        nn.init.normal_(self.sample_queries, std=config.d_model**-0.5)
        self.sample_reader = PerformerCrossAttention(
            dim=config.d_model,
            heads=config.n_heads,
            dim_head=config.dim_head,
            dropout=config.dropout,
            nb_features=config.nb_features,
            feature_redraw_interval=config.feature_redraw_interval,
        )
        self.query_norm = nn.LayerNorm(config.d_model)
        self.query_pool_logits = nn.Parameter(torch.zeros(config.n_sample_queries))
        self.sample_norm = nn.LayerNorm(config.d_model)

        factor_hidden = max(config.d_model, config.n_factors)
        self.factor_head = nn.Sequential(
            nn.Linear(config.d_model, factor_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(factor_hidden, config.n_factors),
            nn.LayerNorm(config.n_factors),
        )

        # L[j, k] is the loading of protein j on supervised factor k.
        self.protein_loadings = nn.Parameter(
            torch.empty(config.n_proteins, config.n_factors)
        )
        nn.init.normal_(self.protein_loadings, std=config.n_factors**-0.5)
        self.protein_bias = nn.Parameter(torch.zeros(config.n_proteins))

        # Protein-specific direct RNA residual and its sample-dependent gate.
        self.cognate_intercept = nn.Parameter(torch.zeros(config.n_proteins))
        self.cognate_slope = nn.Parameter(torch.zeros(config.n_proteins))
        self.gate_bias = nn.Parameter(torch.full((config.n_proteins,), -1.0))
        self.gate_rna_weight = nn.Parameter(torch.zeros(config.n_proteins))
        self.gate_factor_weight = nn.Parameter(torch.zeros(config.n_proteins))

        self.register_buffer("parent_gene_index", parent_index, persistent=True)

    def encode_rna(
        self,
        expression: Tensor,
        rna_valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.rna_encoder(expression, rna_valid_mask)

    def aggregate_sample_state(
        self,
        rna_hidden: Tensor,
        rna_effective_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = rna_hidden.shape[0]
        query = self.sample_queries.unsqueeze(0).expand(batch_size, -1, -1)
        query_state = self.query_norm(
            query
            + self.sample_reader(
                query,
                rna_hidden,
                context_mask=rna_effective_mask,
            )
        )
        pool_weight = torch.softmax(self.query_pool_logits, dim=0)
        sample_state = torch.einsum("q,bqd->bd", pool_weight, query_state)
        return self.sample_norm(sample_state), query_state

    def decode_factors(self, sample_state: Tensor) -> tuple[Tensor, Tensor]:
        factors = self.factor_head(sample_state)
        factor_prediction = factors @ self.protein_loadings.transpose(0, 1)
        factor_prediction = factor_prediction + self.protein_bias
        return factors, factor_prediction

    def _gather_cognate_expression(
        self,
        expression: Tensor,
        rna_effective_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        mapped = self.parent_gene_index >= 0
        safe_index = self.parent_gene_index.clamp_min(0)
        cognate_expression = expression.index_select(1, safe_index)
        sample_valid = rna_effective_mask.index_select(1, safe_index)
        cognate_valid = sample_valid & mapped.unsqueeze(0)
        cognate_expression = cognate_expression * cognate_valid.to(expression.dtype)
        return cognate_expression, cognate_valid

    def apply_cognate_shortcut(
        self,
        expression: Tensor,
        rna_effective_mask: Tensor,
        factor_prediction: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cognate_expression, cognate_valid = self._gather_cognate_expression(
            expression,
            rna_effective_mask,
        )
        direct_residual = (
            self.cognate_intercept.unsqueeze(0)
            + self.cognate_slope.unsqueeze(0) * cognate_expression
        )
        direct_residual = direct_residual * cognate_valid.to(direct_residual.dtype)

        gate_logits = (
            self.gate_bias.unsqueeze(0)
            + self.gate_rna_weight.unsqueeze(0) * cognate_expression
            + self.gate_factor_weight.unsqueeze(0) * factor_prediction
        )
        gate = torch.sigmoid(gate_logits) * cognate_valid.to(gate_logits.dtype)
        prediction = factor_prediction + gate * direct_residual
        return prediction, direct_residual, gate, cognate_valid

    def forward(
        self,
        expression: Tensor,
        rna_valid_mask: Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, Tensor]:
        rna_hidden, effective_mask = self.encode_rna(expression, rna_valid_mask)
        sample_state, query_state = self.aggregate_sample_state(
            rna_hidden,
            effective_mask,
        )
        factors, factor_prediction = self.decode_factors(sample_state)
        protein, direct_residual, gate, cognate_valid = self.apply_cognate_shortcut(
            expression,
            effective_mask,
            factor_prediction,
        )

        output: dict[str, Tensor] = {"protein": protein}
        if return_hidden:
            output.update(
                {
                    "rna_hidden": rna_hidden,
                    "rna_effective_mask": effective_mask,
                    "query_state": query_state,
                    "sample_state": sample_state,
                    "protein_factors": factors,
                    "factor_prediction": factor_prediction,
                    "cognate_direct_residual": direct_residual,
                    "cognate_gate": gate,
                    "cognate_valid": cognate_valid,
                }
            )
        return output

    def fix_projection_matrices_(self) -> "SupervisedFactorProteinTranslator":
        self.rna_encoder.encoder.fix_projection_matrices_()
        self.sample_reader.fix_projection_matrices_()
        return self

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def parameter_breakdown(self) -> dict[str, int]:
        module_groups = {
            "rna_encoder": self.rna_encoder,
            "sample_reader": self.sample_reader,
            "factor_head": self.factor_head,
        }
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in module_groups.items()
        }
        counts.update(
            {
                "sample_queries_and_pool": (
                    self.sample_queries.numel() + self.query_pool_logits.numel()
                ),
                "protein_loadings": self.protein_loadings.numel(),
                "protein_bias": self.protein_bias.numel(),
                "cognate_linear": (
                    self.cognate_intercept.numel() + self.cognate_slope.numel()
                ),
                "cognate_gate": (
                    self.gate_bias.numel()
                    + self.gate_rna_weight.numel()
                    + self.gate_factor_weight.numel()
                ),
            }
        )
        counted = sum(counts.values())
        counts["normalization_remainder"] = self.parameter_count() - counted
        counts["total"] = self.parameter_count()
        return counts

    def complexity(self, batch_size: int = 1) -> dict[str, int | str]:
        return {
            "batch_size": batch_size,
            "rna_tokens": self.config.n_genes,
            "sample_queries": self.config.n_sample_queries,
            "protein_factors": self.config.n_factors,
            "protein_targets": self.config.n_proteins,
            "factor_decode_scaling": "O(B * P * K)",
            "has_gene_by_protein_matrix": 0,
            "largest_gene_protein_attention_matrix": 0,
        }

    @staticmethod
    def equations() -> dict[str, str]:
        return {
            "rna_encoding": "H_i = RNAEncoder(x_i)",
            "query_aggregation": "T_i = Q + LinearAttention(Q, H_i, H_i)",
            "supervised_factors": "f_i = FactorHead(sum_q softmax(a)_q T_iq)",
            "factor_decoder": "p_factor_ij = b_j + <f_i, L_j>",
            "cognate_direct": "d_ij = a_j + beta_j x_i,parent(j)",
            "gate": "g_ij = m_ij sigmoid(eta_j + u_j x_i,parent(j) + v_j p_factor_ij)",
            "prediction": "p_hat_ij = p_factor_ij + g_ij d_ij",
        }


__all__ = ["SupervisedFactorConfig", "SupervisedFactorProteinTranslator"]
