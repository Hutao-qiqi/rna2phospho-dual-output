"""Low-rank RNA-to-protein residual model with a split-local linear anchor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

try:
    from .candidate_lowrank_cognate import LowRankCognateProteinTranslator
    from .protein_model import ProteinModelConfig
except ImportError:
    from candidate_lowrank_cognate import LowRankCognateProteinTranslator
    from protein_model import ProteinModelConfig


@dataclass(frozen=True)
class CognateLinearAnchor:
    """Frozen per-protein coefficients fitted on one explicit training split."""

    intercept: np.ndarray
    slope: np.ndarray
    n_observed: np.ndarray

    def __post_init__(self) -> None:
        arrays = (self.intercept, self.slope, self.n_observed)
        if any(np.asarray(value).ndim != 1 for value in arrays):
            raise ValueError("anchor arrays must be one-dimensional")
        if len({np.asarray(value).size for value in arrays}) != 1:
            raise ValueError("anchor arrays must have equal lengths")
        if not np.isfinite(self.intercept).all() or not np.isfinite(self.slope).all():
            raise ValueError("anchor coefficients must be finite")
        if (np.asarray(self.n_observed) < 0).any():
            raise ValueError("n_observed must be non-negative")


def fit_cognate_linear_anchor(
    expression: np.ndarray,
    protein: np.ndarray,
    protein_mask: np.ndarray,
    parent_gene_index: np.ndarray | Sequence[int],
    train_indices: np.ndarray | Sequence[int],
    *,
    minimum_variance: float = 1e-8,
    chunk_size: int = 1024,
) -> CognateLinearAnchor:
    """Fit ``a_j + b_j x_parent`` using only ``train_indices``.

    The input matrices may contain all samples. Rows outside ``train_indices``
    are never read from the protein matrix. Proteins without a cognate RNA
    mapping receive a zero anchor. The calculation is chunked over proteins and
    does not allocate a protein-by-gene coefficient matrix.
    """

    expression = np.asarray(expression, dtype=np.float64)
    protein = np.asarray(protein, dtype=np.float64)
    protein_mask = np.asarray(protein_mask, dtype=bool)
    parent_index = np.asarray(parent_gene_index, dtype=np.int64)
    fit_rows = np.asarray(train_indices, dtype=np.int64)

    if expression.ndim != 2 or protein.ndim != 2:
        raise ValueError("expression and protein must be two-dimensional")
    if protein_mask.shape != protein.shape:
        raise ValueError("protein_mask must match protein")
    if expression.shape[0] != protein.shape[0]:
        raise ValueError("expression and protein must contain the same samples")
    if parent_index.shape != (protein.shape[1],):
        raise ValueError("parent_gene_index must contain one entry per protein")
    if (parent_index < -1).any() or (parent_index >= expression.shape[1]).any():
        raise ValueError("parent_gene_index contains an invalid RNA index")
    if fit_rows.ndim != 1 or fit_rows.size == 0:
        raise ValueError("train_indices must be a non-empty one-dimensional array")
    if np.unique(fit_rows).size != fit_rows.size:
        raise ValueError("train_indices contains duplicate rows")
    if int(fit_rows.min()) < 0 or int(fit_rows.max()) >= expression.shape[0]:
        raise IndexError("train_indices is outside the sample range")
    if minimum_variance <= 0 or chunk_size < 1:
        raise ValueError("minimum_variance and chunk_size must be positive")
    if not np.isfinite(expression[fit_rows]).all():
        raise ValueError("training expression contains non-finite values")

    n_proteins = protein.shape[1]
    intercept = np.zeros(n_proteins, dtype=np.float32)
    slope = np.zeros(n_proteins, dtype=np.float32)
    n_observed = np.zeros(n_proteins, dtype=np.int64)
    mapped = np.flatnonzero(parent_index >= 0)

    for start in range(0, mapped.size, chunk_size):
        columns = mapped[start : start + chunk_size]
        x = expression[np.ix_(fit_rows, parent_index[columns])]
        y = protein[np.ix_(fit_rows, columns)]
        observed = protein_mask[np.ix_(fit_rows, columns)] & np.isfinite(y)
        count = observed.sum(axis=0, dtype=np.int64)
        n_observed[columns] = count

        safe_count = np.maximum(count, 1)
        x_sum = np.where(observed, x, 0.0).sum(axis=0)
        y_sum = np.where(observed, y, 0.0).sum(axis=0)
        x_mean = x_sum / safe_count
        y_mean = y_sum / safe_count
        centered_x = x - x_mean
        centered_y = y - y_mean
        x_ss = np.where(observed, centered_x * centered_x, 0.0).sum(axis=0)
        xy = np.where(observed, centered_x * centered_y, 0.0).sum(axis=0)

        fitted_slope = np.divide(
            xy,
            x_ss,
            out=np.zeros_like(xy),
            where=(count >= 2) & (x_ss >= minimum_variance),
        )
        fitted_intercept = np.where(count > 0, y_mean - fitted_slope * x_mean, 0.0)
        intercept[columns] = fitted_intercept.astype(np.float32)
        slope[columns] = fitted_slope.astype(np.float32)

    return CognateLinearAnchor(intercept, slope, n_observed)


class ResidualizedLowRankProteinTranslator(nn.Module):
    """Predict a frozen cognate-linear anchor plus a neural residual."""

    def __init__(
        self,
        config: ProteinModelConfig,
        parent_gene_index: Tensor | Sequence[int],
        anchor_intercept: Tensor | Sequence[float],
        anchor_slope: Tensor | Sequence[float],
        *,
        rank: int = 512,
    ) -> None:
        super().__init__()
        parent_index = torch.as_tensor(parent_gene_index, dtype=torch.long)
        intercept = torch.as_tensor(anchor_intercept, dtype=torch.float32)
        slope = torch.as_tensor(anchor_slope, dtype=torch.float32)
        expected = (config.n_proteins,)
        if parent_index.shape != expected:
            raise ValueError("parent_gene_index must contain one entry per protein")
        if intercept.shape != expected or slope.shape != expected:
            raise ValueError("anchor coefficients must contain one entry per protein")
        if not torch.isfinite(intercept).all() or not torch.isfinite(slope).all():
            raise ValueError("anchor coefficients must be finite")
        if (parent_index < -1).any() or (parent_index >= config.n_genes).any():
            raise ValueError("parent_gene_index contains an invalid RNA index")

        self.config = config
        self.residual_model = LowRankCognateProteinTranslator(
            config=config,
            parent_gene_index=parent_index,
            rank=rank,
        )
        self.register_buffer("anchor_intercept", intercept, persistent=True)
        self.register_buffer("anchor_slope", slope, persistent=True)

    @property
    def parent_gene_index(self) -> Tensor:
        return self.residual_model.parent_gene_index

    @property
    def rna_encoder(self) -> nn.Module:
        return self.residual_model.rna_encoder

    @property
    def global_translator(self) -> nn.Module:
        return self.residual_model.global_translator

    def linear_anchor(self, expression: Tensor) -> Tensor:
        if expression.ndim != 2 or expression.shape[1] != self.config.n_genes:
            raise ValueError(
                f"expression must have shape (batch, {self.config.n_genes})"
            )
        safe_index = self.parent_gene_index.clamp_min(0)
        cognate_expression = expression.index_select(1, safe_index)
        return self.anchor_intercept + self.anchor_slope * cognate_expression

    def forward(
        self,
        expression: Tensor,
        rna_valid_mask: Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, Tensor]:
        residual_output = self.residual_model(
            expression,
            rna_valid_mask=rna_valid_mask,
            return_hidden=return_hidden,
        )
        anchor = self.linear_anchor(expression)
        residual = residual_output["protein"]
        output = dict(residual_output)
        output["protein"] = anchor + residual
        if return_hidden:
            output["linear_anchor"] = anchor
            output["protein_residual"] = residual
        return output

    def fix_projection_matrices_(self) -> "ResidualizedLowRankProteinTranslator":
        self.residual_model.fix_projection_matrices_()
        return self

    def parameter_count(self, trainable_only: bool = False) -> int:
        return self.residual_model.parameter_count(trainable_only=trainable_only)

    def parameter_count_by_component(self) -> dict[str, int]:
        return self.residual_model.parameter_count_by_component()


__all__ = [
    "CognateLinearAnchor",
    "ResidualizedLowRankProteinTranslator",
    "fit_cognate_linear_anchor",
]
