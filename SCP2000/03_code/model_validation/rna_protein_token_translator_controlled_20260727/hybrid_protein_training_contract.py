"""Leakage-controlled scaling, loss, and development split for the hybrid model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

try:
    from .splits import make_strict_folds
except ImportError:
    from splits import make_strict_folds


@dataclass(frozen=True)
class TrainFittedProteinScale:
    feature_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    safe_range: np.ndarray
    count: np.ndarray
    lower_quantile: float
    upper_quantile: float
    minimum_scale: float = 1e-6

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        fit_indices: Sequence[int],
        *,
        feature_names: Sequence[str],
        mask: np.ndarray | None = None,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        minimum_scale: float = 1e-6,
    ) -> "TrainFittedProteinScale":
        matrix = np.asarray(values, dtype=np.float64)
        indices = np.asarray(fit_indices, dtype=np.int64)
        names = tuple(map(str, feature_names))
        if matrix.ndim != 2 or matrix.shape[1] != len(names):
            raise ValueError("values and feature_names have incompatible shapes")
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("fit_indices must be a non-empty vector")
        if np.unique(indices).size != indices.size:
            raise ValueError("fit_indices contain duplicates")
        if indices.min() < 0 or indices.max() >= matrix.shape[0]:
            raise IndexError("fit_indices are outside values")
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
        observed = np.isfinite(matrix) if mask is None else np.asarray(mask)
        if observed.shape != matrix.shape or observed.dtype != np.bool_:
            raise ValueError("mask must be boolean and match values")
        train = matrix[indices]
        train_mask = observed[indices]
        count = train_mask.sum(axis=0, dtype=np.int64)
        lower = np.zeros(matrix.shape[1], dtype=np.float64)
        upper = np.zeros(matrix.shape[1], dtype=np.float64)
        for column in range(matrix.shape[1]):
            column_values = train[train_mask[:, column], column]
            if column_values.size:
                lower[column] = np.quantile(column_values, lower_quantile)
                upper[column] = np.quantile(column_values, upper_quantile)
        spread = np.maximum(upper - lower, 0.0)
        safe = np.where(spread >= minimum_scale, spread, 1.0)
        return cls(
            feature_names=names,
            lower=lower,
            upper=upper,
            safe_range=safe,
            count=count,
            lower_quantile=float(lower_quantile),
            upper_quantile=float(upper_quantile),
            minimum_scale=float(minimum_scale),
        )

    def transform(
        self,
        values: np.ndarray,
        *,
        mask: np.ndarray,
        feature_names: Sequence[str],
        clip: bool,
    ) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        observed = np.asarray(mask)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("values have an incompatible shape")
        if tuple(map(str, feature_names)) != self.feature_names:
            raise ValueError("protein feature order differs from fitted scale")
        if observed.shape != matrix.shape or observed.dtype != np.bool_:
            raise ValueError("mask must be boolean and match values")
        changed = (matrix - self.lower) / self.safe_range
        if clip:
            changed = np.clip(changed, 0.0, 1.0)
        return np.where(observed, changed, np.nan).astype(np.float32)

    def state_dict(self) -> dict[str, np.ndarray | float | str]:
        return {
            "schema_version": np.asarray([1], dtype=np.int16),
            "feature_names": np.asarray(self.feature_names, dtype=np.str_),
            "lower": self.lower.copy(),
            "upper": self.upper.copy(),
            "safe_range": self.safe_range.copy(),
            "count": self.count.copy(),
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "minimum_scale": self.minimum_scale,
            "validation_clipping": "disabled",
        }


def masked_per_protein_mse(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    minimum_observations: int = 1,
) -> Tensor:
    """Average each protein's observed MSE, then average eligible proteins."""

    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have equal shapes")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean")
    if minimum_observations < 1:
        raise ValueError("minimum_observations must be positive")
    observed = mask.to(prediction.dtype)
    count = observed.sum(dim=0)
    sum_squared = ((prediction - target).square() * observed).sum(dim=0)
    per_protein = sum_squared / count.clamp_min(1.0)
    eligible = count >= minimum_observations
    if not bool(eligible.any()):
        raise ValueError("batch contains no eligible protein")
    return per_protein[eligible].mean()


def masked_per_protein_pearson_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    minimum_observations: int = 8,
    epsilon: float = 1e-8,
) -> Tensor:
    """One minus the mean cross-sample Pearson correlation per protein."""

    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have equal shapes")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean")
    if minimum_observations < 2:
        raise ValueError("minimum_observations must be at least two")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    working_prediction = prediction.float()
    working_target = target.float()
    observed = mask.to(torch.float32)
    count = observed.sum(dim=0)
    prediction_mean = (working_prediction * observed).sum(dim=0) / count.clamp_min(1.0)
    target_mean = (working_target * observed).sum(dim=0) / count.clamp_min(1.0)
    prediction_centered = (working_prediction - prediction_mean) * observed
    target_centered = (working_target - target_mean) * observed
    covariance = (prediction_centered * target_centered).sum(dim=0)
    prediction_sum_squares = prediction_centered.square().sum(dim=0)
    target_sum_squares = target_centered.square().sum(dim=0)
    denominator = torch.sqrt(
        prediction_sum_squares.clamp_min(epsilon)
        * target_sum_squares.clamp_min(epsilon)
    )
    correlation = covariance / denominator
    eligible = (
        (count >= minimum_observations)
        & (target_sum_squares > epsilon)
    )
    if not bool(eligible.any()):
        raise ValueError("batch contains no protein eligible for Pearson loss")
    return 1.0 - correlation[eligible].mean()


def masked_per_protein_variance_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    minimum_observations: int = 8,
    epsilon: float = 1e-8,
) -> Tensor:
    """Match cross-sample standard deviation for each eligible protein."""

    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have equal shapes")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean")
    if minimum_observations < 2:
        raise ValueError("minimum_observations must be at least two")
    working_prediction = prediction.float()
    working_target = target.float()
    observed = mask.to(torch.float32)
    count = observed.sum(dim=0)
    prediction_mean = (working_prediction * observed).sum(dim=0) / count.clamp_min(1.0)
    target_mean = (working_target * observed).sum(dim=0) / count.clamp_min(1.0)
    prediction_ss = (
        (working_prediction - prediction_mean).square() * observed
    ).sum(dim=0)
    target_ss = ((working_target - target_mean).square() * observed).sum(dim=0)
    denominator = (count - 1.0).clamp_min(1.0)
    prediction_sd = torch.sqrt(prediction_ss / denominator + epsilon)
    target_sd = torch.sqrt(target_ss / denominator + epsilon)
    eligible = (count >= minimum_observations) & (target_ss > epsilon)
    if not bool(eligible.any()):
        raise ValueError("batch contains no protein eligible for variance loss")
    return (prediction_sd[eligible] - target_sd[eligible]).abs().mean()


@dataclass(frozen=True)
class DevelopmentSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray


def make_development_split(
    selection_train_indices: Sequence[int],
    sample_ids: Sequence[str],
    strata: Sequence[object],
    blocking_groups: Sequence[object],
    *,
    n_splits: int = 5,
    seed: int = 20263729,
    fold_index: int = 0,
) -> DevelopmentSplit:
    """Split only the locked 916 selection-training rows for model development."""

    selection = np.asarray(selection_train_indices, dtype=np.int64)
    sample_array = np.asarray(sample_ids).astype(str)
    strata_array = np.asarray(strata)
    group_array = np.asarray(blocking_groups)
    if selection.ndim != 1 or selection.size == 0:
        raise ValueError("selection_train_indices must be non-empty")
    subset_folds = make_strict_folds(
        sample_array[selection],
        strata_array[selection],
        n_splits=n_splits,
        random_state=seed,
        blocking_groups=group_array[selection],
    )
    if fold_index < 0 or fold_index >= len(subset_folds):
        raise ValueError("fold_index is outside development folds")
    split = subset_folds[fold_index]
    return DevelopmentSplit(
        train_indices=selection[split.train_idx],
        validation_indices=selection[split.test_idx],
    )


__all__ = [
    "DevelopmentSplit",
    "TrainFittedProteinScale",
    "make_development_split",
    "masked_per_protein_mse",
    "masked_per_protein_pearson_loss",
    "masked_per_protein_variance_loss",
]
