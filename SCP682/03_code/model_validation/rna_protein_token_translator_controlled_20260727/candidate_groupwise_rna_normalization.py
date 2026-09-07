"""Selection-train-only groupwise RNA normalization for strict protein screens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .splits import make_strict_folds
except ImportError:
    from splits import make_strict_folds


@dataclass(frozen=True)
class StrictInnerPartitions:
    """Nested case-blocked partitions in the original sample coordinate system."""

    selection_train: np.ndarray
    selection_validation: np.ndarray
    outer_test: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            np.asarray(self.selection_train, dtype=np.int64),
            np.asarray(self.selection_validation, dtype=np.int64),
            np.asarray(self.outer_test, dtype=np.int64),
        )
        if any(values.ndim != 1 or values.size == 0 for values in arrays):
            raise ValueError("all strict partitions must be non-empty vectors")
        if any(np.unique(values).size != values.size for values in arrays):
            raise ValueError("strict partitions contain duplicate indices")
        if np.intersect1d(arrays[0], arrays[1]).size:
            raise ValueError("selection train overlaps selection validation")
        if np.intersect1d(arrays[0], arrays[2]).size:
            raise ValueError("selection train overlaps outer test")
        if np.intersect1d(arrays[1], arrays[2]).size:
            raise ValueError("selection validation overlaps outer test")

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "selection_train": np.asarray(self.selection_train, dtype=np.int64),
            "selection_validation": np.asarray(
                self.selection_validation, dtype=np.int64
            ),
            "outer_test": np.asarray(self.outer_test, dtype=np.int64),
        }


def build_strict_inner_partitions(
    sample_ids: Sequence[str],
    strata: Sequence[object],
    case_ids: Sequence[object],
    *,
    seed: int = 20260719,
    n_folds: int = 5,
    inner_folds: int = 5,
    fold_index: int = 0,
) -> StrictInnerPartitions:
    """Reproduce the nested split used by the strict inner protein screen."""

    sample_array = np.asarray(sample_ids).astype(str)
    strata_array = np.asarray(strata)
    case_array = np.asarray(case_ids)
    if sample_array.ndim != 1 or sample_array.size == 0:
        raise ValueError("sample_ids must be a non-empty vector")
    if strata_array.shape != sample_array.shape or case_array.shape != sample_array.shape:
        raise ValueError("strata and case_ids must match sample_ids")
    outer_folds = make_strict_folds(
        sample_array,
        strata_array,
        n_splits=n_folds,
        random_state=seed,
        blocking_groups=case_array,
    )
    if fold_index < 0 or fold_index >= len(outer_folds):
        raise ValueError("fold_index is outside the available folds")
    outer = outer_folds[fold_index]
    inner_folds_created = make_strict_folds(
        sample_array[outer.train_idx],
        strata_array[outer.train_idx],
        n_splits=inner_folds,
        random_state=seed + 1009,
        blocking_groups=case_array[outer.train_idx],
    )
    inner = inner_folds_created[0]
    partitions = StrictInnerPartitions(
        selection_train=outer.train_idx[inner.train_idx],
        selection_validation=outer.train_idx[inner.test_idx],
        outer_test=outer.test_idx,
    )
    validate_case_exclusivity(partitions, case_array)
    return partitions


def validate_case_exclusivity(
    partitions: StrictInnerPartitions,
    case_ids: Sequence[object],
) -> None:
    cases = np.asarray(case_ids).astype(str)
    all_indices = np.concatenate(
        [
            partitions.selection_train,
            partitions.selection_validation,
            partitions.outer_test,
        ]
    )
    if all_indices.min() < 0 or all_indices.max() >= cases.size:
        raise IndexError("strict partition indices exceed the case vector")
    train_cases = set(cases[partitions.selection_train])
    validation_cases = set(cases[partitions.selection_validation])
    outer_cases = set(cases[partitions.outer_test])
    if train_cases & validation_cases:
        raise ValueError("case leakage between selection train and validation")
    if train_cases & outer_cases:
        raise ValueError("case leakage between selection train and outer test")
    if validation_cases & outer_cases:
        raise ValueError("case leakage between selection validation and outer test")


def seal_outer_protein_labels(
    protein: np.ndarray,
    outer_indices: Sequence[int],
) -> np.ndarray:
    """Return a copy with every outer-test protein label made unavailable."""

    matrix = np.asarray(protein)
    indices = np.asarray(outer_indices, dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError("protein must be two-dimensional")
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("outer_indices must be a non-empty vector")
    if np.unique(indices).size != indices.size:
        raise ValueError("outer_indices contain duplicates")
    if indices.min() < 0 or indices.max() >= matrix.shape[0]:
        raise IndexError("outer_indices are outside the protein matrix")
    sealed = matrix.astype(np.float32, copy=True)
    sealed[indices] = np.nan
    return sealed


@dataclass(frozen=True)
class GroupwiseRNAStandardizer:
    """Per-gene group statistics fitted on one declared training partition."""

    feature_names: tuple[str, ...]
    global_mean: np.ndarray
    global_scale: np.ndarray
    global_count: np.ndarray
    known_groups: tuple[str, ...]
    group_mean: np.ndarray
    group_scale: np.ndarray
    group_sample_count: np.ndarray
    minimum_scale: float = 1e-6

    def __post_init__(self) -> None:
        n_features = len(self.feature_names)
        n_groups = len(self.known_groups)
        if n_features == 0 or len(set(self.feature_names)) != n_features:
            raise ValueError("feature_names must be non-empty and unique")
        if len(set(self.known_groups)) != n_groups:
            raise ValueError("known_groups must be unique")
        expected_vector = (n_features,)
        if np.asarray(self.global_mean).shape != expected_vector:
            raise ValueError("global_mean has an incompatible shape")
        if np.asarray(self.global_scale).shape != expected_vector:
            raise ValueError("global_scale has an incompatible shape")
        if np.asarray(self.global_count).shape != expected_vector:
            raise ValueError("global_count has an incompatible shape")
        expected_matrix = (n_groups, n_features)
        if np.asarray(self.group_mean).shape != expected_matrix:
            raise ValueError("group_mean has an incompatible shape")
        if np.asarray(self.group_scale).shape != expected_matrix:
            raise ValueError("group_scale has an incompatible shape")
        if np.asarray(self.group_sample_count).shape != (n_groups,):
            raise ValueError("group_sample_count has an incompatible shape")
        if not np.isfinite(self.global_mean).all():
            raise ValueError("global_mean contains non-finite values")
        if not np.isfinite(self.global_scale).all() or (self.global_scale <= 0).any():
            raise ValueError("global_scale must be finite and positive")
        if not np.isfinite(self.group_mean).all():
            raise ValueError("group_mean contains non-finite values")
        if not np.isfinite(self.group_scale).all() or (self.group_scale <= 0).any():
            raise ValueError("group_scale must be finite and positive")
        if (self.global_count < 1).any() or (self.group_sample_count < 1).any():
            raise ValueError("normalization counts must be positive")

    @property
    def mean(self) -> np.ndarray:
        """Compatibility alias used by the shared checkpoint writer."""

        return self.global_mean

    @property
    def scale(self) -> np.ndarray:
        """Compatibility alias used by the shared checkpoint writer."""

        return self.global_scale

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        groups: Sequence[object],
        fit_indices: Sequence[int],
        *,
        feature_names: Sequence[str],
        minimum_scale: float = 1e-6,
    ) -> "GroupwiseRNAStandardizer":
        matrix = np.asarray(values, dtype=np.float32)
        group_array = np.asarray(groups).astype(str)
        indices = np.asarray(fit_indices, dtype=np.int64)
        names = tuple(str(value) for value in feature_names)
        if matrix.ndim != 2 or matrix.shape[1] != len(names):
            raise ValueError("values and feature_names have incompatible shapes")
        if group_array.shape != (matrix.shape[0],):
            raise ValueError("groups must contain one entry per sample")
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("fit_indices must be a non-empty vector")
        if np.unique(indices).size != indices.size:
            raise ValueError("fit_indices contain duplicates")
        if indices.min() < 0 or indices.max() >= matrix.shape[0]:
            raise IndexError("fit_indices are outside the RNA matrix")
        training = matrix[indices]
        if not np.isfinite(training).all():
            raise ValueError("selection-train RNA contains non-finite values")
        global_mean = training.mean(axis=0, dtype=np.float64).astype(np.float32)
        global_scale = training.std(axis=0, dtype=np.float64).astype(np.float32)
        global_scale = np.where(
            global_scale >= minimum_scale, global_scale, 1.0
        ).astype(np.float32)
        known_groups = tuple(sorted(set(group_array[indices].tolist())))
        group_mean = np.empty((len(known_groups), matrix.shape[1]), dtype=np.float32)
        group_scale = np.empty_like(group_mean)
        group_sample_count = np.empty(len(known_groups), dtype=np.int64)
        for position, group in enumerate(known_groups):
            group_rows = indices[group_array[indices] == group]
            group_values = matrix[group_rows]
            group_sample_count[position] = group_rows.size
            group_mean[position] = group_values.mean(
                axis=0, dtype=np.float64
            ).astype(np.float32)
            scale = group_values.std(axis=0, dtype=np.float64).astype(np.float32)
            group_scale[position] = np.where(
                scale >= minimum_scale, scale, 1.0
            ).astype(np.float32)
        return cls(
            feature_names=names,
            global_mean=global_mean,
            global_scale=global_scale,
            global_count=np.full(matrix.shape[1], indices.size, dtype=np.int64),
            known_groups=known_groups,
            group_mean=group_mean,
            group_scale=group_scale,
            group_sample_count=group_sample_count,
            minimum_scale=float(minimum_scale),
        )

    def statistics_for(self, group: object) -> tuple[np.ndarray, np.ndarray, bool]:
        key = str(group)
        try:
            position = self.known_groups.index(key)
        except ValueError:
            return self.global_mean, self.global_scale, True
        return self.group_mean[position], self.group_scale[position], False

    def transform(
        self,
        values: np.ndarray,
        groups: Sequence[object],
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        group_array = np.asarray(groups).astype(str)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("values have an incompatible feature dimension")
        if group_array.shape != (matrix.shape[0],):
            raise ValueError("groups must contain one entry per sample")
        if feature_names is not None:
            names = tuple(str(value) for value in feature_names)
            if names != self.feature_names:
                raise ValueError("feature order differs from the fitted standardizer")
        if not np.isfinite(matrix).all():
            raise ValueError("RNA transform input contains non-finite values")
        output = np.empty_like(matrix, dtype=np.float32)
        for group in np.unique(group_array):
            rows = group_array == group
            mean, scale, _ = self.statistics_for(group)
            output[rows] = (matrix[rows] - mean) / scale
        return output

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": "selection_train_groupwise_gene_zscore",
            "feature_names": np.asarray(self.feature_names, dtype=np.str_),
            "global_mean": self.global_mean.copy(),
            "global_scale": self.global_scale.copy(),
            "global_count": self.global_count.copy(),
            "known_groups": np.asarray(self.known_groups, dtype=np.str_),
            "group_mean": self.group_mean.copy(),
            "group_scale": self.group_scale.copy(),
            "group_sample_count": self.group_sample_count.copy(),
            "minimum_scale": float(self.minimum_scale),
            "unknown_group_fallback": "selection_train_global_gene_statistics",
        }

    def save(self, path: str | Path, *, fit_indices: Sequence[int]) -> Path:
        destination = Path(path)
        if destination.suffix.lower() != ".npz":
            raise ValueError("normalization state path must end in .npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        state["fit_indices"] = np.asarray(fit_indices, dtype=np.int64)
        np.savez_compressed(destination, **state)
        return destination


__all__ = [
    "GroupwiseRNAStandardizer",
    "StrictInnerPartitions",
    "build_strict_inner_partitions",
    "seal_outer_protein_labels",
    "validate_case_exclusivity",
]
