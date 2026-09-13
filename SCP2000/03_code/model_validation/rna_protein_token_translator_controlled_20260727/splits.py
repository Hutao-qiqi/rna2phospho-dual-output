from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


@dataclass(frozen=True)
class FoldIndices:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def val_idx(self) -> np.ndarray:
        return self.test_idx


def _one_dimensional(values: Sequence[object], name: str, n: int | None = None) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if n is not None and result.size != n:
        raise ValueError(f"{name} length {result.size} does not match {n}")
    return result


def validate_folds(
    folds: Sequence[FoldIndices],
    n_samples: int,
    blocking_groups: Sequence[object] | None = None,
) -> None:
    if len(folds) < 2:
        raise ValueError("at least two folds are required")
    test_counts = np.zeros(n_samples, dtype=np.int64)
    groups = None if blocking_groups is None else _one_dimensional(
        blocking_groups, "blocking_groups", n_samples
    )

    for expected_fold, split in enumerate(folds):
        if split.fold != expected_fold:
            raise ValueError("fold identifiers must be consecutive and zero-based")
        train_idx = np.asarray(split.train_idx, dtype=np.int64)
        test_idx = np.asarray(split.test_idx, dtype=np.int64)
        if train_idx.ndim != 1 or test_idx.ndim != 1:
            raise ValueError("fold indices must be one-dimensional")
        if np.unique(train_idx).size != train_idx.size or np.unique(test_idx).size != test_idx.size:
            raise ValueError(f"fold {split.fold} contains duplicate indices")
        if train_idx.size == 0 or test_idx.size == 0:
            raise ValueError(f"fold {split.fold} has an empty partition")
        if train_idx.min() < 0 or test_idx.min() < 0:
            raise IndexError("fold indices cannot be negative")
        if train_idx.max() >= n_samples or test_idx.max() >= n_samples:
            raise IndexError("fold indices exceed the sample count")
        if np.intersect1d(train_idx, test_idx).size:
            raise ValueError(f"fold {split.fold} has train/test overlap")
        if train_idx.size + test_idx.size != n_samples:
            raise ValueError(f"fold {split.fold} does not partition all samples")
        test_counts[test_idx] += 1
        if groups is not None:
            train_groups = set(groups[train_idx].tolist())
            test_groups = set(groups[test_idx].tolist())
            if train_groups & test_groups:
                raise ValueError(f"fold {split.fold} leaks blocking groups")

    if not np.all(test_counts == 1):
        raise ValueError("each sample must occur in exactly one held-out fold")


def make_strict_folds(
    sample_ids: Sequence[str],
    strata: Sequence[object],
    *,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
    blocking_groups: Sequence[object] | None = None,
) -> list[FoldIndices]:
    """Create deterministic OOF folds, optionally keeping studies disjoint."""

    sample_ids_array = _one_dimensional(sample_ids, "sample_ids")
    if np.unique(sample_ids_array.astype(str)).size != sample_ids_array.size:
        raise ValueError("sample_ids must be unique")
    strata_array = _one_dimensional(strata, "strata", sample_ids_array.size)
    if pd_is_missing(strata_array).any():
        raise ValueError("strata contain missing values")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")

    x = np.zeros((sample_ids_array.size, 1), dtype=np.uint8)
    if blocking_groups is None:
        _, counts = np.unique(strata_array, return_counts=True)
        if counts.min() < n_splits:
            raise ValueError("each stratum must contain at least n_splits samples")
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
        iterator = splitter.split(x, strata_array)
        groups_array = None
    else:
        groups_array = _one_dimensional(
            blocking_groups, "blocking_groups", sample_ids_array.size
        )
        if pd_is_missing(groups_array).any():
            raise ValueError("blocking_groups contain missing values")
        if np.unique(groups_array).size < n_splits:
            raise ValueError("blocking_groups contain fewer groups than n_splits")
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
        iterator = splitter.split(x, strata_array, groups_array)

    folds = [
        FoldIndices(
            fold=fold,
            train_idx=np.asarray(train_idx, dtype=np.int64),
            test_idx=np.asarray(test_idx, dtype=np.int64),
        )
        for fold, (train_idx, test_idx) in enumerate(iterator)
    ]
    validate_folds(folds, sample_ids_array.size, groups_array)
    return folds


def pd_is_missing(values: np.ndarray) -> np.ndarray:
    return np.asarray(pd.isna(values), dtype=bool)


def fold_assignment(folds: Sequence[FoldIndices], n_samples: int) -> np.ndarray:
    validate_folds(folds, n_samples)
    assignment = np.full(n_samples, -1, dtype=np.int16)
    for split in folds:
        assignment[split.test_idx] = split.fold
    return assignment


def save_fold_indices(
    path: str | Path,
    folds: Sequence[FoldIndices],
    sample_ids: Sequence[str],
    strata: Sequence[object],
    blocking_groups: Sequence[object] | None = None,
) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("fold index path must end in .npz")
    sample_ids_array = _one_dimensional(sample_ids, "sample_ids").astype(str)
    strata_array = _one_dimensional(strata, "strata", sample_ids_array.size).astype(str)
    groups_array = None
    if blocking_groups is not None:
        groups_array = _one_dimensional(
            blocking_groups, "blocking_groups", sample_ids_array.size
        ).astype(str)
    validate_folds(folds, sample_ids_array.size, groups_array)

    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray([1], dtype=np.int16),
        "sample_ids": sample_ids_array,
        "strata": strata_array,
        "n_folds": np.asarray([len(folds)], dtype=np.int16),
        "fold_assignment": fold_assignment(folds, sample_ids_array.size),
    }
    if groups_array is not None:
        payload["blocking_groups"] = groups_array
    for split in folds:
        payload[f"train_idx_{split.fold}"] = split.train_idx.astype(np.int64)
        payload[f"test_idx_{split.fold}"] = split.test_idx.astype(np.int64)

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary_path = Path(handle.name)
        np.savez_compressed(handle, **payload)
    temporary_path.replace(path)
    return path


def load_fold_indices(
    path: str | Path,
    *,
    expected_sample_ids: Sequence[str] | None = None,
    expected_strata: Sequence[object] | None = None,
) -> tuple[list[FoldIndices], np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(Path(path), allow_pickle=False) as archive:
        schema_version = int(archive["schema_version"][0])
        if schema_version != 1:
            raise ValueError(f"unsupported fold schema version: {schema_version}")
        sample_ids = archive["sample_ids"].astype(str)
        strata = archive["strata"].astype(str)
        groups = archive["blocking_groups"].astype(str) if "blocking_groups" in archive else None
        n_folds = int(archive["n_folds"][0])
        folds = [
            FoldIndices(
                fold=fold,
                train_idx=archive[f"train_idx_{fold}"].astype(np.int64),
                test_idx=archive[f"test_idx_{fold}"].astype(np.int64),
            )
            for fold in range(n_folds)
        ]

    if expected_sample_ids is not None:
        expected = _one_dimensional(expected_sample_ids, "expected_sample_ids").astype(str)
        if not np.array_equal(sample_ids, expected):
            raise ValueError("saved fold sample order differs from expected_sample_ids")
    if expected_strata is not None:
        expected = _one_dimensional(expected_strata, "expected_strata", sample_ids.size).astype(str)
        if not np.array_equal(strata, expected):
            raise ValueError("saved fold strata differ from expected_strata")
    validate_folds(folds, sample_ids.size, groups)
    return folds, sample_ids, strata, groups
