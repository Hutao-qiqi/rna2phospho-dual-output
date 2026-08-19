"""Canonical cancer grouping and deterministic two-stage training schedules."""

from __future__ import annotations

import math

import numpy as np


_ALIASES = {
    "BRCA_PROSPECTIVE": "BRCA",
    "BRCA_TCGA": "BRCA",
    "GBM_DISCOVERY": "GBM",
    "GBM_CONFIRMATORY": "GBM",
    "LUAD_CONFIRM": "LUAD",
    "UCEC_CONFIRM": "UCEC",
    "OV_PROSPECTIVE": "OV",
    "OV_TCGA": "OV",
    "COAD_PROSPECTIVE": "COAD",
}


def canonical_cancer_labels(values: np.ndarray | list[str]) -> np.ndarray:
    labels = np.asarray(values, dtype=str)
    output = np.asarray([_ALIASES.get(value.upper(), value.upper()) for value in labels], dtype=str)
    if np.any(np.char.str_len(output) == 0):
        raise ValueError("cancer labels contain empty values")
    return output


def fit_cancer_vocabulary(
    training_labels: np.ndarray, all_labels: np.ndarray
) -> tuple[list[str], np.ndarray]:
    training = canonical_cancer_labels(training_labels)
    complete = canonical_cancer_labels(all_labels)
    vocabulary = sorted(set(training.tolist()))
    unknown = sorted(set(complete.tolist()) - set(vocabulary))
    if unknown:
        raise ValueError(f"validation cancers are absent from training: {unknown}")
    lookup = {name: index for index, name in enumerate(vocabulary)}
    return vocabulary, np.asarray([lookup[value] for value in complete], dtype=np.int64)


def pan_cancer_windows(
    index: np.ndarray,
    cancer_index: np.ndarray,
    *,
    window_size: int = 192,
    seed: int,
) -> list[np.ndarray]:
    """Interleave cancers, visit every row, and rotate a small padded tail."""
    rows = np.asarray(index, dtype=np.int64)
    cancer = np.asarray(cancer_index, dtype=np.int64)
    if rows.ndim != 1 or cancer.ndim != 1 or cancer.shape[0] <= rows.max(initial=-1):
        raise ValueError("cancer schedule inputs are inconsistent")
    if window_size < 2:
        raise ValueError("correlation window must contain at least two samples")
    rng = np.random.default_rng(seed)
    queues: dict[int, list[int]] = {}
    for value in sorted(set(cancer[rows].tolist())):
        selected = rows[cancer[rows] == value].copy()
        rng.shuffle(selected)
        queues[value] = selected.tolist()
    ordered: list[int] = []
    active = list(queues)
    while active:
        next_active = []
        for value in active:
            if queues[value]:
                ordered.append(queues[value].pop())
            if queues[value]:
                next_active.append(value)
        active = next_active
    target = int(math.ceil(len(ordered) / window_size) * window_size)
    if target > len(ordered):
        # At most window_size-1 rows repeat.  The offset changes every epoch.
        supplement = rows.copy()
        rng.shuffle(supplement)
        ordered.extend(supplement[: target - len(ordered)].tolist())
    return [
        np.asarray(ordered[start:start + window_size], dtype=np.int64)
        for start in range(0, len(ordered), window_size)
    ]


def cancer_specific_windows(index: np.ndarray, cancer_index: np.ndarray) -> list[np.ndarray]:
    rows = np.asarray(index, dtype=np.int64)
    cancer = np.asarray(cancer_index, dtype=np.int64)
    output = []
    for value in sorted(set(cancer[rows].tolist())):
        selected = rows[cancer[rows] == value]
        if selected.size:
            output.append(selected.copy())
    return output


def physical_batches(window: np.ndarray, batch_size: int) -> list[np.ndarray]:
    rows = np.asarray(window, dtype=np.int64)
    if batch_size < 1:
        raise ValueError("physical batch size must be positive")
    return [rows[start:start + batch_size] for start in range(0, rows.size, batch_size)]
