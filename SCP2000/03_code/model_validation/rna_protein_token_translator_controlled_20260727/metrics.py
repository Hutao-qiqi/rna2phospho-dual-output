from __future__ import annotations

from typing import Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr


def _validate_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must be two-dimensional")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def per_protein_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protein_names: Sequence[str],
    *,
    mask: np.ndarray | None = None,
    min_samples: int = 10,
) -> pd.DataFrame:
    """Calculate held-out Spearman correlation for each fixed protein position."""

    y_true, y_pred = _validate_pair(y_true, y_pred)
    names = tuple(str(name) for name in protein_names)
    if len(names) != y_true.shape[1]:
        raise ValueError("protein_names length does not match target dimension")
    if len(names) != len(set(names)):
        raise ValueError("protein_names contain duplicates")
    if min_samples < 2:
        raise ValueError("min_samples must be at least two")
    if mask is None:
        observed = np.isfinite(y_true)
    else:
        observed = np.asarray(mask, dtype=bool)
        if observed.shape != y_true.shape:
            raise ValueError("mask shape does not match targets")
        observed = observed & np.isfinite(y_true)
    observed &= np.isfinite(y_pred)

    rows: list[dict[str, object]] = []
    for column, protein in enumerate(names):
        valid = observed[:, column]
        n_used = int(valid.sum())
        rho = np.nan
        p_value = np.nan
        if n_used >= min_samples:
            truth = y_true[valid, column]
            prediction = y_pred[valid, column]
            if np.unique(truth).size > 1 and np.unique(prediction).size > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConstantInputWarning)
                    result = spearmanr(truth, prediction)
                rho = float(result.correlation)
                p_value = float(result.pvalue)
        rows.append(
            {
                "protein": protein,
                "n_samples_used": n_used,
                "spearman": rho,
                "rho_p_value": p_value,
            }
        )
    return pd.DataFrame.from_records(rows)


def masked_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    y_true, y_pred = _validate_pair(y_true, y_pred)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != y_true.shape:
            raise ValueError("mask shape does not match targets")
        valid &= mask
    if not valid.any():
        return float("nan")
    difference = y_pred[valid].astype(np.float64) - y_true[valid].astype(np.float64)
    return float(np.mean(difference * difference))


def batchwise_flattened_cosine(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    batch_size: int = 8,
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    """Evaluate direct predictions by flattened valid-value cosine per batch."""

    y_true, y_pred = _validate_pair(y_true, y_pred)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    observed = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask is not None:
        supplied_mask = np.asarray(mask, dtype=bool)
        if supplied_mask.shape != y_true.shape:
            raise ValueError("mask shape does not match targets")
        observed &= supplied_mask

    rows: list[dict[str, float | int]] = []
    for batch_index, start in enumerate(range(0, y_true.shape[0], batch_size)):
        stop = min(start + batch_size, y_true.shape[0])
        valid = observed[start:stop]
        n_valid = int(valid.sum())
        cosine = float("nan")
        if n_valid:
            truth = y_true[start:stop][valid].astype(np.float64, copy=False)
            prediction = y_pred[start:stop][valid].astype(np.float64, copy=False)
            denominator = max(float(np.linalg.norm(truth)), epsilon) * max(
                float(np.linalg.norm(prediction)), epsilon
            )
            cosine = float(np.dot(truth, prediction) / denominator)
        rows.append(
            {
                "batch_index": batch_index,
                "n_samples": int(stop - start),
                "n_valid_values": n_valid,
                "cosine_similarity": cosine,
            }
        )
    return pd.DataFrame.from_records(rows)


def summarize_spearman(per_protein: pd.DataFrame) -> dict[str, float | int]:
    required = {"spearman", "n_samples_used"}
    if not required.issubset(per_protein.columns):
        raise ValueError(f"per_protein table lacks columns: {sorted(required - set(per_protein.columns))}")
    values = pd.to_numeric(per_protein["spearman"], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float))
    return {
        "n_proteins": int(len(per_protein)),
        "n_evaluable_proteins": int(finite.sum()),
        "median_spearman": float(values[finite].median()) if finite.any() else float("nan"),
        "mean_spearman": float(values[finite].mean()) if finite.any() else float("nan"),
    }
