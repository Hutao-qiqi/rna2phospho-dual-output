"""Fixed full-panel centering and training-only residual scaling contracts."""

from __future__ import annotations

import numpy as np


def fit_training_residual_scale(
    residual_target: np.ndarray,
    observed: np.ndarray,
    training_index: np.ndarray,
    *,
    minimum_observations: int = 8,
) -> np.ndarray:
    """Fit a robust per-site scale from training samples only."""
    residual = np.asarray(residual_target, dtype=np.float32)
    mask = np.asarray(observed, dtype=bool)
    train = np.asarray(training_index, dtype=np.int64)
    if residual.shape != mask.shape:
        raise ValueError("residual target and observation mask must align")
    output = np.ones(residual.shape[1], dtype=np.float32)
    for site in range(residual.shape[1]):
        values = residual[train, site][mask[train, site]]
        if values.size < minimum_observations:
            continue
        lower, upper = np.quantile(values, [0.25, 0.75])
        scale = float((upper - lower) / 1.349)
        if not np.isfinite(scale) or scale < 1.0e-4:
            scale = float(np.std(values))
        if np.isfinite(scale) and scale >= 1.0e-4:
            output[site] = scale
    return output


def standardize_residual_target(
    residual_target: np.ndarray,
    observed: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(residual_target, dtype=np.float32)
    mask = np.asarray(observed, dtype=bool)
    scale = np.asarray(scale, dtype=np.float32)
    if scale.shape != (residual.shape[1],):
        raise ValueError("residual scale must cover every site")
    return np.where(mask, residual / scale[None, :], 0.0).astype(np.float32)


def assemble_full_site_matrix(
    chunk_values: list[np.ndarray],
    chunk_site_indices: list[np.ndarray],
    n_sites: int,
) -> np.ndarray:
    """Assemble an exact site partition without applying chunk-wise centering."""
    if len(chunk_values) != len(chunk_site_indices) or not chunk_values:
        raise ValueError("chunk values and indices must be non-empty and aligned")
    n_samples = int(chunk_values[0].shape[0])
    output = np.empty((n_samples, n_sites), dtype=np.float32)
    seen = np.zeros(n_sites, dtype=bool)
    for values, indices in zip(chunk_values, chunk_site_indices):
        values = np.asarray(values, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int64)
        if values.shape != (n_samples, len(indices)):
            raise ValueError("a chunk matrix differs from its site index")
        if indices.size and seen[indices].any():
            raise ValueError("site chunks overlap")
        output[:, indices] = values
        seen[indices] = True
    if not seen.all():
        raise ValueError("site chunks do not cover the complete vocabulary")
    return output


def full_panel_center_prediction(
    residual_prediction: np.ndarray,
    centering_site_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a complete prediction once, after all target chunks are assembled."""
    prediction = np.asarray(residual_prediction, dtype=np.float32)
    panel = np.asarray(centering_site_index, dtype=np.int64)
    if prediction.ndim != 2 or panel.ndim != 1 or panel.size < 1:
        raise ValueError("prediction must be a matrix and panel must be non-empty")
    offset = np.median(prediction[:, panel], axis=1).astype(np.float32)
    return (prediction - offset[:, None]).astype(np.float32), offset
