"""Sample-wise phosphosite-centering utilities for SCP682main-1."""

from __future__ import annotations

import numpy as np
import torch


def sample_median_center_array(
    values: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Center each sample using only its observed phosphosite values."""

    matrix = np.asarray(values, dtype=np.float32)
    observed = np.asarray(mask, dtype=bool)
    if matrix.shape != observed.shape:
        raise ValueError("values and mask must have identical shapes")
    masked = np.where(observed, matrix, np.nan)
    offsets = np.nanmedian(masked, axis=1)
    offsets = np.where(np.isfinite(offsets), offsets, 0.0).astype(np.float32)
    centered = np.where(observed, matrix - offsets[:, None], 0.0)
    return centered.astype(np.float32), offsets


def project_zero_sample_median_torch(
    values: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Project values to zero median over each sample's observed sites."""

    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    masked = torch.where(mask, values, torch.full_like(values, float("nan")))
    offsets = torch.nanquantile(masked, 0.5, dim=1)
    offsets = torch.nan_to_num(offsets, nan=0.0)
    centered = values - offsets.unsqueeze(1)
    return torch.where(mask, centered, values)
