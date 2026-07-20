"""Smoke tests for the SCP682main-1 phosphosite-centering contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


CODE_DIR = Path(__file__).parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))
from centering import project_zero_sample_median_torch, sample_median_center_array


def test_numpy_centering_ignores_missing_values() -> None:
    values = np.asarray([[3.0, 5.0, np.nan, 7.0], [2.0, np.nan, 6.0, 10.0]], dtype=np.float32)
    mask = np.isfinite(values)
    centered, offsets = sample_median_center_array(values, mask)
    assert np.allclose(offsets, [5.0, 6.0])
    assert np.allclose(centered[0, mask[0]], [-2.0, 0.0, 2.0])
    assert np.allclose(centered[1, mask[1]], [-4.0, 0.0, 4.0])


def test_torch_projection_has_zero_observed_median() -> None:
    values = torch.tensor([[3.0, 5.0, -2.0, 7.0], [2.0, -1.0, 6.0, 10.0]])
    mask = torch.tensor([[True, True, False, True], [True, False, True, True]])
    centered = project_zero_sample_median_torch(values, mask)
    for row, observed in zip(centered, mask):
        assert torch.isclose(torch.median(row[observed]), torch.tensor(0.0))


def test_torch_projection_uses_numpy_compatible_even_median() -> None:
    values = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    mask = torch.ones_like(values, dtype=torch.bool)
    centered = project_zero_sample_median_torch(values, mask)
    assert torch.allclose(centered, torch.tensor([[-3.0, -1.0, 1.0, 3.0]]))
