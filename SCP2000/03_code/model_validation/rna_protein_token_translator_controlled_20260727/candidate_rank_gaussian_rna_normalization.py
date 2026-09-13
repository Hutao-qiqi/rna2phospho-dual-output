"""Selection-train-only empirical rank-Gaussian RNA normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np

try:
    from .candidate_groupwise_rna_normalization import (
        StrictInnerPartitions,
        build_strict_inner_partitions,
        seal_outer_protein_labels,
        validate_case_exclusivity,
    )
except ImportError:
    from candidate_groupwise_rna_normalization import (
        StrictInnerPartitions,
        build_strict_inner_partitions,
        seal_outer_protein_labels,
        validate_case_exclusivity,
    )


N_QUANTILE_NODES = 257


def _standard_normal_scores(probabilities: np.ndarray, clip_probability: float) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        float(clip_probability),
        1.0 - float(clip_probability),
    )
    normal = NormalDist()
    return np.asarray([normal.inv_cdf(float(value)) for value in clipped], dtype=np.float32)


@dataclass(frozen=True)
class TrainQuantileGaussianizer:
    """Per-gene empirical quantile map fitted on one declared training set."""

    feature_names: tuple[str, ...]
    probabilities: np.ndarray
    quantile_values: np.ndarray
    normal_scores: np.ndarray
    n_fit_samples: int
    clip_probability: float

    def __post_init__(self) -> None:
        n_features = len(self.feature_names)
        if n_features == 0 or len(set(self.feature_names)) != n_features:
            raise ValueError("feature_names must be non-empty and unique")
        if int(self.n_fit_samples) < 2:
            raise ValueError("n_fit_samples must be at least two")
        if not 0.0 < float(self.clip_probability) < 0.5:
            raise ValueError("clip_probability must be in (0, 0.5)")
        probabilities = np.asarray(self.probabilities)
        quantiles = np.asarray(self.quantile_values)
        scores = np.asarray(self.normal_scores)
        if probabilities.shape != (N_QUANTILE_NODES,):
            raise ValueError("probabilities must contain exactly 257 nodes")
        if quantiles.shape != (n_features, N_QUANTILE_NODES):
            raise ValueError("quantile_values have an incompatible shape")
        if scores.shape != (N_QUANTILE_NODES,):
            raise ValueError("normal_scores must contain exactly 257 nodes")
        if not np.isfinite(quantiles).all() or not np.isfinite(scores).all():
            raise ValueError("rank-Gaussian mapping contains non-finite values")
        if np.any(np.diff(probabilities) <= 0):
            raise ValueError("probabilities must increase strictly")
        if np.any(np.diff(quantiles, axis=1) < 0):
            raise ValueError("per-gene quantile nodes must be non-decreasing")
        if np.any(np.diff(scores) < 0):
            raise ValueError("normal_scores must be non-decreasing")

    @property
    def mean(self) -> np.ndarray:
        """Compatibility vector for the shared checkpoint writer."""

        return np.zeros(len(self.feature_names), dtype=np.float32)

    @property
    def scale(self) -> np.ndarray:
        """Compatibility vector for the shared checkpoint writer."""

        return np.ones(len(self.feature_names), dtype=np.float32)

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        fit_indices: Sequence[int],
        *,
        feature_names: Sequence[str],
        n_nodes: int = N_QUANTILE_NODES,
    ) -> "TrainQuantileGaussianizer":
        matrix = np.asarray(values, dtype=np.float32)
        indices = np.asarray(fit_indices, dtype=np.int64)
        names = tuple(str(value) for value in feature_names)
        if n_nodes != N_QUANTILE_NODES:
            raise ValueError("the candidate contract requires exactly 257 nodes")
        if matrix.ndim != 2 or matrix.shape[1] != len(names):
            raise ValueError("values and feature_names have incompatible shapes")
        if indices.ndim != 1 or indices.size < 2:
            raise ValueError("fit_indices must contain at least two rows")
        if np.unique(indices).size != indices.size:
            raise ValueError("fit_indices contain duplicates")
        if int(indices.min()) < 0 or int(indices.max()) >= matrix.shape[0]:
            raise IndexError("fit_indices are outside the RNA matrix")
        training = matrix[indices]
        if not np.isfinite(training).all():
            raise ValueError("selection-train RNA contains non-finite values")
        probabilities = np.linspace(0.0, 1.0, n_nodes, dtype=np.float64)
        try:
            quantile_values = np.quantile(
                training, probabilities, axis=0, method="linear"
            )
        except TypeError:  # NumPy before 1.22
            quantile_values = np.quantile(
                training, probabilities, axis=0, interpolation="linear"
            )
        clip_probability = 0.5 / float(indices.size)
        normal_scores = _standard_normal_scores(probabilities, clip_probability)
        return cls(
            feature_names=names,
            probabilities=probabilities.astype(np.float32),
            quantile_values=np.asarray(quantile_values.T, dtype=np.float32),
            normal_scores=normal_scores,
            n_fit_samples=int(indices.size),
            clip_probability=float(clip_probability),
        )

    @staticmethod
    def _transform_gene(
        values: np.ndarray,
        quantile_nodes: np.ndarray,
        normal_scores: np.ndarray,
    ) -> np.ndarray:
        unique_nodes, inverse = np.unique(quantile_nodes, return_inverse=True)
        score_sums = np.bincount(
            inverse, weights=normal_scores.astype(np.float64), minlength=unique_nodes.size
        )
        score_counts = np.bincount(inverse, minlength=unique_nodes.size)
        unique_scores = score_sums / score_counts
        return np.interp(
            values,
            unique_nodes,
            unique_scores,
            left=float(unique_scores[0]),
            right=float(unique_scores[-1]),
        ).astype(np.float32)

    def transform(
        self,
        values: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("values have an incompatible feature dimension")
        if feature_names is not None:
            names = tuple(str(value) for value in feature_names)
            if names != self.feature_names:
                raise ValueError("feature order differs from the fitted mapping")
        if not np.isfinite(matrix).all():
            raise ValueError("RNA transform input contains non-finite values")
        output = np.empty_like(matrix, dtype=np.float32)
        for gene_index in range(matrix.shape[1]):
            output[:, gene_index] = self._transform_gene(
                matrix[:, gene_index],
                self.quantile_values[gene_index],
                self.normal_scores,
            )
        return output

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": "selection_train_gene_rank_gaussian_257",
            "feature_names": np.asarray(self.feature_names, dtype=np.str_),
            "probabilities": self.probabilities.copy(),
            "quantile_values": self.quantile_values.copy(),
            "normal_scores": self.normal_scores.copy(),
            "n_fit_samples": int(self.n_fit_samples),
            "n_quantile_nodes": N_QUANTILE_NODES,
            "clip_probability": float(self.clip_probability),
            "validation_uses_training_nodes_only": True,
            "outer_uses_training_nodes_only": True,
        }

    def save(self, path: str | Path, *, fit_indices: Sequence[int]) -> Path:
        destination = Path(path)
        if destination.suffix.lower() != ".npz":
            raise ValueError("mapping state path must end in .npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        state["fit_indices"] = np.asarray(fit_indices, dtype=np.int64)
        np.savez_compressed(destination, **state)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "TrainQuantileGaussianizer":
        with np.load(Path(path), allow_pickle=False) as state:
            if int(state["n_quantile_nodes"]) != N_QUANTILE_NODES:
                raise ValueError("saved mapping does not use 257 nodes")
            return cls(
                feature_names=tuple(state["feature_names"].astype(str).tolist()),
                probabilities=state["probabilities"].astype(np.float32),
                quantile_values=state["quantile_values"].astype(np.float32),
                normal_scores=state["normal_scores"].astype(np.float32),
                n_fit_samples=int(state["n_fit_samples"]),
                clip_probability=float(state["clip_probability"]),
            )


__all__ = [
    "N_QUANTILE_NODES",
    "StrictInnerPartitions",
    "TrainQuantileGaussianizer",
    "build_strict_inner_partitions",
    "seal_outer_protein_labels",
    "validate_case_exclusivity",
]
