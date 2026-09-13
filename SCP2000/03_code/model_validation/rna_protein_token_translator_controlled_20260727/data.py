from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


PathLike = str | Path


def _as_unique_strings(values: Sequence[Any], name: str) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if len(normalized) != len(set(normalized)):
        duplicates = pd.Index(normalized)[pd.Index(normalized).duplicated()].unique().tolist()
        raise ValueError(f"{name} contains duplicate identifiers: {duplicates[:5]}")
    return normalized


def _read_matrix(path: PathLike) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(path, sep="\t", index_col=0)
    elif suffix == ".csv":
        frame = pd.read_csv(path, index_col=0)
    else:
        raise ValueError(f"unsupported matrix format: {path}")

    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    if not frame.index.is_unique:
        raise ValueError(f"sample identifiers are duplicated in {path}")
    if not frame.columns.is_unique:
        raise ValueError(f"feature identifiers are duplicated in {path}")
    return frame


def _ordered_columns(
    frame: pd.DataFrame,
    requested_order: Sequence[str] | None,
    name: str,
) -> tuple[str, ...]:
    if requested_order is None:
        return _as_unique_strings(frame.columns, name)

    order = _as_unique_strings(requested_order, name)
    missing = [feature for feature in order if feature not in frame.columns]
    if missing:
        raise ValueError(f"{name} contains identifiers absent from the matrix: {missing[:5]}")
    return order


@dataclass(frozen=True)
class PairedRNAProteinData:
    """Aligned RNA and protein matrices with immutable token ordering.

    Protein missing values remain at their original coordinates in ``protein``.
    ``protein_mask`` records observed coordinates and ``filled_protein`` provides
    a training-ready matrix without changing protein positions.
    """

    sample_ids: tuple[str, ...]
    gene_names: tuple[str, ...]
    protein_names: tuple[str, ...]
    rna: np.ndarray
    protein: np.ndarray
    protein_mask: np.ndarray
    manifest: pd.DataFrame

    def __post_init__(self) -> None:
        n_samples = len(self.sample_ids)
        expected_rna = (n_samples, len(self.gene_names))
        expected_protein = (n_samples, len(self.protein_names))
        if self.rna.shape != expected_rna:
            raise ValueError(f"RNA shape {self.rna.shape} does not match {expected_rna}")
        if self.protein.shape != expected_protein:
            raise ValueError(f"protein shape {self.protein.shape} does not match {expected_protein}")
        if self.protein_mask.shape != expected_protein:
            raise ValueError("protein mask shape does not match protein matrix")
        if self.protein_mask.dtype != np.bool_:
            raise TypeError("protein_mask must have boolean dtype")
        if tuple(self.manifest.index.map(str)) != self.sample_ids:
            raise ValueError("manifest order does not match sample_ids")
        if not np.isfinite(self.rna).all():
            raise ValueError("RNA matrix contains non-finite values")
        if np.isinf(self.protein).any():
            raise ValueError("protein matrix contains infinite values")
        if not np.array_equal(self.protein_mask, np.isfinite(self.protein)):
            raise ValueError("protein_mask does not match finite protein entries")

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    @property
    def n_proteins(self) -> int:
        return len(self.protein_names)

    @property
    def gene_ids(self) -> np.ndarray:
        return np.arange(self.n_genes, dtype=np.int64)

    @property
    def protein_ids(self) -> np.ndarray:
        return np.arange(self.n_proteins, dtype=np.int64)

    def filled_protein(self, fill_value: float = 0.0) -> np.ndarray:
        return np.where(self.protein_mask, self.protein, fill_value).astype(np.float32, copy=False)

    def context(self, column: str) -> np.ndarray:
        if column not in self.manifest.columns:
            raise KeyError(f"manifest column not found: {column}")
        return self.manifest[column].astype(str).to_numpy(copy=True)


class RNAProteinDataset:
    """Framework-neutral indexed view used by training data loaders."""

    def __init__(
        self,
        data: PairedRNAProteinData,
        indices: Sequence[int] | np.ndarray | None = None,
        protein_fill_value: float = 0.0,
    ) -> None:
        self.data = data
        self.indices = (
            np.arange(data.n_samples, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if self.indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if self.indices.size and (
            int(self.indices.min()) < 0 or int(self.indices.max()) >= data.n_samples
        ):
            raise IndexError("dataset indices are out of range")
        self.protein_fill_value = float(protein_fill_value)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = int(self.indices[item])
        target = np.where(
            self.data.protein_mask[row],
            self.data.protein[row],
            self.protein_fill_value,
        ).astype(np.float32, copy=False)
        return {
            "sample_index": row,
            "sample_id": self.data.sample_ids[row],
            "rna": self.data.rna[row],
            "gene_ids": self.data.gene_ids,
            "protein": target,
            "protein_mask": self.data.protein_mask[row],
            "protein_ids": self.data.protein_ids,
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for item in range(len(self)):
            yield self[item]


@dataclass(frozen=True)
class FeatureStandardizer:
    """Training-fold RNA standardizer with fixed feature ordering."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        feature_names: Sequence[str],
        indices: Sequence[int] | np.ndarray | None = None,
        minimum_scale: float = 1e-6,
    ) -> "FeatureStandardizer":
        values = np.asarray(values, dtype=np.float32)
        names = _as_unique_strings(feature_names, "feature_names")
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError("values and feature_names have incompatible shapes")
        fit_values = values if indices is None else values[np.asarray(indices, dtype=np.int64)]
        if fit_values.shape[0] == 0:
            raise ValueError("cannot fit a standardizer on zero samples")
        if not np.isfinite(fit_values).all():
            raise ValueError("standardizer input contains non-finite values")
        mean = fit_values.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = fit_values.std(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.where(scale >= minimum_scale, scale, 1.0).astype(np.float32)
        return cls(names, mean, scale)

    def transform(
        self,
        values: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("values have an incompatible feature dimension")
        if feature_names is not None:
            names = tuple(str(value) for value in feature_names)
            if names != self.feature_names:
                raise ValueError("feature order differs from the fitted standardizer")
        return ((values - self.mean) / self.scale).astype(np.float32, copy=False)


def load_paired_rna_protein(
    rna_path: PathLike,
    protein_path: PathLike,
    manifest_path: PathLike,
    *,
    rna_gene_order: Sequence[str] | None = None,
    protein_order: Sequence[str] | None = None,
    sample_id_column: str = "sample_id",
    strict_sample_match: bool = True,
) -> PairedRNAProteinData:
    """Load paired matrices while preserving canonical RNA/protein positions."""

    rna_frame = _read_matrix(rna_path)
    protein_frame = _read_matrix(protein_path)
    manifest = pd.read_csv(manifest_path, sep="\t")
    if sample_id_column not in manifest.columns:
        raise ValueError(f"manifest lacks {sample_id_column!r}")
    manifest[sample_id_column] = manifest[sample_id_column].map(str)
    if manifest[sample_id_column].duplicated().any():
        raise ValueError("sample identifiers are duplicated in the manifest")
    manifest = manifest.set_index(sample_id_column, drop=True)
    manifest.index.name = sample_id_column

    rna_samples = set(rna_frame.index)
    protein_samples = set(protein_frame.index)
    manifest_samples = set(manifest.index)
    if strict_sample_match:
        if rna_samples != protein_samples or rna_samples != manifest_samples:
            raise ValueError(
                "RNA, protein and manifest sample sets differ; "
                f"RNA-only={len(rna_samples - protein_samples - manifest_samples)}, "
                f"protein-only={len(protein_samples - rna_samples)}, "
                f"manifest-only={len(manifest_samples - rna_samples)}"
            )
        sample_ids = tuple(rna_frame.index)
    else:
        common = protein_samples & manifest_samples
        sample_ids = tuple(sample for sample in rna_frame.index if sample in common)
        if not sample_ids:
            raise ValueError("the three inputs have no common samples")

    gene_names = _ordered_columns(rna_frame, rna_gene_order, "rna_gene_order")
    protein_names = _ordered_columns(protein_frame, protein_order, "protein_order")
    rna_frame = rna_frame.loc[list(sample_ids), list(gene_names)]
    protein_frame = protein_frame.loc[list(sample_ids), list(protein_names)]
    manifest = manifest.loc[list(sample_ids)].copy()

    rna = rna_frame.to_numpy(dtype=np.float32, copy=True)
    protein = protein_frame.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(rna).all():
        raise ValueError("RNA matrix contains missing or infinite values")
    if np.isinf(protein).any():
        raise ValueError("protein matrix contains infinite values")
    protein_mask = np.isfinite(protein)

    return PairedRNAProteinData(
        sample_ids=sample_ids,
        gene_names=gene_names,
        protein_names=protein_names,
        rna=rna,
        protein=protein,
        protein_mask=protein_mask,
        manifest=manifest,
    )


def dataset_summary(data: PairedRNAProteinData) -> Mapping[str, int | float]:
    return {
        "n_samples": data.n_samples,
        "n_genes": data.n_genes,
        "n_proteins": data.n_proteins,
        "n_observed_protein_values": int(data.protein_mask.sum()),
        "protein_observed_fraction": float(data.protein_mask.mean()),
    }
