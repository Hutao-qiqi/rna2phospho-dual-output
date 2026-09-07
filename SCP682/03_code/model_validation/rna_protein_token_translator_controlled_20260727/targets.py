from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PathLike = str | Path
_STATE_SCHEMA_VERSION = 1


def _as_unique_strings(values: Sequence[Any], name: str) -> tuple[str, ...]:
    raw = tuple(values)
    if any(bool(pd.isna(value)) for value in raw):
        raise ValueError(f"{name} contains missing identifiers")
    normalized = tuple(str(value) for value in raw)
    index = pd.Index(normalized)
    if not index.is_unique:
        duplicates = index[index.duplicated()].unique().tolist()
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
        raise ValueError(f"unsupported target matrix format: {path}")

    if pd.isna(frame.index).any() or pd.isna(frame.columns).any():
        raise ValueError(f"matrix identifiers contain missing values: {path}")
    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    if not frame.index.is_unique:
        raise ValueError(f"sample identifiers are duplicated in {path}")
    if not frame.columns.is_unique:
        raise ValueError(f"feature identifiers are duplicated in {path}")
    return frame


def _align_matrix(
    frame: pd.DataFrame,
    sample_ids: tuple[str, ...],
    feature_names: tuple[str, ...],
    matrix_name: str,
) -> np.ndarray:
    missing_samples = [sample for sample in sample_ids if sample not in frame.index]
    if missing_samples:
        raise ValueError(
            f"{matrix_name} matrix lacks RNA samples: {missing_samples[:5]}"
        )
    missing_features = [feature for feature in feature_names if feature not in frame.columns]
    if missing_features:
        raise ValueError(
            f"{matrix_name} matrix lacks locked features: {missing_features[:5]}"
        )

    aligned = frame.loc[list(sample_ids), list(feature_names)]
    try:
        values = aligned.to_numpy(dtype=np.float32, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{matrix_name} matrix contains non-numeric values") from error
    if np.isinf(values).any():
        raise ValueError(f"{matrix_name} matrix contains infinite values")
    return values


def load_aligned_target_matrix(
    path: PathLike,
    sample_ids: Sequence[str],
    target_names: Sequence[str],
) -> np.ndarray:
    """Load one raw target matrix in the exact requested row and column order."""

    samples = _as_unique_strings(sample_ids, "sample_ids")
    targets = _as_unique_strings(target_names, "target_names")
    return _align_matrix(_read_matrix(path), samples, targets, "target")


@dataclass(frozen=True)
class ProteinPhosphositeTargets:
    """Raw target matrices aligned to fixed RNA samples and target vocabularies."""

    sample_ids: tuple[str, ...]
    protein_names: tuple[str, ...]
    site_names: tuple[str, ...]
    protein: np.ndarray
    phosphosite: np.ndarray
    protein_mask: np.ndarray
    site_mask: np.ndarray

    def __post_init__(self) -> None:
        n_samples = len(self.sample_ids)
        protein_shape = (n_samples, len(self.protein_names))
        site_shape = (n_samples, len(self.site_names))
        if self.protein.shape != protein_shape:
            raise ValueError(
                f"protein shape {self.protein.shape} does not match {protein_shape}"
            )
        if self.phosphosite.shape != site_shape:
            raise ValueError(
                f"phosphosite shape {self.phosphosite.shape} does not match {site_shape}"
            )
        if self.protein_mask.shape != protein_shape:
            raise ValueError("protein mask shape does not match protein matrix")
        if self.site_mask.shape != site_shape:
            raise ValueError("site mask shape does not match phosphosite matrix")
        if self.protein_mask.dtype != np.bool_ or self.site_mask.dtype != np.bool_:
            raise TypeError("target masks must have boolean dtype")
        if np.isinf(self.protein).any() or np.isinf(self.phosphosite).any():
            raise ValueError("target matrices cannot contain infinite values")
        if not np.array_equal(self.protein_mask, np.isfinite(self.protein)):
            raise ValueError("protein_mask does not match finite protein entries")
        if not np.array_equal(self.site_mask, np.isfinite(self.phosphosite)):
            raise ValueError("site_mask does not match finite phosphosite entries")

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)

    @property
    def n_proteins(self) -> int:
        return len(self.protein_names)

    @property
    def n_sites(self) -> int:
        return len(self.site_names)

    @property
    def protein_ids(self) -> np.ndarray:
        return np.arange(self.n_proteins, dtype=np.int64)

    @property
    def site_ids(self) -> np.ndarray:
        return np.arange(self.n_sites, dtype=np.int64)

    @property
    def total_protein(self) -> np.ndarray:
        return self.protein

    @property
    def total_protein_mask(self) -> np.ndarray:
        return self.protein_mask

    @property
    def phosphosite_mask(self) -> np.ndarray:
        return self.site_mask

    def filled_protein(self, fill_value: float = 0.0) -> np.ndarray:
        return np.where(self.protein_mask, self.protein, fill_value).astype(
            np.float32, copy=False
        )

    def filled_phosphosite(self, fill_value: float = 0.0) -> np.ndarray:
        return np.where(self.site_mask, self.phosphosite, fill_value).astype(
            np.float32, copy=False
        )


def load_raw_targets(
    total_protein_path: PathLike,
    phosphosite_path: PathLike,
    sample_ids: Sequence[str],
    protein_names: Sequence[str],
    site_names: Sequence[str],
) -> ProteinPhosphositeTargets:
    """Read raw targets and select the exact RNA sample and vocabulary order."""

    samples = _as_unique_strings(sample_ids, "sample_ids")
    proteins = _as_unique_strings(protein_names, "protein_names")
    sites = _as_unique_strings(site_names, "site_names")
    protein = load_aligned_target_matrix(total_protein_path, samples, proteins)
    phosphosite = load_aligned_target_matrix(phosphosite_path, samples, sites)
    protein_mask = np.isfinite(protein)
    site_mask = np.isfinite(phosphosite)
    return ProteinPhosphositeTargets(
        sample_ids=samples,
        protein_names=proteins,
        site_names=sites,
        protein=protein,
        phosphosite=phosphosite,
        protein_mask=protein_mask,
        site_mask=site_mask,
    )


def _one_dimensional_groups(
    values: Sequence[Any] | np.ndarray,
    n_samples: int,
) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or raw.size != n_samples:
        raise ValueError("pdc_study_ids must be one-dimensional and match values")
    if pd.isna(raw).any():
        raise ValueError("pdc_study_ids contain missing values")
    return np.asarray([str(value) for value in raw], dtype=np.str_)


def _training_indices(
    train_idx: Sequence[int] | np.ndarray,
    n_samples: int,
) -> np.ndarray:
    indices = np.asarray(train_idx)
    if indices.ndim != 1:
        raise ValueError("train_idx must be one-dimensional")
    if indices.dtype == np.bool_:
        if indices.size != n_samples:
            raise ValueError("boolean train_idx must match the sample count")
        indices = np.flatnonzero(indices)
    elif not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("train_idx must contain integer indices or booleans")
    indices = indices.astype(np.int64, copy=False)
    if indices.size == 0:
        raise ValueError("cannot fit a standardizer on zero training samples")
    if np.unique(indices).size != indices.size:
        raise ValueError("train_idx contains duplicate indices")
    if int(indices.min()) < 0 or int(indices.max()) >= n_samples:
        raise IndexError("train_idx contains out-of-range indices")
    return indices


def _feature_names(
    feature_names: Sequence[str] | None,
    n_features: int,
) -> tuple[str, ...]:
    if feature_names is None:
        return tuple(str(index) for index in range(n_features))
    names = _as_unique_strings(feature_names, "feature_names")
    if len(names) != n_features:
        raise ValueError("feature_names length does not match values")
    return names


def _observed_mask(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.isfinite(values)
    observed = np.asarray(mask)
    if observed.shape != values.shape:
        raise ValueError("mask shape does not match values")
    if observed.dtype != np.bool_:
        raise TypeError("mask must have boolean dtype")
    if np.any(observed & ~np.isfinite(values)):
        raise ValueError("mask marks non-finite values as observed")
    return observed


def _masked_stats(
    values: np.ndarray,
    mask: np.ndarray,
    minimum_scale: float,
    fallback_mean: np.ndarray | None = None,
    fallback_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_features = values.shape[1]
    count = mask.sum(axis=0, dtype=np.int64)
    sums = np.where(mask, values, 0.0).sum(axis=0, dtype=np.float64)

    mean = (
        np.zeros(n_features, dtype=np.float64)
        if fallback_mean is None
        else np.asarray(fallback_mean, dtype=np.float64).copy()
    )
    np.divide(sums, count, out=mean, where=count > 0)

    centered = np.where(mask, values - mean, 0.0)
    sum_squares = np.square(centered).sum(axis=0, dtype=np.float64)
    variance = np.zeros(n_features, dtype=np.float64)
    np.divide(sum_squares, count, out=variance, where=count > 0)
    local_std = np.sqrt(np.maximum(variance, 0.0))
    local_std = np.where(local_std >= minimum_scale, local_std, 1.0)

    std = (
        np.ones(n_features, dtype=np.float64)
        if fallback_std is None
        else np.asarray(fallback_std, dtype=np.float64).copy()
    )
    std[count > 0] = local_std[count > 0]
    return mean, std, count


def _stat_vector(values: Any, n_features: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (n_features,):
        raise ValueError(f"{name} has an incompatible shape")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values")
    return vector.copy()


def _count_vector(values: Any, n_features: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.int64)
    if vector.shape != (n_features,):
        raise ValueError(f"{name} has an incompatible shape")
    if (vector < 0).any():
        raise ValueError(f"{name} contains negative counts")
    return vector.copy()


class GroupedMaskedStandardizer:
    """Per-study masked standardizer fitted exclusively on training rows."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        global_mean: np.ndarray,
        global_std: np.ndarray,
        global_count: np.ndarray,
        group_mean: Mapping[str, np.ndarray],
        group_std: Mapping[str, np.ndarray],
        group_count: Mapping[str, np.ndarray],
        minimum_scale: float = 1e-6,
    ) -> None:
        names = _as_unique_strings(feature_names, "feature_names")
        n_features = len(names)
        if not np.isfinite(minimum_scale) or minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive and finite")

        mean_keys = tuple(str(key) for key in group_mean)
        if len(mean_keys) != len(set(mean_keys)):
            raise ValueError("group identifiers collide after string conversion")
        if set(mean_keys) != {str(key) for key in group_std} or set(mean_keys) != {
            str(key) for key in group_count
        }:
            raise ValueError("group statistic keys do not match")

        std_lookup = {str(key): value for key, value in group_std.items()}
        count_lookup = {str(key): value for key, value in group_count.items()}
        mean_lookup = {str(key): value for key, value in group_mean.items()}
        self.feature_names = names
        self.global_mean = _stat_vector(global_mean, n_features, "global_mean")
        self.global_std = _stat_vector(global_std, n_features, "global_std")
        if (self.global_std <= 0).any():
            raise ValueError("global_std must be positive")
        self.global_count = _count_vector(global_count, n_features, "global_count")
        self.group_mean = {
            group: _stat_vector(mean_lookup[group], n_features, f"group_mean[{group!r}]")
            for group in mean_keys
        }
        self.group_std = {
            group: _stat_vector(std_lookup[group], n_features, f"group_std[{group!r}]")
            for group in mean_keys
        }
        if any((std <= 0).any() for std in self.group_std.values()):
            raise ValueError("group standard deviations must be positive")
        self.group_count = {
            group: _count_vector(
                count_lookup[group], n_features, f"group_count[{group!r}]"
            )
            for group in mean_keys
        }
        self.minimum_scale = float(minimum_scale)

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        pdc_study_ids: Sequence[Any] | np.ndarray,
        train_idx: Sequence[int] | np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
        mask: np.ndarray | None = None,
        minimum_scale: float = 1e-6,
    ) -> "GroupedMaskedStandardizer":
        array = np.asarray(values)
        if array.ndim != 2:
            raise ValueError("values must be two-dimensional")
        if not np.isfinite(minimum_scale) or minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive and finite")
        names = _feature_names(feature_names, array.shape[1])
        groups = _one_dimensional_groups(pdc_study_ids, array.shape[0])
        indices = _training_indices(train_idx, array.shape[0])

        train_values = np.asarray(array[indices], dtype=np.float64)
        if mask is None:
            train_mask = np.isfinite(train_values)
        else:
            full_mask = np.asarray(mask)
            if full_mask.shape != array.shape:
                raise ValueError("mask shape does not match values")
            if full_mask.dtype != np.bool_:
                raise TypeError("mask must have boolean dtype")
            train_mask = full_mask[indices]
            if np.any(train_mask & ~np.isfinite(train_values)):
                raise ValueError("mask marks non-finite training values as observed")

        global_mean, global_std, global_count = _masked_stats(
            train_values, train_mask, float(minimum_scale)
        )
        train_groups = groups[indices]
        group_order = tuple(dict.fromkeys(train_groups.tolist()))
        group_mean: dict[str, np.ndarray] = {}
        group_std: dict[str, np.ndarray] = {}
        group_count: dict[str, np.ndarray] = {}
        for group in group_order:
            rows = train_groups == group
            mean, std, count = _masked_stats(
                train_values[rows],
                train_mask[rows],
                float(minimum_scale),
                fallback_mean=global_mean,
                fallback_std=global_std,
            )
            group_mean[group] = mean
            group_std[group] = std
            group_count[group] = count

        return cls(
            feature_names=names,
            global_mean=global_mean,
            global_std=global_std,
            global_count=global_count,
            group_mean=group_mean,
            group_std=group_std,
            group_count=group_count,
            minimum_scale=minimum_scale,
        )

    @property
    def known_groups(self) -> tuple[str, ...]:
        return tuple(self.group_mean)

    @property
    def global_scale(self) -> np.ndarray:
        return self.global_std

    @property
    def group_means(self) -> Mapping[str, np.ndarray]:
        return self.group_mean

    @property
    def group_scales(self) -> Mapping[str, np.ndarray]:
        return self.group_std

    def statistics_for(self, pdc_study_id: Any) -> tuple[np.ndarray, np.ndarray]:
        group = str(pdc_study_id)
        return (
            self.group_mean.get(group, self.global_mean),
            self.group_std.get(group, self.global_std),
        )

    def _validate_feature_order(self, feature_names: Sequence[str] | None) -> None:
        if feature_names is None:
            return
        names = tuple(str(value) for value in feature_names)
        if names != self.feature_names:
            raise ValueError("feature order differs from the fitted standardizer")

    def _apply(
        self,
        values: np.ndarray,
        pdc_study_ids: Sequence[Any] | np.ndarray,
        *,
        mask: np.ndarray | None,
        feature_names: Sequence[str] | None,
        inverse: bool,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            raise ValueError("values have an incompatible feature dimension")
        self._validate_feature_order(feature_names)
        groups = _one_dimensional_groups(pdc_study_ids, array.shape[0])
        observed = _observed_mask(array, mask)
        result = np.full(array.shape, np.nan, dtype=np.float64)

        for group in dict.fromkeys(groups.tolist()):
            rows = groups == group
            mean, std = self.statistics_for(group)
            if inverse:
                changed = array[rows] * std + mean
            else:
                changed = (array[rows] - mean) / std
            result[rows] = np.where(observed[rows], changed, np.nan)
        return result.astype(np.float32)

    def transform(
        self,
        values: np.ndarray,
        pdc_study_ids: Sequence[Any] | np.ndarray,
        *,
        mask: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        return self._apply(
            values,
            pdc_study_ids,
            mask=mask,
            feature_names=feature_names,
            inverse=False,
        )

    def inverse_transform(
        self,
        values: np.ndarray,
        pdc_study_ids: Sequence[Any] | np.ndarray,
        *,
        mask: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        return self._apply(
            values,
            pdc_study_ids,
            mask=mask,
            feature_names=feature_names,
            inverse=True,
        )

    def state_dict(self) -> dict[str, np.ndarray]:
        group_names = self.known_groups
        n_features = len(self.feature_names)
        if group_names:
            group_mean = np.stack([self.group_mean[group] for group in group_names])
            group_std = np.stack([self.group_std[group] for group in group_names])
            group_count = np.stack([self.group_count[group] for group in group_names])
        else:
            group_mean = np.empty((0, n_features), dtype=np.float64)
            group_std = np.empty((0, n_features), dtype=np.float64)
            group_count = np.empty((0, n_features), dtype=np.int64)
        return {
            "schema_version": np.asarray([_STATE_SCHEMA_VERSION], dtype=np.int16),
            "feature_names": np.asarray(self.feature_names, dtype=np.str_),
            "minimum_scale": np.asarray([self.minimum_scale], dtype=np.float64),
            "global_mean": self.global_mean.copy(),
            "global_std": self.global_std.copy(),
            "global_count": self.global_count.copy(),
            "group_names": np.asarray(group_names, dtype=np.str_),
            "group_mean": group_mean,
            "group_std": group_std,
            "group_count": group_count,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
    ) -> "GroupedMaskedStandardizer":
        required = {
            "schema_version",
            "feature_names",
            "minimum_scale",
            "global_mean",
            "global_std",
            "global_count",
            "group_names",
            "group_mean",
            "group_std",
            "group_count",
        }
        missing = required - set(state)
        if missing:
            raise ValueError(f"standardizer state lacks fields: {sorted(missing)}")
        version_values = np.asarray(state["schema_version"]).reshape(-1)
        if version_values.size != 1 or int(version_values[0]) != _STATE_SCHEMA_VERSION:
            raise ValueError("unsupported standardizer state schema version")

        feature_names = tuple(np.asarray(state["feature_names"]).astype(str).tolist())
        group_names = tuple(np.asarray(state["group_names"]).astype(str).tolist())
        if len(group_names) != len(set(group_names)):
            raise ValueError("standardizer state contains duplicate groups")
        n_features = len(feature_names)
        group_mean_array = np.asarray(state["group_mean"], dtype=np.float64)
        group_std_array = np.asarray(state["group_std"], dtype=np.float64)
        group_count_array = np.asarray(state["group_count"], dtype=np.int64)
        expected_group_shape = (len(group_names), n_features)
        if (
            group_mean_array.shape != expected_group_shape
            or group_std_array.shape != expected_group_shape
            or group_count_array.shape != expected_group_shape
        ):
            raise ValueError("saved group statistics have incompatible shapes")
        minimum_scale_values = np.asarray(state["minimum_scale"], dtype=np.float64).reshape(-1)
        if minimum_scale_values.size != 1:
            raise ValueError("saved minimum_scale has an incompatible shape")

        return cls(
            feature_names=feature_names,
            global_mean=np.asarray(state["global_mean"]),
            global_std=np.asarray(state["global_std"]),
            global_count=np.asarray(state["global_count"]),
            group_mean={
                group: group_mean_array[index] for index, group in enumerate(group_names)
            },
            group_std={
                group: group_std_array[index] for index, group in enumerate(group_names)
            },
            group_count={
                group: group_count_array[index] for index, group in enumerate(group_names)
            },
            minimum_scale=float(minimum_scale_values[0]),
        )

    def save(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                dir=path.parent, suffix=".npz", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                np.savez_compressed(handle, **self.state_dict())
            temporary_path.replace(path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return path

    @classmethod
    def load(cls, path: PathLike) -> "GroupedMaskedStandardizer":
        with np.load(Path(path), allow_pickle=False) as archive:
            state = {name: archive[name] for name in archive.files}
        return cls.from_state_dict(state)


AlignedTargets = ProteinPhosphositeTargets
MaskedGroupStandardizer = GroupedMaskedStandardizer
load_aligned_targets = load_raw_targets
load_protein_phosphosite_targets = load_raw_targets


__all__ = [
    "AlignedTargets",
    "GroupedMaskedStandardizer",
    "MaskedGroupStandardizer",
    "ProteinPhosphositeTargets",
    "load_aligned_target_matrix",
    "load_aligned_targets",
    "load_protein_phosphosite_targets",
    "load_raw_targets",
]
