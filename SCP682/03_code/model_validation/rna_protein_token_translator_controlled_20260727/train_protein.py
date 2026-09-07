"""Train one held-out fold of the protein-only Performer translator."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from .candidate_lowrank_cognate import LowRankCognateProteinTranslator
    from .candidate_lowrank_residual_v2 import LowRankResidualProteinTranslatorV2
    from .candidate_moe_lowrank import MoELowRankProteinTranslator
    from .candidate_residualized_lowrank import (
        ResidualizedLowRankProteinTranslator,
        fit_cognate_linear_anchor,
    )
    from .candidate_query_latent import QueryLatentConfig, RNAProteinQueryLatent
    from .candidate_supervised_factor import (
        SupervisedFactorConfig,
        SupervisedFactorProteinTranslator,
    )
    from .data import FeatureStandardizer
    from .losses import masked_huber, masked_mse_loss, protein_translation_loss
    from .metrics import (
        batchwise_flattened_cosine,
        masked_mse as array_masked_mse,
        per_protein_spearman,
        summarize_spearman,
    )
    from .protein_model import ProteinModelConfig, RNAProteinPerformerTranslator
    from .rank_losses import correlation_guided_loss
    from .splits import make_strict_folds, save_fold_indices
    from .targets import GroupedMaskedStandardizer, load_aligned_target_matrix
except ImportError:
    from candidate_lowrank_cognate import LowRankCognateProteinTranslator
    from candidate_lowrank_residual_v2 import LowRankResidualProteinTranslatorV2
    from candidate_moe_lowrank import MoELowRankProteinTranslator
    from candidate_residualized_lowrank import (
        ResidualizedLowRankProteinTranslator,
        fit_cognate_linear_anchor,
    )
    from candidate_query_latent import QueryLatentConfig, RNAProteinQueryLatent
    from candidate_supervised_factor import (
        SupervisedFactorConfig,
        SupervisedFactorProteinTranslator,
    )
    from data import FeatureStandardizer
    from losses import masked_huber, masked_mse_loss, protein_translation_loss
    from metrics import (
        batchwise_flattened_cosine,
        masked_mse as array_masked_mse,
        per_protein_spearman,
        summarize_spearman,
    )
    from protein_model import ProteinModelConfig, RNAProteinPerformerTranslator
    from rank_losses import correlation_guided_loss
    from splits import make_strict_folds, save_fold_indices
    from targets import GroupedMaskedStandardizer, load_aligned_target_matrix


TARGET_SCALE_RAW = "raw_log2_ratio"
TARGET_SCALE_GLOBAL = "global_protein_zscore"
TARGET_SCALE_STUDY = "study_protein_zscore"
TARGET_SCALE_TRAIN_MINMAX = "selection_train_protein_minmax_01"
TARGET_SCALE_SAMPLE_MINMAX = "per_sample_protein_minmax_01"
TARGET_SCALE_CHOICES = (
    TARGET_SCALE_RAW,
    TARGET_SCALE_GLOBAL,
    TARGET_SCALE_STUDY,
    TARGET_SCALE_TRAIN_MINMAX,
    TARGET_SCALE_SAMPLE_MINMAX,
)


@dataclass(frozen=True)
class Runtime:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class ProteinArrays:
    rna: np.ndarray
    protein: np.ndarray
    protein_mask: np.ndarray
    rna_scaler: FeatureStandardizer
    protein_scaler: "ProteinTargetTransform"


@dataclass(frozen=True)
class ProteinTargetTransform:
    """Target transform fitted only on an explicitly declared training partition."""

    mode: str
    feature_names: tuple[str, ...]
    global_mean: np.ndarray | None
    global_std: np.ndarray | None
    global_min: np.ndarray | None
    global_max: np.ndarray | None
    global_range: np.ndarray | None
    global_count: np.ndarray
    grouped: GroupedMaskedStandardizer | None = None
    minimum_scale: float = 1e-6

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        groups: np.ndarray,
        fit_indices: np.ndarray,
        *,
        feature_names: list[str],
        mode: str,
        mask: np.ndarray | None = None,
        minimum_scale: float = 1e-6,
    ) -> "ProteinTargetTransform":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("protein targets must be two-dimensional")
        if mode not in TARGET_SCALE_CHOICES:
            raise ValueError(f"unknown target scale: {mode}")
        names = tuple(str(value) for value in feature_names)
        if len(names) != array.shape[1] or len(set(names)) != len(names):
            raise ValueError("protein feature names must be unique and match targets")
        indices = np.asarray(fit_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("target transform requires non-empty training indices")
        if np.unique(indices).size != indices.size:
            raise ValueError("target-transform training indices contain duplicates")
        if int(indices.min()) < 0 or int(indices.max()) >= array.shape[0]:
            raise IndexError("target-transform training indices are out of range")
        observed = np.isfinite(array) if mask is None else np.asarray(mask)
        if observed.shape != array.shape or observed.dtype != np.bool_:
            raise ValueError("protein target mask must be boolean and match targets")
        if np.any(observed & ~np.isfinite(array)):
            raise ValueError("protein target mask marks non-finite values as observed")

        train_values = array[indices]
        train_mask = observed[indices]
        count = train_mask.sum(axis=0, dtype=np.int64)
        if mode == TARGET_SCALE_RAW:
            return cls(
                mode=mode,
                feature_names=names,
                global_mean=None,
                global_std=None,
                global_min=None,
                global_max=None,
                global_range=None,
                global_count=count,
                grouped=None,
                minimum_scale=float(minimum_scale),
            )

        if mode == TARGET_SCALE_SAMPLE_MINMAX:
            return cls(
                mode=mode,
                feature_names=names,
                global_mean=None,
                global_std=None,
                global_min=None,
                global_max=None,
                global_range=None,
                global_count=count,
                grouped=None,
                minimum_scale=float(minimum_scale),
            )

        if mode == TARGET_SCALE_TRAIN_MINMAX:
            minimum = np.where(train_mask, train_values, np.inf).min(axis=0)
            maximum = np.where(train_mask, train_values, -np.inf).max(axis=0)
            minimum = np.where(count > 0, minimum, 0.0)
            maximum = np.where(count > 0, maximum, 0.0)
            spread = np.maximum(maximum - minimum, 0.0)
            safe_range = np.where(spread >= minimum_scale, spread, 1.0)
            return cls(
                mode=mode,
                feature_names=names,
                global_mean=None,
                global_std=None,
                global_min=minimum,
                global_max=maximum,
                global_range=safe_range,
                global_count=count,
                grouped=None,
                minimum_scale=float(minimum_scale),
            )

        if mode == TARGET_SCALE_STUDY:
            grouped = GroupedMaskedStandardizer.fit(
                array,
                groups,
                indices,
                feature_names=names,
                mask=observed,
                minimum_scale=minimum_scale,
            )
            return cls(
                mode=mode,
                feature_names=names,
                global_mean=grouped.global_mean.copy(),
                global_std=grouped.global_std.copy(),
                global_min=None,
                global_max=None,
                global_range=None,
                global_count=grouped.global_count.copy(),
                grouped=grouped,
                minimum_scale=float(minimum_scale),
            )

        sums = np.where(train_mask, train_values, 0.0).sum(
            axis=0, dtype=np.float64
        )
        mean = np.zeros(array.shape[1], dtype=np.float64)
        np.divide(sums, count, out=mean, where=count > 0)
        centered = np.where(train_mask, train_values - mean, 0.0)
        sum_squares = np.square(centered).sum(axis=0, dtype=np.float64)
        variance = np.zeros(array.shape[1], dtype=np.float64)
        np.divide(sum_squares, count, out=variance, where=count > 0)
        std = np.sqrt(np.maximum(variance, 0.0))
        std = np.where(std >= minimum_scale, std, 1.0)
        return cls(
            mode=mode,
            feature_names=names,
            global_mean=mean,
            global_std=std,
            global_min=None,
            global_max=None,
            global_range=None,
            global_count=count,
            grouped=None,
            minimum_scale=float(minimum_scale),
        )

    def transform(
        self,
        values: np.ndarray,
        groups: np.ndarray,
        *,
        mask: np.ndarray,
        feature_names: list[str],
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        observed = np.asarray(mask)
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            raise ValueError("protein targets have an incompatible shape")
        if tuple(str(value) for value in feature_names) != self.feature_names:
            raise ValueError("protein feature order differs from fitted target transform")
        if observed.shape != array.shape or observed.dtype != np.bool_:
            raise ValueError("protein target mask must be boolean and match targets")
        if self.mode == TARGET_SCALE_RAW:
            changed = array
        elif self.mode == TARGET_SCALE_GLOBAL:
            if self.global_mean is None or self.global_std is None:
                raise RuntimeError("global target statistics are unavailable")
            changed = (array - self.global_mean) / self.global_std
        elif self.mode == TARGET_SCALE_TRAIN_MINMAX:
            if self.global_min is None or self.global_range is None:
                raise RuntimeError("selection-train min-max statistics are unavailable")
            changed = (array - self.global_min) / self.global_range
        elif self.mode == TARGET_SCALE_SAMPLE_MINMAX:
            sample_count = observed.sum(axis=1, dtype=np.int64)
            minimum = np.where(observed, array, np.inf).min(axis=1)
            maximum = np.where(observed, array, -np.inf).max(axis=1)
            minimum = np.where(sample_count > 0, minimum, 0.0)
            maximum = np.where(sample_count > 0, maximum, 0.0)
            spread = np.maximum(maximum - minimum, 0.0)
            safe_range = np.where(spread >= self.minimum_scale, spread, 1.0)
            changed = (array - minimum[:, None]) / safe_range[:, None]
        elif self.mode == TARGET_SCALE_STUDY:
            if self.grouped is None:
                raise RuntimeError("study target standardizer is unavailable")
            return self.grouped.transform(
                array,
                groups,
                mask=observed,
                feature_names=feature_names,
            )
        else:
            raise RuntimeError(f"unsupported fitted target scale: {self.mode}")
        return np.where(observed, changed, np.nan).astype(np.float32)

    def inverse_transform(
        self,
        values: np.ndarray,
        groups: np.ndarray,
        *,
        feature_names: list[str],
    ) -> np.ndarray:
        """Return predictions to the raw protein scale without using labels."""

        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            raise ValueError("protein predictions have an incompatible shape")
        if tuple(str(value) for value in feature_names) != self.feature_names:
            raise ValueError("protein feature order differs from fitted target transform")
        group_array = np.asarray(groups).astype(str)
        if group_array.shape != (array.shape[0],):
            raise ValueError("groups must contain one entry per prediction row")
        if self.mode == TARGET_SCALE_RAW:
            restored = array
        elif self.mode == TARGET_SCALE_GLOBAL:
            if self.global_mean is None or self.global_std is None:
                raise RuntimeError("global target statistics are unavailable")
            restored = array * self.global_std + self.global_mean
        elif self.mode == TARGET_SCALE_TRAIN_MINMAX:
            if self.global_min is None or self.global_range is None:
                raise RuntimeError("selection-train min-max statistics are unavailable")
            restored = array * self.global_range + self.global_min
        elif self.mode == TARGET_SCALE_STUDY:
            if self.grouped is None:
                raise RuntimeError("study target standardizer is unavailable")
            restored = np.empty_like(array, dtype=np.float64)
            for group in np.unique(group_array):
                rows = group_array == group
                mean, scale = self.grouped.statistics_for(group)
                restored[rows] = array[rows] * scale + mean
        elif self.mode == TARGET_SCALE_SAMPLE_MINMAX:
            raise ValueError(
                "per-sample min-max predictions cannot be inverted without sample label statistics"
            )
        else:
            raise RuntimeError(f"unsupported fitted target scale: {self.mode}")
        return restored.astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "schema_version": np.asarray([2], dtype=np.int16),
            "mode": self.mode,
            "feature_names": np.asarray(self.feature_names, dtype=np.str_),
            "minimum_scale": np.asarray([self.minimum_scale], dtype=np.float64),
            "global_count": self.global_count.copy(),
            "normalization_axis": (
                "observed_proteins_within_each_sample"
                if self.mode == TARGET_SCALE_SAMPLE_MINMAX
                else "proteins_across_declared_fitting_partition"
            ),
        }
        if self.global_mean is not None:
            state["global_mean"] = self.global_mean.copy()
        if self.global_std is not None:
            state["global_std"] = self.global_std.copy()
        if self.global_min is not None:
            state["global_min"] = self.global_min.copy()
        if self.global_max is not None:
            state["global_max"] = self.global_max.copy()
        if self.global_range is not None:
            state["global_range"] = self.global_range.copy()
        if self.grouped is not None:
            state["grouped_standardizer"] = self.grouped.state_dict()
        return state


class ProteinTrainingDataset(Dataset):
    def __init__(self, arrays: ProteinArrays, indices: np.ndarray) -> None:
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = int(self.indices[item])
        return {
            "sample_index": row,
            "rna": self.arrays.rna[row],
            "protein": self.arrays.protein[row],
            "protein_mask": self.arrays.protein_mask[row],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--strata-column", default="cancer_label")
    parser.add_argument("--group-column", default="case_submitter_id")
    parser.add_argument(
        "--inner-group-column",
        default=None,
        help=(
            "Optional manifest column used to block the inner selection split. "
            "Omitting it preserves case-level blocking."
        ),
    )
    parser.add_argument("--study-column", default="pdc_study_id")
    parser.add_argument("--rna-file", default="rna_log2_tpm_paired.parquet")
    parser.add_argument(
        "--protein-raw-file", default="total_protein_gene_logratio_all.parquet"
    )
    parser.add_argument(
        "--protein-vocab-file",
        default="total_protein_gene_study_zscore_min20pct.parquet",
    )
    parser.add_argument("--manifest-file", default="sample_manifest.tsv")

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument(
        "--model-family",
        choices=[
            "reference",
            "lowrank_cognate",
            "lowrank_residual_v2",
            "residualized_lowrank",
            "moe_lowrank",
            "query_latent",
            "supervised_factor",
        ],
        default="reference",
    )
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dim-head", type=int, default=64)
    parser.add_argument("--translator-depth", type=int, choices=[1, 2], default=2)
    parser.add_argument("--translator-hidden", type=int, default=0)
    parser.add_argument("--lowrank-rank", type=int, default=512)
    parser.add_argument("--moe-experts", type=int, default=4)
    parser.add_argument("--moe-gate-hidden", type=int, default=64)
    parser.add_argument(
        "--lowrank-ranks",
        type=int,
        nargs="+",
        default=[64, 128, 320],
        help="Branch ranks for the multi-scale low-rank residual model.",
    )
    parser.add_argument("--initial-cognate-gate", type=float, default=0.20)
    parser.add_argument("--state-tokens", type=int, default=256)
    parser.add_argument("--protein-query-chunk", type=int, default=2048)
    parser.add_argument("--protein-factors", type=int, default=256)
    parser.add_argument("--sample-queries", type=int, default=16)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--gene-mask-probability", type=float, default=0.10)
    parser.add_argument("--nb-features", type=int)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--translator-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=0.05)
    parser.add_argument("--rank-temperature", type=float, default=0.20)
    parser.add_argument("--rank-max-targets", type=int, default=2048)
    parser.add_argument(
        "--loss-mode",
        choices=["legacy", "correlation", "mse"],
        default="legacy",
        help="Training objective. The default preserves protein_translation_loss.",
    )
    parser.add_argument(
        "--value-reduction",
        choices=["per_observation", "per_target"],
        default="per_observation",
        help="Reduction for the value term used by correlation loss mode.",
    )
    parser.add_argument("--pearson-weight", type=float, default=0.10)
    parser.add_argument("--spearman-weight", type=float, default=0.05)
    parser.add_argument("--correlation-warmup-fraction", type=float, default=0.10)
    parser.add_argument("--correlation-min-observations", type=int, default=4)
    parser.add_argument(
        "--pearson-max-targets",
        type=int,
        default=None,
        help="Maximum protein columns used by Pearson loss; omit for all columns.",
    )
    parser.add_argument(
        "--gather-correlation-across-ranks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build correlation statistics from the differentiably gathered DDP batch.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["spearman", "huber", "batch_cosine"],
        default="spearman",
        help="Inner-validation metric used for early stopping and epoch selection.",
    )
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--target-scale",
        choices=TARGET_SCALE_CHOICES,
        default=TARGET_SCALE_STUDY,
        help=(
            "Protein-label scale fitted on the declared training partition. "
            "The default preserves study-specific protein z-scoring."
        ),
    )
    parser.add_argument("--refit-full-outer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="Select on the inner validation split and stop without touching the outer test set.",
    )

    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-genes", type=int)
    parser.add_argument("--max-proteins", type=int)
    parser.add_argument(
        "--min-protein-coverage",
        type=float,
        default=None,
        help=(
            "Minimum observed-label fraction computed only on selection-train. "
            "This target screen is currently restricted to --screen-only runs."
        ),
    )
    args = parser.parse_args()
    if args.pearson_weight < 0 or args.spearman_weight < 0:
        parser.error("correlation weights must be non-negative")
    if not 0.0 <= args.correlation_warmup_fraction <= 1.0:
        parser.error("--correlation-warmup-fraction must be in [0, 1]")
    if args.correlation_min_observations < 2:
        parser.error("--correlation-min-observations must be at least 2")
    if args.pearson_max_targets is not None and args.pearson_max_targets <= 0:
        parser.error("--pearson-max-targets must be positive when provided")
    if args.min_protein_coverage is not None:
        if not 0.0 < args.min_protein_coverage <= 1.0:
            parser.error("--min-protein-coverage must be in (0, 1]")
        if not args.screen_only:
            parser.error("--min-protein-coverage currently requires --screen-only")
    if args.moe_experts < 2:
        parser.error("--moe-experts must be at least 2")
    if args.moe_gate_hidden < 1:
        parser.error("--moe-gate-hidden must be positive")
    return args


def setup_runtime() -> Runtime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        try:
            dist.init_process_group(
                backend=backend,
                device_id=device if device.type == "cuda" else None,
            )
        except TypeError:
            dist.init_process_group(backend=backend)
    return Runtime(rank, world_size, local_rank, device)


def barrier(runtime: Runtime) -> None:
    if runtime.world_size > 1:
        dist.barrier()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_parquet_columns(path: Path) -> list[str]:
    # Pandas excludes the physical parquet index column and preserves the
    # exact protein order encoded in the training matrix metadata.
    return list(pd.read_parquet(path).columns.map(str))


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir.resolve()
    paths = {
        "rna": data_dir / args.rna_file,
        "protein_raw": data_dir / args.protein_raw_file,
        "protein_vocab": data_dir / args.protein_vocab_file,
        "manifest": data_dir / args.manifest_file,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing protein-stage inputs:\n" + "\n".join(missing))

    rna_frame = pd.read_parquet(paths["rna"])
    rna_frame.index = rna_frame.index.map(str)
    rna_frame.columns = rna_frame.columns.map(str)
    if not rna_frame.index.is_unique or not rna_frame.columns.is_unique:
        raise ValueError("RNA sample and gene identifiers must be unique")

    manifest = pd.read_csv(paths["manifest"], sep="\t")
    required_manifest = {
        args.sample_id_column,
        args.strata_column,
        args.group_column,
        args.study_column,
    }
    absent = required_manifest - set(manifest.columns)
    if absent:
        raise ValueError(f"manifest lacks columns: {sorted(absent)}")
    manifest[args.sample_id_column] = manifest[args.sample_id_column].map(str)
    manifest = manifest.set_index(args.sample_id_column)
    if not manifest.index.is_unique:
        raise ValueError("manifest sample identifiers must be unique")
    missing_manifest = [sample for sample in rna_frame.index if sample not in manifest.index]
    if missing_manifest:
        raise ValueError(f"manifest lacks RNA samples: {missing_manifest[:5]}")
    manifest = manifest.loc[rna_frame.index].copy()

    gene_names = list(rna_frame.columns)
    protein_names = read_parquet_columns(paths["protein_vocab"])
    if args.max_samples:
        rna_frame = rna_frame.iloc[: args.max_samples]
        manifest = manifest.loc[rna_frame.index]
    if args.max_genes:
        gene_names = gene_names[: args.max_genes]
        rna_frame = rna_frame.loc[:, gene_names]
    if args.max_proteins:
        protein_names = protein_names[: args.max_proteins]

    sample_ids = list(rna_frame.index)
    protein_raw = load_aligned_target_matrix(
        paths["protein_raw"], sample_ids, protein_names
    )
    rna = rna_frame.to_numpy(dtype=np.float32)
    if not np.isfinite(rna).all():
        raise ValueError("RNA contains missing or infinite values")
    if np.isinf(protein_raw).any():
        raise ValueError("protein matrix contains infinite values")
    return {
        "sample_ids": sample_ids,
        "gene_names": gene_names,
        "protein_names": protein_names,
        "rna": rna,
        "protein_raw": protein_raw.astype(np.float32),
        "manifest": manifest,
    }


def prepare_arrays(
    inputs: dict[str, Any],
    fit_indices: np.ndarray,
    study_column: str,
    target_scale: str = TARGET_SCALE_STUDY,
) -> ProteinArrays:
    rna_scaler = FeatureStandardizer.fit(
        inputs["rna"], inputs["gene_names"], indices=fit_indices
    )
    rna = rna_scaler.transform(inputs["rna"], inputs["gene_names"])
    groups = inputs["manifest"][study_column].astype(str).to_numpy()
    raw_mask = np.isfinite(inputs["protein_raw"])
    protein_scaler = ProteinTargetTransform.fit(
        inputs["protein_raw"],
        groups,
        fit_indices,
        feature_names=inputs["protein_names"],
        mode=target_scale,
        mask=raw_mask,
    )
    standardized = protein_scaler.transform(
        inputs["protein_raw"],
        groups,
        mask=raw_mask,
        feature_names=inputs["protein_names"],
    )
    protein_mask = np.isfinite(standardized)
    return ProteinArrays(
        rna=rna.astype(np.float32),
        protein=np.nan_to_num(standardized, nan=0.0).astype(np.float32),
        protein_mask=protein_mask,
        rna_scaler=rna_scaler,
        protein_scaler=protein_scaler,
    )


def make_config(
    args: argparse.Namespace,
    inputs: dict[str, Any],
) -> ProteinModelConfig | QueryLatentConfig | SupervisedFactorConfig:
    common = dict(
        n_genes=len(inputs["gene_names"]),
        n_proteins=len(inputs["protein_names"]),
        d_model=args.d_model,
        encoder_depth=args.encoder_depth,
        n_heads=args.n_heads,
        dim_head=args.dim_head,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        gene_mask_probability=args.gene_mask_probability,
        nb_features=args.nb_features,
        feature_redraw_interval=None,
    )
    if args.model_family == "query_latent":
        return QueryLatentConfig(
            **common,
            n_state_tokens=args.state_tokens,
            protein_query_chunk=args.protein_query_chunk,
        )
    if args.model_family == "supervised_factor":
        return SupervisedFactorConfig(
            **common,
            n_factors=args.protein_factors,
            n_sample_queries=args.sample_queries,
        )
    return ProteinModelConfig(
        **common,
        decoder_depth=args.decoder_depth,
        translator_depth=args.translator_depth,
        translator_hidden=args.translator_hidden,
    )


def build_parent_gene_index(inputs: dict[str, Any]) -> np.ndarray:
    gene_to_index = {
        str(gene): index for index, gene in enumerate(inputs["gene_names"])
    }
    return np.asarray(
        [gene_to_index.get(str(protein), -1) for protein in inputs["protein_names"]],
        dtype=np.int64,
    )


def select_proteins_by_training_coverage(
    inputs: dict[str, Any],
    train_indices: np.ndarray,
    minimum_coverage: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select proteins using observed labels from the selection-train split only."""

    raw = np.asarray(inputs["protein_raw"])
    names = np.asarray(inputs["protein_names"], dtype=object)
    indices = np.asarray(train_indices, dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != names.size:
        raise ValueError("protein_raw and protein_names have incompatible shapes")
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("coverage selection requires non-empty one-dimensional indices")
    if int(indices.min()) < 0 or int(indices.max()) >= raw.shape[0]:
        raise IndexError("coverage-selection indices are out of range")
    if not 0.0 < float(minimum_coverage) <= 1.0:
        raise ValueError("minimum_coverage must be in (0, 1]")

    coverage = np.isfinite(raw[indices]).mean(axis=0, dtype=np.float64)
    selected = coverage >= float(minimum_coverage)
    if not np.any(selected):
        raise ValueError(
            "no proteins meet --min-protein-coverage on selection-train"
        )

    selected_indices = np.flatnonzero(selected)
    subset = dict(inputs)
    subset["protein_names"] = names[selected_indices].astype(str).tolist()
    subset["protein_raw"] = raw[:, selected_indices].astype(np.float32, copy=True)
    audit = pd.DataFrame(
        {
            "protein": names.astype(str),
            "selection_train_coverage": coverage,
            "selected": selected,
            "selected_position": np.where(
                selected,
                np.cumsum(selected, dtype=np.int64) - 1,
                -1,
            ),
            "n_selection_train": int(indices.size),
        }
    )
    return subset, audit


def construct_model(
    config: ProteinModelConfig | QueryLatentConfig | SupervisedFactorConfig,
    runtime: Runtime,
    seed: int,
    *,
    model_family: str = "reference",
    parent_gene_index: np.ndarray | None = None,
    lowrank_rank: int = 512,
    lowrank_ranks: tuple[int, ...] = (64, 128, 320),
    initial_cognate_gate: float = 0.20,
    anchor_intercept: np.ndarray | None = None,
    anchor_slope: np.ndarray | None = None,
    moe_experts: int = 4,
    moe_gate_hidden: int = 64,
) -> nn.Module:
    seed_everything(seed)
    if model_family == "reference":
        if not isinstance(config, ProteinModelConfig):
            raise TypeError("reference model requires ProteinModelConfig")
        model: nn.Module = RNAProteinPerformerTranslator(config)
    elif model_family == "lowrank_cognate":
        if not isinstance(config, ProteinModelConfig):
            raise TypeError("lowrank_cognate requires ProteinModelConfig")
        if parent_gene_index is None:
            raise ValueError("lowrank_cognate requires parent_gene_index")
        model = LowRankCognateProteinTranslator(
            config,
            parent_gene_index,
            rank=lowrank_rank,
        )
    elif model_family == "lowrank_residual_v2":
        if not isinstance(config, ProteinModelConfig):
            raise TypeError("lowrank_residual_v2 requires ProteinModelConfig")
        if parent_gene_index is None:
            raise ValueError("lowrank_residual_v2 requires parent_gene_index")
        model = LowRankResidualProteinTranslatorV2(
            config,
            parent_gene_index,
            ranks=lowrank_ranks,
            initial_cognate_gate=initial_cognate_gate,
        )
    elif model_family == "residualized_lowrank":
        if not isinstance(config, ProteinModelConfig):
            raise TypeError("residualized_lowrank requires ProteinModelConfig")
        if parent_gene_index is None:
            raise ValueError("residualized_lowrank requires parent_gene_index")
        if anchor_intercept is None or anchor_slope is None:
            raise ValueError("residualized_lowrank requires split-local anchor coefficients")
        model = ResidualizedLowRankProteinTranslator(
            config,
            parent_gene_index,
            anchor_intercept,
            anchor_slope,
            rank=lowrank_rank,
        )
    elif model_family == "moe_lowrank":
        if not isinstance(config, ProteinModelConfig):
            raise TypeError("moe_lowrank requires ProteinModelConfig")
        if parent_gene_index is None:
            raise ValueError("moe_lowrank requires parent_gene_index")
        if anchor_intercept is None or anchor_slope is None:
            raise ValueError("moe_lowrank requires split-local anchor coefficients")
        model = MoELowRankProteinTranslator(
            config,
            parent_gene_index,
            anchor_intercept,
            anchor_slope,
            rank=lowrank_rank,
            n_experts=moe_experts,
            gate_hidden=moe_gate_hidden,
        )
    elif model_family == "query_latent":
        if not isinstance(config, QueryLatentConfig):
            raise TypeError("query_latent requires QueryLatentConfig")
        if parent_gene_index is None:
            raise ValueError("query_latent requires parent_gene_index")
        model = RNAProteinQueryLatent(config, parent_gene_index)
    elif model_family == "supervised_factor":
        if not isinstance(config, SupervisedFactorConfig):
            raise TypeError("supervised_factor requires SupervisedFactorConfig")
        if parent_gene_index is None:
            raise ValueError("supervised_factor requires parent_gene_index")
        model = SupervisedFactorProteinTranslator(config, parent_gene_index)
    else:
        raise ValueError(f"unsupported model family: {model_family}")
    model = model.to(runtime.device)
    model.fix_projection_matrices_()
    if runtime.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank] if runtime.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    torch.manual_seed(seed + runtime.rank + 1)
    return model


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def fit_model_anchor(
    model_family: str,
    arrays: ProteinArrays,
    fit_indices: np.ndarray,
    parent_gene_index: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fit optional model state from one declared training partition only."""

    if model_family not in {"residualized_lowrank", "moe_lowrank"}:
        return None, None
    anchor = fit_cognate_linear_anchor(
        arrays.rna,
        arrays.protein,
        arrays.protein_mask,
        parent_gene_index,
        fit_indices,
    )
    return anchor.intercept, anchor.slope


def make_train_loader(
    arrays: ProteinArrays,
    indices: np.ndarray,
    runtime: Runtime,
    args: argparse.Namespace,
    epoch_seed: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = ProteinTrainingDataset(arrays, indices)
    sampler = None
    if runtime.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=True,
            seed=epoch_seed,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=runtime.device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    return loader, sampler


def make_eval_loader(
    arrays: ProteinArrays,
    indices: np.ndarray,
    args: argparse.Namespace,
    runtime: Runtime,
) -> DataLoader:
    return DataLoader(
        ProteinTrainingDataset(arrays, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=runtime.device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
        if name != "sample_index"
    }


def autocast_context(runtime: Runtime, precision: str):
    if runtime.device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type=runtime.device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    core = unwrap(model)
    encoder_parameters = list(core.rna_encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    translator_parameters = [
        parameter for parameter in core.parameters() if id(parameter) not in encoder_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": translator_parameters, "lr": args.translator_lr},
        ],
        weight_decay=args.weight_decay,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def factor(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-3)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def correlation_warmup_scale(
    update_step: int,
    total_update_steps: int,
    warmup_fraction: float,
) -> float:
    """Return the correlation-loss scale for the current optimizer update."""

    if update_step < 0:
        raise ValueError("update_step must be non-negative")
    if total_update_steps <= 0:
        raise ValueError("total_update_steps must be positive")
    if not 0.0 <= warmup_fraction <= 1.0:
        raise ValueError("warmup_fraction must be in [0, 1]")
    if warmup_fraction == 0.0:
        return 1.0
    warmup_steps = max(1, math.ceil(total_update_steps * warmup_fraction))
    return min((update_step + 1) / warmup_steps, 1.0)


def training_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    *,
    update_step: int,
    total_update_steps: int,
) -> dict[str, torch.Tensor]:
    """Dispatch to the configured masked protein training objective."""

    if args.loss_mode == "mse":
        mse = masked_mse_loss(prediction, target, mask)
        return {"loss": mse, "protein_mse": mse}

    if args.loss_mode == "legacy":
        return protein_translation_loss(
            prediction,
            target,
            mask,
            rank_weight=args.rank_weight,
            rank_temperature=args.rank_temperature,
            rank_max_targets=args.rank_max_targets,
        )
    if args.loss_mode != "correlation":
        raise ValueError(f"unsupported loss mode: {args.loss_mode}")

    scale = correlation_warmup_scale(
        update_step,
        total_update_steps,
        args.correlation_warmup_fraction,
    )
    losses = correlation_guided_loss(
        prediction,
        target,
        mask,
        value_kind="huber",
        value_reduction=args.value_reduction,
        value_weight=1.0,
        pearson_weight=args.pearson_weight * scale,
        spearman_weight=args.spearman_weight * scale,
        min_observations=args.correlation_min_observations,
        pearson_max_targets=args.pearson_max_targets,
        spearman_max_targets=args.rank_max_targets,
        gather_distributed=args.gather_correlation_across_ranks,
    )
    losses["correlation_scale"] = prediction.new_tensor(scale)
    return losses


@torch.no_grad()
def validation_metrics(
    model: nn.Module,
    arrays: ProteinArrays,
    indices: np.ndarray,
    runtime: Runtime,
    args: argparse.Namespace,
) -> dict[str, float]:
    core = unwrap(model)
    core.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_observed = 0.0
    prediction_parts: list[np.ndarray] = []
    observed_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    loader = make_eval_loader(arrays, indices, args, runtime)
    for raw_batch in loader:
        batch = move_batch(raw_batch, runtime.device)
        with autocast_context(runtime, args.precision):
            prediction = core(batch["rna"])["protein"]
            loss = masked_huber(prediction, batch["protein"], batch["protein_mask"])
            mse = masked_mse_loss(
                prediction, batch["protein"], batch["protein_mask"]
            )
        observed = float(batch["protein_mask"].sum().cpu())
        total_loss += float(loss.float().cpu()) * observed
        total_mse += float(mse.float().cpu()) * observed
        total_observed += observed
        prediction_parts.append(prediction.float().cpu().numpy())
        observed_parts.append(batch["protein"].float().cpu().numpy())
        mask_parts.append(batch["protein_mask"].bool().cpu().numpy())

    prediction_array = np.vstack(prediction_parts)
    observed_array = np.vstack(observed_parts)
    mask_array = np.vstack(mask_parts)
    # Prediction and observed labels already share the configured target scale.
    # Validation therefore evaluates the model output directly.
    batch_cosine = batchwise_flattened_cosine(
        observed_array,
        prediction_array,
        mask=mask_array,
        batch_size=args.batch_size,
    )
    mean_batch_cosine = float(batch_cosine["cosine_similarity"].mean())
    median_spearman = float("nan")
    if args.selection_metric != "batch_cosine":
        protein_names = [str(index) for index in range(prediction_array.shape[1])]
        per_protein = per_protein_spearman(
            observed_array,
            prediction_array,
            protein_names,
            mask=mask_array,
            min_samples=10,
        )
        median_spearman = float(summarize_spearman(per_protein)["median_spearman"])
    return {
        "huber": total_loss / max(total_observed, 1.0),
        "mse": total_mse / max(total_observed, 1.0),
        "mean_batch_cosine": mean_batch_cosine,
        "n_validation_batches": int(len(batch_cosine)),
        "median_spearman": median_spearman,
    }


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in unwrap(model).state_dict().items()
    }


def train_model(
    model: nn.Module,
    arrays: ProteinArrays,
    train_indices: np.ndarray,
    val_indices: np.ndarray | None,
    runtime: Runtime,
    args: argparse.Namespace,
    *,
    epochs: int,
    early_stop: bool,
) -> tuple[int, list[dict[str, Any]]]:
    loader, sampler = make_train_loader(
        arrays, train_indices, runtime, args, args.seed
    )
    optimizer = make_optimizer(model, args)
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    total_update_steps = max(updates_per_epoch * epochs, 1)
    scheduler = make_scheduler(
        optimizer, total_update_steps, args.warmup_fraction
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=runtime.device.type == "cuda" and args.precision == "fp16"
    )
    best_epoch = 0
    maximize_selection = args.selection_metric in {"spearman", "batch_cosine"}
    best_value = float("-inf") if maximize_selection else float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    global_update_step = 0

    for epoch in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        n_microbatches = 0
        for step, raw_batch in enumerate(loader):
            batch = move_batch(raw_batch, runtime.device)
            window_start = (step // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(
                args.gradient_accumulation, len(loader) - window_start
            )
            synchronize = (
                (step + 1) % args.gradient_accumulation == 0
                or step + 1 == len(loader)
            )
            sync_context = nullcontext()
            if isinstance(model, DistributedDataParallel) and not synchronize:
                sync_context = model.no_sync()
            with sync_context:
                with autocast_context(runtime, args.precision):
                    prediction = model(batch["rna"])["protein"]
                    losses = training_objective(
                        prediction,
                        batch["protein"],
                        batch["protein_mask"],
                        args,
                        update_step=global_update_step,
                        total_update_steps=total_update_steps,
                    )
                    scaled_loss = losses["loss"] / window_size
                scaler.scale(scaled_loss).backward()
            epoch_loss += float(losses["loss"].detach().float().cpu())
            n_microbatches += 1
            if synchronize:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_update_step += 1

        train_value = epoch_loss / max(n_microbatches, 1)
        validation_value = float("nan")
        validation_mse = float("nan")
        validation_spearman = float("nan")
        validation_batch_cosine = float("nan")
        if val_indices is not None:
            validation_result = validation_metrics(
                model, arrays, val_indices, runtime, args
            )
            validation_value = validation_result["huber"]
            validation_mse = validation_result["mse"]
            validation_spearman = validation_result["median_spearman"]
            validation_batch_cosine = validation_result["mean_batch_cosine"]
        if val_indices is None:
            monitor = train_value
        elif args.selection_metric == "spearman":
            monitor = validation_spearman
        elif args.selection_metric == "batch_cosine":
            monitor = validation_batch_cosine
        else:
            monitor = validation_value
        record = {
            "epoch": epoch,
            "train_loss": train_value,
            "validation_huber": validation_value,
            "validation_mse": validation_mse,
            "validation_median_spearman": validation_spearman,
            "validation_mean_batch_cosine": validation_batch_cosine,
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "translator_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(record)
        if runtime.primary:
            print(json.dumps(record), flush=True)

        if not early_stop:
            best_epoch = epoch
            continue
        improved = (
            monitor > best_value + 1e-6
            if maximize_selection
            else monitor < best_value - 1e-6
        )
        if improved:
            best_value = monitor
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            stale = 0
        else:
            stale += 1
        if epoch >= args.min_epochs and stale >= args.patience:
            break

    if early_stop:
        if best_state is None:
            raise RuntimeError("training produced no selectable checkpoint")
        unwrap(model).load_state_dict(best_state)
        del best_state
    barrier(runtime)
    return best_epoch, history


@torch.no_grad()
def predict(
    model: nn.Module,
    arrays: ProteinArrays,
    indices: np.ndarray,
    runtime: Runtime,
    args: argparse.Namespace,
) -> np.ndarray:
    if not runtime.primary:
        return np.empty((0, 0), dtype=np.float32)
    core = unwrap(model)
    core.eval()
    parts: list[np.ndarray] = []
    for raw_batch in make_eval_loader(arrays, indices, args, runtime):
        batch = move_batch(raw_batch, runtime.device)
        with autocast_context(runtime, args.precision):
            output = core(batch["rna"])["protein"]
        parts.append(output.float().cpu().numpy())
    return np.vstack(parts)


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: ProteinModelConfig | QueryLatentConfig | SupervisedFactorConfig,
    arrays: ProteinArrays,
    inputs: dict[str, Any],
    args: argparse.Namespace,
    selected_epochs: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    core = unwrap(model)
    parent_gene_index = getattr(core, "parent_gene_index", None)
    anchor_intercept = getattr(core, "anchor_intercept", None)
    anchor_slope = getattr(core, "anchor_slope", None)
    torch.save(
        {
            "schema_version": 1,
            "model_class": type(core).__name__,
            "model_family": args.model_family,
            "model_state": cpu_state_dict(model),
            "model_config": config.to_dict(),
            "gene_names": inputs["gene_names"],
            "protein_names": inputs["protein_names"],
            "parent_gene_index": (
                parent_gene_index.detach().cpu().numpy()
                if parent_gene_index is not None
                else None
            ),
            "anchor_intercept": (
                anchor_intercept.detach().cpu().numpy()
                if anchor_intercept is not None
                else None
            ),
            "anchor_slope": (
                anchor_slope.detach().cpu().numpy()
                if anchor_slope is not None
                else None
            ),
            "rna_scaler": {
                "feature_names": arrays.rna_scaler.feature_names,
                "mean": arrays.rna_scaler.mean,
                "scale": arrays.rna_scaler.scale,
            },
            "protein_target_scaler": arrays.protein_scaler.state_dict(),
            "target_scale": arrays.protein_scaler.mode,
            "fold_index": args.fold_index,
            "selected_epochs": selected_epochs,
            "training_args": jsonable_args(args),
            "architecture_contract": {
                "protein_is_model_input": False,
                "protein_mask_is_decoder_input": False,
                "complete_fixed_protein_axis": True,
                "phosphosite_head_present": False,
                "split_local_frozen_cognate_anchor": bool(
                    anchor_intercept is not None and anchor_slope is not None
                ),
            },
        },
        path,
    )


def write_table(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, sep="\t", index=False)


def main() -> int:
    args = parse_args()
    runtime = setup_runtime()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        inputs = load_inputs(args)
        manifest = inputs["manifest"]
        folds = make_strict_folds(
            inputs["sample_ids"],
            manifest[args.strata_column].astype(str).to_numpy(),
            n_splits=args.n_folds,
            random_state=args.seed,
            blocking_groups=manifest[args.group_column].astype(str).to_numpy(),
        )
        if args.fold_index < 0 or args.fold_index >= len(folds):
            raise ValueError("fold-index is outside the available folds")
        outer = folds[args.fold_index]
        if runtime.primary:
            save_fold_indices(
                args.output_dir / "fold_indices.npz",
                folds,
                inputs["sample_ids"],
                manifest[args.strata_column].astype(str).to_numpy(),
                manifest[args.group_column].astype(str).to_numpy(),
            )
        barrier(runtime)

        inner_manifest = manifest.iloc[outer.train_idx]
        inner_group_column = args.inner_group_column or args.group_column
        if inner_group_column not in inner_manifest.columns:
            raise ValueError(
                f"manifest lacks inner split group column: {inner_group_column}"
            )
        inner = make_strict_folds(
            inner_manifest.index.to_numpy(),
            inner_manifest[args.strata_column].astype(str).to_numpy(),
            n_splits=args.inner_folds,
            random_state=args.seed + 1009,
            blocking_groups=inner_manifest[inner_group_column].astype(str).to_numpy(),
        )[0]
        selection_train = outer.train_idx[inner.train_idx]
        selection_val = outer.train_idx[inner.test_idx]
        protein_coverage_audit = None
        if args.min_protein_coverage is not None:
            inputs, protein_coverage_audit = select_proteins_by_training_coverage(
                inputs,
                selection_train,
                args.min_protein_coverage,
            )
        config = make_config(args, inputs)
        parent_gene_index = build_parent_gene_index(inputs)

        selection_arrays = prepare_arrays(
            inputs,
            selection_train,
            args.study_column,
            target_scale=args.target_scale,
        )
        selection_anchor_intercept, selection_anchor_slope = fit_model_anchor(
            args.model_family,
            selection_arrays,
            selection_train,
            parent_gene_index,
        )
        selection_model = construct_model(
            config,
            runtime,
            args.seed + 17,
            model_family=args.model_family,
            parent_gene_index=parent_gene_index,
            lowrank_rank=args.lowrank_rank,
            lowrank_ranks=tuple(args.lowrank_ranks),
            initial_cognate_gate=args.initial_cognate_gate,
            anchor_intercept=selection_anchor_intercept,
            anchor_slope=selection_anchor_slope,
            moe_experts=args.moe_experts,
            moe_gate_hidden=args.moe_gate_hidden,
        )
        selected_epochs, selection_history = train_model(
            selection_model,
            selection_arrays,
            selection_train,
            selection_val,
            runtime,
            args,
            epochs=args.max_epochs,
            early_stop=True,
        )

        if args.screen_only:
            validation_prediction = predict(
                selection_model,
                selection_arrays,
                selection_val,
                runtime,
                args,
            )
            barrier(runtime)
            if runtime.primary:
                output_dir = args.output_dir
                table_dir = output_dir / "tables"
                model_dir = output_dir / "models"
                table_dir.mkdir(parents=True, exist_ok=True)
                validation_observed = np.where(
                    selection_arrays.protein_mask[selection_val],
                    selection_arrays.protein[selection_val],
                    np.nan,
                )
                validation_mask = selection_arrays.protein_mask[selection_val]
                batch_cosine = batchwise_flattened_cosine(
                    validation_observed,
                    validation_prediction,
                    mask=validation_mask,
                    batch_size=args.batch_size,
                )
                batch_cosine.to_csv(
                    table_dir / "inner_validation_batch_cosine.tsv",
                    sep="\t",
                    index=False,
                )
                pd.DataFrame(selection_history).to_csv(
                    table_dir / "selection_training_history.tsv",
                    sep="\t",
                    index=False,
                )
                if protein_coverage_audit is not None:
                    protein_coverage_audit.to_csv(
                        table_dir / "selection_train_protein_coverage.tsv",
                        sep="\t",
                        index=False,
                    )
                per_protein = per_protein_spearman(
                    validation_observed,
                    validation_prediction,
                    inputs["protein_names"],
                    mask=validation_mask,
                    min_samples=10,
                )
                selection_train_coverage = selection_arrays.protein_mask[
                    selection_train
                ].sum(axis=0, dtype=np.int64)
                per_protein.insert(
                    2,
                    "selection_train_observed",
                    selection_train_coverage,
                )
                per_protein.to_csv(
                    table_dir / "inner_validation_per_protein_spearman.tsv",
                    sep="\t",
                    index=False,
                )
                summary = summarize_spearman(per_protein)
                high_coverage_count = min(1000, len(per_protein))
                high_coverage_indices = np.argsort(
                    -selection_train_coverage,
                    kind="stable",
                )[:high_coverage_count]
                high_coverage_spearman = pd.to_numeric(
                    per_protein.iloc[high_coverage_indices]["spearman"],
                    errors="coerce",
                ).to_numpy(dtype=float)
                high_coverage_finite = np.isfinite(high_coverage_spearman)
                summary.update(
                    {
                        "high_coverage_protein_count": int(high_coverage_count),
                        "high_coverage_evaluable_proteins": int(
                            high_coverage_finite.sum()
                        ),
                        "high_coverage_median_spearman": (
                            float(np.median(high_coverage_spearman[high_coverage_finite]))
                            if high_coverage_finite.any()
                            else float("nan")
                        ),
                    }
                )
                summary.update(
                    {
                        "mode": "screen_only",
                        "model_family": args.model_family,
                        "fold": args.fold_index,
                        "selected_epochs": int(selected_epochs),
                        "n_selection_train": int(selection_train.size),
                        "n_selection_validation": int(selection_val.size),
                        "min_protein_coverage": args.min_protein_coverage,
                        "n_proteins": len(inputs["protein_names"]),
                        "parameter_count": int(unwrap(selection_model).parameter_count()),
                        "n_cognate_mapped": int(np.sum(parent_gene_index >= 0)),
                        "runtime_seconds": time.time() - started,
                        "outer_test_evaluated": False,
                        "target_scale": args.target_scale,
                        "validation_mse": array_masked_mse(
                            validation_observed,
                            validation_prediction,
                            mask=validation_mask,
                        ),
                        "validation_mean_batch_cosine": float(
                            batch_cosine["cosine_similarity"].mean()
                        ),
                        "n_validation_batches": int(len(batch_cosine)),
                        "validation_prediction_postprocessing": "none",
                        "checkpoint_selection_metric": args.selection_metric,
                        "performance_audit_metric": "median_per_protein_spearman",
                        "performance_target": 0.7,
                        "inner_group_column": inner_group_column,
                    }
                )
                save_checkpoint(
                    model_dir / f"{args.model_family}_screen_fold{args.fold_index}.pt",
                    selection_model,
                    config,
                    selection_arrays,
                    inputs,
                    args,
                    selected_epochs,
                )
                (output_dir / "done.txt").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (output_dir / "run_status.txt").write_text(
                    "completed\n", encoding="utf-8"
                )
                print(json.dumps(summary, ensure_ascii=False), flush=True)
            barrier(runtime)
            return 0

        if args.refit_full_outer:
            del selection_model, selection_arrays
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()
            barrier(runtime)
            arrays = prepare_arrays(
                inputs,
                outer.train_idx,
                args.study_column,
                target_scale=args.target_scale,
            )
            refit_anchor_intercept, refit_anchor_slope = fit_model_anchor(
                args.model_family,
                arrays,
                outer.train_idx,
                parent_gene_index,
            )
            model = construct_model(
                config,
                runtime,
                args.seed + 31,
                model_family=args.model_family,
                parent_gene_index=parent_gene_index,
                lowrank_rank=args.lowrank_rank,
                lowrank_ranks=tuple(args.lowrank_ranks),
                initial_cognate_gate=args.initial_cognate_gate,
                anchor_intercept=refit_anchor_intercept,
                anchor_slope=refit_anchor_slope,
                moe_experts=args.moe_experts,
                moe_gate_hidden=args.moe_gate_hidden,
            )
            _, refit_history = train_model(
                model,
                arrays,
                outer.train_idx,
                None,
                runtime,
                args,
                epochs=selected_epochs,
                early_stop=False,
            )
        else:
            arrays = selection_arrays
            model = selection_model
            refit_history = []

        prediction = predict(model, arrays, outer.test_idx, runtime, args)
        barrier(runtime)
        if runtime.primary:
            output_dir = args.output_dir
            prediction_dir = output_dir / "predictions"
            table_dir = output_dir / "tables"
            model_dir = output_dir / "models"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            table_dir.mkdir(parents=True, exist_ok=True)
            sample_ids = np.asarray(inputs["sample_ids"])[outer.test_idx]
            observed = np.where(
                arrays.protein_mask[outer.test_idx],
                arrays.protein[outer.test_idx],
                np.nan,
            )
            prediction_frame = pd.DataFrame(
                prediction, index=sample_ids, columns=inputs["protein_names"]
            )
            observed_frame = pd.DataFrame(
                observed, index=sample_ids, columns=inputs["protein_names"]
            )
            prediction_frame.to_parquet(
                prediction_dir / f"fold{args.fold_index}_protein_prediction.parquet"
            )
            observed_frame.to_parquet(
                prediction_dir / f"fold{args.fold_index}_protein_observed.parquet"
            )
            per_protein = per_protein_spearman(
                observed,
                prediction,
                inputs["protein_names"],
                mask=arrays.protein_mask[outer.test_idx],
                min_samples=10,
            )
            per_protein.to_csv(
                table_dir / f"fold{args.fold_index}_per_protein_spearman.tsv",
                sep="\t",
                index=False,
            )
            pd.DataFrame(selection_history).to_csv(
                table_dir / "selection_training_history.tsv", sep="\t", index=False
            )
            pd.DataFrame(refit_history).to_csv(
                table_dir / "refit_training_history.tsv", sep="\t", index=False
            )
            summary = summarize_spearman(per_protein)
            summary.update(
                {
                    "fold": args.fold_index,
                    "n_train_outer": int(outer.train_idx.size),
                    "n_test_outer": int(outer.test_idx.size),
                    "n_selection_train": int(selection_train.size),
                    "n_selection_validation": int(selection_val.size),
                    "selected_epochs": int(selected_epochs),
                    "refit_full_outer": bool(args.refit_full_outer),
                    "n_genes": len(inputs["gene_names"]),
                    "n_proteins": len(inputs["protein_names"]),
                    "model_family": args.model_family,
                    "translator_hidden": int(
                        getattr(
                            getattr(unwrap(model), "cross_modal_translator", None),
                            "hidden",
                            0,
                        )
                    ),
                    "lowrank_rank": int(
                        getattr(
                            getattr(unwrap(model), "global_translator", None),
                            "rank",
                            0,
                        )
                    ),
                    "n_state_tokens": int(
                        getattr(getattr(unwrap(model), "config", None), "n_state_tokens", 0)
                    ),
                    "n_protein_factors": int(
                        getattr(getattr(unwrap(model), "config", None), "n_factors", 0)
                    ),
                    "n_cognate_mapped": int(np.sum(parent_gene_index >= 0)),
                    "parameter_count": int(unwrap(model).parameter_count()),
                    "runtime_seconds": time.time() - started,
                    "target_scale": args.target_scale,
                    "inner_group_column": inner_group_column,
                }
            )
            with (table_dir / "protein_fold_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
            save_checkpoint(
                model_dir / f"rna_protein_performer_fold{args.fold_index}.pt",
                model,
                config,
                arrays,
                inputs,
                args,
                selected_epochs,
            )
            (output_dir / "done.txt").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (output_dir / "run_status.txt").write_text("completed\n", encoding="utf-8")
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        barrier(runtime)
        return 0
    except Exception as error:
        if runtime.primary:
            (args.output_dir / "fatal.log").write_text(
                f"{type(error).__name__}: {error}\n", encoding="utf-8"
            )
            (args.output_dir / "run_status.txt").write_text("failed\n", encoding="utf-8")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
