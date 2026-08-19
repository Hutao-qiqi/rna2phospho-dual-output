"""Train the latent-transformer phosphosite operator on the locked split."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "base_snapshot" / "code"
for path in (HERE, BASE):
    sys.path.insert(0, str(path))

from axial_dynamic_data import (  # noqa: E402
    apply_feature_zscore,
    fit_feature_zscore,
    per_site_metrics,
    read_parquet_rows,
    read_prediction_matrix,
    sample_rank_encode,
    sha256_file,
    validate_case_split_disjointness,
    validate_protein_prediction_provenance,
    validate_split_manifest,
    write_json,
)
from cancer_training import (  # noqa: E402
    canonical_cancer_labels,
    fit_cancer_vocabulary,
)
from chunk_manifest import build_site_chunks  # noqa: E402
from cophee_prior_bundle import load_cophee_prior_bundle  # noqa: E402
from decoder_retrieval_model import (  # noqa: E402
    CONFIGURATIONS,
    DecoderRetrievalConfig,
    DecoderRetrievalModel,
)
from experiment_contract import (  # noqa: E402
    assert_development_only_phosphosite_rows,
    padded_kinase_mapping,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "rna",
        "protein-prediction",
        "protein-reliability",
        "protein-provenance",
        "phosphosite",
        "phosphosite-manifest",
        "split-manifest",
        "sample-metadata",
        "cophee-prior-bundle",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--esm2-prior-npz",
        type=Path,
        help="冻结 ESM-2 全蛋白及逐残基序列先验包",
    )
    parser.add_argument(
        "--esm3-prior-npz",
        type=Path,
        help="冻结 ESM-3 逐残基序列先验包",
    )
    parser.add_argument(
        "--esm2-fusion",
        choices=(
            "residue_attention",
            "residue_residual",
            "center_gated",
            "local_window_attention",
            "local_window_pair",
        ),
        default="residue_attention",
    )
    parser.add_argument(
        "--configuration",
        choices=tuple(CONFIGURATIONS),
        default="neural_operator",
    )
    parser.add_argument(
        "--training-stage",
        choices=("pan", "cancer_adapter", "external_full_finetune"),
        default="pan",
    )
    parser.add_argument("--pan-checkpoint", type=Path)
    parser.add_argument(
        "--protein-input-mode",
        choices=("crossfit_prediction", "trainfit_prediction", "measured_oracle"),
        required=True,
    )
    parser.add_argument("--expected-train-samples", type=int, default=916)
    parser.add_argument("--expected-validation-samples", type=int, default=229)
    parser.add_argument("--expected-sealed-samples", type=int, default=286)
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--case-id-column", default="case_submitter_id")
    parser.add_argument("--cancer-column", default="cancer_label")
    parser.add_argument("--study-column", default="study")
    parser.add_argument(
        "--target-study-residual",
        action="store_true",
        help=(
            "将磷酸化目标改为训练集拟合的研究×位点中心残差"
            "R = Y - mu_{study,site}，中心只由训练患者拟合，"
            "低覆盖位点向全局位点均值收缩；验证预测重建时加回中心。"
        ),
    )
    parser.add_argument(
        "--study-residual-tau",
        type=float,
        default=0.0,
        help=(
            "研究×位点中心收缩强度：w = n_obs/(n_obs+tau)，"
            "tau=0 表示不收缩（直接用研究均值），"
            "tau>0 时低观测数位点向（全局位点中心+量纲组整体偏移）收缩。"
        ),
    )
    parser.add_argument(
        "--study-residual-min-overlap",
        type=int,
        default=5,
        help=(
            "估计量纲组整体偏移 b_g 时要求的高覆盖共同位点最小训练观测数。"
            "只有组内和全局训练观测数都不低于该阈值的位点才进入 b_g 的中位数估计。"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--early-stopping-metric",
        choices=("site_spearman", "profile_spearman", "balanced_spearman", "cosine"),
        default="site_spearman",
    )
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument(
        "--drop-last-training-batch",
        action="store_true",
        help=(
            "每轮省略不足sample_batch_size的尾批；患者顺序每轮重排，"
            "用于避免极小尾批形成一次等权优化器更新。"
        ),
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--site-chunk-size", type=int, default=1000)
    parser.add_argument("--maximum-kinases-per-site", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--operator-rank", type=int, default=128)
    parser.add_argument("--modality-tokens", type=int, default=32)
    parser.add_argument("--phospho-latent-tokens", type=int, default=16)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--esm2-warmup-updates", type=int, default=0)
    parser.add_argument("--esm2-warmup-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--esm3-warmup-updates", type=int, default=0)
    parser.add_argument("--esm3-warmup-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--joint-learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--site-huber-weight", type=float, default=1.0)
    parser.add_argument("--site-huber-delta", type=float, default=1.0)
    parser.add_argument("--site-pearson-weight", type=float, default=0.5)
    parser.add_argument("--site-rank-weight", type=float, default=0.0)
    parser.add_argument(
        "--profile-mse-weight",
        type=float,
        default=0.0,
        help=(
            "患者等权、全观测位点masked MSE权重。该受控目标要求"
            "identity_logscale标签、profile_spearman选模，并关闭其余预测损失。"
        ),
    )
    parser.add_argument(
        "--profile-pearson-weight",
        type=float,
        default=0.0,
        help=(
            "患者内全观测位点Pearson损失权重。该项跨全部位点块累计，"
            "不会在单个位点块内分别计算。"
        ),
    )
    parser.add_argument(
        "--profile-pearson-minimum-observations",
        type=int,
        default=2,
        help="患者进入患者内Pearson损失所需的最少有效位点数。",
    )
    parser.add_argument("--rank-temperature", type=float, default=0.2)
    parser.add_argument("--rank-minimum-target-difference", type=float, default=1.0e-4)
    parser.add_argument("--rank-pair-offsets", default="1,3,7,13,29")
    parser.add_argument("--correlation-min-observations", type=int, default=16)
    parser.add_argument("--high-coverage-threshold", type=float, default=0.70)
    parser.add_argument("--high-coverage-panel-size", type=int, default=1000)
    parser.add_argument("--cancer-adapter-rank", type=int, default=8)
    parser.add_argument("--adapter-distill-weight", type=float, default=0.1)
    parser.add_argument("--adapter-parameter-weight", type=float, default=1.0e-4)
    parser.add_argument("--capacity-preflight-batches", type=int, default=0)
    parser.add_argument("--input-hash-cache", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--conservative-pretrain-h5", type=Path)
    parser.add_argument("--conservative-pretrain-protein-npz", type=Path)
    parser.add_argument("--development-protein-npz", type=Path)
    parser.add_argument(
        "--pretrain-scope",
        choices=("cptac916", "combined2067"),
        default="combined2067",
    )
    parser.add_argument("--pretrain-updates", type=int, default=0)
    parser.add_argument("--pretrain-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--maximum-finetune-updates", type=int, default=0)
    parser.add_argument("--validation-interval-updates", type=int, default=0)
    parser.add_argument(
        "--validation-only-at-finetune-end",
        action="store_true",
        help=(
            "跳过训练中间验证，仅在达到maximum_finetune_updates后评估一次；"
            "用于固定外部测试集的单次终点评价。"
        ),
    )
    parser.add_argument(
        "--rna-input-transform",
        choices=("sample_rank", "feature_zscore"),
        default="sample_rank",
        help=(
            "RNA输入变换。统一参考分位训练使用 feature_zscore，"
            "中心和尺度仅由训练患者拟合。"
        ),
    )
    parser.add_argument(
        "--site-scaling",
        choices=(
            "minmax",
            "quantile_01_99",
            "sample_percentile_rank",
            "identity_logscale",
            "identity_reference_quantile",
        ),
        default="minmax",
    )
    parser.add_argument("--validate-inputs-only", action="store_true")
    parser.add_argument("--export-initial-checkpoint-only", action="store_true")
    parser.add_argument(
        "--allow-initial-rna-scaler-mismatch",
        action="store_true",
        help="内层训练集交叉验证允许使用外层训练集RNA标度加载初始主干；不允许验证标签参与标度拟合。",
    )
    return parser.parse_args()


def configure_esm2_warmup(
    model: DecoderRetrievalModel,
    base_trainable_names: set[str],
    enabled: bool,
    parameter_prefix: str = "esm2_",
) -> list[torch.nn.Parameter]:
    """Select one sequence-prior branch during an isolated warm-up phase."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name in base_trainable_names
            and (not enabled or name.startswith(parameter_prefix))
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("sequence-prior warm-up selected no trainable parameters")
    return trainable


def load_pan_trunk_for_cancer_adapter(
    model: DecoderRetrievalModel,
    checkpoint_state: dict[str, torch.Tensor],
) -> dict[str, int]:
    """Load every pan-cancer parameter while keeping a fresh cancer adapter.

    Pan training never enables the cancer branch, although its zero-output
    parameters are present in older checkpoints.  Discarding those tensors
    lets the adapter rank be selected independently without changing a single
    pan-cancer prediction.
    """
    trunk_state = {
        name: value
        for name, value in checkpoint_state.items()
        if not name.startswith("cancer_")
    }
    incompatible = model.load_state_dict(trunk_state, strict=False)
    invalid_missing = [
        name for name in incompatible.missing_keys if not name.startswith("cancer_")
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise ValueError(
            "pan checkpoint differs outside the cancer adapter: "
            f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )
    if model.cancer_identity is None:
        raise ValueError("cancer adapter stage requires at least one cancer")
    if not torch.equal(
        model.cancer_identity.weight.detach(),
        torch.zeros_like(model.cancer_identity.weight),
    ):
        raise RuntimeError("fresh cancer identity must start at zero")
    if not torch.equal(
        model.cancer_adapter_up.detach(),
        torch.zeros_like(model.cancer_adapter_up),
    ):
        raise RuntimeError("fresh cancer adapter output must start at zero")
    return {
        "pan_tensors_loaded": len(trunk_state),
        "cancer_tensors_reinitialized": len(incompatible.missing_keys),
    }


PAN_CHECKPOINT_STAGES = {"cancer_adapter", "external_full_finetune"}


def configure_posttraining_parameters(
    model: DecoderRetrievalModel,
    training_stage: str,
) -> set[str]:
    """Select the disjoint parameter sets for the two post-training controls."""
    if training_stage not in PAN_CHECKPOINT_STAGES:
        raise ValueError(f"unsupported post-training stage: {training_stage}")
    for name, parameter in model.named_parameters():
        if training_stage == "cancer_adapter":
            parameter.requires_grad_(name.startswith("cancer_"))
        else:
            parameter.requires_grad_(not name.startswith("cancer_"))
    return {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }


OUTPUT_CALIBRATION_STATE_NAMES = {
    "site_bias.weight",
    "site_output_scale",
    "site_output_shift",
}

OUTPUT_COORDINATE_PREFIXES = (
    "local_output.",
    "separated_local_output.",
)
OUTPUT_COORDINATE_EXACT_NAMES = {
    "parent_direct_coefficients.weight",
}


def prepare_initial_checkpoint_state(
    checkpoint_state: dict[str, torch.Tensor],
    *,
    target_coordinate_changed: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Prepare a checkpoint for trunk reuse under the requested target coordinate."""
    migrated = dict(checkpoint_state)
    removed_sequence_tensors: list[str] = []
    for name in tuple(migrated):
        if name.startswith("site_esm2_") or name.startswith("site_esm3_"):
            removed_sequence_tensors.append(name)
            migrated.pop(name)
    reset_calibration: list[str] = []
    reset_readout: list[str] = []
    if target_coordinate_changed:
        for name in sorted(OUTPUT_CALIBRATION_STATE_NAMES):
            if name in migrated:
                migrated.pop(name)
            reset_calibration.append(name)
        for name in sorted(tuple(migrated)):
            if name in OUTPUT_COORDINATE_EXACT_NAMES or any(
                name.startswith(prefix) for prefix in OUTPUT_COORDINATE_PREFIXES
            ):
                migrated.pop(name)
                reset_readout.append(name)
    return migrated, {
        "target_coordinate_changed": target_coordinate_changed,
        "old_site_coordinate_conversion_applied": False,
        "reset_output_calibration": reset_calibration,
        "reset_output_readout": reset_readout,
        "removed_frozen_sequence_tensors": sorted(removed_sequence_tensors),
    }


def detach_trainable_context(
    context: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Detach encoder outputs so each site chunk can release its local graph."""
    detached: dict[str, torch.Tensor] = {}
    leaves: dict[str, torch.Tensor] = {}
    for name, value in context.items():
        if value.requires_grad:
            leaf = value.detach().requires_grad_(True)
            detached[name] = leaf
            leaves[name] = leaf
        else:
            detached[name] = value
    return detached, leaves


def accumulate_context_gradients(
    total: dict[str, torch.Tensor], leaves: dict[str, torch.Tensor]
) -> None:
    for name, leaf in leaves.items():
        if leaf.grad is None:
            continue
        value = leaf.grad.detach()
        total[name] = value if name not in total else total[name] + value


def backward_encoder_context(
    context: dict[str, torch.Tensor], gradients: dict[str, torch.Tensor]
) -> None:
    names = [name for name in gradients if context[name].requires_grad]
    if names:
        torch.autograd.backward(
            [context[name] for name in names],
            [gradients[name] for name in names],
        )


def capture_torch_rng_state(device: torch.device) -> dict[str, torch.Tensor | None]:
    """Capture CPU and active-device RNG state for an exact decoder replay."""
    return {
        "cpu": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_torch_rng_state(
    state: dict[str, torch.Tensor | None], device: torch.device
) -> None:
    """Restore a state returned by :func:`capture_torch_rng_state`."""
    cpu_state = state["cpu"]
    if cpu_state is None:
        raise ValueError("captured CPU RNG state is missing")
    torch.random.set_rng_state(cpu_state)
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def nonfinite_gradient_names(model: torch.nn.Module) -> list[str]:
    """Return parameter names whose accumulated gradient is not finite."""
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    ]


def target_metadata(manifest: pd.DataFrame) -> dict[str, object]:
    target_column = next(
        (
            name
            for name in ("scp682_site_id", "gene_site", "gene_site_id")
            if name in manifest
        ),
        None,
    )
    parent_column = next(
        (
            name
            for name in ("parent_gene", "total_protein_gene")
            if name in manifest
        ),
        None,
    )
    if target_column is None or parent_column is None:
        raise ValueError("phosphosite manifest lacks target or parent columns")
    targets = manifest[target_column].astype(str).tolist()
    parents = manifest[parent_column].astype(str).str.upper().tolist()
    if len(targets) != len(set(targets)):
        raise ValueError("phosphosite target vocabulary contains duplicates")
    if any(not value.strip() for value in targets + parents):
        raise ValueError("phosphosite target metadata contains empty values")

    if "residue" in manifest:
        residue_text = manifest["residue"].astype(str).str.upper()
    else:
        residue_text = pd.Series(
            [value.split("|")[-1][:1].upper() for value in targets]
        )
    residue_lookup = {"S": 0, "T": 1, "Y": 2}
    residue_index = np.asarray(
        [residue_lookup.get(value, 3) for value in residue_text], dtype=np.int64
    )
    if "position" in manifest:
        position = pd.to_numeric(manifest["position"], errors="coerce").fillna(0)
    else:
        position = pd.Series(
            [
                "".join(character for character in value.split("|")[-1] if character.isdigit())
                for value in targets
            ]
        )
        position = pd.to_numeric(position, errors="coerce").fillna(0)
    return {
        "targets": targets,
        "parents": parents,
        "residue_index": residue_index,
        "position": position.to_numpy(np.float32),
    }


def read_metadata_column(
    path: Path,
    sample_ids: Sequence[str],
    sample_column: str,
    value_column: str,
) -> np.ndarray:
    separator = "," if path.suffix.lower() == ".csv" else "\t"
    table = pd.read_csv(path, sep=separator, low_memory=False)
    required = {sample_column, value_column}
    if not required.issubset(table.columns):
        raise ValueError(f"sample metadata lacks {value_column}")
    table[sample_column] = table[sample_column].astype(str)
    if table[sample_column].duplicated().any():
        raise ValueError("sample metadata contains duplicate sample identifiers")
    mapping = table.set_index(sample_column)[value_column].astype(str)
    wanted = list(map(str, sample_ids))
    if not set(wanted).issubset(mapping.index):
        raise ValueError(f"sample metadata lacks {value_column} for development samples")
    return mapping.loc[wanted].to_numpy(str)


def validate_measured_provenance(path: Path, split) -> None:
    table = pd.read_csv(path, sep="\t")
    required = {
        "sample_id",
        "input_role",
        "phosphosite_labels_used",
        "transform_fit_role",
    }
    if not required.issubset(table.columns):
        raise ValueError("measured-protein provenance is incomplete")
    table["sample_id"] = table["sample_id"].astype(str)
    if set(table["sample_id"]) != set(split.development_ids.tolist()):
        raise ValueError("measured-protein provenance differs from development samples")
    if set(table["sample_id"]) & set(split.sealed_ids.tolist()):
        raise ValueError("sealed samples occur in measured-protein provenance")
    used = table["phosphosite_labels_used"].astype(str).str.lower()
    if used.isin({"1", "true", "yes"}).any():
        raise ValueError("measured protein declares phosphosite-label use")


def validate_trainfit_prediction_provenance(path: Path, split) -> None:
    """Validate the declared single-split protein predictor used for exploration."""

    table = pd.read_csv(path, sep="\t")
    required = {
        "sample_id",
        "prediction_role",
        "phosphosite_labels_used",
        "source_archive",
    }
    if not required.issubset(table.columns):
        raise ValueError("train-fitted protein provenance is incomplete")
    table["sample_id"] = table["sample_id"].astype(str)
    if table["sample_id"].duplicated().any():
        raise ValueError("train-fitted protein provenance contains duplicate patients")
    if set(table["sample_id"]) != set(split.development_ids.tolist()):
        raise ValueError("train-fitted protein provenance differs from development samples")
    used = table["phosphosite_labels_used"].astype(str).str.lower()
    if used.isin({"1", "true", "yes"}).any():
        raise ValueError("protein predictor declares phosphosite-label use")
    role = table.set_index("sample_id")["prediction_role"].astype(str)
    if not (role.loc[split.train_ids] == "train_full_model").all():
        raise ValueError("training protein predictions have an incorrect role")
    if not (role.loc[split.validation_ids] == "validation_train_only").all():
        raise ValueError("validation protein predictions have an incorrect role")


def sample_percentile_rank_profiles(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank observed phosphosites independently within each patient.

    Average ranks are used for ties and mapped to the closed interval [0, 1].
    Missing entries remain zero in the dense tensor and false in the returned
    observation mask.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("phosphosite matrix must be two-dimensional")
    observed = np.isfinite(array)
    ranked = np.zeros(array.shape, dtype=np.float32)
    for row in range(array.shape[0]):
        columns = np.flatnonzero(observed[row])
        count = len(columns)
        if count == 0:
            continue
        if count == 1:
            ranked[row, columns[0]] = 0.5
            continue
        values_row = array[row, columns]
        order = np.argsort(values_row, kind="mergesort")
        sorted_values = values_row[order]
        ranked_sorted = np.empty(count, dtype=np.float64)
        start = 0
        while start < count:
            stop = start + 1
            while stop < count and sorted_values[stop] == sorted_values[start]:
                stop += 1
            average_zero_based_rank = 0.5 * (start + stop - 1)
            ranked_sorted[start:stop] = average_zero_based_rank / (count - 1)
            start = stop
        ranked[row, columns[order]] = ranked_sorted.astype(np.float32)
    if not np.isfinite(ranked).all():
        raise ValueError("sample-percentile target contains non-finite values")
    if bool(((ranked < 0.0) | (ranked > 1.0))[observed].any()):
        raise ValueError("sample-percentile target lies outside [0, 1]")
    return ranked, observed


def target_normalization_method(scaling: str) -> str:
    if scaling == "sample_percentile_rank":
        return "sample_percentile_rank"
    if scaling == "identity_logscale":
        return "identity_logscale"
    if scaling == "identity_reference_quantile":
        return "reference_quantile_phosphosite"
    if scaling in {"minmax", "quantile_01_99"}:
        return f"training_site_{scaling}"
    raise ValueError(f"unknown site scaling: {scaling}")


def normalize_site_profiles(
    values: np.ndarray,
    train_index: np.ndarray,
    *,
    minimum_training_observations: int = 2,
    scaling: str = "minmax",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct the requested target coordinate and its observation mask."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("phosphosite matrix must be two-dimensional")
    train_index = np.asarray(train_index, dtype=np.int64)
    if train_index.ndim != 1 or not len(train_index):
        raise ValueError("training indices must be a non-empty vector")
    observed = np.isfinite(array)
    training = np.where(observed[train_index], array[train_index], np.nan)
    training_count = np.isfinite(training).sum(axis=0).astype(np.int64)
    if bool((training_count < minimum_training_observations).any()):
        bad = np.flatnonzero(training_count < minimum_training_observations)
        raise ValueError(
            "phosphosites lack enough training observations for site scaling; "
            f"examples={bad[:5].tolist()}"
        )
    if scaling == "sample_percentile_rank":
        normalized, ranked_observed = sample_percentile_rank_profiles(array)
        if not np.array_equal(ranked_observed, observed):
            raise RuntimeError("sample-percentile ranking changed the observation mask")
        minimum = np.zeros(array.shape[1], dtype=np.float32)
        maximum = np.ones(array.shape[1], dtype=np.float32)
        return normalized, observed, minimum, maximum, training_count
    if scaling in {"identity_logscale", "identity_reference_quantile"}:
        normalized = np.where(observed, array, 0.0).astype(np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError("identity-logscale phosphosite target contains non-finite values")
        minimum = np.zeros(array.shape[1], dtype=np.float32)
        maximum = np.ones(array.shape[1], dtype=np.float32)
        return normalized, observed, minimum, maximum, training_count
    if scaling == "minmax":
        minimum = np.nanmin(training, axis=0).astype(np.float32)
        maximum = np.nanmax(training, axis=0).astype(np.float32)
    elif scaling == "quantile_01_99":
        minimum = np.nanquantile(training, 0.01, axis=0).astype(np.float32)
        maximum = np.nanquantile(training, 0.99, axis=0).astype(np.float32)
    else:
        raise ValueError(f"unknown site scaling: {scaling}")
    span = maximum - minimum
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise ValueError("training-site scaling bounds are non-finite")
    nonconstant = span > 1.0e-8
    safe_span = np.where(nonconstant, span, 1.0).astype(np.float32)
    normalized = (array - minimum[None, :]) / safe_span[None, :]
    normalized[:, ~nonconstant] = 0.0
    normalized = np.where(observed, normalized, 0.0).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("site-scaled phosphosite target contains non-finite values")
    return normalized, observed, minimum, maximum, training_count


def apply_site_profile_scaler(
    values: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    *,
    scaling: str = "minmax",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a fitted site transform or an independent patient percentile rank."""
    if scaling == "sample_percentile_rank":
        return sample_percentile_rank_profiles(values)
    if scaling in {"identity_logscale", "identity_reference_quantile"}:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("phosphosite matrix must be two-dimensional")
        observed = np.isfinite(array)
        transformed = np.where(observed, array, 0.0).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise ValueError("identity-logscale phosphosite target contains non-finite values")
        return transformed, observed
    if scaling not in {"minmax", "quantile_01_99"}:
        raise ValueError(f"unknown site scaling: {scaling}")
    array = np.asarray(values, dtype=np.float32)
    observed = np.isfinite(array)
    minimum = np.asarray(minimum, dtype=np.float32)
    maximum = np.asarray(maximum, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != len(minimum) or minimum.shape != maximum.shape:
        raise ValueError("site scaler and phosphosite matrix have inconsistent shapes")
    span = maximum - minimum
    nonconstant = span > 1.0e-8
    safe_span = np.where(nonconstant, span, 1.0).astype(np.float32)
    normalized = (array - minimum[None, :]) / safe_span[None, :]
    normalized[:, ~nonconstant] = 0.0
    normalized = np.where(observed, normalized, 0.0).astype(np.float32)
    return normalized, observed


def normalize_rna_input(
    values: np.ndarray,
    train_index: np.ndarray,
    *,
    transform: str,
    fitted_mean: np.ndarray | None = None,
    fitted_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare RNA without using validation values to fit any parameter.

    Continuous reference-quantile RNA is standardized feature-wise. Missing
    entries become zero after standardization, which is exactly training-center
    imputation. The observed mask is returned for reporting and interface tests.
    """
    array = np.asarray(values, dtype=np.float32)
    train_index = np.asarray(train_index, dtype=np.int64)
    if array.ndim != 2 or train_index.ndim != 1 or not len(train_index):
        raise ValueError("RNA matrix and training indices have incompatible shapes")
    observed = np.isfinite(array)
    if transform == "sample_rank":
        if not observed.all():
            raise ValueError("sample-rank RNA input must be fully imputed before training")
        normalized = sample_rank_encode(array)
        mean = np.zeros(array.shape[1], dtype=np.float32)
        scale = np.ones(array.shape[1], dtype=np.float32)
    elif transform == "feature_zscore":
        if (fitted_mean is None) != (fitted_scale is None):
            raise ValueError(
                "checkpoint RNA mean and scale must be provided together"
            )
        if fitted_mean is None:
            training_count = observed[train_index].sum(axis=0)
            if bool((training_count == 0).any()):
                bad = np.flatnonzero(training_count == 0)
                raise ValueError(
                    "reference-quantile RNA features lack training observations; "
                    f"examples={bad[:5].tolist()}"
                )
            mean, scale = fit_feature_zscore(array, train_index)
        else:
            mean = np.asarray(fitted_mean, dtype=np.float32)
            scale = np.asarray(fitted_scale, dtype=np.float32)
            expected = (array.shape[1],)
            if mean.shape != expected or scale.shape != expected:
                raise ValueError("checkpoint RNA scaler differs from current input")
            if not np.isfinite(mean).all() or not np.isfinite(scale).all():
                raise ValueError("checkpoint RNA scaler contains non-finite values")
            if bool((scale <= 0).any()):
                raise ValueError("checkpoint RNA scale must be positive")
        normalized = apply_feature_zscore(array, mean, scale)
    else:
        raise ValueError(f"unknown RNA input transform: {transform}")
    if not np.isfinite(normalized).all():
        raise ValueError("normalized RNA input contains non-finite values")
    return normalized, mean, scale, observed


def decode_h5_text(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8")
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
            for value in values
        ],
        dtype=str,
    )


def load_esm2_site_prior(
    path: Path | None, targets: Sequence[str]
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, object],
]:
    if path is None:
        return None, None, None, None, {
            "enabled": False,
            "embedding_dimension": 0,
            "site_coverage": 0.0,
        }
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "site_names",
            "site_residue_embedding",
            "site_residue_mask",
            "model_name",
            "summary_json",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"ESM-2 先验包缺少数组: {sorted(missing)}")
        saved_targets = decode_h5_text(archive["site_names"])
        if not np.array_equal(saved_targets, np.asarray(targets, dtype=str)):
            raise ValueError("ESM-2 位点词表或顺序与训练目标不一致")
        embedding = np.asarray(archive["site_residue_embedding"], dtype=np.float16)
        mask = np.asarray(archive["site_residue_mask"], dtype=bool)
        local_embedding = (
            np.asarray(archive["site_local_embedding"], dtype=np.float16)
            if "site_local_embedding" in archive
            else None
        )
        local_mask = (
            np.asarray(archive["site_local_mask"], dtype=bool)
            if "site_local_mask" in archive
            else None
        )
        model_name = str(np.asarray(archive["model_name"]).item())
        summary = json.loads(str(np.asarray(archive["summary_json"]).item()))
    if embedding.ndim != 3 or embedding.shape[:2] != mask.shape:
        raise ValueError("ESM-2 逐残基表示和掩码形状不一致")
    if embedding.shape[0] != len(targets) or embedding.shape[2] < 1:
        raise ValueError("ESM-2 逐残基表示维度无效")
    if not np.isfinite(embedding[mask]).all():
        raise ValueError("ESM-2 有效逐残基表示包含非有限值")
    if (local_embedding is None) != (local_mask is None):
        raise ValueError("ESM-2 局部表示与掩码必须同时存在")
    if local_embedding is not None:
        if local_embedding.ndim != 4 or local_embedding.shape[:-1] != local_mask.shape:
            raise ValueError("ESM-2 局部表示和掩码形状不一致")
        if local_embedding.shape[0] != len(targets):
            raise ValueError("ESM-2 局部表示没有覆盖全部位点")
        if local_embedding.shape[-1] != embedding.shape[-1]:
            raise ValueError("ESM-2 局部表示维度与中心残基不同")
        if not np.isfinite(local_embedding[local_mask]).all():
            raise ValueError("ESM-2 有效局部表示包含非有限值")
    if bool(summary.get("labels_read", True)):
        raise ValueError("ESM-2 先验包未声明标签隔离")
    return embedding, mask, local_embedding, local_mask, {
        "enabled": True,
        "model_name": model_name,
        "embedding_dimension": int(embedding.shape[2]),
        "maximum_residues_per_site": int(embedding.shape[1]),
        "site_coverage": float(mask.any(axis=1).mean()),
        "residue_coverage": float(mask.mean()),
        "local_window_available": local_embedding is not None,
        "local_token_coverage": float(local_mask.mean())
        if local_mask is not None and local_mask.size
        else 0.0,
        "source_summary": summary,
    }


def load_esm3_site_prior(
    path: Path | None, targets: Sequence[str]
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, object]]:
    embedding, mask, local_embedding, local_mask, audit = load_esm2_site_prior(
        path, targets
    )
    if local_embedding is not None or local_mask is not None:
        raise ValueError("ESM-3 prior package must contain residue embeddings only")
    if audit.get("enabled"):
        audit = dict(audit)
        audit["prior_family"] = "ESM-3"
    return embedding, mask, audit


def vocabulary_reindex(source: Sequence[str], target: Sequence[str]) -> np.ndarray:
    source_upper = np.char.upper(np.asarray(source, dtype=str))
    target_upper = np.char.upper(np.asarray(target, dtype=str))
    if len(set(source_upper.tolist())) != source_upper.size:
        raise ValueError("source protein vocabulary contains case-insensitive duplicates")
    if len(set(target_upper.tolist())) != target_upper.size:
        raise ValueError("target protein vocabulary contains case-insensitive duplicates")
    if set(source_upper.tolist()) != set(target_upper.tolist()):
        missing = sorted(set(target_upper.tolist()) - set(source_upper.tolist()))
        extra = sorted(set(source_upper.tolist()) - set(target_upper.tolist()))
        raise ValueError(
            "protein vocabularies contain different members; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    lookup = {value: index for index, value in enumerate(source_upper.tolist())}
    return np.asarray([lookup[value] for value in target_upper.tolist()], dtype=np.int64)


def masked_site_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    full_site_count: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    """Return this chunk's contribution to site-equal masked Huber loss."""
    if prediction.shape != target.shape or mask.shape != prediction.shape:
        raise ValueError("masked Huber tensors have inconsistent shapes")
    error = prediction.float() - target.float()
    absolute = error.abs()
    elementwise = torch.where(
        absolute <= delta,
        0.5 * error.square(),
        delta * (absolute - 0.5 * delta),
    )
    count = mask.sum(dim=0)
    per_site = (elementwise * mask.float()).sum(dim=0) / count.clamp_min(1)
    return per_site[count > 0].sum() / full_site_count.float().clamp_min(1)


def masked_profile_mse_chunk(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    full_patient_observation_count: torch.Tensor,
) -> torch.Tensor:
    """Return one chunk's exact contribution to patient-equal masked MSE.

    ``full_patient_observation_count`` is computed once from the complete
    patient-by-site mask. Summing this function over every site chunk equals
    ``mean_i(mean_{s in Omega_i}((prediction_is - target_is)^2))`` and retains
    the same gradient as an unchunked calculation.
    """
    if prediction.shape != target.shape or mask.shape != prediction.shape:
        raise ValueError("masked profile MSE tensors have inconsistent shapes")
    if prediction.ndim != 2:
        raise ValueError("masked profile MSE tensors must be two-dimensional")
    if (
        full_patient_observation_count.ndim != 1
        or full_patient_observation_count.numel() != prediction.shape[0]
    ):
        raise ValueError(
            "masked profile MSE patient observation count has an inconsistent shape"
        )

    prediction_float = prediction.float()
    target_float = target.float()
    observed = mask.bool()
    if not bool(torch.isfinite(prediction_float[observed]).all()):
        raise FloatingPointError("profile MSE observed prediction is non-finite")
    if not bool(torch.isfinite(target_float[observed]).all()):
        raise FloatingPointError("profile MSE observed target is non-finite")
    count = full_patient_observation_count.to(
        device=prediction.device, dtype=torch.float32
    )
    if not bool(torch.isfinite(count).all()) or bool((count < 0).any()):
        raise ValueError("profile MSE patient observation count is invalid")
    if bool((observed.sum(dim=1).float() > count).any()):
        raise ValueError(
            "profile MSE chunk observations exceed the full patient count"
        )

    safe_prediction = torch.where(observed, prediction_float, 0.0)
    safe_target = torch.where(observed, target_float, 0.0)
    per_patient_chunk = (safe_prediction - safe_target).square().sum(dim=1)
    per_patient_chunk = per_patient_chunk / count.clamp_min(1.0)
    eligible = count > 0
    if bool(eligible.any()):
        return per_patient_chunk[eligible].sum() / eligible.sum().float()
    return safe_prediction.sum() * 0.0


def masked_site_pearson_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    full_site_count: torch.Tensor,
    *,
    minimum_observations: int,
) -> torch.Tensor:
    """Return a site-equal cross-patient Pearson loss for one site chunk."""
    if prediction.shape != target.shape or mask.shape != prediction.shape:
        raise ValueError("masked Pearson tensors have inconsistent shapes")
    weight = mask.float()
    count = weight.sum(dim=0)
    safe_count = count.clamp_min(1.0)
    prediction = prediction.float()
    target = target.float()
    prediction_mean = (prediction * weight).sum(dim=0) / safe_count
    target_mean = (target * weight).sum(dim=0) / safe_count
    prediction_centered = (prediction - prediction_mean) * weight
    target_centered = (target - target_mean) * weight
    covariance = (prediction_centered * target_centered).sum(dim=0)
    prediction_square_norm = prediction_centered.square().sum(dim=0)
    target_square_norm = target_centered.square().sum(dim=0)
    # Some sites can be exactly constant at initialization.  Taking sqrt(0)
    # creates an infinite derivative even when the site is excluded below,
    # which can turn the first backward pass into NaN through 0 * inf.
    # Clamp before sqrt while retaining the original eligibility criterion.
    denominator = (
        prediction_square_norm.clamp_min(1.0e-8)
        * target_square_norm.clamp_min(1.0e-8)
    ).sqrt()
    eligible = (
        (count >= minimum_observations)
        & (prediction_square_norm > 1.0e-8)
        & (target_square_norm > 1.0e-8)
    )
    correlation = covariance / denominator
    return (1.0 - correlation[eligible]).sum() / full_site_count.float().clamp_min(1)


def masked_profile_pearson_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    minimum_observations: int = 2,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Return patient-equal Pearson loss across every observed site.

    Non-finite values outside ``mask`` are ignored.  A non-finite observed value
    is an invalid training batch and raises immediately.  Patients with too few
    observed sites or a zero-variance target do not contribute to the mean.
    """
    if prediction.shape != target.shape or mask.shape != prediction.shape:
        raise ValueError("masked profile Pearson tensors have inconsistent shapes")
    if prediction.ndim != 2:
        raise ValueError("masked profile Pearson tensors must be two-dimensional")
    if minimum_observations < 2:
        raise ValueError("profile Pearson minimum_observations must be at least two")

    prediction_float = prediction.float()
    target_float = target.float()
    valid = mask.bool()
    if not bool(torch.isfinite(prediction_float[valid]).all()):
        raise FloatingPointError("profile Pearson observed prediction is non-finite")
    if not bool(torch.isfinite(target_float[valid]).all()):
        raise FloatingPointError("profile Pearson observed target is non-finite")
    weight = valid.float()
    safe_prediction = torch.where(valid, prediction_float, 0.0)
    safe_target = torch.where(valid, target_float, 0.0)
    count = weight.sum(dim=1)
    safe_count = count.clamp_min(1.0)
    prediction_mean = safe_prediction.sum(dim=1) / safe_count
    target_mean = safe_target.sum(dim=1) / safe_count
    prediction_centered = torch.where(
        valid, prediction_float - prediction_mean[:, None], 0.0
    )
    target_centered = torch.where(
        valid, target_float - target_mean[:, None], 0.0
    )
    covariance = (prediction_centered * target_centered).sum(dim=1)
    prediction_square_norm = prediction_centered.square().sum(dim=1)
    target_square_norm = target_centered.square().sum(dim=1)
    eligible = (
        (count >= minimum_observations)
        & (target_square_norm > epsilon)
        & torch.isfinite(covariance)
        & torch.isfinite(prediction_square_norm)
        & torch.isfinite(target_square_norm)
    )
    denominator = (
        (prediction_square_norm + epsilon)
        * (target_square_norm + epsilon)
    ).sqrt()
    correlation = covariance / denominator
    if bool(eligible.any()):
        return (1.0 - correlation[eligible]).mean()
    return safe_prediction.sum() * 0.0


def new_profile_pearson_statistics(
    patient_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create detached sufficient statistics for a site-chunked profile loss."""
    if patient_count < 1:
        raise ValueError("profile Pearson patient_count must be positive")
    return {
        name: torch.zeros(patient_count, dtype=torch.float32, device=device)
        for name in (
            "count",
            "prediction_sum",
            "target_sum",
            "prediction_square_sum",
            "target_square_sum",
            "cross_sum",
        )
    }


def update_profile_pearson_statistics(
    statistics: dict[str, torch.Tensor],
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Accumulate one site chunk without retaining its autograd graph."""
    if prediction.shape != target.shape or mask.shape != prediction.shape:
        raise ValueError("profile Pearson chunk tensors have inconsistent shapes")
    if prediction.ndim != 2 or prediction.shape[0] != statistics["count"].numel():
        raise ValueError("profile Pearson chunk has an inconsistent patient axis")
    with torch.no_grad():
        prediction_float = prediction.detach().float()
        target_float = target.detach().float()
        valid = mask.bool()
        if not bool(torch.isfinite(prediction_float[valid]).all()):
            raise FloatingPointError(
                "profile Pearson observed prediction is non-finite"
            )
        if not bool(torch.isfinite(target_float[valid]).all()):
            raise FloatingPointError("profile Pearson observed target is non-finite")
        safe_prediction = torch.where(valid, prediction_float, 0.0)
        safe_target = torch.where(valid, target_float, 0.0)
        statistics["count"].add_(valid.sum(dim=1).float())
        statistics["prediction_sum"].add_(safe_prediction.sum(dim=1))
        statistics["target_sum"].add_(safe_target.sum(dim=1))
        statistics["prediction_square_sum"].add_(
            safe_prediction.square().sum(dim=1)
        )
        statistics["target_square_sum"].add_(safe_target.square().sum(dim=1))
        statistics["cross_sum"].add_((safe_prediction * safe_target).sum(dim=1))


def finalize_profile_pearson_statistics(
    statistics: dict[str, torch.Tensor],
    *,
    minimum_observations: int = 2,
    epsilon: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Finalize global patient statistics used by every site chunk."""
    if minimum_observations < 2:
        raise ValueError("profile Pearson minimum_observations must be at least two")
    count = statistics["count"]
    safe_count = count.clamp_min(1.0)
    prediction_mean = statistics["prediction_sum"] / safe_count
    target_mean = statistics["target_sum"] / safe_count
    prediction_square_norm = (
        statistics["prediction_square_sum"]
        - statistics["prediction_sum"].square() / safe_count
    ).clamp_min(0.0)
    target_square_norm = (
        statistics["target_square_sum"]
        - statistics["target_sum"].square() / safe_count
    ).clamp_min(0.0)
    covariance = (
        statistics["cross_sum"]
        - statistics["prediction_sum"] * statistics["target_sum"] / safe_count
    )
    eligible = (
        (count >= minimum_observations)
        & (target_square_norm > epsilon)
        & torch.isfinite(covariance)
        & torch.isfinite(prediction_square_norm)
        & torch.isfinite(target_square_norm)
    )
    denominator = (
        (prediction_square_norm + epsilon)
        * (target_square_norm + epsilon)
    ).sqrt()
    correlation = torch.nan_to_num(covariance / denominator)
    eligible_count = eligible.sum().clamp_min(1).float()
    if bool(eligible.any()):
        loss = (1.0 - correlation[eligible]).mean()
    else:
        loss = statistics["prediction_sum"].sum() * 0.0
    return {
        "prediction_mean": prediction_mean,
        "target_mean": target_mean,
        "prediction_square_norm": prediction_square_norm,
        "target_square_norm": target_square_norm,
        "correlation": correlation,
        "eligible": eligible,
        "eligible_count": eligible_count,
        "epsilon": torch.as_tensor(epsilon, dtype=torch.float32, device=count.device),
        "loss": loss,
    }


def masked_profile_pearson_chunk_surrogate(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    state: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return a chunk-local surrogate carrying the exact global-loss gradient.

    ``state`` must come from a detached full-profile pass over the same model
    outputs.  Summing this surrogate over all site chunks yields the exact
    gradient of the patient-equal full-profile Pearson loss while each decoder
    graph can be released after its chunk backward call.
    """
    if prediction.shape != target.shape or mask.shape != prediction.shape:
        raise ValueError("profile Pearson surrogate tensors have inconsistent shapes")
    if prediction.ndim != 2 or prediction.shape[0] != state["eligible"].numel():
        raise ValueError("profile Pearson surrogate has an inconsistent patient axis")
    prediction_float = prediction.float()
    target_float = target.float()
    observed = mask.bool()
    if not bool(torch.isfinite(prediction_float[observed]).all()):
        raise FloatingPointError("profile Pearson observed prediction is non-finite")
    if not bool(torch.isfinite(target_float[observed]).all()):
        raise FloatingPointError("profile Pearson observed target is non-finite")
    valid = observed & state["eligible"][:, None]
    safe_prediction = torch.where(valid, prediction_float, 0.0)
    prediction_centered = torch.where(
        valid, prediction_float - state["prediction_mean"][:, None], 0.0
    )
    target_centered = torch.where(
        valid, target_float - state["target_mean"][:, None], 0.0
    )
    correlation_gradient = (
        target_centered
        / (
            (state["prediction_square_norm"] + state["epsilon"])
            * (state["target_square_norm"] + state["epsilon"])
        ).sqrt()[:, None]
        - state["correlation"][:, None]
        * prediction_centered
        / (state["prediction_square_norm"] + state["epsilon"])[:, None]
    )
    loss_gradient = -correlation_gradient / state["eligible_count"]
    loss_gradient = torch.where(valid, loss_gradient, 0.0).detach()
    return (safe_prediction * loss_gradient).sum()


def parse_rank_pair_offsets(value: str) -> tuple[int, ...]:
    """Parse a comma-separated collection of positive patient-pair offsets."""
    try:
        offsets = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise ValueError("rank_pair_offsets must contain comma-separated integers") from error
    if not offsets or offsets[0] < 1:
        raise ValueError("rank_pair_offsets must contain positive integers")
    return offsets


def pairwise_rank_site_count(mask: torch.Tensor, offsets: Sequence[int]) -> torch.Tensor:
    """Count sites that contribute at least one sampled observed patient pair."""
    if mask.ndim != 2:
        raise ValueError("pairwise rank mask must be two-dimensional")
    eligible = torch.zeros(mask.shape[1], dtype=torch.bool, device=mask.device)
    for offset in offsets:
        if offset >= mask.shape[0]:
            continue
        eligible |= (mask[:-offset] & mask[offset:]).any(dim=0)
    return eligible.sum()


def masked_site_pairwise_rank_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    full_site_count: torch.Tensor,
    *,
    offsets: Sequence[int],
    temperature: float,
    minimum_target_difference: float,
) -> torch.Tensor:
    """Return a site-equal RankNet loss over sampled observed patient pairs."""
    if prediction.shape != target.shape or mask.shape != prediction.shape:
        raise ValueError("masked pairwise-rank tensors have inconsistent shapes")
    prediction = prediction.float()
    target = target.float()
    site_loss = torch.zeros(prediction.shape[1], device=prediction.device)
    site_pairs = torch.zeros(prediction.shape[1], device=prediction.device)
    for offset in offsets:
        if offset >= prediction.shape[0]:
            continue
        target_difference = target[:-offset] - target[offset:]
        valid = (
            mask[:-offset]
            & mask[offset:]
            & (target_difference.abs() > minimum_target_difference)
        )
        direction = target_difference.sign()
        prediction_difference = prediction[:-offset] - prediction[offset:]
        pair_loss = F.softplus(
            -direction * prediction_difference / temperature
        ) / math.log(2.0)
        site_loss = site_loss + (pair_loss * valid.float()).sum(dim=0)
        site_pairs = site_pairs + valid.sum(dim=0)
    per_site = site_loss / site_pairs.clamp_min(1.0)
    eligible = site_pairs > 0
    return per_site[eligible].sum() / full_site_count.float().clamp_min(1.0)


def masked_batch_cosine(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    *,
    batch_size: int = 8,
) -> tuple[float, np.ndarray]:
    values: list[float] = []
    for start in range(0, len(target), batch_size):
        rows = slice(start, min(start + batch_size, len(target)))
        valid = mask[rows] & np.isfinite(target[rows]) & np.isfinite(prediction[rows])
        truth = target[rows][valid].astype(np.float64)
        estimate = prediction[rows][valid].astype(np.float64)
        denominator = np.linalg.norm(truth) * np.linalg.norm(estimate)
        values.append(
            float(np.dot(truth, estimate) / max(float(denominator), 1.0e-12))
        )
    result = np.asarray(values, dtype=np.float64)
    return float(result.mean()), result


def per_sample_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    sample_ids: Sequence[str],
    *,
    pearson_minimum_observations: int = 2,
    pearson_epsilon: float = 1.0e-6,
) -> pd.DataFrame:
    if pearson_minimum_observations < 2:
        raise ValueError("pearson_minimum_observations must be at least two")
    rows: list[dict[str, object]] = []
    for row, sample in enumerate(sample_ids):
        valid = mask[row].astype(bool)
        if not np.isfinite(target[row, valid]).all():
            raise FloatingPointError("observed validation target is non-finite")
        if not np.isfinite(prediction[row, valid]).all():
            raise FloatingPointError("observed validation prediction is non-finite")
        truth = target[row, valid].astype(np.float64)
        estimate = prediction[row, valid].astype(np.float64)
        denominator = np.linalg.norm(truth) * np.linalg.norm(estimate)
        cosine = float(np.dot(truth, estimate) / max(float(denominator), 1.0e-12))
        spearman = pd.Series(truth).corr(pd.Series(estimate), method="spearman")
        truth_centered = truth - truth.mean()
        estimate_centered = estimate - estimate.mean()
        truth_square_norm = float(np.square(truth_centered).sum())
        estimate_square_norm = float(np.square(estimate_centered).sum())
        pearson_eligible = bool(
            len(truth) >= pearson_minimum_observations
            and truth_square_norm > pearson_epsilon
        )
        within_sample_pearson = (
            float(
                np.dot(truth_centered, estimate_centered)
                / np.sqrt(
                    (truth_square_norm + pearson_epsilon)
                    * (estimate_square_norm + pearson_epsilon)
                )
            )
            if pearson_eligible
            else np.nan
        )
        rows.append(
            {
                "sample_id": str(sample),
                "n_observed_sites": int(valid.sum()),
                "cosine": cosine,
                "within_sample_spearman": float(spearman),
                "within_sample_pearson": within_sample_pearson,
                "within_sample_pearson_eligible": pearson_eligible,
                "mse": float(np.mean(np.square(truth - estimate))),
                "mae": float(np.mean(np.abs(truth - estimate))),
            }
        )
    return pd.DataFrame(rows)


def input_hash_contract(path: Path, args: argparse.Namespace) -> None:
    sources = [
        ("rna", args.rna),
        ("protein_prediction", args.protein_prediction),
        ("protein_reliability", args.protein_reliability),
        ("protein_provenance", args.protein_provenance),
        ("phosphosite", args.phosphosite),
        ("phosphosite_manifest", args.phosphosite_manifest),
        ("split_manifest", args.split_manifest),
        ("sample_metadata", args.sample_metadata),
        (
            "cophee_prior_bundle_manifest",
            args.cophee_prior_bundle / "artifact_sha256.tsv",
        ),
    ]
    if args.esm2_prior_npz is not None:
        sources.append(("esm2_sequence_prior", args.esm2_prior_npz))
    if args.esm3_prior_npz is not None:
        sources.append(("esm3_sequence_prior", args.esm3_prior_npz))
    if args.pan_checkpoint is not None:
        sources.append(("pan_checkpoint", args.pan_checkpoint))
    if args.initial_checkpoint is not None:
        sources.append(("initial_checkpoint", args.initial_checkpoint))
    if args.conservative_pretrain_h5 is not None:
        sources.append(("conservative_pretrain_h5", args.conservative_pretrain_h5))
    if args.conservative_pretrain_protein_npz is not None:
        sources.append(
            ("conservative_pretrain_protein_npz", args.conservative_pretrain_protein_npz)
        )
    if args.development_protein_npz is not None:
        sources.append(("development_protein_npz", args.development_protein_npz))
    rows = []
    for name, source in sources:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        rows.append(
            {
                "argument": name,
                "path": str(source),
                "size_bytes": int(source.stat().st_size),
                "sha256": sha256_file(source),
            }
        )
    table = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def validate_loss_configuration(args: argparse.Namespace) -> None:
    """Validate loss weights and the controlled continuous-profile objective."""
    weights = (
        args.site_huber_weight,
        args.site_pearson_weight,
        args.site_rank_weight,
        args.profile_pearson_weight,
        args.profile_mse_weight,
    )
    if any(weight < 0 for weight in weights):
        raise ValueError("site loss weights must be non-negative")
    if sum(weights) <= 0:
        raise ValueError("at least one site loss must be enabled")
    if args.profile_mse_weight:
        competing = (
            args.site_huber_weight,
            args.site_pearson_weight,
            args.site_rank_weight,
            args.profile_pearson_weight,
        )
        if any(competing):
            raise ValueError(
                "profile MSE is an exclusive controlled objective; disable Huber, "
                "site Pearson, rank, and profile Pearson"
            )
        if args.site_scaling not in {"identity_logscale", "identity_reference_quantile"}:
            raise ValueError(
                "profile MSE requires continuous identity_logscale phosphosite targets"
            )
        if args.early_stopping_metric != "profile_spearman":
            raise ValueError(
                "profile MSE requires profile_spearman checkpoint selection"
            )
        if args.training_stage not in {
            "pan",
            "cancer_adapter",
            "external_full_finetune",
        }:
            raise ValueError(
                "profile MSE supports pan training, a frozen cancer adapter, "
                "or external full-model fine-tuning"
            )


def main() -> int:
    args = arguments()
    rank_pair_offsets = parse_rank_pair_offsets(args.rank_pair_offsets)
    if args.sample_batch_size < 1 or args.evaluation_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.site_chunk_size < 1 or args.site_chunk_size > 1000:
        raise ValueError("site_chunk_size must lie in [1, 1000]")
    validate_loss_configuration(args)
    if args.site_huber_delta <= 0:
        raise ValueError("site_huber_delta must be positive")
    if args.rank_temperature <= 0:
        raise ValueError("rank_temperature must be positive")
    if args.rank_minimum_target_difference < 0:
        raise ValueError("rank_minimum_target_difference must be non-negative")
    if args.correlation_min_observations < 2:
        raise ValueError("correlation_min_observations must be at least two")
    if args.profile_pearson_minimum_observations < 2:
        raise ValueError(
            "profile_pearson_minimum_observations must be at least two"
        )
    if args.training_stage in PAN_CHECKPOINT_STAGES and args.pan_checkpoint is None:
        raise ValueError(
            f"{args.training_stage} training requires --pan-checkpoint"
        )
    if args.pretrain_updates and args.conservative_pretrain_h5 is None:
        raise ValueError("pretraining updates require --conservative-pretrain-h5")
    if (
        args.pretrain_updates < 0
        or args.maximum_finetune_updates < 0
        or args.validation_interval_updates < 0
    ):
        raise ValueError("update budgets must be non-negative")
    if (
        args.validation_only_at_finetune_end
        and args.maximum_finetune_updates < 1
    ):
        raise ValueError(
            "end-only validation requires a positive maximum-finetune-updates"
        )
    if args.esm2_warmup_updates < 0 or args.esm3_warmup_updates < 0:
        raise ValueError("sequence-prior warm-up updates cannot be negative")
    if args.esm2_warmup_learning_rate <= 0:
        raise ValueError("esm2_warmup_learning_rate must be positive")
    if args.joint_learning_rate is not None and args.joint_learning_rate <= 0:
        raise ValueError("joint_learning_rate must be positive")
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            "this direct-profile model uses single-device execution; launch one process"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    started = time.time()
    output = args.output_dir
    for name in ("models", "predictions", "tables", "logs", "reports"):
        (output / name).mkdir(parents=True, exist_ok=True)
    input_hash_contract(output / "reports" / "input_artifact_sha256.tsv", args)

    split = validate_split_manifest(
        args.split_manifest,
        expected_sizes=(
            args.expected_train_samples,
            args.expected_validation_samples,
            args.expected_sealed_samples,
        ),
    )
    validate_case_split_disjointness(
        args.sample_metadata,
        split,
        sample_id_column=args.sample_id_column,
        case_id_column=args.case_id_column,
    )
    if args.protein_input_mode == "crossfit_prediction":
        validate_protein_prediction_provenance(args.protein_provenance, split)
    elif args.protein_input_mode == "trainfit_prediction":
        validate_trainfit_prediction_provenance(args.protein_provenance, split)
    else:
        validate_measured_provenance(args.protein_provenance, split)

    development_ids = split.development_ids.tolist()
    assert_development_only_phosphosite_rows(
        development_ids,
        split.train_ids.tolist(),
        split.validation_ids.tolist(),
        split.sealed_ids.tolist(),
    )
    n_train = len(split.train_ids)
    train_index = np.arange(n_train, dtype=np.int64)
    validation_index = np.arange(n_train, len(development_ids), dtype=np.int64)

    manifest = pd.read_csv(args.phosphosite_manifest, sep="\t")
    metadata = target_metadata(manifest)
    targets = metadata["targets"]
    parents = metadata["parents"]
    rna_frame = read_parquet_rows(args.rna, development_ids)
    protein_frame = read_prediction_matrix(args.protein_prediction, split)
    reliability_frame = read_parquet_rows(
        args.protein_reliability,
        development_ids,
        columns=protein_frame.columns.astype(str).tolist(),
    )
    phosphosite_frame = read_parquet_rows(
        args.phosphosite, development_ids, columns=targets
    )

    rna_raw = rna_frame.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    protein_raw = protein_frame.to_numpy(np.float32)
    reliability = reliability_frame.to_numpy(np.float32)
    pan_checkpoint_data = None
    if args.training_stage in PAN_CHECKPOINT_STAGES:
        if args.pan_checkpoint is None:
            raise ValueError(
                f"{args.training_stage} training requires --pan-checkpoint"
            )
        pan_checkpoint_data = torch.load(
            args.pan_checkpoint, map_location="cpu", weights_only=False
        )
    if args.development_protein_npz is not None:
        with np.load(args.development_protein_npz, allow_pickle=False) as archive:
            required = {
                "development_sample_id",
                "development_prediction_z",
                "development_role",
                "protein_names",
                "reliability",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(
                    f"development protein archive lacks required arrays: {missing}"
                )
            saved_ids = decode_h5_text(archive["development_sample_id"])
            saved_roles = decode_h5_text(archive["development_role"])
            saved_proteins = decode_h5_text(archive["protein_names"])
            saved_prediction = np.asarray(
                archive["development_prediction_z"], dtype=np.float32
            )
            saved_reliability = np.asarray(archive["reliability"], dtype=np.float32)
        if saved_ids.tolist() != list(map(str, development_ids)):
            raise ValueError("development protein predictions are not in locked split order")
        expected_roles = np.asarray(
            ["selection_train"] * n_train
            + ["selection_validation"] * len(validation_index)
        )
        role_alias = {
            "selection_train": "selection_train",
            "selection_train_oof": "selection_train",
            "train_full_model": "selection_train",
            "selection_validation": "selection_validation",
            "selection_validation_full916": "selection_validation",
            "selection_validation_full2067": "selection_validation",
            "validation_full_model": "selection_validation",
        }
        canonical_roles = np.asarray(
            [role_alias.get(str(role), "<invalid>") for role in saved_roles],
            dtype=str,
        )
        if not np.array_equal(canonical_roles, expected_roles):
            raise ValueError("development protein prediction roles violate locked split")
        protein_order = vocabulary_reindex(
            saved_proteins, protein_frame.columns.astype(str).tolist()
        )
        saved_prediction = saved_prediction[:, protein_order]
        if saved_reliability.ndim == 1:
            saved_reliability = saved_reliability[protein_order]
        else:
            saved_reliability = saved_reliability[:, protein_order]
        if saved_prediction.shape != protein_raw.shape:
            raise ValueError("development protein prediction matrix has incorrect shape")
        if saved_reliability.ndim == 1:
            saved_reliability = np.broadcast_to(
                saved_reliability[None, :], saved_prediction.shape
            ).copy()
        if saved_reliability.shape != saved_prediction.shape:
            raise ValueError("development protein reliability has incorrect shape")
        protein_raw = saved_prediction
        reliability = saved_reliability
    if pan_checkpoint_data is not None:
        checkpoint_rna_transform = str(
            pan_checkpoint_data.get("rna_input_transform", "sample_rank")
        )
        if checkpoint_rna_transform != args.rna_input_transform:
            raise ValueError(
                "pan checkpoint RNA transform differs from adaptation input: "
                f"{checkpoint_rna_transform} != {args.rna_input_transform}"
            )
        checkpoint_rna_mean = pan_checkpoint_data.get("rna_mean")
        checkpoint_rna_scale = pan_checkpoint_data.get("rna_scale")
    else:
        checkpoint_rna_mean = None
        checkpoint_rna_scale = None
    rna_input, rna_mean, rna_scale, rna_observed = normalize_rna_input(
        rna_raw,
        train_index,
        transform=args.rna_input_transform,
        fitted_mean=checkpoint_rna_mean,
        fitted_scale=checkpoint_rna_scale,
    )
    if not np.isfinite(reliability).all() or bool(
        ((reliability < 0) | (reliability > 1)).any()
    ):
        raise ValueError("protein reliability must lie in [0, 1]")
    if pan_checkpoint_data is not None:
        protein_mean = np.asarray(
            pan_checkpoint_data["protein_mean"], dtype=np.float32
        )
        protein_scale = np.asarray(
            pan_checkpoint_data["protein_scale"], dtype=np.float32
        )
        if protein_mean.shape != (protein_raw.shape[1],) or protein_scale.shape != (
            protein_raw.shape[1],
        ):
            raise ValueError("pan checkpoint protein scaler differs from adaptation input")
    else:
        protein_mean, protein_scale = fit_feature_zscore(protein_raw, train_index)
    protein = apply_feature_zscore(protein_raw, protein_mean, protein_scale)
    phosphosite_raw = phosphosite_frame.apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(np.float32)
    current_target_normalization = target_normalization_method(args.site_scaling)
    prediction_coordinate_label = (
        "raw_log"
        if args.site_scaling == "identity_logscale"
        else "reference_quantile"
        if args.site_scaling == "identity_reference_quantile"
        else "site_scaled"
    )
    if pan_checkpoint_data is not None:
        normalization = pan_checkpoint_data.get("normalization", {})
        pan_target_normalization = str(
            normalization.get("method", "training_site_minmax")
        )
        if pan_target_normalization != current_target_normalization:
            raise ValueError(
                "pan checkpoint target normalization differs from adaptation target"
            )
        site_minimum = np.asarray(
            normalization.get("site_minimum", []), dtype=np.float32
        )
        site_maximum = np.asarray(
            normalization.get("site_maximum", []), dtype=np.float32
        )
        if site_minimum.shape != (phosphosite_raw.shape[1],) or site_maximum.shape != (
            phosphosite_raw.shape[1],
        ):
            raise ValueError("pan checkpoint phosphosite scaler differs from adaptation targets")
        if args.site_scaling == "sample_percentile_rank":
            normalized_target, observed = sample_percentile_rank_profiles(
                phosphosite_raw
            )
        elif args.site_scaling in {"identity_logscale", "identity_reference_quantile"}:
            observed = np.isfinite(phosphosite_raw)
            normalized_target = np.where(
                observed, phosphosite_raw, 0.0
            ).astype(np.float32)
        else:
            span = np.maximum(site_maximum - site_minimum, 1.0e-6)
            observed = np.isfinite(phosphosite_raw)
            normalized_target = np.where(
                observed,
                (phosphosite_raw - site_minimum[None, :]) / span[None, :],
                0.0,
            ).astype(np.float32)
        site_training_count = observed[train_index].sum(axis=0).astype(np.int64)
    else:
        normalized_target, observed, site_minimum, site_maximum, site_training_count = (
            normalize_site_profiles(
                phosphosite_raw,
                train_index,
                scaling=args.site_scaling,
            )
        )

    study_center = None
    if args.target_study_residual:
        if args.site_scaling not in {"identity_logscale", "identity_reference_quantile"}:
            raise ValueError(
                "target-study-residual currently requires --site-scaling identity_logscale"
            )
        study_labels = read_metadata_column(
            args.sample_metadata,
            development_ids,
            args.sample_id_column,
            args.study_column,
        )
        observed_flat = np.where(observed, normalized_target, np.nan)
        global_column_mean = np.nanmean(
            observed_flat[train_index], axis=0
        ).astype(np.float32)
        global_column_mean = np.nan_to_num(global_column_mean, nan=0.0)
        global_column_count = np.isfinite(observed_flat[train_index]).sum(
            axis=0
        ).astype(np.float32)
        study_center = np.zeros_like(normalized_target, dtype=np.float32)
        for study_name in np.unique(study_labels):
            study_rows = np.flatnonzero(study_labels == study_name)
            train_rows_in_study = np.intersect1d(study_rows, train_index)
            if len(train_rows_in_study) == 0:
                study_center[study_rows] = global_column_mean[None, :]
                continue
            study_values = observed_flat[train_rows_in_study]
            site_mean = np.nanmean(study_values, axis=0).astype(np.float32)
            site_mean = np.nan_to_num(site_mean, nan=0.0)
            site_n = np.isfinite(study_values).sum(axis=0).astype(np.float32)
            # 量纲组整体偏移 b_g：只用高覆盖共同位点估计
            # （组内与全局训练观测数都不低于阈值的位点），取差值中位数。
            min_overlap = args.study_residual_min_overlap
            common_high = (site_n >= min_overlap) & (global_column_count >= min_overlap)
            if bool(common_high.any()):
                offset = np.median(
                    site_mean[common_high] - global_column_mean[common_high]
                )
            else:
                offset = 0.0
            offset = float(np.nan_to_num(offset, nan=0.0))
            shrink_target = global_column_mean + offset
            if args.study_residual_tau > 0:
                weight = site_n / (site_n + args.study_residual_tau)
                site_mean = (
                    weight[None, :] * site_mean
                    + (1.0 - weight)[None, :] * shrink_target[None, :]
                )
            study_center[study_rows] = site_mean[None, :]
        normalized_target = np.where(
            observed, normalized_target - study_center, 0.0
        ).astype(np.float32)
        if not np.isfinite(normalized_target).all():
            raise ValueError(
                "study-residual target contains non-finite values"
            )

    rna_vocabulary = {str(value).upper(): index for index, value in enumerate(rna_frame.columns)}
    protein_vocabulary = {
        str(value).upper(): index for index, value in enumerate(protein_frame.columns)
    }
    parent_protein_index = np.asarray(
        [protein_vocabulary.get(parent, 0) for parent in parents], dtype=np.int64
    )
    parent_protein_mask = np.asarray(
        [parent in protein_vocabulary for parent in parents], dtype=bool
    )
    parent_rna_index = np.asarray(
        [rna_vocabulary.get(parent, 0) for parent in parents], dtype=np.int64
    )
    parent_rna_mask = np.asarray(
        [parent in rna_vocabulary for parent in parents], dtype=bool
    )

    bundle = load_cophee_prior_bundle(
        args.cophee_prior_bundle,
        targets,
        include_site_site=False,
        include_kinase_site=True,
    )
    kinase_rna_index = np.asarray(
        [rna_vocabulary.get(value.upper(), 0) for value in bundle.kinase_vocabulary],
        dtype=np.int64,
    )
    kinase_rna_mask = np.asarray(
        [value.upper() in rna_vocabulary for value in bundle.kinase_vocabulary],
        dtype=bool,
    )
    kinase_protein_index = np.asarray(
        [
            protein_vocabulary.get(value.upper(), 0)
            for value in bundle.kinase_vocabulary
        ],
        dtype=np.int64,
    )
    kinase_protein_mask = np.asarray(
        [value.upper() in protein_vocabulary for value in bundle.kinase_vocabulary],
        dtype=bool,
    )
    kinase_count = np.bincount(
        bundle.kinase_site_edge_index[1], minlength=len(targets)
    ).astype(np.int64)
    if kinase_count.max(initial=0) > args.maximum_kinases_per_site:
        raise ValueError(
            "maximum_kinases_per_site would truncate the CoPhee prior"
        )
    chunks = build_site_chunks(
        parent_protein_index,
        parent_protein_mask,
        bundle.kinase_site_edge_index,
        chunk_size=args.site_chunk_size,
        maximum_kinases_per_site=args.maximum_kinases_per_site,
    )

    cancer_labels = canonical_cancer_labels(
        read_metadata_column(
            args.sample_metadata,
            development_ids,
            args.sample_id_column,
            args.cancer_column,
        )
    )
    cancer_vocabulary, cancer_index = fit_cancer_vocabulary(
        cancer_labels[train_index], cancer_labels
    )
    if bool((cancer_index < 0).any()):
        missing = sorted(set(cancer_labels[cancer_index < 0].tolist()))
        raise ValueError(f"validation cancers absent from training vocabulary: {missing}")
    cancer_counts = pd.DataFrame(
        {
            "cancer": cancer_vocabulary,
            "training_samples": np.bincount(
                cancer_index[train_index], minlength=len(cancer_vocabulary)
            ),
            "validation_samples": np.bincount(
                cancer_index[validation_index], minlength=len(cancer_vocabulary)
            ),
        }
    )
    cancer_counts.to_csv(
        output / "tables" / "cancer_sample_counts.tsv", sep="\t", index=False
    )

    (
        esm2_embedding,
        esm2_mask,
        esm2_local_embedding,
        esm2_local_mask,
        esm2_audit,
    ) = load_esm2_site_prior(
        args.esm2_prior_npz, targets
    )
    esm3_embedding, esm3_mask, esm3_audit = load_esm3_site_prior(
        args.esm3_prior_npz, targets
    )
    if (
        esm3_audit.get("enabled")
        and bool(esm3_audit.get("source_summary", {}).get("capacity_test_only", False))
        and not args.capacity_preflight_batches
    ):
        raise ValueError("ESM-3 capacity placeholder cannot enter formal training")
    coverage = observed[train_index].mean(axis=0).astype(np.float32)
    high_coverage = np.flatnonzero(coverage >= args.high_coverage_threshold)
    top_coverage = np.argsort(-coverage)[: min(args.high_coverage_panel_size, len(targets))]
    pd.DataFrame(
        {
            "target": targets,
            "training_coverage": coverage,
            "kinase_count": kinase_count,
            "parent_protein_available": parent_protein_mask,
            "parent_rna_available": parent_rna_mask,
            "esm2_residue_available": (
                np.zeros(len(targets), dtype=bool)
                if esm2_mask is None
                else esm2_mask.any(axis=1)
            ),
            "esm2_matched_residue_count": (
                np.zeros(len(targets), dtype=np.int16)
                if esm2_mask is None
                else esm2_mask.sum(axis=1)
            ),
            "esm3_residue_available": (
                np.zeros(len(targets), dtype=bool)
                if esm3_mask is None
                else esm3_mask.any(axis=1)
            ),
        }
    ).to_csv(output / "tables" / "target_training_metadata.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "target": targets,
            "scaling": args.site_scaling,
            "training_observations": site_training_count,
            "training_lower_bound": site_minimum,
            "training_upper_bound": site_maximum,
            "training_scale_span": site_maximum - site_minimum,
        }
    ).to_csv(output / "tables" / "site_normalization.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "rna_feature": rna_frame.columns.astype(str),
            "input_transform": args.rna_input_transform,
            "training_observations": rna_observed[train_index].sum(axis=0),
            "training_center": rna_mean,
            "training_scale": rna_scale,
        }
    ).to_csv(output / "tables" / "rna_normalization.tsv", sep="\t", index=False)

    config = DecoderRetrievalConfig(
        n_rna=rna_input.shape[1],
        n_proteins=protein.shape[1],
        n_sites=len(targets),
        n_kinases=len(bundle.kinase_vocabulary),
        n_cancers=len(cancer_vocabulary),
        hidden=args.hidden,
        heads=args.heads,
        decoder_layers=2,
        hgt_layers=0,
        global_tokens=1,
        maximum_kinases_per_site=args.maximum_kinases_per_site,
        site_output_rank=args.operator_rank,
        dropout=args.dropout,
        cancer_adapter_rank=args.cancer_adapter_rank,
        esm2_dimension=int(esm2_audit["embedding_dimension"]),
        esm2_fusion=args.esm2_fusion,
        esm3_dimension=int(esm3_audit["embedding_dimension"]),
        modality_tokens=args.modality_tokens,
        phospho_latent_tokens=args.phospho_latent_tokens,
        transformer_layers=args.transformer_layers,
    )
    model = DecoderRetrievalModel(
        config,
        configuration=args.configuration,
        parent_protein_index=torch.from_numpy(parent_protein_index),
        parent_protein_mask=torch.from_numpy(parent_protein_mask),
        parent_rna_index=torch.from_numpy(parent_rna_index),
        parent_rna_mask=torch.from_numpy(parent_rna_mask),
        kinase_rna_index=torch.from_numpy(kinase_rna_index),
        kinase_rna_mask=torch.from_numpy(kinase_rna_mask),
        kinase_protein_index=torch.from_numpy(kinase_protein_index),
        kinase_protein_mask=torch.from_numpy(kinase_protein_mask),
        site_residue_index=torch.from_numpy(metadata["residue_index"]),
        site_position=torch.from_numpy(metadata["position"]),
        site_esm2_residue_embedding=(
            None if esm2_embedding is None else torch.from_numpy(esm2_embedding)
        ),
        site_esm2_residue_mask=(
            None if esm2_mask is None else torch.from_numpy(esm2_mask)
        ),
        site_esm2_local_embedding=(
            None
            if esm2_local_embedding is None
            else torch.from_numpy(esm2_local_embedding)
        ),
        site_esm2_local_mask=(
            None if esm2_local_mask is None else torch.from_numpy(esm2_local_mask)
        ),
        site_esm3_residue_embedding=(
            None if esm3_embedding is None else torch.from_numpy(esm3_embedding)
        ),
        site_esm3_residue_mask=(
            None if esm3_mask is None else torch.from_numpy(esm3_mask)
        ),
    )
    initial_target_coordinate_audit: dict[str, object] | None = None
    if args.initial_checkpoint is not None:
        checkpoint = torch.load(
            args.initial_checkpoint, map_location="cpu", weights_only=False
        )
        checkpoint_metadata = checkpoint.get("metadata", {})
        if checkpoint_metadata.get("architecture") != "latent_transformer_biology_operator":
            raise ValueError("initial checkpoint belongs to a different architecture")
        if checkpoint_metadata.get("configuration") != args.configuration:
            raise ValueError("initial checkpoint configuration differs from requested model")
        checkpoint_rna_transform = str(
            checkpoint.get("rna_input_transform", "sample_rank")
        )
        if checkpoint_rna_transform != args.rna_input_transform:
            raise ValueError(
                "initial checkpoint RNA transform differs from current input: "
                f"{checkpoint_rna_transform} != {args.rna_input_transform}"
            )
        if args.rna_input_transform == "feature_zscore":
            prior_rna_mean = np.asarray(checkpoint.get("rna_mean", []), dtype=np.float32)
            prior_rna_scale = np.asarray(checkpoint.get("rna_scale", []), dtype=np.float32)
            if (
                prior_rna_mean.shape != rna_mean.shape
                or prior_rna_scale.shape != rna_scale.shape
                or not np.allclose(prior_rna_mean, rna_mean, rtol=0.0, atol=1.0e-6)
                or not np.allclose(
                    prior_rna_scale, rna_scale, rtol=0.0, atol=1.0e-6
                )
            ) and not args.allow_initial_rna_scaler_mismatch:
                raise ValueError("initial checkpoint RNA scaler differs from current training split")
        prior_normalization = checkpoint.get("normalization", {})
        prior_target_normalization = str(
            prior_normalization.get("method", "training_site_minmax")
        )
        target_coordinate_changed = (
            prior_target_normalization != current_target_normalization
        ) or bool(args.target_study_residual)
        prior_checkpoint = bool(checkpoint_metadata.get("esm2_sequence_prior", False))
        migrated_state, initial_target_coordinate_audit = prepare_initial_checkpoint_state(
            checkpoint["model_state"],
            target_coordinate_changed=target_coordinate_changed,
        )
        initial_target_coordinate_audit.update(
            {
                "prior_target_normalization": prior_target_normalization,
                "current_target_normalization": current_target_normalization,
            }
        )
        incompatible = model.load_state_dict(migrated_state, strict=False)
        invalid_missing = [
            name
            for name in incompatible.missing_keys
            if not name.startswith("site_esm2_")
            and not name.startswith("site_esm3_")
            and not (
                target_coordinate_changed
                and name in OUTPUT_CALIBRATION_STATE_NAMES
            )
            and not (
                target_coordinate_changed
                and (
                    name in OUTPUT_COORDINATE_EXACT_NAMES
                    or any(
                        name.startswith(prefix)
                        for prefix in OUTPUT_COORDINATE_PREFIXES
                    )
                )
            )
            and not (
                config.esm2_dimension
                and not prior_checkpoint
                and name.startswith("esm2_")
            )
            and not (config.esm3_dimension and name.startswith("esm3_"))
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise ValueError(
                "检查点迁移到当前序列先验模型时出现非序列层差异: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
        if target_coordinate_changed:
            if not torch.equal(
                model.site_bias.weight.detach(),
                torch.zeros_like(model.site_bias.weight),
            ):
                raise RuntimeError("site bias was not reset for the new target coordinate")
            if not torch.equal(
                model.site_output_scale.detach(),
                torch.ones_like(model.site_output_scale),
            ) or not torch.equal(
                model.site_output_shift.detach(),
                torch.zeros_like(model.site_output_shift),
            ):
                raise RuntimeError(
                    "site output calibration was not reset for the new target coordinate"
                )
        else:
            if (
                "site_minimum" not in prior_normalization
                or "site_maximum" not in prior_normalization
            ):
                raise ValueError("initial checkpoint lacks per-site normalization bounds")
            prior_minimum = np.asarray(
                prior_normalization["site_minimum"], dtype=np.float32
            )
            prior_maximum = np.asarray(
                prior_normalization["site_maximum"], dtype=np.float32
            )
            if (
                prior_minimum.shape != site_minimum.shape
                or prior_maximum.shape != site_maximum.shape
            ):
                raise ValueError("initial checkpoint normalization differs from current targets")
            prior_span = np.where(
                prior_maximum - prior_minimum > 1.0e-8,
                prior_maximum - prior_minimum,
                1.0,
            ).astype(np.float32)
            current_span = np.where(
                site_maximum - site_minimum > 1.0e-8,
                site_maximum - site_minimum,
                1.0,
            ).astype(np.float32)
            coordinate_scale = torch.from_numpy(prior_span / current_span)
            coordinate_shift = torch.from_numpy(
                (prior_minimum - site_minimum) / current_span
            )
            with torch.no_grad():
                model.site_output_shift.copy_(
                    model.site_output_shift * coordinate_scale + coordinate_shift
                )
                model.site_output_scale.mul_(coordinate_scale)
            initial_target_coordinate_audit[
                "old_site_coordinate_conversion_applied"
            ] = True
    pan_trunk_load_audit: dict[str, int] | None = None
    if args.training_stage in PAN_CHECKPOINT_STAGES:
        checkpoint = pan_checkpoint_data
        if checkpoint is None:
            raise RuntimeError("pan checkpoint was not loaded")
        checkpoint_metadata = checkpoint.get("metadata", {})
        if checkpoint_metadata.get("architecture") != "latent_transformer_biology_operator":
            raise ValueError("pan checkpoint belongs to a different architecture")
        if checkpoint_metadata.get("configuration") != args.configuration:
            raise ValueError("pan checkpoint configuration differs from requested model")
        checkpoint_targets = np.asarray(checkpoint.get("targets", []), dtype=str)
        if not np.array_equal(checkpoint_targets, np.asarray(targets, dtype=str)):
            raise ValueError("pan checkpoint phosphosite vocabulary differs from current data")
        pan_trunk_load_audit = load_pan_trunk_for_cancer_adapter(
            model, checkpoint["model_state"]
        )
        configure_posttraining_parameters(model, args.training_stage)

    # Reset the stochastic training stream after configuration-specific module
    # construction so controlled configurations receive identical dropout draws.
    torch.manual_seed(args.seed + 1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 1000)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but CUDA is unavailable")
    model.to(device)
    autocast_enabled = args.precision == "bfloat16" and device.type == "cuda"

    if args.export_initial_checkpoint_only:
        initial_path = output / "models" / "decoder_retrieval_initial.pt"
        torch.save(
            {
                "model_state": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "metadata": model.checkpoint_metadata(),
                "targets": targets,
                "rna_input_transform": args.rna_input_transform,
                "rna_mean": rna_mean,
                "rna_scale": rna_scale,
                "esm2_prior_audit": esm2_audit,
                "esm3_prior_audit": esm3_audit,
                "normalization": {
                    "method": current_target_normalization,
                    "site_minimum": site_minimum,
                    "site_maximum": site_maximum,
                    "validation_clipped": False,
                    "study_residual": bool(args.target_study_residual),
                    "study_center": (
                        study_center if args.target_study_residual else None
                    ),
                },
                "initial_target_coordinate_audit": initial_target_coordinate_audit,
                "sealed_phosphosite_rows_loaded": False,
            },
            initial_path,
        )
        write_json(
            output / "reports" / "initial_checkpoint.json",
            {
                "status": "complete",
                "checkpoint": str(initial_path),
                "architecture": "latent_transformer_biology_operator",
                "esm2_prior_audit": esm2_audit,
                "esm3_prior_audit": esm3_audit,
                "target_normalization": current_target_normalization,
                "initial_target_coordinate_audit": initial_target_coordinate_audit,
                "sealed_phosphosite_rows_loaded": False,
            },
        )
        return 0

    if args.esm2_warmup_updates and not config.esm2_dimension:
        raise ValueError("ESM-2 warm-up requires --esm2-prior-npz")
    if args.esm3_warmup_updates and not config.esm3_dimension:
        raise ValueError("ESM-3 warm-up requires --esm3-prior-npz")
    if args.esm2_warmup_updates and args.esm3_warmup_updates:
        raise ValueError("only one sequence-prior warm-up may be active")
    if args.esm2_warmup_updates and args.training_stage != "pan":
        raise ValueError("ESM-2 warm-up is supported only for pan training")
    if args.esm3_warmup_updates and args.training_stage != "pan":
        raise ValueError("ESM-3 warm-up is supported only for pan training")
    base_trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    trainable_parameter_names = sorted(base_trainable_names)
    frozen_parameter_names = sorted(
        name for name, parameter in model.named_parameters() if not parameter.requires_grad
    )
    trainable_scalar_count = int(
        sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name in base_trainable_names
        )
    )
    if args.training_stage == "cancer_adapter":
        unexpected_trainable = [
            name
            for name in trainable_parameter_names
            if not name.startswith("cancer_")
        ]
        if not trainable_parameter_names or unexpected_trainable:
            raise RuntimeError(
                "cancer-adapter training must update only cancer_ parameters; "
                f"unexpected={unexpected_trainable}"
            )
    if args.training_stage == "external_full_finetune":
        unexpected_trainable = [
            name for name in trainable_parameter_names if name.startswith("cancer_")
        ]
        missing_trainable = [
            name
            for name, _ in model.named_parameters()
            if not name.startswith("cancer_") and name not in base_trainable_names
        ]
        if not trainable_parameter_names or unexpected_trainable or missing_trainable:
            raise RuntimeError(
                "external full-model fine-tuning must update every non-cancer "
                "parameter and freeze every cancer_ parameter; "
                f"unexpected={unexpected_trainable}, missing={missing_trainable}"
            )
    sequence_warmup_updates = args.esm3_warmup_updates or args.esm2_warmup_updates
    sequence_warmup_prefix = "esm3_" if args.esm3_warmup_updates else "esm2_"
    sequence_warmup_learning_rate = (
        args.esm3_warmup_learning_rate
        if args.esm3_warmup_updates
        else args.esm2_warmup_learning_rate
    )
    warmup_active = bool(sequence_warmup_updates)
    trainable = configure_esm2_warmup(
        model,
        base_trainable_names,
        enabled=warmup_active,
        parameter_prefix=sequence_warmup_prefix,
    )
    if warmup_active:
        sequence_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name in base_trainable_names and name.startswith(sequence_warmup_prefix)
        ]
        shared_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name in base_trainable_names and not name.startswith(sequence_warmup_prefix)
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": sequence_parameters, "lr": sequence_warmup_learning_rate},
                {"params": shared_parameters, "lr": 0.0},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable, lr=args.learning_rate, weight_decay=args.weight_decay
        )

    tensors = {
        "rna": torch.from_numpy(rna_input),
        "protein": torch.from_numpy(protein),
        "reliability": torch.from_numpy(reliability),
        "target": torch.from_numpy(normalized_target),
        "mask": torch.from_numpy(observed),
        "cancer": torch.from_numpy(cancer_index.astype(np.int64)),
    }
    chunk_runtime = []
    for chunk in chunks:
        site_array = np.asarray(chunk.site_index, dtype=np.int64)
        kinase_index, kinase_mask = padded_kinase_mapping(
            bundle.kinase_site_edge_index,
            site_array,
            maximum_kinases=args.maximum_kinases_per_site,
        )
        chunk_runtime.append(
            (
                torch.from_numpy(site_array),
                torch.from_numpy(kinase_index),
                torch.from_numpy(kinase_mask),
            )
        )

    pretrain_tensors = None
    pretrain_probability = None
    pretrain_sample_ids: np.ndarray | None = None
    if args.conservative_pretrain_h5 is not None:
        with h5py.File(args.conservative_pretrain_h5, "r") as handle:
            required = {
                "sample_id",
                "source_group",
                "study_balancing_weight",
                "rna_sample_rank",
                "phosphosite_log_scale_sample_centered",
                "phosphosite_observed",
                "rna_vocabulary",
                "protein_vocabulary",
                "target_vocabulary",
            }
            if not required.issubset(handle.keys()):
                missing = sorted(required - set(handle.keys()))
                raise ValueError(f"conservative pretraining HDF5 lacks {missing}")
            pretrain_sample_ids = decode_h5_text(handle["sample_id"][:])
            all_pretrain_sample_ids = pretrain_sample_ids.copy()
            pretrain_source = decode_h5_text(handle["source_group"][:])
            pretrain_rna_vocabulary = decode_h5_text(handle["rna_vocabulary"][:])
            pretrain_protein_vocabulary = decode_h5_text(
                handle["protein_vocabulary"][:]
            )
            pretrain_target_vocabulary = decode_h5_text(
                handle["target_vocabulary"][:]
            )
            if not np.array_equal(
                np.char.upper(pretrain_rna_vocabulary),
                np.char.upper(np.asarray(rna_frame.columns, dtype=str)),
            ):
                raise ValueError("pretraining RNA vocabulary differs from development data")
            vocabulary_reindex(
                pretrain_protein_vocabulary,
                protein_frame.columns.astype(str).tolist(),
            )
            if not np.array_equal(
                pretrain_target_vocabulary, np.asarray(targets, dtype=str)
            ):
                raise ValueError(
                    "pretraining phosphosite vocabulary differs from development data"
                )

            train_id_set = set(map(str, split.train_ids.tolist()))
            cptac_rows = np.asarray(
                [sample in train_id_set for sample in pretrain_sample_ids], dtype=bool
            )
            if int(cptac_rows.sum()) != len(split.train_ids):
                raise ValueError("pretraining HDF5 does not contain the exact fixed 916")
            if set(pretrain_sample_ids[cptac_rows]) != train_id_set:
                raise ValueError("pretraining fixed-916 identifiers differ from split")
            if set(pretrain_sample_ids) & set(map(str, split.validation_ids.tolist())):
                raise ValueError("pretraining HDF5 contains fixed validation patients")
            if set(pretrain_sample_ids) & set(map(str, split.sealed_ids.tolist())):
                raise ValueError("pretraining HDF5 contains sealed patients")

            if args.pretrain_scope == "cptac916":
                selected = np.flatnonzero(cptac_rows)
            else:
                selected = np.arange(len(pretrain_sample_ids), dtype=np.int64)
                if (
                    len(selected) != 2067
                    or int((pretrain_source == "external_public").sum()) != 1151
                ):
                    raise ValueError("combined pretraining scope is not 916+1151")
            pretrain_sample_ids = pretrain_sample_ids[selected]
            pretrain_rna = (
                handle["rna_sample_rank"][selected].astype(np.float32) * 2.0 - 1.0
            )
            pretrain_phosphosite = handle[
                "phosphosite_log_scale_sample_centered"
            ][selected].astype(np.float32)
            pretrain_observed = handle["phosphosite_observed"][selected].astype(bool)
            if not np.array_equal(
                pretrain_observed, np.isfinite(pretrain_phosphosite)
            ):
                raise ValueError("pretraining phosphosite mask differs from finite values")
            pretrain_target, normalized_mask = apply_site_profile_scaler(
                pretrain_phosphosite,
                site_minimum,
                site_maximum,
                scaling=args.site_scaling,
            )
            if not np.array_equal(pretrain_observed, normalized_mask):
                raise ValueError("pretraining normalization changed the mask")
            weights = handle["study_balancing_weight"][selected].astype(np.float64)
            if not np.isfinite(weights).all() or bool((weights <= 0).any()):
                raise ValueError("pretraining study weights are invalid")
            pretrain_probability = weights / weights.sum()

        pretrain_tensors = {
            "rna": torch.from_numpy(pretrain_rna),
            "protein": torch.zeros((len(pretrain_rna), protein.shape[1]), dtype=torch.float32),
            "reliability": torch.zeros((len(pretrain_rna), protein.shape[1]), dtype=torch.float32),
            "target": torch.from_numpy(pretrain_target),
            "mask": torch.from_numpy(pretrain_observed),
        }
        if args.conservative_pretrain_protein_npz is not None:
            with np.load(args.conservative_pretrain_protein_npz, allow_pickle=False) as archive:
                required = {"sample_id", "protein_names", "prediction_z", "reliability"}
                if not required.issubset(archive.files):
                    raise ValueError("pretraining protein archive lacks required arrays")
                saved_ids = archive["sample_id"].astype(str)
                saved_proteins = archive["protein_names"].astype(str)
                if not np.array_equal(saved_ids, all_pretrain_sample_ids):
                    raise ValueError("pretraining protein sample order differs from HDF5")
                protein_order = vocabulary_reindex(
                    saved_proteins, protein_frame.columns.astype(str).tolist()
                )
                selected_prediction = np.asarray(
                    archive["prediction_z"][selected][:, protein_order],
                    dtype=np.float32,
                )
                saved_reliability = np.asarray(archive["reliability"], dtype=np.float32)
                if saved_reliability.ndim == 1:
                    saved_reliability = saved_reliability[protein_order]
                    saved_reliability = np.broadcast_to(
                        saved_reliability[None, :], selected_prediction.shape
                    ).copy()
                else:
                    saved_reliability = saved_reliability[selected][:, protein_order]
                if selected_prediction.shape != pretrain_tensors["protein"].shape:
                    raise ValueError("pretraining protein prediction shape differs")
                if not np.isfinite(selected_prediction).all() or not np.isfinite(saved_reliability).all():
                    raise ValueError("pretraining protein archive contains non-finite values")
                if np.any((saved_reliability < 0) | (saved_reliability > 1)):
                    raise ValueError("pretraining protein reliability lies outside [0, 1]")
                pretrain_tensors["protein"] = torch.from_numpy(selected_prediction)
                pretrain_tensors["reliability"] = torch.from_numpy(saved_reliability)

    if args.validate_inputs_only:
        write_json(
            output / "reports" / "input_validation.json",
            {
                "status": "complete",
                "split_sizes": [n_train, len(validation_index), len(split.sealed_ids)],
                "n_sites": len(targets),
                "n_chunks": len(chunks),
                "n_high_coverage_sites": int(len(high_coverage)),
                "rna_input_transform": args.rna_input_transform,
                "rna_scaler_source": (
                    "pan_checkpoint"
                    if pan_checkpoint_data is not None
                    else "selection_train"
                ),
                "rna_scaler_fit_samples": (
                    None
                    if pan_checkpoint_data is not None
                    else int(len(train_index))
                ),
                "adaptation_training_samples": int(len(train_index)),
                "rna_observed_fraction": float(rna_observed.mean()),
                "rna_missing_after_transform": int(
                    np.count_nonzero(~np.isfinite(rna_input))
                ),
                "maximum_observed_kinases_per_site": int(kinase_count.max(initial=0)),
                "pretraining_samples": (
                    int(len(pretrain_sample_ids))
                    if pretrain_sample_ids is not None
                    else 0
                ),
                "trainable_parameter_names": trainable_parameter_names,
                "trainable_parameter_count": int(len(trainable_parameter_names)),
                "trainable_scalar_count": trainable_scalar_count,
                "frozen_parameter_names": frozen_parameter_names,
                "frozen_parameter_count": int(len(frozen_parameter_names)),
                "esm2_prior_audit": esm2_audit,
                "sealed_phosphosite_rows_loaded": False,
            },
        )
        return 0

    def predict(
        row_index: np.ndarray,
        *,
        enable_cancer_adapter: bool | None = None,
    ) -> np.ndarray:
        adapter_enabled = (
            args.training_stage == "cancer_adapter"
            if enable_cancer_adapter is None
            else bool(enable_cancer_adapter)
        )
        model.eval()
        result = np.empty((len(row_index), len(targets)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(row_index), args.evaluation_batch_size):
                rows = row_index[start : start + args.evaluation_batch_size]
                ids = torch.from_numpy(rows)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    context = model.encode_context(
                        tensors["rna"][ids].to(device),
                        tensors["protein"][ids].to(device),
                        tensors["reliability"][ids].to(device),
                    )
                    for sites, kinase_index, kinase_mask in chunk_runtime:
                        output_value = model.decode_sites(
                            context,
                            sites.to(device),
                            kinase_index.to(device),
                            kinase_mask.to(device),
                            cancer_index=tensors["cancer"][ids].to(device),
                            enable_cancer_adapter=adapter_enabled,
                        )["normalized_profile"]
                        result[start : start + len(rows), sites.numpy()] = (
                            output_value.float().cpu().numpy()
                        )
        return result

    external_full_finetune_zero_shot_prediction = None
    if args.training_stage == "external_full_finetune":
        # The pan trunk changes during this control, so its zero-shot output must
        # be captured before the first optimizer update.  The fixed validation
        # labels are not consulted by this forward pass.
        external_full_finetune_zero_shot_prediction = predict(
            validation_index,
            enable_cancer_adapter=False,
        )

    template = np.nanmean(
        np.where(observed[train_index], normalized_target[train_index], np.nan), axis=0
    )
    template = np.nan_to_num(template, nan=0.5).astype(np.float32)
    validation_template = np.broadcast_to(
        template[None, :], (len(validation_index), len(targets))
    ).copy()
    template_cosine, _ = masked_batch_cosine(
        normalized_target[validation_index],
        validation_template,
        observed[validation_index],
        batch_size=8,
    )

    def checkpoint_payload(
        state: dict[str, torch.Tensor],
        epoch: int,
        metric: dict[str, float],
    ) -> dict[str, object]:
        checkpoint_metadata = model.checkpoint_metadata()
        checkpoint_metadata["prediction_domain"] = current_target_normalization
        return {
            "model_state": state,
            "metadata": checkpoint_metadata,
            "targets": targets,
            "esm2_prior_audit": esm2_audit,
            "esm3_prior_audit": esm3_audit,
            "rna_vocabulary": rna_frame.columns.astype(str).tolist(),
            "rna_input_transform": args.rna_input_transform,
            "rna_mean": rna_mean,
            "rna_scale": rna_scale,
            "protein_vocabulary": protein_frame.columns.astype(str).tolist(),
            "protein_mean": protein_mean,
            "protein_scale": protein_scale,
            "parent_protein_index": parent_protein_index,
            "parent_protein_mask": parent_protein_mask,
            "parent_rna_index": parent_rna_index,
            "parent_rna_mask": parent_rna_mask,
            "site_residue_index": metadata["residue_index"],
            "site_position": metadata["position"],
            "kinase_vocabulary": bundle.kinase_vocabulary,
            "kinase_rna_index": kinase_rna_index,
            "kinase_rna_mask": kinase_rna_mask,
            "kinase_protein_index": kinase_protein_index,
            "kinase_protein_mask": kinase_protein_mask,
            "cancer_vocabulary": cancer_vocabulary,
            "normalization": {
                "method": current_target_normalization,
                "site_minimum": site_minimum,
                "site_maximum": site_maximum,
                "validation_clipped": False,
                "study_residual": bool(args.target_study_residual),
                "study_center": (
                    study_center if args.target_study_residual else None
                ),
            },
            "initial_target_coordinate_audit": initial_target_coordinate_audit,
            "training_template": template,
            "best_epoch": epoch,
            "sealed_phosphosite_rows_loaded": False,
            **metric,
        }

    def profile_pearson_state_for_batch(
        context: dict[str, torch.Tensor],
        ids: torch.Tensor,
        target_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
        *,
        cancer_index: torch.Tensor | None,
        enable_cancer_adapter: bool,
    ) -> dict[str, torch.Tensor] | None:
        """Collect full-profile statistics while preserving chunked memory use.

        Decoder dropout must be identical in the detached statistics pass and
        the following gradient pass.  Saving and restoring the RNG state makes
        both passes use the same masks without changing later random draws.
        """
        if not args.profile_pearson_weight:
            return None
        statistics = new_profile_pearson_statistics(len(ids), device)
        decoder_rng_state = capture_torch_rng_state(device)
        try:
            with torch.no_grad():
                for sites, kinase_index, kinase_mask in chunk_runtime:
                    result = model.decode_sites(
                        context,
                        sites.to(device),
                        kinase_index.to(device),
                        kinase_mask.to(device),
                        cancer_index=cancer_index,
                        enable_cancer_adapter=enable_cancer_adapter,
                    )
                    target = target_tensor[ids][:, sites].to(device)
                    mask = mask_tensor[ids][:, sites].to(device)
                    update_profile_pearson_statistics(
                        statistics,
                        result["normalized_profile"],
                        target,
                        mask,
                    )
        finally:
            restore_torch_rng_state(decoder_rng_state, device)
        return finalize_profile_pearson_statistics(
            statistics,
            minimum_observations=args.profile_pearson_minimum_observations,
        )

    history: list[dict[str, object]] = []
    best_score = -math.inf
    best_state = None
    best_profile_spearman = -math.inf
    best_profile_spearman_state = None
    best_profile_spearman_epoch = 0
    best_site_spearman = -math.inf
    best_site_spearman_state = None
    best_site_spearman_epoch = 0
    best_balanced_spearman = -math.inf
    best_balanced_spearman_epoch = 0
    best_early_stopping_score = -math.inf
    zero_shot_validation_summary: dict[str, float] | None = None
    stale = 0
    updates = 0
    rng = np.random.default_rng(args.seed)
    pretrain_updates_completed = 0
    if args.pretrain_updates:
        if pretrain_tensors is None or pretrain_probability is None:
            raise RuntimeError("pretraining tensors were not constructed")
        for group in optimizer.param_groups:
            group["lr"] = args.pretrain_learning_rate
        model.train()
        for _ in range(args.pretrain_updates):
            rows = rng.choice(
                len(pretrain_sample_ids),
                size=args.sample_batch_size,
                replace=True,
                p=pretrain_probability,
            ).astype(np.int64)
            ids = torch.from_numpy(rows)
            optimizer.zero_grad(set_to_none=True)
            full_mask = pretrain_tensors["mask"][ids]
            full_site_count = (
                (full_mask.sum(dim=0) > 0).sum().to(device)
                if args.site_huber_weight
                else torch.ones((), device=device)
            )
            full_correlation_count = (
                (
                    full_mask.sum(dim=0)
                    >= args.correlation_min_observations
                ).sum().to(device)
                if args.site_pearson_weight
                else torch.ones((), device=device)
            )
            full_rank_count = (
                pairwise_rank_site_count(full_mask, rank_pair_offsets).to(device)
                if args.site_rank_weight
                else torch.ones((), device=device)
            )
            full_patient_observation_count = (
                full_mask.sum(dim=1).to(device=device, dtype=torch.float32)
                if args.profile_mse_weight
                else torch.ones(len(ids), device=device)
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                context = model.encode_context(
                    pretrain_tensors["rna"][ids].to(device),
                    pretrain_tensors["protein"][ids].to(device),
                    pretrain_tensors["reliability"][ids].to(device),
                )
                profile_state = profile_pearson_state_for_batch(
                    context,
                    ids,
                    pretrain_tensors["target"],
                    pretrain_tensors["mask"],
                    cancer_index=None,
                    enable_cancer_adapter=False,
                )
                context_gradients: dict[str, torch.Tensor] = {}
                for chunk_number, (sites, kinase_index, kinase_mask) in enumerate(
                    chunk_runtime
                ):
                    chunk_context, context_leaves = detach_trainable_context(context)
                    result = model.decode_sites(
                        chunk_context,
                        sites.to(device),
                        kinase_index.to(device),
                        kinase_mask.to(device),
                        enable_cancer_adapter=False,
                    )
                    target = pretrain_tensors["target"][ids][:, sites].to(device)
                    mask = pretrain_tensors["mask"][ids][:, sites].to(device)
                    zero_loss = torch.zeros((), dtype=torch.float32, device=device)
                    value_loss = (
                        masked_site_huber(
                            result["normalized_profile"],
                            target,
                            mask,
                            full_site_count,
                            delta=args.site_huber_delta,
                        )
                        if args.site_huber_weight
                        else zero_loss
                    )
                    correlation_loss = (
                        masked_site_pearson_loss(
                            result["normalized_profile"],
                            target,
                            mask,
                            full_correlation_count,
                            minimum_observations=args.correlation_min_observations,
                        )
                        if args.site_pearson_weight
                        else zero_loss
                    )
                    rank_loss = (
                        masked_site_pairwise_rank_loss(
                            result["normalized_profile"],
                            target,
                            mask,
                            full_rank_count,
                            offsets=rank_pair_offsets,
                            temperature=args.rank_temperature,
                            minimum_target_difference=(
                                args.rank_minimum_target_difference
                            ),
                        )
                        if args.site_rank_weight
                        else zero_loss
                    )
                    profile_surrogate = (
                        masked_profile_pearson_chunk_surrogate(
                            result["normalized_profile"],
                            target,
                            mask,
                            profile_state,
                        )
                        if profile_state is not None
                        else zero_loss
                    )
                    profile_mse_loss = (
                        masked_profile_mse_chunk(
                            result["normalized_profile"],
                            target,
                            mask,
                            full_patient_observation_count,
                        )
                        if args.profile_mse_weight
                        else zero_loss
                    )
                    loss = (
                        args.site_huber_weight * value_loss
                        + args.site_pearson_weight * correlation_loss
                        + args.site_rank_weight * rank_loss
                        + args.profile_pearson_weight * profile_surrogate
                        + args.profile_mse_weight * profile_mse_loss
                    )
                    loss.backward()
                    accumulate_context_gradients(context_gradients, context_leaves)
                backward_encoder_context(context, context_gradients)
            torch.nn.utils.clip_grad_norm_(
                trainable, 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            pretrain_updates_completed += 1
            updates += 1
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate

    finetune_updates_completed = 0
    esm2_warmup_updates_completed = 0
    joint_updates_completed = 0
    reached_finetune_budget = False
    last_validation_finetune_update = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_index.copy()
        rng.shuffle(order)
        losses: list[float] = []
        huber_losses: list[float] = []
        pearson_losses: list[float] = []
        profile_pearson_losses: list[float] = []
        profile_mse_losses: list[float] = []
        rank_losses: list[float] = []
        distill_losses: list[float] = []
        for start in range(0, len(order), args.sample_batch_size):
            rows = order[start : start + args.sample_batch_size]
            if args.drop_last_training_batch and len(rows) < args.sample_batch_size:
                break
            ids = torch.from_numpy(rows)
            optimizer.zero_grad(set_to_none=True)
            full_mask = tensors["mask"][ids]
            full_site_count = (
                (full_mask.sum(dim=0) > 0).sum().to(device)
                if args.site_huber_weight or args.training_stage == "cancer_adapter"
                else torch.ones((), device=device)
            )
            full_correlation_count = (
                (
                    full_mask.sum(dim=0)
                    >= args.correlation_min_observations
                ).sum().to(device)
                if args.site_pearson_weight
                else torch.ones((), device=device)
            )
            full_rank_count = (
                pairwise_rank_site_count(full_mask, rank_pair_offsets).to(device)
                if args.site_rank_weight
                else torch.ones((), device=device)
            )
            full_patient_observation_count = (
                full_mask.sum(dim=1).to(device=device, dtype=torch.float32)
                if args.profile_mse_weight
                else torch.ones(len(ids), device=device)
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                context = model.encode_context(
                    tensors["rna"][ids].to(device),
                    tensors["protein"][ids].to(device),
                    tensors["reliability"][ids].to(device),
                )
                cancer_index = tensors["cancer"][ids].to(device)
                profile_state = profile_pearson_state_for_batch(
                    context,
                    ids,
                    tensors["target"],
                    tensors["mask"],
                    cancer_index=cancer_index,
                    enable_cancer_adapter=args.training_stage == "cancer_adapter",
                )
                context_gradients: dict[str, torch.Tensor] = {}
                batch_loss = torch.zeros((), device=device)
                batch_huber = torch.zeros((), device=device)
                batch_pearson = torch.zeros((), device=device)
                batch_profile_pearson = (
                    profile_state["loss"].detach()
                    if profile_state is not None
                    else torch.zeros((), device=device)
                )
                batch_profile_mse = torch.zeros((), device=device)
                batch_rank = torch.zeros((), device=device)
                batch_distill = torch.zeros((), device=device)
                for chunk_number, (sites, kinase_index, kinase_mask) in enumerate(
                    chunk_runtime
                ):
                    chunk_context, context_leaves = detach_trainable_context(context)
                    result = model.decode_sites(
                        chunk_context,
                        sites.to(device),
                        kinase_index.to(device),
                        kinase_mask.to(device),
                        cancer_index=cancer_index,
                        enable_cancer_adapter=args.training_stage == "cancer_adapter",
                    )
                    target = tensors["target"][ids][:, sites].to(device)
                    mask = tensors["mask"][ids][:, sites].to(device)
                    zero_loss = torch.zeros((), dtype=torch.float32, device=device)
                    value_loss = (
                        masked_site_huber(
                            result["normalized_profile"],
                            target,
                            mask,
                            full_site_count,
                            delta=args.site_huber_delta,
                        )
                        if args.site_huber_weight
                        else zero_loss
                    )
                    correlation_loss = (
                        masked_site_pearson_loss(
                            result["normalized_profile"],
                            target,
                            mask,
                            full_correlation_count,
                            minimum_observations=args.correlation_min_observations,
                        )
                        if args.site_pearson_weight
                        else zero_loss
                    )
                    rank_loss = (
                        masked_site_pairwise_rank_loss(
                            result["normalized_profile"],
                            target,
                            mask,
                            full_rank_count,
                            offsets=rank_pair_offsets,
                            temperature=args.rank_temperature,
                            minimum_target_difference=(
                                args.rank_minimum_target_difference
                            ),
                        )
                        if args.site_rank_weight
                        else zero_loss
                    )
                    profile_surrogate = (
                        masked_profile_pearson_chunk_surrogate(
                            result["normalized_profile"],
                            target,
                            mask,
                            profile_state,
                        )
                        if profile_state is not None
                        else zero_loss
                    )
                    profile_mse_loss = (
                        masked_profile_mse_chunk(
                            result["normalized_profile"],
                            target,
                            mask,
                            full_patient_observation_count,
                        )
                        if args.profile_mse_weight
                        else zero_loss
                    )
                    reported_loss = (
                        args.site_huber_weight * value_loss
                        + args.site_pearson_weight * correlation_loss
                        + args.site_rank_weight * rank_loss
                        + args.profile_mse_weight * profile_mse_loss
                    )
                    loss = (
                        reported_loss
                        + args.profile_pearson_weight * profile_surrogate
                    )
                    if args.training_stage == "cancer_adapter":
                        distill = masked_site_huber(
                            result["normalized_profile"],
                            result["pan_normalized_profile"].detach(),
                            mask,
                            full_site_count,
                            delta=args.site_huber_delta,
                        )
                        reported_loss = (
                            reported_loss + args.adapter_distill_weight * distill
                        )
                        loss = loss + args.adapter_distill_weight * distill
                        batch_distill = batch_distill + distill.detach()
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError(
                            "non-finite loss before backward: "
                            f"chunk={chunk_number}, huber={float(value_loss.detach())}, "
                            f"pearson={float(correlation_loss.detach())}, "
                            f"profile_pearson={float(batch_profile_pearson)}, "
                            f"profile_mse={float(profile_mse_loss.detach())}, "
                            f"prediction_finite={bool(torch.isfinite(result['normalized_profile']).all())}, "
                            f"target_finite={bool(torch.isfinite(target).all())}"
                        )
                    loss.backward()
                    if args.training_stage == "cancer_adapter":
                        chunk_bad_gradients = nonfinite_gradient_names(model)
                        if chunk_bad_gradients:
                            prediction = result["normalized_profile"].detach().float()
                            raise RuntimeError(
                                "non-finite adapter gradients after site chunk: "
                                f"chunk={chunk_number}, sites={int(sites[0])}:{int(sites[-1])}, "
                                f"prediction_range=({float(prediction.min())}, "
                                f"{float(prediction.max())}), parameters="
                                + ", ".join(chunk_bad_gradients[:20])
                            )
                    accumulate_context_gradients(context_gradients, context_leaves)
                    batch_loss = batch_loss + reported_loss.detach()
                    batch_huber = batch_huber + value_loss.detach()
                    batch_pearson = batch_pearson + correlation_loss.detach()
                    batch_profile_mse = (
                        batch_profile_mse + profile_mse_loss.detach()
                    )
                    batch_rank = batch_rank + rank_loss.detach()
                batch_loss = (
                    batch_loss
                    + args.profile_pearson_weight * batch_profile_pearson
                )
                backward_encoder_context(context, context_gradients)
            if args.training_stage == "cancer_adapter" and args.adapter_parameter_weight:
                penalty = sum(parameter.square().mean() for parameter in trainable)
                (args.adapter_parameter_weight * penalty).backward()
            bad_gradients = nonfinite_gradient_names(model)
            if bad_gradients:
                raise RuntimeError(
                    "non-finite gradients before clipping: "
                    + ", ".join(bad_gradients[:20])
                )
            torch.nn.utils.clip_grad_norm_(trainable, 5.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            finetune_updates_completed += 1
            if warmup_active:
                esm2_warmup_updates_completed += 1
                if esm2_warmup_updates_completed >= sequence_warmup_updates:
                    warmup_active = False
                    trainable = configure_esm2_warmup(
                        model,
                        base_trainable_names,
                        enabled=False,
                        parameter_prefix=sequence_warmup_prefix,
                    )
                    joint_learning_rate = (
                        args.joint_learning_rate
                        if args.joint_learning_rate is not None
                        else args.learning_rate
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = joint_learning_rate
            else:
                joint_updates_completed += 1
            losses.append(float(batch_loss))
            huber_losses.append(float(batch_huber))
            pearson_losses.append(float(batch_pearson))
            profile_pearson_losses.append(float(batch_profile_pearson))
            profile_mse_losses.append(float(batch_profile_mse))
            rank_losses.append(float(batch_rank))
            distill_losses.append(float(batch_distill))
            if args.capacity_preflight_batches and updates >= args.capacity_preflight_batches:
                write_json(
                    output / "reports" / "capacity_preflight.json",
                    {
                        "status": "complete",
                        "architecture": "latent_transformer_biology_operator",
                        "sample_batch_size": args.sample_batch_size,
                        "site_chunk_size": args.site_chunk_size,
                        "training_loss": float(losses[-1]),
                        "training_site_equal_huber": float(huber_losses[-1]),
                        "training_cross_patient_pearson_loss": float(
                            pearson_losses[-1]
                        ),
                        "training_within_patient_pearson_loss": float(
                            profile_pearson_losses[-1]
                        ),
                        "training_patient_equal_masked_mse": float(
                            profile_mse_losses[-1]
                        ),
                        "training_pairwise_rank_loss": float(rank_losses[-1]),
                        "loss_finite": bool(np.isfinite(losses[-1])),
                        "elapsed_seconds": time.time() - started,
                        "peak_cuda_memory_allocated_gib": (
                            torch.cuda.max_memory_allocated(device) / 1024**3
                            if device.type == "cuda"
                            else 0.0
                        ),
                        "peak_cuda_memory_reserved_gib": (
                            torch.cuda.max_memory_reserved(device) / 1024**3
                            if device.type == "cuda"
                            else 0.0
                        ),
                        "esm2_prior_audit": esm2_audit,
                        "esm3_prior_audit": esm3_audit,
                        "sealed_phosphosite_rows_loaded": False,
                    },
                )
                return 0
            if (
                args.maximum_finetune_updates
                and finetune_updates_completed >= args.maximum_finetune_updates
            ):
                reached_finetune_budget = True
                break

        if args.validation_only_at_finetune_end and not reached_finetune_budget:
            continue
        if (
            args.validation_interval_updates
            and not reached_finetune_budget
            and finetune_updates_completed - last_validation_finetune_update
            < args.validation_interval_updates
        ):
            continue
        last_validation_finetune_update = finetune_updates_completed

        prediction = predict(validation_index)
        prediction_reconstructed = prediction
        if args.target_study_residual:
            prediction_reconstructed = (
                prediction + study_center[validation_index]
            ).astype(np.float32)
        validation_target = normalized_target[validation_index]
        validation_mask = observed[validation_index]
        score, batch_cosines = masked_batch_cosine(
            validation_target,
            prediction,
            validation_mask,
            batch_size=8,
        )
        high_score = float("nan")
        if len(high_coverage):
            high_score, _ = masked_batch_cosine(
                validation_target[:, high_coverage],
                prediction[:, high_coverage],
                validation_mask[:, high_coverage],
                batch_size=8,
            )
        top_score, _ = masked_batch_cosine(
            validation_target[:, top_coverage],
            prediction[:, top_coverage],
            validation_mask[:, top_coverage],
            batch_size=8,
        )
        valid_prediction = prediction[validation_mask]
        valid_target = validation_target[validation_mask]
        validation_mse = float(np.mean(np.square(valid_prediction - valid_target)))
        validation_mae = float(np.mean(np.abs(valid_prediction - valid_target)))
        site_metrics = per_site_metrics(
            validation_target, prediction, validation_mask, targets
        )
        sample_metrics = per_sample_metrics(
            validation_target,
            prediction,
            validation_mask,
            split.validation_ids,
            pearson_minimum_observations=(
                args.profile_pearson_minimum_observations
            ),
        )
        raw_sample_metrics = per_sample_metrics(
            phosphosite_raw[validation_index],
            prediction_reconstructed,
            validation_mask,
            split.validation_ids,
            pearson_minimum_observations=(
                args.profile_pearson_minimum_observations
            ),
        )
        target_coordinate_spearman = sample_metrics[
            "within_sample_spearman"
        ].to_numpy(np.float64)
        original_logscale_spearman = raw_sample_metrics[
            "within_sample_spearman"
        ].to_numpy(np.float64)
        if args.site_scaling == "sample_percentile_rank" and not np.allclose(
            target_coordinate_spearman,
            original_logscale_spearman,
            rtol=0.0,
            atol=1.0e-12,
            equal_nan=True,
        ):
            raise RuntimeError(
                "sample-percentile target changed validation within-patient ranks"
            )
        sample_metrics["within_sample_spearman_target_coordinate"] = (
            target_coordinate_spearman
        )
        sample_metrics["within_sample_spearman"] = original_logscale_spearman
        sample_metrics["within_sample_spearman_original_logscale"] = (
            original_logscale_spearman
        )
        validation_adapter_groups = cancer_labels[validation_index].astype(str)
        sample_metrics["adapter_group"] = validation_adapter_groups
        validation_patient_equal_mse = float(sample_metrics["mse"].mean())
        validation_patient_equal_mse_median = float(sample_metrics["mse"].median())
        median_site_spearman = float(
            pd.to_numeric(site_metrics["spearman"], errors="coerce").median()
        )
        if args.training_stage == "cancer_adapter":
            zero_shot_prediction = predict(
                validation_index,
                enable_cancer_adapter=False,
            )
            zero_shot_site_metrics = per_site_metrics(
                validation_target,
                zero_shot_prediction,
                validation_mask,
                targets,
            )
            zero_shot_sample_metrics = per_sample_metrics(
                validation_target,
                zero_shot_prediction,
                validation_mask,
                split.validation_ids,
                pearson_minimum_observations=(
                    args.profile_pearson_minimum_observations
                ),
            )
            zero_shot_raw_metrics = per_sample_metrics(
                phosphosite_raw[validation_index],
                zero_shot_prediction,
                validation_mask,
                split.validation_ids,
                pearson_minimum_observations=(
                    args.profile_pearson_minimum_observations
                ),
            )
            zero_shot_sample_metrics["within_sample_spearman"] = (
                zero_shot_raw_metrics["within_sample_spearman"].to_numpy(
                    np.float64
                )
            )
            zero_shot_sample_metrics[
                "within_sample_spearman_original_logscale"
            ] = zero_shot_sample_metrics["within_sample_spearman"]
            zero_shot_sample_metrics["adapter_group"] = (
                validation_adapter_groups
            )
            zero_shot_profile_spearman = float(
                zero_shot_sample_metrics["within_sample_spearman"].median()
            )
            zero_shot_patient_equal_mse = float(
                zero_shot_sample_metrics["mse"].mean()
            )
            zero_shot_site_spearman = float(
                pd.to_numeric(
                    zero_shot_site_metrics["spearman"], errors="coerce"
                ).median()
            )
            adapted_profile_spearman = float(
                sample_metrics["within_sample_spearman"].median()
            )
            zero_shot_validation_summary = {
                "zero_shot_per_sample_spearman_median": (
                    zero_shot_profile_spearman
                ),
                "adapted_per_sample_spearman_median": adapted_profile_spearman,
                "adaptation_gain_per_sample_spearman_median": (
                    adapted_profile_spearman - zero_shot_profile_spearman
                ),
                "zero_shot_patient_equal_mse": zero_shot_patient_equal_mse,
                "adapted_patient_equal_mse": validation_patient_equal_mse,
                "adaptation_reduction_patient_equal_mse": (
                    zero_shot_patient_equal_mse - validation_patient_equal_mse
                ),
                "zero_shot_per_site_spearman_median": zero_shot_site_spearman,
                "adapted_per_site_spearman_median": median_site_spearman,
                "adaptation_gain_per_site_spearman_median": (
                    median_site_spearman - zero_shot_site_spearman
                ),
            }
            paired_sample_metrics = pd.DataFrame(
                {
                    "sample_id": split.validation_ids,
                    "adapter_group": validation_adapter_groups,
                    "zero_shot_within_sample_spearman": (
                        zero_shot_sample_metrics[
                            "within_sample_spearman"
                        ].to_numpy(np.float64)
                    ),
                    "adapted_within_sample_spearman": sample_metrics[
                        "within_sample_spearman"
                    ].to_numpy(np.float64),
                    "zero_shot_mse": zero_shot_sample_metrics["mse"].to_numpy(
                        np.float64
                    ),
                    "adapted_mse": sample_metrics["mse"].to_numpy(np.float64),
                }
            )
            paired_sample_metrics["spearman_gain"] = (
                paired_sample_metrics["adapted_within_sample_spearman"]
                - paired_sample_metrics["zero_shot_within_sample_spearman"]
            )
            paired_sample_metrics["mse_reduction"] = (
                paired_sample_metrics["zero_shot_mse"]
                - paired_sample_metrics["adapted_mse"]
            )
            cohort_comparison_rows: list[dict[str, float | int | str]] = []
            for adapter_group in sorted(set(validation_adapter_groups.tolist())):
                cohort_mask = validation_adapter_groups == adapter_group
                cohort_adapted_site_metrics = per_site_metrics(
                    validation_target[cohort_mask],
                    prediction[cohort_mask],
                    validation_mask[cohort_mask],
                    targets,
                )
                cohort_zero_shot_site_metrics = per_site_metrics(
                    validation_target[cohort_mask],
                    zero_shot_prediction[cohort_mask],
                    validation_mask[cohort_mask],
                    targets,
                )
                cohort_paired = paired_sample_metrics.loc[cohort_mask]
                cohort_adapted_site_spearman = float(
                    pd.to_numeric(
                        cohort_adapted_site_metrics["spearman"],
                        errors="coerce",
                    ).median()
                )
                cohort_zero_shot_site_spearman = float(
                    pd.to_numeric(
                        cohort_zero_shot_site_metrics["spearman"],
                        errors="coerce",
                    ).median()
                )
                cohort_comparison_rows.append(
                    {
                        "adapter_group": adapter_group,
                        "n_test_samples": int(cohort_mask.sum()),
                        "zero_shot_per_sample_spearman_median": float(
                            cohort_paired[
                                "zero_shot_within_sample_spearman"
                            ].median()
                        ),
                        "adapted_per_sample_spearman_median": float(
                            cohort_paired[
                                "adapted_within_sample_spearman"
                            ].median()
                        ),
                        "adaptation_gain_per_sample_spearman_median": float(
                            cohort_paired["adapted_within_sample_spearman"].median()
                            - cohort_paired[
                                "zero_shot_within_sample_spearman"
                            ].median()
                        ),
                        "zero_shot_patient_equal_mse": float(
                            cohort_paired["zero_shot_mse"].mean()
                        ),
                        "adapted_patient_equal_mse": float(
                            cohort_paired["adapted_mse"].mean()
                        ),
                        "adaptation_reduction_patient_equal_mse": float(
                            cohort_paired["zero_shot_mse"].mean()
                            - cohort_paired["adapted_mse"].mean()
                        ),
                        "zero_shot_per_site_spearman_median": (
                            cohort_zero_shot_site_spearman
                        ),
                        "adapted_per_site_spearman_median": (
                            cohort_adapted_site_spearman
                        ),
                        "adaptation_gain_per_site_spearman_median": (
                            cohort_adapted_site_spearman
                            - cohort_zero_shot_site_spearman
                        ),
                    }
                )
            zero_shot_site_metrics.to_csv(
                output / "tables/validation_zero_shot_per_site.tsv",
                sep="\t",
                index=False,
            )
            zero_shot_sample_metrics.to_csv(
                output / "tables/validation_zero_shot_per_sample.tsv",
                sep="\t",
                index=False,
            )
            paired_sample_metrics.to_csv(
                output / "tables/validation_zero_shot_vs_adapter_per_sample.tsv",
                sep="\t",
                index=False,
            )
            pd.DataFrame(cohort_comparison_rows).to_csv(
                output / "tables/validation_zero_shot_vs_adapter_by_group.tsv",
                sep="\t",
                index=False,
            )
            pd.DataFrame(
                zero_shot_prediction,
                index=split.validation_ids,
                columns=targets,
            ).to_parquet(
                output
                / "predictions"
                / f"validation_{prediction_coordinate_label}_zero_shot.parquet"
            )
            pd.DataFrame(
                prediction,
                index=split.validation_ids,
                columns=targets,
            ).to_parquet(
                output
                / "predictions"
                / f"validation_{prediction_coordinate_label}_adapter_final.parquet"
            )
            pd.DataFrame([zero_shot_validation_summary]).to_csv(
                output / "tables/validation_zero_shot_vs_adapter.tsv",
                sep="\t",
                index=False,
            )
        elif args.training_stage == "external_full_finetune":
            zero_shot_prediction = external_full_finetune_zero_shot_prediction
            if zero_shot_prediction is None:
                raise RuntimeError("external zero-shot prediction was not captured")
            zero_shot_site_metrics = per_site_metrics(
                validation_target,
                zero_shot_prediction,
                validation_mask,
                targets,
            )
            zero_shot_sample_metrics = per_sample_metrics(
                validation_target,
                zero_shot_prediction,
                validation_mask,
                split.validation_ids,
                pearson_minimum_observations=(
                    args.profile_pearson_minimum_observations
                ),
            )
            zero_shot_raw_metrics = per_sample_metrics(
                phosphosite_raw[validation_index],
                zero_shot_prediction,
                validation_mask,
                split.validation_ids,
                pearson_minimum_observations=(
                    args.profile_pearson_minimum_observations
                ),
            )
            zero_shot_sample_metrics["within_sample_spearman"] = (
                zero_shot_raw_metrics["within_sample_spearman"].to_numpy(
                    np.float64
                )
            )
            zero_shot_sample_metrics[
                "within_sample_spearman_original_logscale"
            ] = zero_shot_sample_metrics["within_sample_spearman"]
            zero_shot_sample_metrics["cohort"] = validation_adapter_groups
            sample_metrics["cohort"] = validation_adapter_groups
            zero_shot_profile_spearman = float(
                zero_shot_sample_metrics["within_sample_spearman"].median()
            )
            zero_shot_patient_equal_mse = float(
                zero_shot_sample_metrics["mse"].mean()
            )
            zero_shot_site_spearman = float(
                pd.to_numeric(
                    zero_shot_site_metrics["spearman"], errors="coerce"
                ).median()
            )
            final_profile_spearman = float(
                sample_metrics["within_sample_spearman"].median()
            )
            zero_shot_validation_summary = {
                "zero_shot_per_sample_spearman_median": (
                    zero_shot_profile_spearman
                ),
                "final_per_sample_spearman_median": final_profile_spearman,
                "full_finetune_gain_per_sample_spearman_median": (
                    final_profile_spearman - zero_shot_profile_spearman
                ),
                "zero_shot_patient_equal_mse": zero_shot_patient_equal_mse,
                "final_patient_equal_mse": validation_patient_equal_mse,
                "full_finetune_reduction_patient_equal_mse": (
                    zero_shot_patient_equal_mse - validation_patient_equal_mse
                ),
                "zero_shot_per_site_spearman_median": zero_shot_site_spearman,
                "final_per_site_spearman_median": median_site_spearman,
                "full_finetune_gain_per_site_spearman_median": (
                    median_site_spearman - zero_shot_site_spearman
                ),
            }
            paired_sample_metrics = pd.DataFrame(
                {
                    "sample_id": split.validation_ids,
                    "cohort": validation_adapter_groups,
                    "zero_shot_within_sample_spearman": (
                        zero_shot_sample_metrics[
                            "within_sample_spearman"
                        ].to_numpy(np.float64)
                    ),
                    "final_within_sample_spearman": sample_metrics[
                        "within_sample_spearman"
                    ].to_numpy(np.float64),
                    "zero_shot_mse": zero_shot_sample_metrics["mse"].to_numpy(
                        np.float64
                    ),
                    "final_mse": sample_metrics["mse"].to_numpy(np.float64),
                }
            )
            paired_sample_metrics["spearman_gain"] = (
                paired_sample_metrics["final_within_sample_spearman"]
                - paired_sample_metrics["zero_shot_within_sample_spearman"]
            )
            paired_sample_metrics["mse_reduction"] = (
                paired_sample_metrics["zero_shot_mse"]
                - paired_sample_metrics["final_mse"]
            )
            cohort_comparison_rows = []
            for cohort in sorted(set(validation_adapter_groups.tolist())):
                cohort_mask = validation_adapter_groups == cohort
                cohort_final_site_metrics = per_site_metrics(
                    validation_target[cohort_mask],
                    prediction[cohort_mask],
                    validation_mask[cohort_mask],
                    targets,
                )
                cohort_zero_shot_site_metrics = per_site_metrics(
                    validation_target[cohort_mask],
                    zero_shot_prediction[cohort_mask],
                    validation_mask[cohort_mask],
                    targets,
                )
                cohort_paired = paired_sample_metrics.loc[cohort_mask]
                cohort_final_site_spearman = float(
                    pd.to_numeric(
                        cohort_final_site_metrics["spearman"], errors="coerce"
                    ).median()
                )
                cohort_zero_shot_site_spearman = float(
                    pd.to_numeric(
                        cohort_zero_shot_site_metrics["spearman"],
                        errors="coerce",
                    ).median()
                )
                cohort_comparison_rows.append(
                    {
                        "cohort": cohort,
                        "n_test_samples": int(cohort_mask.sum()),
                        "zero_shot_per_sample_spearman_median": float(
                            cohort_paired[
                                "zero_shot_within_sample_spearman"
                            ].median()
                        ),
                        "final_per_sample_spearman_median": float(
                            cohort_paired[
                                "final_within_sample_spearman"
                            ].median()
                        ),
                        "full_finetune_gain_per_sample_spearman_median": float(
                            cohort_paired[
                                "final_within_sample_spearman"
                            ].median()
                            - cohort_paired[
                                "zero_shot_within_sample_spearman"
                            ].median()
                        ),
                        "zero_shot_patient_equal_mse": float(
                            cohort_paired["zero_shot_mse"].mean()
                        ),
                        "final_patient_equal_mse": float(
                            cohort_paired["final_mse"].mean()
                        ),
                        "full_finetune_reduction_patient_equal_mse": float(
                            cohort_paired["zero_shot_mse"].mean()
                            - cohort_paired["final_mse"].mean()
                        ),
                        "zero_shot_per_site_spearman_median": (
                            cohort_zero_shot_site_spearman
                        ),
                        "final_per_site_spearman_median": (
                            cohort_final_site_spearman
                        ),
                        "full_finetune_gain_per_site_spearman_median": (
                            cohort_final_site_spearman
                            - cohort_zero_shot_site_spearman
                        ),
                    }
                )
            zero_shot_site_metrics.to_csv(
                output / "tables/validation_zero_shot_per_site.tsv",
                sep="\t",
                index=False,
            )
            zero_shot_sample_metrics.to_csv(
                output / "tables/validation_zero_shot_per_sample.tsv",
                sep="\t",
                index=False,
            )
            site_metrics.to_csv(
                output / "tables/validation_full_finetune_final_per_site.tsv",
                sep="\t",
                index=False,
            )
            sample_metrics.to_csv(
                output / "tables/validation_full_finetune_final_per_sample.tsv",
                sep="\t",
                index=False,
            )
            paired_sample_metrics.to_csv(
                output
                / "tables/validation_zero_shot_vs_full_finetune_per_sample.tsv",
                sep="\t",
                index=False,
            )
            pd.DataFrame(cohort_comparison_rows).to_csv(
                output
                / "tables/validation_zero_shot_vs_full_finetune_by_cohort.tsv",
                sep="\t",
                index=False,
            )
            pd.DataFrame(
                zero_shot_prediction,
                index=split.validation_ids,
                columns=targets,
            ).to_parquet(
                output
                / "predictions"
                / f"validation_{prediction_coordinate_label}_zero_shot.parquet"
            )
            pd.DataFrame(
                prediction,
                index=split.validation_ids,
                columns=targets,
            ).to_parquet(
                output
                / "predictions"
                / (
                    f"validation_{prediction_coordinate_label}_"
                    "full_finetune_final.parquet"
                )
            )
            pd.DataFrame([zero_shot_validation_summary]).to_csv(
                output
                / "tables/validation_zero_shot_vs_full_finetune.tsv",
                sep="\t",
                index=False,
            )
        row = {
            "epoch": epoch,
            "optimizer_updates": updates,
            "esm2_warmup_updates": esm2_warmup_updates_completed,
            "joint_updates": joint_updates_completed,
            "training_phase": (
                f"{sequence_warmup_prefix.rstrip('_')}_warmup"
                if warmup_active
                else "joint"
            ),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_total_loss": float(np.mean(losses)),
            "train_site_equal_huber": float(np.mean(huber_losses)),
            "train_cross_patient_pearson_loss": float(np.mean(pearson_losses)),
            "train_within_patient_pearson_loss": float(
                np.mean(profile_pearson_losses)
            ),
            "train_patient_equal_masked_mse": float(np.mean(profile_mse_losses)),
            "train_pairwise_rank_loss": float(np.mean(rank_losses)),
            "train_adapter_distill": float(np.mean(distill_losses)),
            "validation_batch8_flattened_cosine": score,
            "validation_high_coverage_batch8_cosine": high_score,
            "validation_top_coverage_1000_batch8_cosine": top_score,
            "validation_per_sample_cosine_median": float(
                sample_metrics["cosine"].median()
            ),
            "validation_per_sample_spearman_median": float(
                sample_metrics["within_sample_spearman"].median()
            ),
            "validation_per_sample_spearman_target_coordinate_median": float(
                sample_metrics["within_sample_spearman_target_coordinate"].median()
            ),
            "validation_per_sample_pearson_median": float(
                sample_metrics["within_sample_pearson"].median()
            ),
            "validation_per_sample_pearson_mean": float(
                sample_metrics["within_sample_pearson"].mean()
            ),
            "validation_per_sample_pearson_effective_n": int(
                sample_metrics["within_sample_pearson"].notna().sum()
            ),
            "validation_per_site_spearman_median": median_site_spearman,
            "validation_mse": validation_mse,
            "validation_patient_equal_mse": validation_patient_equal_mse,
            "validation_patient_equal_mse_median": (
                validation_patient_equal_mse_median
            ),
            "validation_mae": validation_mae,
            "validation_template_batch8_cosine": template_cosine,
            "validation_cosine_gain_over_template": score - template_cosine,
            "sample_batch_size": args.sample_batch_size,
        }
        if zero_shot_validation_summary is not None:
            row.update(
                {
                    f"validation_{name}": value
                    for name, value in zero_shot_validation_summary.items()
                }
            )
        balanced_spearman = 0.5 * (
            float(row["validation_per_sample_spearman_median"])
            + float(row["validation_per_site_spearman_median"])
        )
        row["validation_balanced_spearman"] = balanced_spearman
        history.append(row)
        pd.DataFrame(history).to_csv(
            output / "logs" / "training_history.tsv", sep="\t", index=False
        )
        site_metrics.to_csv(
            output / f"tables/validation_per_site_epoch_{epoch:03d}.tsv",
            sep="\t",
            index=False,
        )
        sample_metrics.to_csv(
            output / f"tables/validation_per_sample_epoch_{epoch:03d}.tsv",
            sep="\t",
            index=False,
        )
        pd.DataFrame(
            {"batch_number": np.arange(len(batch_cosines)), "cosine": batch_cosines}
        ).to_csv(
            output / f"tables/validation_batch8_cosine_epoch_{epoch:03d}.tsv",
            sep="\t",
            index=False,
        )

        cosine_improved = bool(np.isfinite(score) and score > best_score)
        if cosine_improved:
            best_score = score
            best_state = copy.deepcopy(
                {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                }
            )
            site_metrics.to_csv(
                output / "tables" / "validation_per_site_best.tsv",
                sep="\t",
                index=False,
            )
            sample_metrics.to_csv(
                output / "tables" / "validation_per_sample_best.tsv",
                sep="\t",
                index=False,
            )
            pd.DataFrame(
                prediction_reconstructed, index=split.validation_ids, columns=targets
            ).to_parquet(
                output
                / "predictions"
                / f"validation_{prediction_coordinate_label}_best.parquet"
            )
            torch.save(
                checkpoint_payload(
                    best_state,
                    epoch,
                    {"best_validation_batch8_flattened_cosine": best_score},
                ),
                output / "models" / "decoder_retrieval_cosine_best.pt",
            )
        profile_spearman = row["validation_per_sample_spearman_median"]
        if np.isfinite(profile_spearman) and profile_spearman > best_profile_spearman:
            best_profile_spearman = float(profile_spearman)
            best_profile_spearman_epoch = int(epoch)
            best_profile_spearman_state = copy.deepcopy(
                {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                }
            )
            site_metrics.to_csv(
                output / "tables/validation_per_site_profile_spearman_best.tsv",
                sep="\t",
                index=False,
            )
            sample_metrics.to_csv(
                output / "tables/validation_per_sample_profile_spearman_best.tsv",
                sep="\t",
                index=False,
            )
            pd.DataFrame(
                prediction_reconstructed, index=split.validation_ids, columns=targets
            ).to_parquet(
                output
                / "predictions"
                / (
                    f"validation_{prediction_coordinate_label}_"
                    "profile_spearman_best.parquet"
                )
            )
            torch.save(
                checkpoint_payload(
                    best_profile_spearman_state,
                    epoch,
                    {
                        "best_validation_per_sample_spearman": (
                            best_profile_spearman
                        )
                    },
                ),
                output / "models" / "decoder_retrieval_profile_spearman_best.pt",
            )
        if np.isfinite(median_site_spearman) and median_site_spearman > best_site_spearman:
            best_site_spearman = float(median_site_spearman)
            best_site_spearman_epoch = int(epoch)
            site_metrics.to_csv(
                output / "tables/validation_per_site_site_spearman_best.tsv",
                sep="\t",
                index=False,
            )
            sample_metrics.to_csv(
                output / "tables/validation_per_sample_site_spearman_best.tsv",
                sep="\t",
                index=False,
            )
            pd.DataFrame(
                prediction, index=split.validation_ids, columns=targets
            ).to_parquet(
                output
                / "predictions"
                / (
                    f"validation_{prediction_coordinate_label}_"
                    "site_spearman_best.parquet"
                )
            )
            best_site_spearman_state = {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            }
            site_checkpoint = checkpoint_payload(
                best_site_spearman_state,
                epoch,
                {"best_validation_per_site_spearman": best_site_spearman},
            )
            torch.save(
                site_checkpoint,
                output / "models/decoder_retrieval_site_spearman_best.pt",
            )
            torch.save(
                site_checkpoint,
                output / "models/decoder_retrieval_best.pt",
            )
        if np.isfinite(balanced_spearman) and balanced_spearman > best_balanced_spearman:
            best_balanced_spearman = float(balanced_spearman)
            best_balanced_spearman_epoch = int(epoch)
            site_metrics.to_csv(
                output / "tables/validation_per_site_balanced_spearman_best.tsv",
                sep="\t",
                index=False,
            )
            sample_metrics.to_csv(
                output / "tables/validation_per_sample_balanced_spearman_best.tsv",
                sep="\t",
                index=False,
            )
            balanced_state = {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            }
            torch.save(
                checkpoint_payload(
                    balanced_state,
                    epoch,
                    {"best_validation_balanced_spearman": best_balanced_spearman},
                ),
                output / "models/decoder_retrieval_balanced_spearman_best.pt",
            )
        early_stopping_score = {
            "site_spearman": float(median_site_spearman),
            "profile_spearman": float(profile_spearman),
            "balanced_spearman": float(balanced_spearman),
            "cosine": float(score),
        }[args.early_stopping_metric]
        if np.isfinite(early_stopping_score) and early_stopping_score > best_early_stopping_score:
            best_early_stopping_score = early_stopping_score
            stale = 0
        else:
            stale += 1
        if reached_finetune_budget:
            break
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("no finite validation cosine checkpoint")
    if best_site_spearman_state is None or best_site_spearman_epoch < 1:
        raise RuntimeError("no finite cross-patient per-site Spearman checkpoint")
    if best_profile_spearman_state is None or best_profile_spearman_epoch < 1:
        raise RuntimeError("no finite within-sample Spearman checkpoint")
    saved_profile_table = pd.read_csv(
        output / "tables/validation_per_sample_profile_spearman_best.tsv",
        sep="\t",
    )
    saved_profile_median = float(
        saved_profile_table["within_sample_spearman"].median(skipna=True)
    )
    if not np.isclose(
        saved_profile_median,
        best_profile_spearman,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "within-sample Spearman best table does not match the selected checkpoint: "
            f"table={saved_profile_median}, selected={best_profile_spearman}"
        )
    pearson_history = [
        row
        for row in history
        if np.isfinite(float(row["validation_per_sample_pearson_median"]))
    ]
    if not pearson_history:
        raise RuntimeError("no finite validation within-patient Pearson result")
    best_pearson_row = max(
        pearson_history,
        key=lambda row: float(row["validation_per_sample_pearson_median"]),
    )
    write_json(
        output / "reports" / "run_summary.json",
        {
            "status": "complete",
            "architecture": "latent_transformer_biology_operator",
            "configuration": args.configuration,
            "training_stage": args.training_stage,
            "pan_checkpoint": (
                str(args.pan_checkpoint) if args.pan_checkpoint is not None else None
            ),
            "cancer_adapter_rank": args.cancer_adapter_rank,
            "adapter_distill_weight": args.adapter_distill_weight,
            "adapter_parameter_weight": args.adapter_parameter_weight,
            "pan_trunk_load_audit": pan_trunk_load_audit,
            "trainable_parameter_names": trainable_parameter_names,
            "trainable_parameter_count": int(len(trainable_parameter_names)),
            "trainable_scalar_count": trainable_scalar_count,
            "frozen_parameter_names": frozen_parameter_names,
            "frozen_parameter_count": int(len(frozen_parameter_names)),
            "rna_input_transform": args.rna_input_transform,
            "rna_scaler_source": (
                "pan_checkpoint"
                if pan_checkpoint_data is not None
                else "selection_train"
            ),
            "rna_scaler_fit_samples": (
                None
                if pan_checkpoint_data is not None
                else int(len(train_index))
            ),
            "adaptation_training_samples": int(len(train_index)),
            "rna_observed_fraction": float(rna_observed.mean()),
            "target_normalization": current_target_normalization,
            "prediction_domain": current_target_normalization,
            "prediction_file_coordinate": prediction_coordinate_label,
            "profile_spearman_target_coordinate": "original_logscale",
            "study_residual": bool(args.target_study_residual),
            "study_residual_tau": float(args.study_residual_tau),
            "study_residual_min_overlap": int(args.study_residual_min_overlap),
            "initial_target_coordinate_audit": initial_target_coordinate_audit,
            "training_objective": "_plus_".join(
                name
                for enabled, name in (
                    (args.site_huber_weight, "site_equal_huber"),
                    (args.site_pearson_weight, "cross_patient_pearson"),
                    (args.site_rank_weight, "pairwise_rank"),
                    (args.profile_pearson_weight, "within_patient_pearson"),
                    (args.profile_mse_weight, "patient_equal_masked_mse"),
                )
                if enabled
            ),
            "site_huber_weight": args.site_huber_weight,
            "site_huber_delta": args.site_huber_delta,
            "site_pearson_weight": args.site_pearson_weight,
            "site_rank_weight": args.site_rank_weight,
            "profile_pearson_weight": args.profile_pearson_weight,
            "profile_mse_weight": args.profile_mse_weight,
            "profile_pearson_minimum_observations": (
                args.profile_pearson_minimum_observations
            ),
            "drop_last_training_batch": args.drop_last_training_batch,
            "rank_temperature": args.rank_temperature,
            "rank_minimum_target_difference": (
                args.rank_minimum_target_difference
            ),
            "rank_pair_offsets": list(rank_pair_offsets),
            "correlation_min_observations": args.correlation_min_observations,
            "best_validation_batch8_flattened_cosine": best_score,
            "best_validation_per_sample_spearman": best_profile_spearman,
            "best_validation_per_sample_spearman_epoch": best_profile_spearman_epoch,
            "best_validation_per_sample_pearson_median": float(
                best_pearson_row["validation_per_sample_pearson_median"]
            ),
            "best_validation_per_sample_pearson_mean": float(
                best_pearson_row["validation_per_sample_pearson_mean"]
            ),
            "best_validation_per_sample_pearson_effective_n": int(
                best_pearson_row["validation_per_sample_pearson_effective_n"]
            ),
            "best_validation_per_sample_pearson_epoch": int(
                best_pearson_row["epoch"]
            ),
            "best_validation_per_site_spearman": best_site_spearman,
            "best_validation_per_site_spearman_epoch": best_site_spearman_epoch,
            "best_validation_balanced_spearman": best_balanced_spearman,
            "best_validation_balanced_spearman_epoch": best_balanced_spearman_epoch,
            "early_stopping_metric": args.early_stopping_metric,
            "best_early_stopping_score": best_early_stopping_score,
            "zero_shot_validation_summary": zero_shot_validation_summary,
            "training_template_batch8_flattened_cosine": template_cosine,
            "epochs_completed": len(history),
            "optimizer_updates": updates,
            "training_patient_exposures": int(
                finetune_updates_completed * args.sample_batch_size
            ),
            "validation_interval_updates": args.validation_interval_updates,
            "validation_only_at_finetune_end": (
                args.validation_only_at_finetune_end
            ),
            "pretrain_scope": (
                args.pretrain_scope if args.pretrain_updates else "none"
            ),
            "pretraining_samples": (
                int(len(pretrain_sample_ids))
                if pretrain_sample_ids is not None
                else 0
            ),
            "pretrain_updates": pretrain_updates_completed,
            "finetune_updates": finetune_updates_completed,
            "esm2_warmup_updates": esm2_warmup_updates_completed,
            "esm3_warmup_updates": (
                esm2_warmup_updates_completed if args.esm3_warmup_updates else 0
            ),
            "joint_updates": joint_updates_completed,
            "esm2_warmup_learning_rate": args.esm2_warmup_learning_rate,
            "esm3_warmup_learning_rate": args.esm3_warmup_learning_rate,
            "joint_learning_rate": (
                args.joint_learning_rate
                if args.joint_learning_rate is not None
                else args.learning_rate
            ),
            "runtime_seconds": time.time() - started,
            "maximum_observed_kinases_per_site": int(kinase_count.max(initial=0)),
            "sealed_phosphosite_rows_loaded": False,
        },
    )
    (output / "SUCCESS").write_text("complete\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
