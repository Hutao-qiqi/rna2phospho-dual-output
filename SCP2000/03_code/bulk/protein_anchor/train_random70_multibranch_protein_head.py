"""在统一参考分位坐标的 2,566 人随机 70:30 队列上训练三分支总蛋白头。"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold


RNA_DATASET = "rna_reference_quantile"
PROTEIN_DATASET = "protein_reference_quantile"
RNA_INPUT_COORDINATE = "reference_quantile_rna"
PROTEIN_TARGET_COORDINATE = "reference_quantile_protein"
EXPECTED_NORMALIZATION_FIT_SCOPE = "training_patients_only"
EXPECTED_NORMALIZATION_REFERENCE = "training_CPTAC_patients"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument(
        "--raw-input-h5",
        type=Path,
        help=(
            "可选的统一坐标源文件，仅核对患者、蛋白词表和缺失掩码；"
            "数值目标固定从 input-h5 的 protein_reference_quantile 读取。"
        ),
    )
    parser.add_argument("--protein-graph-artifact", type=Path, required=True)
    parser.add_argument("--translator-code-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--components", type=int, default=512)
    parser.add_argument("--local-genes", type=int, default=64)
    parser.add_argument("--supervised-genes", type=int, default=128)
    parser.add_argument("--global-ridge-alpha", type=float, default=100.0)
    parser.add_argument("--local-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--supervised-ridge-alpha", type=float, default=100.0)
    parser.add_argument("--protein-block-size", type=int, default=128)
    parser.add_argument("--calibration-shrinkage", type=float, default=0.5)
    parser.add_argument("--protein-limit", type=int, default=None)
    parser.add_argument("--outer-workers", type=int, default=5)
    parser.add_argument(
        "--external-prediction-only",
        action="store_true",
        help=(
            "Fit the complete 1,796-patient protein head and predict rows whose "
            "split is external; skip construction of a new five-fold OOF package."
        ),
    )
    parser.add_argument("--frozen-protein-results", type=Path)
    parser.add_argument("--frozen-supervised-prior", type=Path)
    return parser.parse_args()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=str,
    )


def require_reference_coordinate_contract(handle: h5py.File) -> dict[str, str]:
    """Validate the canonical train-fitted reference-quantile HDF5 contract."""

    required = {
        "sample_id",
        "split",
        "study",
        "cancer",
        "rna_vocabulary",
        "protein_vocabulary",
        RNA_DATASET,
        "rna_observed",
        PROTEIN_DATASET,
        "protein_observed",
    }
    missing = sorted(required.difference(handle.keys()))
    if missing:
        raise KeyError(f"统一参考分位输入缺少字段: {missing}")
    fit_scope = str(handle.attrs.get("normalization_fit_scope", ""))
    if fit_scope != EXPECTED_NORMALIZATION_FIT_SCOPE:
        raise ValueError("输入矩阵的归一化参数并非只由训练患者拟合")
    normalization_reference = str(handle.attrs.get("normalization_reference", ""))
    if normalization_reference != EXPECTED_NORMALIZATION_REFERENCE:
        raise ValueError("输入矩阵未使用训练集CPTAC参考分布")
    if not bool(handle.attrs.get("coordinate_scale_validated", False)):
        raise ValueError("统一参考分位输入尚未通过坐标量纲检查")
    if not bool(handle.attrs.get("training_ready", False)):
        raise ValueError("统一参考分位输入未标记为可训练")
    return {
        "normalization_fit_scope": fit_scope,
        "normalization_reference": normalization_reference,
        "rna_dataset": RNA_DATASET,
        "protein_dataset": PROTEIN_DATASET,
    }


def validate_reference_arrays(
    rna: np.ndarray,
    rna_observed: np.ndarray,
    protein: np.ndarray,
    protein_observed: np.ndarray,
) -> None:
    """Reject shape, mask and finite-value errors before fitting any fold."""

    if rna.ndim != 2 or protein.ndim != 2:
        raise ValueError("RNA和总蛋白参考分位矩阵必须为二维")
    if rna.shape != rna_observed.shape:
        raise ValueError("RNA参考分位矩阵与缺失掩码形状不一致")
    if protein.shape != protein_observed.shape:
        raise ValueError("总蛋白参考分位矩阵与缺失掩码形状不一致")
    if rna.shape[0] != protein.shape[0]:
        raise ValueError("RNA与总蛋白患者轴不一致")
    if rna_observed.dtype != np.bool_ or protein_observed.dtype != np.bool_:
        raise ValueError("RNA与总蛋白缺失掩码必须为布尔值")
    if not np.isfinite(rna[rna_observed]).all():
        raise ValueError("RNA参考分位矩阵的观测值含非有限值")
    if not np.isfinite(protein[protein_observed]).all():
        raise ValueError("总蛋白参考分位矩阵的观测值含非有限值")


def fill_reference_rna_from_fit(
    rna: np.ndarray,
    observed: np.ndarray,
    fit_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill missing RNA with fit-only gene means and return the fitted means."""

    values = np.asarray(rna, dtype=np.float32)
    mask = np.asarray(observed, dtype=bool)
    fit = np.asarray(fit_rows, dtype=np.int64)
    if values.ndim != 2 or values.shape != mask.shape:
        raise ValueError("RNA参考分位矩阵与掩码形状不一致")
    if fit.ndim != 1 or fit.size == 0:
        raise ValueError("RNA缺失填补需要非空训练患者索引")
    if np.unique(fit).size != fit.size:
        raise ValueError("RNA缺失填补训练患者索引重复")
    if fit.min() < 0 or fit.max() >= values.shape[0]:
        raise IndexError("RNA缺失填补训练患者索引越界")
    if not np.isfinite(values[mask]).all():
        raise ValueError("RNA参考分位矩阵的观测值含非有限值")
    fit_mask = mask[fit]
    fit_values = values[fit]
    counts = fit_mask.sum(axis=0, dtype=np.int64)
    sums = np.where(fit_mask, fit_values, 0.0).sum(axis=0, dtype=np.float64)
    gene_mean = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=np.float64),
        where=counts > 0,
    ).astype(np.float32)
    filled = np.where(mask, values, gene_mean[None, :]).astype(np.float32)
    if not np.isfinite(filled).all():
        raise RuntimeError("训练折内RNA缺失填补后仍含非有限值")
    return filled, gene_mean


def build_development_role_metadata(
    training_sample_id: np.ndarray,
    validation_sample_id: np.ndarray,
    training_source_fold: np.ndarray,
    fold_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build and validate the OOF/full-model role vector stored in the package."""

    training_id = np.asarray(training_sample_id, dtype=str)
    validation_id = np.asarray(validation_sample_id, dtype=str)
    source_fold = np.asarray(training_source_fold, dtype=np.int16)
    if training_id.ndim != 1 or validation_id.ndim != 1:
        raise ValueError("训练与验证患者编号必须为一维")
    if source_fold.shape != training_id.shape:
        raise ValueError("训练患者来源折与患者编号形状不一致")
    if fold_count < 2 or bool((source_fold < 0).any()) or bool(
        (source_fold >= fold_count).any()
    ):
        raise ValueError("训练患者来源折不在规定范围内")
    combined_id = np.concatenate([training_id, validation_id])
    if len(np.unique(combined_id)) != len(combined_id):
        raise ValueError("训练与验证角色包含重复患者")
    role = np.concatenate(
        [
            np.repeat("selection_train_oof", len(training_id)),
            np.repeat("selection_validation_full1796", len(validation_id)),
        ]
    )
    combined_source_fold = np.concatenate(
        [source_fold, np.full(len(validation_id), -1, dtype=np.int16)]
    )
    return combined_id, role, combined_source_fold


def stratification_labels(study: np.ndarray, cancer: np.ndarray) -> np.ndarray:
    detailed = np.char.add(np.char.add(study, "|"), cancer)
    counts = pd.Series(detailed).value_counts()
    return np.asarray(
        [value if counts[value] >= 2 else study[index] for index, value in enumerate(detailed)],
        dtype=str,
    )


def per_protein_spearman(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    result = np.full(prediction.shape[1], np.nan, dtype=np.float64)
    for protein in range(prediction.shape[1]):
        use = mask[:, protein]
        if int(use.sum()) < 8:
            continue
        left = rankdata(prediction[use, protein], method="average")
        right = rankdata(target[use, protein], method="average")
        left -= left.mean()
        right -= right.mean()
        denominator = np.sqrt(np.square(left).sum() * np.square(right).sum())
        if denominator > 0:
            result[protein] = float(left @ right / denominator)
    return result


def per_sample_spearman(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    result = np.full(prediction.shape[0], np.nan, dtype=np.float64)
    for sample in range(prediction.shape[0]):
        use = mask[sample]
        if int(use.sum()) < 8:
            continue
        left = rankdata(prediction[sample, use], method="average")
        right = rankdata(target[sample, use], method="average")
        left -= left.mean()
        right -= right.mean()
        denominator = np.sqrt(np.square(left).sum() * np.square(right).sum())
        if denominator > 0:
            result[sample] = float(left @ right / denominator)
    return result


def per_sample_cosine(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    result = np.full(prediction.shape[0], np.nan, dtype=np.float64)
    for sample in range(prediction.shape[0]):
        use = mask[sample]
        if int(use.sum()) < 2:
            continue
        left = prediction[sample, use]
        right = target[sample, use]
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        if denominator > 0:
            result[sample] = float(left @ right / denominator)
    return result


def metric_record(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    training_coverage: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    protein_score = per_protein_spearman(prediction, target, mask)
    sample_score = per_sample_spearman(prediction, target, mask)
    sample_cosine = per_sample_cosine(prediction, target, mask)
    high = np.argsort(-training_coverage, kind="stable")[: min(1000, training_coverage.size)]
    observed = mask
    record = {
        "evaluated_proteins": int(np.isfinite(protein_score).sum()),
        "median_per_protein_spearman": float(np.nanmedian(protein_score)),
        "median_high_coverage_1000_spearman": float(np.nanmedian(protein_score[high])),
        "median_within_sample_spearman": float(np.nanmedian(sample_score)),
        "median_within_sample_cosine": float(np.nanmedian(sample_cosine)),
        "mse": float(np.square(prediction[observed] - target[observed]).mean()),
        "mae": float(np.abs(prediction[observed] - target[observed]).mean()),
    }
    return record, protein_score, sample_score


def inverse_protein_scale(
    values: np.ndarray,
    scaler: object,
) -> np.ndarray:
    """Return fold-scaled predictions to the shared reference-quantile coordinate."""
    matrix = np.asarray(values, dtype=np.float64)
    lower = np.asarray(getattr(scaler, "lower"), dtype=np.float64)
    safe_range = np.asarray(getattr(scaler, "safe_range"), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != lower.size:
        raise ValueError("scaled protein predictions and fitted scale disagree")
    return (matrix * safe_range[None, :] + lower[None, :]).astype(np.float32)


def main() -> int:
    args = arguments()
    sys.path.insert(0, str(args.translator_code_dir))
    from audit_locked_oof_adaptive_study_calibration import (
        apply_group_calibration,
        fit_group_calibration,
    )
    from audit_locked_oof_string_pca_blend import (
        convert_scale,
        fit_global_prediction,
        local_prediction,
        transform_subset,
    )
    from audit_locked_oof_three_branch_blend import learn_shrunk_weights
    from candidate_multiscale_protein_query import LocalGenePrior, build_protein_local_gene_prior
    from hybrid_protein_training_contract import TrainFittedProteinScale
    from protein_graph_prior import ProteinGraphArtifact
    from train_cptac_pretrain_tcpa_finetune import (
        fit_direct_local_ridge_initialization,
        fit_supervised_local_gene_prior,
        fit_train_study_protein_baseline,
        fit_transform_train_study_centered_rna,
    )

    started = time.time()
    if args.folds != 5:
        raise ValueError("总蛋白训练患者折外预测固定使用五折")
    if not 1 <= args.outer_workers <= args.folds:
        raise ValueError("outer-workers 必须位于 1 到 folds 之间")
    output = args.output_dir
    for path in (
        output,
        output / "tables",
        output / "models",
        output / "logs",
        output / "shards",
    ):
        path.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input_h5, "r") as handle:
        coordinate_contract = require_reference_coordinate_contract(handle)
        sample_id = decode(handle["sample_id"][:])
        split = decode(handle["split"][:])
        study = decode(handle["study"][:])
        cancer = decode(handle["cancer"][:])
        rna_names = decode(handle["rna_vocabulary"][:])
        protein_names = decode(handle["protein_vocabulary"][:])
        rna = np.asarray(handle[RNA_DATASET][:], dtype=np.float32)
        rna_observed = np.asarray(handle["rna_observed"][:], dtype=bool)
        protein = np.asarray(handle[PROTEIN_DATASET][:], dtype=np.float32)
        protein_mask = np.asarray(handle["protein_observed"][:], dtype=bool)
    validate_reference_arrays(rna, rna_observed, protein, protein_mask)
    if len(rna_names) != rna.shape[1]:
        raise ValueError("RNA参考分位矩阵与基因词表不一致")
    if len(protein_names) != protein.shape[1]:
        raise ValueError("总蛋白参考分位矩阵与蛋白词表不一致")
    if args.raw_input_h5 is not None:
        with h5py.File(args.raw_input_h5, "r") as handle:
            raw_sample_id = decode(handle["sample_id"][:])
            raw_protein_names = decode(handle["protein_vocabulary"][:])
            raw_protein_mask = np.asarray(
                handle["protein_observed"][:], dtype=bool
            )
        if not np.array_equal(raw_sample_id, sample_id):
            raise ValueError("统一坐标源文件与随机划分患者顺序不一致")
        if not np.array_equal(raw_protein_names, protein_names):
            raise ValueError("统一坐标源文件与随机划分蛋白词表不一致")
        if not np.array_equal(raw_protein_mask, protein_mask):
            raise ValueError("统一坐标源文件与随机划分缺失掩码不一致")
    if len(np.unique(sample_id)) != len(sample_id):
        raise ValueError("患者编号重复")
    training_rows = np.flatnonzero((split == "train") & protein_mask.any(axis=1))
    if args.external_prediction_only:
        validation_rows = np.flatnonzero(split == "external")
        if len(training_rows) != 1796 or len(validation_rows) == 0:
            raise ValueError("外部预测要求1,796名训练患者和至少一名external患者")
    else:
        validation_rows = np.flatnonzero(
            (split == "validation") & protein_mask.any(axis=1)
        )
        if (len(training_rows), len(validation_rows)) != (1796, 770):
            raise ValueError("总蛋白监督患者数与随机划分摘要不一致")

    complete_protein_names = protein_names.copy()
    kept = np.arange(len(protein_names), dtype=np.int64)
    if args.protein_limit is not None:
        kept = np.argsort(-protein_mask[training_rows].sum(axis=0), kind="stable")[
            : args.protein_limit
        ]
        protein_names = protein_names[kept]
        protein = protein[:, kept]
        protein_mask = protein_mask[:, kept]
    names = tuple(map(str, protein_names))
    genes = tuple(map(str, rna_names))

    graph = ProteinGraphArtifact.load(args.protein_graph_artifact)
    gene_lookup = {name.upper(): index for index, name in enumerate(genes)}
    graph_names = tuple(map(str, graph.protein_names))
    graph_position = {name.upper(): index for index, name in enumerate(graph_names)}
    complete_names = tuple(map(str, complete_protein_names))
    if set(graph_position) != {name.upper() for name in complete_names}:
        raise ValueError("STRING 图与蛋白词表不一致")
    complete_parent = np.asarray(
        [gene_lookup.get(name.upper(), -1) for name in graph_names], dtype=np.int64
    )
    complete_prior = build_protein_local_gene_prior(
        genes,
        graph_names,
        complete_parent,
        graph,
        max_local_genes=args.local_genes,
    )
    prior_rows = np.asarray(
        [graph_position[complete_names[index].upper()] for index in kept], dtype=np.int64
    )
    string_prior = LocalGenePrior(
        gene_index=complete_prior.gene_index[prior_rows],
        gene_mask=complete_prior.gene_mask[prior_rows],
        relation_index=complete_prior.relation_index[prior_rows],
        edge_strength=complete_prior.edge_strength[prior_rows],
        relation_names=complete_prior.relation_names,
        metadata={**complete_prior.metadata, "selected_protein_count": len(names)},
    )

    def fit_three_branches(
        fit_rows: np.ndarray,
        predict_rows: np.ndarray,
        scaler: TrainFittedProteinScale,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        target = np.zeros_like(protein, dtype=np.float32)
        mask = np.zeros_like(protein_mask, dtype=bool)
        target[fit_rows], mask[fit_rows] = transform_subset(
            scaler, protein, fit_rows, names, clip=True
        )
        target[predict_rows], mask[predict_rows] = transform_subset(
            scaler, protein, predict_rows, names, clip=False
        )
        baseline, _ = fit_train_study_protein_baseline(target, mask, fit_rows, study)
        filled_rna, rna_gene_mean = fill_reference_rna_from_fit(
            rna, rna_observed, fit_rows
        )
        transformed_rna, _ = fit_transform_train_study_centered_rna(
            filled_rna, fit_rows, study
        )
        local_weight, local_bias, local_state = fit_direct_local_ridge_initialization(
            transformed_rna,
            target,
            mask,
            baseline,
            fit_rows,
            string_prior,
            alpha=args.local_ridge_alpha,
        )
        local_value = local_prediction(
            transformed_rna,
            baseline,
            local_weight,
            local_bias,
            string_prior,
            predict_rows,
        )
        global_value, explained = fit_global_prediction(
            transformed_rna,
            target,
            mask,
            baseline,
            fit_rows,
            predict_rows,
            components=min(args.components, len(fit_rows) - 1),
            alpha=args.global_ridge_alpha,
            seed=seed,
        )
        supervised_prior = fit_supervised_local_gene_prior(
            transformed_rna,
            target,
            mask,
            baseline,
            fit_rows,
            genes,
            names,
            selected_genes=args.supervised_genes,
            protein_block_size=args.protein_block_size,
        )
        supervised_weight, supervised_bias, supervised_state = (
            fit_direct_local_ridge_initialization(
                transformed_rna,
                target,
                mask,
                baseline,
                fit_rows,
                supervised_prior,
                alpha=args.supervised_ridge_alpha,
            )
        )
        supervised_value = local_prediction(
            transformed_rna,
            baseline,
            supervised_weight,
            supervised_bias,
            supervised_prior,
            predict_rows,
        )
        state = {
            "local_fitted_proteins": int(local_state["fitted_proteins"]),
            "supervised_fitted_proteins": int(supervised_state["fitted_proteins"]),
            "global_explained_fraction": float(explained),
            "supervised_prior": supervised_prior,
            "rna_imputed_genes": int(
                np.count_nonzero(~rna_observed[fit_rows].all(axis=0))
            ),
            "rna_fit_gene_mean_finite": bool(np.isfinite(rna_gene_mean).all()),
        }
        return local_value, global_value, supervised_value, target, mask, state

    def fit_complete_pipeline(
        fit_rows: np.ndarray,
        predict_rows: np.ndarray,
        *,
        seed: int,
        inner_folds: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, object],
        dict[str, object],
    ]:
        """Fit scaling, branches, fusion and calibration inside one outer fold."""
        fit_rows = np.asarray(fit_rows, dtype=np.int64)
        predict_rows = np.asarray(predict_rows, dtype=np.int64)
        outer_scaler = TrainFittedProteinScale.fit(
            protein,
            fit_rows,
            feature_names=names,
            mask=protein_mask,
            lower_quantile=0.01,
            upper_quantile=0.99,
        )
        outer_target = np.zeros_like(protein, dtype=np.float32)
        outer_mask = np.zeros_like(protein_mask, dtype=bool)
        outer_target[fit_rows], outer_mask[fit_rows] = transform_subset(
            outer_scaler, protein, fit_rows, names, clip=True
        )
        labels = stratification_labels(study[fit_rows], cancer[fit_rows])
        splitter = StratifiedKFold(
            n_splits=inner_folds, shuffle=True, random_state=seed
        )
        inner_pairs = list(splitter.split(np.arange(len(fit_rows)), labels))
        shape = (len(fit_rows), len(names))
        inner_local = np.full(shape, np.nan, dtype=np.float32)
        inner_global = np.full(shape, np.nan, dtype=np.float32)
        inner_supervised = np.full(shape, np.nan, dtype=np.float32)
        for inner_index, (inner_fit_position, inner_holdout_position) in enumerate(
            inner_pairs
        ):
            inner_fit_position = np.asarray(inner_fit_position, dtype=np.int64)
            inner_holdout_position = np.asarray(
                inner_holdout_position, dtype=np.int64
            )
            inner_fit_rows = fit_rows[inner_fit_position]
            inner_holdout_rows = fit_rows[inner_holdout_position]
            inner_scaler = TrainFittedProteinScale.fit(
                protein,
                inner_fit_rows,
                feature_names=names,
                mask=protein_mask,
                lower_quantile=0.01,
                upper_quantile=0.99,
            )
            local_value, global_value, supervised_value, _, _, _ = (
                fit_three_branches(
                    inner_fit_rows,
                    inner_holdout_rows,
                    inner_scaler,
                    seed + inner_index,
                )
            )
            inner_local[inner_holdout_position] = convert_scale(
                local_value, inner_scaler, outer_scaler
            )
            inner_global[inner_holdout_position] = convert_scale(
                global_value, inner_scaler, outer_scaler
            )
            inner_supervised[inner_holdout_position] = convert_scale(
                supervised_value, inner_scaler, outer_scaler
            )
        if not all(
            np.isfinite(value).all()
            for value in (inner_local, inner_global, inner_supervised)
        ):
            raise RuntimeError("内层总蛋白折外分支预测不完整")

        fit_target = outer_target[fit_rows]
        fit_mask = outer_mask[fit_rows]
        base_weights = learn_shrunk_weights(
            inner_local, inner_global, fit_target, fit_mask
        )[4]
        inner_base = (
            (1.0 - base_weights[None, :]) * inner_local
            + base_weights[None, :] * inner_global
        ).astype(np.float32)
        supervised_weights = learn_shrunk_weights(
            inner_base, inner_supervised, fit_target, fit_mask
        )[4]
        inner_selected = (
            (1.0 - supervised_weights[None, :]) * inner_base
            + supervised_weights[None, :] * inner_supervised
        ).astype(np.float32)

        study_names = tuple(sorted(set(study[fit_rows].tolist())))
        study_lookup = {name: index for index, name in enumerate(study_names)}
        fit_study_index = np.asarray(
            [study_lookup[name] for name in study[fit_rows]], dtype=np.int64
        )
        inner_calibrated = np.full_like(inner_selected, np.nan)
        for inner_fit_position, inner_holdout_position in inner_pairs:
            inner_fit_position = np.asarray(inner_fit_position, dtype=np.int64)
            inner_holdout_position = np.asarray(
                inner_holdout_position, dtype=np.int64
            )
            state = fit_group_calibration(
                inner_selected,
                fit_target,
                fit_mask,
                fit_study_index,
                inner_fit_position,
                len(study_names),
            )
            inner_calibrated[inner_holdout_position] = apply_group_calibration(
                inner_selected,
                fit_study_index,
                inner_holdout_position,
                state,
                args.calibration_shrinkage,
            )
        coverage = fit_mask.sum(axis=0)
        inner_uncalibrated_metric, _, _ = metric_record(
            inner_selected, fit_target, fit_mask, coverage
        )
        inner_calibrated_metric, _, _ = metric_record(
            inner_calibrated, fit_target, fit_mask, coverage
        )
        use_calibration = (
            inner_calibrated_metric["median_per_protein_spearman"]
            > inner_uncalibrated_metric["median_per_protein_spearman"]
        )

        local_value, global_value, supervised_value, _, _, full_state = (
            fit_three_branches(
                fit_rows, predict_rows, outer_scaler, seed + inner_folds + 1
            )
        )
        base_value = (
            (1.0 - base_weights[None, :]) * local_value
            + base_weights[None, :] * global_value
        ).astype(np.float32)
        prediction = (
            (1.0 - supervised_weights[None, :]) * base_value
            + supervised_weights[None, :] * supervised_value
        ).astype(np.float32)
        if use_calibration:
            calibration_state = fit_group_calibration(
                inner_selected,
                fit_target,
                fit_mask,
                fit_study_index,
                np.arange(len(fit_rows), dtype=np.int64),
                len(study_names),
            )
            predict_study_index = np.asarray(
                [study_lookup.get(name, -1) for name in study[predict_rows]],
                dtype=np.int64,
            )
            combined_prediction = np.concatenate(
                [inner_selected, prediction], axis=0
            )
            combined_study_index = np.concatenate(
                [fit_study_index, predict_study_index]
            )
            prediction_position = np.arange(
                len(fit_rows), len(fit_rows) + len(predict_rows)
            )
            prediction = apply_group_calibration(
                combined_prediction,
                combined_study_index,
                prediction_position,
                calibration_state,
                args.calibration_shrinkage,
            )
        # Each outer fold owns a different training-fitted scale.  Return every
        # prediction to the shared reference-quantile coordinate before concatenating
        # outer holdouts; otherwise fold identity changes a protein's numeric
        # coordinate and can disturb cross-patient ordering downstream.
        prediction = inverse_protein_scale(prediction, outer_scaler)
        prediction_target = np.where(
            protein_mask[predict_rows], protein[predict_rows], np.nan
        ).astype(np.float32)
        prediction_mask = protein_mask[predict_rows].copy()
        metadata = {
            "base_weights": base_weights,
            "supervised_weights": supervised_weights,
            "calibration_selected": bool(use_calibration),
            "inner_uncalibrated": inner_uncalibrated_metric,
            "inner_calibrated": inner_calibrated_metric,
        }
        return prediction, prediction_target, prediction_mask, metadata, full_state

    training_coverage = protein_mask[training_rows].sum(axis=0)
    if args.external_prediction_only:
        frozen = (args.frozen_protein_results, args.frozen_supervised_prior)
        if all(value is not None for value in frozen):
            outer_scaler = TrainFittedProteinScale.fit(
                protein,
                training_rows,
                feature_names=names,
                mask=protein_mask,
                lower_quantile=0.01,
                upper_quantile=0.99,
            )
            target = np.zeros_like(protein, dtype=np.float32)
            target[training_rows], _ = transform_subset(
                outer_scaler, protein, training_rows, names, clip=True
            )
            baseline, _ = fit_train_study_protein_baseline(
                target, protein_mask, training_rows, study
            )
            filled_rna, _ = fill_reference_rna_from_fit(
                rna, rna_observed, training_rows
            )
            transformed_rna, _ = fit_transform_train_study_centered_rna(
                filled_rna, training_rows, study
            )
            local_weight, local_bias, _ = fit_direct_local_ridge_initialization(
                transformed_rna,
                target,
                protein_mask,
                baseline,
                training_rows,
                string_prior,
                alpha=args.local_ridge_alpha,
            )
            local_value = local_prediction(
                transformed_rna,
                baseline,
                local_weight,
                local_bias,
                string_prior,
                validation_rows,
            )
            global_value, _ = fit_global_prediction(
                transformed_rna,
                target,
                protein_mask,
                baseline,
                training_rows,
                validation_rows,
                components=min(args.components, len(training_rows) - 1),
                alpha=args.global_ridge_alpha,
                seed=args.seed + 1006,
            )
            with np.load(args.frozen_supervised_prior, allow_pickle=False) as saved:
                supervised_prior = LocalGenePrior(
                    gene_index=np.asarray(saved["gene_index"], dtype=np.int64),
                    gene_mask=np.asarray(saved["gene_mask"], dtype=bool),
                    relation_index=np.zeros_like(
                        np.asarray(saved["gene_index"], dtype=np.int64)
                    ),
                    edge_strength=np.asarray(saved["edge_strength"], dtype=np.float32),
                    relation_names=("supervised",),
                    metadata={"source": "frozen_full1796_supervised_prior"},
                )
            supervised_weight, supervised_bias, _ = (
                fit_direct_local_ridge_initialization(
                    transformed_rna,
                    target,
                    protein_mask,
                    baseline,
                    training_rows,
                    supervised_prior,
                    alpha=args.supervised_ridge_alpha,
                )
            )
            supervised_value = local_prediction(
                transformed_rna,
                baseline,
                supervised_weight,
                supervised_bias,
                supervised_prior,
                validation_rows,
            )
            frozen_results = pd.read_csv(args.frozen_protein_results, sep="\t")
            if frozen_results["protein"].astype(str).tolist() != list(names):
                raise ValueError("冻结的总蛋白融合权重与蛋白词表不一致")
            base_weights = frozen_results["base_global_weight"].to_numpy(np.float32)
            supervised_weights = frozen_results["supervised_weight"].to_numpy(np.float32)
            base_value = (
                (1.0 - base_weights[None, :]) * local_value
                + base_weights[None, :] * global_value
            ).astype(np.float32)
            external_prediction = (
                (1.0 - supervised_weights[None, :]) * base_value
                + supervised_weights[None, :] * supervised_value
            ).astype(np.float32)
            external_prediction = inverse_protein_scale(
                external_prediction, outer_scaler
            )
            external_metadata = {
                "calibration_selected": False,
                "frozen_selection_reused": True,
            }
        elif any(value is not None for value in frozen):
            raise ValueError("冻结融合权重和监督先验必须同时提供")
        else:
            (
                external_prediction,
                _,
                _,
                external_metadata,
                _,
            ) = fit_complete_pipeline(
                training_rows,
                validation_rows,
                seed=args.seed + 1000,
                inner_folds=5,
            )
        if not np.isfinite(external_prediction).all():
            raise RuntimeError("外部总蛋白预测含非有限值")
        reliability = training_coverage.astype(np.float32)
        reliability /= max(float(reliability.max()), 1.0)
        reliability_matrix = np.broadcast_to(
            reliability[None, :], external_prediction.shape
        ).copy()
        package_path = output / "external_multibranch_protein_anchor.npz"
        np.savez_compressed(
            package_path,
            sample_id=sample_id[validation_rows],
            target=np.asarray(names, dtype=str),
            prediction=external_prediction,
            reliability=reliability_matrix,
            external_sample_usable=np.ones(len(validation_rows), dtype=bool),
            external_sample_exclusion_reason=np.repeat("", len(validation_rows)),
            rna_input_coordinate=np.asarray(RNA_INPUT_COORDINATE, dtype=str),
            protein_target_coordinate=np.asarray(
                PROTEIN_TARGET_COORDINATE, dtype=str
            ),
            prediction_coordinate=np.asarray(PROTEIN_TARGET_COORDINATE, dtype=str),
        )
        result = {
            "status": "complete",
            "mode": "external_prediction_only",
            "model_family": (
                "string64_pca512_supervised128_full1796_blend"
            ),
            "training_samples": int(len(training_rows)),
            "external_samples": int(len(validation_rows)),
            "external_labels_used_for_fit": False,
            "rna_dataset": coordinate_contract["rna_dataset"],
            "rna_input_coordinate": RNA_INPUT_COORDINATE,
            "protein_dataset": coordinate_contract["protein_dataset"],
            "protein_target_coordinate": PROTEIN_TARGET_COORDINATE,
            "protein_package_coordinate": PROTEIN_TARGET_COORDINATE,
            "normalization_fit_scope": coordinate_contract[
                "normalization_fit_scope"
            ],
            "normalization_reference": coordinate_contract[
                "normalization_reference"
            ],
            "raw_input_h5_role": (
                "axis_and_mask_integrity_only"
                if args.raw_input_h5 is not None
                else "unused"
            ),
            "calibration_selected": bool(
                external_metadata["calibration_selected"]
            ),
            "frozen_selection_reused": bool(
                external_metadata.get("frozen_selection_reused", False)
            ),
            "package": str(package_path),
            "elapsed_seconds": float(time.time() - started),
        }
        (output / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0

    labels = stratification_labels(study[training_rows], cancer[training_rows])
    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    fold_pairs = list(splitter.split(np.arange(len(training_rows)), labels))
    training_prediction = np.full(
        (len(training_rows), len(names)), np.nan, dtype=np.float32
    )
    training_target = np.full_like(training_prediction, np.nan)
    training_mask = np.zeros_like(training_prediction, dtype=bool)
    training_source_fold = np.full(len(training_rows), -1, dtype=np.int16)
    fold_records: list[dict[str, object]] = []

    def run_outer_fold(
        fold_index: int,
        fit_position: np.ndarray,
        holdout_position: np.ndarray,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        fit_position = np.asarray(fit_position, dtype=np.int64)
        holdout_position = np.asarray(holdout_position, dtype=np.int64)
        (
            outer_prediction,
            outer_target,
            outer_mask,
            outer_metadata,
            _,
        ) = fit_complete_pipeline(
            training_rows[fit_position],
            training_rows[holdout_position],
            seed=args.seed + fold_index * 100,
            inner_folds=4,
        )
        record = {
            "fold": fold_index,
            "fit_samples": int(len(fit_position)),
            "holdout_samples": int(len(holdout_position)),
            "inner_folds": 4,
            "calibration_selected": outer_metadata["calibration_selected"],
            "elapsed_seconds": float(time.time() - started),
        }
        return (
            fold_index,
            holdout_position,
            outer_prediction,
            outer_target,
            outer_mask,
            record,
        )

    def write_outer_fold_shard(
        fold_index: int,
        fit_position: np.ndarray,
        holdout_position: np.ndarray,
        shard_path: Path,
        error_path: Path,
    ) -> None:
        try:
            (
                fold_index,
                holdout_position,
                outer_prediction,
                outer_target,
                outer_mask,
                record,
            ) = run_outer_fold(fold_index, fit_position, holdout_position)
            np.savez(
                shard_path,
                fold_index=np.asarray(fold_index, dtype=np.int16),
                holdout_position=holdout_position,
                prediction=outer_prediction,
                target=outer_target,
                mask=outer_mask,
                record_json=np.asarray(json.dumps(record, ensure_ascii=False)),
            )
        except BaseException:
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            raise

    shard_paths = [output / "shards" / f"outer_fold_{fold}.npz" for fold in range(args.folds)]
    error_paths = [output / "logs" / f"outer_fold_{fold}.error.log" for fold in range(args.folds)]
    if args.outer_workers == 1:
        for fold_index, (fit_position, holdout_position) in enumerate(fold_pairs):
            write_outer_fold_shard(
                fold_index,
                fit_position,
                holdout_position,
                shard_paths[fold_index],
                error_paths[fold_index],
            )
    else:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("多外折进程训练需要支持 fork 的运行环境")
        context = mp.get_context("fork")
        active: list[mp.Process] = []
        next_fold = 0
        while next_fold < len(fold_pairs) or active:
            while next_fold < len(fold_pairs) and len(active) < args.outer_workers:
                fit_position, holdout_position = fold_pairs[next_fold]
                process = context.Process(
                    target=write_outer_fold_shard,
                    args=(
                        next_fold,
                        fit_position,
                        holdout_position,
                        shard_paths[next_fold],
                        error_paths[next_fold],
                    ),
                    name=f"protein_outer_fold_{next_fold}",
                )
                process.start()
                active.append(process)
                next_fold += 1
            for process in list(active):
                process.join(timeout=0.2)
                if process.is_alive():
                    continue
                active.remove(process)
                if process.exitcode != 0:
                    raise RuntimeError(
                        f"外折子进程 {process.name} 失败，退出码={process.exitcode}"
                    )

    for shard_path in shard_paths:
        if not shard_path.exists():
            raise RuntimeError(f"缺少外折结果分片: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as shard:
            fold_index = int(np.asarray(shard["fold_index"]).item())
            holdout_position = np.asarray(shard["holdout_position"], dtype=np.int64)
            outer_prediction = np.asarray(shard["prediction"], dtype=np.float32)
            outer_target = np.asarray(shard["target"], dtype=np.float32)
            outer_mask = np.asarray(shard["mask"], dtype=bool)
            record = json.loads(str(np.asarray(shard["record_json"]).item()))
            training_prediction[holdout_position] = outer_prediction
            training_target[holdout_position] = outer_target
            training_mask[holdout_position] = outer_mask
            training_source_fold[holdout_position] = fold_index
            fold_records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    fold_records.sort(key=lambda row: int(row["fold"]))
    if not np.isfinite(training_prediction).all() or bool(
        (training_source_fold < 0).any()
    ):
        raise RuntimeError("端到端总蛋白折外预测不完整")

    (
        validation_prediction,
        validation_target,
        validation_mask,
        full_metadata,
        full_state,
    ) = fit_complete_pipeline(
        training_rows, validation_rows, seed=args.seed + 1000, inner_folds=5
    )
    oof_metric, _, _ = metric_record(
        training_prediction,
        training_target,
        training_mask,
        training_coverage,
    )
    validation_metric, protein_score, sample_score = metric_record(
        validation_prediction,
        validation_target,
        validation_mask,
        training_coverage,
    )

    reliability = training_coverage.astype(np.float32)
    reliability /= max(float(reliability.max()), 1.0)
    development_sample_id, development_role, development_source_fold = (
        build_development_role_metadata(
            sample_id[training_rows],
            sample_id[validation_rows],
            training_source_fold,
            args.folds,
        )
    )
    package_path = output / "random70_multibranch_protein_anchor.npz"
    np.savez_compressed(
        package_path,
        protein_names=np.asarray(names, dtype=str),
        reliability=reliability,
        training_sample_id=sample_id[training_rows],
        training_prediction_z=training_prediction,
        rna_input_coordinate=np.asarray(RNA_INPUT_COORDINATE, dtype=str),
        protein_target_coordinate=np.asarray(PROTEIN_TARGET_COORDINATE, dtype=str),
        prediction_coordinate=np.asarray(PROTEIN_TARGET_COORDINATE, dtype=str),
        validation_sample_id=sample_id[validation_rows],
        validation_prediction_z=validation_prediction,
        development_sample_id=development_sample_id,
        development_prediction_z=np.concatenate(
            [training_prediction, validation_prediction], axis=0
        ),
        development_role=development_role,
        development_source_fold=development_source_fold,
    )
    pd.DataFrame(
        {
            "protein": names,
            "training_observations": training_coverage,
            "base_global_weight": full_metadata["base_weights"],
            "supervised_weight": full_metadata["supervised_weights"],
            "validation_spearman": protein_score,
        }
    ).to_csv(output / "tables/protein_results.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "sample_id": sample_id[validation_rows],
            "within_sample_spearman": sample_score,
        }
    ).to_csv(output / "tables/validation_per_sample.tsv", sep="\t", index=False)
    pd.DataFrame(fold_records).to_csv(
        output / "tables/fold_runtime.tsv", sep="\t", index=False
    )
    supervised_prior = full_state["supervised_prior"]
    np.savez_compressed(
        output / "models/full_supervised_gene_prior.npz",
        gene_index=supervised_prior.gene_index,
        gene_mask=supervised_prior.gene_mask,
        edge_strength=supervised_prior.edge_strength,
    )
    result = {
        "status": "complete",
        "model_family": "string64_pca512_supervised128_five_fold_oof_blend",
        "training_samples": int(len(training_rows)),
        "validation_samples": int(len(validation_rows)),
        "folds": args.folds,
        "parallel_outer_workers": args.outer_workers,
        "fold_records": fold_records,
        "rna_dataset": coordinate_contract["rna_dataset"],
        "rna_input": "continuous_reference_quantile_fit_only_imputation",
        "rna_input_coordinate": RNA_INPUT_COORDINATE,
        "protein_dataset": coordinate_contract["protein_dataset"],
        "protein_branch_training_target": (
            "reference_quantile_protein_outer_fold_q01_q99_scale"
        ),
        "protein_target_coordinate": PROTEIN_TARGET_COORDINATE,
        "protein_package_coordinate": PROTEIN_TARGET_COORDINATE,
        "normalization_fit_scope": coordinate_contract[
            "normalization_fit_scope"
        ],
        "normalization_reference": coordinate_contract[
            "normalization_reference"
        ],
        "raw_input_h5_role": (
            "axis_and_mask_integrity_only"
            if args.raw_input_h5 is not None
            else "unused"
        ),
        "training_oof": oof_metric,
        "full_training_inner_uncalibrated": full_metadata[
            "inner_uncalibrated"
        ],
        "full_training_inner_calibrated": full_metadata["inner_calibrated"],
        "full_training_calibration_selected": full_metadata[
            "calibration_selected"
        ],
        "validation": validation_metric,
        "package": str(package_path),
        "elapsed_seconds": float(time.time() - started),
    }
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
