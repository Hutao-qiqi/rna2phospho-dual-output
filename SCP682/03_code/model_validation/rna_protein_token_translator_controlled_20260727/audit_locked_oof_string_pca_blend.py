"""Learn STRING/PCA blend weights from strict out-of-fold training predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.utils.extmath import randomized_svd

import train_protein as base
import train_hybrid_protein_development_screen as dev
from candidate_groupwise_rna_normalization import (
    build_strict_inner_partitions,
    seal_outer_protein_labels,
)
from candidate_multiscale_protein_query import build_protein_local_gene_prior
from hybrid_protein_training_contract import TrainFittedProteinScale
from protein_graph_prior import ProteinGraphArtifact
from splits import make_strict_folds
from train_cptac_pretrain_tcpa_finetune import (
    fit_direct_local_ridge_initialization,
    fit_train_study_protein_baseline,
    fit_transform_train_study_centered_rna,
)


GLOBAL_WEIGHT_GRID = np.linspace(0.0, 1.0, 21)
SHRINKAGE_GRID = (20.0, 50.0, 100.0, 200.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--protein-graph-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--components", type=int, default=512)
    parser.add_argument("--global-ridge-alpha", type=float, default=100.0)
    parser.add_argument("--local-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--local-genes", type=int, default=64)
    parser.add_argument("--meta-folds", type=int, default=5)
    return parser.parse_args()


def input_args(data_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=data_dir,
        rna_file="rna_log2_tpm_paired.parquet",
        protein_raw_file="total_protein_gene_logratio_all.parquet",
        protein_vocab_file="total_protein_gene_study_zscore_min20pct.parquet",
        manifest_file="sample_manifest.tsv",
        sample_id_column="sample_id",
        strata_column="cancer_label",
        group_column="case_submitter_id",
        study_column="pdc_study_id",
        max_samples=None,
        max_genes=None,
        max_proteins=None,
    )


def transform_subset(
    scaler: TrainFittedProteinScale,
    raw: np.ndarray,
    indices: np.ndarray,
    protein_names: tuple[str, ...],
    *,
    clip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.isfinite(raw[indices])
    changed = scaler.transform(
        raw[indices], mask=observed, feature_names=protein_names, clip=clip
    )
    return np.nan_to_num(changed, nan=0.0), np.isfinite(changed)


def local_prediction(
    rna: np.ndarray,
    baseline: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
    prior,
    indices: np.ndarray,
) -> np.ndarray:
    prediction = baseline[indices].copy()
    for protein in range(prior.n_targets):
        positions = np.flatnonzero(prior.gene_mask[protein])
        if positions.size == 0:
            continue
        genes = prior.gene_index[protein, positions]
        prediction[:, protein] += (
            rna[indices][:, genes] @ weights[protein, positions] + biases[protein]
        )
    return prediction.astype(np.float32)


def fit_global_prediction(
    rna: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    baseline: np.ndarray,
    fit: np.ndarray,
    predict: np.ndarray,
    *,
    components: int,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    _, singular_values, axes = randomized_svd(
        np.asarray(rna[fit], dtype=np.float64),
        n_components=components,
        n_iter=5,
        random_state=seed,
    )
    fit_scores = np.asarray(rna[fit], dtype=np.float64) @ axes.T
    prediction_scores = np.asarray(rna[predict], dtype=np.float64) @ axes.T
    scale = fit_scores.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    fit_scores /= scale
    prediction_scores /= scale
    residual = target - baseline
    prediction = baseline[predict].copy()
    for protein in range(target.shape[1]):
        use = mask[fit, protein]
        if use.sum() < 8:
            continue
        x = fit_scores[use]
        y = residual[fit[use], protein].astype(np.float64)
        x_mean = x.mean(axis=0)
        y_mean = float(y.mean())
        centered_x = x - x_mean
        gram = centered_x.T @ centered_x
        gram.flat[:: gram.shape[0] + 1] += alpha
        coefficient = np.linalg.solve(gram, centered_x.T @ (y - y_mean))
        prediction[:, protein] += (
            (prediction_scores - x_mean) @ coefficient + y_mean
        ).astype(np.float32)
    explained = float(
        np.square(singular_values).sum()
        / np.square(np.linalg.norm(rna[fit], ord="fro"))
    )
    return prediction.astype(np.float32), explained


def convert_scale(
    prediction: np.ndarray,
    source: TrainFittedProteinScale,
    destination: TrainFittedProteinScale,
) -> np.ndarray:
    raw_prediction = source.lower + prediction * source.safe_range
    return ((raw_prediction - destination.lower) / destination.safe_range).astype(
        np.float32
    )


def protein_spearman(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    values = np.full(target.shape[1], np.nan, dtype=np.float64)
    for protein in range(target.shape[1]):
        use = mask[:, protein] & np.isfinite(prediction[:, protein])
        if use.sum() < 3 or np.unique(prediction[use, protein]).size < 2:
            continue
        values[protein] = float(
            spearmanr(prediction[use, protein], target[use, protein]).statistic
        )
    return values


def choose_global_weight(
    local: np.ndarray, global_value: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for global_weight in GLOBAL_WEIGHT_GRID:
        prediction = (1.0 - global_weight) * local + global_weight * global_value
        rho = protein_spearman(prediction, target, mask)
        rows.append(
            {
                "global_weight": float(global_weight),
                "median_oof_spearman": float(np.nanmedian(rho)),
            }
        )
    best = max(rows, key=lambda row: row["median_oof_spearman"])
    return float(best["global_weight"]), rows


def individual_weights(
    local: np.ndarray,
    global_value: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    fallback: float,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.full(target.shape[1], fallback, dtype=np.float64)
    counts = mask.sum(axis=0).astype(np.int64)
    for protein in range(target.shape[1]):
        use = mask[:, protein]
        if use.sum() < 8:
            continue
        best_rho = float("-inf")
        best_weight = fallback
        for global_weight in GLOBAL_WEIGHT_GRID:
            prediction = (
                (1.0 - global_weight) * local[use, protein]
                + global_weight * global_value[use, protein]
            )
            if np.unique(prediction).size < 2:
                continue
            current = float(spearmanr(prediction, target[use, protein]).statistic)
            if np.isfinite(current) and current > best_rho:
                best_rho = current
                best_weight = float(global_weight)
        weights[protein] = best_weight
    return weights, counts


def evaluate(
    name: str,
    prediction: np.ndarray,
    arrays: dev.DevelopmentArrays,
    validation: np.ndarray,
    protein_names: tuple[str, ...],
    top: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame]:
    table, _, summary = dev.validation_tables(
        prediction, arrays, validation, protein_names, 8
    )
    top_values = table.iloc[top]["spearman"].to_numpy(dtype=float)
    record = {
        "candidate": name,
        "median_spearman_all": float(summary["median_spearman"]),
        "median_spearman_high_coverage_1000": float(
            np.nanmedian(top_values)
        ),
        "validation_mse": float(summary["validation_mse"]),
        "validation_batch_cosine": float(
            summary["validation_mean_batch_cosine"]
        ),
        "median_sd_ratio": float(
            summary["median_predicted_to_observed_sd_ratio"]
        ),
    }
    return record, table


def main() -> int:
    args = parse_args()
    if args.components < 1 or args.local_genes < 1 or args.meta_folds < 2:
        raise ValueError("component, local gene and meta-fold counts must be positive")
    if args.global_ridge_alpha < 0 or args.local_ridge_alpha < 0:
        raise ValueError("ridge parameters must be non-negative")
    inputs = base.load_inputs(input_args(args.data_dir))
    protein_names = tuple(map(str, inputs["protein_names"]))
    manifest = inputs["manifest"]
    studies = manifest["pdc_study_id"].astype(str).to_numpy()
    cancer = manifest["cancer_label"].astype(str).to_numpy()
    cases = manifest["case_submitter_id"].astype(str).to_numpy()
    partitions = build_strict_inner_partitions(
        inputs["sample_ids"], cancer, cases, seed=args.seed,
        n_folds=5, inner_folds=5, fold_index=0,
    )
    train = partitions.selection_train
    validation = partitions.selection_validation
    sizes = (train.size, validation.size, partitions.outer_test.size)
    if sizes != (916, 229, 286):
        raise RuntimeError(f"locked split differs from 916/229/286: {sizes}")
    raw = seal_outer_protein_labels(inputs["protein_raw"], partitions.outer_test)
    raw_mask = np.isfinite(raw)
    full_scaler = TrainFittedProteinScale.fit(
        raw, train, feature_names=protein_names, mask=raw_mask,
        lower_quantile=0.01, upper_quantile=0.99,
    )
    target = np.zeros_like(raw, dtype=np.float32)
    mask = np.zeros_like(raw_mask, dtype=bool)
    for indices, clip in ((train, True), (validation, False)):
        target[indices], mask[indices] = transform_subset(
            full_scaler, raw, indices, protein_names, clip=clip
        )
    graph = ProteinGraphArtifact.load(args.protein_graph_artifact)
    parent = base.build_parent_gene_index(inputs)
    prior = build_protein_local_gene_prior(
        inputs["gene_names"], protein_names, parent, graph,
        max_local_genes=args.local_genes,
    )

    oof_local = np.full((train.size, len(protein_names)), np.nan, dtype=np.float32)
    oof_global = np.full_like(oof_local, np.nan)
    train_position = {int(sample): position for position, sample in enumerate(train)}
    folds = make_strict_folds(
        np.asarray(inputs["sample_ids"])[train], cancer[train],
        n_splits=args.meta_folds, random_state=args.seed + 2017,
        blocking_groups=cases[train],
    )
    fold_records: list[dict[str, object]] = []
    for fold_index, fold in enumerate(folds):
        fit = train[fold.train_idx]
        holdout = train[fold.test_idx]
        fold_scaler = TrainFittedProteinScale.fit(
            raw, fit, feature_names=protein_names, mask=raw_mask,
            lower_quantile=0.01, upper_quantile=0.99,
        )
        fold_target = np.zeros_like(raw, dtype=np.float32)
        fold_mask = np.zeros_like(raw_mask, dtype=bool)
        for indices, clip in ((fit, True), (holdout, False)):
            fold_target[indices], fold_mask[indices] = transform_subset(
                fold_scaler, raw, indices, protein_names, clip=clip
            )
        fold_baseline, _ = fit_train_study_protein_baseline(
            fold_target, fold_mask, fit, studies
        )
        fold_rna, _ = fit_transform_train_study_centered_rna(
            inputs["rna"], fit, studies
        )
        weights, biases, local_state = fit_direct_local_ridge_initialization(
            fold_rna, fold_target, fold_mask, fold_baseline, fit, prior,
            alpha=args.local_ridge_alpha,
        )
        local_fold = local_prediction(
            fold_rna, fold_baseline, weights, biases, prior, holdout
        )
        global_fold, explained = fit_global_prediction(
            fold_rna, fold_target, fold_mask, fold_baseline, fit, holdout,
            components=args.components, alpha=args.global_ridge_alpha,
            seed=args.seed + fold_index,
        )
        positions = np.asarray([train_position[int(i)] for i in holdout])
        oof_local[positions] = convert_scale(local_fold, fold_scaler, full_scaler)
        oof_global[positions] = convert_scale(global_fold, fold_scaler, full_scaler)
        fold_records.append(
            {
                "fold": fold_index,
                "fit_samples": int(fit.size),
                "holdout_samples": int(holdout.size),
                "local_fitted_proteins": int(local_state["fitted_proteins"]),
                "explained_singular_value_fraction": explained,
            }
        )
    if not np.isfinite(oof_local).all() or not np.isfinite(oof_global).all():
        raise RuntimeError("out-of-fold prediction matrix is incomplete")

    global_weight, global_screen = choose_global_weight(
        oof_local, oof_global, target[train], mask[train]
    )
    raw_individual, observation_counts = individual_weights(
        oof_local, oof_global, target[train], mask[train], global_weight
    )
    shrink_screen: list[dict[str, float]] = []
    best_shrinkage = SHRINKAGE_GRID[0]
    best_oof_median = float("-inf")
    for shrinkage in SHRINKAGE_GRID:
        reliability = observation_counts / (observation_counts + shrinkage)
        weights = global_weight + reliability * (raw_individual - global_weight)
        prediction = (
            (1.0 - weights[None, :]) * oof_local
            + weights[None, :] * oof_global
        )
        rho = protein_spearman(prediction, target[train], mask[train])
        current = float(np.nanmedian(rho))
        shrink_screen.append(
            {"shrinkage": float(shrinkage), "median_oof_spearman": current}
        )
        if current > best_oof_median:
            best_oof_median = current
            best_shrinkage = shrinkage
    reliability = observation_counts / (observation_counts + best_shrinkage)
    protein_weights = global_weight + reliability * (
        raw_individual - global_weight
    )

    full_baseline, _ = fit_train_study_protein_baseline(
        target, mask, train, studies
    )
    full_rna, _ = fit_transform_train_study_centered_rna(
        inputs["rna"], train, studies
    )
    weights, biases, full_local_state = fit_direct_local_ridge_initialization(
        full_rna, target, mask, full_baseline, train, prior,
        alpha=args.local_ridge_alpha,
    )
    validation_local = local_prediction(
        full_rna, full_baseline, weights, biases, prior, validation
    )
    validation_global, full_explained = fit_global_prediction(
        full_rna, target, mask, full_baseline, train, validation,
        components=args.components, alpha=args.global_ridge_alpha,
        seed=args.seed,
    )
    predictions = {
        "oof_global_weight": (
            (1.0 - global_weight) * validation_local
            + global_weight * validation_global
        ).astype(np.float32),
        "oof_protein_weight_shrunk": (
            (1.0 - protein_weights[None, :]) * validation_local
            + protein_weights[None, :] * validation_global
        ).astype(np.float32),
    }
    arrays = dev.DevelopmentArrays(
        rna=full_rna, target=target, mask=mask,
        cancer_index=np.zeros(target.shape[0], dtype=np.int64),
    )
    coverage = mask[train].sum(axis=0)
    top = np.argsort(-coverage, kind="stable")[:1000]
    validation_records: list[dict[str, object]] = []
    tables: dict[str, pd.DataFrame] = {}
    for name, prediction in predictions.items():
        record, table = evaluate(
            name, prediction, arrays, validation, protein_names, top
        )
        validation_records.append(record)
        tables[name] = table
    selected = max(validation_records, key=lambda row: row["median_spearman_all"])
    selected_name = str(selected["candidate"])
    result = {
        "seed": args.seed,
        "split_sizes": list(map(int, sizes)),
        "meta_folds": args.meta_folds,
        "components": args.components,
        "global_ridge_alpha": args.global_ridge_alpha,
        "local_ridge_alpha": args.local_ridge_alpha,
        "local_genes": args.local_genes,
        "folds": fold_records,
        "full_explained_singular_value_fraction": full_explained,
        "full_local_fitted_proteins": int(full_local_state["fitted_proteins"]),
        "oof_global_weight": global_weight,
        "oof_global_weight_screen": global_screen,
        "oof_selected_shrinkage": float(best_shrinkage),
        "oof_shrinkage_screen": shrink_screen,
        "protein_global_weight_quantiles": {
            str(q): float(np.quantile(protein_weights, q))
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
        "locked_229_validation": validation_records,
        "selected_by_locked_229_median_spearman": selected,
        "outer_test_evaluated": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(global_screen).to_csv(
        args.output_dir / "oof_global_weight_screen.tsv", sep="\t", index=False
    )
    pd.DataFrame(shrink_screen).to_csv(
        args.output_dir / "oof_shrinkage_screen.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {
            "protein": protein_names,
            "train_observations": observation_counts,
            "raw_oof_global_weight": raw_individual,
            "shrunk_oof_global_weight": protein_weights,
        }
    ).to_csv(args.output_dir / "protein_blend_weights.tsv", sep="\t", index=False)
    for name, table in tables.items():
        table.to_csv(
            args.output_dir / f"{name}_per_protein.tsv", sep="\t", index=False
        )
    np.savez_compressed(
        args.output_dir / "selected_validation_predictions.npz",
        prediction=predictions[selected_name],
        validation_indices=validation,
        protein_names=np.asarray(protein_names, dtype=str),
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
