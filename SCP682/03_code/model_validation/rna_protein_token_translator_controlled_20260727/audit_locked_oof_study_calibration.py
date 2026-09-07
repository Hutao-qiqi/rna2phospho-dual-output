"""Calibrate a selected three-branch predictor using train-only OOF study slopes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_protein as base
import train_hybrid_protein_development_screen as dev
from audit_locked_oof_string_pca_blend import input_args, transform_subset
from candidate_groupwise_rna_normalization import (
    build_strict_inner_partitions,
    seal_outer_protein_labels,
)
from hybrid_protein_training_contract import TrainFittedProteinScale
from train_cptac_pretrain_tcpa_finetune import fit_train_study_protein_baseline


SHRINKAGE_GRID = (1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--oof-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument(
        "--shrinkage-grid",
        type=float,
        nargs="+",
        default=list(SHRINKAGE_GRID),
    )
    return parser.parse_args()


def selected_oof_prediction(
    result_dir: Path,
    summary: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], str]:
    saved = np.load(result_dir / "oof_predictions.npz", allow_pickle=False)
    train = np.asarray(saved["train_indices"], dtype=np.int64)
    protein_names = tuple(map(str, saved["protein_names"].tolist()))
    selected = summary["selected_by_locked_229_median_spearman"]
    if not isinstance(selected, dict):
        raise ValueError("OOF summary selected candidate is invalid")
    name = str(selected["candidate"])
    if "selected" in saved.files:
        prediction = np.asarray(saved["selected"], dtype=np.float32)
        if prediction.shape != (train.size, len(protein_names)):
            raise ValueError("generic selected OOF prediction dimensions differ")
        return prediction, train, protein_names, name
    base_prediction = np.asarray(saved["two_branch_base"], dtype=np.float32)
    supervised = np.asarray(saved["supervised"], dtype=np.float32)
    if name == "oof_two_branch_base":
        prediction = base_prediction
    elif name == "oof_three_branch_global_weight":
        weight = float(summary["supervised_oof_global_weight"])
        prediction = (1.0 - weight) * base_prediction + weight * supervised
    elif name == "oof_three_branch_protein_weight_shrunk":
        weights = pd.read_csv(
            result_dir / "protein_blend_weights.tsv", sep="\t"
        )["supervised_weight"].to_numpy(dtype=np.float32)
        if weights.shape != (base_prediction.shape[1],):
            raise ValueError("OOF supervised weight dimensions differ")
        prediction = (
            (1.0 - weights[None, :]) * base_prediction
            + weights[None, :] * supervised
        )
    else:
        raise ValueError(f"unsupported selected OOF candidate: {name}")
    return prediction.astype(np.float32), train, protein_names, name


def main() -> int:
    args = parse_args()
    summary = json.loads(
        (args.oof_result_dir / "summary.json").read_text(encoding="utf-8")
    )
    if summary.get("outer_test_evaluated") is not False:
        raise ValueError("source OOF result used outer labels")
    oof_prediction, saved_train, saved_names, source_name = selected_oof_prediction(
        args.oof_result_dir, summary
    )
    validation_saved = np.load(
        args.oof_result_dir / "selected_validation_predictions.npz",
        allow_pickle=False,
    )
    validation_prediction = np.asarray(
        validation_saved["prediction"], dtype=np.float32
    )

    inputs = base.load_inputs(input_args(args.data_dir))
    manifest = inputs["manifest"]
    cancer = manifest["cancer_label"].astype(str).to_numpy()
    cases = manifest["case_submitter_id"].astype(str).to_numpy()
    studies = manifest["pdc_study_id"].astype(str).to_numpy()
    partitions = build_strict_inner_partitions(
        inputs["sample_ids"], cancer, cases, seed=args.seed,
        n_folds=5, inner_folds=5, fold_index=0,
    )
    train = partitions.selection_train
    validation = partitions.selection_validation
    sizes = (train.size, validation.size, partitions.outer_test.size)
    if sizes != (916, 229, 286):
        raise RuntimeError(f"locked split differs from 916/229/286: {sizes}")
    protein_names = tuple(map(str, inputs["protein_names"]))
    if not np.array_equal(saved_train, train) or saved_names != protein_names:
        raise ValueError("OOF prediction split or protein order differs")
    if not np.array_equal(validation_saved["validation_indices"], validation):
        raise ValueError("OOF validation indices differ")
    if tuple(map(str, validation_saved["protein_names"].tolist())) != protein_names:
        raise ValueError("OOF validation protein order differs")

    raw = seal_outer_protein_labels(inputs["protein_raw"], partitions.outer_test)
    raw_mask = np.isfinite(raw)
    scaler = TrainFittedProteinScale.fit(
        raw, train, feature_names=protein_names, mask=raw_mask,
        lower_quantile=0.01, upper_quantile=0.99,
    )
    target = np.zeros_like(raw, dtype=np.float32)
    mask = np.zeros_like(raw_mask, dtype=bool)
    for indices, clip in ((train, True), (validation, False)):
        target[indices], mask[indices] = transform_subset(
            scaler, raw, indices, protein_names, clip=clip
        )
    baseline, _ = fit_train_study_protein_baseline(
        target, mask, train, studies
    )
    train_studies = studies[train]
    validation_studies = studies[validation]
    unique_studies = tuple(sorted(set(train_studies.tolist())))
    study_to_index = {study: index for index, study in enumerate(unique_studies)}
    train_study_index = np.asarray(
        [study_to_index[study] for study in train_studies], dtype=np.int64
    )
    validation_study_index = np.asarray(
        [study_to_index.get(study, -1) for study in validation_studies],
        dtype=np.int64,
    )
    n_studies = len(unique_studies)
    n_proteins = len(protein_names)
    global_slope = np.ones(n_proteins, dtype=np.float32)
    numerator = np.zeros((n_studies, n_proteins), dtype=np.float32)
    denominator = np.zeros_like(numerator)
    prediction_mean = np.zeros_like(numerator)
    fitted = np.zeros(n_proteins, dtype=bool)
    train_baseline = baseline[train]
    target_residual = target[train] - train_baseline
    prediction_residual = oof_prediction - train_baseline

    for protein in range(n_proteins):
        observed = mask[train, protein]
        if int(observed.sum()) < 8:
            continue
        total_numerator = 0.0
        total_denominator = 0.0
        for study_index in range(n_studies):
            use = observed & (train_study_index == study_index)
            if not use.any():
                continue
            mean = float(prediction_residual[use, protein].mean())
            x = prediction_residual[use, protein].astype(np.float64) - mean
            y = target_residual[use, protein].astype(np.float64)
            current_numerator = float(x @ y)
            current_denominator = float(x @ x)
            prediction_mean[study_index, protein] = mean
            numerator[study_index, protein] = current_numerator
            denominator[study_index, protein] = current_denominator
            total_numerator += current_numerator
            total_denominator += current_denominator
        if total_denominator > 1e-12:
            global_slope[protein] = np.float32(
                total_numerator / total_denominator
            )
            fitted[protein] = True

    arrays = dev.DevelopmentArrays(
        rna=np.zeros((target.shape[0], 1), dtype=np.float32),
        target=target,
        mask=mask,
        cancer_index=np.zeros(target.shape[0], dtype=np.int64),
    )
    coverage = mask[train].sum(axis=0)
    high = np.argsort(-coverage, kind="stable")[:1000]
    predictions: dict[str, np.ndarray] = {
        "uncalibrated_three_branch": validation_prediction
    }
    shrinkage_grid = tuple(float(value) for value in args.shrinkage_grid)
    if not shrinkage_grid or any(value < 0 for value in shrinkage_grid):
        raise ValueError("shrinkage grid must contain nonnegative values")
    for shrinkage in shrinkage_grid:
        slopes = np.divide(
            numerator + shrinkage * global_slope[None, :],
            denominator + shrinkage,
            out=np.broadcast_to(global_slope, numerator.shape).copy(),
            where=(denominator + shrinkage) > 1e-12,
        )
        prediction = baseline[validation].copy()
        residual = validation_prediction - baseline[validation]
        for study_index in range(n_studies):
            rows = validation_study_index == study_index
            if not rows.any():
                continue
            prediction[rows] += (
                residual[rows] - prediction_mean[study_index][None, :]
            ) * slopes[study_index][None, :]
        unseen = validation_study_index < 0
        if unseen.any():
            prediction[unseen] += residual[unseen] * global_slope[None, :]
        predictions[f"oof_study_calibration_shrinkage_{shrinkage:g}"] = (
            prediction.astype(np.float32)
        )

    records: list[dict[str, object]] = []
    tables: dict[str, pd.DataFrame] = {}
    for name, prediction in predictions.items():
        table, _, metric = dev.validation_tables(
            prediction, arrays, validation, protein_names, 8
        )
        high_values = table.iloc[high]["spearman"].to_numpy(dtype=float)
        records.append(
            {
                "candidate": name,
                "median_spearman_all": float(metric["median_spearman"]),
                "median_spearman_high_coverage_1000": float(
                    np.nanmedian(high_values)
                ),
                "validation_mse": float(metric["validation_mse"]),
                "validation_batch_cosine": float(
                    metric["validation_mean_batch_cosine"]
                ),
                "median_sd_ratio": float(
                    metric["median_predicted_to_observed_sd_ratio"]
                ),
            }
        )
        tables[name] = table
    selected = max(records, key=lambda row: row["median_spearman_all"])
    selected_name = str(selected["candidate"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(
        args.output_dir / "calibration_screen.tsv", sep="\t", index=False
    )
    tables[selected_name].to_csv(
        args.output_dir / "selected_per_protein.tsv", sep="\t", index=False
    )
    np.savez_compressed(
        args.output_dir / "train_oof_calibration_parameters.npz",
        global_slope=global_slope,
        study_numerator=numerator,
        study_denominator=denominator,
        prediction_mean=prediction_mean,
        fitted=fitted,
        study_names=np.asarray(unique_studies, dtype=str),
        protein_names=np.asarray(protein_names, dtype=str),
    )
    np.savez_compressed(
        args.output_dir / "selected_validation_predictions.npz",
        prediction=predictions[selected_name],
        validation_indices=validation,
        protein_names=np.asarray(protein_names, dtype=str),
    )
    result = {
        "seed": args.seed,
        "split_sizes": list(map(int, sizes)),
        "source_selected_candidate": source_name,
        "training_studies": n_studies,
        "fitted_proteins": int(fitted.sum()),
        "calibration_fit_source": "selection_train_916_out_of_fold_predictions",
        "screen": records,
        "selected_by_locked_229_median_spearman": selected,
        "outer_test_evaluated": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
