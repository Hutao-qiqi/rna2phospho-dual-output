"""Select protein-specific study-calibration shrinkage from train-only cross-fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_protein as base
import train_hybrid_protein_development_screen as dev
from audit_locked_oof_study_calibration import selected_oof_prediction
from audit_locked_oof_string_pca_blend import input_args, protein_spearman, transform_subset
from candidate_groupwise_rna_normalization import (
    build_strict_inner_partitions,
    seal_outer_protein_labels,
)
from hybrid_protein_training_contract import TrainFittedProteinScale
from splits import make_strict_folds
from train_cptac_pretrain_tcpa_finetune import fit_train_study_protein_baseline


DEFAULT_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--oof-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--meta-folds", type=int, default=5)
    parser.add_argument("--shrinkage-grid", type=float, nargs="+", default=list(DEFAULT_GRID))
    parser.add_argument("--default-shrinkage", type=float, default=0.5)
    parser.add_argument("--minimum-oof-improvement", type=float, default=0.005)
    return parser.parse_args()


def fit_group_calibration(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    study_index: np.ndarray,
    rows: np.ndarray,
    n_studies: int,
) -> dict[str, np.ndarray]:
    """Fit study means and centered slopes on the requested row positions."""

    x = np.asarray(prediction[rows], dtype=np.float64)
    y = np.asarray(target[rows], dtype=np.float64)
    observed = np.asarray(mask[rows], dtype=bool)
    groups = np.asarray(study_index[rows], dtype=np.int64)
    n_proteins = target.shape[1]
    count = np.zeros((n_studies, n_proteins), dtype=np.int32)
    x_mean = np.zeros((n_studies, n_proteins), dtype=np.float64)
    y_mean = np.zeros_like(x_mean)
    numerator = np.zeros_like(x_mean)
    denominator = np.zeros_like(x_mean)
    global_count = observed.sum(axis=0).astype(np.int32)
    safe_global = np.maximum(global_count, 1)
    global_x_mean = np.where(observed, x, 0.0).sum(axis=0) / safe_global
    global_y_mean = np.where(observed, y, 0.0).sum(axis=0) / safe_global
    for group in range(n_studies):
        use_rows = groups == group
        if not use_rows.any():
            continue
        current_mask = observed[use_rows]
        current_x = x[use_rows]
        current_y = y[use_rows]
        current_count = current_mask.sum(axis=0).astype(np.int32)
        safe_count = np.maximum(current_count, 1)
        current_x_mean = np.where(current_mask, current_x, 0.0).sum(axis=0) / safe_count
        current_y_mean = np.where(current_mask, current_y, 0.0).sum(axis=0) / safe_count
        centered_x = current_x - current_x_mean[None, :]
        centered_y = current_y - current_y_mean[None, :]
        count[group] = current_count
        x_mean[group] = current_x_mean
        y_mean[group] = current_y_mean
        numerator[group] = np.where(
            current_mask, centered_x * centered_y, 0.0
        ).sum(axis=0)
        denominator[group] = np.where(
            current_mask, np.square(centered_x), 0.0
        ).sum(axis=0)
    total_numerator = numerator.sum(axis=0)
    total_denominator = denominator.sum(axis=0)
    global_slope = np.divide(
        total_numerator,
        total_denominator,
        out=np.ones(n_proteins, dtype=np.float64),
        where=total_denominator > 1e-12,
    )
    return {
        "count": count,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "numerator": numerator,
        "denominator": denominator,
        "global_count": global_count,
        "global_x_mean": global_x_mean,
        "global_y_mean": global_y_mean,
        "global_slope": global_slope,
    }


def apply_group_calibration(
    prediction: np.ndarray,
    study_index: np.ndarray,
    rows: np.ndarray,
    state: dict[str, np.ndarray],
    shrinkage: float | np.ndarray,
) -> np.ndarray:
    """Apply a fitted calibration to rows, with scalar or protein shrinkage."""

    values = np.asarray(prediction[rows], dtype=np.float64)
    groups = np.asarray(study_index[rows], dtype=np.int64)
    n_proteins = values.shape[1]
    shrink = np.broadcast_to(np.asarray(shrinkage, dtype=np.float64), (n_proteins,))
    slopes = np.divide(
        state["numerator"] + shrink[None, :] * state["global_slope"][None, :],
        state["denominator"] + shrink[None, :],
        out=np.broadcast_to(state["global_slope"], state["numerator"].shape).copy(),
        where=(state["denominator"] + shrink[None, :]) > 1e-12,
    )
    output = np.empty_like(values)
    for row_offset, group in enumerate(groups):
        if group >= 0:
            supported = state["count"][group] >= 3
            group_value = state["y_mean"][group] + slopes[group] * (
                values[row_offset] - state["x_mean"][group]
            )
            global_value = state["global_y_mean"] + state["global_slope"] * (
                values[row_offset] - state["global_x_mean"]
            )
            output[row_offset] = np.where(supported, group_value, global_value)
        else:
            output[row_offset] = state["global_y_mean"] + state["global_slope"] * (
                values[row_offset] - state["global_x_mean"]
            )
    return output.astype(np.float32)


def choose_protein_shrinkage(
    scores: np.ndarray,
    grid: np.ndarray,
    default_shrinkage: float,
    minimum_improvement: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose per-protein shrinkage, retaining the default for weak OOF gains."""

    default_index = int(np.argmin(np.abs(grid - default_shrinkage)))
    if not np.isclose(grid[default_index], default_shrinkage):
        raise ValueError("default shrinkage must occur in the shrinkage grid")
    safe = np.where(np.isfinite(scores), scores, -np.inf)
    best_index = np.argmax(safe, axis=0)
    best_score = safe[best_index, np.arange(scores.shape[1])]
    default_score = safe[default_index]
    keep_default = (
        ~np.isfinite(best_score)
        | ~np.isfinite(default_score)
        | (best_score < default_score + minimum_improvement)
    )
    best_index[keep_default] = default_index
    return grid[best_index].astype(np.float32), best_index.astype(np.int32)


def main() -> int:
    args = parse_args()
    grid = np.asarray(args.shrinkage_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2 or np.any(grid < 0) or args.meta_folds < 2:
        raise ValueError("adaptive calibration controls are invalid")
    summary = json.loads((args.oof_result_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("outer_test_evaluated") is not False:
        raise ValueError("source OOF result used outer labels")
    oof_prediction, saved_train, saved_names, source_name = selected_oof_prediction(
        args.oof_result_dir, summary
    )
    validation_saved = np.load(
        args.oof_result_dir / "selected_validation_predictions.npz", allow_pickle=False
    )
    validation_prediction = np.asarray(validation_saved["prediction"], dtype=np.float32)

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

    train_studies = studies[train]
    unique_studies = tuple(sorted(set(train_studies.tolist())))
    study_to_index = {study: index for index, study in enumerate(unique_studies)}
    train_study_index = np.asarray([study_to_index[x] for x in train_studies], dtype=np.int64)
    validation_study_index = np.asarray(
        [study_to_index.get(x, -1) for x in studies[validation]], dtype=np.int64
    )
    train_target = target[train]
    train_mask = mask[train]
    folds = make_strict_folds(
        np.asarray(inputs["sample_ids"])[train], cancer[train],
        n_splits=args.meta_folds, random_state=args.seed + 2017,
        blocking_groups=cases[train],
    )
    crossfit = np.full(
        (grid.size, train.size, len(protein_names)), np.nan, dtype=np.float32
    )
    for fold_index, fold in enumerate(folds):
        state = fit_group_calibration(
            oof_prediction, train_target, train_mask, train_study_index,
            np.asarray(fold.train_idx, dtype=np.int64), len(unique_studies),
        )
        holdout = np.asarray(fold.test_idx, dtype=np.int64)
        for grid_index, shrinkage in enumerate(grid):
            crossfit[grid_index, holdout] = apply_group_calibration(
                oof_prediction, train_study_index, holdout, state, float(shrinkage)
            )
        print(json.dumps({"calibration_fold": fold_index, "holdout_samples": int(holdout.size)}), flush=True)
    scores = np.stack(
        [protein_spearman(crossfit[index], train_target, train_mask) for index in range(grid.size)]
    )
    selected_shrinkage, selected_index = choose_protein_shrinkage(
        scores, grid, args.default_shrinkage, args.minimum_oof_improvement
    )

    full_rows = np.arange(train.size, dtype=np.int64)
    full_state = fit_group_calibration(
        oof_prediction, train_target, train_mask, train_study_index,
        full_rows, len(unique_studies),
    )
    adaptive_prediction = apply_group_calibration(
        validation_prediction, validation_study_index,
        np.arange(validation.size, dtype=np.int64), full_state, selected_shrinkage,
    )
    baseline, _ = fit_train_study_protein_baseline(target, mask, train, studies)
    # The group calibrator estimates the complete study intercept, so baseline is
    # retained only as a separately audited reference from the original method.
    del baseline
    arrays = dev.DevelopmentArrays(
        rna=np.zeros((target.shape[0], 1), dtype=np.float32), target=target, mask=mask,
        cancer_index=np.zeros(target.shape[0], dtype=np.int64),
    )
    coverage = mask[train].sum(axis=0)
    high = np.argsort(-coverage, kind="stable")[:1000]
    candidates = {
        "uncalibrated_three_branch": validation_prediction,
        "adaptive_crossfit_study_calibration": adaptive_prediction,
    }
    records: list[dict[str, object]] = []
    tables: dict[str, pd.DataFrame] = {}
    for name, prediction in candidates.items():
        table, _, metric = dev.validation_tables(
            prediction, arrays, validation, protein_names, 8
        )
        records.append({
            "candidate": name,
            "median_spearman_all": float(metric["median_spearman"]),
            "median_spearman_high_coverage_1000": float(
                np.nanmedian(table.iloc[high]["spearman"].to_numpy(dtype=float))
            ),
            "validation_mse": float(metric["validation_mse"]),
            "validation_batch_cosine": float(metric["validation_mean_batch_cosine"]),
            "median_sd_ratio": float(metric["median_predicted_to_observed_sd_ratio"]),
        })
        tables[name] = table
    selected = max(records, key=lambda row: row["median_spearman_all"])
    selected_name = str(selected["candidate"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_dir / "calibration_screen.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "protein": protein_names,
            "train_observations": mask[train].sum(axis=0),
            "selected_shrinkage": selected_shrinkage,
            **{f"oof_spearman_shrinkage_{value:g}": scores[index] for index, value in enumerate(grid)},
        }
    ).to_csv(args.output_dir / "protein_shrinkage_selection.tsv", sep="\t", index=False)
    tables[selected_name].to_csv(args.output_dir / "selected_per_protein.tsv", sep="\t", index=False)
    np.savez_compressed(
        args.output_dir / "selected_validation_predictions.npz",
        prediction=candidates[selected_name], validation_indices=validation,
        protein_names=np.asarray(protein_names, dtype=str),
    )
    result = {
        "seed": args.seed,
        "split_sizes": list(map(int, sizes)),
        "source_selected_candidate": source_name,
        "meta_folds": args.meta_folds,
        "shrinkage_grid": grid.tolist(),
        "default_shrinkage": args.default_shrinkage,
        "minimum_oof_improvement": args.minimum_oof_improvement,
        "selection_source": "selection_train_916_crossfit_predictions_only",
        "selected_shrinkage_counts": {
            f"{value:g}": int(np.sum(selected_index == index))
            for index, value in enumerate(grid)
        },
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
