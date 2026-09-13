#!/usr/bin/env python3
"""M2.2 + PT7 的研究×位点折外收缩分位数校准。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
METRIC_SCRIPT = ROOT / "03_code/model_validation/scp682_v2_m22_study_centered_evaluation/02_rerun_external_pt7_centered.py"
EXTERNAL_INPUT = ROOT / "01_data/bulk/intermediate/independent_external_posttraining_current_model_inputs_20260815"
BASE_PREDICTION = ROOT / "02_results/external_validation/scp682_v2_m22_f_pt7_fusion_20260831_v2/predictions/fusion_selected_predictions.npz"
OUT = ROOT / "02_results/external_validation/scp682_v2_m22_pt7_quantile_calibration_20260831"
LAMBDAS = (0.0, 0.25, 0.50, 0.75, 1.00)
MIN_OBSERVED = 10
QUANTILE_GRID = np.linspace(0.0, 1.0, 41)


def load_metrics_module():
    spec = importlib.util.spec_from_file_location("centered_metrics", METRIC_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {METRIC_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_quantile_nodes(prediction: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    observed = np.isfinite(prediction) & np.isfinite(truth)
    if int(observed.sum()) < MIN_OBSERVED:
        return None
    x = np.quantile(prediction[observed], QUANTILE_GRID)
    y = np.quantile(truth[observed], QUANTILE_GRID)
    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) < 4:
        return None
    unique_y = np.zeros(len(unique_x), dtype=np.float64)
    count = np.zeros(len(unique_x), dtype=np.float64)
    np.add.at(unique_y, inverse, y)
    np.add.at(count, inverse, 1.0)
    unique_y /= count
    unique_y = np.maximum.accumulate(unique_y)
    return unique_x, unique_y


def apply_nodes(values: np.ndarray, nodes: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    x, y = nodes
    mapped = np.interp(values, x, y)
    if len(x) >= 2:
        left_dx = max(float(x[1] - x[0]), 1e-6)
        right_dx = max(float(x[-1] - x[-2]), 1e-6)
        left_slope = float(np.clip((y[1] - y[0]) / left_dx, 0.1, 5.0))
        right_slope = float(np.clip((y[-1] - y[-2]) / right_dx, 0.1, 5.0))
        left = values < x[0]
        right = values > x[-1]
        mapped[left] = y[0] + left_slope * (values[left] - x[0])
        mapped[right] = y[-1] + right_slope * (values[right] - x[-1])
    return mapped


def map_fold(
    prediction: np.ndarray,
    truth: np.ndarray,
    studies: np.ndarray,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
) -> np.ndarray:
    output = prediction[apply_rows].copy().astype(np.float32)
    for study in np.unique(studies[apply_rows]):
        fit = fit_rows[studies[fit_rows] == study]
        held_local = np.flatnonzero(studies[apply_rows] == study)
        held = apply_rows[held_local]
        if len(fit) < MIN_OBSERVED:
            continue
        for site in range(prediction.shape[1]):
            nodes = fit_quantile_nodes(prediction[fit, site], truth[fit, site])
            if nodes is None:
                continue
            valid = np.isfinite(prediction[held, site])
            if not valid.any():
                continue
            output[held_local[valid], site] = apply_nodes(prediction[held[valid], site], nodes)
    return output


def crossfit_mapping(
    prediction: np.ndarray,
    truth: np.ndarray,
    studies: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    output = np.full_like(prediction, np.nan, dtype=np.float32)
    rows = np.arange(len(prediction))
    for fold in sorted(np.unique(folds)):
        fit = rows[folds != fold]
        held = rows[folds == fold]
        output[held] = map_fold(prediction, truth, studies, fit, held)
        print(f"completed quantile fold {int(fold) + 1}/{len(np.unique(folds))}", flush=True)
    return output


def full_mapping(
    calibration_prediction: np.ndarray,
    calibration_truth: np.ndarray,
    calibration_studies: np.ndarray,
    test_prediction: np.ndarray,
    test_studies: np.ndarray,
) -> np.ndarray:
    combined_prediction = np.concatenate([calibration_prediction, test_prediction], axis=0)
    combined_truth = np.concatenate(
        [calibration_truth, np.full_like(test_prediction, np.nan, dtype=np.float32)], axis=0
    )
    combined_studies = np.concatenate([calibration_studies, test_studies])
    fit = np.arange(len(calibration_prediction))
    apply_rows = np.arange(len(calibration_prediction), len(combined_prediction))
    return map_fold(combined_prediction, combined_truth, combined_studies, fit, apply_rows)


def centered_dynamic_range(module, truth: np.ndarray, prediction: np.ndarray, studies: np.ndarray) -> tuple[float, float]:
    y, p = module.center_pair_within_study(truth, prediction, studies)
    observed = np.isfinite(y) & np.isfinite(p)
    global_ratio = float(np.nanstd(np.where(observed, p, np.nan)) / np.nanstd(np.where(observed, y, np.nan)))
    ratios = []
    for site in range(y.shape[1]):
        valid = observed[:, site]
        if valid.sum() < 10:
            continue
        true_sd = float(np.std(y[valid, site]))
        if true_sd > 1e-8:
            ratios.append(float(np.std(p[valid, site])) / true_sd)
    return global_ratio, float(np.median(ratios))


def metric_rows(module, truth, base, mapped, studies, scope):
    rows = []
    for lam in LAMBDAS:
        prediction = base + lam * (mapped - base)
        for row in module.metric_pair(truth, prediction, studies, f"quantile_lambda_{lam:g}", scope):
            global_ratio, median_ratio = centered_dynamic_range(module, truth, prediction, studies)
            rows.append(
                {
                    "lambda": lam,
                    **row,
                    "centered_global_sd_ratio": global_ratio,
                    "centered_median_site_sd_ratio": median_ratio,
                }
            )
    return rows


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    for name in ("tables", "predictions", "reports", "logs"):
        (OUT / name).mkdir(parents=True, exist_ok=True)

    module = load_metrics_module()
    package = np.load(BASE_PREDICTION, allow_pickle=True)
    calibration_ids = package["calibration_ids"].astype(str)
    test_ids = package["test_ids"].astype(str)
    sites = package["site_ids"].astype(str)
    base_oof = package["m2_oof"].astype(np.float32)
    base_test = package["selected_test"].astype(np.float32)

    truth_frame = pd.read_parquet(EXTERNAL_INPUT / "phosphosite_logscale_aligned.parquet")
    metadata = pd.read_csv(EXTERNAL_INPUT / "sample_metadata.tsv", sep="\t").set_index("sample_id")
    truth_oof = truth_frame.reindex(index=calibration_ids, columns=sites).to_numpy(np.float32)
    truth_test = truth_frame.reindex(index=test_ids, columns=sites).to_numpy(np.float32)
    studies_oof = metadata.reindex(calibration_ids).study.astype(str).to_numpy()
    studies_test = metadata.reindex(test_ids).study.astype(str).to_numpy()
    fold_file = module.PROTOCOL / "calibration_fold_assignments.tsv"
    folds = pd.read_csv(fold_file, sep="\t").set_index("sample_id").reindex(calibration_ids).fold.to_numpy(np.int64)

    mapped_oof = crossfit_mapping(base_oof, truth_oof, studies_oof, folds)
    mapped_test = full_mapping(base_oof, truth_oof, studies_oof, base_test, studies_test)

    table_oof = pd.DataFrame(metric_rows(module, truth_oof, base_oof, mapped_oof, studies_oof, "external_247_OOF"))
    centered_oof = table_oof.query("coordinate == 'study_centered'").copy()
    centered_oof["selection_score"] = 0.5 * (
        centered_oof.patient_pearson_median + centered_oof.site_pearson_median
    )
    selected_lambda = float(centered_oof.sort_values(
        ["selection_score", "patient_pearson_median", "site_pearson_median"], ascending=False
    ).iloc[0]["lambda"])
    table_test = pd.DataFrame(metric_rows(module, truth_test, base_test, mapped_test, studies_test, "external_106_locked_test"))
    table_oof["selected"] = table_oof["lambda"].eq(selected_lambda)
    table_test["selected"] = table_test["lambda"].eq(selected_lambda)
    table_oof.to_csv(OUT / "tables/quantile_calibration_247_oof.tsv", sep="\t", index=False)
    table_test.to_csv(OUT / "tables/quantile_calibration_106_test.tsv", sep="\t", index=False)

    selected_test = base_test + selected_lambda * (mapped_test - base_test)
    np.savez_compressed(
        OUT / "predictions/m22_pt7_quantile_calibrated_predictions.npz",
        selected_lambda=np.asarray(selected_lambda),
        calibration_ids=calibration_ids,
        test_ids=test_ids,
        site_ids=sites,
        base_oof=base_oof,
        mapped_oof=mapped_oof,
        base_test=base_test,
        mapped_test=mapped_test,
        selected_test=selected_test.astype(np.float32),
    )

    chosen_oof = table_oof.query("selected and coordinate == 'study_centered'").iloc[0]
    chosen_test = table_test.query("selected and coordinate == 'study_centered'").iloc[0]
    baseline_oof = table_oof.loc[
        table_oof["lambda"].eq(0) & table_oof.coordinate.eq("study_centered")
    ].iloc[0]
    baseline_test = table_test.loc[
        table_test["lambda"].eq(0) & table_test.coordinate.eq("study_centered")
    ].iloc[0]
    report = "\n".join(
        [
            "# M2.2 + PT7 收缩分位数校准",
            "",
            f"247例折外选择 lambda = `{selected_lambda:g}`。",
            "",
            "| 数据 | 模型 | 患者 Pearson | 患者 Spearman | 位点 Pearson | 位点 Spearman | MSE | 全局SD比 | 位点SD比中位数 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| 247例 | M2.2 + PT7 | {baseline_oof.patient_pearson_median:.6f} | {baseline_oof.patient_spearman_median:.6f} | {baseline_oof.site_pearson_median:.6f} | {baseline_oof.site_spearman_median:.6f} | {baseline_oof.patient_equal_mse:.6f} | {baseline_oof.centered_global_sd_ratio:.6f} | {baseline_oof.centered_median_site_sd_ratio:.6f} |",
            f"| 247例 | 分位数校准 | {chosen_oof.patient_pearson_median:.6f} | {chosen_oof.patient_spearman_median:.6f} | {chosen_oof.site_pearson_median:.6f} | {chosen_oof.site_spearman_median:.6f} | {chosen_oof.patient_equal_mse:.6f} | {chosen_oof.centered_global_sd_ratio:.6f} | {chosen_oof.centered_median_site_sd_ratio:.6f} |",
            f"| 106例 | M2.2 + PT7 | {baseline_test.patient_pearson_median:.6f} | {baseline_test.patient_spearman_median:.6f} | {baseline_test.site_pearson_median:.6f} | {baseline_test.site_spearman_median:.6f} | {baseline_test.patient_equal_mse:.6f} | {baseline_test.centered_global_sd_ratio:.6f} | {baseline_test.centered_median_site_sd_ratio:.6f} |",
            f"| 106例 | 分位数校准 | {chosen_test.patient_pearson_median:.6f} | {chosen_test.patient_spearman_median:.6f} | {chosen_test.site_pearson_median:.6f} | {chosen_test.site_spearman_median:.6f} | {chosen_test.patient_equal_mse:.6f} | {chosen_test.centered_global_sd_ratio:.6f} | {chosen_test.centered_median_site_sd_ratio:.6f} |",
            "",
        ]
    )
    (OUT / "RESULTS_20260831.md").write_text(report, encoding="utf-8")
    (OUT / "reports/run_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "selected_lambda": selected_lambda,
                "minimum_observed_per_study_site": MIN_OBSERVED,
                "quantile_nodes": len(QUANTILE_GRID),
                "selection_metric": "mean of centered patient Pearson and centered site Pearson on 247 OOF",
                "selected_247": chosen_oof.to_dict(),
                "selected_106": chosen_test.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (OUT / "SUCCESS").touch()
    print(centered_oof[["lambda", "patient_pearson_median", "patient_spearman_median", "site_pearson_median", "site_spearman_median", "patient_equal_mse", "centered_global_sd_ratio", "centered_median_site_sd_ratio", "selection_score"]].to_string(index=False), flush=True)
    print(table_test.query("coordinate == 'study_centered'")[["lambda", "patient_pearson_median", "patient_spearman_median", "site_pearson_median", "site_spearman_median", "patient_equal_mse", "centered_global_sd_ratio", "centered_median_site_sd_ratio"]].to_string(index=False), flush=True)
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
