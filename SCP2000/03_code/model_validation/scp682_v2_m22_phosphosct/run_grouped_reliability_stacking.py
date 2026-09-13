#!/usr/bin/env python3
"""按位点增量可靠性分组收缩 M2.2/S1 连续堆叠。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BASE_SCRIPT = ROOT / "03_code/model_validation/scp682_v2_m22_phosphosct/run_patient_pearson_output_calibration.py"
OUT = ROOT / "02_results/external_validation/scp682_v2_grouped_reliability_stacking_20260831"
GROUPS = (25, 50)
ALPHA_MAX = (0.05, 0.10, 0.15, 0.20, 0.30)
POWERS = (0.25, 0.50, 1.0, 2.0)
STACK_RIDGE = 30.0
MIN_OBSERVED = 10


def load_base():
    spec = importlib.util.spec_from_file_location("output_calibration", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def site_pearson(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    observed = np.isfinite(truth) & np.isfinite(prediction)
    count = observed.sum(0)
    y = np.where(observed, truth, 0.0)
    p = np.where(observed, prediction, 0.0)
    ym = np.divide(y.sum(0), count, out=np.zeros(truth.shape[1]), where=count > 0)
    pm = np.divide(p.sum(0), count, out=np.zeros(truth.shape[1]), where=count > 0)
    dy = np.where(observed, truth - ym[None, :], 0.0)
    dp = np.where(observed, prediction - pm[None, :], 0.0)
    denominator = np.sqrt(np.square(dy).sum(0) * np.square(dp).sum(0))
    return np.divide(
        (dy * dp).sum(0), denominator, out=np.full(truth.shape[1], np.nan),
        where=(count >= MIN_OBSERVED) & (denominator > 1e-12),
    )


def grouped_weight(utility: np.ndarray, groups: int, alpha_max: float, power: float) -> np.ndarray:
    weight = np.zeros(len(utility), dtype=np.float32)
    positive = np.flatnonzero(np.isfinite(utility) & (utility > 0))
    if len(positive) == 0:
        return weight
    order = positive[np.argsort(utility[positive], kind="mergesort")]
    percentile = (np.arange(len(order), dtype=np.float32) + 0.5) / len(order)
    group_index = np.minimum((percentile * groups).astype(int), groups - 1)
    group_score = (group_index.astype(np.float32) + 0.5) / groups
    weight[order] = alpha_max * np.power(group_score, power)
    return weight


def fold_components(base, truth, m2, s1, studies, fit, groups, alpha_max, power):
    truth_centers = base.fit_centers(truth, studies, fit)
    m2_centers = base.fit_centers(m2, studies, fit)
    s1_centers = base.fit_centers(s1, studies, fit)
    y = base.apply_centers(truth, studies, truth_centers)
    m = base.apply_centers(m2, studies, m2_centers)
    s = base.apply_centers(s1, studies, s1_centers)
    beta_m, beta_s = base.fit_site_stack(y[fit], m[fit], s[fit], STACK_RIDGE)
    stacked = m * beta_m[None, :] + s * beta_s[None, :]
    utility = site_pearson(y[fit], stacked[fit]) - site_pearson(y[fit], m[fit])
    weight = grouped_weight(utility, groups, alpha_max, power)
    prediction_centered = m + weight[None, :] * (stacked - m)
    return prediction_centered, m2_centers, weight, utility, beta_m, beta_s


def crossfit(base, truth, m2, s1, studies, folds, groups, alpha_max, power):
    output = np.full_like(m2, np.nan, dtype=np.float32)
    rows = np.arange(len(truth))
    for fold in sorted(np.unique(folds)):
        fit = rows[folds != fold]
        held = rows[folds == fold]
        prediction, centers, *_ = fold_components(
            base, truth, m2, s1, studies, fit, groups, alpha_max, power
        )
        output[held] = base.restore_centers(prediction[held], studies[held], centers)
    return output


def full_apply(base, truth, m2, s1, studies, m2_test, s1_test, studies_test, groups, alpha_max, power):
    fit = np.arange(len(truth))
    prediction_cal, centers, weight, utility, beta_m, beta_s = fold_components(
        base, truth, m2, s1, studies, fit, groups, alpha_max, power
    )
    m_test_center = base.apply_centers(m2_test, studies_test, centers)
    s1_centers = base.fit_centers(s1, studies, fit)
    s_test_center = base.apply_centers(s1_test, studies_test, s1_centers)
    stacked_test = m_test_center * beta_m[None, :] + s_test_center * beta_s[None, :]
    test_centered = m_test_center + weight[None, :] * (stacked_test - m_test_center)
    return base.restore_centers(test_centered, studies_test, centers), {
        "weight": weight, "utility": utility, "beta_m2": beta_m, "beta_s1": beta_s,
    }


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    for name in ("tables", "predictions", "reports"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    base = load_base()
    module = base.load_module()
    m2_package = np.load(base.M2_FILE, allow_pickle=True)
    s1_package = np.load(base.S1_FILE, allow_pickle=True)
    calibration_ids = m2_package["calibration_ids"].astype(str)
    test_ids = m2_package["test_ids"].astype(str)
    sites = m2_package["site_ids"].astype(str)
    m2_cal = m2_package["m2_oof"].astype(np.float32)
    m2_test = m2_package["selected_test"].astype(np.float32)
    s1_cal = s1_package["oof"].astype(np.float32)
    s1_test = s1_package["test"].astype(np.float32)
    truth_frame = pd.read_parquet(base.EXTERNAL_INPUT / "phosphosite_logscale_aligned.parquet")
    metadata = pd.read_csv(base.EXTERNAL_INPUT / "sample_metadata.tsv", sep="\t").set_index("sample_id")
    truth_cal = truth_frame.reindex(index=calibration_ids, columns=sites).to_numpy(np.float32)
    truth_test = truth_frame.reindex(index=test_ids, columns=sites).to_numpy(np.float32)
    studies_cal = metadata.reindex(calibration_ids).study.astype(str).to_numpy()
    studies_test = metadata.reindex(test_ids).study.astype(str).to_numpy()
    folds = pd.read_csv(module.PROTOCOL / "calibration_fold_assignments.tsv", sep="\t").set_index("sample_id").reindex(calibration_ids).fold.to_numpy(np.int64)
    baseline_patient, baseline_site, baseline_mse = base.fast_metrics(module, truth_cal, m2_cal, studies_cal)

    rows = []
    for groups in GROUPS:
        for alpha_max in ALPHA_MAX:
            for power in POWERS:
                prediction = crossfit(
                    base, truth_cal, m2_cal, s1_cal, studies_cal, folds,
                    groups, alpha_max, power,
                )
                patient, site, mse = base.fast_metrics(module, truth_cal, prediction, studies_cal)
                rows.append(
                    {
                        "groups": groups, "alpha_max": alpha_max, "power": power,
                        "patient_pearson": patient, "site_pearson": site,
                        "patient_equal_mse": mse,
                        "delta_patient_pearson": patient - baseline_patient,
                        "delta_site_pearson": site - baseline_site,
                    }
                )
                print(f"groups={groups} alpha={alpha_max:g} power={power:g}", flush=True)
    grid = pd.DataFrame(rows)
    eligible = grid[(grid.patient_pearson >= baseline_patient) & (grid.site_pearson >= baseline_site)]
    if eligible.empty:
        selected = None
        selected_oof = m2_cal
        selected_test = m2_test
        diagnostics = {}
        selection = {"model": "M2.2_PT7"}
    else:
        selected = eligible.sort_values(
            ["patient_pearson", "site_pearson", "patient_equal_mse"], ascending=[False, False, True]
        ).iloc[0]
        groups = int(selected.groups)
        alpha_max = float(selected.alpha_max)
        power = float(selected.power)
        selected_oof = crossfit(
            base, truth_cal, m2_cal, s1_cal, studies_cal, folds, groups, alpha_max, power
        )
        selected_test, diagnostics = full_apply(
            base, truth_cal, m2_cal, s1_cal, studies_cal, m2_test, s1_test, studies_test,
            groups, alpha_max, power,
        )
        selection = {"groups": groups, "alpha_max": alpha_max, "power": power, "stack_ridge": STACK_RIDGE}
    grid["selected"] = False
    if selected is not None:
        grid.loc[selected.name, "selected"] = True
    grid.sort_values(["selected", "patient_pearson"], ascending=[False, False]).to_csv(
        OUT / "tables/grouped_reliability_grid.tsv", sep="\t", index=False
    )
    if diagnostics:
        pd.DataFrame({"phosphosite": sites, **diagnostics}).to_csv(
            OUT / "tables/selected_site_coefficients.tsv", sep="\t", index=False
        )
    final_rows = []
    for model_name, scope, truth, prediction, studies in (
        ("M2.2_PT7", "external_247_OOF", truth_cal, m2_cal, studies_cal),
        ("grouped_reliability_stacking", "external_247_OOF", truth_cal, selected_oof, studies_cal),
        ("M2.2_PT7", "external_106_locked_test", truth_test, m2_test, studies_test),
        ("grouped_reliability_stacking", "external_106_locked_test", truth_test, selected_test, studies_test),
    ):
        final_rows.extend(module.metric_pair(truth, prediction, studies, model_name, scope))
    final = pd.DataFrame(final_rows)
    final.to_csv(OUT / "tables/final_metrics.tsv", sep="\t", index=False)
    np.savez_compressed(
        OUT / "predictions/grouped_reliability_stacking_predictions.npz",
        calibration_ids=calibration_ids, test_ids=test_ids, site_ids=sites,
        oof=selected_oof, test=selected_test,
    )
    centered = final.query("coordinate == 'study_centered'")
    report = "\n".join(
        [
            "# 分组可靠性连续堆叠",
            "",
            f"选择：`{selection}`。",
            "",
            "| 数据 | 模型 | 患者 Pearson | 患者 Spearman | 位点 Pearson | 位点 Spearman | MSE |",
            "|---|---|---:|---:|---:|---:|---:|",
            *[
                f"| {row.scope} | {row.model} | {row.patient_pearson_median:.6f} | {row.patient_spearman_median:.6f} | {row.site_pearson_median:.6f} | {row.site_spearman_median:.6f} | {row.patient_equal_mse:.6f} |"
                for _, row in centered.iterrows()
            ],
            "",
        ]
    )
    (OUT / "RESULTS_20260831.md").write_text(report, encoding="utf-8")
    (OUT / "reports/run_summary.json").write_text(
        json.dumps({"status": "complete", "selection": selection, "metrics": final.to_dict("records")}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "SUCCESS").touch()
    print(grid.sort_values("patient_pearson", ascending=False).head(20).to_string(index=False), flush=True)
    print(centered.to_string(index=False), flush=True)
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
