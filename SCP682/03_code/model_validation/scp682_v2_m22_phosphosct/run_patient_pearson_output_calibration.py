#!/usr/bin/env python3
"""M2.2/S1 位点尺度校准与连续堆叠。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
METRIC_SCRIPT = ROOT / "03_code/model_validation/scp682_v2_m22_study_centered_evaluation/02_rerun_external_pt7_centered.py"
EXTERNAL_INPUT = ROOT / "01_data/bulk/intermediate/independent_external_posttraining_current_model_inputs_20260815"
M2_FILE = ROOT / "02_results/external_validation/scp682_v2_m22_f_pt7_fusion_20260831_v2/predictions/fusion_selected_predictions.npz"
S1_FILE = ROOT / "02_results/external_validation/scp682_v2_m23_phosphosct_s1_dedicated_pt7_20260831/predictions/s1_dedicated_pt7_predictions.npz"
OUT = ROOT / "02_results/external_validation/scp682_v2_patient_pearson_output_calibration_20260831"
Q_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
SCALE_KAPPA_GRID = (10.0, 30.0, 100.0, 300.0)
SCALE_CAP_GRID = (2.0, 3.0)
STACK_LAMBDA_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
SITE_FLOOR_TOLERANCE = 0.0005
MIN_OBSERVED = 10


def load_module():
    spec = importlib.util.spec_from_file_location("centered_metrics", METRIC_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {METRIC_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_centers(values: np.ndarray, studies: np.ndarray, rows: np.ndarray) -> dict[str, np.ndarray]:
    centers = {}
    for study in np.unique(studies):
        local = rows[studies[rows] == study]
        centers[str(study)] = np.nanmean(values[local], axis=0).astype(np.float32)
    return centers


def apply_centers(values: np.ndarray, studies: np.ndarray, centers: dict[str, np.ndarray]) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float32)
    for study in np.unique(studies):
        rows = np.flatnonzero(studies == study)
        output[rows] = values[rows] - centers[str(study)][None, :]
    return output


def restore_centers(centered: np.ndarray, studies: np.ndarray, centers: dict[str, np.ndarray]) -> np.ndarray:
    output = np.empty_like(centered, dtype=np.float32)
    for study in np.unique(studies):
        rows = np.flatnonzero(studies == study)
        output[rows] = centered[rows] + centers[str(study)][None, :]
    return output


def fit_site_scale(truth: np.ndarray, prediction: np.ndarray, q: float, kappa: float, cap: float) -> np.ndarray:
    observed = np.isfinite(truth) & np.isfinite(prediction)
    count = observed.sum(axis=0).astype(np.float64)
    y = np.where(observed, truth, 0.0)
    p = np.where(observed, prediction, 0.0)
    y_mean = np.divide(y.sum(0), count, out=np.zeros(truth.shape[1]), where=count > 0)
    p_mean = np.divide(p.sum(0), count, out=np.zeros(truth.shape[1]), where=count > 0)
    dy = np.where(observed, truth - y_mean[None, :], 0.0)
    dp = np.where(observed, prediction - p_mean[None, :], 0.0)
    y_sd = np.sqrt(np.divide(np.square(dy).sum(0), count, out=np.zeros(truth.shape[1]), where=count > 0))
    p_sd = np.sqrt(np.divide(np.square(dp).sum(0), count, out=np.ones(truth.shape[1]), where=count > 0))
    denominator = np.sqrt(np.square(dy).sum(0) * np.square(dp).sum(0))
    correlation = np.divide(
        (dy * dp).sum(0), denominator,
        out=np.zeros(truth.shape[1]), where=(count >= MIN_OBSERVED) & (denominator > 1e-12),
    )
    ratio = np.divide(y_sd, p_sd, out=np.ones(truth.shape[1]), where=p_sd > 1e-8)
    reliability = np.ones_like(correlation) if q == 0 else np.power(np.maximum(correlation, 0.0), q)
    raw = ratio * reliability
    weight = count / (count + kappa)
    scale = weight * raw + (1.0 - weight)
    scale[count < MIN_OBSERVED] = 1.0
    return np.clip(scale, 0.0, cap).astype(np.float32)


def fit_site_stack(truth: np.ndarray, m2: np.ndarray, s1: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    observed = np.isfinite(truth) & np.isfinite(m2) & np.isfinite(s1)
    count = observed.sum(axis=0)
    x1 = np.where(observed, m2, 0.0).astype(np.float64)
    x2 = np.where(observed, s1, 0.0).astype(np.float64)
    y = np.where(observed, truth, 0.0).astype(np.float64)
    a = np.square(x1).sum(axis=0) + ridge
    b = (x1 * x2).sum(axis=0)
    d = np.square(x2).sum(axis=0) + ridge
    r1 = (x1 * y).sum(axis=0) + ridge
    r2 = (x2 * y).sum(axis=0)
    determinant = a * d - np.square(b)
    beta_m = np.divide(r1 * d - b * r2, determinant, out=np.ones_like(a), where=determinant > 1e-12)
    beta_s = np.divide(a * r2 - b * r1, determinant, out=np.zeros_like(a), where=determinant > 1e-12)
    beta_m[count < MIN_OBSERVED] = 1.0
    beta_s[count < MIN_OBSERVED] = 0.0
    return np.clip(beta_m, 0.0, 1.5).astype(np.float32), np.clip(beta_s, -0.75, 1.0).astype(np.float32)


def fast_metrics(module, truth, prediction, studies):
    metric = module.fast_pearson_metrics(truth, prediction, studies)
    y, p = module.center_pair_within_study(truth, prediction, studies)
    observed = np.isfinite(y) & np.isfinite(p)
    patient_mse = np.nanmean(np.where(observed, np.square(y - p), np.nan), axis=1)
    return (
        metric["centered_patient_pearson"],
        metric["centered_site_pearson"],
        float(np.nanmean(patient_mse)),
    )


def crossfit_candidate(
    truth, m2, s1, studies, folds, mode, stack_ridge=None, q=None, scale_kappa=None, scale_cap=None,
):
    output = np.full_like(m2, np.nan, dtype=np.float32)
    rows = np.arange(len(truth))
    for fold in sorted(np.unique(folds)):
        fit = rows[folds != fold]
        held = rows[folds == fold]
        truth_centers = fit_centers(truth, studies, fit)
        m2_centers = fit_centers(m2, studies, fit)
        s1_centers = fit_centers(s1, studies, fit)
        y_center = apply_centers(truth, studies, truth_centers)
        m_center = apply_centers(m2, studies, m2_centers)
        s_center = apply_centers(s1, studies, s1_centers)

        if mode == "scale_m2":
            scale = fit_site_scale(y_center[fit], m_center[fit], float(q), float(scale_kappa), float(scale_cap))
            held_centered = m_center[held] * scale[None, :]
        else:
            beta_m, beta_s = fit_site_stack(y_center[fit], m_center[fit], s_center[fit], float(stack_ridge))
            fit_stacked = m_center[fit] * beta_m[None, :] + s_center[fit] * beta_s[None, :]
            held_centered = m_center[held] * beta_m[None, :] + s_center[held] * beta_s[None, :]
            if mode == "stack_scale":
                scale = fit_site_scale(y_center[fit], fit_stacked, float(q), float(scale_kappa), float(scale_cap))
                held_centered *= scale[None, :]
        output[held] = restore_centers(held_centered, studies[held], m2_centers)
    return output


def full_apply(
    truth_cal, m2_cal, s1_cal, studies_cal, m2_test, s1_test, studies_test,
    mode, stack_ridge=None, q=None, scale_kappa=None, scale_cap=None,
):
    fit = np.arange(len(truth_cal))
    truth_centers = fit_centers(truth_cal, studies_cal, fit)
    m2_centers = fit_centers(m2_cal, studies_cal, fit)
    s1_centers = fit_centers(s1_cal, studies_cal, fit)
    y_center = apply_centers(truth_cal, studies_cal, truth_centers)
    m_cal_center = apply_centers(m2_cal, studies_cal, m2_centers)
    s_cal_center = apply_centers(s1_cal, studies_cal, s1_centers)
    m_test_center = apply_centers(m2_test, studies_test, m2_centers)
    s_test_center = apply_centers(s1_test, studies_test, s1_centers)
    diagnostics = {}
    if mode == "scale_m2":
        scale = fit_site_scale(y_center, m_cal_center, float(q), float(scale_kappa), float(scale_cap))
        test_centered = m_test_center * scale[None, :]
        diagnostics["scale"] = scale
    else:
        beta_m, beta_s = fit_site_stack(y_center, m_cal_center, s_cal_center, float(stack_ridge))
        test_centered = m_test_center * beta_m[None, :] + s_test_center * beta_s[None, :]
        diagnostics["beta_m2"] = beta_m
        diagnostics["beta_s1"] = beta_s
        if mode == "stack_scale":
            fit_stacked = m_cal_center * beta_m[None, :] + s_cal_center * beta_s[None, :]
            scale = fit_site_scale(y_center, fit_stacked, float(q), float(scale_kappa), float(scale_cap))
            test_centered *= scale[None, :]
            diagnostics["scale"] = scale
    return restore_centers(test_centered, studies_test, m2_centers), diagnostics


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    for name in ("tables", "predictions", "reports", "logs"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    module = load_module()
    m2_package = np.load(M2_FILE, allow_pickle=True)
    s1_package = np.load(S1_FILE, allow_pickle=True)
    calibration_ids = m2_package["calibration_ids"].astype(str)
    test_ids = m2_package["test_ids"].astype(str)
    sites = m2_package["site_ids"].astype(str)
    m2_cal = m2_package["m2_oof"].astype(np.float32)
    m2_test = m2_package["selected_test"].astype(np.float32)
    s1_cal = s1_package["oof"].astype(np.float32)
    s1_test = s1_package["test"].astype(np.float32)
    truth_frame = pd.read_parquet(EXTERNAL_INPUT / "phosphosite_logscale_aligned.parquet")
    metadata = pd.read_csv(EXTERNAL_INPUT / "sample_metadata.tsv", sep="\t").set_index("sample_id")
    truth_cal = truth_frame.reindex(index=calibration_ids, columns=sites).to_numpy(np.float32)
    truth_test = truth_frame.reindex(index=test_ids, columns=sites).to_numpy(np.float32)
    studies_cal = metadata.reindex(calibration_ids).study.astype(str).to_numpy()
    studies_test = metadata.reindex(test_ids).study.astype(str).to_numpy()
    folds = pd.read_csv(module.PROTOCOL / "calibration_fold_assignments.tsv", sep="\t").set_index("sample_id").reindex(calibration_ids).fold.to_numpy(np.int64)

    baseline_patient, baseline_site, baseline_mse = fast_metrics(module, truth_cal, m2_cal, studies_cal)
    rows = [
        {
            "mode": "baseline_m2",
            "stack_ridge": np.nan,
            "q": np.nan,
            "scale_kappa": np.nan,
            "scale_cap": np.nan,
            "patient_pearson": baseline_patient,
            "site_pearson": baseline_site,
            "patient_equal_mse": baseline_mse,
        }
    ]
    candidates = []
    for q in Q_GRID:
        for kappa in SCALE_KAPPA_GRID:
            for cap in SCALE_CAP_GRID:
                candidates.append(("scale_m2", None, q, kappa, cap))
    for ridge in STACK_LAMBDA_GRID:
        candidates.append(("stack", ridge, None, None, None))
        for q in Q_GRID:
            for kappa in SCALE_KAPPA_GRID:
                candidates.append(("stack_scale", ridge, q, kappa, 3.0))
    for index, (mode, ridge, q, kappa, cap) in enumerate(candidates):
        prediction = crossfit_candidate(
            truth_cal, m2_cal, s1_cal, studies_cal, folds, mode,
            stack_ridge=ridge, q=q, scale_kappa=kappa, scale_cap=cap,
        )
        patient, site, mse = fast_metrics(module, truth_cal, prediction, studies_cal)
        rows.append(
            {
                "mode": mode,
                "stack_ridge": ridge,
                "q": q,
                "scale_kappa": kappa,
                "scale_cap": cap,
                "patient_pearson": patient,
                "site_pearson": site,
                "patient_equal_mse": mse,
            }
        )
        if (index + 1) % 20 == 0:
            print(f"evaluated {index + 1}/{len(candidates)}", flush=True)
    grid = pd.DataFrame(rows)
    grid["delta_patient_pearson"] = grid.patient_pearson - baseline_patient
    grid["delta_site_pearson"] = grid.site_pearson - baseline_site
    eligible = grid[
        (grid.patient_pearson >= baseline_patient)
        & (grid.site_pearson >= baseline_site - SITE_FLOOR_TOLERANCE)
    ]
    if eligible.empty:
        selected = grid.iloc[0]
    else:
        selected = eligible.sort_values(
            ["patient_pearson", "site_pearson", "patient_equal_mse"], ascending=[False, False, True]
        ).iloc[0]
    grid["selected"] = False
    grid.loc[selected.name, "selected"] = True
    grid.sort_values(["selected", "patient_pearson"], ascending=[False, False]).to_csv(
        OUT / "tables/output_calibration_grid.tsv", sep="\t", index=False
    )

    selected_mode = str(selected["mode"])
    if selected_mode == "baseline_m2":
        selected_oof = m2_cal
        selected_test = m2_test
        diagnostics = {}
    else:
        def optional(name):
            value = selected[name]
            return None if pd.isna(value) else float(value)
        selected_oof = crossfit_candidate(
            truth_cal, m2_cal, s1_cal, studies_cal, folds, selected_mode,
            stack_ridge=optional("stack_ridge"), q=optional("q"),
            scale_kappa=optional("scale_kappa"), scale_cap=optional("scale_cap"),
        )
        selected_test, diagnostics = full_apply(
            truth_cal, m2_cal, s1_cal, studies_cal, m2_test, s1_test, studies_test,
            selected_mode, stack_ridge=optional("stack_ridge"), q=optional("q"),
            scale_kappa=optional("scale_kappa"), scale_cap=optional("scale_cap"),
        )

    final_rows = []
    for model_name, scope, truth, prediction, studies in (
        ("M2.2_PT7", "external_247_OOF", truth_cal, m2_cal, studies_cal),
        ("selected_output_calibration", "external_247_OOF", truth_cal, selected_oof, studies_cal),
        ("M2.2_PT7", "external_106_locked_test", truth_test, m2_test, studies_test),
        ("selected_output_calibration", "external_106_locked_test", truth_test, selected_test, studies_test),
    ):
        final_rows.extend(module.metric_pair(truth, prediction, studies, model_name, scope))
    final = pd.DataFrame(final_rows)
    final.to_csv(OUT / "tables/final_metrics.tsv", sep="\t", index=False)
    if diagnostics:
        diagnostic_table = pd.DataFrame({"phosphosite": sites, **diagnostics})
        diagnostic_table.to_csv(OUT / "tables/selected_site_coefficients.tsv", sep="\t", index=False)
    np.savez_compressed(
        OUT / "predictions/selected_output_calibration_predictions.npz",
        calibration_ids=calibration_ids, test_ids=test_ids, site_ids=sites,
        oof=selected_oof, test=selected_test,
    )

    centered = final.query("coordinate == 'study_centered'")
    selection_text = ", ".join(
        f"{name}={selected[name]}" for name in ("mode", "stack_ridge", "q", "scale_kappa", "scale_cap")
    )
    report = "\n".join(
        [
            "# 患者 Pearson 输出空间校准与连续堆叠",
            "",
            f"247例选择：`{selection_text}`。",
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
        json.dumps(
            {"status": "complete", "selection": selected.to_dict(), "metrics": final.to_dict("records")},
            ensure_ascii=False, indent=2,
        ) + "\n", encoding="utf-8",
    )
    (OUT / "SUCCESS").touch()
    print(grid.sort_values("patient_pearson", ascending=False).head(25).to_string(index=False), flush=True)
    print(centered.to_string(index=False), flush=True)
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
