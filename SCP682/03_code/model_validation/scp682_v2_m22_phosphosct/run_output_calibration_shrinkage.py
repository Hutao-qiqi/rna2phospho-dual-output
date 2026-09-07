#!/usr/bin/env python3
"""对高患者 Pearson 输出校准候选进行全局收缩。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BASE_SCRIPT = ROOT / "03_code/model_validation/scp682_v2_m22_phosphosct/run_patient_pearson_output_calibration.py"
GRID_DIR = ROOT / "02_results/external_validation/scp682_v2_patient_pearson_output_calibration_20260831"
OUT = ROOT / "02_results/external_validation/scp682_v2_patient_pearson_output_shrinkage_20260831"
ALPHAS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.0)
TOP_CANDIDATES = 20
SITE_FLOOR_TOLERANCE = 0.0005


def load_base():
    spec = importlib.util.spec_from_file_location("output_calibration", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def optional(row, name):
    value = row[name]
    return None if pd.isna(value) else float(value)


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

    grid = pd.read_csv(GRID_DIR / "tables/output_calibration_grid.tsv", sep="\t")
    top = grid.query("mode != 'baseline_m2'").sort_values("patient_pearson", ascending=False).head(TOP_CANDIDATES)
    rows = []
    cached = []
    for candidate_id, (_, candidate) in enumerate(top.iterrows()):
        mode = str(candidate["mode"])
        ridge = optional(candidate, "stack_ridge")
        q = optional(candidate, "q")
        kappa = optional(candidate, "scale_kappa")
        cap = optional(candidate, "scale_cap")
        candidate_oof = base.crossfit_candidate(
            truth_cal, m2_cal, s1_cal, studies_cal, folds, mode,
            stack_ridge=ridge, q=q, scale_kappa=kappa, scale_cap=cap,
        )
        cached.append((candidate_id, candidate, candidate_oof))
        for alpha in ALPHAS:
            prediction = m2_cal + alpha * (candidate_oof - m2_cal)
            patient, site, mse = base.fast_metrics(module, truth_cal, prediction, studies_cal)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "mode": mode,
                    "stack_ridge": ridge,
                    "q": q,
                    "scale_kappa": kappa,
                    "scale_cap": cap,
                    "alpha": alpha,
                    "patient_pearson": patient,
                    "site_pearson": site,
                    "patient_equal_mse": mse,
                    "delta_patient_pearson": patient - baseline_patient,
                    "delta_site_pearson": site - baseline_site,
                }
            )
        print(f"completed candidate {candidate_id + 1}/{len(top)}", flush=True)
    table = pd.DataFrame(rows)
    eligible = table[
        (table.patient_pearson >= baseline_patient)
        & (table.site_pearson >= baseline_site - SITE_FLOOR_TOLERANCE)
    ]
    if eligible.empty:
        selected = None
        selected_oof = m2_cal
        selected_test = m2_test
        selection = {"mode": "baseline_m2", "alpha": 0.0}
    else:
        selected = eligible.sort_values(
            ["patient_pearson", "site_pearson", "patient_equal_mse"], ascending=[False, False, True]
        ).iloc[0]
        candidate_id = int(selected.candidate_id)
        _, candidate, candidate_oof = cached[candidate_id]
        mode = str(candidate["mode"])
        ridge = optional(candidate, "stack_ridge")
        q = optional(candidate, "q")
        kappa = optional(candidate, "scale_kappa")
        cap = optional(candidate, "scale_cap")
        candidate_test, diagnostics = base.full_apply(
            truth_cal, m2_cal, s1_cal, studies_cal, m2_test, s1_test, studies_test,
            mode, stack_ridge=ridge, q=q, scale_kappa=kappa, scale_cap=cap,
        )
        alpha = float(selected.alpha)
        selected_oof = m2_cal + alpha * (candidate_oof - m2_cal)
        selected_test = m2_test + alpha * (candidate_test - m2_test)
        selection = {
            "mode": mode, "stack_ridge": ridge, "q": q, "scale_kappa": kappa,
            "scale_cap": cap, "alpha": alpha,
        }
        if diagnostics:
            pd.DataFrame({"phosphosite": sites, **diagnostics}).to_csv(
                OUT / "tables/selected_site_coefficients.tsv", sep="\t", index=False
            )
    table["selected"] = False
    if selected is not None:
        table.loc[selected.name, "selected"] = True
    table.sort_values(["selected", "patient_pearson"], ascending=[False, False]).to_csv(
        OUT / "tables/shrinkage_grid.tsv", sep="\t", index=False
    )

    final_rows = []
    for model_name, scope, truth, prediction, studies in (
        ("M2.2_PT7", "external_247_OOF", truth_cal, m2_cal, studies_cal),
        ("shrunken_output_calibration", "external_247_OOF", truth_cal, selected_oof, studies_cal),
        ("M2.2_PT7", "external_106_locked_test", truth_test, m2_test, studies_test),
        ("shrunken_output_calibration", "external_106_locked_test", truth_test, selected_test, studies_test),
    ):
        final_rows.extend(module.metric_pair(truth, prediction, studies, model_name, scope))
    final = pd.DataFrame(final_rows)
    final.to_csv(OUT / "tables/final_metrics.tsv", sep="\t", index=False)
    np.savez_compressed(
        OUT / "predictions/shrunken_output_calibration_predictions.npz",
        calibration_ids=calibration_ids, test_ids=test_ids, site_ids=sites,
        oof=selected_oof, test=selected_test,
    )
    centered = final.query("coordinate == 'study_centered'")
    report = "\n".join(
        [
            "# 患者 Pearson 输出校准收缩",
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
    print(table.sort_values("patient_pearson", ascending=False).head(30).to_string(index=False), flush=True)
    print(centered.to_string(index=False), flush=True)
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
