#!/usr/bin/env python3
"""用研究内中心化指标重跑 M2.2 外部 PT7 参数选择与 106 例评价。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
PT3_CODE = ROOT / "03_code/model_validation/scp682_v2_m22_pt3_comparison"
sys.path[:0] = [str(CODE), str(PT3_CODE)]
from study_centered_metrics import (  # noqa: E402
    center_pair_within_study,
    metric_pair,
    observed_axis_pearson,
    site_spearman,
)
import run_m22_pt3 as core  # noqa: E402


PRIMARY = ROOT / "01_data/bulk/intermediate/random70_reference_rna_protein_logscale_phosphosite_inputs_20260815"
EXTERNAL = ROOT / "01_data/bulk/intermediate/independent_external_posttraining_current_model_inputs_20260815"
LATENT = ROOT / "01_data/multi_omics/intermediate/scp682_v2_m2_residual_latent_20260823"
M1 = ROOT / "01_data/multi_omics/intermediate/scp682_v2_m1_parent_ptm_decomposition_20260822"
MANIFEST = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2/residual_target_manifest.tsv"
PROTOCOL = ROOT / "01_data/multi_omics/intermediate/scp682_v2_m22_pt4_protocol_20260826"
OUT = ROOT / "02_results/model_validation/scp682_v2_m22_study_centered_evaluation_20260830"
ETA = 0.001
ALPHA = 10.0
RANK = 24
GAMMA_GRID = (0.10, 0.15, 0.20, 0.25, 0.30)
LAMBDA_GRID = (1.0, 3.0, 10.0, 30.0)
KAPPA_GRID = (10.0, 30.0, 100.0)


def masked_projection(residual: np.ndarray, basis: np.ndarray, eta: float) -> np.ndarray:
    output = np.zeros((len(residual), len(basis)), dtype=np.float32)
    eye = np.eye(len(basis), dtype=np.float64) * eta
    for row in range(len(residual)):
        observed = np.isfinite(residual[row])
        local = basis[:, observed].astype(np.float64)
        output[row] = np.linalg.solve(local @ local.T + eye, local @ residual[row, observed].astype(np.float64)).astype(np.float32)
    return output


def center_coordinates(coordinates: np.ndarray, studies: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    output = np.full_like(coordinates, np.nan, dtype=np.float32)
    for study in np.unique(studies):
        rows = np.flatnonzero(studies == study)
        fit_rows = train_rows[studies[train_rows] == study]
        center = np.median(coordinates[fit_rows], axis=0).astype(np.float32)
        output[rows] = coordinates[rows] - center[None, :]
    return output


def reduced_delta(model: Ridge, x_train: np.ndarray, x_all: np.ndarray, rank: int) -> np.ndarray:
    fitted = model.predict(x_train).astype(np.float32)
    _, _, right = np.linalg.svd(fitted, full_matrices=False)
    predicted = model.predict(x_all).astype(np.float32)
    return (predicted @ right[:rank].T @ right[:rank]).astype(np.float32)


def slopes(centered: np.ndarray, target: np.ndarray, rows: np.ndarray, gamma: float, lam: float, kappa: float):
    observed = np.isfinite(centered[rows]) & np.isfinite(target[rows])
    count = observed.sum(axis=0).astype(np.float32)
    numerator = np.nansum(np.where(observed, centered[rows] * target[rows], np.nan), axis=0).astype(np.float32)
    square = np.nansum(np.where(observed, centered[rows] ** 2, np.nan), axis=0).astype(np.float32)
    raw = numerator / (square + lam)
    slope = count / (count + kappa) * raw + kappa / (count + kappa) * gamma
    return np.clip(slope, -0.5, 0.5).astype(np.float32), raw, count.astype(np.int32)


def fast_pearson_metrics(truth: np.ndarray, prediction: np.ndarray, studies: np.ndarray) -> dict[str, float]:
    y, p = center_pair_within_study(truth, prediction, studies)
    return {
        "absolute_patient_pearson": float(np.nanmedian(observed_axis_pearson(truth, prediction, axis=1))),
        "absolute_site_pearson": float(np.nanmedian(observed_axis_pearson(truth, prediction, axis=0))),
        "centered_patient_pearson": float(np.nanmedian(observed_axis_pearson(y, p, axis=1))),
        "centered_site_pearson": float(np.nanmedian(observed_axis_pearson(y, p, axis=0))),
    }


def per_site_centered(truth: np.ndarray, prediction: np.ndarray, studies: np.ndarray, sites: np.ndarray) -> pd.DataFrame:
    y, p = center_pair_within_study(truth, prediction, studies)
    observed = np.isfinite(y) & np.isfinite(p)
    return pd.DataFrame(
        {
            "phosphosite": sites,
            "patients_observed": observed.sum(axis=0),
            "study_centered_pearson": observed_axis_pearson(y, p, axis=0),
            "study_centered_spearman": site_spearman(y, p),
            "study_centered_mse": np.nanmean(np.where(observed, (y - p) ** 2, np.nan), axis=0),
        }
    ).sort_values(["study_centered_pearson", "study_centered_spearman"], ascending=False, ignore_index=True)


def main() -> int:
    for name in ("tables", "predictions", "reports", "logs"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    core.LATENT = LATENT
    primary_rna = core.numeric_frame(PRIMARY / "rna_reference_quantile.parquet")
    primary_protein = core.numeric_frame(PRIMARY / "protein_prediction.parquet").reindex(primary_rna.index)
    primary_truth = core.numeric_frame(PRIMARY / "phosphosite_logscale_aligned.parquet").reindex(primary_rna.index)
    primary_split = pd.read_csv(PRIMARY / "split_manifest.tsv", sep="\t").set_index("sample_id").reindex(primary_rna.index)
    primary_train = primary_split.role.eq("selection_train").to_numpy()
    external_rna = core.numeric_frame(EXTERNAL / "rna_reference_quantile.parquet")
    external_protein = core.numeric_frame(EXTERNAL / "protein_prediction.parquet").reindex(external_rna.index)
    external_truth = core.numeric_frame(EXTERNAL / "phosphosite_logscale_aligned.parquet").reindex(external_rna.index)
    external_split = pd.read_csv(EXTERNAL / "split_manifest.tsv", sep="\t").set_index("sample_id").reindex(external_rna.index)
    external_meta = pd.read_csv(EXTERNAL / "sample_metadata.tsv", sep="\t").set_index("sample_id").reindex(external_rna.index)
    calibration_mask = external_split.role.eq("selection_train").to_numpy()
    test_mask = external_split.role.eq("selection_validation").to_numpy()
    calibration_ids = external_rna.index[calibration_mask]
    test_ids = external_rna.index[test_mask]
    all_ids = calibration_ids.append(test_ids)
    sites = external_truth.columns.astype(str).to_numpy()

    manifest = pd.read_csv(MANIFEST, sep="\t")
    protein_lookup = {str(gene).upper(): index for index, gene in enumerate(primary_protein.columns.astype(str))}
    parent_index = np.asarray([protein_lookup[str(gene).upper()] for gene in manifest.total_protein_gene], dtype=np.int64)
    beta = pd.read_csv(M1 / "parent_beta.tsv", sep="\t").beta.to_numpy(np.float32)
    primary_y = primary_truth.to_numpy(np.float32)
    primary_p = primary_protein.to_numpy(np.float32)
    primary_mean = np.nanmean(primary_y[primary_train], axis=0).astype(np.float32)
    primary_count = np.isfinite(primary_y[primary_train]).sum(axis=0)
    primary_protein_center = np.nanmean(primary_p[primary_train], axis=0).astype(np.float32)

    coordinate = core.m22_coordinates_all(
        primary_rna.to_numpy(np.float32), primary_p, primary_train,
        external_rna.to_numpy(np.float32), external_protein.to_numpy(np.float32),
    )
    basis = np.load(LATENT / "models/site_basis_rank128.npy").astype(np.float32)
    residual = coordinate @ basis
    ordered_truth = external_truth.loc[all_ids].to_numpy(np.float32)
    ordered_protein = external_protein.loc[all_ids].to_numpy(np.float32)
    ordered_coordinate = np.vstack([coordinate[calibration_mask], coordinate[test_mask]]).astype(np.float32)
    ordered_residual = np.vstack([residual[calibration_mask], residual[test_mask]]).astype(np.float32)
    studies = external_meta.loc[all_ids].study.astype(str).to_numpy()
    fold_id = pd.read_csv(PROTOCOL / "calibration_fold_assignments.tsv", sep="\t").set_index("sample_id").reindex(calibration_ids).fold.to_numpy(np.int64)
    n_cal = len(calibration_ids)
    cal_rows = np.arange(n_cal)
    candidates = [(g, lam, k) for g in GAMMA_GRID for lam in LAMBDA_GRID for k in KAPPA_GRID]
    oof = {key: np.full((n_cal, len(sites)), np.nan, dtype=np.float32) for key in candidates}
    for fold in sorted(np.unique(fold_id)):
        fit_rows = cal_rows[fold_id != fold]
        held_rows = cal_rows[fold_id == fold]
        fixed, _, target = core.fold_components(
            ordered_truth[:n_cal], ordered_protein[:n_cal], ordered_residual[:n_cal], studies[:n_cal], fit_rows,
            primary_mean, primary_count, primary_protein_center, parent_index, beta,
        )
        z_pred = center_coordinates(ordered_coordinate[:n_cal], studies[:n_cal], fit_rows)
        z_target = masked_projection(target, basis, ETA)
        adapter = Ridge(alpha=ALPHA, fit_intercept=False).fit(z_pred[fit_rows], z_target[fit_rows] - z_pred[fit_rows])
        z_aligned = z_pred + reduced_delta(adapter, z_pred[fit_rows], z_pred, RANK)
        aligned_residual = z_aligned @ basis
        fixed, centered, target = core.fold_components(
            ordered_truth[:n_cal], ordered_protein[:n_cal], aligned_residual, studies[:n_cal], fit_rows,
            primary_mean, primary_count, primary_protein_center, parent_index, beta,
        )
        for gamma, lam, kappa in candidates:
            slope, _, _ = slopes(centered, target, fit_rows, gamma, lam, kappa)
            oof[(gamma, lam, kappa)][held_rows] = fixed[held_rows] + centered[held_rows] * slope[None, :]
        print(f"completed centered PT7 fold {fold + 1}/5", flush=True)

    truth_cal = ordered_truth[:n_cal]
    studies_cal = studies[:n_cal]
    candidate_rows = []
    for gamma, lam, kappa in candidates:
        row = {"gamma": gamma, "lambda": lam, "kappa": kappa, **fast_pearson_metrics(truth_cal, oof[(gamma, lam, kappa)], studies_cal)}
        candidate_rows.append(row)
    grid = pd.DataFrame(candidate_rows)
    baseline = grid[(grid.gamma == 0.20) & (grid["lambda"] == 10.0) & (grid.kappa == 30.0)].iloc[0]
    eligible = grid[grid.centered_site_pearson >= float(baseline.centered_site_pearson) - 1e-8]
    selected = eligible.sort_values(["centered_patient_pearson", "centered_site_pearson", "absolute_patient_pearson"], ascending=False).iloc[0]
    grid["selected"] = (grid.gamma == selected.gamma) & (grid["lambda"] == selected["lambda"]) & (grid.kappa == selected.kappa)
    grid["centered_site_floor"] = float(baseline.centered_site_pearson)
    grid.sort_values(["centered_patient_pearson", "centered_site_pearson"], ascending=False).to_csv(
        OUT / "tables/m22_external_247_centered_pt7_grid.tsv", sep="\t", index=False
    )

    selected_key = (float(selected.gamma), float(selected["lambda"]), float(selected.kappa))
    oof_rows = metric_pair(truth_cal, oof[selected_key], studies_cal, "M2.2_PT7_centered_selected", "external_247_OOF")
    pd.DataFrame(oof_rows).to_csv(OUT / "tables/m22_external_247_corrected_metrics.tsv", sep="\t", index=False)

    fixed, _, target = core.fold_components(
        ordered_truth, ordered_protein, ordered_residual, studies, cal_rows,
        primary_mean, primary_count, primary_protein_center, parent_index, beta,
    )
    z_pred = center_coordinates(ordered_coordinate, studies, cal_rows)
    z_target = masked_projection(target, basis, ETA)
    adapter = Ridge(alpha=ALPHA, fit_intercept=False).fit(z_pred[cal_rows], z_target[cal_rows] - z_pred[cal_rows])
    z_aligned = z_pred + reduced_delta(adapter, z_pred[cal_rows], z_pred, RANK)
    fixed, centered, target = core.fold_components(
        ordered_truth, ordered_protein, z_aligned @ basis, studies, cal_rows,
        primary_mean, primary_count, primary_protein_center, parent_index, beta,
    )
    slope, raw_slope, count = slopes(centered, target, cal_rows, *selected_key)
    test_rows = np.arange(n_cal, len(all_ids))
    prediction_test = fixed[test_rows] + centered[test_rows] * slope[None, :]
    truth_test = ordered_truth[test_rows]
    studies_test = studies[test_rows]
    final_rows = metric_pair(truth_test, prediction_test, studies_test, "M2.2_PT7_centered_selected", "external_106_locked_test")
    for study in np.unique(studies_test):
        keep = studies_test == study
        final_rows += metric_pair(truth_test[keep], prediction_test[keep], studies_test[keep], "M2.2_PT7_centered_selected", str(study))
    final = pd.DataFrame(final_rows)
    final.to_csv(OUT / "tables/m22_external_106_corrected_metrics.tsv", sep="\t", index=False)
    per_site_centered(truth_test, prediction_test, studies_test, sites).to_csv(
        OUT / "tables/m22_external_106_study_centered_per_site.tsv", sep="\t", index=False
    )
    np.savez_compressed(
        OUT / "predictions/m22_external_106_centered_selected_predictions.npz",
        sample_ids=np.asarray(test_ids, str), site_ids=sites, prediction=prediction_test.astype(np.float32), z_aligned=z_aligned[test_rows],
    )
    pd.DataFrame({"phosphosite": sites, "n_calibration_observed": count, "raw_slope": raw_slope, "shrunk_slope": slope}).to_csv(
        OUT / "tables/m22_external_centered_selected_site_slopes.tsv", sep="\t", index=False
    )
    report = {
        "status": "complete",
        "selection_metric": "study-centered patient Pearson with study-centered site Pearson floor",
        "baseline_centered_site_pearson": float(baseline.centered_site_pearson),
        "selected": {"gamma": selected_key[0], "lambda": selected_key[1], "kappa": selected_key[2]},
        "oof": oof_rows,
        "locked_test": final[final.scope.eq("external_106_locked_test")].to_dict("records"),
        "test_labels_used_for_parameter_selection": False,
    }
    (OUT / "reports/external_run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(grid.sort_values(["centered_patient_pearson", "centered_site_pearson"], ascending=False).head(12).to_string(index=False), flush=True)
    print(pd.DataFrame(oof_rows).to_string(index=False), flush=True)
    print(final.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
