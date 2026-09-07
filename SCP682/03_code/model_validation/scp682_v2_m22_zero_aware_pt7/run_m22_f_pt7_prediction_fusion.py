#!/usr/bin/env python3
"""在相同 PT7 流程下融合 M2.2 与 F_lr2e4 的外部预测。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[3]
BASE_SCRIPT = ROOT / "03_code/model_validation/scp682_v2_m22_study_centered_evaluation/02_rerun_external_pt7_centered.py"
F_MODEL = ROOT / "02_results/model_validation/scp682_v2_m22_zero_aware_20260830/F_lr2e4"
LATENT_MODELS = ROOT / "01_data/multi_omics/intermediate/scp682_v2_m2_residual_latent_20260823/models"
M2_RESULT = ROOT / "02_results/model_validation/scp682_v2_m22_study_centered_evaluation_20260830"
F_RESULT = ROOT / "02_results/external_validation/scp682_v2_m22_zero_aware_f_lr2e4_pt7_20260830"
OUT = ROOT / "02_results/external_validation/scp682_v2_m22_f_pt7_fusion_20260831_v2"
ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
M2_PT7 = (0.10, 1.0, 100.0)
F_PT7 = (0.25, 10.0, 100.0)
SITE_FLOOR_TOLERANCE = 0.0005


def load_pipeline():
    spec = importlib.util.spec_from_file_location("m22_centered_external", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transform_inputs(rna: np.ndarray, protein: np.ndarray, package: np.lib.npyio.NpzFile) -> np.ndarray:
    rna_z = np.nan_to_num((rna - package["rna_mean"]) / package["rna_scale"], nan=0.0).astype(np.float32)
    protein_z = np.nan_to_num(
        (protein - package["protein_mean"]) / package["protein_scale"], nan=0.0
    ).astype(np.float32)
    feature = np.concatenate(
        [rna_z @ package["rna_components"].T, protein_z @ package["protein_components"].T], axis=1
    )
    return ((feature - package["feature_mean"]) / package["feature_scale"]).astype(np.float32)


def f_coordinates(rna: np.ndarray, protein: np.ndarray) -> np.ndarray:
    package = np.load(LATENT_MODELS / "input_projection_and_ridge.npz")
    weights = np.load(F_MODEL / "deployment_linear_weights.npz")
    feature = transform_inputs(rna, protein, package)
    return (feature @ weights["weight"].T + weights["bias"][None, :]).astype(np.float32)


def pt7_oof(
    module,
    coordinate: np.ndarray,
    truth: np.ndarray,
    protein: np.ndarray,
    studies: np.ndarray,
    fold_id: np.ndarray,
    basis: np.ndarray,
    primary_mean: np.ndarray,
    primary_count: np.ndarray,
    primary_protein_center: np.ndarray,
    parent_index: np.ndarray,
    beta: np.ndarray,
    parameters: tuple[float, float, float],
    label: str,
) -> np.ndarray:
    core = module.core
    rows = np.arange(len(truth))
    residual = coordinate @ basis
    output = np.full_like(truth, np.nan, dtype=np.float32)
    for fold in sorted(np.unique(fold_id)):
        fit = rows[fold_id != fold]
        held = rows[fold_id == fold]
        fixed, _, target = core.fold_components(
            truth, protein, residual, studies, fit,
            primary_mean, primary_count, primary_protein_center, parent_index, beta,
        )
        z_pred = module.center_coordinates(coordinate, studies, fit)
        z_target = module.masked_projection(target, basis, module.ETA)
        adapter = Ridge(alpha=module.ALPHA, fit_intercept=False).fit(
            z_pred[fit], z_target[fit] - z_pred[fit]
        )
        z_aligned = z_pred + module.reduced_delta(adapter, z_pred[fit], z_pred, module.RANK)
        fixed, centered, target = core.fold_components(
            truth, protein, z_aligned @ basis, studies, fit,
            primary_mean, primary_count, primary_protein_center, parent_index, beta,
        )
        slope, _, _ = module.slopes(centered, target, fit, *parameters)
        output[held] = fixed[held] + centered[held] * slope[None, :]
        print(f"completed {label} fold {fold + 1}/5", flush=True)
    return output


def pt7_full(
    module,
    coordinate: np.ndarray,
    truth: np.ndarray,
    protein: np.ndarray,
    studies: np.ndarray,
    n_calibration: int,
    basis: np.ndarray,
    primary_mean: np.ndarray,
    primary_count: np.ndarray,
    primary_protein_center: np.ndarray,
    parent_index: np.ndarray,
    beta: np.ndarray,
    parameters: tuple[float, float, float],
) -> np.ndarray:
    core = module.core
    calibration = np.arange(n_calibration)
    residual = coordinate @ basis
    fixed, _, target = core.fold_components(
        truth, protein, residual, studies, calibration,
        primary_mean, primary_count, primary_protein_center, parent_index, beta,
    )
    z_pred = module.center_coordinates(coordinate, studies, calibration)
    z_target = module.masked_projection(target, basis, module.ETA)
    adapter = Ridge(alpha=module.ALPHA, fit_intercept=False).fit(
        z_pred[calibration], z_target[calibration] - z_pred[calibration]
    )
    z_aligned = z_pred + module.reduced_delta(
        adapter, z_pred[calibration], z_pred, module.RANK
    )
    fixed, centered, target = core.fold_components(
        truth, protein, z_aligned @ basis, studies, calibration,
        primary_mean, primary_count, primary_protein_center, parent_index, beta,
    )
    slope, _, _ = module.slopes(centered, target, calibration, *parameters)
    test = np.arange(n_calibration, len(truth))
    return (fixed[test] + centered[test] * slope[None, :]).astype(np.float32)


def centered_site_table(module, truth: np.ndarray, prediction: np.ndarray, studies: np.ndarray, sites: np.ndarray, prefix: str) -> pd.DataFrame:
    y, p = module.center_pair_within_study(truth, prediction, studies)
    return pd.DataFrame(
        {
            "phosphosite": sites,
            f"{prefix}_pearson": module.observed_axis_pearson(y, p, axis=0),
            f"{prefix}_spearman": module.site_spearman(y, p),
            f"{prefix}_n": (np.isfinite(y) & np.isfinite(p)).sum(axis=0),
        }
    )


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    for name in ("tables", "predictions", "reports", "logs"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    module = load_pipeline()
    core = module.core
    core.LATENT = module.LATENT

    primary_rna = core.numeric_frame(module.PRIMARY / "rna_reference_quantile.parquet")
    primary_protein = core.numeric_frame(module.PRIMARY / "protein_prediction.parquet").reindex(primary_rna.index)
    primary_truth = core.numeric_frame(module.PRIMARY / "phosphosite_logscale_aligned.parquet").reindex(primary_rna.index)
    primary_split = pd.read_csv(module.PRIMARY / "split_manifest.tsv", sep="\t").set_index("sample_id").reindex(primary_rna.index)
    primary_train = primary_split.role.eq("selection_train").to_numpy()
    external_rna = core.numeric_frame(module.EXTERNAL / "rna_reference_quantile.parquet")
    external_protein = core.numeric_frame(module.EXTERNAL / "protein_prediction.parquet").reindex(external_rna.index)
    external_truth = core.numeric_frame(module.EXTERNAL / "phosphosite_logscale_aligned.parquet").reindex(external_rna.index)
    external_split = pd.read_csv(module.EXTERNAL / "split_manifest.tsv", sep="\t").set_index("sample_id").reindex(external_rna.index)
    external_meta = pd.read_csv(module.EXTERNAL / "sample_metadata.tsv", sep="\t").set_index("sample_id").reindex(external_rna.index)
    calibration_ids = external_rna.index[external_split.role.eq("selection_train")]
    test_ids = external_rna.index[external_split.role.eq("selection_validation")]
    all_ids = calibration_ids.append(test_ids)
    calibration_mask = external_rna.index.isin(calibration_ids)

    manifest = pd.read_csv(module.MANIFEST, sep="\t")
    protein_lookup = {str(gene).upper(): index for index, gene in enumerate(primary_protein.columns.astype(str))}
    parent_index = np.asarray([protein_lookup[str(gene).upper()] for gene in manifest.total_protein_gene], dtype=np.int64)
    beta = pd.read_csv(module.M1 / "parent_beta.tsv", sep="\t").beta.to_numpy(np.float32)
    primary_y = primary_truth.to_numpy(np.float32)
    primary_p = primary_protein.to_numpy(np.float32)
    primary_mean = np.nanmean(primary_y[primary_train], axis=0).astype(np.float32)
    primary_count = np.isfinite(primary_y[primary_train]).sum(axis=0)
    primary_protein_center = np.nanmean(primary_p[primary_train], axis=0).astype(np.float32)
    basis = np.load(module.LATENT / "models/site_basis_rank128.npy").astype(np.float32)

    external_rna_values = external_rna.to_numpy(np.float32)
    external_protein_values = external_protein.to_numpy(np.float32)
    m2_coordinate = core.m22_coordinates_all(
        primary_rna.to_numpy(np.float32), primary_p, primary_train,
        external_rna_values, external_protein_values,
    )
    f_coordinate = f_coordinates(external_rna_values, external_protein_values)
    truth_cal = external_truth.loc[calibration_ids].to_numpy(np.float32)
    protein_cal = external_protein.loc[calibration_ids].to_numpy(np.float32)
    studies_cal = external_meta.loc[calibration_ids].study.astype(str).to_numpy()
    fold_id = pd.read_csv(module.PROTOCOL / "calibration_fold_assignments.tsv", sep="\t").set_index("sample_id").reindex(calibration_ids).fold.to_numpy(np.int64)
    m2_oof = pt7_oof(
        module, m2_coordinate[calibration_mask], truth_cal, protein_cal, studies_cal, fold_id, basis,
        primary_mean, primary_count, primary_protein_center, parent_index, beta, M2_PT7, "M2.2",
    )
    f_oof = pt7_oof(
        module, f_coordinate[calibration_mask], truth_cal, protein_cal, studies_cal, fold_id, basis,
        primary_mean, primary_count, primary_protein_center, parent_index, beta, F_PT7, "F_lr2e4",
    )

    order = external_rna.index.get_indexer(all_ids)
    truth_ordered = external_truth.loc[all_ids].to_numpy(np.float32)
    protein_ordered = external_protein.loc[all_ids].to_numpy(np.float32)
    studies_ordered = external_meta.loc[all_ids].study.astype(str).to_numpy()
    m2_test = pt7_full(
        module, m2_coordinate[order], truth_ordered, protein_ordered, studies_ordered,
        len(calibration_ids), basis, primary_mean, primary_count, primary_protein_center,
        parent_index, beta, M2_PT7,
    )
    f_test = pt7_full(
        module, f_coordinate[order], truth_ordered, protein_ordered, studies_ordered,
        len(calibration_ids), basis, primary_mean, primary_count, primary_protein_center,
        parent_index, beta, F_PT7,
    )
    m2_ids = np.asarray(test_ids, str)
    m2_sites = external_truth.columns.astype(str).to_numpy()
    truth_test = external_truth.loc[test_ids, m2_sites].to_numpy(np.float32)
    studies_test = external_meta.loc[test_ids].study.astype(str).to_numpy()

    calibration_rows = []
    test_rows = []
    for alpha in ALPHAS:
        blend_cal = m2_oof + alpha * (f_oof - m2_oof)
        blend_test = m2_test + alpha * (f_test - m2_test)
        for row in module.metric_pair(truth_cal, blend_cal, studies_cal, f"blend_alpha_{alpha:g}", "external_247_OOF"):
            calibration_rows.append({"alpha": alpha, **row})
        for row in module.metric_pair(truth_test, blend_test, studies_test, f"blend_alpha_{alpha:g}", "external_106_locked_test"):
            test_rows.append({"alpha": alpha, **row})
    calibration_table = pd.DataFrame(calibration_rows)
    test_table = pd.DataFrame(test_rows)
    centered = calibration_table.query("coordinate == 'study_centered'").copy()
    baseline_site = float(centered.loc[centered.alpha.eq(0), "site_pearson_median"].iloc[0])
    eligible = centered[centered.site_pearson_median >= baseline_site - SITE_FLOOR_TOLERANCE]
    selected = eligible.sort_values(
        ["patient_pearson_median", "patient_spearman_median", "site_pearson_median"], ascending=False
    ).iloc[0]
    selected_alpha = float(selected.alpha)
    calibration_table["selected"] = calibration_table.alpha.eq(selected_alpha)
    test_table["selected"] = test_table.alpha.eq(selected_alpha)
    calibration_table.to_csv(OUT / "tables/fusion_247_grid.tsv", sep="\t", index=False)
    test_table.to_csv(OUT / "tables/fusion_106_grid.tsv", sep="\t", index=False)

    m2_site = centered_site_table(module, truth_cal, m2_oof, studies_cal, m2_sites, "m2")
    f_site = centered_site_table(module, truth_cal, f_oof, studies_cal, m2_sites, "f")
    site = m2_site.merge(f_site, on="phosphosite", how="inner")
    site["delta_pearson"] = site.f_pearson - site.m2_pearson
    site["delta_spearman"] = site.f_spearman - site.m2_spearman
    site.to_csv(OUT / "tables/f_vs_m2_247_per_site.tsv", sep="\t", index=False)

    selected_test = m2_test + selected_alpha * (f_test - m2_test)
    np.savez_compressed(
        OUT / "predictions/fusion_selected_predictions.npz",
        selected_alpha=np.asarray(selected_alpha),
        calibration_ids=np.asarray(calibration_ids, str),
        test_ids=m2_ids,
        site_ids=m2_sites,
        m2_oof=m2_oof,
        f_oof=f_oof,
        selected_test=selected_test.astype(np.float32),
    )
    chosen_247 = calibration_table.query("selected and coordinate == 'study_centered'").iloc[0]
    chosen_106 = test_table.query("selected and coordinate == 'study_centered'").iloc[0]
    report = "\n".join(
        [
            "# M2.2 与 F_lr2e4 的 PT7 预测融合",
            "",
            f"247例选择 alpha = `{selected_alpha:g}`。",
            "",
            "| 数据 | 患者 Pearson | 患者 Spearman | 位点 Pearson | 位点 Spearman | MSE |",
            "|---|---:|---:|---:|---:|---:|",
            f"| 247例中心化 | {chosen_247.patient_pearson_median:.6f} | {chosen_247.patient_spearman_median:.6f} | {chosen_247.site_pearson_median:.6f} | {chosen_247.site_spearman_median:.6f} | {chosen_247.patient_equal_mse:.6f} |",
            f"| 106例中心化 | {chosen_106.patient_pearson_median:.6f} | {chosen_106.patient_spearman_median:.6f} | {chosen_106.site_pearson_median:.6f} | {chosen_106.site_spearman_median:.6f} | {chosen_106.patient_equal_mse:.6f} |",
            "",
        ]
    )
    (OUT / "RESULTS_20260831.md").write_text(report, encoding="utf-8")
    (OUT / "reports/run_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "alphas": list(ALPHAS),
                "selected_alpha": selected_alpha,
                "site_floor_tolerance": SITE_FLOOR_TOLERANCE,
                "selected_247_centered": chosen_247.to_dict(),
                "selected_106_centered": chosen_106.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (OUT / "SUCCESS").touch()
    print(centered[["alpha", "patient_pearson_median", "patient_spearman_median", "site_pearson_median", "site_spearman_median", "patient_equal_mse"]].to_string(index=False), flush=True)
    print(test_table.query("coordinate == 'study_centered'")[["alpha", "patient_pearson_median", "patient_spearman_median", "site_pearson_median", "site_spearman_median", "patient_equal_mse"]].to_string(index=False), flush=True)
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
