#!/usr/bin/env python3
"""M2.2 residual under the locked Main-BORP PT3 calibration protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge


ROOT = Path("/data/lsy/Infinite_Stream/SCP682-main")
PRIMARY = ROOT / "01_data/bulk/intermediate/random70_reference_rna_protein_logscale_phosphosite_inputs_20260815"
EXTERNAL = ROOT / "01_data/bulk/intermediate/independent_external_posttraining_current_model_inputs_20260815"
TARGET_MANIFEST = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2/residual_target_manifest.tsv"
M1 = ROOT / "01_data/multi_omics/intermediate/scp682_v2_m1_parent_ptm_decomposition_20260822"
LATENT = ROOT / "01_data/multi_omics/intermediate/scp682_v2_m2_residual_latent_20260823"
M22_LOCKED = ROOT / "02_results/external_validation/20260823_scp682_v2_m22_locked_external_test"
PT4_PROTOCOL = ROOT / "01_data/multi_omics/intermediate/scp682_v2_m22_pt4_protocol_20260826"
OUT = ROOT / "02_results/external_validation/scp682_v2_m22_pt3_matched_20260826"

GAMMA_J = 0.20
LAMBDA = 10.0
KAPPA = 30.0
SLOPE_MIN = -0.5
SLOPE_MAX = 0.5
SELECTIVE = np.asarray([15, 18, 23, 30, 41, 48, 55, 58], dtype=int)


def numeric_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame.apply(pd.to_numeric, errors="coerce")


def transform(rna: np.ndarray, protein: np.ndarray, package) -> np.ndarray:
    rna_z = np.nan_to_num((rna - package["rna_mean"]) / package["rna_scale"], nan=0.0).astype(np.float32)
    protein_z = np.nan_to_num((protein - package["protein_mean"]) / package["protein_scale"], nan=0.0).astype(np.float32)
    feature = np.concatenate(
        [rna_z @ package["rna_components"].T, protein_z @ package["protein_components"].T], axis=1
    )
    return ((feature - package["feature_mean"]) / package["feature_scale"]).astype(np.float32)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0)
    scale = np.nanstd(train, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return (
        np.nan_to_num((train - mean) / scale, nan=0.0).astype(np.float32),
        np.nan_to_num((test - mean) / scale, nan=0.0).astype(np.float32),
    )


def patient_spearman(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    output = np.full(len(truth), np.nan, dtype=np.float64)
    for row in range(len(truth)):
        observed = np.isfinite(truth[row]) & np.isfinite(prediction[row])
        if observed.sum() >= 10:
            output[row] = np.corrcoef(
                rankdata(truth[row, observed]), rankdata(prediction[row, observed])
            )[0, 1]
    return output


def site_summary(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    observed = np.isfinite(truth) & np.isfinite(prediction)
    count = observed.sum(axis=0)
    truth_rank = pd.DataFrame(np.where(observed, truth, np.nan)).rank(axis=0, method="average").to_numpy(np.float64)
    prediction_rank = pd.DataFrame(np.where(observed, prediction, np.nan)).rank(axis=0, method="average").to_numpy(np.float64)
    truth_mean = np.divide(np.nansum(truth_rank, axis=0), count, out=np.zeros(truth.shape[1]), where=count > 0)
    prediction_mean = np.divide(np.nansum(prediction_rank, axis=0), count, out=np.zeros(truth.shape[1]), where=count > 0)
    truth_centered = truth_rank - truth_mean
    prediction_centered = prediction_rank - prediction_mean
    numerator = np.nansum(truth_centered * prediction_centered, axis=0)
    denominator = np.sqrt(
        np.nansum(truth_centered**2, axis=0) * np.nansum(prediction_centered**2, axis=0)
    )
    score = np.divide(
        numerator,
        denominator,
        out=np.full(truth.shape[1], np.nan),
        where=(count >= 10) & (denominator > 0),
    )
    return {
        "effective_sites": int(np.isfinite(score).sum()),
        "median_site_spearman": float(np.nanmedian(score)),
        "mean_site_spearman": float(np.nanmean(score)),
    }


def observed_median(values: np.ndarray, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    count = np.isfinite(values).sum(axis=0)
    center = np.zeros(values.shape[1], dtype=np.float32)
    valid = count >= minimum
    if valid.any():
        center[valid] = np.nanmedian(values[:, valid], axis=0).astype(np.float32)
    return center, count


def fold_components(
    truth: np.ndarray,
    protein: np.ndarray,
    residual: np.ndarray,
    studies: np.ndarray,
    train_rows: np.ndarray,
    primary_mean: np.ndarray,
    primary_count: np.ndarray,
    primary_protein_center: np.ndarray,
    parent_index: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixed = np.full_like(truth, np.nan, dtype=np.float32)
    centered_residual = np.full_like(truth, np.nan, dtype=np.float32)
    centered_target = np.full_like(truth, np.nan, dtype=np.float32)
    for study in sorted(np.unique(studies)):
        study_rows = np.flatnonzero(studies == study)
        study_train = train_rows[studies[train_rows] == study]
        train_truth = truth[study_train]
        site_count = np.isfinite(train_truth).sum(axis=0).astype(np.float32)
        site_mean = np.divide(
            np.nansum(train_truth, axis=0, dtype=np.float64),
            site_count,
            out=np.full(truth.shape[1], np.nan, dtype=np.float64),
            where=site_count > 0,
        ).astype(np.float32)
        common = (site_count >= 5) & (primary_count >= 5) & np.isfinite(site_mean) & np.isfinite(primary_mean)
        delta = float(np.median(site_mean[common] - primary_mean[common])) if common.any() else 0.0
        fallback = primary_mean + delta
        site_mean = np.where(np.isfinite(site_mean), site_mean, fallback)
        weight = site_count / (site_count + 2.0)
        adapted_site = (weight * site_mean + (1.0 - weight) * fallback).astype(np.float32)

        protein_center = np.nanmean(protein[study_train], axis=0).astype(np.float32)
        protein_center = np.where(np.isfinite(protein_center), protein_center, primary_protein_center)
        parent_term = np.nan_to_num(
            protein[study_rows][:, parent_index] - protein_center[parent_index][None, :], nan=0.0
        ) * beta[None, :]
        fixed[study_rows] = adapted_site[None, :] + parent_term

        residual_center = np.median(residual[study_train], axis=0).astype(np.float32)
        centered_residual[study_rows] = residual[study_rows] - residual_center[None, :]
        raw_target_train = truth[study_train] - fixed[study_train]
        target_center, _ = observed_median(raw_target_train, minimum=10)
        centered_target[study_rows] = truth[study_rows] - fixed[study_rows] - target_center[None, :]
    return fixed, centered_residual, centered_target


def m22_coordinates_all(
    primary_rna: np.ndarray,
    primary_protein: np.ndarray,
    primary_train: np.ndarray,
    external_rna: np.ndarray,
    external_protein: np.ndarray,
) -> np.ndarray:
    package = np.load(LATENT / "models/input_projection_and_ridge.npz")
    train_feature = transform(primary_rna[primary_train], primary_protein[primary_train], package)
    external_feature = transform(external_rna, external_protein, package)
    coordinate = np.load(LATENT / "models/train_coordinates_rank128.npy").astype(np.float32)
    mean = coordinate.mean(axis=0)
    scale = np.where(coordinate.std(axis=0) > 1e-6, coordinate.std(axis=0), 1.0)
    target = (coordinate - mean) / scale
    base = Ridge(alpha=1000.0).fit(train_feature, target)
    predicted = base.predict(external_feature).astype(np.float32) * scale + mean

    rna_train, rna_external = standardize(primary_rna[primary_train], external_rna)
    protein_train, protein_external = standardize(primary_protein[primary_train], external_protein)
    for latent in SELECTIVE:
        rna_columns = np.argsort(-np.abs(target[:, latent] @ rna_train))[:16]
        protein_columns = np.argsort(-np.abs(target[:, latent] @ protein_train))[:8]
        feature_train = np.concatenate([train_feature, rna_train[:, rna_columns], protein_train[:, protein_columns]], axis=1)
        feature_external = np.concatenate([external_feature, rna_external[:, rna_columns], protein_external[:, protein_columns]], axis=1)
        model = Ridge(alpha=1000.0, solver="lsqr").fit(feature_train, target[:, latent])
        predicted[:, latent] = model.predict(feature_external).astype(np.float32) * scale[latent] + mean[latent]
    return predicted


def fit_slope(centered_residual: np.ndarray, centered_target: np.ndarray, fit_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.isfinite(centered_residual[fit_rows]) & np.isfinite(centered_target[fit_rows])
    n_site = observed.sum(axis=0).astype(np.float32)
    numerator = np.nansum(
        np.where(observed, centered_residual[fit_rows] * centered_target[fit_rows], np.nan), axis=0
    ).astype(np.float32)
    square = np.nansum(np.where(observed, centered_residual[fit_rows] ** 2, np.nan), axis=0).astype(np.float32)
    raw = numerator / (square + LAMBDA)
    slope = (n_site / (n_site + KAPPA)) * raw + (KAPPA / (n_site + KAPPA)) * GAMMA_J
    return np.clip(slope, SLOPE_MIN, SLOPE_MAX).astype(np.float32), raw.astype(np.float32), n_site.astype(np.int32)


def add_metrics(rows: list[dict], truth: np.ndarray, prediction: np.ndarray, model: str, scope: str, studies: np.ndarray | None = None) -> None:
    patient = patient_spearman(truth, prediction)
    row = {
        "scope": scope,
        "model": model,
        "patients": len(truth),
        "median_patient_spearman": float(np.nanmedian(patient)),
        "mean_patient_spearman": float(np.nanmean(patient)),
        **site_summary(truth, prediction),
    }
    rows.append(row)
    if studies is not None:
        for study in sorted(np.unique(studies)):
            keep = studies == study
            rows.append(
                {
                    "scope": str(study),
                    "model": model,
                    "patients": int(keep.sum()),
                    "median_patient_spearman": float(np.nanmedian(patient[keep])),
                    "mean_patient_spearman": float(np.nanmean(patient[keep])),
                    **site_summary(truth[keep], prediction[keep]),
                }
            )


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    for subdir in ("tables", "predictions", "reports", "logs"):
        (OUT / subdir).mkdir(parents=True, exist_ok=True)

    primary_rna = numeric_frame(PRIMARY / "rna_reference_quantile.parquet")
    primary_protein = numeric_frame(PRIMARY / "protein_prediction.parquet").reindex(primary_rna.index)
    primary_truth = numeric_frame(PRIMARY / "phosphosite_logscale_aligned.parquet").reindex(primary_rna.index)
    primary_split = pd.read_csv(PRIMARY / "split_manifest.tsv", sep="\t").set_index("sample_id").reindex(primary_rna.index)
    primary_train = primary_split.role.eq("selection_train").to_numpy()

    external_rna = numeric_frame(EXTERNAL / "rna_reference_quantile.parquet")
    external_protein = numeric_frame(EXTERNAL / "protein_prediction.parquet").reindex(external_rna.index)
    external_truth = numeric_frame(EXTERNAL / "phosphosite_logscale_aligned.parquet").reindex(external_rna.index)
    external_split = pd.read_csv(EXTERNAL / "split_manifest.tsv", sep="\t").set_index("sample_id").reindex(external_rna.index)
    external_meta = pd.read_csv(EXTERNAL / "sample_metadata.tsv", sep="\t").set_index("sample_id").reindex(external_rna.index)
    calibration_mask = external_split.role.eq("selection_train").to_numpy()
    test_mask = external_split.role.eq("selection_validation").to_numpy()
    if (int(calibration_mask.sum()), int(test_mask.sum())) != (247, 106):
        raise ValueError("external split differs from 247/106")
    calibration_ids = external_rna.index[calibration_mask]
    test_ids = external_rna.index[test_mask]
    all_ids = calibration_ids.append(test_ids)

    targets = list(primary_truth.columns)
    if targets != list(external_truth.columns) or len(targets) != 18592:
        raise ValueError("phosphosite order differs")
    manifest = pd.read_csv(TARGET_MANIFEST, sep="\t")
    protein_lookup = {str(gene).upper(): index for index, gene in enumerate(primary_protein.columns)}
    parent_index = np.asarray([protein_lookup[str(gene).upper()] for gene in manifest.total_protein_gene], dtype=np.int64)
    beta = pd.read_csv(M1 / "parent_beta.tsv", sep="\t").beta.to_numpy(np.float32)
    primary_y = primary_truth.to_numpy(np.float32)
    primary_p = primary_protein.to_numpy(np.float32)
    primary_mean = np.nanmean(primary_y[primary_train], axis=0).astype(np.float32)
    primary_count = np.isfinite(primary_y[primary_train]).sum(axis=0)
    primary_protein_center = np.nanmean(primary_p[primary_train], axis=0).astype(np.float32)

    print("building frozen M2.2 coordinates for all 353 external patients", flush=True)
    all_coordinate = m22_coordinates_all(
        primary_rna.to_numpy(np.float32),
        primary_p,
        primary_train,
        external_rna.to_numpy(np.float32),
        external_protein.to_numpy(np.float32),
    )
    basis = np.load(LATENT / "models/site_basis_rank128.npy").astype(np.float32)
    site_mean = np.load(LATENT / "models/site_mean.npy").astype(np.float32)
    residual_all_external = all_coordinate @ basis

    external_p = external_protein.to_numpy(np.float32)
    strict_parent = np.nan_to_num(
        external_p[:, parent_index] - primary_protein_center[parent_index][None, :], nan=0.0
    ) * beta[None, :]
    strict_fixed = primary_mean[None, :] + strict_parent
    strict_full = strict_fixed + residual_all_external + site_mean[None, :]
    locked_test = numeric_frame(M22_LOCKED / "predictions/strict_cold_start_lr.parquet").reindex(index=test_ids, columns=targets).to_numpy(np.float32)
    strict_difference = float(np.nanmax(np.abs(strict_full[test_mask] - locked_test)))

    ordered_truth = np.full((len(all_ids), len(targets)), np.nan, dtype=np.float32)
    ordered_truth[: len(calibration_ids)] = external_truth.loc[calibration_ids].to_numpy(np.float32)
    ordered_protein = external_protein.loc[all_ids].to_numpy(np.float32)
    ordered_residual = np.vstack([residual_all_external[calibration_mask], residual_all_external[test_mask]]).astype(np.float32)
    studies = external_meta.loc[all_ids, "study"].astype(str).to_numpy()
    fold_table = pd.read_csv(PT4_PROTOCOL / "calibration_fold_assignments.tsv", sep="\t").set_index("sample_id").reindex(calibration_ids)
    if fold_table.isna().any().any():
        raise ValueError("Main PT3 fold assignments do not cover M2.2 calibration patients")
    fold_id = fold_table.fold.to_numpy(np.int64)
    if set(fold_id) != {0, 1, 2, 3, 4}:
        raise ValueError("unexpected fold identifiers")
    fold_table.assign(study=studies[: len(calibration_ids)]).to_csv(OUT / "tables/m22_pt3_fold_assignments.tsv", sep="\t")

    calibration_rows = np.arange(len(calibration_ids))
    oof_fixed = np.full((len(calibration_ids), len(targets)), np.nan, dtype=np.float32)
    oof_centered = np.full_like(oof_fixed, np.nan)
    oof_target = np.full_like(oof_fixed, np.nan)
    oof_pt3 = np.full_like(oof_fixed, np.nan)
    slope_rows = []
    for fold in sorted(np.unique(fold_id)):
        train_rows = calibration_rows[fold_id != fold]
        held_rows = calibration_rows[fold_id == fold]
        fixed, centered, target = fold_components(
            ordered_truth[: len(calibration_ids)],
            ordered_protein[: len(calibration_ids)],
            ordered_residual[: len(calibration_ids)],
            studies[: len(calibration_ids)],
            train_rows,
            primary_mean,
            primary_count,
            primary_protein_center,
            parent_index,
            beta,
        )
        slope, raw, n_site = fit_slope(centered, target, train_rows)
        oof_fixed[held_rows] = fixed[held_rows]
        oof_centered[held_rows] = centered[held_rows]
        oof_target[held_rows] = target[held_rows]
        oof_pt3[held_rows] = fixed[held_rows] + centered[held_rows] * slope[None, :]
        slope_rows.append(
            {
                "fold": int(fold), "gamma_J": GAMMA_J, "lambda": LAMBDA, "kappa": KAPPA,
                "median_raw_slope": float(np.median(raw)), "median_shrunk_slope": float(np.median(slope)),
                "median_abs_shrunk_slope": float(np.median(np.abs(slope))),
                "lower_clip_fraction": float(np.mean(slope <= SLOPE_MIN)), "upper_clip_fraction": float(np.mean(slope >= SLOPE_MAX)),
                "median_n_site": float(np.median(n_site)),
            }
        )
    if not np.isfinite(oof_fixed).all() or not np.isfinite(oof_centered).all() or not np.isfinite(oof_pt3).all():
        raise RuntimeError("incomplete five-fold OOF predictions")

    oof_models = {
        "PT0_adapted_fixed": oof_fixed,
        "M2.2_PT2_gamma_0.20": oof_fixed + GAMMA_J * oof_centered,
        "M2.2_PT3_sitewise": oof_pt3,
    }
    oof_rows: list[dict] = []
    for name, prediction in oof_models.items():
        add_metrics(oof_rows, ordered_truth[: len(calibration_ids)], prediction, name, "calibration_247_OOF", studies[: len(calibration_ids)])
    oof_table = pd.DataFrame(oof_rows)
    oof_table.to_csv(OUT / "tables/m22_pt3_247_oof_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(slope_rows).to_csv(OUT / "tables/m22_pt3_oof_slope_diagnostics.tsv", sep="\t", index=False)

    fixed_all, centered_all, target_all = fold_components(
        ordered_truth,
        ordered_protein,
        ordered_residual,
        studies,
        calibration_rows,
        primary_mean,
        primary_count,
        primary_protein_center,
        parent_index,
        beta,
    )
    slope, raw, n_site = fit_slope(centered_all, target_all, calibration_rows)
    test_rows = np.arange(len(calibration_ids), len(all_ids))
    final_models = {
        "PT0_adapted_fixed": fixed_all[test_rows],
        "M2.2_PT2_gamma_0.20": fixed_all[test_rows] + GAMMA_J * centered_all[test_rows],
        "M2.2_PT3_sitewise": fixed_all[test_rows] + centered_all[test_rows] * slope[None, :],
        "M2.2_frozen_residual": fixed_all[test_rows] + centered_all[test_rows],
    }
    truth_test = external_truth.loc[test_ids].to_numpy(np.float32)
    test_studies = external_meta.loc[test_ids, "study"].astype(str).to_numpy()
    final_rows: list[dict] = []
    patient_rows: list[dict] = []
    for name, prediction in final_models.items():
        add_metrics(final_rows, truth_test, prediction, name, "combined_106", test_studies)
        patient = patient_spearman(truth_test, prediction)
        patient_rows.extend(
            {
                "sample_id": sample_id, "study": study, "model": name,
                "patient_spearman": score, "n_observed_sites": int(np.isfinite(truth_test[row]).sum()),
            }
            for row, (sample_id, study, score) in enumerate(zip(test_ids, test_studies, patient))
        )
    final_table = pd.DataFrame(final_rows)
    final_table.to_csv(OUT / "m22_pt3_106_model_summary.tsv", sep="\t", index=False)
    pd.DataFrame(patient_rows).to_csv(OUT / "tables/m22_pt3_106_per_patient.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "target": targets, "n_calibration_observed": n_site, "raw_slope": raw, "shrunk_slope": slope,
            "at_lower_clip": slope <= SLOPE_MIN, "at_upper_clip": slope >= SLOPE_MAX,
        }
    ).to_csv(OUT / "tables/m22_pt3_final_site_slopes.tsv", sep="\t", index=False)
    np.savez_compressed(
        OUT / "predictions/m22_pt3_106_predictions.npz",
        sample_ids=np.asarray(test_ids, dtype=str), site_ids=np.asarray(targets, dtype=str),
        PT0_adapted_fixed=final_models["PT0_adapted_fixed"],
        M22_PT2=final_models["M2.2_PT2_gamma_0.20"],
        M22_PT3=final_models["M2.2_PT3_sitewise"],
    )

    oof_pt3_score = float(
        oof_table.loc[
            oof_table.scope.eq("calibration_247_OOF") & oof_table.model.eq("M2.2_PT3_sitewise"),
            "median_patient_spearman",
        ].iloc[0]
    )
    combined = final_table[final_table.scope.eq("combined_106")].copy()
    pt0_score = float(combined.loc[combined.model.eq("PT0_adapted_fixed"), "median_patient_spearman"].iloc[0])
    pt3_score = float(combined.loc[combined.model.eq("M2.2_PT3_sitewise"), "median_patient_spearman"].iloc[0])
    comparison = pd.DataFrame(
        [
            {"model": "PT0_adapted_fixed_shared", "median_patient_spearman": pt0_score},
            {"model": "M2.2_PT3_matched", "median_patient_spearman": pt3_score},
        ]
    )
    comparison["delta_vs_PT0"] = comparison.median_patient_spearman - pt0_score
    comparison.to_csv(OUT / "fair_main_vs_m22_pt3_106.tsv", sep="\t", index=False)

    checks = pd.DataFrame(
        [
            {"check": "M2.2严格106预测复现（容差1e-4）", "status": "PASS" if strict_difference < 1e-4 else "FAIL", "value": strict_difference},
            {"check": "247例五折划分使用M2.2_PT4协议", "status": "PASS", "value": str(len(calibration_ids))},
            {"check": "锁定参数", "status": "PASS", "value": f"gamma={GAMMA_J},lambda={LAMBDA},kappa={KAPPA}"},
            {"check": "graph关闭", "status": "PASS", "value": "0"},
            {"check": "106例测试标签未进入斜率拟合", "status": "PASS", "value": str(len(test_ids))},
            {"check": "预测维度", "status": "PASS" if final_models["M2.2_PT3_sitewise"].shape == (106, 18592) else "FAIL", "value": str(final_models["M2.2_PT3_sitewise"].shape)},
        ]
    )
    checks.to_csv(OUT / "tables/m22_pt3_identity_checks.tsv", sep="\t", index=False)
    if not checks.status.eq("PASS").all():
        raise RuntimeError(checks.to_string(index=False))

    report = "\n".join(
        [
            "# M2.2-PT3 matched Main-BORP calibration", "",
            f"固定参数：gamma_J={GAMMA_J:g}，lambda={LAMBDA:g}，kappa={KAPPA:g}，graph=0。",
            f"M2.2 PT3的247例五折OOF患者内Spearman中位数为`{oof_pt3_score:.6f}`。",
            f"M2.2 PT3的106例最终患者内Spearman中位数为`{pt3_score:.6f}`。",
            f"共享PT0为`{pt0_score:.6f}`，M2.2 PT3相对PT0变化`{pt3_score - pt0_score:+.6f}`。",
            "247例用于斜率拟合和五折OOF；106例仅用于冻结最终评价。",
            "",
        ]
    )
    (OUT / "RESULTS_20260826.md").write_text(report, encoding="utf-8")
    summary = {
        "status": "complete", "locked_parameters": {"gamma_J": GAMMA_J, "lambda": LAMBDA, "kappa": KAPPA, "graph": 0},
        "strict_m22_reproduction_max_abs_error": strict_difference,
        "m22_pt3_oof": oof_pt3_score,
        "m22_pt3_106": pt3_score, "pt0_106": pt0_score,
        "test_labels_used_for_selection": False,
    }
    (OUT / "reports/run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "SUCCESS").touch()
    print(comparison.to_string(index=False), flush=True)
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
