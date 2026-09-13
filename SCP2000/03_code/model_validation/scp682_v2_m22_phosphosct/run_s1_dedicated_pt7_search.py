#!/usr/bin/env python3
"""PhosphoSCT-S1 专属 PT7-R/PT7-Y 搜索及 106 例评价。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[3]
FUSION_SCRIPT = ROOT / "03_code/model_validation/scp682_v2_m22_zero_aware_pt7/run_m22_f_pt7_prediction_fusion.py"
S1 = ROOT / "02_results/model_validation/scp682_v2_m23_phosphosct_s1_20260831/models/phosphosct_s1_model.npz"
M2_PREDICTION = ROOT / "02_results/external_validation/scp682_v2_m22_f_pt7_fusion_20260831_v2/predictions/fusion_selected_predictions.npz"
OUT = ROOT / "02_results/external_validation/scp682_v2_m23_phosphosct_s1_dedicated_pt7_20260831"
GAMMAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50)
LAMBDAS = (0.3, 1.0, 3.0, 10.0, 30.0)
KAPPA_COARSE = 100.0
KAPPAS_LOCAL = (50.0, 100.0, 200.0, 400.0)
ETA = 0.001
ADAPTER_ALPHA = 10.0
ADAPTER_RANK = 24
SCALE_SHRINKAGE = 50.0
SLOPE_LIMIT = 0.5


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(values, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(values - center[None, :]), axis=0)
    count = np.isfinite(values).sum(axis=0).astype(np.float32)
    return scale.astype(np.float32), count


def scale_by_study(
    target: np.ndarray,
    studies: np.ndarray,
    fit_rows: np.ndarray,
    prior_scale: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    output = np.empty((len(np.unique(studies)), target.shape[1]), dtype=np.float32)
    names = sorted(np.unique(studies))
    for index, study in enumerate(names):
        fit = fit_rows[studies[fit_rows] == study]
        local, count = robust_scale(target[fit])
        local = np.where(np.isfinite(local) & (local > 1e-6), local, prior_scale)
        weight = count / (count + SCALE_SHRINKAGE)
        scale = np.sqrt(weight * np.square(local) + (1.0 - weight) * np.square(prior_scale))
        output[index] = np.maximum(scale, tau)
    lookup = {study: output[index] for index, study in enumerate(names)}
    return np.stack([lookup[str(study)] for study in studies]).astype(np.float32)


def slopes(source: np.ndarray, target: np.ndarray, rows: np.ndarray, gamma: float, lam: float, kappa: float):
    observed = np.isfinite(source[rows]) & np.isfinite(target[rows])
    count = observed.sum(axis=0).astype(np.float32)
    numerator = np.nansum(np.where(observed, source[rows] * target[rows], np.nan), axis=0).astype(np.float32)
    square = np.nansum(np.where(observed, np.square(source[rows]), np.nan), axis=0).astype(np.float32)
    raw = numerator / (square + lam)
    shrunk = count / (count + kappa) * raw + kappa / (count + kappa) * gamma
    return np.clip(shrunk, -SLOPE_LIMIT, SLOPE_LIMIT).astype(np.float32)


def center_source(source: np.ndarray, studies: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    output = np.empty_like(source, dtype=np.float32)
    for study in np.unique(studies):
        rows = np.flatnonzero(studies == study)
        fit = fit_rows[studies[fit_rows] == study]
        output[rows] = source[rows] - np.median(source[fit], axis=0).astype(np.float32)[None, :]
    return output


def reduced_delta(model: Ridge, train_prediction: np.ndarray, all_prediction: np.ndarray) -> np.ndarray:
    fitted = model.predict(train_prediction).astype(np.float32)
    _, _, right = np.linalg.svd(fitted, full_matrices=False)
    predicted = model.predict(all_prediction).astype(np.float32)
    return (predicted @ right[:ADAPTER_RANK].T @ right[:ADAPTER_RANK]).astype(np.float32)


def prepare_contexts(
    module,
    truth,
    protein,
    studies,
    coordinate,
    basis,
    fold_ids,
    primary_mean,
    primary_count,
    primary_protein_center,
    parent_index,
    beta,
    prior_scale,
    tau,
):
    rows = np.arange(len(truth))
    contexts = []
    source_standard = coordinate @ basis
    for fold in sorted(np.unique(fold_ids)):
        fit = rows[fold_ids != fold]
        held = rows[fold_ids == fold]
        fixed, _, target_raw = module.core.fold_components(
            truth, protein, source_standard, studies, fit,
            primary_mean, primary_count, primary_protein_center, parent_index, beta,
        )
        scale = scale_by_study(target_raw, studies, fit, prior_scale, tau)
        target_standard = target_raw / scale
        z_prediction = module.center_coordinates(coordinate, studies, fit)
        z_target = module.masked_projection(target_standard, basis, ETA)
        adapter = Ridge(alpha=ADAPTER_ALPHA, fit_intercept=False).fit(
            z_prediction[fit], z_target[fit] - z_prediction[fit]
        )
        z_aligned = z_prediction + reduced_delta(adapter, z_prediction[fit], z_prediction)
        aligned_standard = center_source(z_aligned @ basis, studies, fit)
        contexts.append(
            {
                "fit": fit,
                "held": held,
                "fixed": fixed,
                "scale": scale,
                "target_raw": target_raw,
                "target_standard": target_standard,
                "aligned_standard": aligned_standard,
            }
        )
        print(f"prepared fold {int(fold) + 1}/{len(np.unique(fold_ids))}", flush=True)
    return contexts


def predict_candidate(contexts, shape, route, gamma, lam, kappa):
    prediction = np.full(shape, np.nan, dtype=np.float32)
    for context in contexts:
        fit = context["fit"]
        held = context["held"]
        aligned_standard = context["aligned_standard"]
        scale = context["scale"]
        if route == "PT7_R":
            slope = slopes(aligned_standard, context["target_standard"], fit, gamma, lam, kappa)
            correction = scale[held] * aligned_standard[held] * slope[None, :]
        else:
            aligned_raw = scale * aligned_standard
            slope = slopes(aligned_raw, context["target_raw"], fit, gamma, lam, kappa)
            correction = aligned_raw[held] * slope[None, :]
        prediction[held] = context["fixed"][held] + correction
    return prediction


def fast_row(module, truth, prediction, studies, route, gamma, lam, kappa, stage):
    metric = module.fast_pearson_metrics(truth, prediction, studies)
    y, p = module.center_pair_within_study(truth, prediction, studies)
    observed = np.isfinite(y) & np.isfinite(p)
    patient_mse = np.nanmean(np.where(observed, np.square(y - p), np.nan), axis=1)
    score = 0.60 * metric["centered_patient_pearson"] + 0.40 * metric["centered_site_pearson"]
    return {
        "stage": stage,
        "route": route,
        "gamma": gamma,
        "lambda": lam,
        "kappa": kappa,
        "patient_pearson": metric["centered_patient_pearson"],
        "site_pearson": metric["centered_site_pearson"],
        "patient_equal_mse": float(np.nanmean(patient_mse)),
        "selection_score": score,
    }


def evaluate_grid(module, contexts, truth, studies, candidates, stage):
    rows = []
    for index, (route, gamma, lam, kappa) in enumerate(candidates):
        prediction = predict_candidate(contexts, truth.shape, route, gamma, lam, kappa)
        rows.append(fast_row(module, truth, prediction, studies, route, gamma, lam, kappa, stage))
        if (index + 1) % 20 == 0:
            print(f"evaluated {stage} {index + 1}/{len(candidates)}", flush=True)
    return pd.DataFrame(rows)


def local_candidates(top: pd.DataFrame):
    candidates = set()
    for _, row in top.iterrows():
        for gamma in (row["gamma"] - 0.05, row["gamma"] - 0.025, row["gamma"], row["gamma"] + 0.025, row["gamma"] + 0.05):
            if gamma < 0:
                continue
            for lam in (row["lambda"] * 0.5, row["lambda"], row["lambda"] * 2.0):
                for kappa in KAPPAS_LOCAL:
                    candidates.add((row["route"], round(float(gamma), 6), round(float(lam), 6), float(kappa)))
    return sorted(candidates)


def fit_full(
    module, truth, protein, studies, coordinate, basis, n_calibration,
    primary_mean, primary_count, primary_protein_center, parent_index, beta,
    prior_scale, tau, route, gamma, lam, kappa,
):
    calibration = np.arange(n_calibration)
    source_standard = coordinate @ basis
    fixed, _, target_raw = module.core.fold_components(
        truth, protein, source_standard, studies, calibration,
        primary_mean, primary_count, primary_protein_center, parent_index, beta,
    )
    scale = scale_by_study(target_raw, studies, calibration, prior_scale, tau)
    target_standard = target_raw / scale
    z_prediction = module.center_coordinates(coordinate, studies, calibration)
    z_target = module.masked_projection(target_standard[:n_calibration], basis, ETA)
    adapter = Ridge(alpha=ADAPTER_ALPHA, fit_intercept=False).fit(
        z_prediction[calibration], z_target - z_prediction[calibration]
    )
    z_aligned = z_prediction + reduced_delta(adapter, z_prediction[calibration], z_prediction)
    aligned_standard = center_source(z_aligned @ basis, studies, calibration)
    if route == "PT7_R":
        slope = slopes(aligned_standard, target_standard, calibration, gamma, lam, kappa)
        correction = scale * aligned_standard * slope[None, :]
    else:
        aligned_raw = scale * aligned_standard
        slope = slopes(aligned_raw, target_raw, calibration, gamma, lam, kappa)
        correction = aligned_raw * slope[None, :]
    return fixed + correction, slope


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    for name in ("tables", "predictions", "reports", "logs"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    fusion = load_module(FUSION_SCRIPT, "fusion_pipeline")
    module = fusion.load_pipeline()
    module.core.LATENT = module.LATENT

    primary_rna = module.core.numeric_frame(module.PRIMARY / "rna_reference_quantile.parquet")
    primary_protein = module.core.numeric_frame(module.PRIMARY / "protein_prediction.parquet").reindex(primary_rna.index)
    primary_truth = module.core.numeric_frame(module.PRIMARY / "phosphosite_logscale_aligned.parquet").reindex(primary_rna.index)
    primary_split = pd.read_csv(module.PRIMARY / "split_manifest.tsv", sep="\t").set_index("sample_id").reindex(primary_rna.index)
    primary_train = primary_split.role.eq("selection_train").to_numpy()
    external_rna = module.core.numeric_frame(module.EXTERNAL / "rna_reference_quantile.parquet")
    external_protein = module.core.numeric_frame(module.EXTERNAL / "protein_prediction.parquet").reindex(external_rna.index)
    external_truth = module.core.numeric_frame(module.EXTERNAL / "phosphosite_logscale_aligned.parquet").reindex(external_rna.index)
    external_split = pd.read_csv(module.EXTERNAL / "split_manifest.tsv", sep="\t").set_index("sample_id").reindex(external_rna.index)
    external_meta = pd.read_csv(module.EXTERNAL / "sample_metadata.tsv", sep="\t").set_index("sample_id").reindex(external_rna.index)
    calibration_ids = external_rna.index[external_split.role.eq("selection_train")]
    test_ids = external_rna.index[external_split.role.eq("selection_validation")]
    all_ids = calibration_ids.append(test_ids)

    manifest = pd.read_csv(module.MANIFEST, sep="\t")
    lookup = {str(gene).upper(): index for index, gene in enumerate(primary_protein.columns.astype(str))}
    parent_index = np.asarray([lookup[str(gene).upper()] for gene in manifest.total_protein_gene], dtype=np.int64)
    beta = pd.read_csv(module.M1 / "parent_beta.tsv", sep="\t").beta.to_numpy(np.float32)
    primary_y = primary_truth.to_numpy(np.float32)
    primary_p = primary_protein.to_numpy(np.float32)
    primary_mean = np.nanmean(primary_y[primary_train], axis=0).astype(np.float32)
    primary_count = np.isfinite(primary_y[primary_train]).sum(axis=0)
    primary_protein_center = np.nanmean(primary_p[primary_train], axis=0).astype(np.float32)

    old_package = np.load(module.LATENT / "models/input_projection_and_ridge.npz")
    s1 = np.load(S1)
    external_feature = fusion.transform_inputs(
        external_rna.to_numpy(np.float32), external_protein.to_numpy(np.float32), old_package
    )
    all_coordinate = (external_feature @ s1["ridge_coef"].T + s1["ridge_intercept"][None, :]).astype(np.float32)
    basis = s1["basis"].astype(np.float32)
    prior_scale = s1["global_scale"].astype(np.float32)
    tau = s1["tau"].astype(np.float32)

    calibration_position = external_rna.index.get_indexer(calibration_ids)
    ordered_position = external_rna.index.get_indexer(all_ids)
    truth_cal = external_truth.loc[calibration_ids].to_numpy(np.float32)
    protein_cal = external_protein.loc[calibration_ids].to_numpy(np.float32)
    studies_cal = external_meta.loc[calibration_ids].study.astype(str).to_numpy()
    coordinate_cal = all_coordinate[calibration_position]
    five_fold = pd.read_csv(module.PROTOCOL / "calibration_fold_assignments.tsv", sep="\t").set_index("sample_id").reindex(calibration_ids).fold.to_numpy(np.int64)

    contexts5 = prepare_contexts(
        module, truth_cal, protein_cal, studies_cal, coordinate_cal, basis, five_fold,
        primary_mean, primary_count, primary_protein_center, parent_index, beta, prior_scale, tau,
    )
    coarse_candidates = [
        (route, gamma, lam, KAPPA_COARSE)
        for route in ("PT7_R", "PT7_Y") for gamma in GAMMAS for lam in LAMBDAS
    ]
    coarse = evaluate_grid(module, contexts5, truth_cal, studies_cal, coarse_candidates, "coarse_5fold")
    coarse["coarse_rank"] = coarse.groupby("route")["selection_score"].rank(method="first", ascending=False)
    top8 = coarse.sort_values("selection_score", ascending=False).head(8)

    three_fold = np.empty(len(studies_cal), dtype=np.int64)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=20260831)
    for fold, (_, held) in enumerate(splitter.split(np.zeros(len(studies_cal)), studies_cal)):
        three_fold[held] = fold
    contexts3 = prepare_contexts(
        module, truth_cal, protein_cal, studies_cal, coordinate_cal, basis, three_fold,
        primary_mean, primary_count, primary_protein_center, parent_index, beta, prior_scale, tau,
    )
    top8_candidates = [
        (row["route"], float(row["gamma"]), float(row["lambda"]), float(row["kappa"]))
        for _, row in top8.iterrows()
    ]
    top8_threefold = evaluate_grid(module, contexts3, truth_cal, studies_cal, top8_candidates, "top8_3fold")
    top3 = top8_threefold.sort_values("selection_score", ascending=False).head(3)
    local = evaluate_grid(module, contexts3, truth_cal, studies_cal, local_candidates(top3), "local_3fold")
    selected = local.sort_values(
        ["selection_score", "patient_pearson", "site_pearson", "patient_equal_mse"],
        ascending=[False, False, False, True],
    ).iloc[0]
    route = str(selected.route)
    gamma = float(selected.gamma)
    lam = float(selected["lambda"])
    kappa = float(selected.kappa)

    selected_oof = predict_candidate(contexts5, truth_cal.shape, route, gamma, lam, kappa)
    ordered_truth = external_truth.loc[all_ids].to_numpy(np.float32)
    ordered_protein = external_protein.loc[all_ids].to_numpy(np.float32)
    ordered_studies = external_meta.loc[all_ids].study.astype(str).to_numpy()
    ordered_coordinate = all_coordinate[ordered_position]
    full_prediction, final_slope = fit_full(
        module, ordered_truth, ordered_protein, ordered_studies, ordered_coordinate, basis,
        len(calibration_ids), primary_mean, primary_count, primary_protein_center,
        parent_index, beta, prior_scale, tau, route, gamma, lam, kappa,
    )
    selected_test = full_prediction[len(calibration_ids):]
    truth_test = external_truth.loc[test_ids].to_numpy(np.float32)
    studies_test = external_meta.loc[test_ids].study.astype(str).to_numpy()

    m2 = np.load(M2_PREDICTION, allow_pickle=True)
    comparison_rows = []
    for model_name, scope, truth, prediction, studies in (
        ("M2.2_PT7", "external_247_OOF", truth_cal, m2["m2_oof"].astype(np.float32), studies_cal),
        ("PhosphoSCT_S1_dedicated_PT7", "external_247_OOF", truth_cal, selected_oof, studies_cal),
        ("M2.2_PT7", "external_106_locked_test", truth_test, m2["selected_test"].astype(np.float32), studies_test),
        ("PhosphoSCT_S1_dedicated_PT7", "external_106_locked_test", truth_test, selected_test, studies_test),
    ):
        comparison_rows.extend(module.metric_pair(truth, prediction, studies, model_name, scope))
    comparison = pd.DataFrame(comparison_rows)
    coarse.to_csv(OUT / "tables/coarse_grid.tsv", sep="\t", index=False)
    top8_threefold.to_csv(OUT / "tables/top8_threefold.tsv", sep="\t", index=False)
    local.to_csv(OUT / "tables/local_grid.tsv", sep="\t", index=False)
    comparison.to_csv(OUT / "tables/final_comparison.tsv", sep="\t", index=False)
    pd.DataFrame({"phosphosite": external_truth.columns.astype(str), "slope": final_slope}).to_csv(
        OUT / "tables/final_site_slopes.tsv", sep="\t", index=False
    )
    np.savez_compressed(
        OUT / "predictions/s1_dedicated_pt7_predictions.npz",
        route=np.asarray(route), gamma=np.asarray(gamma), lambda_value=np.asarray(lam), kappa=np.asarray(kappa),
        calibration_ids=np.asarray(calibration_ids, str), test_ids=np.asarray(test_ids, str),
        site_ids=external_truth.columns.astype(str).to_numpy(), oof=selected_oof, test=selected_test,
    )
    centered = comparison.query("coordinate == 'study_centered'")
    report = "\n".join(
        [
            "# PhosphoSCT-S1 专属 PT7",
            "",
            f"选择 `{route}`：gamma=`{gamma:g}`，lambda=`{lam:g}`，kappa=`{kappa:g}`。",
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
            {"status": "complete", "selected": {"route": route, "gamma": gamma, "lambda": lam, "kappa": kappa}, "metrics": comparison.to_dict("records")},
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (OUT / "SUCCESS").touch()
    print(top8_threefold.sort_values("selection_score", ascending=False).to_string(index=False), flush=True)
    print(local.sort_values("selection_score", ascending=False).head(20).to_string(index=False), flush=True)
    print(centered.to_string(index=False), flush=True)
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
