#!/usr/bin/env python3
"""重建 M2.2 的训练五折和开发验证预测，并生成研究内中心化指标。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from study_centered_metrics import (  # noqa: E402
    center_pair_within_study,
    metric_pair,
    observed_axis_pearson,
    site_spearman,
)


INPUT = ROOT / "01_data/bulk/intermediate/random70_reference_rna_protein_logscale_phosphosite_inputs_20260815"
LATENT = ROOT / "01_data/multi_omics/intermediate/scp682_v2_m2_residual_latent_20260823/models"
M1 = ROOT / "01_data/multi_omics/intermediate/scp682_v2_m1_parent_ptm_decomposition_20260822"
MANIFEST = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2/residual_target_manifest.tsv"
OUT = ROOT / "02_results/model_validation/scp682_v2_m22_study_centered_evaluation_20260830"
SELECTIVE = np.asarray([15, 18, 23, 30, 41, 48, 55, 58], dtype=np.int64)
SEED = 20260823
MINIMUM_PLOT_PATIENTS = 1000


def transform_inputs(rna: np.ndarray, protein: np.ndarray, package: np.lib.npyio.NpzFile) -> np.ndarray:
    rna_z = np.nan_to_num((rna - package["rna_mean"]) / package["rna_scale"], nan=0.0).astype(np.float32)
    protein_z = np.nan_to_num((protein - package["protein_mean"]) / package["protein_scale"], nan=0.0).astype(np.float32)
    feature = np.concatenate(
        [rna_z @ package["rna_components"].T, protein_z @ package["protein_components"].T], axis=1
    )
    return ((feature - package["feature_mean"]) / package["feature_scale"]).astype(np.float32)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0).astype(np.float32)
    scale = np.nanstd(train, axis=0).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    return (
        np.nan_to_num((train - mean) / scale, nan=0.0).astype(np.float32),
        np.nan_to_num((test - mean) / scale, nan=0.0).astype(np.float32),
    )


def fit_fixed_offset(
    truth: np.ndarray,
    protein: np.ndarray,
    studies: np.ndarray,
    training: np.ndarray,
    parent_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    global_site = np.nan_to_num(np.nanmean(truth[training], axis=0), nan=0.0).astype(np.float32)
    global_count = np.isfinite(truth[training]).sum(axis=0).astype(np.float32)
    global_protein = np.nan_to_num(np.nanmean(protein[training], axis=0), nan=0.0).astype(np.float32)
    site_center = np.empty_like(truth, dtype=np.float32)
    protein_center = np.empty_like(protein, dtype=np.float32)
    training_mask = np.zeros(len(truth), dtype=bool)
    training_mask[training] = True
    for study in np.unique(studies):
        rows = np.flatnonzero(studies == study)
        fit_rows = rows[training_mask[rows]]
        if len(fit_rows):
            values = truth[fit_rows]
            count = np.isfinite(values).sum(axis=0).astype(np.float32)
            mean = np.divide(
                np.nansum(values, axis=0, dtype=np.float64),
                count,
                out=np.zeros(truth.shape[1], dtype=np.float64),
                where=count > 0,
            ).astype(np.float32)
            common = (count >= 5) & (global_count >= 5)
            offset = float(np.median(mean[common] - global_site[common])) if common.any() else 0.0
            fallback = global_site + offset
            mean = np.where(count > 0, mean, fallback)
            weight = count / (count + 2.0)
            site_center[rows] = (weight * mean + (1.0 - weight) * fallback)[None, :]
            local_protein = np.nanmean(protein[fit_rows], axis=0).astype(np.float32)
            local_protein = np.where(np.isfinite(local_protein), local_protein, global_protein)
            protein_center[rows] = local_protein[None, :]
        else:
            site_center[rows] = global_site[None, :]
            protein_center[rows] = global_protein[None, :]

    protein_centered = protein - protein_center
    beta = np.zeros(truth.shape[1], dtype=np.float32)
    residual_after_study = truth - site_center
    for start in range(0, truth.shape[1], 1000):
        stop = min(start + 1000, truth.shape[1])
        x = protein_centered[:, parent_index[start:stop]]
        y = residual_after_study[:, start:stop]
        observed = np.isfinite(x[training]) & np.isfinite(y[training])
        count = observed.sum(axis=0)
        numerator = np.where(observed, x[training] * y[training], 0.0).sum(axis=0)
        denominator = np.where(observed, x[training] ** 2, 0.0).sum(axis=0)
        slope = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-8)
        slope *= count / (count + 20.0)
        slope[count < 16] = 0.0
        beta[start:stop] = slope.astype(np.float32)
    parent = np.nan_to_num(protein_centered[:, parent_index], nan=0.0) * beta[None, :]
    return (site_center + parent).astype(np.float32), beta


def fit_m22_coordinates(
    feature_train: np.ndarray,
    feature_dev: np.ndarray,
    rna_train: np.ndarray,
    rna_dev: np.ndarray,
    protein_train: np.ndarray,
    protein_dev: np.ndarray,
    coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_mean = coordinate.mean(axis=0).astype(np.float32)
    target_scale = np.where(coordinate.std(axis=0) > 1e-6, coordinate.std(axis=0), 1.0).astype(np.float32)
    target = ((coordinate - target_mean) / target_scale).astype(np.float32)
    oof = np.empty_like(coordinate, dtype=np.float32)
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fit, held in folds.split(feature_train):
        base = Ridge(alpha=1000.0).fit(feature_train[fit], target[fit])
        oof[held] = base.predict(feature_train[held]).astype(np.float32) * target_scale + target_mean
        rna_fit, rna_held = standardize(rna_train[fit], rna_train[held])
        protein_fit, protein_held = standardize(protein_train[fit], protein_train[held])
        for latent in SELECTIVE:
            rna_columns = np.argsort(-np.abs(target[fit, latent] @ rna_fit))[:16]
            protein_columns = np.argsort(-np.abs(target[fit, latent] @ protein_fit))[:8]
            x_fit = np.concatenate([feature_train[fit], rna_fit[:, rna_columns], protein_fit[:, protein_columns]], axis=1)
            x_held = np.concatenate([feature_train[held], rna_held[:, rna_columns], protein_held[:, protein_columns]], axis=1)
            model = Ridge(alpha=1000.0, solver="lsqr").fit(x_fit, target[fit, latent])
            oof[held, latent] = model.predict(x_held).astype(np.float32) * target_scale[latent] + target_mean[latent]

    base = Ridge(alpha=1000.0).fit(feature_train, target)
    dev = base.predict(feature_dev).astype(np.float32) * target_scale + target_mean
    rna_fit, rna_test = standardize(rna_train, rna_dev)
    protein_fit, protein_test = standardize(protein_train, protein_dev)
    for latent in SELECTIVE:
        rna_columns = np.argsort(-np.abs(target[:, latent] @ rna_fit))[:16]
        protein_columns = np.argsort(-np.abs(target[:, latent] @ protein_fit))[:8]
        x_fit = np.concatenate([feature_train, rna_fit[:, rna_columns], protein_fit[:, protein_columns]], axis=1)
        x_test = np.concatenate([feature_dev, rna_test[:, rna_columns], protein_test[:, protein_columns]], axis=1)
        model = Ridge(alpha=1000.0, solver="lsqr").fit(x_fit, target[:, latent])
        dev[:, latent] = model.predict(x_test).astype(np.float32) * target_scale[latent] + target_mean[latent]
    return oof, dev


def per_site_table(truth: np.ndarray, prediction: np.ndarray, studies: np.ndarray, sites: np.ndarray) -> pd.DataFrame:
    y, p = center_pair_within_study(truth, prediction, studies)
    pearson = observed_axis_pearson(y, p, axis=0)
    spearman = site_spearman(y, p)
    observed = np.isfinite(y) & np.isfinite(p)
    count = observed.sum(axis=0)
    mse = np.nanmean(np.where(observed, (y - p) ** 2, np.nan), axis=0)
    return pd.DataFrame(
        {"phosphosite": sites, "patients_observed": count, "study_centered_pearson": pearson, "study_centered_spearman": spearman, "study_centered_mse": mse}
    ).sort_values(["study_centered_pearson", "study_centered_spearman"], ascending=False, ignore_index=True)


def marginal(axis: plt.Axes, values: np.ndarray, bins: np.ndarray, horizontal: bool = False) -> None:
    count, edges = np.histogram(values, bins=bins, density=True)
    center = (edges[:-1] + edges[1:]) / 2
    if horizontal:
        axis.fill_betweenx(center, 0, count, color="#A9BBDC", alpha=0.9, linewidth=0)
        axis.plot(count, center, color="#8FA7CE", linewidth=0.7)
    else:
        axis.fill_between(center, 0, count, color="#A9BBDC", alpha=0.9, linewidth=0)
        axis.plot(center, count, color="#8FA7CE", linewidth=0.7)


def draw_panel(subfigure, measured: np.ndarray, fitted: np.ndarray, record: pd.Series) -> None:
    grid = subfigure.add_gridspec(4, 4, height_ratios=(0.30, 1, 1, 1), width_ratios=(1, 1, 1, 0.30), hspace=0.05, wspace=0.05)
    top = subfigure.add_subplot(grid[0, :3])
    main = subfigure.add_subplot(grid[1:, :3])
    right = subfigure.add_subplot(grid[1:, 3])
    low, high = np.nanpercentile(np.concatenate([measured, fitted]), [0.5, 99.5])
    pad = max((high - low) * 0.07, 0.05)
    low -= pad
    high += pad
    bins = np.linspace(low, high, 28)
    marginal(top, measured, bins)
    marginal(right, fitted, bins, horizontal=True)
    main.scatter(measured, fitted, s=2.2, color="#7899CA", alpha=0.17, linewidths=0, rasterized=True)
    main.plot([low, high], [low, high], color="#777777", linewidth=0.75, linestyle=(0, (2, 2)))
    main.set(xlim=(low, high), ylim=(low, high), aspect="equal")
    main.set_xlabel("Study-centered measured value", fontsize=5.7)
    main.set_ylabel("Study-centered predicted value", fontsize=5.7)
    main.tick_params(labelsize=5.0, width=0.6, length=2.5)
    main.text(
        0.96, 0.06,
        f"Pearson = {record.study_centered_pearson:.3f}\nSpearman = {record.study_centered_spearman:.3f}\nn = {int(record.patients_observed):,}",
        transform=main.transAxes, ha="right", va="bottom", fontsize=4.9, color="#4A4A4A",
    )
    top.set_title(str(record.phosphosite).replace("|", " "), fontsize=7.0, pad=2)
    top.set(xlim=(low, high), xticks=[], yticks=[])
    right.set(ylim=(low, high), xticks=[], yticks=[])
    for axis in (top, right):
        for spine in axis.spines.values():
            spine.set_visible(False)
    main.spines[["top", "right"]].set_visible(False)


def main() -> int:
    for name in ("tables", "predictions", "figures", "reports", "logs"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    rna_frame = pd.read_parquet(INPUT / "rna_reference_quantile.parquet")
    protein_frame = pd.read_parquet(INPUT / "protein_prediction.parquet").reindex(rna_frame.index)
    truth_frame = pd.read_parquet(INPUT / "phosphosite_logscale_aligned.parquet").reindex(rna_frame.index)
    split = pd.read_csv(INPUT / "split_manifest.tsv", sep="\t").set_index("sample_id").reindex(rna_frame.index)
    metadata = pd.read_csv(INPUT / "sample_metadata.tsv", sep="\t").set_index("sample_id").reindex(rna_frame.index)
    training = np.flatnonzero(split.role.eq("selection_train").to_numpy())
    development = np.flatnonzero(split.role.eq("selection_validation").to_numpy())
    truth = truth_frame.to_numpy(np.float32)
    protein = protein_frame.to_numpy(np.float32)
    rna = rna_frame.to_numpy(np.float32)
    studies = metadata.study.astype(str).to_numpy()
    manifest = pd.read_csv(MANIFEST, sep="\t")
    protein_lookup = {str(gene).upper(): index for index, gene in enumerate(protein_frame.columns.astype(str))}
    parent_index = np.asarray([protein_lookup[str(gene).upper()] for gene in manifest.total_protein_gene], dtype=np.int64)
    fixed, beta = fit_fixed_offset(truth, protein, studies, training, parent_index)
    stored_beta = pd.read_csv(M1 / "parent_beta.tsv", sep="\t").beta.to_numpy(np.float32)
    beta_error = float(np.max(np.abs(beta - stored_beta)))

    package = np.load(LATENT / "input_projection_and_ridge.npz")
    feature = transform_inputs(rna, protein, package)
    coordinate = np.load(LATENT / "train_coordinates_rank128.npy").astype(np.float32)
    basis = np.load(LATENT / "site_basis_rank128.npy").astype(np.float32)
    site_mean = np.load(LATENT / "site_mean.npy").astype(np.float32)
    coordinate_oof, coordinate_dev = fit_m22_coordinates(
        feature[training], feature[development], rna[training], rna[development], protein[training], protein[development], coordinate
    )
    prediction_oof = fixed[training] + coordinate_oof @ basis + site_mean[None, :]
    prediction_dev = fixed[development] + coordinate_dev @ basis + site_mean[None, :]

    rows = metric_pair(truth[training], prediction_oof, studies[training], "M2.2", "train_1796_OOF")
    rows += metric_pair(truth[development], prediction_dev, studies[development], "M2.2", "development_770")
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "tables/m22_primary_corrected_metrics.tsv", sep="\t", index=False)
    train_sites = per_site_table(truth[training], prediction_oof, studies[training], truth_frame.columns.astype(str).to_numpy())
    dev_sites = per_site_table(truth[development], prediction_dev, studies[development], truth_frame.columns.astype(str).to_numpy())
    train_sites.to_csv(OUT / "tables/m22_train1796_study_centered_per_site.tsv", sep="\t", index=False)
    dev_sites.to_csv(OUT / "tables/m22_dev770_study_centered_per_site.tsv", sep="\t", index=False)
    np.savez_compressed(
        OUT / "predictions/m22_primary_train_oof_dev_predictions.npz",
        train_ids=rna_frame.index[training].astype(str).to_numpy(),
        dev_ids=rna_frame.index[development].astype(str).to_numpy(),
        site_ids=truth_frame.columns.astype(str).to_numpy(),
        train_oof=prediction_oof.astype(np.float32),
        dev=prediction_dev.astype(np.float32),
    )

    top = train_sites[train_sites.patients_observed >= MINIMUM_PLOT_PATIENTS].head(4).copy()
    top.to_csv(OUT / "tables/m22_train1796_study_centered_top4.tsv", sep="\t", index=False)
    y_center, p_center = center_pair_within_study(truth[training], prediction_oof, studies[training])
    site_lookup = {site: column for column, site in enumerate(truth_frame.columns.astype(str))}
    source_rows = []
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})
    figure = plt.figure(figsize=(7.09, 1.80), layout="constrained")
    subfigures = figure.subfigures(1, 4, wspace=0.065)
    for subfigure, (_, record) in zip(np.ravel(subfigures), top.iterrows()):
        column = site_lookup[str(record.phosphosite)]
        observed = np.isfinite(y_center[:, column]) & np.isfinite(p_center[:, column])
        draw_panel(subfigure, y_center[observed, column], p_center[observed, column], record)
        local_ids = rna_frame.index[training][observed]
        source_rows.extend(
            {"sample_id": sample_id, "study": study, "phosphosite": record.phosphosite, "measured_centered": measured, "predicted_centered": predicted}
            for sample_id, study, measured, predicted in zip(local_ids, studies[training][observed], y_center[observed, column], p_center[observed, column])
        )
    figure.text(0.006, 0.986, "a", fontsize=10, fontweight="bold", va="top")
    figure.savefig(OUT / "figures/m22_train1796_study_centered_top4.pdf", bbox_inches="tight")
    figure.savefig(OUT / "figures/m22_train1796_study_centered_top4.png", dpi=600, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(source_rows).to_csv(OUT / "tables/m22_train1796_study_centered_top4_source.tsv", sep="\t", index=False)

    report = {
        "status": "complete",
        "training_patients": int(len(training)),
        "development_patients": int(len(development)),
        "parent_beta_max_abs_error_vs_locked": beta_error,
        "selected_latents_one_based": (SELECTIVE + 1).tolist(),
        "metric_definition": "truth and prediction centered separately on the same pairwise-observed patients within each study and phosphosite",
        "results": summary.to_dict("records"),
    }
    (OUT / "reports/primary_run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(top.to_string(index=False), flush=True)
    print(f"parent beta max abs error: {beta_error:.8g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
