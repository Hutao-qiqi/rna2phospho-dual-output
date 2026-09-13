#!/usr/bin/env python3
"""在 M2.2 固定输入表示上训练 PhosphoSCT-S1 低秩残差模型。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.utils.extmath import randomized_svd


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "01_data/bulk/intermediate/random70_reference_rna_protein_logscale_phosphosite_inputs_20260815"
LATENT = ROOT / "01_data/multi_omics/intermediate/scp682_v2_m2_residual_latent_20260823/models"
MANIFEST = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2/residual_target_manifest.tsv"
BASELINE = ROOT / "02_results/model_validation/scp682_v2_m22_study_centered_evaluation_20260830/predictions/m22_primary_train_oof_dev_predictions.npz"
METRIC_SCRIPT = ROOT / "03_code/model_validation/scp682_v2_m22_study_centered_evaluation/02_rerun_external_pt7_centered.py"
ZERO_SCRIPT = ROOT / "03_code/model_validation/scp682_v2_m22_zero_aware/train_zero_aware.py"
OUT = ROOT / "02_results/model_validation/scp682_v2_m23_phosphosct_s1_20260831"
ALPHAS = (100.0, 1000.0, 10000.0)
RANK = 128
SHRINKAGE = 50.0
CLIP = 5.0
SEED = 20260831


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.nanmedian(values, axis=0).astype(np.float32)
    scale = (1.4826 * np.nanmedian(np.abs(values - center[None, :]), axis=0)).astype(np.float32)
    count = np.isfinite(values).sum(axis=0).astype(np.float32)
    center = np.nan_to_num(center, nan=0.0)
    return center, scale, count


def fit_phosphosct(residual: np.ndarray, studies: np.ndarray, training: np.ndarray, tau: np.ndarray):
    global_center, global_scale, _ = robust_center_scale(residual[training])
    finite_scale = global_scale[np.isfinite(global_scale) & (global_scale > 1e-6)]
    fallback_scale = float(np.median(finite_scale)) if finite_scale.size else 1.0
    global_scale = np.where(np.isfinite(global_scale) & (global_scale > 1e-6), global_scale, fallback_scale)
    global_scale = np.maximum(global_scale, tau).astype(np.float32)
    parameters = {}
    training_mask = np.zeros(len(residual), dtype=bool)
    training_mask[training] = True
    for study in np.unique(studies):
        rows = np.flatnonzero((studies == study) & training_mask)
        if len(rows) == 0:
            parameters[str(study)] = (global_center.copy(), global_scale.copy())
            continue
        local_center, local_scale, count = robust_center_scale(residual[rows])
        local_center = np.where(np.isfinite(local_center), local_center, global_center)
        local_scale = np.where(np.isfinite(local_scale) & (local_scale > 1e-6), local_scale, global_scale)
        weight = count / (count + SHRINKAGE)
        center = (weight * local_center + (1.0 - weight) * global_center).astype(np.float32)
        scale = np.sqrt(weight * np.square(local_scale) + (1.0 - weight) * np.square(global_scale))
        scale = np.maximum(scale, tau).astype(np.float32)
        parameters[str(study)] = (center, scale)
    return parameters, global_center.astype(np.float32), global_scale.astype(np.float32)


def transform_target(residual, studies, parameters, global_center, global_scale):
    output = np.full_like(residual, np.nan, dtype=np.float32)
    for study in np.unique(studies):
        rows = np.flatnonzero(studies == study)
        center, scale = parameters.get(str(study), (global_center, global_scale))
        output[rows] = np.clip((residual[rows] - center[None, :]) / scale[None, :], -CLIP, CLIP)
    return output


def inverse_target(prediction, studies, parameters, global_center, global_scale):
    output = np.empty_like(prediction, dtype=np.float32)
    for study in np.unique(studies):
        rows = np.flatnonzero(studies == study)
        center, scale = parameters.get(str(study), (global_center, global_scale))
        output[rows] = center[None, :] + scale[None, :] * prediction[rows]
    return output


def pearson_selection(module, truth, prediction, studies):
    y, p = module.center_pair_within_study(truth, prediction, studies)
    patient = module.observed_axis_pearson(y, p, axis=1)
    site = module.observed_axis_pearson(y, p, axis=0)
    observed = np.isfinite(y) & np.isfinite(p)
    mse = np.nanmean(np.where(observed, (y - p) ** 2, np.nan), axis=1)
    return float(np.nanmedian(patient)), float(np.nanmedian(site)), float(np.nanmean(mse))


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    for name in ("tables", "models", "predictions", "reports", "logs"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    metrics = load_module(METRIC_SCRIPT, "centered_metrics")
    zero = load_module(ZERO_SCRIPT, "zero_training")

    rna = pd.read_parquet(INPUT / "rna_reference_quantile.parquet")
    protein_frame = pd.read_parquet(INPUT / "protein_prediction.parquet").reindex(rna.index)
    truth_frame = pd.read_parquet(INPUT / "phosphosite_logscale_aligned.parquet").reindex(rna.index)
    split = pd.read_csv(INPUT / "split_manifest.tsv", sep="\t").set_index("sample_id").reindex(rna.index)
    metadata = pd.read_csv(INPUT / "sample_metadata.tsv", sep="\t").set_index("sample_id").reindex(rna.index)
    training = np.flatnonzero(split.role.eq("selection_train"))
    development = np.flatnonzero(split.role.eq("selection_validation"))
    studies = metadata.study.astype(str).to_numpy()
    truth = truth_frame.to_numpy(np.float32)
    protein = protein_frame.to_numpy(np.float32)

    manifest = pd.read_csv(MANIFEST, sep="\t")
    lookup = {gene.upper(): i for i, gene in enumerate(protein_frame.columns.astype(str))}
    parent_index = np.asarray([lookup[str(gene).upper()] for gene in manifest.total_protein_gene], dtype=np.int64)
    fixed = zero.fit_fixed_offset(truth, protein, studies, training, parent_index)
    residual = truth - fixed
    _, tau = zero.empirical_tau(residual[training])
    parameters, global_center, global_scale = fit_phosphosct(residual, studies, training, tau)
    target = transform_target(residual, studies, parameters, global_center, global_scale)

    package = np.load(LATENT / "input_projection_and_ridge.npz")
    feature = zero.transform_inputs(rna.to_numpy(np.float32), protein, package)
    train_target = target[training]
    site_mean = np.nan_to_num(np.nanmean(train_target, axis=0), nan=0.0).astype(np.float32)
    filled = np.where(np.isfinite(train_target), train_target - site_mean[None, :], 0.0).astype(np.float32)
    print("fitting PhosphoSCT rank-128 basis", flush=True)
    left, singular, basis = randomized_svd(
        filled, n_components=RANK, n_iter=5, random_state=SEED
    )
    coordinate = (left * singular[None, :]).astype(np.float32)
    basis = basis.astype(np.float32)
    folds = KFold(n_splits=5, shuffle=True, random_state=20260823)

    candidates = []
    predictions = {}
    for alpha in ALPHAS:
        oof_standard = np.empty_like(train_target, dtype=np.float32)
        for fold, (fit_local, held_local) in enumerate(folds.split(training)):
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(feature[training[fit_local]], coordinate[fit_local])
            z = model.predict(feature[training[held_local]]).astype(np.float32)
            oof_standard[held_local] = z @ basis + site_mean[None, :]
            print(f"alpha={alpha:g} fold={fold + 1}/5", flush=True)
        oof_residual = inverse_target(
            oof_standard, studies[training], parameters, global_center, global_scale
        )
        oof_total = fixed[training] + oof_residual
        patient_p, site_p, mse = pearson_selection(
            metrics, truth[training], oof_total, studies[training]
        )
        candidates.append(
            {
                "ridge_alpha": alpha,
                "patient_pearson": patient_p,
                "site_pearson": site_p,
                "patient_equal_mse": mse,
                "selection_score": 0.5 * (patient_p + site_p),
            }
        )
        predictions[alpha] = oof_total.astype(np.float32)
    candidate_table = pd.DataFrame(candidates).sort_values(
        ["selection_score", "patient_pearson", "site_pearson"], ascending=False
    )
    selected_alpha = float(candidate_table.iloc[0].ridge_alpha)
    candidate_table["selected"] = candidate_table.ridge_alpha.eq(selected_alpha)
    candidate_table.to_csv(OUT / "tables/ridge_selection.tsv", sep="\t", index=False)
    train_oof = predictions[selected_alpha]

    full_model = Ridge(alpha=selected_alpha, fit_intercept=True)
    full_model.fit(feature[training], coordinate)
    dev_coordinate = full_model.predict(feature[development]).astype(np.float32)
    dev_standard = dev_coordinate @ basis + site_mean[None, :]
    dev_residual = inverse_target(
        dev_standard, studies[development], parameters, global_center, global_scale
    )
    dev_prediction = fixed[development] + dev_residual

    baseline = np.load(BASELINE, allow_pickle=True)
    if not np.array_equal(baseline["train_ids"].astype(str), rna.index[training].astype(str)):
        raise ValueError("baseline training order differs")
    if not np.array_equal(baseline["dev_ids"].astype(str), rna.index[development].astype(str)):
        raise ValueError("baseline development order differs")
    rows = []
    for model_name, scope, local_truth, prediction, local_studies in (
        ("M2.2", "train_1796_OOF", truth[training], baseline["train_oof"], studies[training]),
        ("PhosphoSCT_S1", "train_1796_OOF", truth[training], train_oof, studies[training]),
        ("M2.2", "development_770", truth[development], baseline["dev"], studies[development]),
        ("PhosphoSCT_S1", "development_770", truth[development], dev_prediction, studies[development]),
    ):
        rows.extend(metrics.metric_pair(local_truth, prediction, local_studies, model_name, scope))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "tables/phosphosct_s1_metrics.tsv", sep="\t", index=False)
    np.savez_compressed(
        OUT / "predictions/phosphosct_s1_predictions.npz",
        train_ids=rna.index[training].astype(str).to_numpy(),
        dev_ids=rna.index[development].astype(str).to_numpy(),
        site_ids=truth_frame.columns.astype(str).to_numpy(),
        train_oof=train_oof,
        dev=dev_prediction.astype(np.float32),
    )
    study_names = np.asarray(sorted(parameters), dtype=str)
    np.savez_compressed(
        OUT / "models/phosphosct_s1_model.npz",
        basis=basis,
        site_mean=site_mean,
        singular_values=singular.astype(np.float32),
        ridge_coef=full_model.coef_.astype(np.float32),
        ridge_intercept=full_model.intercept_.astype(np.float32),
        selected_alpha=np.asarray(selected_alpha),
        tau=tau,
        global_center=global_center,
        global_scale=global_scale,
        study_names=study_names,
        study_center=np.stack([parameters[name][0] for name in study_names]),
        study_scale=np.stack([parameters[name][1] for name in study_names]),
    )

    centered = result.query("coordinate == 'study_centered'")
    base_dev = centered.query("model == 'M2.2' and scope == 'development_770'").iloc[0]
    sct_dev = centered.query("model == 'PhosphoSCT_S1' and scope == 'development_770'").iloc[0]
    report = "\n".join(
        [
            "# SCP682 PhosphoSCT-S1",
            "",
            f"选择 ridge alpha = `{selected_alpha:g}`。",
            "",
            "| 模型 | 患者 Pearson | 患者 Spearman | 位点 Pearson | 位点 Spearman | MSE |",
            "|---|---:|---:|---:|---:|---:|",
            f"| M2.2 开发770 | {base_dev.patient_pearson_median:.6f} | {base_dev.patient_spearman_median:.6f} | {base_dev.site_pearson_median:.6f} | {base_dev.site_spearman_median:.6f} | {base_dev.patient_equal_mse:.6f} |",
            f"| PhosphoSCT-S1 开发770 | {sct_dev.patient_pearson_median:.6f} | {sct_dev.patient_spearman_median:.6f} | {sct_dev.site_pearson_median:.6f} | {sct_dev.site_spearman_median:.6f} | {sct_dev.patient_equal_mse:.6f} |",
            "",
        ]
    )
    (OUT / "RESULTS_20260831.md").write_text(report, encoding="utf-8")
    (OUT / "reports/run_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "rank": RANK,
                "selected_alpha": selected_alpha,
                "shrinkage": SHRINKAGE,
                "clip": CLIP,
                "development_centered": sct_dev.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (OUT / "SUCCESS").touch()
    print(candidate_table.to_string(index=False), flush=True)
    print(centered.to_string(index=False), flush=True)
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
