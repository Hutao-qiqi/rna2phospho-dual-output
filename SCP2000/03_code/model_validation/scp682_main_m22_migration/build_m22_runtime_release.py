#!/usr/bin/env python3
"""Build the deployable SCP682 M2.2 runtime bundle from the locked training assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


SELECTIVE = np.asarray([15, 18, 23, 30, 41, 48, 55, 58], dtype=np.int64)


def transform(rna: np.ndarray, protein: np.ndarray, package: np.lib.npyio.NpzFile) -> np.ndarray:
    rz = np.nan_to_num((rna - package["rna_mean"]) / package["rna_scale"], nan=0.0).astype(np.float32)
    pz = np.nan_to_num((protein - package["protein_mean"]) / package["protein_scale"], nan=0.0).astype(np.float32)
    x = np.concatenate([rz @ package["rna_components"].T, pz @ package["protein_components"].T], axis=1)
    return ((x - package["feature_mean"]) / package["feature_scale"]).astype(np.float32)


def standardization(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0).astype(np.float32)
    scale = np.nanstd(x, axis=0).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    return np.nan_to_num((x - mean) / scale, nan=0.0).astype(np.float32), mean, scale


def copy_if_present(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    release = args.release.resolve()
    if root not in release.parents:
        raise ValueError("release must be inside the SCP682 project root")

    source = root / "01_data/bulk/intermediate/random70_reference_rna_protein_logscale_phosphosite_inputs_20260815"
    latent = root / "01_data/multi_omics/intermediate/scp682_v2_m2_residual_latent_20260823/models"
    m1 = root / "01_data/multi_omics/intermediate/scp682_v2_m1_parent_ptm_decomposition_20260822"
    target_manifest = root / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2/residual_target_manifest.tsv"
    required = [
        source / "rna_reference_quantile.parquet",
        source / "protein_prediction.parquet",
        source / "phosphosite_logscale_aligned.parquet",
        source / "split_manifest.tsv",
        latent / "input_projection_and_ridge.npz",
        latent / "train_coordinates_rank128.npy",
        latent / "site_basis_rank128.npy",
        latent / "site_mean.npy",
        m1 / "parent_beta.tsv",
        target_manifest,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing locked M2.2 assets:\n" + "\n".join(missing))

    rna_frame = pd.read_parquet(source / "rna_reference_quantile.parquet")
    protein_frame = pd.read_parquet(source / "protein_prediction.parquet").reindex(rna_frame.index)
    truth_frame = pd.read_parquet(source / "phosphosite_logscale_aligned.parquet").reindex(rna_frame.index)
    split = pd.read_csv(source / "split_manifest.tsv", sep="\t").set_index("sample_id").reindex(rna_frame.index)
    train = split.role.eq("selection_train").to_numpy()
    if int(train.sum()) != 1796:
        raise ValueError(f"expected 1796 M2.2 training patients, found {int(train.sum())}")

    package = np.load(latent / "input_projection_and_ridge.npz")
    coordinate = np.load(latent / "train_coordinates_rank128.npy").astype(np.float32)
    feature = transform(
        rna_frame.to_numpy(np.float32)[train],
        protein_frame.to_numpy(np.float32)[train],
        package,
    )
    rna = rna_frame.to_numpy(np.float32)[train]
    protein = protein_frame.to_numpy(np.float32)[train]
    rna_z, rna_mean, rna_scale = standardization(rna)
    protein_z, protein_mean, protein_scale = standardization(protein)
    coordinate_mean = coordinate.mean(axis=0).astype(np.float32)
    coordinate_scale = np.where(coordinate.std(axis=0) > 1e-6, coordinate.std(axis=0), 1.0).astype(np.float32)
    target = ((coordinate - coordinate_mean) / coordinate_scale).astype(np.float32)

    rescue_rna_index = np.empty((len(SELECTIVE), 16), dtype=np.int64)
    rescue_protein_index = np.empty((len(SELECTIVE), 8), dtype=np.int64)
    rescue_coef = np.empty((len(SELECTIVE), feature.shape[1] + 24), dtype=np.float32)
    rescue_intercept = np.empty(len(SELECTIVE), dtype=np.float32)
    for row, latent_id in enumerate(SELECTIVE):
        rna_idx = np.argsort(-np.abs(target[:, latent_id] @ rna_z))[:16]
        protein_idx = np.argsort(-np.abs(target[:, latent_id] @ protein_z))[:8]
        x = np.concatenate([feature, rna_z[:, rna_idx], protein_z[:, protein_idx]], axis=1)
        model = Ridge(alpha=1000.0, solver="lsqr").fit(x, target[:, latent_id])
        rescue_rna_index[row] = rna_idx
        rescue_protein_index[row] = protein_idx
        rescue_coef[row] = model.coef_.astype(np.float32)
        rescue_intercept[row] = np.float32(model.intercept_)

    manifest = pd.read_csv(target_manifest, sep="\t")
    protein_lookup = {str(gene).upper(): index for index, gene in enumerate(protein_frame.columns.astype(str))}
    parent_index = np.asarray([protein_lookup[str(gene).upper()] for gene in manifest.total_protein_gene], dtype=np.int64)
    beta = pd.read_csv(m1 / "parent_beta.tsv", sep="\t").beta.to_numpy(np.float32)

    runtime = release / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        runtime / "scp682_m22_runtime.npz",
        rna_mean=package["rna_mean"],
        rna_scale=package["rna_scale"],
        rna_components=package["rna_components"],
        protein_mean=package["protein_mean"],
        protein_scale=package["protein_scale"],
        protein_components=package["protein_components"],
        feature_mean=package["feature_mean"],
        feature_scale=package["feature_scale"],
        ridge_coef=package["ridge_coef"],
        ridge_intercept=package["ridge_intercept"],
        coordinate_mean=coordinate_mean,
        coordinate_scale=coordinate_scale,
        rescue_latent=SELECTIVE,
        rescue_rna_index=rescue_rna_index,
        rescue_protein_index=rescue_protein_index,
        rescue_rna_mean=rna_mean,
        rescue_rna_scale=rna_scale,
        rescue_protein_mean=protein_mean,
        rescue_protein_scale=protein_scale,
        rescue_coef=rescue_coef,
        rescue_intercept=rescue_intercept,
        site_basis=np.load(latent / "site_basis_rank128.npy").astype(np.float32),
        site_mean=np.load(latent / "site_mean.npy").astype(np.float32),
        global_site_mean=np.nan_to_num(np.nanmean(truth_frame.to_numpy(np.float32)[train], axis=0), nan=0.0).astype(np.float32),
        global_protein_mean=np.nan_to_num(np.nanmean(protein, axis=0), nan=0.0).astype(np.float32),
        parent_index=parent_index,
        parent_beta=beta,
        rna_vocabulary=np.asarray(rna_frame.columns.astype(str).tolist(), dtype="U"),
        protein_vocabulary=np.asarray(protein_frame.columns.astype(str).tolist(), dtype="U"),
        phosphosite_vocabulary=np.asarray(truth_frame.columns.astype(str).tolist(), dtype="U"),
        model_id=np.asarray("SCP682-M2.2-LR-rank128"),
        training_patients=np.asarray(1796),
    )

    copy_if_present(
        root / "02_results/model_validation/scp682_v2_final_output_lock_20260831",
        release / "final_output_lock",
    )
    copy_if_present(
        root / "02_results/external_validation/scp682_v2_patient_pearson_output_shrinkage_20260831/predictions/shrunken_output_calibration_predictions.npz",
        release / "reference_predictions/default_patient_profile_predictions.npz",
    )
    copy_if_present(
        root / "02_results/external_validation/scp682_v2_m22_f_pt7_fusion_20260831_v2/predictions/fusion_selected_predictions.npz",
        release / "reference_predictions/base_site_predictions.npz",
    )
    release_manifest = {
        "status": "active",
        "model_id": "SCP682-M2.2-LR-rank128",
        "release_date": "2026-09-05",
        "runtime_bundle": "runtime/scp682_m22_runtime.npz",
        "training_patients": 1796,
        "phosphosites": int(truth_frame.shape[1]),
        "selective_rescue_latents_one_based": (SELECTIVE + 1).tolist(),
        "pt7": {"gamma_J": 0.1, "lambda": 1.0, "kappa": 100.0, "adapter_ridge": 10.0, "adapter_rank": 24},
        "default_patient_profile": {"stack_ridge": 30.0, "alpha": 0.1},
        "inference_requires": ["reference-quantile RNA", "RNA-predicted total protein"],
    }
    (release / "RELEASE.json").write_text(json.dumps(release_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(release_manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
