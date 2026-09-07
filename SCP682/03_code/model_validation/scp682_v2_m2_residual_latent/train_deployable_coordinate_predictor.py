#!/usr/bin/env python3
"""Predict train-derived PTM residual coordinates from deployable inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.utils.extmath import randomized_svd


def row_spearman(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = np.full(len(truth), np.nan)
    for row in range(len(truth)):
        mask = np.isfinite(truth[row]) & np.isfinite(prediction[row])
        if mask.sum() < 2:
            continue
        x = rankdata(truth[row, mask])
        y = rankdata(prediction[row, mask])
        result[row] = np.corrcoef(x, y)[0, 1]
    return result


def fit_input_basis(
    train: np.ndarray,
    validation: np.ndarray,
    components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    mean = np.nanmean(train, axis=0).astype(np.float32)
    scale = np.nanstd(train, axis=0).astype(np.float32)
    scale = np.where(scale > 1.0e-6, scale, 1.0).astype(np.float32)
    train_z = np.nan_to_num((train - mean) / scale, nan=0.0).astype(np.float32)
    validation_z = np.nan_to_num((validation - mean) / scale, nan=0.0).astype(np.float32)
    left, singular, right = randomized_svd(
        train_z,
        n_components=min(components, train_z.shape[0] - 1, train_z.shape[1]),
        n_iter=4,
        random_state=seed,
    )
    train_score = (left * singular[None, :]).astype(np.float32)
    validation_score = (validation_z @ right.T).astype(np.float32)
    return train_score, validation_score, {
        "mean": mean,
        "scale": scale,
        "components": right.astype(np.float32),
        "singular_values": singular.astype(np.float32),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--protein", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--fixed-offset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--latent-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument("--ridge-alphas", default="1,10,100,1000")
    parser.add_argument("--rna-components", type=int, default=256)
    parser.add_argument("--protein-components", type=int, default=128)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    ranks = tuple(sorted({int(value) for value in args.ranks.split(",")}))
    alphas = tuple(float(value) for value in args.ridge_alphas.split(","))
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for name in ("tables", "reports", "models", "predictions"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    rna = pd.read_parquet(args.rna)
    protein = pd.read_parquet(args.protein).reindex(rna.index)
    truth = pd.read_parquet(args.truth).reindex(rna.index)
    offset = pd.read_parquet(args.fixed_offset).reindex_like(truth)
    split = pd.read_csv(args.split, sep="\t").set_index("sample_id").reindex(rna.index)
    train_mask = split["role"].eq("selection_train").to_numpy()
    validation_mask = split["role"].eq("selection_validation").to_numpy()
    train_ids = rna.index[train_mask]
    validation_ids = rna.index[validation_mask]
    package_ids = pd.read_csv(args.latent_package / "metadata/train_ids.tsv", sep="\t")["sample_id"].astype(str)
    if not np.array_equal(package_ids, train_ids.astype(str)):
        raise ValueError("latent coordinates differ from current training patient order")

    latent = np.load(args.latent_package / "models/train_coordinates_rank128.npy")
    basis = np.load(args.latent_package / "models/site_basis_rank128.npy")
    site_mean = np.load(args.latent_package / "models/site_mean.npy")
    if latent.shape != (len(train_ids), 128) or basis.shape != (128, truth.shape[1]):
        raise ValueError("latent package dimensions differ from the dataset")

    rna_train = rna.loc[train_ids].to_numpy(np.float32)
    rna_validation = rna.loc[validation_ids].to_numpy(np.float32)
    protein_train = protein.loc[train_ids].to_numpy(np.float32)
    protein_validation = protein.loc[validation_ids].to_numpy(np.float32)
    rna_score, rna_validation_score, rna_model = fit_input_basis(
        rna_train, rna_validation, args.rna_components, args.seed
    )
    protein_score, protein_validation_score, protein_model = fit_input_basis(
        protein_train, protein_validation, args.protein_components, args.seed + 1
    )
    feature = np.concatenate([rna_score, protein_score], axis=1)
    validation_feature = np.concatenate(
        [rna_validation_score, protein_validation_score], axis=1
    )
    feature_mean = feature.mean(axis=0).astype(np.float32)
    feature_scale = feature.std(axis=0).astype(np.float32)
    feature_scale = np.where(feature_scale > 1.0e-6, feature_scale, 1.0)
    feature = ((feature - feature_mean) / feature_scale).astype(np.float32)
    validation_feature = ((validation_feature - feature_mean) / feature_scale).astype(np.float32)

    train_truth = truth.loc[train_ids].to_numpy(np.float32)
    train_offset = offset.loc[train_ids].to_numpy(np.float32)
    folds = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows: list[dict[str, float | int]] = []
    best: dict[str, float | int] | None = None
    for alpha in alphas:
        oof_coordinate = np.zeros_like(latent, dtype=np.float32)
        for fit_index, test_index in folds.split(feature):
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(feature[fit_index], latent[fit_index])
            oof_coordinate[test_index] = model.predict(feature[test_index]).astype(np.float32)
        for rank in ranks:
            residual_prediction = (
                oof_coordinate[:, :rank] @ basis[:rank] + site_mean[None, :]
            )
            prediction = train_offset + residual_prediction
            score = row_spearman(train_truth, prediction)
            row = {
                "rank": rank,
                "ridge_alpha": alpha,
                "oof_profile_spearman_median": float(np.nanmedian(score)),
                "oof_profile_spearman_mean": float(np.nanmean(score)),
            }
            rows.append(row)
            if best is None or row["oof_profile_spearman_median"] > best["oof_profile_spearman_median"]:
                best = row
    if best is None:
        raise RuntimeError("no coordinate predictor candidate was evaluated")
    table = pd.DataFrame(rows).sort_values(
        ["oof_profile_spearman_median", "rank"], ascending=[False, True]
    )
    table.to_csv(args.output_dir / "tables/oof_model_selection.tsv", sep="\t", index=False)

    selected_rank = int(best["rank"])
    selected_alpha = float(best["ridge_alpha"])
    coordinate_model = Ridge(alpha=selected_alpha, fit_intercept=True)
    coordinate_model.fit(feature, latent)
    validation_coordinate = coordinate_model.predict(validation_feature).astype(np.float32)
    validation_residual = (
        validation_coordinate[:, :selected_rank] @ basis[:selected_rank]
        + site_mean[None, :]
    )
    validation_offset = offset.loc[validation_ids].to_numpy(np.float32)
    validation_prediction = validation_offset + validation_residual
    validation_truth = truth.loc[validation_ids].to_numpy(np.float32)
    validation_score = row_spearman(validation_truth, validation_prediction)
    fixed_score = row_spearman(validation_truth, validation_offset)
    pd.DataFrame(
        validation_prediction,
        index=validation_ids,
        columns=truth.columns,
    ).to_parquet(args.output_dir / "predictions/validation_prediction.parquet")
    pd.DataFrame(
        {
            "sample_id": validation_ids,
            "fixed_offset_spearman": fixed_score,
            "latent_model_spearman": validation_score,
            "gain": validation_score - fixed_score,
        }
    ).to_csv(args.output_dir / "tables/validation_patient_metrics.tsv", sep="\t", index=False)

    np.savez_compressed(
        args.output_dir / "models/input_projection_and_ridge.npz",
        rna_mean=rna_model["mean"],
        rna_scale=rna_model["scale"],
        rna_components=rna_model["components"],
        protein_mean=protein_model["mean"],
        protein_scale=protein_model["scale"],
        protein_components=protein_model["components"],
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        ridge_coef=coordinate_model.coef_.astype(np.float32),
        ridge_intercept=coordinate_model.intercept_.astype(np.float32),
        selected_rank=np.asarray(selected_rank),
        selected_alpha=np.asarray(selected_alpha),
    )
    report = {
        "status": "complete",
        "selection_scope": "selection_train_oof",
        "selected_rank": selected_rank,
        "selected_ridge_alpha": selected_alpha,
        "selected_oof_profile_spearman_median": float(best["oof_profile_spearman_median"]),
        "fixed_offset_validation_spearman_median": float(np.nanmedian(fixed_score)),
        "deployable_latent_validation_spearman_median": float(np.nanmedian(validation_score)),
        "validation_gain": float(np.nanmedian(validation_score) - np.nanmedian(fixed_score)),
        "patients_improved": int(np.sum(validation_score > fixed_score)),
        "validation_phosphosite_used_for_selection": False,
    }
    (args.output_dir / "reports/run_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(table.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
