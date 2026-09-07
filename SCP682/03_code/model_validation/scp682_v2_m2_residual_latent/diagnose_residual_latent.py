#!/usr/bin/env python3
"""Train-only low-rank analysis of M1 PTM-specific residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
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


def masked_mse(truth: np.ndarray, prediction: np.ndarray) -> float:
    mask = np.isfinite(truth) & np.isfinite(prediction)
    return float(np.mean(np.square(truth[mask] - prediction[mask])))


def oracle_coordinates(
    residual: np.ndarray,
    site_mean: np.ndarray,
    basis: np.ndarray,
    ridge: float,
) -> np.ndarray:
    coordinates = np.zeros((len(residual), basis.shape[0]), dtype=np.float32)
    eye = np.eye(basis.shape[0], dtype=np.float64)
    for row in range(len(residual)):
        observed = np.isfinite(residual[row])
        if observed.sum() < basis.shape[0]:
            continue
        local_basis = basis[:, observed].astype(np.float64)
        target = (residual[row, observed] - site_mean[observed]).astype(np.float64)
        gram = local_basis @ local_basis.T + ridge * eye
        coordinates[row] = np.linalg.solve(gram, local_basis @ target).astype(np.float32)
    return coordinates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--fixed-offset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument("--svd-iterations", type=int, default=5)
    parser.add_argument("--oracle-ridge", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    ranks = tuple(sorted({int(value) for value in args.ranks.split(",")}))
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    (args.output_dir / "tables").mkdir(parents=True)
    (args.output_dir / "reports").mkdir()
    (args.output_dir / "models").mkdir()

    residual_frame = pd.read_parquet(args.residual)
    truth_frame = pd.read_parquet(args.truth).reindex_like(residual_frame)
    fixed_frame = pd.read_parquet(args.fixed_offset).reindex_like(residual_frame)
    split = pd.read_csv(args.split, sep="\t").set_index("sample_id").reindex(residual_frame.index)
    train_mask = split["role"].eq("selection_train").to_numpy()
    validation_mask = split["role"].eq("selection_validation").to_numpy()
    train_ids = residual_frame.index[train_mask]
    validation_ids = residual_frame.index[validation_mask]
    train = residual_frame.loc[train_ids].to_numpy(np.float32)
    validation = residual_frame.loc[validation_ids].to_numpy(np.float32)
    validation_truth = truth_frame.loc[validation_ids].to_numpy(np.float32)
    validation_fixed = fixed_frame.loc[validation_ids].to_numpy(np.float32)

    site_mean = np.nan_to_num(np.nanmean(train, axis=0), nan=0.0).astype(np.float32)
    train_observed = np.isfinite(train)
    train_filled = np.where(train_observed, train - site_mean[None, :], 0.0).astype(np.float32)
    maximum_rank = max(ranks)
    left, singular, right = randomized_svd(
        train_filled,
        n_components=maximum_rank,
        n_iter=args.svd_iterations,
        random_state=args.seed,
    )
    coordinates = (left * singular[None, :]).astype(np.float32)
    right = right.astype(np.float32)
    total_energy = float(np.square(train_filled).sum())
    cumulative = np.cumsum(np.square(singular)) / max(total_energy, 1.0e-12)

    np.save(args.output_dir / "models/site_mean.npy", site_mean)
    np.save(args.output_dir / "models/site_basis_rank128.npy", right)
    np.save(args.output_dir / "models/train_coordinates_rank128.npy", coordinates)
    pd.DataFrame({"sample_id": train_ids}).to_csv(
        args.output_dir / "tables/train_ids.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {
            "component": np.arange(1, maximum_rank + 1),
            "singular_value": singular,
            "cumulative_zero_filled_energy": cumulative,
        }
    ).to_csv(args.output_dir / "tables/singular_values.tsv", sep="\t", index=False)

    baseline_spearman = row_spearman(validation_truth, validation_fixed)
    rows: list[dict[str, float | int | str]] = [
        {
            "candidate": "fixed_offset",
            "rank": 0,
            "validation_profile_spearman_median": float(np.nanmedian(baseline_spearman)),
            "validation_profile_spearman_mean": float(np.nanmean(baseline_spearman)),
            "validation_residual_mse": masked_mse(validation, np.zeros_like(validation)),
            "cumulative_zero_filled_energy": 0.0,
            "coordinate_source": "none",
        }
    ]
    for rank in ranks:
        local_basis = right[:rank]
        train_reconstruction = coordinates[:, :rank] @ local_basis + site_mean[None, :]
        train_mse = masked_mse(train, train_reconstruction)
        validation_coordinate = oracle_coordinates(
            validation,
            site_mean,
            local_basis,
            args.oracle_ridge,
        )
        validation_reconstruction = validation_coordinate @ local_basis + site_mean[None, :]
        reconstructed_truth = validation_fixed + validation_reconstruction
        profile = row_spearman(validation_truth, reconstructed_truth)
        rows.append(
            {
                "candidate": f"oracle_rank{rank}",
                "rank": rank,
                "validation_profile_spearman_median": float(np.nanmedian(profile)),
                "validation_profile_spearman_mean": float(np.nanmean(profile)),
                "validation_residual_mse": masked_mse(validation, validation_reconstruction),
                "train_residual_mse": train_mse,
                "cumulative_zero_filled_energy": float(cumulative[rank - 1]),
                "coordinate_source": "validation_residual_oracle",
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "tables/latent_rank_diagnostic.tsv", sep="\t", index=False)
    report = {
        "status": "complete",
        "fit_scope": "selection_train_only",
        "training_samples": int(len(train_ids)),
        "validation_samples": int(len(validation_ids)),
        "sites": int(train.shape[1]),
        "maximum_rank": maximum_rank,
        "fixed_offset_profile_spearman_median": float(np.nanmedian(baseline_spearman)),
        "best_oracle": table.sort_values(
            "validation_profile_spearman_median", ascending=False
        ).iloc[0].to_dict(),
        "oracle_warning": "validation phosphosite residuals estimate coordinates; ceiling only",
    }
    (args.output_dir / "reports/diagnostic_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
