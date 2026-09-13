#!/usr/bin/env python3
"""Fit train-only study and parent-protein components for SCP682-v2 M1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phosphosite", type=Path, required=True)
    parser.add_argument("--protein", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--study-tau", type=float, default=2.0)
    parser.add_argument("--study-min-overlap", type=int, default=5)
    parser.add_argument("--parent-slope-tau", type=float, default=20.0)
    parser.add_argument("--parent-min-observations", type=int, default=16)
    parser.add_argument("--site-chunk-size", type=int, default=1000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_study_site_center(
    values: np.ndarray,
    observed: np.ndarray,
    studies: np.ndarray,
    train_index: np.ndarray,
    *,
    tau: float,
    minimum_overlap: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    observed_values = np.where(observed, values, np.nan)
    train_values = observed_values[train_index]
    global_mean = np.nan_to_num(np.nanmean(train_values, axis=0), nan=0.0)
    global_count = np.isfinite(train_values).sum(axis=0).astype(np.float32)
    center = np.zeros_like(values, dtype=np.float32)
    rows: list[dict[str, float | int | str]] = []
    train_mask = np.zeros(len(values), dtype=bool)
    train_mask[train_index] = True
    for study in np.unique(studies):
        study_rows = np.flatnonzero(studies == study)
        study_train = study_rows[train_mask[study_rows]]
        if not len(study_train):
            center[study_rows] = global_mean
            rows.append({"study": str(study), "training_samples": 0, "offset": 0.0})
            continue
        study_values = observed_values[study_train]
        site_mean = np.nan_to_num(np.nanmean(study_values, axis=0), nan=0.0)
        site_count = np.isfinite(study_values).sum(axis=0).astype(np.float32)
        common = (site_count >= minimum_overlap) & (global_count >= minimum_overlap)
        offset = float(np.median(site_mean[common] - global_mean[common])) if common.any() else 0.0
        target = global_mean + float(np.nan_to_num(offset, nan=0.0))
        weight = site_count / (site_count + tau) if tau > 0 else (site_count > 0)
        shrunk = weight * site_mean + (1.0 - weight) * target
        center[study_rows] = shrunk.astype(np.float32)
        rows.append(
            {
                "study": str(study),
                "training_samples": int(len(study_train)),
                "offset": offset,
            }
        )
    return center, pd.DataFrame(rows)


def fit_study_protein_center(
    protein: np.ndarray,
    studies: np.ndarray,
    train_index: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    global_mean = np.nanmean(protein[train_index], axis=0)
    global_mean = np.nan_to_num(global_mean, nan=0.0).astype(np.float32)
    center = np.zeros_like(protein, dtype=np.float32)
    train_mask = np.zeros(len(protein), dtype=bool)
    train_mask[train_index] = True
    rows: list[dict[str, float | int | str]] = []
    for study in np.unique(studies):
        study_rows = np.flatnonzero(studies == study)
        study_train = study_rows[train_mask[study_rows]]
        value = (
            np.nan_to_num(np.nanmean(protein[study_train], axis=0), nan=global_mean)
            if len(study_train)
            else global_mean
        )
        center[study_rows] = value.astype(np.float32)
        rows.append({"study": str(study), "training_samples": int(len(study_train))})
    return center, pd.DataFrame(rows)


def rowwise_spearman(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = np.full(len(truth), np.nan, dtype=np.float64)
    for row in range(len(truth)):
        mask = np.isfinite(truth[row]) & np.isfinite(prediction[row])
        if mask.sum() < 2:
            continue
        x = rankdata(truth[row, mask], method="average")
        y = rankdata(prediction[row, mask], method="average")
        if np.std(x) > 0 and np.std(y) > 0:
            result[row] = np.corrcoef(x, y)[0, 1]
    return result


def masked_mse(truth: np.ndarray, prediction: np.ndarray, rows: np.ndarray) -> float:
    x = truth[rows]
    y = prediction[rows]
    mask = np.isfinite(x) & np.isfinite(y)
    return float(np.mean(np.square(x[mask] - y[mask])))


def main() -> int:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    truth_frame = pd.read_parquet(args.phosphosite)
    protein_frame = pd.read_parquet(args.protein).reindex(truth_frame.index)
    manifest = pd.read_csv(args.manifest, sep="\t")
    split = pd.read_csv(args.split, sep="\t").set_index("sample_id")
    metadata = pd.read_csv(args.metadata, sep="\t").set_index("sample_id")
    if protein_frame.isna().all(axis=1).any():
        raise ValueError("protein matrix is missing complete phosphosite patients")
    if not truth_frame.index.is_unique or not truth_frame.columns.is_unique:
        raise ValueError("phosphosite axes must be unique")
    targets = truth_frame.columns.astype(str)
    if not np.array_equal(manifest["gene_site_id"].astype(str), targets):
        raise ValueError("target manifest differs from phosphosite columns")
    split = split.reindex(truth_frame.index)
    metadata = metadata.reindex(truth_frame.index)
    if split["role"].isna().any() or metadata["study"].isna().any():
        raise ValueError("split or study metadata is incomplete")

    truth = truth_frame.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    protein = protein_frame.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    observed = np.isfinite(truth)
    train_index = np.flatnonzero(split["role"].eq("selection_train").to_numpy())
    validation_index = np.flatnonzero(split["role"].eq("selection_validation").to_numpy())
    studies = metadata["study"].astype(str).to_numpy()

    study_center, study_table = fit_study_site_center(
        truth,
        observed,
        studies,
        train_index,
        tau=args.study_tau,
        minimum_overlap=args.study_min_overlap,
    )
    protein_center, protein_study_table = fit_study_protein_center(
        protein, studies, train_index
    )
    protein_centered = protein - protein_center
    protein_lookup = {
        str(gene).upper(): index for index, gene in enumerate(protein_frame.columns)
    }
    parents = manifest["total_protein_gene"].astype(str).str.upper().to_numpy()
    parent_index = np.asarray([protein_lookup.get(gene, -1) for gene in parents])
    mapped = parent_index >= 0

    beta = np.zeros(len(targets), dtype=np.float32)
    n_observed = np.zeros(len(targets), dtype=np.int64)
    unshrunk = np.zeros(len(targets), dtype=np.float32)
    residual_after_study = truth - study_center
    for start in range(0, len(targets), args.site_chunk_size):
        stop = min(start + args.site_chunk_size, len(targets))
        local_parent = parent_index[start:stop]
        local_mapped = local_parent >= 0
        x = np.zeros((len(truth), stop - start), dtype=np.float32)
        if local_mapped.any():
            x[:, local_mapped] = protein_centered[:, local_parent[local_mapped]]
        y = residual_after_study[:, start:stop]
        mask = observed[:, start:stop] & np.isfinite(x)
        train_mask = mask[train_index]
        xt = x[train_index]
        yt = y[train_index]
        count = train_mask.sum(axis=0)
        numerator = np.where(train_mask, xt * yt, 0.0).sum(axis=0)
        denominator = np.where(train_mask, xt * xt, 0.0).sum(axis=0)
        slope = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1.0e-8,
        )
        shrink = count / (count + args.parent_slope_tau)
        slope = slope * shrink
        slope[(count < args.parent_min_observations) | ~local_mapped] = 0.0
        beta[start:stop] = slope.astype(np.float32)
        unshrunk[start:stop] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1.0e-8,
        ).astype(np.float32)
        n_observed[start:stop] = count

    parent_values = np.zeros_like(truth, dtype=np.float32)
    parent_values[:, mapped] = protein_centered[:, parent_index[mapped]]
    parent_component = parent_values * beta[None, :]
    fixed_offset = study_center + parent_component
    residual = np.where(observed, truth - fixed_offset, np.nan).astype(np.float32)

    pd.DataFrame(study_center, index=truth_frame.index, columns=targets).to_parquet(
        args.output_dir / "study_site_center.parquet"
    )
    pd.DataFrame(parent_component, index=truth_frame.index, columns=targets).to_parquet(
        args.output_dir / "parent_protein_component.parquet"
    )
    pd.DataFrame(fixed_offset, index=truth_frame.index, columns=targets).to_parquet(
        args.output_dir / "fixed_target_offset.parquet"
    )
    pd.DataFrame(residual, index=truth_frame.index, columns=targets).to_parquet(
        args.output_dir / "ptm_specific_residual.parquet"
    )
    beta_table = pd.DataFrame(
        {
            "target": targets,
            "parent_gene": parents,
            "parent_mapped": mapped,
            "training_observations": n_observed,
            "beta_unshrunk": unshrunk,
            "beta": beta,
        }
    )
    beta_table.to_csv(args.output_dir / "parent_beta.tsv", sep="\t", index=False)
    study_table.to_csv(args.output_dir / "study_site_center_summary.tsv", sep="\t", index=False)
    protein_study_table.to_csv(
        args.output_dir / "study_protein_center_summary.tsv", sep="\t", index=False
    )

    baseline_spearman = rowwise_spearman(truth[validation_index], study_center[validation_index])
    parent_spearman = rowwise_spearman(truth[validation_index], fixed_offset[validation_index])
    report = {
        "status": "complete",
        "model": "scp682_v2_m1_parent_ptm_decomposition",
        "fit_scope": "selection_train_only",
        "training_samples": int(len(train_index)),
        "validation_samples": int(len(validation_index)),
        "sites": int(len(targets)),
        "mapped_parent_sites": int(mapped.sum()),
        "nonzero_beta_sites": int(np.count_nonzero(beta)),
        "median_absolute_beta": float(np.median(np.abs(beta[beta != 0]))) if np.any(beta != 0) else 0.0,
        "study_tau": args.study_tau,
        "parent_slope_tau": args.parent_slope_tau,
        "train_study_only_mse": masked_mse(truth, study_center, train_index),
        "train_study_parent_mse": masked_mse(truth, fixed_offset, train_index),
        "validation_study_only_mse": masked_mse(truth, study_center, validation_index),
        "validation_study_parent_mse": masked_mse(truth, fixed_offset, validation_index),
        "validation_study_only_profile_spearman_median": float(np.nanmedian(baseline_spearman)),
        "validation_study_parent_profile_spearman_median": float(np.nanmedian(parent_spearman)),
        "sources": {
            "phosphosite": {"path": str(args.phosphosite), "sha256": sha256(args.phosphosite)},
            "protein": {"path": str(args.protein), "sha256": sha256(args.protein)},
            "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            "split": {"path": str(args.split), "sha256": sha256(args.split)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
        },
    }
    (args.output_dir / "decomposition_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "SUCCESS").touch()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
