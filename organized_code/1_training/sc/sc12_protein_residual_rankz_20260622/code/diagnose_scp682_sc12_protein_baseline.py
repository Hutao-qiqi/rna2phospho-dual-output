import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def spearman_np(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def pearson_np(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def read_tsv(path):
    return pd.read_csv(path, sep="\t", low_memory=False)


def parse_indices(text):
    if pd.isna(text) or str(text).strip() == "":
        return []
    return [int(x) for x in str(text).split(";") if str(x).strip() != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    ap.add_argument("--model-input-dir", required=True)
    ap.add_argument("--protein-cache-dir", required=True)
    ap.add_argument("--sc12-result-dir", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    input_dir = root / args.model_input_dir
    cache_dir = root / args.protein_cache_dir
    result_dir = root / args.sc12_result_dir
    out_dir = result_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_tsv(input_dir / "cell_metadata.tsv")
    target_table = read_tsv(input_dir / "phospho_target_table.tsv")
    y = np.load(input_dir / "targets.npy", mmap_mode="r")
    mask = np.load(input_dir / "target_mask.npy", mmap_mode="r").astype(bool)
    protein = np.load(cache_dir / "protein_predicted.npy", mmap_mode="r")
    protein_table = read_tsv(cache_dir / "protein_features.tsv")
    cache_meta = read_tsv(cache_dir / "cell_metadata.tsv")
    mapping = read_tsv(out_dir / "scp682_sc12_parent_protein_mapping.tsv")
    transform = read_tsv(out_dir / "scp682_sc12_parent_protein_transform.tsv")
    baseline = read_tsv(out_dir / "scp682_sc12_frozen_parent_protein_baseline.tsv")
    component = read_tsv(out_dir / "scp682_sc12_component_performance.tsv")

    if len(meta) != protein.shape[0]:
        raise ValueError(f"cell count mismatch: model={len(meta)} protein_cache={protein.shape[0]}")
    if "cell_id" in meta.columns and "cell_id" in cache_meta.columns:
        if not meta["cell_id"].astype(str).equals(cache_meta["cell_id"].astype(str)):
            raise ValueError("cell_id order differs between model input and protein cache")

    beta_lookup = {
        str(r["target_id"]): (float(r["protein_beta"]), float(r["protein_bias"]))
        for _, r in baseline.iterrows()
    }
    rows = []
    for _, mrow in mapping.iterrows():
        tid = str(mrow["target_id"])
        j = int(mrow["target_index"])
        indices = parse_indices(mrow.get("matched_cache_indices", ""))
        if not indices:
            continue
        parent_mat = np.asarray(protein[:, indices], dtype=np.float32)
        parent_mean = np.nanmean(parent_mat, axis=1)
        obs = np.asarray(y[:, j], dtype=np.float32)
        obs_mask = np.asarray(mask[:, j], dtype=bool)
        beta, bias = beta_lookup.get(tid, (np.nan, np.nan))
        protein_component = parent_mean * beta + bias
        dataset_values = list(meta["dataset_id"].astype(str).unique())
        for ds in ["all"] + dataset_values:
            keep = obs_mask.copy()
            if ds != "all":
                keep &= meta["dataset_id"].astype(str).to_numpy() == ds
            if keep.sum() < 3:
                continue
            row = {
                "dataset_id": ds,
                "target_id": tid,
                "target_index": j,
                "n_cells": int(keep.sum()),
                "parent_genes": mrow.get("matched_cache_names", ""),
                "n_parent_proteins": int(mrow.get("n_matched_parent_proteins", len(indices))),
                "protein_beta": beta,
                "protein_abs_beta": abs(beta) if np.isfinite(beta) else np.nan,
                "parent_mean_vs_phospho_spearman": spearman_np(parent_mean[keep], obs[keep]),
                "parent_mean_vs_phospho_pearson": pearson_np(parent_mean[keep], obs[keep]),
                "protein_component_vs_phospho_spearman": spearman_np(protein_component[keep], obs[keep]),
                "protein_component_vs_phospho_pearson": pearson_np(protein_component[keep], obs[keep]),
                "parent_mean_sd": float(np.nanstd(parent_mean[keep])),
                "phospho_sd": float(np.nanstd(obs[keep])),
            }
            if len(indices) > 1:
                single_vals = []
                for idx in indices:
                    single_vals.append(spearman_np(np.asarray(protein[:, idx], dtype=np.float32)[keep], obs[keep]))
                finite = [v for v in single_vals if np.isfinite(v)]
                row["best_single_parent_spearman"] = float(np.max(finite)) if finite else np.nan
                row["mean_minus_best_single_parent_spearman"] = (
                    row["parent_mean_vs_phospho_spearman"] - row["best_single_parent_spearman"]
                    if np.isfinite(row["parent_mean_vs_phospho_spearman"]) and finite
                    else np.nan
                )
            else:
                row["best_single_parent_spearman"] = row["parent_mean_vs_phospho_spearman"]
                row["mean_minus_best_single_parent_spearman"] = 0.0
            rows.append(row)

    diag = pd.DataFrame(rows)
    diag.to_csv(out_dir / "scp682_sc12_parent_protein_diagnostic_per_target.tsv", sep="\t", index=False)

    comp = component.copy()
    comp = comp[comp["component"].isin(["protein_only", "phospho_residual_only", "protein_plus_residual"])]
    comp_summary = (
        comp.groupby(["evaluation", "test_dataset", "component"])["spearman"]
        .median()
        .reset_index()
        .pivot_table(index=["evaluation", "test_dataset"], columns="component", values="spearman")
        .reset_index()
    )
    for col in ["protein_only", "phospho_residual_only", "protein_plus_residual"]:
        if col not in comp_summary.columns:
            comp_summary[col] = np.nan
    comp_summary["residual_gain_over_protein_only"] = comp_summary["protein_plus_residual"] - comp_summary["protein_only"]
    comp_summary["protein_component_gain_over_residual_only"] = comp_summary["protein_plus_residual"] - comp_summary["phospho_residual_only"]

    diag_summary = diag.groupby("dataset_id").agg(
        n_targets=("target_id", "nunique"),
        median_parent_mean_vs_phospho_spearman=("parent_mean_vs_phospho_spearman", "median"),
        median_protein_component_vs_phospho_spearman=("protein_component_vs_phospho_spearman", "median"),
        median_best_single_parent_spearman=("best_single_parent_spearman", "median"),
        median_mean_minus_best_single_parent_spearman=("mean_minus_best_single_parent_spearman", "median"),
        median_abs_beta=("protein_abs_beta", "median"),
        median_parent_mean_sd=("parent_mean_sd", "median"),
        median_phospho_sd=("phospho_sd", "median"),
    ).reset_index()
    diag_summary.to_csv(out_dir / "scp682_sc12_parent_protein_diagnostic_summary.tsv", sep="\t", index=False)
    comp_summary.to_csv(out_dir / "scp682_sc12_component_diagnostic_summary.tsv", sep="\t", index=False)

    payload = {
        "n_cells": int(len(meta)),
        "n_targets_mapped": int(mapping["n_matched_parent_proteins"].fillna(0).astype(int).gt(0).sum()),
        "n_targets_unmapped": int(mapping["n_matched_parent_proteins"].fillna(0).astype(int).le(0).sum()),
        "median_abs_beta": float(np.nanmedian(np.abs(baseline["protein_beta"].to_numpy(dtype=float)))),
        "tables": {
            "per_target": str(out_dir / "scp682_sc12_parent_protein_diagnostic_per_target.tsv"),
            "parent_summary": str(out_dir / "scp682_sc12_parent_protein_diagnostic_summary.tsv"),
            "component_summary": str(out_dir / "scp682_sc12_component_diagnostic_summary.tsv"),
        },
    }
    (result_dir / "reports").mkdir(exist_ok=True, parents=True)
    with (result_dir / "reports" / "scp682_sc12_parent_protein_diagnostic.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))
    print("\nParent summary")
    print(diag_summary.to_string(index=False))
    print("\nComponent summary")
    print(comp_summary.to_string(index=False))


if __name__ == "__main__":
    main()
