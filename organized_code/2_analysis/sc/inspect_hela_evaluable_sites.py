from pathlib import Path

import numpy as np
import pandas as pd


ROOTS = [
    Path(r"D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\phospho_model_inputs\scgpt_frozen_gse300551_signal_seq_multidomain_v1"),
    Path(r"D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\phospho_model_inputs\geneformer_pathway_flatten_benchmark_v1"),
    Path(r"D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_masked_multisite_v1"),
]

RESULT_ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell")
OUT = RESULT_ROOT / "20260531_scp682_sc_kirc_rps6_validation_v1" / "tables" / "debug_hela_evaluable_sites.tsv"


def find_result_tables():
    hits = []
    for p in RESULT_ROOT.rglob("*.tsv"):
        name = p.name.lower()
        if ("reconstruction" in name or "external" in name or "performance" in name) and "scp682" in str(p).lower():
            hits.append(p)
    return hits


def summarize_input(root: Path):
    meta_p = root / "cell_metadata.tsv"
    target_p = root / "phospho_target_table.tsv"
    y_p = root / "targets.npy"
    mask_p = root / "target_mask.npy"
    if not (meta_p.exists() and target_p.exists() and y_p.exists() and mask_p.exists()):
        return pd.DataFrame()
    meta = pd.read_csv(meta_p, sep="\t", low_memory=False)
    if "dataset_id" not in meta.columns:
        return pd.DataFrame()
    idx = np.flatnonzero(meta["dataset_id"].astype(str).to_numpy() == "signal_seq_gse256403_hela_2024")
    if len(idx) == 0:
        return pd.DataFrame()
    targets = pd.read_csv(target_p, sep="\t")
    y = np.load(y_p, mmap_mode="r")
    mask = np.load(mask_p, mmap_mode="r")
    rows = []
    for j in range(y.shape[1]):
        mm = np.asarray(mask[idx, j]).astype(bool)
        if mm.sum() == 0:
            continue
        vals = np.asarray(y[idx, j])[mm]
        sub = targets[targets["target_index"].astype(int) == j] if "target_index" in targets.columns else pd.DataFrame()
        hela_sub = sub[sub["dataset_id"].astype(str).eq("signal_seq_gse256403_hela_2024")] if not sub.empty and "dataset_id" in sub.columns else pd.DataFrame()
        rows.append(
            {
                "input_root": str(root),
                "target_index": j,
                "target_id": ";".join(sub["target_id"].astype(str).drop_duplicates().tolist()) if "target_id" in sub else "",
                "hela_feature_id": ";".join(hela_sub["feature_id"].astype(str).drop_duplicates().tolist()) if "feature_id" in hela_sub else "",
                "all_feature_dataset_ids": ";".join(sub["dataset_id"].astype(str).drop_duplicates().tolist()) if "dataset_id" in sub else "",
                "protein_symbol": ";".join(sub["protein_symbol"].astype(str).drop_duplicates().tolist()) if "protein_symbol" in sub else "",
                "residue": ";".join(sub["residue"].astype(str).drop_duplicates().tolist()) if "residue" in sub else "",
                "evaluation_tier": ";".join(sub["evaluation_tier"].astype(str).drop_duplicates().tolist()) if "evaluation_tier" in sub else "",
                "n_hela_cells": len(idx),
                "n_masked": int(mm.sum()),
                "sd": float(np.nanstd(vals)),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
                "n_unique": int(len(np.unique(vals[np.isfinite(vals)]))),
                "can_compute_spearman": bool(np.nanstd(vals) > 0 and len(np.unique(vals[np.isfinite(vals)])) > 1),
            }
        )
    return pd.DataFrame(rows)


def main():
    parts = []
    for root in ROOTS:
        df = summarize_input(root)
        if not df.empty:
            parts.append(df)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, sep="\t", index=False)
    print("OUT", OUT)
    if out.empty:
        print("no input rows")
        return
    for root, sub in out.groupby("input_root"):
        print("\n###", root)
        cols = [
            "target_index",
            "target_id",
            "hela_feature_id",
            "all_feature_dataset_ids",
            "n_masked",
            "sd",
            "n_unique",
            "can_compute_spearman",
        ]
        print(sub[cols].sort_values(["can_compute_spearman", "sd"], ascending=[False, False]).to_string(index=False))
    print("\n### possible result tables")
    for p in find_result_tables()[:80]:
        print(p)


if __name__ == "__main__":
    main()
