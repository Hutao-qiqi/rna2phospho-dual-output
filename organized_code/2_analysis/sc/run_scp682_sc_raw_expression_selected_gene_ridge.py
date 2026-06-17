#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_scp682_sc_reviewer_missing_tables_v2 import (
    DATASET_NAMES,
    load_blair_expression,
    load_gse_expression,
    load_mtx_expression,
    load_signal_expression,
    norm_gene_name,
    target_gene_candidates,
)


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def safe_spearman(x, y, min_n=20):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < min_n:
        return math.nan, math.nan, n, "too_few_cells"
    if np.nanstd(x[ok]) < 1e-12:
        return math.nan, math.nan, n, "constant_prediction"
    if np.nanstd(y[ok]) < 1e-12:
        return math.nan, math.nan, n, "constant_observed"
    r = spearmanr(x[ok], y[ok])
    return float(r.correlation), float(r.pvalue), n, "ok"


def read_gene_list(path: Path) -> list[str]:
    return pd.read_csv(path, sep="\t", header=None).iloc[:, 0].astype(str).tolist()


def load_iccite_selected(root: Path, genes: list[str], n_cells: int) -> dict[str, np.ndarray]:
    base = root / "01_data" / "single_cell" / "intermediate" / "paired_matrices" / "iccite_seq_tcell_2025" / "rna_full_counts"
    feature_path = base / "rna_full_counts_features.tsv"
    matrix_path = base / "rna_full_counts.mtx"
    features = read_gene_list(feature_path)
    gene_to_row = {}
    for i, g in enumerate(features, start=1):
        gene_to_row.setdefault(norm_gene_name(g), i)
    selected = {gene_to_row[norm_gene_name(g)]: g for g in genes if norm_gene_name(g) in gene_to_row}
    out = {g: np.zeros(n_cells, dtype=np.float32) for g in genes}
    if not selected:
        return out
    log(f"icCITE raw mtx scan selected_genes={len(selected)}")
    opener = gzip.open if str(matrix_path).endswith(".gz") else open
    with opener(matrix_path, "rt", encoding="utf-8", errors="ignore") as fh:
        dims_seen = False
        for line in fh:
            if not line or line.startswith("%"):
                continue
            if not dims_seen:
                dims_seen = True
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            i = int(parts[0])
            if i not in selected:
                continue
            j = int(parts[1]) - 1
            if 0 <= j < n_cells:
                out[selected[i]][j] = float(parts[2])
    return out


def expr_dict_to_matrix(expr: dict[str, np.ndarray], genes: list[str]) -> np.ndarray:
    arr = np.vstack([expr.get(g, np.full_like(next(iter(expr.values())), np.nan)) for g in genes]).T.astype(np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    arr = np.where(arr >= 0, np.log1p(arr), arr).astype(np.float32)
    return arr


def load_dataset_expr(dataset_id: str, meta_sub: pd.DataFrame, genes: list[str], root: Path, paired_root: Path, input_dir: Path) -> dict[str, np.ndarray]:
    if dataset_id == "iccite_seq_tcell_2025":
        return load_iccite_selected(root, genes, len(meta_sub))
    if dataset_id.startswith("signal_seq"):
        return load_signal_expression(meta_sub, genes)
    if dataset_id == "gse300551_iccite_plex_kinase_2025":
        extracted = input_dir.parents[1] / "gse300551_iccite_plex_kinase_2025_extracted"
        return load_gse_expression(meta_sub, genes, extracted)
    if dataset_id == "phospho_seq_blair_2025_phospho_multi":
        return load_blair_expression(meta_sub, genes, paired_root / dataset_id / "rna_counts.tsv")
    if dataset_id == "vivo_seq_th17_2025":
        return load_mtx_expression(
            meta_sub,
            genes,
            paired_root / dataset_id / "rna_counts.mtx",
            paired_root / dataset_id / "genes.tsv",
            paired_root / dataset_id / "barcodes.tsv",
            "barcodes",
        )
    if dataset_id == "qurie_seq_bjab_2021":
        return load_mtx_expression(
            meta_sub,
            genes,
            paired_root / dataset_id / "rna_counts.mtx.gz",
            paired_root / dataset_id / "genes.tsv",
            paired_root / dataset_id / "barcodes.tsv",
            "cell_id",
        )
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    ap.add_argument("--input-dir", default=r"D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_gse300551_signal_seq_multidomain_v1")
    ap.add_argument("--output-dir", default=r"D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260531_scp682_sc_raw_expression_selected_gene_ridge_v1")
    ap.add_argument("--train-datasets", default="iccite_seq_tcell_2025,qurie_seq_bjab_2021")
    ap.add_argument("--test-datasets", default="gse300551_iccite_plex_kinase_2025,signal_seq_gse256403_hela_2024,signal_seq_gse256404_pdo_caf_2024,phospho_seq_blair_2025_phospho_multi,vivo_seq_th17_2025")
    ap.add_argument("--min-train", type=int, default=50)
    ap.add_argument("--min-test", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.root)
    input_dir = Path(args.input_dir)
    out = Path(args.output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    paired_root = root / "01_data" / "single_cell" / "intermediate" / "paired_matrices"

    meta = pd.read_csv(input_dir / "cell_metadata.tsv", sep="\t", low_memory=False)
    targets = pd.read_csv(input_dir / "phospho_target_table.tsv", sep="\t").drop_duplicates("target_index").sort_values("target_index")
    y = np.load(input_dir / "targets.npy", mmap_mode="r")
    mask = np.load(input_dir / "target_mask.npy", mmap_mode="r").astype(bool)
    genes = sorted({g for _, r in targets.iterrows() for g in target_gene_candidates(r)})
    pd.DataFrame({"gene": genes}).to_csv(out / "tables" / "raw_expression_selected_genes.tsv", sep="\t", index=False)
    train_datasets = [x.strip() for x in args.train_datasets.split(",") if x.strip()]
    test_datasets = [x.strip() for x in args.test_datasets.split(",") if x.strip()]

    expr_by_ds = {}
    status_rows = []
    for dataset_id in train_datasets + test_datasets:
        sub = meta[meta["dataset_id"].astype(str).eq(dataset_id)].reset_index(drop=True)
        if sub.empty:
            continue
        try:
            expr = load_dataset_expr(dataset_id, sub, genes, root, paired_root, input_dir)
            status = "computed"
        except Exception as exc:
            expr = {}
            status = f"failed:{type(exc).__name__}:{exc}"
        expr_by_ds[dataset_id] = expr
        n_values = int(sum(np.isfinite(v).sum() for v in expr.values())) if expr else 0
        status_rows.append({"dataset_id": dataset_id, "status": status, "n_expression_values": n_values})
        log(f"{dataset_id} {status} values={n_values}")

    train_meta_parts = []
    train_x_parts = []
    train_row_positions = []
    dataset_arr = meta["dataset_id"].astype(str).to_numpy()
    for dataset_id in train_datasets:
        idx = np.flatnonzero(dataset_arr == dataset_id)
        if dataset_id not in expr_by_ds:
            continue
        train_x_parts.append(expr_dict_to_matrix(expr_by_ds[dataset_id], genes))
        train_row_positions.append(idx)
        train_meta_parts.append(dataset_id)
    x_train = np.vstack(train_x_parts).astype(np.float32)
    train_idx = np.concatenate(train_row_positions)
    scaler = StandardScaler()
    x_train_z = scaler.fit_transform(x_train).astype(np.float32)

    rows = []
    alphas = np.logspace(-3, 3, 13)
    for _, t in targets.iterrows():
        j = int(t["target_index"])
        tr_ok = mask[train_idx, j]
        if int(tr_ok.sum()) < args.min_train:
            continue
        model = RidgeCV(alphas=alphas)
        model.fit(x_train_z[tr_ok], np.asarray(y[train_idx[tr_ok], j], dtype=np.float32))
        for dataset_id in test_datasets:
            if dataset_id not in expr_by_ds:
                continue
            idx = np.flatnonzero(dataset_arr == dataset_id)
            te_ok = mask[idx, j]
            if int(te_ok.sum()) < args.min_test:
                continue
            x_test = expr_dict_to_matrix(expr_by_ds[dataset_id], genes)
            x_test_z = scaler.transform(x_test).astype(np.float32)
            pred = model.predict(x_test_z[te_ok]).astype(np.float32)
            obs = np.asarray(y[idx[te_ok], j], dtype=np.float32)
            rho, pval, n, score_status = safe_spearman(pred, obs, min_n=args.min_test)
            rows.append({
                "method": "raw_expression_selected_gene_ridge",
                "evaluation": "scFoundation_removed_raw_expression_ablation",
                "train_dataset": ";".join(train_datasets),
                "test_dataset": dataset_id,
                "cohort_name": DATASET_NAMES.get(dataset_id, dataset_id),
                "target_id": str(t["target_id"]),
                "target_index": j,
                "protein_symbol": str(t.get("protein_symbol", "")),
                "residue": str(t.get("residue", "")),
                "n_train": int(tr_ok.sum()),
                "n_test": n,
                "spearman": rho,
                "spearman_pvalue": pval,
                "score_status": score_status,
                "alpha": float(model.alpha_),
                "n_raw_genes": int(len(genes)),
            })
    perf = pd.DataFrame(rows)
    perf.to_csv(out / "tables" / "raw_expression_selected_gene_ridge_per_target.tsv", sep="\t", index=False, na_rep="NA")
    summary = []
    if not perf.empty:
        for ds, sub in perf.groupby("test_dataset"):
            vals = pd.to_numeric(sub["spearman"], errors="coerce")
            summary.append({
                "test_dataset": ds,
                "cohort_name": DATASET_NAMES.get(ds, ds),
                "n_targets": int(vals.notna().sum()),
                "median_spearman": float(np.nanmedian(vals)) if vals.notna().any() else math.nan,
                "mean_spearman": float(np.nanmean(vals)) if vals.notna().any() else math.nan,
            })
    pd.DataFrame(summary).to_csv(out / "tables" / "raw_expression_selected_gene_ridge_summary.tsv", sep="\t", index=False, na_rep="NA")
    pd.DataFrame(status_rows).to_csv(out / "tables" / "raw_expression_loading_status.tsv", sep="\t", index=False)
    (out / "reports" / "run_manifest.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "done.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    log(f"done rows={len(rows)} out={out}")


if __name__ == "__main__":
    main()
