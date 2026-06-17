#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.stats import spearmanr


DATASET_NAMES = {
    "iccite_seq_tcell_2025": "icCITE",
    "qurie_seq_bjab_2021": "QuRIE",
    "gse300551_iccite_plex_kinase_2025": "GSE300551",
    "signal_seq_gse256403_hela_2024": "SIGNAL-seq HeLa",
    "signal_seq_gse256404_pdo_caf_2024": "SIGNAL-seq PDO/CAF",
    "phospho_seq_blair_2025_phospho_multi": "Blair",
    "vivo_seq_th17_2025": "Vivo-seq Th17",
}


def safe_float(x):
    try:
        x = float(x)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return math.nan


def safe_spearman(x, y, min_n=10):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < min_n:
        return math.nan, math.nan, n, "too_few_cells"
    if np.nanstd(x[ok]) < 1e-12:
        return math.nan, math.nan, n, "constant_predictor"
    if np.nanstd(y[ok]) < 1e-12:
        return math.nan, math.nan, n, "constant_observed"
    out = spearmanr(x[ok], y[ok])
    rho = float(out.correlation) if math.isfinite(float(out.correlation)) else math.nan
    pval = float(out.pvalue) if math.isfinite(float(out.pvalue)) else math.nan
    return rho, pval, n, "ok"


def write_table(path: Path, df: pd.DataFrame, description: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, na_rep="NA")
    md = path.with_suffix(".md")
    md.write_text(description + "\n", encoding="utf-8")


def norm_gene_name(x: str) -> str:
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    if "_" in s and (s.startswith("GRCh38_") or s.startswith("mm10_")):
        s = s.split("_", 1)[1]
    return s.upper()


def target_gene_candidates(row: pd.Series) -> list[str]:
    raw = str(row.get("protein_symbol", "")).strip()
    if not raw or raw.lower() in {"nan", "pending"}:
        tid = str(row.get("target_id", ""))
        raw = tid.split("_", 1)[0]
    vals = []
    for part in re.split(r"[;,/|]+", raw):
        g = norm_gene_name(part)
        if g and g not in {"PENDING", "UNKNOWN"} and not g.startswith("HM"):
            vals.append(g)
    out = []
    for g in vals:
        if g not in out:
            out.append(g)
    return out


def values_from_sparse_slice(x):
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray()).reshape(-1)
    return np.asarray(x).reshape(-1)


def load_signal_expression(meta_sub: pd.DataFrame, genes: list[str]) -> dict[str, np.ndarray]:
    import anndata as ad

    meta_sub = meta_sub.reset_index(drop=True)
    out = {g: np.full(len(meta_sub), np.nan, dtype=np.float32) for g in genes}
    if "rna_h5ad" not in meta_sub.columns:
        return out
    for path, idx in meta_sub.groupby("rna_h5ad", dropna=True).groups.items():
        if not isinstance(path, str) or not path or not os.path.exists(path):
            continue
        rows = np.asarray(list(idx), dtype=int)
        sub = meta_sub.loc[rows]
        adata = ad.read_h5ad(path, backed="r")
        try:
            obs_pos = {str(x): i for i, x in enumerate(adata.obs_names)}
            pos = np.array([obs_pos.get(str(x), -1) for x in sub["raw_barcode"].astype(str)], dtype=int)
            keep = pos >= 0
            if not keep.any():
                continue
            kept_rows = rows[keep]
            kept_pos = pos[keep]
            unique_pos, inverse = np.unique(kept_pos, return_inverse=True)
            var_map = {norm_gene_name(x): i for i, x in enumerate(adata.var_names)}
            for g in genes:
                j = var_map.get(norm_gene_name(g))
                if j is None:
                    continue
                vals_unique = values_from_sparse_slice(adata.X[unique_pos, j]).astype(np.float32)
                vals = vals_unique[inverse]
                out[g][kept_rows] = vals
        finally:
            adata.file.close()
    return out


def load_gse_expression(meta_sub: pd.DataFrame, genes: list[str], extracted_dir: Path) -> dict[str, np.ndarray]:
    meta_sub = meta_sub.reset_index(drop=True)
    out = {g: np.full(len(meta_sub), np.nan, dtype=np.float32) for g in genes}
    if "rna_h5" not in meta_sub.columns:
        return out
    for rna_h5, idx in meta_sub.groupby("rna_h5", dropna=True).groups.items():
        if not isinstance(rna_h5, str) or not rna_h5:
            continue
        path = Path(rna_h5)
        if not path.is_absolute():
            path = extracted_dir / rna_h5
        if not path.exists():
            continue
        rows = np.asarray(list(idx), dtype=int)
        sub = meta_sub.loc[rows]
        with h5py.File(path, "r") as f:
            barcodes = [x.decode() if isinstance(x, bytes) else str(x) for x in f["matrix/barcodes"][:]]
            barcode_pos = {b: i for i, b in enumerate(barcodes)}
            features = [x.decode() if isinstance(x, bytes) else str(x) for x in f["matrix/features/name"][:]]
            gene_to_idx = {}
            for i, name in enumerate(features):
                g = norm_gene_name(name)
                gene_to_idx.setdefault(g, i)
            gene_idx = {g: gene_to_idx.get(norm_gene_name(g)) for g in genes}
            wanted_idx = {v for v in gene_idx.values() if v is not None}
            indptr = f["matrix/indptr"]
            indices = f["matrix/indices"]
            data = f["matrix/data"]
            for local_i, bc in enumerate(sub["raw_barcode"].astype(str)):
                col = barcode_pos.get(bc)
                if col is None:
                    continue
                start = int(indptr[col])
                end = int(indptr[col + 1])
                idxs = indices[start:end]
                vals = data[start:end]
                if len(idxs) == 0:
                    continue
                hit = np.isin(idxs, list(wanted_idx))
                if not hit.any():
                    continue
                val_map = {int(i): float(v) for i, v in zip(idxs[hit], vals[hit])}
                for g, gi in gene_idx.items():
                    if gi is not None and gi in val_map:
                        out[g][rows[local_i]] = val_map[gi]
    return out


def load_mtx_expression(meta_sub: pd.DataFrame, genes: list[str], matrix_path: Path, genes_path: Path, barcodes_path: Path, barcode_col: str) -> dict[str, np.ndarray]:
    out = {g: np.full(len(meta_sub), np.nan, dtype=np.float32) for g in genes}
    if not (matrix_path.exists() and genes_path.exists() and barcodes_path.exists()):
        return out
    gene_names = pd.read_csv(genes_path, sep="\t", header=None).iloc[:, 0].astype(str).tolist()
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None).iloc[:, 0].astype(str).tolist()
    gene_to_idx = {norm_gene_name(g): i for i, g in enumerate(gene_names)}
    mat = mmread(str(matrix_path)).tocsr()
    if mat.shape[0] == len(barcodes) and mat.shape[1] == len(gene_names):
        mat = mat.T.tocsr()
    barcode_pos = {b: i for i, b in enumerate(barcodes)}
    pos = np.array([barcode_pos.get(str(x), -1) for x in meta_sub[barcode_col].astype(str)], dtype=int)
    keep = pos >= 0
    for g in genes:
        gi = gene_to_idx.get(norm_gene_name(g))
        if gi is None:
            continue
        vals = np.asarray(mat[gi, pos[keep]].todense()).reshape(-1).astype(np.float32)
        out[g][keep] = vals
    return out


def load_blair_expression(meta_sub: pd.DataFrame, genes: list[str], tsv_path: Path) -> dict[str, np.ndarray]:
    out = {g: np.full(len(meta_sub), np.nan, dtype=np.float32) for g in genes}
    if not tsv_path.exists():
        return out
    wanted = {norm_gene_name(g) for g in genes}
    chunks = []
    for chunk in pd.read_csv(tsv_path, sep="\t", chunksize=2000):
        key = chunk.iloc[:, 0].astype(str).map(norm_gene_name)
        sub = chunk[key.isin(wanted)]
        if not sub.empty:
            chunks.append(sub)
    if not chunks:
        return out
    df = pd.concat(chunks, ignore_index=True)
    df["_gene_norm"] = df.iloc[:, 0].astype(str).map(norm_gene_name)
    barcode_pos = {str(x): i for i, x in enumerate(df.columns[1:-1])}
    cols = [str(x) for x in meta_sub["cell_id"]]
    for g in genes:
        sub = df[df["_gene_norm"].eq(norm_gene_name(g))]
        if sub.empty:
            continue
        row = sub.iloc[0]
        vals = np.full(len(meta_sub), np.nan, dtype=np.float32)
        for i, bc in enumerate(cols):
            if bc in row.index:
                vals[i] = safe_float(row[bc])
        out[g] = vals
    return out


def summarize_constant_baseline(input_dir: Path, out_dir: Path):
    meta = pd.read_csv(input_dir / "cell_metadata.tsv", sep="\t", low_memory=False)
    targets = pd.read_csv(input_dir / "phospho_target_table.tsv", sep="\t").drop_duplicates("target_index").sort_values("target_index")
    y = np.load(input_dir / "targets.npy", mmap_mode="r")
    mask = np.load(input_dir / "target_mask.npy", mmap_mode="r").astype(bool)
    ds = meta["dataset_id"].astype(str).to_numpy()
    train = np.isin(ds, ["iccite_seq_tcell_2025", "qurie_seq_bjab_2021"])
    rows = []
    train_mean = np.full(y.shape[1], np.nan, dtype=np.float64)
    for _, t in targets.iterrows():
        j = int(t["target_index"])
        ok = train & mask[:, j]
        if ok.sum() > 0:
            train_mean[j] = float(np.nanmean(y[ok, j]))
    for dataset_id in sorted(pd.unique(meta["dataset_id"].astype(str))):
        idx = ds == dataset_id
        for _, t in targets.iterrows():
            j = int(t["target_index"])
            ok = idx & mask[:, j]
            n = int(ok.sum())
            if n < 3:
                continue
            obs = np.asarray(y[ok, j], dtype=np.float64)
            pred = np.full(n, train_mean[j], dtype=np.float64)
            rho, pval, _, reason = safe_spearman(pred, obs, min_n=3)
            rows.append({
                "baseline_name": "train_target_mean",
                "dataset_id": dataset_id,
                "cohort_name": DATASET_NAMES.get(dataset_id, dataset_id),
                "target_id": str(t["target_id"]),
                "target_index": j,
                "protein_symbol": str(t.get("protein_symbol", "")),
                "residue": str(t.get("residue", "")),
                "n_cells": n,
                "train_mean": train_mean[j],
                "observed_mean": float(np.nanmean(obs)),
                "observed_sd": float(np.nanstd(obs)),
                "mae": float(np.nanmean(np.abs(pred - obs))),
                "rmse": float(np.sqrt(np.nanmean((pred - obs) ** 2))),
                "spearman": rho,
                "spearman_pvalue": pval,
                "spearman_status": reason,
                "source_file": str(input_dir / "targets.npy"),
            })
    df = pd.DataFrame(rows)
    write_table(
        out_dir / "mean_baseline_per_target.tsv",
        df,
        "训练集每个位点均值基线。逐细胞 Spearman 对常数预测不适用，表中保留 MAE/RMSE 和常数预测状态。",
    )
    return meta, targets, y, mask


def build_cognate_mrna(input_dir: Path, paired_root: Path, meta: pd.DataFrame, targets: pd.DataFrame, y, mask, out_dir: Path):
    all_genes = []
    for _, row in targets.iterrows():
        all_genes.extend(target_gene_candidates(row))
    genes = sorted(set(all_genes))
    ds_to_expr = {}
    ds_to_status = []
    for dataset_id in [
        "gse300551_iccite_plex_kinase_2025",
        "signal_seq_gse256403_hela_2024",
        "signal_seq_gse256404_pdo_caf_2024",
        "phospho_seq_blair_2025_phospho_multi",
        "vivo_seq_th17_2025",
        "qurie_seq_bjab_2021",
    ]:
        sub = meta[meta["dataset_id"].astype(str).eq(dataset_id)].copy()
        if sub.empty:
            continue
        expr = {}
        status = "not_attempted"
        try:
            if dataset_id.startswith("signal_seq"):
                expr = load_signal_expression(sub, genes)
                status = "computed_from_h5ad"
            elif dataset_id == "gse300551_iccite_plex_kinase_2025":
                extracted = input_dir.parents[1] / "gse300551_iccite_plex_kinase_2025_extracted"
                expr = load_gse_expression(sub, genes, extracted)
                status = "computed_from_10x_h5"
            elif dataset_id == "phospho_seq_blair_2025_phospho_multi":
                expr = load_blair_expression(sub, genes, paired_root / dataset_id / "rna_counts.tsv")
                status = "computed_from_tsv"
            elif dataset_id == "vivo_seq_th17_2025":
                expr = load_mtx_expression(
                    sub,
                    genes,
                    paired_root / dataset_id / "rna_counts.mtx",
                    paired_root / dataset_id / "genes.tsv",
                    paired_root / dataset_id / "barcodes.tsv",
                    "barcodes",
                )
                status = "computed_from_mtx"
            elif dataset_id == "qurie_seq_bjab_2021":
                expr = load_mtx_expression(
                    sub,
                    genes,
                    paired_root / dataset_id / "rna_counts.mtx.gz",
                    paired_root / dataset_id / "genes.tsv",
                    paired_root / dataset_id / "barcodes.tsv",
                    "cell_id",
                )
                status = "computed_from_mtx_gz"
        except Exception as exc:
            status = f"failed:{type(exc).__name__}:{exc}"
            expr = {}
        ds_to_expr[dataset_id] = expr
        n_nonmissing = int(sum(np.isfinite(v).sum() for v in expr.values())) if expr else 0
        ds_to_status.append({"dataset_id": dataset_id, "status": status, "n_expression_values_loaded": n_nonmissing})
    rows = []
    ds = meta["dataset_id"].astype(str).to_numpy()
    for dataset_id, expr in ds_to_expr.items():
        if not expr:
            continue
        idx = np.flatnonzero(ds == dataset_id)
        for _, t in targets.iterrows():
            j = int(t["target_index"])
            genes_used = [g for g in target_gene_candidates(t) if g in expr and np.isfinite(expr[g]).any()]
            if not genes_used or j >= y.shape[1]:
                continue
            ok = mask[idx, j]
            if int(ok.sum()) < 10:
                continue
            mat = np.vstack([expr[g] for g in genes_used]).astype(np.float64)
            x = np.nanmean(mat, axis=0)
            obs = np.asarray(y[idx, j], dtype=np.float64)
            rho, pval, n, reason = safe_spearman(x[ok], obs[ok], min_n=10)
            rows.append({
                "baseline_name": "cognate_mRNA",
                "dataset_id": dataset_id,
                "cohort_name": DATASET_NAMES.get(dataset_id, dataset_id),
                "target_id": str(t["target_id"]),
                "target_index": j,
                "protein_symbol": str(t.get("protein_symbol", "")),
                "residue": str(t.get("residue", "")),
                "genes_used": ";".join(genes_used),
                "n_cells": n,
                "spearman": rho,
                "spearman_pvalue": pval,
                "spearman_status": reason,
                "expression_source": dict((r["dataset_id"], r["status"]) for r in ds_to_status).get(dataset_id, ""),
            })
    write_table(
        out_dir / "cognate_mRNA_per_target.tsv",
        pd.DataFrame(rows),
        "同源 mRNA 与对应磷酸化读数的逐细胞 Spearman。多基因读数用可获得基因表达均值。",
    )
    write_table(
        out_dir / "cognate_mRNA_expression_loading_status.tsv",
        pd.DataFrame(ds_to_status),
        "同源 mRNA 表达提取状态表。",
    )


def build_high_error_qc(result_dir: Path, meta: pd.DataFrame, out_dir: Path):
    tables = result_dir / "tables"
    files = sorted(tables.glob("scp682_sc11_predicted_observed_*.tsv"))
    meta_cols = [
        "row_index", "cell_id", "dataset_id", "nCount_RNA", "nFeature_RNA", "percent.mt",
        "n_counts", "n_genes", "mito_proportion", "ribo_proportion", "sample_key",
        "cell_type_label", "state_label", "condition_label", "ibrutinib", "treatment",
    ]
    have_cols = [c for c in meta_cols if c in meta.columns]
    meta_small = meta[have_cols].copy()
    if "row_index" in meta_small.columns:
        meta_small["row_index"] = pd.to_numeric(meta_small["row_index"], errors="coerce").astype("Int64")
    rows = []
    for path in files:
        cell_rows = []
        for chunk in pd.read_csv(path, sep="\t", chunksize=500000):
            chunk["abs_error"] = (pd.to_numeric(chunk["predicted"], errors="coerce") - pd.to_numeric(chunk["observed"], errors="coerce")).abs()
            chunk["signed_error"] = pd.to_numeric(chunk["predicted"], errors="coerce") - pd.to_numeric(chunk["observed"], errors="coerce")
            grp = chunk.groupby(["cohort_id", "cell_id", "row_index"], dropna=False).agg(
                mean_abs_error=("abs_error", "mean"),
                median_abs_error=("abs_error", "median"),
                mean_signed_error=("signed_error", "mean"),
                n_readouts=("target_id", "nunique"),
            ).reset_index()
            cell_rows.append(grp)
        if cell_rows:
            rows.append(pd.concat(cell_rows, ignore_index=True).groupby(["cohort_id", "cell_id", "row_index"], dropna=False).agg(
                mean_abs_error=("mean_abs_error", "mean"),
                median_abs_error=("median_abs_error", "median"),
                mean_signed_error=("mean_signed_error", "mean"),
                n_readouts=("n_readouts", "max"),
            ).reset_index())
    if rows:
        err = pd.concat(rows, ignore_index=True)
        err["row_index"] = pd.to_numeric(err["row_index"], errors="coerce").astype("Int64")
        merged = err.merge(meta_small, on="row_index", how="left", suffixes=("", "_meta"))
    else:
        merged = pd.DataFrame()
    write_table(
        out_dir / "high_error_cell_qc_metadata.tsv",
        merged,
        "外部验证逐细胞平均绝对误差与 RNA 质控字段的合并原表。",
    )
    assoc_rows = []
    if not merged.empty:
        qc_cols = [c for c in ["nCount_RNA", "nFeature_RNA", "percent.mt", "n_counts", "n_genes", "mito_proportion", "ribo_proportion"] if c in merged.columns]
        for cohort, sub in merged.groupby("cohort_id"):
            for col in qc_cols:
                rho, pval, n, status = safe_spearman(pd.to_numeric(sub[col], errors="coerce"), sub["mean_abs_error"], min_n=20)
                assoc_rows.append({"cohort_id": cohort, "qc_metric": col, "spearman_with_mean_abs_error": rho, "p_value": pval, "n_cells": n, "status": status})
    write_table(
        out_dir / "high_error_qc_association.tsv",
        pd.DataFrame(assoc_rows),
        "平均绝对误差与细胞质控指标的 Spearman 相关。",
    )


def build_pseudobulk_status(meta: pd.DataFrame, out_dir: Path):
    group_cols = [c for c in ["dataset_id", "sample_key", "sample_id", "condition_label", "treatment", "ibrutinib", "cell_type_label"] if c in meta.columns]
    groups = meta[group_cols].fillna("NA").groupby(group_cols, dropna=False).size().reset_index(name="n_cells") if group_cols else pd.DataFrame()
    status = pd.DataFrame([
        {
            "item": "SC_to_pseudobulk_bulk_match",
            "status": "no_shared_bulk_phospho_measurement_found",
            "reason": "当前单细胞验证队列没有在结果包中提供同一样本的实测 bulk 磷酸化矩阵；可生成单细胞伪 bulk，但不能计算与 bulk 实测的一致性。",
            "candidate_group_table": str(out_dir / "sc_to_pseudobulk_candidate_groups.tsv"),
        }
    ])
    write_table(out_dir / "sc_to_pseudobulk_candidate_groups.tsv", groups, "可聚合成伪 bulk 的单细胞分组候选表。")
    write_table(out_dir / "sc_to_pseudobulk_bulk_match_status.tsv", status, "单细胞伪 bulk 与实测 bulk 磷酸化匹配状态。")


def copy_v1(v1: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if v1.exists():
        for p in v1.iterdir():
            if p.is_file():
                shutil.copy2(p, out_dir / p.name)


def build_source_index(out_dir: Path):
    rows = []
    for p in sorted(out_dir.glob("*.tsv")):
        try:
            df = pd.read_csv(p, sep="\t", nrows=5)
            n_cols = int(df.shape[1])
            cols = ";".join(df.columns.astype(str).tolist())
            with p.open("r", encoding="utf-8", errors="ignore") as fh:
                n_rows = max(sum(1 for _ in fh) - 1, 0)
            status = "available"
        except Exception as exc:
            n_cols = 0
            n_rows = 0
            cols = ""
            status = f"unreadable:{type(exc).__name__}"
        rows.append({
            "table_name": p.name,
            "status": status,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "columns": cols,
            "path": str(p),
        })
    df = pd.DataFrame(rows)
    write_table(out_dir / "reviewer_requested_source_table.tsv", df, "新版审稿补充表格索引。每一行是一份可追溯原表。")
    manifest = {
        "output_dir": str(out_dir),
        "n_tables": int(len(rows)),
        "tables": rows,
    }
    (out_dir / "TABLE_MANIFEST_V2.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    ap.add_argument("--input-dir", default=r"D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_gse300551_signal_seq_multidomain_v1")
    ap.add_argument("--result-dir", default=r"D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260522_scp682_sc11_expanded_scnet_site_gnn_v1")
    ap.add_argument("--output-dir", default=r"D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260522_scp682_sc11_expanded_scnet_site_gnn_v1\reviewer_requested_tables_v2")
    args = ap.parse_args()

    root = Path(args.root)
    input_dir = Path(args.input_dir)
    result_dir = Path(args.result_dir)
    out_dir = Path(args.output_dir)
    v1 = result_dir / "reviewer_requested_tables_v1"
    paired_root = root / "01_data" / "single_cell" / "intermediate" / "paired_matrices"

    copy_v1(v1, out_dir)
    meta, targets, y, mask = summarize_constant_baseline(input_dir, out_dir)
    build_cognate_mrna(input_dir, paired_root, meta, targets, y, mask, out_dir)
    build_high_error_qc(result_dir, meta, out_dir)
    build_pseudobulk_status(meta, out_dir)

    rerun = pd.DataFrame([
        {
            "item": "pathway_attention_removed_ablation",
            "status": "requires_model_rerun",
            "recommended_script": "train_scp682_sc11_expanded_scnet_site_gnn.py with fixed or disabled cross-attention branch",
            "notes": "当前正式训练脚本没有已完成的去通路注意力变体；不能从现有结果倒推。",
        },
        {
            "item": "scFoundation_removed_raw_expression_ablation",
            "status": "requires_model_rerun",
            "recommended_script": "raw RNA selected-gene input + same masked phospho target evaluation",
            "notes": "已找到原始 RNA 矩阵；需要单独训练原始表达输入模型。",
        },
    ])
    write_table(out_dir / "missing_ablation_rerun_status.tsv", rerun, "需要重训的消融项状态，不填充虚构数字。")
    build_source_index(out_dir)
    print(f"done\t{out_dir}", flush=True)


if __name__ == "__main__":
    main()
