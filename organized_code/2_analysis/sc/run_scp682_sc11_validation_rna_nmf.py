#!/usr/bin/env python
# 模型: SCP682-SC
# 作用: 对外部验证队列原始 RNA 矩阵做全基因组 NMF，并关联 hallmark 富集和预测误差。
# 输入: 远程 single_cell 原始/中间 RNA 矩阵、SC11 predicted-observed 表、MSigDB hallmark GMT。
# 输出: 每个验证集的 RNA NMF program、hallmark 富集、program 与预测误差关联、论文图源和成图。
# 依赖: pandas, numpy, scipy, scikit-learn, h5py, anndata, matplotlib, seaborn。
# 原始路径: remote_scripts/run_scp682_sc11_validation_rna_nmf.py
# 原始版本: 2026-05-28 RNA NMF v2

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.io import mmread
from scipy.stats import hypergeom, mannwhitneyu, spearmanr
from sklearn.decomposition import MiniBatchNMF

try:
    import anndata as ad
except Exception:
    ad = None


PROJECT = Path(os.environ.get("SCP682_PROJECT_ROOT", ".")).resolve()
SC_ROOT = PROJECT / "01_data" / "single_cell"
PAIR = SC_ROOT / "intermediate" / "paired_matrices"
SC11 = PROJECT / "02_results" / "single_cell" / "20260522_scp682_sc11_expanded_scnet_site_gnn_v1"
DEFAULT_OUT = PROJECT / "02_results" / "single_cell" / "20260528_scp682_sc11_validation_rna_nmf_v2"


COHORT_FILES = {
    "gse300551_iccite_plex_kinase_2025": SC11 / "tables" / "scp682_sc11_predicted_observed_gse300551_iccite_plex_kinase_2025.tsv",
    "phospho_seq_blair_2025_phospho_multi": SC11 / "tables" / "scp682_sc11_predicted_observed_phospho_seq_blair_2025_phospho_multi.tsv",
    "signal_seq_gse256403_hela_2024": SC11 / "tables" / "scp682_sc11_predicted_observed_signal_seq_gse256403_hela_2024.tsv",
    "signal_seq_gse256404_pdo_caf_2024": SC11 / "tables" / "scp682_sc11_predicted_observed_signal_seq_gse256404_pdo_caf_2024.tsv",
    "vivo_seq_th17_2025": SC11 / "tables" / "scp682_sc11_predicted_observed_vivo_seq_th17_2025.tsv",
}

DISPLAY = {
    "gse300551_iccite_plex_kinase_2025": "GSE300551",
    "phospho_seq_blair_2025_phospho_multi": "Blair",
    "signal_seq_gse256403_hela_2024": "SIGNAL-seq HeLa",
    "signal_seq_gse256404_pdo_caf_2024": "SIGNAL-seq PDO/CAF",
    "vivo_seq_th17_2025": "Vivo-seq Th17",
}

CMAP_SEQ = LinearSegmentedColormap.from_list("seq", ["#F7F7F7", "#C8D7D2", "#6CBFB5", "#1F3A5F"], N=256)
CMAP_DIV = LinearSegmentedColormap.from_list("div", ["#92B1D9", "#F7F7F7", "#D98973"], N=256)


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def snake_case(name: str) -> str:
    text = str(name)
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or "column"


def clean_gene_symbol(gene: str) -> str:
    text = str(gene).strip()
    for prefix in ["GRCh38_", "GRCH38_", "hg38_", "HG38_", "mm10_", "MM10_"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if "|" in text:
        text = text.split("|")[0]
    text = text.strip("_")
    return text.upper()


def read_lines(path: Path) -> list[str]:
    return [line.rstrip("\n\r") for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]


def write_table(df: pd.DataFrame, path: Path, desc: str) -> None:
    out = df.copy()
    out.columns = [snake_case(c) for c in out.columns]
    out = out.replace([np.inf, -np.inf], np.nan).fillna("NA")
    out.to_csv(path, sep="\t", index=False, na_rep="NA")
    path.with_suffix(".md").write_text(desc + "\n", encoding="utf-8")


def parse_gmt(path: Path) -> dict[str, set[str]]:
    gene_sets: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gene_sets[parts[0]] = {clean_gene_symbol(x) for x in parts[2:] if x}
    return gene_sets


def bh_fdr(pvalues: list[float]) -> list[float]:
    p = np.array([1.0 if not np.isfinite(x) else float(x) for x in pvalues], dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    prev = 1.0
    n = len(p)
    for rank, idx in enumerate(order[::-1], start=1):
        original_rank = n - rank + 1
        val = min(prev, p[idx] * n / original_rank)
        out[idx] = val
        prev = val
    return out.tolist()


def build_error_table(cohort_id: str) -> pd.DataFrame:
    path = COHORT_FILES[cohort_id]
    df = pd.read_csv(path, sep="\t", usecols=["cell_id", "target_id", "predicted_raw_scale", "observed_raw_scale"])
    df["predicted_raw_scale"] = pd.to_numeric(df["predicted_raw_scale"], errors="coerce")
    df["observed_raw_scale"] = pd.to_numeric(df["observed_raw_scale"], errors="coerce")
    df["abs_raw_error"] = (df["predicted_raw_scale"] - df["observed_raw_scale"]).abs()
    z_parts = []
    for target, sub in df.groupby("target_id", sort=False):
        obs = sub["observed_raw_scale"].astype(float)
        pred = sub["predicted_raw_scale"].astype(float)
        obs_sd = obs.std(ddof=0)
        pred_sd = pred.std(ddof=0)
        if not np.isfinite(obs_sd) or obs_sd <= 1e-12 or not np.isfinite(pred_sd) or pred_sd <= 1e-12:
            continue
        tmp = sub[["cell_id", "target_id"]].copy()
        tmp["abs_z_error"] = (((pred - pred.mean()) / pred_sd) - ((obs - obs.mean()) / obs_sd)).abs().to_numpy()
        z_parts.append(tmp)
    raw = df.groupby("cell_id", as_index=False).agg(
        mean_abs_raw_error=("abs_raw_error", "mean"),
        n_error_targets_raw=("target_id", "nunique"),
    )
    if z_parts:
        zdf = pd.concat(z_parts, ignore_index=True)
        zsum = zdf.groupby("cell_id", as_index=False).agg(
            mean_abs_z_error=("abs_z_error", "mean"),
            n_error_targets_z=("target_id", "nunique"),
        )
        out = raw.merge(zsum, on="cell_id", how="left")
    else:
        out = raw
        out["mean_abs_z_error"] = np.nan
        out["n_error_targets_z"] = 0
    return out


def read_10x_h5_selected(h5_path: Path, plate: str, keep_ids: set[str]) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    with h5py.File(h5_path, "r") as handle:
        g = handle["matrix"]
        barcodes = [x.decode() if isinstance(x, bytes) else str(x) for x in g["barcodes"][:]]
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in g["features"]["name"][:]]
        shape = tuple(int(x) for x in g["shape"][:])
        mat = sparse.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)
    full_ids = [f"{plate}_{bc}" for bc in barcodes]
    idx = [i for i, cid in enumerate(full_ids) if cid in keep_ids]
    if not idx:
        return sparse.csr_matrix((0, len(names)), dtype=np.float32), [], names
    sub = mat[:, idx].T.tocsr().astype(np.float32)
    return sub, [full_ids[i] for i in idx], names


def load_gse300551(error_ids: set[str]) -> tuple[sparse.csr_matrix, list[str], list[str], str]:
    paired = PAIR / "gse300551_iccite_plex_kinase_2025"
    manifest = pd.read_csv(paired / "rna_h5_manifest.tsv", sep="\t")
    mats = []
    ids = []
    genes_ref = None
    for _, row in manifest.iterrows():
        plate = str(row["plate"])
        mat, sub_ids, genes = read_10x_h5_selected(Path(row["rna_h5"]), plate, error_ids)
        if genes_ref is None:
            genes_ref = genes
        elif genes != genes_ref:
            raise RuntimeError("GSE300551 10x h5 gene order differs across plates")
        if mat.shape[0] > 0:
            mats.append(mat)
            ids.extend(sub_ids)
    if not mats:
        raise RuntimeError("GSE300551 no matched RNA cells")
    return sparse.vstack(mats, format="csr"), ids, list(genes_ref), "five_10x_h5"


def load_h5ad_selected(cohort_id: str, error_ids: set[str]) -> tuple[sparse.csr_matrix, list[str], list[str], str]:
    if ad is None:
        raise RuntimeError("anndata is not available")
    manifest = pd.read_csv(PAIR / "signal_seq_gse256405_processed_h5ad_v1" / "rna_adt_h5ad_manifest.tsv", sep="\t")
    row = manifest[manifest["dataset_id"] == cohort_id].iloc[0]
    path = Path(row["rna_h5ad"])
    adata = ad.read_h5ad(path)
    obs_names = [str(x) for x in adata.obs_names]
    obs_df = adata.obs
    barcode_values = obs_df["barcode_seq"].astype(str).tolist() if "barcode_seq" in obs_df.columns else obs_names
    suffix_values = [None] * len(obs_names)
    if "sub_lib_id" in obs_df.columns:
        suffix_values = []
        for value in obs_df["sub_lib_id"].astype(str).tolist():
            match = re.search(r"_(\d+)$", value)
            suffix_values.append(match.group(1) if match else None)

    full_ids = []
    for obs_name, barcode, suffix in zip(obs_names, barcode_values, suffix_values):
        bases = []
        for base in [obs_name, barcode]:
            if base and base not in bases and base.lower() != "nan":
                bases.append(base)
        candidates = []
        for base in bases:
            candidates.extend([base, f"{cohort_id}:{base}"])
            if suffix:
                candidates.extend([f"{base}_{suffix}", f"{cohort_id}:{base}_{suffix}"])
        matched = next((cid for cid in candidates if cid in error_ids), None)
        full_ids.append(matched)
    idx = [i for i, cid in enumerate(full_ids) if cid is not None]
    if not idx:
        raise RuntimeError(f"{cohort_id} no matched h5ad cells")
    if "counts" in adata.layers:
        x = adata.layers["counts"][idx, :]
        matrix_slot = "layers/counts"
        genes = [str(g) for g in adata.var["Gene_Name"]] if "Gene_Name" in adata.var.columns else [str(g) for g in adata.var_names]
    elif adata.raw is not None:
        x = adata.raw.X[idx, :]
        matrix_slot = "raw.X"
        genes = [str(g) for g in adata.raw.var_names]
    else:
        x = adata.X[idx, :]
        matrix_slot = "X"
        genes = [str(g) for g in adata.var["Gene_Name"]] if "Gene_Name" in adata.var.columns else [str(g) for g in adata.var_names]
    if not sparse.issparse(x):
        x = sparse.csr_matrix(np.asarray(x))
    x = x.tocsr().astype(np.float32)
    sub_ids = [str(full_ids[i]) for i in idx]
    del adata
    gc.collect()
    return x, sub_ids, genes, f"{path}::{matrix_slot}"


def load_vivo(error_ids: set[str]) -> tuple[sparse.csr_matrix, list[str], list[str], str]:
    paired = PAIR / "vivo_seq_th17_2025"
    x = mmread(paired / "rna_counts.mtx").tocsr().astype(np.float32)
    genes = read_lines(paired / "genes.tsv")
    barcodes = read_lines(paired / "barcodes.tsv")
    if x.shape[0] != len(barcodes):
        x = x.T.tocsr()
    idx = [i for i, cid in enumerate(barcodes) if cid in error_ids]
    return x[idx, :].tocsr(), [barcodes[i] for i in idx], genes, "paired_matrices/vivo_seq_th17_2025/rna_counts.mtx"


def load_blair(error_ids: set[str]) -> tuple[sparse.csr_matrix, list[str], list[str], str]:
    path = PAIR / "phospho_seq_blair_2025_phospho_multi" / "rna_counts.tsv"
    df = pd.read_csv(path, sep="\t")
    genes = df.iloc[:, 0].astype(str).tolist()
    cols = [str(c) for c in df.columns[1:]]
    keep = [i for i, cid in enumerate(cols) if cid in error_ids]
    sub = df.iloc[:, [0] + [i + 1 for i in keep]]
    mat = sparse.csr_matrix(sub.iloc[:, 1:].T.to_numpy(dtype=np.float32))
    ids = [cols[i] for i in keep]
    del df, sub
    gc.collect()
    return mat, ids, genes, "paired_matrices/phospho_seq_blair_2025_phospho_multi/rna_counts.tsv"


def load_rna_matrix(cohort_id: str, error_ids: set[str]) -> tuple[sparse.csr_matrix, list[str], list[str], str]:
    if cohort_id == "gse300551_iccite_plex_kinase_2025":
        return load_gse300551(error_ids)
    if cohort_id in {"signal_seq_gse256403_hela_2024", "signal_seq_gse256404_pdo_caf_2024"}:
        return load_h5ad_selected(cohort_id, error_ids)
    if cohort_id == "vivo_seq_th17_2025":
        return load_vivo(error_ids)
    if cohort_id == "phospho_seq_blair_2025_phospho_multi":
        return load_blair(error_ids)
    raise KeyError(cohort_id)


def preprocess_for_nmf(x: sparse.csr_matrix, genes: list[str], min_cells: int) -> tuple[sparse.csr_matrix, list[str], dict, np.ndarray]:
    x = x.tocsr().astype(np.float32)
    x.eliminate_zeros()
    if x.data.size and np.nanmin(x.data) < 0:
        raise RuntimeError("RNA NMF input contains negative values; raw counts layer is required")
    if x.data.size and not np.isfinite(x.data).all():
        raise RuntimeError("RNA NMF input contains non-finite values")
    detected = np.asarray((x > 0).sum(axis=0)).ravel()
    keep = detected >= min_cells
    x = x[:, keep].tocsr()
    kept_genes = [g for g, ok in zip(genes, keep) if ok]
    lib = np.asarray(x.sum(axis=1)).ravel()
    ok_cells = lib > 0
    if not np.all(ok_cells):
        x = x[ok_cells, :].tocsr()
        lib = lib[ok_cells]
    scale = np.divide(1e4, lib, out=np.zeros_like(lib, dtype=np.float32), where=lib > 0)
    x = sparse.diags(scale).dot(x).tocsr()
    x.data = np.log1p(x.data).astype(np.float32)
    x.eliminate_zeros()
    meta = {
        "n_cells_after_library_filter": int(x.shape[0]),
        "n_genes_after_min_cells": int(x.shape[1]),
        "min_cells": int(min_cells),
    }
    return x, kept_genes, meta, np.asarray(ok_cells, dtype=bool)


def fit_nmf(x: sparse.csr_matrix, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, float, int]:
    model = MiniBatchNMF(
        n_components=k,
        init="nndsvda",
        random_state=seed,
        batch_size=4096,
        max_iter=500,
        tol=1e-4,
        beta_loss="frobenius",
    )
    w = model.fit_transform(x)
    h = model.components_.astype(np.float32)
    denom = float(sparse.linalg.norm(x))
    rel_error = float(model.reconstruction_err_ / denom) if denom > 0 else np.nan
    return w.astype(np.float32), h, rel_error, int(model.n_iter_)


def top_unique_genes(genes: list[str], loadings: np.ndarray, top_n: int = 200) -> list[tuple[str, float]]:
    order = np.argsort(-loadings)
    out = []
    seen = set()
    for idx in order:
        symbol = clean_gene_symbol(genes[idx])
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append((symbol, float(loadings[idx])))
        if len(out) >= top_n:
            break
    return out


def enrich_program(
    cohort_id: str,
    k_label: str,
    component: str,
    query: list[str],
    universe: set[str],
    gene_sets: dict[str, set[str]],
) -> list[dict]:
    qset = set(query) & universe
    rows = []
    m = len(universe)
    n = len(qset)
    for term, genes in gene_sets.items():
        g = genes & universe
        if not g:
            continue
        overlap = sorted(qset & g)
        k = len(overlap)
        p = float(hypergeom.sf(k - 1, m, len(g), n)) if k > 0 else 1.0
        fold = (k / n) / (len(g) / m) if n > 0 and len(g) > 0 else np.nan
        rows.append(
            {
                "cohort_id": cohort_id,
                "display_cohort": DISPLAY.get(cohort_id, cohort_id),
                "k_label": k_label,
                "component": component,
                "gene_set": term,
                "universe_size": m,
                "query_size": n,
                "gene_set_size": len(g),
                "overlap_count": k,
                "fold_enrichment": fold,
                "p_value": p,
                "overlap_genes": ";".join(overlap[:80]),
            }
        )
    qvals = bh_fdr([r["p_value"] for r in rows])
    for r, q in zip(rows, qvals):
        r["q_value"] = q
        r["neg_log10_q"] = -math.log10(max(q, 1e-300))
    return rows


def choose_min_cells(n_cells: int) -> int:
    if n_cells < 2000:
        return 5
    if n_cells < 10000:
        return 10
    return 20


def choose_k_values(n_cells: int, requested: str) -> list[int]:
    if requested == "both":
        return [10, 30]
    if requested in {"10", "30"}:
        return [int(requested)]
    return [10] if n_cells < 3000 else [30]


def process_cohort(cohort_id: str, gene_sets: dict[str, set[str]], out: Path, requested_k: str, seed: int) -> dict:
    log(f"{cohort_id}: build error table")
    err = build_error_table(cohort_id)
    error_ids = set(err["cell_id"].astype(str))
    log(f"{cohort_id}: load RNA matrix")
    x, cell_ids, genes, source = load_rna_matrix(cohort_id, error_ids)
    if x.shape[0] == 0:
        raise RuntimeError(f"{cohort_id}: no matched cells")
    cell_df = pd.DataFrame({"cell_id": cell_ids})
    cell_df = cell_df.merge(err, on="cell_id", how="left")
    min_cells = choose_min_cells(x.shape[0])
    log(f"{cohort_id}: preprocess RNA, raw shape={x.shape}, min_cells={min_cells}")
    x, kept_genes, prep, keep_cells = preprocess_for_nmf(x, genes, min_cells=min_cells)
    if x.shape[0] != len(cell_df):
        cell_df = cell_df.loc[keep_cells].reset_index(drop=True)
    clean_universe = {clean_gene_symbol(g) for g in kept_genes if clean_gene_symbol(g)}
    k_values = choose_k_values(x.shape[0], requested_k)

    summary_rows = []
    program_rows = []
    top_gene_rows = []
    enrichment_rows = []
    association_rows = []
    score_tables = []
    high_error_cut = float(cell_df["mean_abs_z_error"].quantile(0.75)) if cell_df["mean_abs_z_error"].notna().any() else np.nan
    high_error = cell_df["mean_abs_z_error"] >= high_error_cut if np.isfinite(high_error_cut) else pd.Series(False, index=cell_df.index)

    for k in k_values:
        k_label = f"k{k}"
        log(f"{cohort_id}: fit RNA NMF {k_label}")
        w, h, rel_error, n_iter = fit_nmf(x, k=k, seed=seed + k)
        score = cell_df[["cell_id", "mean_abs_z_error", "mean_abs_raw_error", "n_error_targets_z", "n_error_targets_raw"]].copy()
        score.insert(1, "cohort_id", cohort_id)
        score.insert(2, "display_cohort", DISPLAY.get(cohort_id, cohort_id))
        score.insert(3, "k_label", k_label)
        for comp_idx in range(k):
            comp = f"rna_nmf{comp_idx + 1:02d}"
            score[comp] = w[:, comp_idx]
            loadings = h[comp_idx, :]
            top = top_unique_genes(kept_genes, loadings, top_n=200)
            top_symbols = [g for g, _ in top]
            top_gene_rows.extend(
                {
                    "cohort_id": cohort_id,
                    "display_cohort": DISPLAY.get(cohort_id, cohort_id),
                    "k_label": k_label,
                    "component": comp,
                    "rank": rank + 1,
                    "gene_symbol": gene,
                    "loading": loading,
                }
                for rank, (gene, loading) in enumerate(top)
            )
            program_rows.append(
                {
                    "cohort_id": cohort_id,
                    "display_cohort": DISPLAY.get(cohort_id, cohort_id),
                    "k_label": k_label,
                    "component": comp,
                    "top_20_genes": ";".join(top_symbols[:20]),
                    "component_loading_sum": float(loadings.sum()),
                    "component_loading_max": float(loadings.max()),
                }
            )
            enrichment_rows.extend(enrich_program(cohort_id, k_label, comp, top_symbols, clean_universe, gene_sets))

            rho_z = spearmanr(w[:, comp_idx], cell_df["mean_abs_z_error"], nan_policy="omit").statistic
            rho_raw = spearmanr(w[:, comp_idx], cell_df["mean_abs_raw_error"], nan_policy="omit").statistic
            p_u = np.nan
            delta_high = np.nan
            if high_error.sum() > 0 and (~high_error).sum() > 0:
                hi = w[high_error.to_numpy(), comp_idx]
                lo = w[(~high_error).to_numpy(), comp_idx]
                try:
                    p_u = float(mannwhitneyu(hi, lo, alternative="two-sided").pvalue)
                except Exception:
                    p_u = np.nan
                delta_high = float(np.median(hi) - np.median(lo))
            association_rows.append(
                {
                    "cohort_id": cohort_id,
                    "display_cohort": DISPLAY.get(cohort_id, cohort_id),
                    "k_label": k_label,
                    "component": comp,
                    "spearman_program_vs_z_error": float(rho_z) if np.isfinite(rho_z) else np.nan,
                    "spearman_program_vs_raw_error": float(rho_raw) if np.isfinite(rho_raw) else np.nan,
                    "high_error_q75_cutoff": high_error_cut,
                    "median_score_high_error_minus_other": delta_high,
                    "mannwhitney_p_value": p_u,
                    "top_20_genes": ";".join(top_symbols[:20]),
                }
            )
        score_tables.append(score)
        summary_rows.append(
            {
                "cohort_id": cohort_id,
                "display_cohort": DISPLAY.get(cohort_id, cohort_id),
                "k_label": k_label,
                "n_cells": int(x.shape[0]),
                "n_genes_raw": int(len(genes)),
                "n_genes_used": int(len(kept_genes)),
                "min_cells": int(min_cells),
                "relative_reconstruction_error": rel_error,
                "n_iter": n_iter,
                "rna_source": source,
                "matrix_transform": "normalize_total_1e4_log1p",
                "gene_filter": "all_detected_genes_min_cells",
            }
        )
    score_df = pd.concat(score_tables, ignore_index=True, sort=False)
    cohort_safe = snake_case(cohort_id)
    score_path = out / "source_data" / f"{cohort_safe}_rna_nmf_program_scores.tsv.gz"
    score_df.to_csv(score_path, sep="\t", index=False, compression="gzip", na_rep="NA")
    score_path.with_suffix(score_path.suffix + ".md").write_text("每个细胞的 RNA NMF program 得分与 SCP682-SC 预测误差。\n", encoding="utf-8")
    return {
        "summary": summary_rows,
        "programs": program_rows,
        "top_genes": top_gene_rows,
        "enrichment": enrichment_rows,
        "association": association_rows,
        "score_file": str(score_path),
    }


def draw_error_association(assoc: pd.DataFrame, fig_dir: Path) -> None:
    sub = assoc[assoc["k_label"].isin(["k30", "k10"])].copy()
    sub["abs_rho"] = sub["spearman_program_vs_z_error"].abs()
    top = sub.sort_values("abs_rho", ascending=False).groupby("cohort_id").head(8)
    top = top.sort_values(["display_cohort", "spearman_program_vs_z_error"])
    fig, ax = plt.subplots(figsize=(8.2, max(3.2, 0.22 * len(top))))
    labels = top["display_cohort"] + " " + top["k_label"] + " " + top["component"]
    colors = ["#D98973" if v > 0 else "#92B1D9" for v in top["spearman_program_vs_z_error"]]
    ax.barh(np.arange(len(top)), top["spearman_program_vs_z_error"], color=colors)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color="#606060", lw=0.7)
    ax.set_xlabel("Spearman(program score, prediction error)")
    ax.set_title("RNA NMF programs associated with prediction error")
    fig.tight_layout()
    fig.savefig(fig_dir / "rna_nmf_error_association_barplot.png", dpi=350, bbox_inches="tight")
    fig.savefig(fig_dir / "rna_nmf_error_association_barplot.svg", bbox_inches="tight")
    fig.savefig(fig_dir / "rna_nmf_error_association_barplot.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_enrichment_dotplot(enrich: pd.DataFrame, fig_dir: Path) -> None:
    sig = enrich.copy()
    sig = sig[sig["overlap_count"] > 0]
    sig = sig.sort_values(["q_value", "p_value"]).groupby(["cohort_id", "k_label", "component"]).head(1)
    sig = sig.sort_values("neg_log10_q", ascending=False).head(35)
    if sig.empty:
        return
    sig["program"] = sig["display_cohort"] + " " + sig["k_label"] + " " + sig["component"]
    sig["hallmark"] = sig["gene_set"].str.replace("HALLMARK_", "", regex=False).str.replace("_", " ", regex=False)
    fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.25 * sig["program"].nunique())))
    x_levels = list(dict.fromkeys(sig["hallmark"]))
    y_levels = list(dict.fromkeys(sig["program"]))
    x = [x_levels.index(v) for v in sig["hallmark"]]
    y = [y_levels.index(v) for v in sig["program"]]
    sizes = 18 + 7 * sig["overlap_count"].clip(upper=30)
    sc = ax.scatter(x, y, s=sizes, c=sig["neg_log10_q"], cmap=CMAP_SEQ, edgecolor="#333333", lw=0.25)
    ax.set_xticks(range(len(x_levels)))
    ax.set_xticklabels(x_levels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_levels)))
    ax.set_yticklabels(y_levels)
    ax.set_title("Top hallmark enrichment for RNA NMF programs")
    ax.set_xlabel("")
    ax.set_ylabel("")
    cb = fig.colorbar(sc, ax=ax, shrink=0.72)
    cb.set_label("-log10(q)")
    fig.tight_layout()
    fig.savefig(fig_dir / "rna_nmf_hallmark_enrichment_dotplot.png", dpi=350, bbox_inches="tight")
    fig.savefig(fig_dir / "rna_nmf_hallmark_enrichment_dotplot.svg", bbox_inches="tight")
    fig.savefig(fig_dir / "rna_nmf_hallmark_enrichment_dotplot.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_cohort_summary(summary: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    sub = summary.copy()
    sub["label"] = sub["display_cohort"] + "\n" + sub["k_label"]
    ax.bar(np.arange(len(sub)), sub["n_genes_used"], color="#6CBFB5")
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(sub["label"], rotation=30, ha="right")
    ax.set_ylabel("genes used in RNA NMF")
    ax.set_title("RNA NMF input scale")
    for i, row in sub.iterrows():
        ax.text(i, row["n_genes_used"], f"n={int(row['n_cells'])}", ha="center", va="bottom", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "rna_nmf_input_summary.png", dpi=350, bbox_inches="tight")
    fig.savefig(fig_dir / "rna_nmf_input_summary.svg", bbox_inches="tight")
    fig.savefig(fig_dir / "rna_nmf_input_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hallmark-gmt", type=Path, required=True)
    parser.add_argument("--cohorts", default=",".join(COHORT_FILES.keys()))
    parser.add_argument("--k", choices=["auto", "10", "30", "both"], default="auto")
    parser.add_argument("--seed", type=int, default=682)
    args = parser.parse_args()

    out = args.out_dir
    source = out / "source_data"
    fig = out / "figures"
    logs = out / "logs"
    for folder in [out, source, fig, logs]:
        folder.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    gene_sets = parse_gmt(args.hallmark_gmt)
    log(f"loaded hallmark gene sets: {len(gene_sets)}")
    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    all_summary = []
    all_programs = []
    all_top_genes = []
    all_enrichment = []
    all_association = []
    score_files = []
    failures = []
    for cohort_id in cohorts:
        try:
            result = process_cohort(cohort_id, gene_sets, out, args.k, args.seed)
            all_summary.extend(result["summary"])
            all_programs.extend(result["programs"])
            all_top_genes.extend(result["top_genes"])
            all_enrichment.extend(result["enrichment"])
            all_association.extend(result["association"])
            score_files.append({"cohort_id": cohort_id, "score_file": result["score_file"]})
        except Exception as exc:
            log(f"{cohort_id}: failed: {exc}")
            failures.append({"cohort_id": cohort_id, "error": str(exc)})
        gc.collect()

    summary = pd.DataFrame(all_summary)
    programs = pd.DataFrame(all_programs)
    top_genes = pd.DataFrame(all_top_genes)
    enrichment = pd.DataFrame(all_enrichment)
    association = pd.DataFrame(all_association)
    if not association.empty:
        association["mannwhitney_q_value"] = bh_fdr(association["mannwhitney_p_value"].fillna(1).astype(float).tolist())

    write_table(summary, source / "rna_nmf_summary.tsv", "每个验证集 RNA NMF 的输入规模、分解维度、基因过滤和重构误差。")
    write_table(programs, source / "rna_nmf_program_top20.tsv", "每个 RNA NMF program 的前 20 个载荷基因。")
    write_table(top_genes, source / "rna_nmf_program_top200_genes.tsv", "每个 RNA NMF program 的前 200 个载荷基因，用于 hallmark 富集。")
    write_table(enrichment, source / "rna_nmf_hallmark_enrichment.tsv", "RNA NMF program 前 200 基因对 MSigDB hallmark gene sets 的超几何富集。")
    write_table(association, source / "rna_nmf_program_error_association.tsv", "每个 RNA NMF program 得分和 SCP682-SC 预测误差的关联。")
    write_table(pd.DataFrame(score_files), source / "rna_nmf_program_score_files.tsv", "每个验证集每细胞 program 得分表的位置。")
    write_table(pd.DataFrame(failures), source / "rna_nmf_failures.tsv", "运行失败的队列及原因；空表表示无失败。")

    if not summary.empty:
        draw_cohort_summary(summary, fig)
    if not association.empty:
        draw_error_association(association, fig)
    if not enrichment.empty:
        draw_enrichment_dotplot(enrichment, fig)

    manifest = {
        "out_dir": str(out),
        "hallmark_gmt": str(args.hallmark_gmt),
        "cohorts": cohorts,
        "k": args.k,
        "score_files": score_files,
        "failures": failures,
        "notes": [
            "NMF 输入来自原始 RNA cells × genes 矩阵；不使用 scFoundation、Geneformer 或磷酸化靶点子集。",
            "NMF 前只做测序深度归一化到 1e4 和 log1p；基因层面保留所有达到最小检出细胞数的基因，不做 HVG 筛选。",
            "hallmark 富集使用每个 program 载荷最高的 200 个基因，背景为该队列 NMF 实际使用的表达基因。",
            "预测误差使用 SC11 predicted-observed 表，主关联指标为每细胞 mean_abs_z_error。",
        ],
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "done.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log("done")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
