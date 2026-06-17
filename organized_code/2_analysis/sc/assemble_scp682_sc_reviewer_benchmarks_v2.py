#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


RESULT_DIR = Path(r"D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260522_scp682_sc11_expanded_scnet_site_gnn_v1")
V2_DIR = RESULT_DIR / "reviewer_requested_tables_v2"
RIDGE_DIR = Path(r"D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260531_scp682_sc_reviewer_scfoundation_ridge_baseline_v1")
PAPER_LOCAL_LABEL = "paper_materials_SCP682_SC11"


COHORT_NAMES = {
    "gse300551_iccite_plex_kinase_2025": "GSE300551",
    "signal_seq_gse256403_hela_2024": "SIGNAL-seq HeLa",
    "signal_seq_gse256404_pdo_caf_2024": "SIGNAL-seq PDO/CAF",
    "phospho_seq_blair_2025_phospho_multi": "Blair",
    "vivo_seq_th17_2025": "Vivo-seq Th17",
    "iccite_seq_tcell_2025": "icCITE",
    "qurie_seq_bjab_2021": "QuRIE",
}


def finite_median(x):
    vals = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    return float(np.median(vals)) if len(vals) else math.nan


def write(path: Path, df: pd.DataFrame, note: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, na_rep="NA")
    path.with_suffix(".md").write_text(note + "\n", encoding="utf-8")


def load_external_model():
    p = V2_DIR / "external_per_target_from_predicted_observed.tsv"
    df = pd.read_csv(p, sep="\t")
    out = pd.DataFrame({
        "evaluation_scope": "external_validation",
        "method_name": "SCP682-SC",
        "cohort_id": df["cohort_id"].astype(str),
        "cohort_name": df["cohort_id"].astype(str).map(COHORT_NAMES).fillna(df["cohort_id"].astype(str)),
        "target_id": df["target_id"].astype(str),
        "n_cells": pd.to_numeric(df["sample_size_used"], errors="coerce"),
        "spearman": pd.to_numeric(df["spearman"], errors="coerce"),
        "p_value": pd.to_numeric(df["spearman_pvalue"], errors="coerce"),
        "mae": np.nan,
        "rmse": np.nan,
        "source_file": str(p),
        "notes": "正式模型外部验证逐读数 Spearman",
    })
    return out


def load_mean():
    p = V2_DIR / "mean_baseline_per_target.tsv"
    df = pd.read_csv(p, sep="\t")
    df = df[~df["dataset_id"].astype(str).isin(["iccite_seq_tcell_2025", "qurie_seq_bjab_2021"])].copy()
    out = pd.DataFrame({
        "evaluation_scope": "external_validation",
        "method_name": "train_target_mean",
        "cohort_id": df["dataset_id"].astype(str),
        "cohort_name": df["cohort_name"].astype(str),
        "target_id": df["target_id"].astype(str),
        "n_cells": pd.to_numeric(df["n_cells"], errors="coerce"),
        "spearman": pd.to_numeric(df["spearman"], errors="coerce"),
        "p_value": pd.to_numeric(df["spearman_pvalue"], errors="coerce"),
        "mae": pd.to_numeric(df["mae"], errors="coerce"),
        "rmse": pd.to_numeric(df["rmse"], errors="coerce"),
        "source_file": str(p),
        "notes": "训练集位点均值常数基线；逐细胞 Spearman 对常数预测不适用",
    })
    return out


def load_cognate():
    p = V2_DIR / "cognate_mRNA_per_target.tsv"
    df = pd.read_csv(p, sep="\t")
    df = df[~df["dataset_id"].astype(str).isin(["qurie_seq_bjab_2021"])].copy()
    out = pd.DataFrame({
        "evaluation_scope": "external_validation",
        "method_name": "cognate_mRNA",
        "cohort_id": df["dataset_id"].astype(str),
        "cohort_name": df["cohort_name"].astype(str),
        "target_id": df["target_id"].astype(str),
        "n_cells": pd.to_numeric(df["n_cells"], errors="coerce"),
        "spearman": pd.to_numeric(df["spearman"], errors="coerce"),
        "p_value": pd.to_numeric(df["spearman_pvalue"], errors="coerce"),
        "mae": np.nan,
        "rmse": np.nan,
        "source_file": str(p),
        "notes": "同源 mRNA 与磷酸化读数的逐细胞 Spearman",
    })
    return out


def load_ridge():
    src = RIDGE_DIR / "tables" / "persite_ridge_performance.tsv"
    dst = V2_DIR / "scfoundation_persite_ridge_external_per_target.tsv"
    if src.exists():
        shutil.copy2(src, dst)
    summary = RIDGE_DIR / "tables" / "external_summary_by_dataset.tsv"
    if summary.exists():
        shutil.copy2(summary, V2_DIR / "scfoundation_persite_ridge_external_summary.tsv")
    df = pd.read_csv(src, sep="\t")
    out = pd.DataFrame({
        "evaluation_scope": "external_validation",
        "method_name": "scFoundation_persite_ridge",
        "cohort_id": df["test_dataset"].astype(str),
        "cohort_name": df["test_dataset"].astype(str).map(COHORT_NAMES).fillna(df["test_dataset"].astype(str)),
        "target_id": df["target_id"].astype(str),
        "n_cells": pd.to_numeric(df["n_test"], errors="coerce"),
        "spearman": pd.to_numeric(df["spearman"], errors="coerce"),
        "p_value": np.nan,
        "mae": np.nan,
        "rmse": pd.to_numeric(df["rmse"], errors="coerce") if "rmse" in df.columns else np.nan,
        "source_file": str(src),
        "notes": "同一训练集 icCITE+QuRIE 下逐位点 ridge 基线，特征为 scFoundation 细胞 embedding",
    })
    return out


def load_internal_mlp_and_model():
    rows = []
    model_p = V2_DIR / "scp682_sc11_formal_internal_5fold_per_target.tsv"
    if model_p.exists():
        m = pd.read_csv(model_p, sep="\t")
        fold_col = "fold" if "fold" in m.columns else None
        test_col = "test_dataset" if "test_dataset" in m.columns else ("dataset_id" if "dataset_id" in m.columns else None)
        for _, r in m.iterrows():
            rows.append({
                "evaluation_scope": "internal_5fold",
                "method_name": "SCP682-SC",
                "cohort_id": str(r.get(test_col, "internal")) if test_col else "internal",
                "cohort_name": COHORT_NAMES.get(str(r.get(test_col, "internal")), str(r.get(test_col, "internal"))) if test_col else "internal",
                "target_id": str(r.get("target_id", "")),
                "n_cells": pd.to_numeric(r.get("n", r.get("sample_size_used", np.nan)), errors="coerce"),
                "spearman": pd.to_numeric(r.get("spearman", np.nan), errors="coerce"),
                "p_value": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "source_file": str(model_p),
                "notes": f"fold={r.get(fold_col, 'NA')}" if fold_col else "internal 5fold",
            })
    mlp_p = V2_DIR / "internal_5fold_scfoundation_site_aware_mlp_per_target.tsv"
    if mlp_p.exists():
        d = pd.read_csv(mlp_p, sep="\t")
        for _, r in d.iterrows():
            rows.append({
                "evaluation_scope": "internal_5fold",
                "method_name": "scFoundation_site_aware_MLP",
                "cohort_id": str(r.get("test_dataset", r.get("dataset_id", "internal"))),
                "cohort_name": COHORT_NAMES.get(str(r.get("test_dataset", r.get("dataset_id", "internal"))), str(r.get("test_dataset", r.get("dataset_id", "internal")))),
                "target_id": str(r.get("target_id", "")),
                "n_cells": pd.to_numeric(r.get("n", r.get("sample_size_used", np.nan)), errors="coerce"),
                "spearman": pd.to_numeric(r.get("spearman", np.nan), errors="coerce"),
                "p_value": np.nan,
                "mae": np.nan,
                "rmse": pd.to_numeric(r.get("rmse", np.nan), errors="coerce"),
                "source_file": str(mlp_p),
                "notes": f"fold={r.get('fold', 'NA')}",
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    rows = []
    for keys, sub in df.groupby(["evaluation_scope", "method_name", "cohort_id", "cohort_name"], dropna=False):
        vals = pd.to_numeric(sub["spearman"], errors="coerce")
        rows.append({
            "evaluation_scope": keys[0],
            "method_name": keys[1],
            "cohort_id": keys[2],
            "cohort_name": keys[3],
            "n_targets_with_spearman": int(vals.notna().sum()),
            "median_spearman": finite_median(vals),
            "mean_spearman": float(vals.dropna().mean()) if vals.notna().any() else math.nan,
            "min_spearman": float(vals.dropna().min()) if vals.notna().any() else math.nan,
            "max_spearman": float(vals.dropna().max()) if vals.notna().any() else math.nan,
            "n_rows": int(len(sub)),
        })
    return pd.DataFrame(rows)


def paired_wilcoxon(df: pd.DataFrame):
    rows = []
    model = df[df["method_name"].eq("SCP682-SC") & df["evaluation_scope"].eq("external_validation")]
    for method, sub in df[df["evaluation_scope"].eq("external_validation")].groupby("method_name"):
        if method == "SCP682-SC":
            continue
        merged = model.merge(sub, on=["cohort_id", "target_id"], suffixes=("_model", "_baseline"))
        vals_m = pd.to_numeric(merged["spearman_model"], errors="coerce")
        vals_b = pd.to_numeric(merged["spearman_baseline"], errors="coerce")
        ok = vals_m.notna() & vals_b.notna()
        n = int(ok.sum())
        if n >= 2:
            diff = (vals_m[ok] - vals_b[ok]).to_numpy(float)
            try:
                p = float(wilcoxon(diff).pvalue)
            except Exception:
                p = math.nan
            delta = float(np.median(diff))
        else:
            p = math.nan
            delta = math.nan
        rows.append({
            "baseline_method": method,
            "n_paired_targets": n,
            "median_delta_spearman_model_minus_baseline": delta,
            "paired_wilcoxon_p_value": p,
        })
    return pd.DataFrame(rows)


def rebuild_source_index():
    rows = []
    for p in sorted(V2_DIR.glob("*.tsv")):
        try:
            head = pd.read_csv(p, sep="\t", nrows=5)
            with p.open("r", encoding="utf-8", errors="ignore") as fh:
                n_rows = max(sum(1 for _ in fh) - 1, 0)
            rows.append({"table_name": p.name, "status": "available", "n_rows": n_rows, "n_cols": int(head.shape[1]), "columns": ";".join(head.columns), "path": str(p)})
        except Exception as exc:
            rows.append({"table_name": p.name, "status": f"unreadable:{type(exc).__name__}", "n_rows": 0, "n_cols": 0, "columns": "", "path": str(p)})
    df = pd.DataFrame(rows)
    write(V2_DIR / "reviewer_requested_source_table.tsv", df, "新版审稿补充表格索引。每一行是一份可追溯原表。")
    (V2_DIR / "TABLE_MANIFEST_V2.json").write_text(json.dumps({"n_tables": len(rows), "tables": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parts = [load_external_model(), load_mean(), load_cognate(), load_ridge(), load_internal_mlp_and_model()]
    full = pd.concat(parts, ignore_index=True, sort=False)
    write(V2_DIR / "benchmark_table_reviewer_full_per_target.tsv", full, "审稿要求的基准对照逐读数原表，包含正式模型、均值、同源 mRNA、ridge、内部 MLP 对照。")
    write(V2_DIR / "benchmark_table_reviewer_full_summary.tsv", summarize(full), "基准对照按队列和方法汇总。")
    write(V2_DIR / "benchmark_paired_wilcoxon_summary.tsv", paired_wilcoxon(full), "外部验证中正式模型与可配对基线的逐读数配对 Wilcoxon 检验。")
    rebuild_source_index()
    print("done", V2_DIR, flush=True)


if __name__ == "__main__":
    main()
