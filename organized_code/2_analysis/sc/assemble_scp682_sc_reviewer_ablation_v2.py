#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(r"D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260522_scp682_sc11_expanded_scnet_site_gnn_v1")
V2 = BASE / "reviewer_requested_tables_v2"
NO_ATT = Path(r"D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260531_scp682_sc11_no_pathway_attention_ablation_v1")
RAW = Path(r"D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260531_scp682_sc_raw_expression_selected_gene_ridge_v1")


COHORT_NAMES = {
    "gse300551_iccite_plex_kinase_2025": "GSE300551",
    "signal_seq_gse256403_hela_2024": "SIGNAL-seq HeLa",
    "signal_seq_gse256404_pdo_caf_2024": "SIGNAL-seq PDO/CAF",
    "phospho_seq_blair_2025_phospho_multi": "Blair",
    "vivo_seq_th17_2025": "Vivo-seq Th17",
}


def write(path: Path, df: pd.DataFrame, note: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, na_rep="NA")
    path.with_suffix(".md").write_text(note + "\n", encoding="utf-8")


def med(x):
    vals = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    return float(np.median(vals)) if len(vals) else math.nan


def copy_sources():
    pairs = [
        (NO_ATT / "tables" / "scp682_sc11_reconstruction_performance.tsv", V2 / "pathway_attention_removed_ablation_per_target.tsv"),
        (RAW / "tables" / "raw_expression_selected_gene_ridge_per_target.tsv", V2 / "scFoundation_removed_raw_expression_ablation_per_target.tsv"),
        (RAW / "tables" / "raw_expression_selected_gene_ridge_summary.tsv", V2 / "scFoundation_removed_raw_expression_ablation_summary.tsv"),
        (RAW / "tables" / "raw_expression_selected_genes.tsv", V2 / "scFoundation_removed_raw_expression_selected_genes.tsv"),
        (RAW / "tables" / "raw_expression_loading_status.tsv", V2 / "scFoundation_removed_raw_expression_loading_status.tsv"),
    ]
    for src, dst in pairs:
        if src.exists():
            shutil.copy2(src, dst)


def build_component_table():
    rows = []
    formal = pd.read_csv(V2 / "external_per_target_from_predicted_observed.tsv", sep="\t")
    for _, r in formal.iterrows():
        ds = str(r["cohort_id"])
        rows.append({
            "component_variant": "full_SCP682_SC",
            "evaluation_scope": "external_validation",
            "cohort_id": ds,
            "cohort_name": COHORT_NAMES.get(ds, ds),
            "target_id": str(r["target_id"]),
            "n_cells": pd.to_numeric(r.get("sample_size_used"), errors="coerce"),
            "spearman": pd.to_numeric(r.get("spearman"), errors="coerce"),
            "source_file": str(V2 / "external_per_target_from_predicted_observed.tsv"),
        })
    no_att = pd.read_csv(NO_ATT / "tables" / "scp682_sc11_reconstruction_performance.tsv", sep="\t")
    no_att = no_att[no_att["evaluation"].astype(str).eq("external_reconstruction") & ~no_att["test_dataset"].astype(str).eq("all")]
    for _, r in no_att.iterrows():
        ds = str(r["test_dataset"])
        rows.append({
            "component_variant": "pathway_attention_removed",
            "evaluation_scope": "external_validation",
            "cohort_id": ds,
            "cohort_name": COHORT_NAMES.get(ds, ds),
            "target_id": str(r["target_id"]),
            "n_cells": pd.to_numeric(r.get("n"), errors="coerce"),
            "spearman": pd.to_numeric(r.get("spearman"), errors="coerce"),
            "source_file": str(NO_ATT / "tables" / "scp682_sc11_reconstruction_performance.tsv"),
        })
    raw = pd.read_csv(RAW / "tables" / "raw_expression_selected_gene_ridge_per_target.tsv", sep="\t")
    for _, r in raw.iterrows():
        ds = str(r["test_dataset"])
        rows.append({
            "component_variant": "scFoundation_removed_raw_expression_ridge",
            "evaluation_scope": "external_validation",
            "cohort_id": ds,
            "cohort_name": COHORT_NAMES.get(ds, ds),
            "target_id": str(r["target_id"]),
            "n_cells": pd.to_numeric(r.get("n_test"), errors="coerce"),
            "spearman": pd.to_numeric(r.get("spearman"), errors="coerce"),
            "source_file": str(RAW / "tables" / "raw_expression_selected_gene_ridge_per_target.tsv"),
        })
    graph = pd.read_csv(V2 / "bulk_site_graph_matched_ablation_per_target.tsv", sep="\t")
    for _, r in graph.iterrows():
        ds = str(r["test_dataset"])
        rows.append({
            "component_variant": "expanded_site_graph_removed_matched",
            "evaluation_scope": "external_validation",
            "cohort_id": ds,
            "cohort_name": COHORT_NAMES.get(ds, ds),
            "target_id": str(r["target_id"]),
            "n_cells": pd.to_numeric(r.get("n_matched_no_site_graph"), errors="coerce"),
            "spearman": pd.to_numeric(r.get("spearman_matched_no_site_graph"), errors="coerce"),
            "source_file": str(V2 / "bulk_site_graph_matched_ablation_per_target.tsv"),
        })
    df = pd.DataFrame(rows)
    write(V2 / "component_ablation_reviewer_per_target.tsv", df, "审稿要求的组件消融逐读数原表。")
    summary = []
    for keys, sub in df.groupby(["component_variant", "cohort_id", "cohort_name"], dropna=False):
        vals = pd.to_numeric(sub["spearman"], errors="coerce")
        summary.append({
            "component_variant": keys[0],
            "cohort_id": keys[1],
            "cohort_name": keys[2],
            "n_targets": int(vals.notna().sum()),
            "median_spearman": med(vals),
            "mean_spearman": float(vals.dropna().mean()) if vals.notna().any() else math.nan,
            "min_spearman": float(vals.dropna().min()) if vals.notna().any() else math.nan,
            "max_spearman": float(vals.dropna().max()) if vals.notna().any() else math.nan,
        })
    s = pd.DataFrame(summary)
    full = s[s["component_variant"].eq("full_SCP682_SC")][["cohort_id", "median_spearman"]].rename(columns={"median_spearman": "full_median_spearman"})
    s = s.merge(full, on="cohort_id", how="left")
    s["delta_median_vs_full"] = pd.to_numeric(s["median_spearman"], errors="coerce") - pd.to_numeric(s["full_median_spearman"], errors="coerce")
    write(V2 / "component_ablation_reviewer_summary.tsv", s, "组件消融按外部队列汇总，delta 为变体中位 Spearman 减正式模型中位 Spearman。")


def update_benchmark():
    bench = pd.read_csv(V2 / "benchmark_table_reviewer_full_per_target.tsv", sep="\t")
    raw = pd.read_csv(RAW / "tables" / "raw_expression_selected_gene_ridge_per_target.tsv", sep="\t")
    raw_rows = pd.DataFrame({
        "evaluation_scope": "external_validation",
        "method_name": "raw_expression_selected_gene_ridge",
        "cohort_id": raw["test_dataset"].astype(str),
        "cohort_name": raw["cohort_name"].astype(str),
        "target_id": raw["target_id"].astype(str),
        "n_cells": pd.to_numeric(raw["n_test"], errors="coerce"),
        "spearman": pd.to_numeric(raw["spearman"], errors="coerce"),
        "p_value": pd.to_numeric(raw["spearman_pvalue"], errors="coerce"),
        "mae": np.nan,
        "rmse": np.nan,
        "source_file": str(RAW / "tables" / "raw_expression_selected_gene_ridge_per_target.tsv"),
        "notes": "原始 RNA 选定基因 ridge；用于 scFoundation 去除消融",
    })
    bench = pd.concat([bench, raw_rows], ignore_index=True)
    bench = bench.drop_duplicates(["evaluation_scope", "method_name", "cohort_id", "target_id"], keep="last")
    write(V2 / "benchmark_table_reviewer_full_per_target.tsv", bench, "审稿要求的基准对照逐读数原表，包含正式模型、均值、同源 mRNA、ridge、原始表达 ridge、内部 MLP 对照。")
    summary = []
    for keys, sub in bench.groupby(["evaluation_scope", "method_name", "cohort_id", "cohort_name"], dropna=False):
        vals = pd.to_numeric(sub["spearman"], errors="coerce")
        summary.append({
            "evaluation_scope": keys[0],
            "method_name": keys[1],
            "cohort_id": keys[2],
            "cohort_name": keys[3],
            "n_targets_with_spearman": int(vals.notna().sum()),
            "median_spearman": med(vals),
            "mean_spearman": float(vals.dropna().mean()) if vals.notna().any() else math.nan,
            "min_spearman": float(vals.dropna().min()) if vals.notna().any() else math.nan,
            "max_spearman": float(vals.dropna().max()) if vals.notna().any() else math.nan,
            "n_rows": int(len(sub)),
        })
    write(V2 / "benchmark_table_reviewer_full_summary.tsv", pd.DataFrame(summary), "基准对照按队列和方法汇总。")


def remove_missing_status():
    p = V2 / "missing_ablation_rerun_status.tsv"
    if not p.exists():
        return
    df = pd.read_csv(p, sep="\t")
    df["status"] = df["item"].map({
        "pathway_attention_removed_ablation": "rerun_completed",
        "scFoundation_removed_raw_expression_ablation": "rerun_completed",
    }).fillna(df["status"])
    df["notes"] = df["item"].map({
        "pathway_attention_removed_ablation": f"结果表: {V2 / 'pathway_attention_removed_ablation_per_target.tsv'}",
        "scFoundation_removed_raw_expression_ablation": f"结果表: {V2 / 'scFoundation_removed_raw_expression_ablation_per_target.tsv'}",
    }).fillna(df["notes"])
    write(p, df, "原缺失消融项的重跑状态。")


def rebuild_source_index():
    rows = []
    for p in sorted(V2.glob("*.tsv")):
        try:
            head = pd.read_csv(p, sep="\t", nrows=5)
            with p.open("r", encoding="utf-8", errors="ignore") as fh:
                n_rows = max(sum(1 for _ in fh) - 1, 0)
            rows.append({"table_name": p.name, "status": "available", "n_rows": n_rows, "n_cols": int(head.shape[1]), "columns": ";".join(head.columns), "path": str(p)})
        except Exception as exc:
            rows.append({"table_name": p.name, "status": f"unreadable:{type(exc).__name__}", "n_rows": 0, "n_cols": 0, "columns": "", "path": str(p)})
    df = pd.DataFrame(rows)
    write(V2 / "reviewer_requested_source_table.tsv", df, "新版审稿补充表格索引。每一行是一份可追溯原表。")
    (V2 / "TABLE_MANIFEST_V2.json").write_text(json.dumps({"n_tables": len(rows), "tables": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    copy_sources()
    build_component_table()
    update_benchmark()
    remove_missing_status()
    rebuild_source_index()
    print("done", V2, flush=True)


if __name__ == "__main__":
    main()
