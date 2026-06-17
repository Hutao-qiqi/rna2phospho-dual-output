# 模型: SCP682-SC
# 作用: 汇总 SC11 正式架构内部五折交叉验证的逐折、逐位点和总体指标。
# 输入: ./results/SCP682_SC_internal_5fold/fold_<n>/tables/scp682_sc11_reconstruction_performance.tsv。
# 输出: scp682_sc11_internal_5fold_per_target.tsv、summary_by_fold.tsv 和 pooled_summary.tsv。
# 依赖: Python, pandas, numpy。
# 原始路径: D:\data\lsy\vm_lsy_parent\lsy\03_code\single_cell\modeling\summarize_scp682_sc11_internal_5fold.py
# 原始版本: 2026-05-27 SC11 正式五折汇总脚本。

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_values(values):
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) == 0:
        return {
            "n_targets": 0,
            "median_spearman": "NA",
            "mean_spearman": "NA",
            "min_spearman": "NA",
            "max_spearman": "NA",
        }
    return {
        "n_targets": int(len(v)),
        "median_spearman": float(v.median()),
        "mean_spearman": float(v.mean()),
        "min_spearman": float(v.min()),
        "max_spearman": float(v.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    args = ap.parse_args()
    base = Path(args.base_dir)
    table_dir = base / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    perf_rows = []
    summary_rows = []
    for fold in range(1, 6):
        perf_path = base / f"fold_{fold}" / "tables" / "scp682_sc11_reconstruction_performance.tsv"
        if not perf_path.exists():
            summary_rows.append(
                {
                    "fold": fold,
                    "status": "missing",
                    "test_dataset": "all",
                    "n_cells": "NA",
                    **summarize_values([]),
                    "source_file": str(perf_path),
                }
            )
            continue
        df = pd.read_csv(perf_path, sep="\t")
        df["fold"] = fold
        perf_rows.append(df)
        sub = df[df["evaluation"].astype(str).eq("internal_cv_reconstruction")].copy()
        for ds, block in sub.groupby("test_dataset", dropna=False):
            stat = summarize_values(block["spearman"])
            summary_rows.append(
                {
                    "fold": fold,
                    "status": "complete",
                    "test_dataset": ds,
                    "n_cells": int(pd.to_numeric(block["n"], errors="coerce").max()) if len(block) else "NA",
                    **stat,
                    "source_file": str(perf_path),
                }
            )

    if perf_rows:
        pd.concat(perf_rows, ignore_index=True).to_csv(
            table_dir / "scp682_sc11_internal_5fold_per_target.tsv",
            sep="\t",
            index=False,
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(table_dir / "scp682_sc11_internal_5fold_summary_by_fold.tsv", sep="\t", index=False)

    complete = summary[(summary["status"] == "complete") & (summary["test_dataset"] == "all")].copy()
    if not complete.empty:
        vals = pd.to_numeric(complete["median_spearman"], errors="coerce")
        pooled = pd.DataFrame(
            [
                {
                    "metric": "fold_median_spearman",
                    "n_folds": int(vals.notna().sum()),
                    "median_across_folds": float(vals.median()),
                    "mean_across_folds": float(vals.mean()),
                    "min_fold": float(vals.min()),
                    "max_fold": float(vals.max()),
                }
            ]
        )
    else:
        pooled = pd.DataFrame(
            [
                {
                    "metric": "fold_median_spearman",
                    "n_folds": 0,
                    "median_across_folds": "NA",
                    "mean_across_folds": "NA",
                    "min_fold": "NA",
                    "max_fold": "NA",
                }
            ]
        )
    pooled.to_csv(table_dir / "scp682_sc11_internal_5fold_pooled_summary.tsv", sep="\t", index=False)
    print("wrote", table_dir)


if __name__ == "__main__":
    main()
