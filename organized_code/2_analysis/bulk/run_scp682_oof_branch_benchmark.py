"""
模型: SCP682
作用: 从原始实验脚本提取的最小可复现代码片段，文件名 run_scp682_oof_branch_benchmark.py
输入: ./data_root 下的训练数据、图先验和冻结基线预测
输出: ./paper_materials_SCP682 或结果目录中的模型、表格、报告
依赖: Python、pandas、numpy、torch、torch_geometric
原始路径: remote_scripts/run_scp682_oof_branch_benchmark.py
原始版本: 20260523 结果目录对应脚本
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("./data_root")
DEFAULT_BRANCH = ROOT / "SCP682_PORTABLE/training_set"
DEFAULT_GENERAL = ROOT / "SCP682-main/inputs/general_baseline_predictions/general_baseline_internal_cptac_pdc_phosphosite.parquet"
DEFAULT_OUT = ROOT / "SCP682-main/results/20260523_oof_branch_benchmark"


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def spearman_cols(y: np.ndarray, pred: np.ndarray, mask: np.ndarray, targets: list[str]) -> pd.DataFrame:
    rows = []
    for j, t in enumerate(targets):
        ok = mask[:, j] & np.isfinite(y[:, j]) & np.isfinite(pred[:, j])
        if int(ok.sum()) < 3:
            sp = np.nan
        else:
            sp = pd.Series(y[ok, j]).corr(pd.Series(pred[ok, j]), method="spearman")
        rows.append({"target": t, "n": int(ok.sum()), "spearman": sp})
    return pd.DataFrame(rows)


def summarize(per: pd.DataFrame, method: str) -> dict:
    v = pd.to_numeric(per["spearman"], errors="coerce").dropna()
    return {
        "method": method,
        "n_targets": int(v.shape[0]),
        "median_spearman": float(v.median()) if len(v) else np.nan,
        "mean_spearman": float(v.mean()) if len(v) else np.nan,
        "ge_0_3": int((v >= 0.3).sum()),
        "ge_0_5": int((v >= 0.5).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch-dir", default=str(DEFAULT_BRANCH))
    ap.add_argument("--general-baseline", default=str(DEFAULT_GENERAL))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.output_dir)
    for sub in ["tables", "reports"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    branch = Path(args.branch_dir)

    y_df = clean_numeric(pd.read_parquet(branch / "observed_phosphosite.parquet"))
    matrices = {
        "mean_prediction_constant": None,
        "parent_only": clean_numeric(pd.read_parquet(branch / "oof_candidate_parent_only_phosphosite.parquet")),
        "ridge_direct": clean_numeric(pd.read_parquet(branch / "oof_candidate_ridge_direct_phosphosite.parquet")),
        "rna_direct": clean_numeric(pd.read_parquet(branch / "oof_candidate_rna_direct_phosphosite.parquet")),
        "general_baseline": clean_numeric(pd.read_parquet(args.general_baseline)),
    }
    combo_keys = ["parent_only", "ridge_direct", "rna_direct"]
    common_samples = y_df.index
    common_targets = list(y_df.columns)
    for key, df in matrices.items():
        if df is None:
            continue
        common_samples = common_samples.intersection(df.index)
        common_targets = [t for t in common_targets if t in df.columns]

    y = y_df.loc[common_samples, common_targets].to_numpy(np.float32)
    summaries = []
    per_site_frames = []
    for method, df in matrices.items():
        if df is None:
            pred = np.tile(np.nanmean(y, axis=0, keepdims=True), (y.shape[0], 1)).astype(np.float32)
        else:
            pred = df.loc[common_samples, common_targets].to_numpy(np.float32)
        mask = np.isfinite(y) & np.isfinite(pred)
        per = spearman_cols(y, pred, mask, common_targets)
        per["method"] = method
        per_site_frames.append(per)
        summaries.append(summarize(per, method))

    combo = np.zeros_like(y, dtype=np.float32)
    combo_mask = np.ones_like(y, dtype=bool)
    for key in combo_keys:
        arr = matrices[key].loc[common_samples, common_targets].to_numpy(np.float32)
        combo += np.nan_to_num(arr, nan=0.0)
        combo_mask &= np.isfinite(arr)
    combo = combo / float(len(combo_keys))
    mask = np.isfinite(y) & combo_mask
    per = spearman_cols(y, combo, mask, common_targets)
    per["method"] = "three_branch_mean"
    per_site_frames.append(per)
    summaries.append(summarize(per, "three_branch_mean"))

    pd.DataFrame(summaries).fillna("NA").to_csv(out / "tables/oof_branch_benchmark_summary.tsv", sep="\t", index=False)
    pd.concat(per_site_frames, axis=0, ignore_index=True).fillna("NA").to_csv(out / "tables/oof_branch_benchmark_per_site.tsv", sep="\t", index=False)
    (out / "reports/run_summary.json").write_text(
        json.dumps(
            {
                "branch_dir": str(branch),
                "general_baseline": str(args.general_baseline),
                "n_samples": int(len(common_samples)),
                "n_targets": int(len(common_targets)),
                "methods": list(matrices.keys()) + ["three_branch_mean"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out / "done.txt").write_text("done\n", encoding="utf-8")


if __name__ == "__main__":
    main()

