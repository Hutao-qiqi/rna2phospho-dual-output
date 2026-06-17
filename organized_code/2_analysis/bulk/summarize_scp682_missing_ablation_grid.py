"""
模型: SCP682
作用: 从原始实验脚本提取的最小可复现代码片段，文件名 summarize_scp682_missing_ablation_grid.py
输入: ./data_root 下的训练数据、图先验和冻结基线预测
输出: ./paper_materials_SCP682 或结果目录中的模型、表格、报告
依赖: Python、pandas、numpy、torch、torch_geometric
原始路径: remote_scripts/summarize_scp682_missing_ablation_grid.py
原始版本: 20260523 结果目录对应脚本
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    grid = Path(args.grid_dir)
    rows = []
    for run_dir in sorted(p for p in grid.iterdir() if p.is_dir() and p.name != "logs"):
        summary_path = run_dir / "tables/model_summary_best.tsv"
        graph_path = run_dir / "reports/input_graph_summary.json"
        if not summary_path.exists():
            rows.append(
                {
                    "run_name": run_dir.name,
                    "axis_mode": "NA",
                    "edge_mode": "NA",
                    "model": "NA",
                    "n_targets": "NA",
                    "median_spearman": "NA",
                    "mean_spearman": "NA",
                    "ge_0_3": "NA",
                    "ge_0_5": "NA",
                    "n_site_edges": "NA",
                    "n_sample_edges": "NA",
                    "source_file": str(summary_path),
                    "status": "missing_summary",
                }
            )
            continue
        meta = {}
        if graph_path.exists():
            meta = json.loads(graph_path.read_text(encoding="utf-8"))
        df = pd.read_csv(summary_path, sep="\t")
        for _, r in df.iterrows():
            rows.append(
                {
                    "run_name": run_dir.name,
                    "axis_mode": meta.get("axis_mode", "NA"),
                    "edge_mode": meta.get("edge_mode", "NA"),
                    "model": r.get("model", "NA"),
                    "n_targets": r.get("n_targets", "NA"),
                    "median_spearman": r.get("median_spearman", "NA"),
                    "mean_spearman": r.get("mean_spearman", "NA"),
                    "ge_0_3": r.get("ge_0_3", "NA"),
                    "ge_0_5": r.get("ge_0_5", "NA"),
                    "n_site_edges": meta.get("n_site_edges", "NA"),
                    "n_sample_edges": meta.get("n_sample_edges", "NA"),
                    "source_file": str(summary_path),
                    "status": "ok",
                }
            )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).fillna("NA").to_csv(out, sep="\t", index=False)


if __name__ == "__main__":
    main()
