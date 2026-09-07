#!/usr/bin/env python3
"""生成 M2.2 研究内中心化评价的最终结果工件。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "02_results/model_validation/scp682_v2_m22_study_centered_evaluation_20260830"


def main() -> int:
    primary = pd.read_csv(OUT / "tables/m22_primary_corrected_metrics.tsv", sep="\t")
    calibration = pd.read_csv(OUT / "tables/m22_external_247_corrected_metrics.tsv", sep="\t")
    external = pd.read_csv(OUT / "tables/m22_external_106_corrected_metrics.tsv", sep="\t")
    selected_external = external[external.scope.eq("external_106_locked_test")]
    combined = pd.concat(
        [
            primary[primary.coordinate.eq("study_centered")],
            calibration[calibration.coordinate.eq("study_centered")],
            selected_external[selected_external.coordinate.eq("study_centered")],
        ],
        ignore_index=True,
    )
    combined.to_csv(OUT / "tables/m22_study_centered_reference_metrics.tsv", sep="\t", index=False)
    grid = pd.read_csv(OUT / "tables/m22_external_247_centered_pt7_grid.tsv", sep="\t")
    selected = grid[grid.selected.astype(bool)].iloc[0]
    top = pd.read_csv(OUT / "tables/m22_train1796_study_centered_top4.tsv", sep="\t")
    policy = {
        "status": "active",
        "primary_metric_coordinate": "study_centered",
        "centering_axis": ["study", "phosphosite"],
        "centering_operation": "truth and prediction means are computed separately on the same paired observations within each study and phosphosite",
        "missing_values": "preserved; centering and metrics use the same pairwise-observed patients",
        "minimum_observations": 10,
        "selection": {
            "primary": "study-centered patient Pearson median",
            "guard": "study-centered site Pearson not below the locked PT6 centered baseline",
        },
        "locked_external_parameters": {
            "projection_ridge": 0.001,
            "adapter_ridge_alpha": 10.0,
            "adapter_rank": 24,
            "gamma_J": float(selected.gamma),
            "lambda": float(selected["lambda"]),
            "kappa": float(selected.kappa),
            "graph": 0,
        },
    }
    (OUT / "reports/METRIC_POLICY.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    labels = {
        "train_1796_OOF": "训练集五折交叉预测",
        "development_770": "内部开发验证",
        "external_247_OOF": "外部后训练五折",
        "external_106_locked_test": "外部锁定测试",
    }
    lines = [
        "# SCP682-v2 M2.2 研究内中心化评价结果",
        "",
        "## 指标口径",
        "",
        "对每个研究和磷酸化位点，在真实值与预测值均可用的同一组患者上分别计算均值并中心化，再计算患者与位点 Pearson、Spearman 和均方误差。缺失值保留，中心化与指标计算使用相同的成对观测。",
        "",
        "## 主结果",
        "",
        "| 数据范围 | 患者 Pearson | 患者 Spearman | 位点 Pearson | 位点 Spearman | 患者等权均方误差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in combined.itertuples(index=False):
        lines.append(
            f"| {labels.get(row.scope, row.scope)} | {row.patient_pearson_median:.6f} | {row.patient_spearman_median:.6f} | {row.site_pearson_median:.6f} | {row.site_spearman_median:.6f} | {row.patient_equal_mse:.6f} |"
        )
    lines += [
        "",
        "## 外部后训练参数",
        "",
        f"中心化五折选择参数为 `gamma_J={float(selected.gamma):g}`、`lambda={float(selected['lambda']):g}`、`kappa={float(selected.kappa):g}`；latent projection ridge 为 `0.001`，adapter ridge alpha 为 `10`，rank 为 `24`。",
        "",
        "## 训练集高覆盖位点",
        "",
        "| 磷酸化位点 | 观测患者 | 中心化 Pearson | 中心化 Spearman |",
        "|---|---:|---:|---:|",
    ]
    for row in top.itertuples(index=False):
        lines.append(f"| {row.phosphosite} | {int(row.patients_observed)} | {row.study_centered_pearson:.6f} | {row.study_centered_spearman:.6f} |")
    lines += [
        "",
        "## 活跃结果文件",
        "",
        "- `tables/m22_study_centered_reference_metrics.tsv`",
        "- `tables/m22_train1796_study_centered_per_site.tsv`",
        "- `tables/m22_dev770_study_centered_per_site.tsv`",
        "- `tables/m22_external_247_centered_pt7_grid.tsv`",
        "- `tables/m22_external_106_study_centered_per_site.tsv`",
        "- `figures/m22_train1796_study_centered_top4.pdf`",
        "- `reports/METRIC_POLICY.json`",
        "",
    ]
    (OUT / "RESULTS_20260830.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "SUCCESS").touch()
    print(combined.to_string(index=False))
    print(f"selected gamma={selected.gamma:g}, lambda={selected['lambda']:g}, kappa={selected.kappa:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
