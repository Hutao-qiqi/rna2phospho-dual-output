#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\data\gongke\TCGA-TCPA\paper_materials_SCP682")
FIG2 = ROOT / "04_figure_source_data" / "fig2"
OUT = ROOT / "04_figure_source_data" / "extended_data_tables"
SRC = FIG2 / "fig2_panel_b_literature_comparison.tsv"


DISPLAY = {
    "DREAM_best_phosphoproteomics_BRCA": "DREAM phosphoproteomics, BRCA blind test",
    "DREAM_best_phosphoproteomics_OV": "DREAM phosphoproteomics, OV blind test",
    "DeepGxP": "DeepGxP, TCGA test set",
    "DeepGxP_ICGC": "DeepGxP, independent ICGC validation",
    "CPTAC_8_protein_prediction": "CPTAC_8 RNA-to-protein prediction",
    "KinPred_RNA": "KinPred-RNA",
    "CoPheeMap": "CoPheeMap",
    "CoPheeKSA": "CoPheeKSA",
    "FunMap": "FunMap",
}

RELATION = {
    "not_direct_protein_input": "使用蛋白组输入，不是纯 RNA 输入",
    "not_direct_rppa_not_ms_site": "预测 RPPA 靶标，不是大规模 MS phosphosite",
    "not_direct_protein_not_phosphosite": "预测 total protein，不是 phosphosite",
    "not_direct_kinase_activity": "预测 kinase activity，不是 phosphosite abundance",
    "not_direct_network_edge": "预测 phosphosite 共调控边，不是样本级 abundance",
    "not_direct_ksa": "预测 kinase-substrate association，不是样本级 abundance",
    "not_direct_gene_network": "预测功能网络，不是 phosphosite abundance",
}

METRIC = {
    "mean_pearson": "mean Pearson r",
    "median_pearson": "median Pearson r",
    "auroc": "AUROC",
    "median_auroc": "median AUROC",
    "llr_threshold": "LLR threshold",
    "r2_high_predictability_fraction": "fraction of kinases with R² > 0.5",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC, sep="\t").fillna("not_reported")
    adj = df[df["include_in_main_panel"] == "no"].copy().reset_index(drop=True)
    adj.insert(0, "table_order", range(1, len(adj) + 1))
    adj["method_label"] = adj["method"].map(DISPLAY).fillna(adj["method"])
    adj["publication"] = adj["paper"].astype(str) + " (" + adj["year"].astype(str) + ")"
    adj["relationship_to_scp682"] = adj["direct_comparison_level"].map(RELATION).fillna(adj["direct_comparison_level"])
    adj["reported_metric"] = adj["metric"].map(METRIC).fillna(adj["metric"])
    adj["reported_value"] = adj["value"].map(lambda x: f"{float(x):.4g}" if str(x) != "not_reported" else "not_reported")
    adj["sample_count"] = adj["n_samples"].astype(str)
    adj["target_count"] = adj["n_targets"].astype(str)

    out_cols = [
        "table_order",
        "method_label",
        "publication",
        "input_data",
        "output_target",
        "validation_scope",
        "reported_metric",
        "reported_value",
        "sample_count",
        "target_count",
        "relationship_to_scp682",
        "source_url",
        "notes",
    ]
    out = adj[out_cols].copy()
    tsv = OUT / "extended_data_table_1_adjacent_task_models.tsv"
    out.to_csv(tsv, sep="\t", index=False)

    md = OUT / "extended_data_table_1_adjacent_task_models.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Extended Data Table 1. 与 SCP682 相邻的大规模机器学习模型\n\n")
        f.write("本表列出 Fig. 2b 未放入主图同轴比较的 9 个相邻任务模型。它们用于界定 SCP682 所处的方法学背景，但不作为主图的直接数值基线。主图只保留直接或近直接的 bulk RNA → MS phosphosite abundance 对照。\n\n")
        f.write("**注意事项**: 指标按原文报告，不假设数值可直接换算。\n\n")
        f.write("| 序号 | 方法 | 文献 | 输入 | 输出 | 验证口径 | 原文指标 | 原文数值 | 样本数 | 靶标数 | 为什么不进入主图同轴比较 |\n")
        f.write("|---:|---|---|---|---|---|---|---:|---:|---:|---|\n")
        for _, r in out.iterrows():
            f.write(
                "| {table_order} | {method_label} | {publication} | {input_data} | {output_target} | "
                "{validation_scope} | {reported_metric} | {reported_value} | {sample_count} | "
                "{target_count} | {relationship_to_scp682} |\n".format(**r.to_dict())
            )
        f.write("\n## 数据字典\n\n")
        f.write("| 列 | 含义 |\n|---|---|\n")
        f.write("| `table_order` | 表内排序 |\n")
        f.write("| `method_label` | 方法或队列标签 |\n")
        f.write("| `publication` | 原始文献与年份 |\n")
        f.write("| `input_data` | 原文模型输入 |\n")
        f.write("| `output_target` | 原文预测对象 |\n")
        f.write("| `validation_scope` | 原文验证口径 |\n")
        f.write("| `reported_metric` | 原文报告指标 |\n")
        f.write("| `reported_value` | 原文报告数值 |\n")
        f.write("| `sample_count` | 原文样本量；`not_reported` 表示主文未按该字段报告 |\n")
        f.write("| `target_count` | 原文靶标或边数量；`not_reported` 表示主文未按该字段报告 |\n")
        f.write("| `relationship_to_scp682` | 与 SCP682 主任务的差异 |\n")
        f.write("| `source_url` | 文献或原始数据地址 |\n")
        f.write("| `notes` | 备注 |\n")
        f.write("\n## 源文件\n\n")
        f.write(f"- 源表: `{SRC}`\n")
        f.write(f"- 输出 TSV: `{tsv}`\n")
    print(tsv)
    print(md)


if __name__ == "__main__":
    main()
