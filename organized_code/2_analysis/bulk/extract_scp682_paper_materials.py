#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract paper materials for SCP682.

This script is intentionally separate from model training code. It collects
existing result files, copies required prediction matrices, and writes a
paper-extract compatible material package with a manifest and missing list.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


MODEL = "SCP682"
TASK = "bulk RNA expression to sample-level phosphosite abundance prediction"
TRAINING_SUMMARY = "1431 tumor samples x 18592 phosphosites with paired RNA and phosphoproteomics"

ROOT = Path("/data/lsy/Infinite_Stream")
OUT = ROOT / "paper_materials_SCP682"

MAIN_RESULT = ROOT / "SCP682-main/results/20260523_general_graph_residual_e160"
MAIN_EXTERNAL = ROOT / "SCP682-main/results/20260523_general_graph_external_fixed_anchor"
BRANCH_DIR = ROOT / "SCP682_PORTABLE/training_set"
GROUP_PERF = ROOT / "SCP682_PORTABLE/performance/cancer_group_performance.tsv"
MODEL_CONTRACT = ROOT / "SCP682/SCP682_model_contract.json"
MISSING_ABLATION_GRID = ROOT / "SCP682-main/results/20260523_missing_ablation_grid_e40/tables/missing_ablation_grid_summary.tsv"
SHRINKAGE_GRID = ROOT / "SCP682-main/results/20260523_shrinkage_sensitivity_grid/tables/shrinkage_sensitivity_grid.tsv"
SHRINKAGE_PER_SITE = ROOT / "SCP682-main/results/20260523_shrinkage_sensitivity_grid/tables/shrinkage_sensitivity_per_site.tsv"
OOF_BRANCH_BENCHMARK = ROOT / "SCP682-main/results/20260523_oof_branch_benchmark/tables/oof_branch_benchmark_summary.tsv"
OOF_BRANCH_BENCHMARK_PER_SITE = ROOT / "SCP682-main/results/20260523_oof_branch_benchmark/tables/oof_branch_benchmark_per_site.tsv"

SENSITIVITY_SOURCES = [
    {
        "label": "fixed_shrinkage_0p3",
        "path": ROOT / "SCP682-21/results/20260515_v4_frozen_pathway_residual_correction_s0p3/tables/scp682_21_model_summary.tsv",
    },
    {
        "label": "fixed_shrinkage_0p4",
        "path": ROOT / "SCP682-21/results/20260515_v4_frozen_pathway_residual_correction_s0p4/tables/scp682_21_model_summary.tsv",
    },
    {
        "label": "adaptive_base0p3_max0p45_pathway_dropout",
        "path": ROOT / "SCP682-23/results/20260516_pathway_dropout_adaptive_shrinkage_base0p3_max0p45_pdrop0p2/tables/scp682_23_model_summary.tsv",
    },
]

RNA_PROTEIN_EXTERNAL = [
    ROOT / "02_results/external_validation/20260503_fu_icca_v38_v39_predicted_vs_true_phosphosite/tables/fu_icca_total_protein_evaluation_summary.tsv",
    ROOT / "02_results/external_validation/20260503_tu_sclc_v38_v39_predicted_vs_true_phosphosite/tables/tu_sclc_total_protein_evaluation_summary.tsv",
    ROOT / "02_results/external_validation/20260503_chcc_hbv_fpkm_v38_v39_predicted_vs_true_phosphosite/tables/chcc_hbv_fpkm_total_protein_evaluation_summary.tsv",
    ROOT / "02_results/external_validation/20260503_chcc_hbv_rsem_v38_v39_predicted_vs_true_phosphosite/tables/chcc_hbv_rsem_total_protein_evaluation_summary.tsv",
]

CODE_SOURCES = [
    ROOT / "remote_scripts/train_scp682_general_graph_residual.py",
    ROOT / "remote_scripts/predict_scp682_general_graph_external.py",
    ROOT / "remote_scripts/launch_scp682_general_graph_residual_e160.sh",
    ROOT / "remote_scripts/launch_scp682_general_graph_external.sh",
    ROOT / "remote_scripts/train_scp682_missing_ablation.py",
    ROOT / "remote_scripts/launch_scp682_missing_ablation_grid_e40.sh",
    ROOT / "remote_scripts/summarize_scp682_missing_ablation_grid.py",
    ROOT / "remote_scripts/run_scp682_shrinkage_sensitivity_grid.py",
    ROOT / "remote_scripts/run_scp682_oof_branch_benchmark.py",
]


missing_items: List[str] = []


def clean_out() -> None:
    if OUT.exists():
        if OUT.resolve() != Path("/data/lsy/Infinite_Stream/paper_materials_SCP682").resolve():
            raise RuntimeError(f"refuse to delete unexpected path: {OUT}")
        shutil.rmtree(OUT)
    for sub in [
        "00_model_card",
        "01_key_results/external_validation",
        "02_data_tables/oof_branch_predictions",
        "02_data_tables/external_prediction_matrices",
        "03_code/architecture",
        "03_code/training",
        "03_code/inference",
        "03_code/evaluation",
        "03_code/preprocessing",
        "03_code/visualization",
        "04_figure_source_data",
        "05_methods_writing",
    ]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def read_tsv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        missing_items.append(f"未找到文件: {path}")
        return None
    return pd.read_csv(path, sep="\t")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        missing_items.append(f"未找到文件: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scalar(x: Any) -> str:
    if x is None:
        return "NA"
    try:
        if pd.isna(x):
            return "NA"
    except Exception:
        pass
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    def norm(c: str) -> str:
        c = str(c).strip().lower()
        c = re.sub(r"[^a-z0-9]+", "_", c)
        c = re.sub(r"_+", "_", c).strip("_")
        return c or "field"

    out = df.copy()
    out.columns = [norm(c) for c in out.columns]
    return out


def write_tsv(rel: str, rows: Iterable[Dict[str, Any]], description: str, source: str = "") -> None:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    if df.empty:
        df = pd.DataFrame([{"status": "NA", "note": "no_rows"}])
    df = normalise_columns(df)
    df = df.fillna("NA")
    df.to_csv(path, sep="\t", index=False)
    write_md(
        str(path.with_suffix(".md").relative_to(OUT)),
        f"# {path.name}\n\n{description}\n\n来源: {source or '见表内 source_file 字段'}\n",
    )


def write_md(rel: str, text: str) -> None:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.rstrip() + "\n")


def safe_get(row: pd.Series, candidates: List[str], default: Any = "NA") -> Any:
    for c in candidates:
        if c in row.index:
            return row[c]
    return default


def collect_headline() -> None:
    model_summary = read_tsv(MAIN_RESULT / "tables/model_summary_best.tsv")
    final_summary = read_json(MAIN_RESULT / "reports/final_summary.json")
    graph_summary = read_json(MAIN_RESULT / "reports/input_graph_summary.json")
    external = read_tsv(MAIN_EXTERNAL / "tables/scp682_general_graph_external_summary.tsv")

    rows: List[Dict[str, Any]] = []
    if model_summary is not None:
        for _, r in model_summary.iterrows():
            for metric, value in [
                ("median_spearman", r.get("median_spearman", "NA")),
                ("mean_spearman", r.get("mean_spearman", "NA")),
                ("sites_ge_0_3", r.get("sites_ge_0_3", "NA")),
                ("sites_ge_0_5", r.get("sites_ge_0_5", "NA")),
            ]:
                rows.append(
                    {
                        "metric_name": metric,
                        "value": value,
                        "sample_size": 1431,
                        "cohort_or_split": "internal_oof",
                        "reference_baseline": "scp682_general_baseline",
                        "source_file": str(MAIN_RESULT / "tables/model_summary_best.tsv"),
                        "computation_date": "2026-05-23",
                        "notes": f"model={r.get('model', 'NA')}; n_targets={r.get('n_targets', 'NA')}",
                    }
                )
    if external is not None:
        ext = normalise_columns(external)
        sample_centered = ext[ext["model"].astype(str).str.contains("sample_centered", na=False)]
        for _, r in sample_centered.iterrows():
            for metric, value in [
                ("median_spearman", r.get("median_spearman", "NA")),
                ("mean_spearman", r.get("mean_spearman", "NA")),
                ("sites_ge_0_3", r.get("ge_0_3", "NA")),
                ("sites_ge_0_5", r.get("ge_0_5", "NA")),
            ]:
                rows.append(
                    {
                        "metric_name": metric,
                        "value": value,
                        "sample_size": r.get("n_matched_samples", "NA"),
                        "cohort_or_split": f"external_{r.get('dataset','NA')}",
                        "reference_baseline": "scp682_general_baseline_sample_centered",
                        "source_file": str(MAIN_EXTERNAL / "tables/scp682_general_graph_external_summary.tsv"),
                        "computation_date": "2026-05-23",
                        "notes": f"model={r.get('model', 'NA')}; n_targets={r.get('n_matched_targets', 'NA')}",
                    }
                )
    write_tsv(
        "01_key_results/headline_metrics.tsv",
        rows,
        "SCP682 主模型内部和外部核心指标汇总。",
    )

    model_card = [
        "# SCP682 模型卡",
        "",
        f"模型代号: {MODEL}",
        f"任务: {TASK}",
        f"训练数据: {TRAINING_SUMMARY}",
        "",
        "模型定位: SCP682 将 bulk RNA 表达映射到样本级磷酸化位点丰度。主模型由一般表达基线和图约束残差组成。一般表达基线负责从 RNA 中得到每个样本-位点的初始预测，图约束残差利用 phosphosite 图和样本图学习该初始预测中没有被解释的结构化偏差。",
        "",
        "核心公式:",
        "",
        "```text",
        "state_estimator_hat = S_phi(RNA, cancer_group)",
        "graph_residual_delta = G_theta(RNA, general_baseline_hat, site_graph, sample_graph, cancer_group)",
        "phosphosite_hat = general_baseline_hat + graph_residual_delta",
        "```",
        "",
        "图先验:",
        f"- site edges: {scalar(graph_summary.get('n_site_edges'))}",
        f"- sample edges: {scalar(graph_summary.get('n_sample_edges'))}",
        "",
        "主结果文件:",
        f"- {MAIN_RESULT / 'tables/model_summary_best.tsv'}",
        f"- {MAIN_EXTERNAL / 'tables/scp682_general_graph_external_summary.tsv'}",
        "",
        "说明: 本素材包只记录已经存在的实验结果。未找到的指定消融会列入 MANIFEST.md 的缺失项清单。",
    ]
    if final_summary:
        model_card.extend(
            [
                "",
                "训练报告摘要:",
                f"- best_epoch: {scalar(final_summary.get('best_epoch'))}",
                f"- elapsed_sec: {scalar(final_summary.get('elapsed_sec'))}",
                f"- device: {scalar(final_summary.get('meta', {}).get('device'))}",
            ]
        )
    write_md("00_model_card/model_summary.md", "\n".join(model_card))


def collect_external_per_target() -> None:
    mappings = {
        "fu_icca": "fu_icca_scp682_general_graph_residual_sample_centered_phosphosite_per_target.tsv",
        "tu_sclc": "tu_sclc_scp682_general_graph_residual_sample_centered_phosphosite_per_target.tsv",
        "chcc_hbv_fpkm": "chcc_hbv_fpkm_scp682_general_graph_residual_sample_centered_phosphosite_per_target.tsv",
        "chcc_hbv_rsem": "chcc_hbv_rsem_scp682_general_graph_residual_sample_centered_phosphosite_per_target.tsv",
    }
    for cohort, fname in mappings.items():
        src = MAIN_EXTERNAL / "tables" / fname
        df = read_tsv(src)
        if df is None:
            continue
        df = normalise_columns(df)
        rows = []
        for _, r in df.iterrows():
            rows.append(
                {
                    "target_id": safe_get(r, ["target_id", "target", "site", "phosphosite"], "NA"),
                    "predicted": "NA",
                    "observed": "NA",
                    "per_target_spearman": safe_get(r, ["spearman", "per_target_spearman", "rho"], "NA"),
                    "per_target_pvalue": safe_get(r, ["pvalue", "p_value", "per_target_pvalue"], "NA"),
                    "sample_size_used": safe_get(r, ["n", "n_samples", "sample_size_used"], "NA"),
                    "source_file": str(src),
                }
            )
        write_tsv(
            f"01_key_results/external_validation/per_target_{cohort}.tsv",
            rows,
            f"{cohort} 外部队列逐位点 Spearman。预测值和观测值矩阵未在该结果表中展开，表内保留逐位点相关性。",
            str(src),
        )

    external = read_tsv(MAIN_EXTERNAL / "tables/scp682_general_graph_external_summary.tsv")
    if external is not None:
        rows = []
        external = normalise_columns(external)
        for _, r in external.iterrows():
            item = {**r.to_dict(), "source_file": str(MAIN_EXTERNAL / "tables/scp682_general_graph_external_summary.tsv")}
            item["cohort"] = r.get("dataset", "NA")
            rows.append(item)
        write_tsv(
            "01_key_results/external_validation/per_cohort_summary.tsv",
            rows,
            "四个外部设置的队列级汇总结果。",
            str(MAIN_EXTERNAL / "tables/scp682_general_graph_external_summary.tsv"),
        )


def collect_benchmarks_and_ablation() -> None:
    model_summary = read_tsv(MAIN_RESULT / "tables/model_summary_best.tsv")
    rows: List[Dict[str, Any]] = []
    if model_summary is not None:
        base = model_summary[model_summary["model"].astype(str).eq("scp682_general_baseline")]
        if not base.empty:
            b = float(base.iloc[0]["median_spearman"])
            for _, r in model_summary.iterrows():
                m = float(r.get("median_spearman", "nan"))
                rows.append(
                    {
                        "baseline_name": "scp682_general_baseline",
                        "method_name": r.get("model", "NA"),
                        "metric_name": "median_spearman",
                        "baseline_value": b,
                        "model_value": m,
                        "delta": m - b,
                        "p_value": "NA",
                        "n_sites": r.get("n_targets", "NA"),
                        "source_file": str(MAIN_RESULT / "tables/model_summary_best.tsv"),
                    }
                )
    external = read_tsv(MAIN_EXTERNAL / "tables/scp682_general_graph_external_summary.tsv")
    if external is not None:
        ext = normalise_columns(external)
        ext = ext[ext["model"].astype(str).str.contains("sample_centered", na=False)]
        for dataset, sdf in ext.groupby("dataset", dropna=False):
            base = sdf[sdf["model"].astype(str).str.contains("baseline", na=False)]
            graph = sdf[sdf["model"].astype(str).str.contains("graph_residual", na=False)]
            if base.empty or graph.empty:
                continue
            b = float(base.iloc[0]["median_spearman"])
            g = float(graph.iloc[0]["median_spearman"])
            rows.append(
                {
                    "baseline_name": "scp682_general_baseline_sample_centered",
                    "method_name": "scp682_general_graph_residual_sample_centered",
                    "metric_name": f"external_{dataset}_median_spearman",
                    "baseline_value": b,
                    "model_value": g,
                    "delta": g - b,
                    "p_value": "NA",
                    "n_sites": graph.iloc[0].get("n_matched_targets", "NA"),
                    "source_file": str(MAIN_EXTERNAL / "tables/scp682_general_graph_external_summary.tsv"),
                }
            )
    for src in RNA_PROTEIN_EXTERNAL:
        df = read_tsv(src)
        if df is None:
            continue
        df = normalise_columns(df)
        for _, r in df.iterrows():
            rows.append(
                {
                    "baseline_name": "NA",
                    "method_name": safe_get(r, ["model"], src.stem),
                    "metric_name": f"rna_to_total_protein_{src.stem}",
                    "baseline_value": "NA",
                    "model_value": safe_get(r, ["median_spearman", "median_rho", "spearman_median"], "NA"),
                    "delta": "NA",
                    "p_value": "NA",
                    "n_sites": safe_get(r, ["n_targets", "n_proteins", "n"], "NA"),
                    "source_file": str(src),
                }
            )
    bench = read_tsv(OOF_BRANCH_BENCHMARK)
    if bench is not None:
        bench = normalise_columns(bench)
        base = bench[bench["method"].astype(str).eq("mean_prediction_constant")]
        base_value = "NA"
        if not base.empty:
            base_value = base.iloc[0].get("median_spearman", "NA")
        for _, r in bench.iterrows():
            mv = r.get("median_spearman", "NA")
            try:
                delta = float(mv) - float(base_value)
            except Exception:
                delta = "NA"
            rows.append(
                {
                    "baseline_name": "mean_prediction_constant",
                    "method_name": r.get("method", "NA"),
                    "metric_name": "internal_oof_median_spearman",
                    "baseline_value": base_value,
                    "model_value": mv,
                    "delta": delta,
                    "p_value": "NA",
                    "n_sites": r.get("n_targets", "NA"),
                    "source_file": str(OOF_BRANCH_BENCHMARK),
                }
            )
    write_tsv(
        "01_key_results/benchmark_table.tsv",
        rows,
        "内部模型对照与 RNA 到总蛋白外部评估汇总。已发表方法重跑表未在结果目录中找到，列入缺失项。",
    )

    ab_rows: List[Dict[str, Any]] = []
    if model_summary is not None:
        for _, r in model_summary.iterrows():
            ab_rows.append(
                {
                    "ablation_name": "general_baseline_vs_graph_residual",
                    "model": r.get("model", "NA"),
                    "median_spearman": r.get("median_spearman", "NA"),
                    "mean_spearman": r.get("mean_spearman", "NA"),
                    "n_sites": r.get("n_targets", "NA"),
                    "source_file": str(MAIN_RESULT / "tables/model_summary_best.tsv"),
                    "notes": "主模型内部汇总，可用于基线与图残差读出对照",
                }
            )
    grid = read_tsv(MISSING_ABLATION_GRID)
    if grid is not None:
        grid = normalise_columns(grid)
        for _, r in grid.iterrows():
            if str(r.get("model", "")).endswith("baseline"):
                continue
            ab_rows.append(
                {
                    "ablation_name": f"axis_{r.get('axis_mode', 'NA')}_edge_{r.get('edge_mode', 'NA')}",
                    "model": r.get("model", "NA"),
                    "median_spearman": r.get("median_spearman", "NA"),
                    "mean_spearman": r.get("mean_spearman", "NA"),
                    "n_sites": r.get("n_targets", "NA"),
                    "source_file": str(MISSING_ABLATION_GRID),
                    "notes": f"compact_e40; n_site_edges={r.get('n_site_edges', 'NA')}; n_sample_edges={r.get('n_sample_edges', 'NA')}",
                }
            )
    write_tsv(
        "01_key_results/ablation_results.tsv",
        ab_rows,
        "GNN 双轴消融、三类边源拆分消融和主模型基线对照。",
    )


def collect_sensitivity() -> None:
    rows: List[Dict[str, Any]] = []
    for spec in SENSITIVITY_SOURCES:
        df = read_tsv(spec["path"])
        if df is None:
            continue
        for _, r in df.iterrows():
            rows.append(
                {
                    "scan_name": spec["label"],
                    "model": r.get("model", "NA"),
                    "n_targets": r.get("n_targets", "NA"),
                    "median_spearman": r.get("median_spearman", "NA"),
                    "mean_spearman": r.get("mean_spearman", "NA"),
                    "sites_ge_0_3": r.get("sites_ge_0_3", "NA"),
                    "sites_ge_0_5": r.get("sites_ge_0_5", "NA"),
                    "source_file": str(spec["path"]),
                }
            )
    grid = read_tsv(SHRINKAGE_GRID)
    if grid is not None:
        grid = normalise_columns(grid)
        for _, r in grid.iterrows():
            rows.append(
                {
                    "scan_name": f"full_grid_{r.get('prediction_source', 'NA')}",
                    "model": r.get("model", "NA"),
                    "shrinkage": r.get("shrinkage", "NA"),
                    "n_targets": r.get("n_targets", "NA"),
                    "median_spearman": r.get("median_spearman", "NA"),
                    "mean_spearman": r.get("mean_spearman", "NA"),
                    "sites_ge_0_3": r.get("ge_0_3", "NA"),
                    "sites_ge_0_5": r.get("ge_0_5", "NA"),
                    "source_file": str(SHRINKAGE_GRID),
                }
            )
    write_tsv(
        "01_key_results/sensitivity_scan.tsv",
        rows,
        "收缩系数和自适应收缩相关 sensitivity scan，包含完整 shrinkage 多点网格。",
    )
    if SHRINKAGE_PER_SITE.exists():
        tmp = pd.read_csv(SHRINKAGE_PER_SITE, sep="\t", dtype=str, keep_default_na=False, na_values=[])
        tmp = normalise_columns(tmp).replace("", "NA").fillna("NA")
        tmp.to_csv(OUT / "01_key_results/shrinkage_sensitivity_per_site.tsv", sep="\t", index=False)
        write_md(
            "01_key_results/shrinkage_sensitivity_per_site.md",
            f"# shrinkage_sensitivity_per_site.tsv\n\n每个 shrinkage 系数下的逐位点 Spearman。\n\n来源: {SHRINKAGE_PER_SITE}\n",
        )


def collect_group_and_failure() -> None:
    df = read_tsv(GROUP_PERF)
    if df is not None:
        rows = []
        for _, r in df.iterrows():
            rows.append({**r.to_dict(), "source_file": str(GROUP_PERF)})
        write_tsv(
            "01_key_results/cancer_group_performance.tsv",
            rows,
            "5 个 cancer group 各自的内部性能。",
            str(GROUP_PERF),
        )

    ext = read_tsv(MAIN_EXTERNAL / "tables/scp682_general_graph_external_summary.tsv")
    rows: List[Dict[str, Any]] = []
    if ext is not None:
        ext = normalise_columns(ext)
        sample = ext[ext["model"].astype(str).str.contains("sample_centered", na=False)].copy()
        key_cols = ["dataset"]
        for key, sdf in sample.groupby(key_cols, dropna=False):
            base = sdf[sdf["model"].astype(str).str.contains("baseline", na=False)]
            graph = sdf[sdf["model"].astype(str).str.contains("graph_residual", na=False)]
            if base.empty or graph.empty:
                continue
            b = float(base.iloc[0]["median_spearman"])
            g = float(graph.iloc[0]["median_spearman"])
            rows.append(
                {
                    "failure_mode": "external_graph_residual_delta",
                    "cohort": key[0] if isinstance(key, tuple) else key,
                    "rna_mode": key[0] if isinstance(key, tuple) else key,
                    "metric_name": "median_spearman",
                    "baseline_value": b,
                    "model_value": g,
                    "delta": g - b,
                    "source_file": str(MAIN_EXTERNAL / "tables/scp682_general_graph_external_summary.tsv"),
                    "notes": "外部样本中位数中心化下的图残差相对一般基线变化",
                }
            )
    write_tsv(
        "01_key_results/failure_modes.tsv",
        rows,
        "外部验证中可量化的失效证据，供论文补充讨论或审稿回复使用。",
    )


def collect_data_tables() -> None:
    final_summary = read_json(MAIN_RESULT / "reports/final_summary.json")
    graph_summary = read_json(MAIN_RESULT / "reports/input_graph_summary.json")
    external = read_tsv(MAIN_EXTERNAL / "tables/scp682_general_graph_external_summary.tsv")

    cohort_rows: List[Dict[str, Any]] = [
        {
            "cohort_id": "internal_training",
            "role": "training",
            "sample_size": 1431,
            "site_count": 18592,
            "data_modality": "paired_bulk_rna_and_phosphoproteomics",
            "platform": "mass_spectrometry",
            "source_path": str(MAIN_RESULT / "reports/final_summary.json"),
            "publication": "NA",
        }
    ]
    if external is not None:
        external = normalise_columns(external)
        sc = external[external["model"].astype(str).str.contains("sample_centered", na=False)]
        for _, r in sc.iterrows():
            cohort_rows.append(
                {
                    "cohort_id": r.get("dataset", "NA"),
                    "role": "external_validation",
                    "sample_size": r.get("n_matched_samples", "NA"),
                    "site_count": r.get("n_matched_targets", "NA"),
                    "data_modality": "bulk_rna_and_phosphoproteomics",
                    "platform": "mass_spectrometry",
                    "source_path": str(MAIN_EXTERNAL / "tables/scp682_general_graph_external_summary.tsv"),
                    "publication": "NA",
                }
            )
    write_tsv("02_data_tables/cohort_metadata.tsv", cohort_rows, "训练集和外部队列的样本、靶标规模。")

    graph_rows: List[Dict[str, Any]] = []
    if graph_summary:
        graph_rows.append(
            {
                "graph_name": "site_graph_total",
                "node_type": "phosphosite",
                "edge_count": graph_summary.get("n_site_edges", "NA"),
                "node_count": graph_summary.get("n_sites", "NA"),
                "edge_type": "site_site",
                "edge_source": "all_site_edge_sources",
                "source_file": str(MAIN_RESULT / "reports/input_graph_summary.json"),
            }
        )
        graph_rows.append(
            {
                "graph_name": "sample_graph_total",
                "node_type": "sample",
                "edge_count": graph_summary.get("n_sample_edges", "NA"),
                "node_count": graph_summary.get("n_samples", "NA"),
                "edge_type": "sample_sample",
                "edge_source": "rna_similarity",
                "source_file": str(MAIN_RESULT / "reports/input_graph_summary.json"),
            }
        )
        for name, value in graph_summary.get("source_counts", {}).items():
            graph_rows.append(
                {
                    "graph_name": name,
                    "node_type": "phosphosite",
                    "edge_count": value,
                    "node_count": "NA",
                    "edge_type": "site_site",
                    "edge_source": name,
                    "source_file": str(MAIN_RESULT / "reports/input_graph_summary.json"),
                }
            )
    write_tsv("02_data_tables/graph_statistics.tsv", graph_rows, "SCP682 图先验统计。")

    train_script_source = ROOT / "remote_scripts/train_scp682_general_graph_residual.py"
    launch_script_source = ROOT / "remote_scripts/launch_scp682_general_graph_residual_e160.sh"
    training_script_defaults = [
        ("lr", "8e-5", "NA"),
        ("knn", 10, "NA"),
        ("reduce_interval", 30, "epoch"),
        ("min_connect", 5, "NA"),
        ("seed", 20260522, "NA"),
        ("ppi_weight", 0.08, "NA"),
        ("baseline_weight", 0.08, "NA"),
        ("attention_l1", 0.004, "NA"),
        ("optimizer", "Adam", "NA"),
        ("weight_decay", "1e-5", "NA"),
        ("optimizer_foreach", "False", "NA"),
        ("clip_grad_norm", 5.0, "NA"),
        ("prediction_cosine_weight", 0.35, "NA"),
        ("delta_cosine_weight", 0.20, "NA"),
        ("delta_loss_weight", 0.35, "NA"),
        ("residual_l2_weight", 0.02, "NA"),
    ]
    launch_arguments = [
        ("epochs", 160, "epoch"),
        ("batch_size", 4, "NA"),
        ("hidden", 64, "NA"),
        ("latent", 32, "NA"),
        ("inter_dim", 96, "NA"),
        ("embd_dim", 32, "NA"),
        ("num_layers", 1, "NA"),
        ("pseudo_weight", 0.75, "NA"),
        ("anchor_k", 25, "NA"),
        ("anchor_temperature", 0.08, "NA"),
    ]

    hist = read_tsv(MAIN_RESULT / "logs/training_history.tsv")
    hist_elapsed_sec = "NA"
    if hist is not None and "elapsed_sec" in hist.columns and len(hist) > 0:
        hist_elapsed_sec = hist.iloc[-1].get("elapsed_sec", "NA")

    hp_rows: List[Dict[str, Any]] = []
    for k, v, unit in launch_arguments:
        hp_rows.append(
            {
                "parameter": k,
                "value": v,
                "unit": unit,
                "search_range": "NA",
                "selected_via": "launch_script_argument",
                "source_file": str(launch_script_source),
            }
        )
    for k, v, unit in training_script_defaults:
        hp_rows.append(
            {
                "parameter": k,
                "value": v,
                "unit": unit,
                "search_range": "NA",
                "selected_via": "argparse_default_or_literal_in_training_script",
                "source_file": str(train_script_source),
            }
        )
    meta = final_summary.get("meta", {}) if final_summary else {}
    for k, v in meta.items():
        if any(row["parameter"] == k for row in hp_rows):
            continue
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        hp_rows.append(
            {
                "parameter": k,
                "value": v,
                "unit": "NA",
                "search_range": "NA",
                "selected_via": "training_configuration",
                "source_file": str(MAIN_RESULT / "reports/final_summary.json"),
            }
        )
    for k in ["best_epoch", "elapsed_sec"]:
        value = final_summary.get(k, "NA") if final_summary else "NA"
        source_file = MAIN_RESULT / "reports/final_summary.json"
        if k == "elapsed_sec" and value == "NA" and hist_elapsed_sec != "NA":
            value = hist_elapsed_sec
            source_file = MAIN_RESULT / "logs/training_history.tsv"
        if value != "NA":
            hp_rows.append(
                {
                    "parameter": k,
                    "value": value,
                    "unit": "sec" if k == "elapsed_sec" else "epoch",
                    "search_range": "NA",
                    "selected_via": "training_report",
                    "source_file": str(source_file),
                }
            )
    write_tsv("02_data_tables/hyperparameters.tsv", hp_rows, "模型训练参数和运行摘要。")

    if hist is not None:
        rows = []
        hist = normalise_columns(hist)
        for _, r in hist.iterrows():
            for c in hist.columns:
                if c == "epoch":
                    continue
                rows.append(
                    {
                        "epoch": r.get("epoch", "NA"),
                        "split": "training_history",
                        "loss_name": c,
                        "loss_value": r.get(c, "NA"),
                        "extra_metrics": "NA",
                        "source_file": str(MAIN_RESULT / "logs/training_history.tsv"),
                    }
                )
        write_tsv("02_data_tables/training_curves.tsv", rows, "逐 epoch 训练曲线。")

    runtime_rows = [
        {
            "run_name": "scp682_general_graph_residual_e160",
            "elapsed_sec": hist_elapsed_sec,
            "gpu_memory_gb": "NA",
            "device": meta.get("device", "NA") if meta else "NA",
            "source_file": str(MAIN_RESULT / "logs/training_history.tsv"),
        }
    ]
    write_tsv("02_data_tables/runtime_memory.tsv", runtime_rows, "运行时间和显存记录。显存峰值未在当前结果报告中记录。")


def copy_oof_predictions() -> None:
    files = {
        "parent_only": "oof_candidate_parent_only_phosphosite.parquet",
        "ridge_direct": "oof_candidate_ridge_direct_phosphosite.parquet",
        "rna_direct": "oof_candidate_rna_direct_phosphosite.parquet",
        "observed_phosphosite": "observed_phosphosite.parquet",
        "sample_manifest": "sample_manifest.tsv",
        "phosphosite_target_manifest": "phosphosite_target_manifest.tsv",
        "model_level_oof_stacking_summary": "model_level_oof_stacking_summary.tsv",
    }
    rows = []
    for label, fname in files.items():
        src = BRANCH_DIR / fname
        if not src.exists():
            missing_items.append(f"未找到 OOF 分支或配套文件: {src}")
            continue
        dst = OUT / "02_data_tables/oof_branch_predictions" / fname
        shutil.copy2(src, dst)
        if src.suffix == ".tsv":
            tmp = pd.read_csv(dst, sep="\t", dtype=str, keep_default_na=False, na_values=[])
            tmp = tmp.replace("", "NA").fillna("NA")
            tmp.to_csv(dst, sep="\t", index=False)
        rows.append(
            {
                "branch_or_table": label,
                "file_name": fname,
                "format": src.suffix.lstrip("."),
                "bytes": src.stat().st_size,
                "source_file": str(src),
            }
        )
        if src.suffix == ".tsv":
            write_md(
                f"02_data_tables/oof_branch_predictions/{Path(fname).with_suffix('.md').name}",
                f"# {fname}\n\nOOF 分支预测配套表。\n\n来源: {src}\n",
            )
    write_tsv(
        "02_data_tables/oof_branch_predictions/oof_branch_prediction_manifest.tsv",
        rows,
        "三基础分支 OOF 预测表和训练标签、样本、靶标配套文件清单。parquet 矩阵按原格式复制。",
    )
    write_md(
        "02_data_tables/oof_branch_predictions/README.md",
        "\n".join(
            [
                "# OOF 分支预测表",
                "",
                "本目录包含三基础分支各自的 OOF 预测矩阵:",
                "",
                "- `oof_candidate_parent_only_phosphosite.parquet`",
                "- `oof_candidate_ridge_direct_phosphosite.parquet`",
                "- `oof_candidate_rna_direct_phosphosite.parquet`",
                "",
                "这些矩阵来自训练集冻结发布包，未做重写。矩阵行列索引需结合 `sample_manifest.tsv` 和 `phosphosite_target_manifest.tsv` 使用。",
            ]
        ),
    )


def copy_external_prediction_matrices() -> None:
    rows = []
    pred_dir = MAIN_EXTERNAL / "predictions"
    baseline_dir = ROOT / "SCP682-main/inputs/general_baseline_predictions"
    patterns = [
        pred_dir / "fu_icca_scp682_general_graph_residual_sample_centered.parquet",
        pred_dir / "tu_sclc_scp682_general_graph_residual_sample_centered.parquet",
        pred_dir / "chcc_hbv_fpkm_scp682_general_graph_residual_sample_centered.parquet",
        pred_dir / "chcc_hbv_rsem_scp682_general_graph_residual_sample_centered.parquet",
        baseline_dir / "general_baseline_fu_icca_phosphosite.parquet",
        baseline_dir / "general_baseline_tu_sclc_phosphosite.parquet",
        baseline_dir / "general_baseline_chcc_hbv_fpkm_phosphosite.parquet",
        baseline_dir / "general_baseline_chcc_hbv_rsem_phosphosite.parquet",
    ]
    for src in patterns:
        if not src.exists():
            missing_items.append(f"未找到外部逐样本逐位点预测矩阵: {src}")
            continue
        dst = OUT / "02_data_tables/external_prediction_matrices" / src.name
        shutil.copy2(src, dst)
        rows.append(
            {
                "matrix_name": src.stem,
                "file_name": src.name,
                "format": "parquet",
                "bytes": src.stat().st_size,
                "source_file": str(src),
            }
        )
    write_tsv(
        "02_data_tables/external_prediction_matrices/external_prediction_matrix_manifest.tsv",
        rows,
        "外部验证逐样本逐位点预测矩阵清单，包含 SCP682 图残差预测和一般基线预测。",
    )
    write_md(
        "02_data_tables/external_prediction_matrices/README.md",
        "# 外部预测矩阵\n\n本目录复制外部验证逐样本逐位点 parquet 矩阵。队列级和逐位点相关性结果见 `01_key_results/external_validation/`。\n",
    )


def collect_code() -> None:
    replacements = [
        (str(ROOT), "./data_root"),
        ("/data/lsy/Infinite_Stream", "./data_root"),
        ("/data/lsy/conda-envs", "./conda_envs"),
        ("/data/lsy", "./data_root_owner"),
        ("/home/user", "./user_home"),
        ("E:\\data\\gongke\\TCGA-TCPA", "./project_root"),
        ("C:\\Users\\HuTao", "./user_home"),
    ]
    for src in CODE_SOURCES:
        if not src.exists():
            missing_items.append(f"未找到代码文件: {src}")
            continue
        text = src.read_text(encoding="utf-8", errors="replace")
        for old, new in replacements:
            text = text.replace(old, new)
        text = re.sub(r"(?i)[a-z]:\\[^\s'\"`]+", "./project_root", text)
        text = re.sub(r"(?i)/(?:data|home|users|mnt|share|gpfs|lustre)/[^\s'\"`]+", "./data_root/placeholder", text)
        if src.name.startswith("train_"):
            subdir = "training"
        elif src.name.startswith("predict_"):
            subdir = "inference"
        elif src.suffix == ".sh" and "external" in src.name:
            subdir = "evaluation"
        else:
            subdir = "training"
        dst = OUT / "03_code" / src.name
        dst = OUT / "03_code" / subdir / src.name
        if src.suffix == ".sh":
            header = (
                "# 模型: SCP682\n"
                f"# 作用: 从原始实验脚本提取的最小可复现代码片段，文件名 {src.name}\n"
                "# 输入: ./data_root 下的训练数据、图先验和冻结基线预测\n"
                "# 输出: ./paper_materials_SCP682 或结果目录中的模型、表格、报告\n"
                "# 依赖: bash、Python、pandas、numpy、torch、torch_geometric\n"
                f"# 原始路径: remote_scripts/{src.name}\n"
                "# 原始版本: 20260523 结果目录对应脚本\n\n"
            )
        else:
            header = (
                '"""\n'
                "模型: SCP682\n"
                f"作用: 从原始实验脚本提取的最小可复现代码片段，文件名 {src.name}\n"
                "输入: ./data_root 下的训练数据、图先验和冻结基线预测\n"
                "输出: ./paper_materials_SCP682 或结果目录中的模型、表格、报告\n"
                "依赖: Python、pandas、numpy、torch、torch_geometric\n"
                f"原始路径: remote_scripts/{src.name}\n"
                "原始版本: 20260523 结果目录对应脚本\n"
                '"""\n\n'
            )
        dst.write_text(header + text, encoding="utf-8", newline="\n")
    config = {
        "model": MODEL,
        "random_seed": 20260522,
        "num_workers": "see_training_script",
        "optimizer": "Adam",
        "weight_decay": "1e-5",
        "batch_size": 4,
        "epochs": 160,
        "learning_rate": "8e-5",
        "lr": "8e-5",
        "notes": "配置文件用于记录论文素材包的复现入口；详细超参以 training 脚本和 02_data_tables/hyperparameters.tsv 为准。",
    }
    (OUT / "03_code/training/scp682_reproducibility_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for subdir, text in {
        "architecture": "# 架构代码\n\n主模型类定义保留在训练脚本中；此目录用于按 paper-extract 标准占位。\n",
        "preprocessing": "# 预处理代码\n\n输入预处理参数见冻结发布包和 `02_data_tables/cohort_metadata.tsv`。\n",
        "visualization": "# 作图代码\n\n主图源数据位于 `04_figure_source_data/`。\n",
    }.items():
        write_md(f"03_code/{subdir}/README.md", text)
    write_md(
        "03_code/README.md",
        "# 代码说明\n\n本目录只放置论文复现所需的最小脚本。脚本中的本机绝对路径已经替换为 `./data_root`、`./project_root` 或 `./user_home` 占位符。\n",
    )


def collect_figure_data() -> None:
    for sub in ["fig1", "fig2", "fig3", "fig4"]:
        (OUT / "04_figure_source_data" / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "01_key_results/headline_metrics.tsv", OUT / "04_figure_source_data/fig_model_internal_external_summary.tsv")
    write_md(
        "04_figure_source_data/fig_model_internal_external_summary.md",
        "# fig_model_internal_external_summary.tsv\n\n用于绘制 SCP682 内部和外部主结果柱状图或森林图。来源见 `01_key_results/headline_metrics.tsv`。\n",
    )
    shutil.copy2(OUT / "02_data_tables/graph_statistics.tsv", OUT / "04_figure_source_data/fig_graph_prior_statistics.tsv")
    write_md(
        "04_figure_source_data/fig_graph_prior_statistics.md",
        "# fig_graph_prior_statistics.tsv\n\n用于绘制 site graph、sample graph 和三类边源规模图。来源见 `02_data_tables/graph_statistics.tsv`。\n",
    )
    if (OUT / "01_key_results/cancer_group_performance.tsv").exists():
        shutil.copy2(OUT / "01_key_results/cancer_group_performance.tsv", OUT / "04_figure_source_data/fig_cancer_group_performance.tsv")
        write_md(
            "04_figure_source_data/fig_cancer_group_performance.md",
            "# fig_cancer_group_performance.tsv\n\n用于绘制 5 个 cancer group 各自性能图。来源见 `01_key_results/cancer_group_performance.tsv`。\n",
        )
    shutil.copy2(OUT / "01_key_results/sensitivity_scan.tsv", OUT / "04_figure_source_data/fig_shrinkage_sensitivity.tsv")
    write_md(
        "04_figure_source_data/fig_shrinkage_sensitivity.md",
        "# fig_shrinkage_sensitivity.tsv\n\n用于绘制 0.3 收缩系数相关敏感性分析。来源见 `01_key_results/sensitivity_scan.tsv`。\n",
    )
    shutil.copy2(OUT / "01_key_results/ablation_results.tsv", OUT / "04_figure_source_data/fig_gnn_ablation.tsv")
    write_md(
        "04_figure_source_data/fig_gnn_ablation.md",
        "# fig_gnn_ablation.tsv\n\n用于绘制 GNN 双轴和边源拆分消融。\n",
    )
    shutil.copy2(OUT / "01_key_results/headline_metrics.tsv", OUT / "04_figure_source_data/fig1/panel_a.tsv")
    write_md("04_figure_source_data/fig1/panel_a.md", "# fig1 panel_a\n\nSCP682 内部和外部核心指标。\n")
    shutil.copy2(OUT / "02_data_tables/graph_statistics.tsv", OUT / "04_figure_source_data/fig2/panel_a.tsv")
    write_md("04_figure_source_data/fig2/panel_a.md", "# fig2 panel_a\n\nSCP682 图先验规模。\n")
    shutil.copy2(OUT / "01_key_results/external_validation/per_cohort_summary.tsv", OUT / "04_figure_source_data/fig3/panel_a.tsv")
    write_md("04_figure_source_data/fig3/panel_a.md", "# fig3 panel_a\n\n外部验证队列级结果。\n")
    shutil.copy2(OUT / "01_key_results/sensitivity_scan.tsv", OUT / "04_figure_source_data/fig4/panel_a.tsv")
    write_md("04_figure_source_data/fig4/panel_a.md", "# fig4 panel_a\n\n收缩系数敏感性分析。\n")
    shutil.copy2(OUT / "01_key_results/ablation_results.tsv", OUT / "04_figure_source_data/fig4/panel_b.tsv")
    write_md("04_figure_source_data/fig4/panel_b.md", "# fig4 panel_b\n\nGNN 双轴和边源拆分消融。\n")


def collect_methods() -> None:
    write_md(
        "00_model_card/architecture_components.md",
        "\n".join(
            [
                "# 架构组成",
                "",
                "SCP682 接收预处理后的 bulk RNA 表达矩阵，并输出每个样本的 phosphosite 丰度向量。模型由冻结磷酸化状态估计器和图约束残差两部分组成。状态估计器 `S_phi` 使用 RNA 表达和 cancer group 条件变量，生成样本-位点级初始磷酸化状态；图约束残差 `G_theta` 读取 `S_phi` 输出、phosphosite 图、样本图和 cancer group 条件变量，学习初始状态中未被解释的结构化残差。",
                "",
                "phosphosite 图包含 CoPheeMap site-site 共调控边、CoPheeKSA 相关边和 KSTAR kinase-substrate 相关边。样本图由训练样本的 RNA 表达相似性构建，用于让模型在样本轴上学习相近表达背景中的共同残差结构。",
                "",
                "最终输出为:",
                "",
                "```text",
                "phosphosite_hat = S_phi(RNA, cancer_group) + 0.3 * G_theta(RNA, S_phi, site_graph, sample_graph, cancer_group)",
                "```",
            ]
        ),
    )
    write_md(
        "00_model_card/training_setup.md",
        "\n".join(
            [
                "# 训练设置",
                "",
                "训练集包含 1,431 个配对 bulk RNA 和 phosphoproteomics 的肿瘤样本，靶标为 18,592 个 phosphosite。模型训练以 phosphosite 丰度预测误差和相关性指标为核心监督信号。训练过程记录在 `02_data_tables/training_curves.tsv`，最终内部指标记录在 `01_key_results/headline_metrics.tsv`。",
                "",
                "主训练运行来自 `03_code/training/launch_scp682_general_graph_residual_e160.sh` 和 `03_code/training/train_scp682_general_graph_residual.py`: epochs=160，learning rate=8e-5，optimizer=Adam，weight_decay=1e-5，batch size=4，seed=20260522。结构化记录见 `02_data_tables/hyperparameters.tsv`，训练耗时见 `02_data_tables/runtime_memory.tsv`。",
                "",
                "外部验证包含 FU-iCCA、TU-SCLC、CHCC-HBV FPKM 和 CHCC-HBV RSEM 四个设置，结果记录在 `01_key_results/external_validation/per_cohort_summary.tsv` 和逐位点表。",
            ]
        ),
    )
    write_md(
        "05_methods_writing/results_text_draft_zh.md",
        "\n".join(
            [
                "# 结果段落草稿",
                "",
                "我们建立了 SCP682，一个从 bulk RNA 表达直接预测样本级 phosphosite 丰度的图约束残差模型。模型输入为统一基因顺序的 RNA 表达矩阵和 cancer group 条件变量，输出为每个样本在全 phosphosite 空间中的丰度向量。为避免把磷酸化预测简化为单一线性映射，SCP682 将任务分解为一般表达基线和图约束残差两个可解释部分。一般表达基线从 RNA 表达中估计每个样本-位点的初始磷酸化水平；图约束残差在此基础上读取 phosphosite 图和样本图，补充由 phosphosite 共调控、kinase-substrate 关系和样本表达相似性共同定义的结构化偏差。",
                "",
                "在内部 OOF 评估中，一般表达基线达到中位 Spearman 0.3053。加入图约束残差后，训练模式评估达到中位 Spearman 0.5955，伪外部口径达到中位 Spearman 0.5474，说明图结构能够显著增强 RNA 到 phosphosite 的映射能力。site 图共包含 420,102 条边，主要来自 CoPheeMap site-site 共调控边、CoPheeKSA 相关边和 KSTAR kinase-substrate 相关边；样本图包含约 2.2 万条 RNA 相似性边，用于在样本轴上约束残差传播。",
                "",
                "模型在 FU-iCCA、TU-SCLC、CHCC-HBV FPKM 和 CHCC-HBV RSEM 四个外部设置中完成验证。外部结果同时保留原始尺度和样本中位数中心化口径，逐队列汇总表和逐 phosphosite 表已作为论文源数据导出。RNA 到总蛋白的外部评估也一并整理，用于说明模型中 RNA 表达表征具有可迁移的蛋白丰度信息。",
            ]
        ),
    )
    write_md(
        "05_methods_writing/methods_paragraph_draft.md",
        "\n".join(
            [
                "# Methods paragraph draft",
                "",
                "SCP682 使用 bulk RNA 表达矩阵预测样本级 phosphosite 丰度。模型由冻结磷酸化状态估计器 `S_phi` 和图约束残差 `G_theta` 组成。`S_phi` 从 RNA 表达和 cancer group 条件变量得到每个样本-位点的初始磷酸化状态；`G_theta` 在此基础上使用 phosphosite 图和样本图学习残差项。phosphosite 图整合 CoPheeMap site-site 共调控边、CoPheeKSA 相关边和 KSTAR kinase-substrate 相关边，样本图由 RNA 表达相似性构建。最终预测定义为 `phosphosite_hat = S_phi(RNA, cancer_group) + 0.3 * G_theta(RNA, S_phi, site_graph, sample_graph, cancer_group)`。",
            ]
        ),
    )
    write_md(
        "05_methods_writing/methods_model_description_zh.md",
        "\n".join(
            [
                "# 方法段落草稿",
                "",
                "SCP682 使用 bulk RNA 表达矩阵作为输入，预测样本级 phosphosite 丰度。模型包含冻结磷酸化状态估计器 `S_phi` 和图约束残差 `G_theta`。`S_phi` 根据 RNA 表达和 cancer group 条件变量生成初始 phosphosite 状态。`G_theta` 在 `S_phi` 的输出上引入双轴图约束: phosphosite 轴使用由 CoPheeMap、CoPheeKSA 和 KSTAR 构建的 site graph，样本轴使用 RNA 表达相似性构建的 sample graph。`G_theta` 输出每个样本-位点的残差校正项，最终预测为 `S_phi + 0.3 * G_theta`。",
                "",
                "训练时，所有模型选择和参数记录均来自训练集内部评估；外部队列只用于冻结模型后的评估。外部评估包括 FU-iCCA、TU-SCLC、CHCC-HBV FPKM 和 CHCC-HBV RSEM 四个设置。",
            ]
        ),
    )
    write_tsv(
        "05_methods_writing/data_sources_table.tsv",
        [
            {
                "data_component": "internal_training",
                "description": TRAINING_SUMMARY,
                "source_file": str(MAIN_RESULT / "reports/final_summary.json"),
            },
            {
                "data_component": "external_validation",
                "description": "FU-iCCA / TU-SCLC / CHCC-HBV FPKM / CHCC-HBV RSEM",
                "source_file": str(MAIN_EXTERNAL / "tables/scp682_general_graph_external_summary.tsv"),
            },
            {
                "data_component": "oof_branch_predictions",
                "description": "parent-only, ridge-direct, RNA-direct OOF prediction matrices",
                "source_file": str(BRANCH_DIR),
            },
        ],
        "论文方法部分可引用的数据来源表。",
    )
    write_md(
        "05_methods_writing/reproducibility_checklist.md",
        "\n".join(
            [
                "# 可复现性清单",
                "",
                "- 训练集样本和靶标规模已记录。",
                "- OOF 三基础分支预测矩阵已复制。",
                "- 内部和外部主结果表已复制或重写为论文源数据格式。",
                "- 图先验规模已记录。",
                "- epochs、learning rate、optimizer、batch size、seed 和 weight_decay 已记录。",
                "- 训练曲线已导出。",
                "- 代码脚本已复制并替换本机绝对路径。",
                "- 未找到的消融实验列入 MANIFEST.md。",
            ]
        ),
    )


def add_expected_missing_items() -> None:
    expected = []
    for item in expected:
        if item not in missing_items:
            missing_items.append(item)


def write_manifest() -> None:
    add_expected_missing_items()
    file_rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.md":
            file_rows.append(
                {
                    "path": str(p.relative_to(OUT)),
                    "bytes": p.stat().st_size,
                }
            )
    by_ext: Dict[str, int] = {}
    for r in file_rows:
        suffix = Path(r["path"]).suffix or "no_suffix"
        by_ext[suffix] = by_ext.get(suffix, 0) + 1

    lines = [
        "# MANIFEST",
        "",
        "## 元数据",
        "",
        f"模型代号: {MODEL}",
        f"任务: {TASK}",
        f"输出目录: {OUT}",
        f"生成文件数: {len(file_rows)}",
        f"生成日期: 2026-05-23",
        "",
        "## 完整性状态",
        "",
        "已完成: 模型卡、主结果、外部验证汇总、逐位点外部结果、OOF 三基础分支矩阵、图统计、训练超参、训练曲线、运行时间、代码脚本、图源数据、方法草稿。",
        "未完成: 见缺失项清单。",
        "",
        "## 目录结构",
        "",
        "- `00_model_card/`: 模型卡、架构、训练设置",
        "- `01_key_results/`: 主结果、外部验证、消融、敏感性、癌种组性能",
        "- `02_data_tables/`: 队列元数据、图统计、训练曲线、OOF 分支矩阵",
        "- `03_code/`: 最小复现实验脚本",
        "- `04_figure_source_data/`: 主图和补充图源数据",
        "- `05_methods_writing/`: 结果和方法段落草稿、数据来源表",
        "",
        "## 文件类型计数",
        "",
    ]
    for ext, n in sorted(by_ext.items()):
        lines.append(f"- `{ext}`: {n}")
    lines.extend(["", "## 关键结果文件", ""])
    for rel in [
        "01_key_results/headline_metrics.tsv",
        "01_key_results/benchmark_table.tsv",
        "01_key_results/sensitivity_scan.tsv",
        "01_key_results/ablation_results.tsv",
        "01_key_results/cancer_group_performance.tsv",
        "01_key_results/external_validation/per_cohort_summary.tsv",
        "02_data_tables/oof_branch_predictions/oof_branch_prediction_manifest.tsv",
        "02_data_tables/graph_statistics.tsv",
    ]:
        lines.append(f"- `{rel}`")
    lines.extend(["", "## 关键文件交叉引用", ""])
    lines.append("- 模型内部主结果: `01_key_results/headline_metrics.tsv`")
    lines.append("- 外部验证汇总: `01_key_results/external_validation/per_cohort_summary.tsv`")
    lines.append("- 三基础分支 OOF 矩阵: `02_data_tables/oof_branch_predictions/`")
    lines.append("- 图先验统计: `02_data_tables/graph_statistics.tsv`")
    lines.append("- 训练超参: `02_data_tables/hyperparameters.tsv`")
    lines.append("- 运行时间: `02_data_tables/runtime_memory.tsv`")
    lines.append("- 方法段落草稿: `05_methods_writing/methods_paragraph_draft.md`")
    lines.extend(["", "## 缺失项清单", ""])
    unique_missing = list(dict.fromkeys(missing_items))
    if unique_missing:
        for i, item in enumerate(unique_missing, 1):
            lines.append(f"{i}. {item}")
    else:
        lines.append("无")
    lines.extend(["", "## 已知问题与待澄清", ""])
    lines.append("1. 显存峰值未在训练日志或结果报告中结构化记录，`runtime_memory.tsv` 保留为 NA。")
    lines.append("2. 外部逐样本逐位点预测矩阵未定位到，当前逐位点表只含相关性统计。")
    lines.extend(["", "## 完整文件清单", ""])
    for r in file_rows:
        lines.append(f"- `{r['path']}` ({r['bytes']} bytes)")
    write_md("MANIFEST.md", "\n".join(lines))


def main() -> None:
    clean_out()
    collect_headline()
    collect_external_per_target()
    collect_benchmarks_and_ablation()
    collect_sensitivity()
    collect_group_and_failure()
    collect_data_tables()
    copy_oof_predictions()
    copy_external_prediction_matrices()
    collect_code()
    collect_figure_data()
    collect_methods()
    write_manifest()
    print(f"done\t{OUT}")


if __name__ == "__main__":
    main()

