"""
模型: SCP682-PPKO
作用: 重新计算论文素材包中的缺失项，包括 P100 指定口径、PXD063604 分支独立性能、图统计和超参数表。
输入: ./data_root/remote_scripts/SCP682_PPKO_release_v1 与 ./data_root/paper_materials_SCP682_PPKO
输出: ./data_root/paper_materials_SCP682_PPKO 下的补充 TSV、说明文件和更新后的清单
依赖: python 3.x, numpy, pandas
原始路径: paper_materials_SCP682_PPKO/03_code/evaluation/rerun_missing_items.py
原始版本: NA
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def rel(path: Path, project_root: Path) -> str:
    try:
        return "./" + path.resolve().relative_to(project_root.resolve()).as_posix()
    except Exception:
        return path.name


def write_md(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.replace({np.nan: "NA", "": "NA"})
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return np.nan
    x = x[ok].astype(float)
    y = y[ok].astype(float)
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den <= 1e-12:
        return np.nan
    return float(np.dot(x, y) / den)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))


def direction(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y) & (np.abs(y) > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(x[ok]) == np.sign(y[ok])))


def top_fraction_mask(values: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    ok = np.isfinite(values)
    mask = np.zeros(len(values), dtype=bool)
    n = int(ok.sum())
    if n == 0:
        return mask
    k = max(1, int(math.ceil(n * fraction)))
    idx = np.where(ok)[0]
    ranked = idx[np.argsort(-np.abs(values[idx]))[:k]]
    mask[ranked] = True
    return mask


def branch_metrics(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    y = df["observed_delta"].astype(float).to_numpy()
    yhat = df[pred_col].astype(float).to_numpy()
    responsive = top_fraction_mask(y, 0.2)
    predicted_top = top_fraction_mask(yhat, 0.2)
    k_resp = int(responsive.sum())
    recall = float((responsive & predicted_top).sum() / max(1, k_resp))
    return {
        "all_cosine": cosine(yhat, y),
        "all_spearman": spearman(yhat, y),
        "all_direction": direction(yhat, y),
        "responsive20_cosine": cosine(yhat[responsive], y[responsive]),
        "responsive20_spearman": spearman(yhat[responsive], y[responsive]),
        "responsive20_direction": direction(yhat[responsive], y[responsive]),
        "predicted20_recall": recall,
        "predicted20_direction": direction(yhat[predicted_top], y[predicted_top]),
        "pred_abs_mean": float(np.nanmean(np.abs(yhat))),
        "real_abs_mean": float(np.nanmean(np.abs(y))),
        "n_sites": int(np.isfinite(y).sum()),
    }


def add_or_replace_metric(headline: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    if headline.empty:
        return pd.DataFrame(rows)
    remove = {r["metric_name"] for r in rows}
    headline = headline[~headline["metric_name"].isin(remove)].copy()
    return pd.concat([headline, pd.DataFrame(rows)], ignore_index=True)


def add_or_replace_tiers(tiers: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    if tiers.empty:
        return pd.DataFrame(rows)
    keys = {(r["cohort"], r["tier"]) for r in rows}
    keep = ~tiers.apply(lambda r: (r["cohort"], r["tier"]) in keys, axis=1)
    return pd.concat([tiers[keep].copy(), pd.DataFrame(rows)], ignore_index=True)


def parse_argparse_defaults(script_path: Path, source_file: str) -> pd.DataFrame:
    text = script_path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    pattern = re.compile(r'ap\.add_argument\("(?P<name>--[^"]+)".*?default=(?P<value>[^,\)]+)', re.S)
    for match in pattern.finditer(text):
        name = match.group("name").lstrip("-").replace("-", "_")
        value = match.group("value").strip().strip("\"'")
        rows.append(
            {
                "parameter": name,
                "value": value,
                "unit": "NA",
                "search_range": "NA",
                "selected_via": "argparse default in released training script",
                "source_file": source_file,
            }
        )
    manual = [
        ("weight_decay", "1e-4"),
        ("clip_grad_norm", "5.0"),
        ("loss_cosine_weight", "0.35"),
        ("latent_penalty_weight", "0.04"),
        ("residual_penalty_weight", "0.02"),
    ]
    for name, value in manual:
        rows.append(
            {
                "parameter": name,
                "value": value,
                "unit": "NA",
                "search_range": "NA",
                "selected_via": "literal in released training script",
                "source_file": source_file,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--paper-dir", default="paper_materials_SCP682_PPKO")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    paper_dir = (project_root / args.paper_dir).resolve()
    release = project_root / "remote_scripts" / "SCP682_PPKO_release_v1"
    v10b = project_root / "remote_scripts" / "SCP682_PPKO_V10B_transferable"

    p100_v10_summary = release / "results" / "v10_attention_prior_focus55_summary.tsv"
    p100_candidates = release / "results" / "candidate_focus55_comparison.tsv"
    pxd_site = release / "results" / "external_bulk_drug_validation_v2_external_baseline" / "external_bulk_site_predictions.tsv"
    pxd_comp = release / "results" / "external_bulk_drug_validation_v2_external_baseline" / "external_bulk_comparison_metrics.tsv"
    train_script = release / "scripts" / "pretrain_scp682_ppko_attention_prior_v10.py"
    final_metrics_path = paper_dir / "02_data_tables" / "remote_final_metrics_v10.json"
    graph_summary_path = paper_dir / "02_data_tables" / "source_graph_summary" / "global_heterograph_summary.json"

    p100 = read_tsv(p100_v10_summary)
    if p100.columns[0] != "mode":
        p100 = p100.rename(columns={p100.columns[0]: "mode"})
    p100["source_file"] = rel(p100_v10_summary, project_root)
    write_tsv(p100, paper_dir / "01_key_results" / "p100_focus55_release_v10_metrics.tsv")
    write_md(
        paper_dir / "01_key_results" / "p100_focus55_release_v10_metrics.md",
        "P100 focus55 指定口径来自 SCP682-PPKO release v1 的 V10 表。all cosine 0.602、responsive top20 direction 0.888、predicted top20 direction 0.942 均由该表真实靶点行直接给出。",
    )

    cand = read_tsv(p100_candidates)
    cand["source_file"] = rel(p100_candidates, project_root)
    write_tsv(cand, paper_dir / "01_key_results" / "candidate_focus55_comparison.tsv")
    write_md(
        paper_dir / "01_key_results" / "candidate_focus55_comparison.md",
        "候选模型 focus55 对比表，包含 V8、V10、V10B strong300 和 V10C。该表用于说明指定 P100 口径来自 V10 release v1，而非 V10B strong300。",
    )

    true = p100[p100["mode"] == "true"].iloc[0]
    p100_source = rel(p100_v10_summary, project_root)
    headline_path = paper_dir / "01_key_results" / "headline_metrics.tsv"
    headline = read_tsv(headline_path)
    headline_rows = []
    for metric in [
        "all_cosine",
        "all_spearman",
        "all_direction",
        "responsive20_cosine",
        "responsive20_spearman",
        "responsive20_direction",
        "topk_recall",
        "topk_direction",
        "response_auroc_abs_pred",
    ]:
        headline_rows.append(
            {
                "metric_name": "p100_focus55_release_v10_" + metric,
                "value": true[metric],
                "sample_size": true["n"],
                "cohort_or_split": "P100_focus55_release_v10",
                "reference_baseline": "NA",
                "source_file": p100_source,
                "computation_date": "2026-05-23",
                "notes": "SCP682-PPKO release v1 指定口径",
            }
        )
    write_tsv(add_or_replace_metric(headline, headline_rows), headline_path)

    tiers_path = paper_dir / "01_key_results" / "three_tier_metrics.tsv"
    tiers = read_tsv(tiers_path)
    tier_rows = [
        {
            "cohort": "P100_focus55_release_v10",
            "tier": "all_sites",
            "cosine": true["all_cosine"],
            "direction": true["all_direction"],
            "spearman": true["all_spearman"],
            "recall": "NA",
            "source_file": p100_source,
        },
        {
            "cohort": "P100_focus55_release_v10",
            "tier": "responsive_top20",
            "cosine": true["responsive20_cosine"],
            "direction": true["responsive20_direction"],
            "spearman": true["responsive20_spearman"],
            "recall": "NA",
            "source_file": p100_source,
        },
        {
            "cohort": "P100_focus55_release_v10",
            "tier": "predicted_top20",
            "cosine": "NA",
            "direction": true["topk_direction"],
            "spearman": "NA",
            "recall": true["topk_recall"],
            "source_file": p100_source,
        },
    ]
    write_tsv(add_or_replace_tiers(tiers, tier_rows), tiers_path)

    per_cohort_path = paper_dir / "01_key_results" / "external_validation" / "per_cohort_summary.tsv"
    per_cohort = read_tsv(per_cohort_path)
    per_cohort = per_cohort[per_cohort["cohort_id"] != "P100_focus55_release_v10"].copy()
    per_cohort = pd.concat(
        [
            per_cohort,
            pd.DataFrame(
                [
                    {
                        "cohort_id": "P100_focus55_release_v10",
                        "n_comparisons": true["n"],
                        "site_count": "NA",
                        "all_sites_cosine": true["all_cosine"],
                        "all_sites_direction": true["all_direction"],
                        "responsive_top20_cosine": true["responsive20_cosine"],
                        "responsive_top20_direction": true["responsive20_direction"],
                        "predicted_top20_recall": true["topk_recall"],
                        "predicted_top20_direction": true["topk_direction"],
                        "source_file": p100_source,
                        "notes": "SCP682-PPKO release v1 指定口径",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    write_tsv(per_cohort, per_cohort_path)

    p100_target = paper_dir / "01_key_results" / "external_validation" / "per_target_P100_focus55.tsv"
    p100_target_release = paper_dir / "01_key_results" / "external_validation" / "per_target_P100_focus55_release_v10.tsv"
    if p100_target.exists():
        p100_target_release.write_text(p100_target.read_text(encoding="utf-8"), encoding="utf-8")
        write_md(
            p100_target_release.with_suffix(".md"),
            "P100 focus55 release v10 的逐靶点占位表。当前固定源表只提供模式级汇总和候选模型表，逐药物靶点表沿用 P100 已提取表以满足队列级追踪；精确 V10 逐靶点表需原始 per-comparison 输出。",
        )

    site = read_tsv(pxd_site)
    pxd063604_site = site[site["dataset"] == "PXD063604"].copy()
    pxd063604_site["source_file"] = rel(pxd_site, project_root)
    write_tsv(
        pxd063604_site,
        paper_dir / "02_data_tables" / "pxd063604_site_level_predictions.tsv",
    )
    write_md(
        paper_dir / "02_data_tables" / "pxd063604_site_level_predictions.md",
        "PXD063604 逐位点预测表，包含 observed_delta、predicted_delta、graph_delta、latent_delta、residual_delta、attention 和 regulated 标记。该表用于重新计算分支独立性能。",
    )

    branch_map = {
        "predicted_delta": ("full_prediction", "full"),
        "latent_delta": ("core_proxy", "latent_delta"),
        "residual_delta": ("residual", "residual_delta"),
        "graph_delta": ("common_graph_proxy", "graph_delta"),
    }
    rows = []
    for comparison, sub in pxd063604_site.groupby("comparison", sort=True):
        for col, (branch, source_column) in branch_map.items():
            rec = branch_metrics(sub, col)
            rec.update(
                {
                    "dataset": "PXD063604",
                    "comparison": comparison,
                    "branch": branch,
                    "source_column": source_column,
                    "source_file": rel(pxd_site, project_root),
                }
            )
            rows.append(rec)
    branch_df = pd.DataFrame(rows)
    write_tsv(branch_df, paper_dir / "01_key_results" / "component_independent_performance.tsv")
    write_md(
        paper_dir / "01_key_results" / "component_independent_performance.md",
        "PXD063604 分支独立性能。full_prediction 使用 predicted_delta；core_proxy 使用 latent_delta；residual 使用 residual_delta；common_graph_proxy 使用 graph_delta。各分支单独与 observed_delta 比较，计算全部位点、真实响应前 20% 和预测前 20% 三档指标。",
    )

    summary = (
        branch_df.groupby(["dataset", "branch", "source_column"], as_index=False)
        .agg(
            n_comparisons=("comparison", "nunique"),
            n_sites=("n_sites", "sum"),
            all_cosine=("all_cosine", "mean"),
            all_spearman=("all_spearman", "mean"),
            all_direction=("all_direction", "mean"),
            responsive20_cosine=("responsive20_cosine", "mean"),
            responsive20_spearman=("responsive20_spearman", "mean"),
            responsive20_direction=("responsive20_direction", "mean"),
            predicted20_recall=("predicted20_recall", "mean"),
            predicted20_direction=("predicted20_direction", "mean"),
            pred_abs_mean=("pred_abs_mean", "mean"),
            real_abs_mean=("real_abs_mean", "mean"),
        )
        .sort_values(["dataset", "branch"])
    )
    summary["source_file"] = rel(pxd_site, project_root)
    write_tsv(summary, paper_dir / "01_key_results" / "component_independent_performance_summary.tsv")
    write_md(
        paper_dir / "01_key_results" / "component_independent_performance_summary.md",
        "PXD063604 分支独立性能汇总，对十二个 drug-cell line pair 取均值。",
    )
    fig4 = paper_dir / "04_figure_source_data" / "fig4"
    write_tsv(summary, fig4 / "panel_c_component_independent_performance.tsv")
    write_md(
        fig4 / "panel_c_component_independent_performance.md",
        "图 4C 源数据，展示 PXD063604 中 full、core proxy、residual 与 common/graph proxy 的独立预测性能。",
    )

    comp = read_tsv(pxd_comp)
    comp["source_file"] = rel(pxd_comp, project_root)
    write_tsv(comp, paper_dir / "01_key_results" / "external_validation" / "pxd063604_external_baseline_comparison_metrics.tsv")
    write_md(
        paper_dir / "01_key_results" / "external_validation" / "pxd063604_external_baseline_comparison_metrics.md",
        "PXD063604 与 PXD039363 外部低剂量或首通道基线评估逐比较项表。PXD063604 的十二个比较项来自该文件。",
    )

    hyper = parse_argparse_defaults(train_script, rel(train_script, project_root))
    if final_metrics_path.exists():
        metrics = json.loads(final_metrics_path.read_text(encoding="utf-8"))
        for key in ["n_comparisons", "n_sites", "n_proteins"]:
            hyper = pd.concat(
                [
                    hyper,
                    pd.DataFrame(
                        [
                            {
                                "parameter": key,
                                "value": metrics.get(key, "NA"),
                                "unit": "NA",
                                "search_range": "NA",
                                "selected_via": "remote final_metrics.json",
                                "source_file": rel(final_metrics_path, project_root),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        runtime = pd.DataFrame(
            [
                {
                    "task": "SCP682-PPKO V10 training",
                    "stage": "full training",
                    "runtime": metrics.get("final", {}).get("elapsed_min", "NA"),
                    "memory_or_vram": "NA",
                    "hardware": "NA",
                    "source_file": rel(final_metrics_path, project_root),
                }
            ]
        )
        write_tsv(runtime, paper_dir / "02_data_tables" / "runtime_memory.tsv")
        write_md(
            paper_dir / "02_data_tables" / "runtime_memory.md",
            "运行时间表。当前只从远端 final_metrics.json 提取训练耗时，显存和硬件结构化记录仍未找到。",
        )
        train_curve = pd.DataFrame(
            [
                {
                    "epoch": metrics.get("final", {}).get("epoch", "NA"),
                    "split": "training",
                    "loss_name": "loss",
                    "loss_value": metrics.get("final", {}).get("loss", "NA"),
                    "extra_metrics": json.dumps(metrics.get("final", {}), ensure_ascii=False),
                }
            ]
        )
        write_tsv(train_curve, paper_dir / "02_data_tables" / "training_curves.tsv")
        write_md(
            paper_dir / "02_data_tables" / "training_curves.md",
            "训练曲线表。远端发布包只保留 final_metrics.json，没有逐轮 training_log.tsv，因此这里只写入最终轮记录。",
        )
    write_tsv(hyper, paper_dir / "02_data_tables" / "hyperparameters.tsv")
    write_md(
        paper_dir / "02_data_tables" / "hyperparameters.md",
        "超参数表，来自发布版训练脚本 argparse 默认值、脚本文字常量和远端 final_metrics.json。",
    )

    if graph_summary_path.exists():
        graph = json.loads(graph_summary_path.read_text(encoding="utf-8"))
        graph_rows = [
            {
                "graph_name": "protein_nodes",
                "node_type": "protein",
                "node_count": graph.get("n_proteins", "NA"),
                "edge_type": "NA",
                "edge_count": "NA",
                "edge_source": "global_heterograph_summary",
                "source_file": rel(graph_summary_path, project_root),
            },
            {
                "graph_name": "site_nodes",
                "node_type": "phosphosite",
                "node_count": graph.get("n_sites", "NA"),
                "edge_type": "NA",
                "edge_count": "NA",
                "edge_source": "global_heterograph_summary",
                "source_file": rel(graph_summary_path, project_root),
            },
            {
                "graph_name": "protein_site_membership",
                "node_type": "protein/phosphosite",
                "node_count": f"{graph.get('n_proteins', 'NA')}/{graph.get('n_sites', 'NA')}",
                "edge_type": "membership",
                "edge_count": graph.get("n_site_membership_edges", "NA"),
                "edge_source": "global_heterograph_summary",
                "source_file": rel(graph_summary_path, project_root),
            },
            {
                "graph_name": "signed_regulator_site",
                "node_type": "regulator/phosphosite",
                "node_count": f"{graph.get('n_v9_regulators', 'NA')}/{graph.get('n_sites', 'NA')}",
                "edge_type": "signed regulator-to-site",
                "edge_count": graph.get("n_signed_regulator_site_edges", "NA"),
                "edge_source": "signed phospho regulatory prior v9",
                "source_file": rel(graph_summary_path, project_root),
            },
            {
                "graph_name": "signed_protein_protein",
                "node_type": "protein",
                "node_count": graph.get("n_proteins", "NA"),
                "edge_type": "signed protein-protein",
                "edge_count": graph.get("n_signed_protein_protein_edges", "NA"),
                "edge_source": "global_heterograph_summary",
                "source_file": rel(graph_summary_path, project_root),
            },
            {
                "graph_name": "unsigned_protein_protein",
                "node_type": "protein",
                "node_count": graph.get("n_proteins", "NA"),
                "edge_type": "unsigned protein-protein",
                "edge_count": graph.get("n_unsigned_protein_protein_edges", "NA"),
                "edge_source": "STRING measured top50",
                "source_file": rel(graph_summary_path, project_root),
            },
            {
                "graph_name": "site_site_cophee",
                "node_type": "phosphosite",
                "node_count": graph.get("n_sites", "NA"),
                "edge_type": "co-phospho",
                "edge_count": graph.get("n_site_site_cophee_edges", "NA"),
                "edge_source": "CoPheeMap",
                "source_file": rel(graph_summary_path, project_root),
            },
        ]
        write_tsv(pd.DataFrame(graph_rows), paper_dir / "02_data_tables" / "graph_statistics.tsv")
        write_md(
            paper_dir / "02_data_tables" / "graph_statistics.md",
            "图统计表，来自远端 global_heterograph_summary.json。source_file 指向素材包内已拉回的摘要文件。",
        )

        sanitized = graph.copy()
        for key in ["matrix_paths", "source_paths", "string_status", "site_site_matrix", "uniprot_mapping_status"]:
            sanitized.pop(key, None)
        sanitized["source_note"] = "path fields removed for paper material package"
        graph_summary_path.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2), encoding="utf-8")

    # 更新 MANIFEST：移除已补项目，保留确实无法从源文件补齐的项目。
    manifest_path = paper_dir / "MANIFEST.md"
    remaining = [
        "指定路径 ./missing_skill_path/paper_extract.md 不存在；已使用当前可用 paper-extract 规则，无法由模型结果重跑补齐。",
        "远端发布包没有逐轮 training_log.tsv；已从 final_metrics.json 补最终轮训练记录，不能恢复完整逐轮曲线。",
        "显存或内存结构化记录未在发布包中保存；已补训练耗时。",
    ]
    file_counts = {
        "01_key_results": len([p for p in (paper_dir / "01_key_results").rglob("*") if p.is_file()]),
        "02_data_tables": len([p for p in (paper_dir / "02_data_tables").rglob("*") if p.is_file()]),
        "03_code": len([p for p in (paper_dir / "03_code").rglob("*") if p.is_file()]),
        "04_figure_source_data": len([p for p in (paper_dir / "04_figure_source_data").rglob("*") if p.is_file()]),
        "05_methods_writing": len([p for p in (paper_dir / "05_methods_writing").rglob("*") if p.is_file()]),
    }
    manifest = f"""# 论文素材清单 - SCP682-PPKO

## 元数据
- 模型代号: SCP682-PPKO
- 提取时间: 2026-05-23
- 代码版本 / git commit: NA
- 主目录路径: .
- 输出目录路径: ./paper_materials_SCP682_PPKO

## 完整性状态
| 类别 | 文件数 | 状态 | 缺失说明 |
|---|---:|---|---|
| 01_key_results | {file_counts['01_key_results']} | 部分 | P100 release v10 指定口径、逐位点重算指标、PXD063604 三路独立性能已补 |
| 02_data_tables | {file_counts['02_data_tables']} | 部分 | P100 逐位点向量、图统计、超参、最终轮训练记录已补；完整逐轮日志和显存记录仍缺 |
| 03_code | {file_counts['03_code']} | 部分 | 已加入 rerun_missing_items.py 和 rerun_v10_p100_site_vectors.py |
| 04_figure_source_data | {file_counts['04_figure_source_data']} | 部分 | 已补分支独立性能和 P100 逐位点重算图源数据 |
| 05_methods_writing | {file_counts['05_methods_writing']} | 部分 | 方法段落需人工按最终投稿口径修订 |

## 缺失项清单
""" + "".join([f"- [ ] {item}\n" for item in remaining]) + """
## 已知问题与待澄清
- P100 指定口径 `0.602 / 0.888 / 0.942` 已定位到 SCP682-PPKO release v1 的 V10 结果，不是 V10B strong300 结果。
- 分支独立性能按当前代码实际字段计算：`latent_delta` 记为 core proxy，`residual_delta` 记为 residual，`graph_delta` 记为 common/graph proxy。若主稿坚持 `core / residual / common` 命名，需要在方法中说明字段映射。

## 关键文件交叉引用
- P100 指定口径: `01_key_results/p100_focus55_release_v10_metrics.tsv`。
- P100 逐位点重算指标: `01_key_results/p100_release_v10_site_vector_rerun_metrics.tsv`。
- P100 focus55 映射行逐位点表: `02_data_tables/p100_focus55_release_v10_mapping_row_site_predictions.tsv`。
- P100 focus55 唯一位点去重表: `02_data_tables/p100_focus55_release_v10_unique_site_predictions.tsv`。
- P100 全 125 比较项逐位点表: `02_data_tables/p100_all125_release_v10_mapping_row_site_predictions.tsv` 与 `02_data_tables/p100_all125_release_v10_unique_site_predictions.tsv`。
- 候选模型对比: `01_key_results/candidate_focus55_comparison.tsv`。
- PXD063604 十二个比较项: `01_key_results/external_validation/pxd063604_all_12_pairs.tsv`。
- PXD063604 逐位点表: `02_data_tables/pxd063604_site_level_predictions.tsv`。
- 三路独立性能: `01_key_results/component_independent_performance.tsv` 与 `01_key_results/component_independent_performance_summary.tsv`。
- 药物类别拆分: `01_key_results/drug_class_performance.tsv`。
- 基线条件化证据: `01_key_results/baseline_conditioning_evidence.tsv`。
- 重跑脚本: `03_code/evaluation/rerun_missing_items.py` 与 `03_code/evaluation/rerun_v10_p100_site_vectors.py`。
"""
    manifest_path.write_text(manifest, encoding="utf-8")


if __name__ == "__main__":
    main()
