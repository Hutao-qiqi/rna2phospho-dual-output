#!/usr/bin/env python
"""panel_f_heatmap_data.py

为 Fig 2 panel f（200 位点 × 6 数据集热图）准备数据。

输入：
  01_key_results/per_site_spearman_with_deep_learning.tsv  (SCP682 × 6 datasets)
  04_figure_source_data/fig2/_scripts/enrichment_top_bottom_query.tsv  (200 sites)
  04_figure_source_data/fig2/_scripts/enrichment_top_bottom_hallmark.tsv (cluster terms)

输出：
  panel_f_heatmap_matrix.tsv     —— 200 rows × (target, gene, direction, row_order, 6 datasets ρ)
  panel_f_marker_labels.tsv      —— ~30 行 (15 top + 15 bottom) marker label
  panel_f_cluster_annotations.tsv —— 8 行 cluster × pathway × q × top_genes 用于右侧框

布局：
  Top 100 行（直接按 pan-cancer ρ 降序）→ Bottom 100 行（按 pan-cancer ρ 降序，即最差在底）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("E:/data/gongke/TCGA-TCPA/paper_materials_SCP682")
PER_SITE = ROOT / "01_key_results" / "per_site_spearman_with_deep_learning.tsv"

SCRIPTS = ROOT / "04_figure_source_data" / "fig2" / "_scripts"
QUERY_TSV = SCRIPTS / "enrichment_top_bottom_query.tsv"
HALLMARK_TSV = SCRIPTS / "enrichment_top_bottom_hallmark.tsv"

OUT_MATRIX = SCRIPTS / "panel_f_heatmap_matrix.tsv"
OUT_MARKERS = SCRIPTS / "panel_f_marker_labels.tsv"
OUT_ANNOT = SCRIPTS / "panel_f_cluster_annotations.tsv"

DATASETS_ORDER = [
    "CPTAC_all",
    "CPTAC_kidney",
    "CPTAC_pancreas_HN",
    "CPTAC_gynecologic",
    "CPTAC_gi_hepato",
    "CPTAC_lung",
]
DATASET_LABELS = {
    "CPTAC_all":          "Pan-cancer\n(n = 1,431)",
    "CPTAC_kidney":       "Kidney\n(n = 132)",
    "CPTAC_pancreas_HN":  "Pancreas / H&N\n(n = 172)",
    "CPTAC_gynecologic":  "Gynaecological\n(n = 167)",
    "CPTAC_gi_hepato":    "GI / hepatobiliary\n(n = 79)",
    "CPTAC_lung":         "Lung\n(n = 284)",
}

N_MARKERS_PER_CLUSTER = 20  # v2：1000 sites per cluster 后增加 marker 数


def build_matrix() -> pd.DataFrame:
    """加载 200 位点 ρ × 6 数据集宽矩阵。"""
    q = pd.read_csv(QUERY_TSV, sep="\t")
    target_set = set(q["target"])

    df = pd.read_csv(PER_SITE, sep="\t", low_memory=False)
    df = df[df["method"] == "SCP682"].copy()
    df["spearman"] = pd.to_numeric(df["spearman"], errors="coerce")
    df = df[df["target"].isin(target_set)]
    print(f"hit rows: {len(df):,}; unique targets: {df['target'].nunique()}")

    # 宽化
    wide = df.pivot(index="target", columns="dataset", values="spearman")
    wide = wide.reindex(columns=DATASETS_ORDER)

    # 加上 direction / gene
    meta = q[["target", "gene", "direction", "spearman"]].rename(
        columns={"spearman": "rho_pancancer_q"})
    wide = wide.reset_index().merge(meta, on="target", how="left")

    # 行排序：Top 100 由 ρ 降序（最高在最上）；Bottom 100 也按 ρ 降序（最低在最下）
    wide["sort_key"] = wide["CPTAC_all"]
    top = wide[wide["direction"] == "top"].sort_values("sort_key", ascending=False).reset_index(drop=True)
    bot = wide[wide["direction"] == "bottom"].sort_values("sort_key", ascending=False).reset_index(drop=True)
    ordered = pd.concat([top, bot], ignore_index=True)
    ordered["row_order"] = np.arange(1, len(ordered) + 1)

    # 输出列序
    out_cols = ["row_order", "target", "gene", "direction"] + DATASETS_ORDER
    ordered = ordered[out_cols]
    ordered.to_csv(OUT_MATRIX, sep="\t", index=False, float_format="%.6f")
    print(f"wrote {OUT_MATRIX}  ({len(ordered)} rows)")
    return ordered


def select_markers(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    选 top/bottom 各 ~15 个 marker gene 在左侧显示。
    选择规则：
      Top: 优先来自 E2F_TARGETS / G2M_CHECKPOINT / ESTROGEN_RESPONSE 重叠基因，
            且 ρ_pancancer 高，每 gene 只选 1 个位点（每 gene 取 ρ 最高的位点）
      Bottom: 来自 HEME_METABOLISM / COAGULATION / COMPLEMENT / IL2_STAT5 重叠基因
    """
    h = pd.read_csv(HALLMARK_TSV, sep="\t")

    def gather_overlap(direction: str, hallmark_terms: list[str]) -> list[str]:
        col = "top_genes" if direction == "top" else "bot_genes"
        genes = set()
        for term in hallmark_terms:
            row = h[h["gene_set"] == term]
            if row.empty:
                continue
            val = row.iloc[0][col]
            if isinstance(val, str) and val:
                genes.update(val.split(";"))
        return sorted(genes)

    # v2：1000 sites per cluster 下显著的通路有所变化
    top_priority_genes = gather_overlap("top", [
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_G2M_CHECKPOINT",
        "HALLMARK_ESTROGEN_RESPONSE_EARLY",
        "HALLMARK_SPERMATOGENESIS",
    ])
    bot_priority_genes = gather_overlap("bottom", [
        "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
        "HALLMARK_HEME_METABOLISM",
        "HALLMARK_COAGULATION",
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    ])
    print(f"top priority genes: {len(top_priority_genes)}: {top_priority_genes}")
    print(f"bottom priority genes: {len(bot_priority_genes)}: {bot_priority_genes}")

    def pick(direction: str, priority: list[str], n: int) -> pd.DataFrame:
        sub = matrix[matrix["direction"] == direction].copy()
        # 每 gene 只保留 ρ 最极端的一行
        if direction == "top":
            sub = sub.sort_values("CPTAC_all", ascending=False)
        else:
            sub = sub.sort_values("CPTAC_all", ascending=True)
        sub = sub.drop_duplicates(subset=["gene"], keep="first")
        # 先取在 priority list 里的，按 ρ 极端度排序
        in_pri = sub[sub["gene"].isin(priority)].copy()
        # 不够则从非 priority 里补
        if len(in_pri) < n:
            others = sub[~sub["gene"].isin(priority)].head(n - len(in_pri))
            picks = pd.concat([in_pri.head(n), others], ignore_index=True)
        else:
            picks = in_pri.head(n)
        picks["marker_direction"] = direction
        return picks

    top_picks = pick("top", top_priority_genes, N_MARKERS_PER_CLUSTER)
    bot_picks = pick("bottom", bot_priority_genes, N_MARKERS_PER_CLUSTER)
    markers = pd.concat([top_picks, bot_picks], ignore_index=True)
    markers = markers[["row_order", "target", "gene", "direction", "CPTAC_all", "marker_direction"]]
    markers = markers.rename(columns={"CPTAC_all": "rho_pancancer"})
    markers.to_csv(OUT_MARKERS, sep="\t", index=False, float_format="%.6f")
    print(f"wrote {OUT_MARKERS}  ({len(markers)} rows)")
    return markers


def cluster_annotations() -> pd.DataFrame:
    """右侧每个 cluster 显示的 Hallmark 通路。"""
    h = pd.read_csv(HALLMARK_TSV, sep="\t")

    # 手动指定漂亮 label（避免 .title() 把 PI3K 变 Pi3K）
    pretty_labels = {
        "HALLMARK_E2F_TARGETS":                       "E2F targets",
        "HALLMARK_G2M_CHECKPOINT":                    "G2M checkpoint",
        "HALLMARK_ESTROGEN_RESPONSE_EARLY":           "Estrogen response (early)",
        "HALLMARK_ESTROGEN_RESPONSE_LATE":            "Estrogen response (late)",
        "HALLMARK_SPERMATOGENESIS":                   "Spermatogenesis",
        "HALLMARK_PI3K_AKT_MTOR_SIGNALING":           "PI3K-AKT-mTOR signaling",
        "HALLMARK_HEME_METABOLISM":                   "Heme metabolism",
        "HALLMARK_COAGULATION":                       "Coagulation",
        "HALLMARK_COMPLEMENT":                        "Complement",
        "HALLMARK_IL2_STAT5_SIGNALING":               "IL2-STAT5 signaling",
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": "Epithelial-mesenchymal transition",
        "HALLMARK_MITOTIC_SPINDLE":                   "Mitotic spindle",
    }

    def pack(direction: str, terms: list[str]) -> pd.DataFrame:
        col_q = "top_q" if direction == "top" else "bot_q"
        col_k = "top_k" if direction == "top" else "bot_k"
        col_f = "top_fold" if direction == "top" else "bot_fold"
        rows = []
        for term in terms:
            row = h[h["gene_set"] == term]
            if row.empty:
                continue
            rows.append({
                "cluster": direction,
                "rank": len(rows) + 1,
                "gene_set": term,
                "label": pretty_labels.get(
                    term, term.replace("HALLMARK_", "").replace("_", " ").title()),
                "k": int(row.iloc[0][col_k]),
                "q": float(row.iloc[0][col_q]),
                "fold": float(row.iloc[0][col_f]),
            })
        return pd.DataFrame(rows)

    top_annot = pack("top", [
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_G2M_CHECKPOINT",
        "HALLMARK_ESTROGEN_RESPONSE_EARLY",
        "HALLMARK_SPERMATOGENESIS",
    ])
    bot_annot = pack("bottom", [
        "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
        "HALLMARK_HEME_METABOLISM",
        "HALLMARK_COAGULATION",
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    ])
    annot = pd.concat([top_annot, bot_annot], ignore_index=True)
    annot.to_csv(OUT_ANNOT, sep="\t", index=False, float_format="%.4g")
    print(f"wrote {OUT_ANNOT}  ({len(annot)} rows)")
    return annot


def main() -> int:
    matrix = build_matrix()
    select_markers(matrix)
    cluster_annotations()
    # NaN 统计
    print("\nNaN per dataset (out of 200 sites):")
    for ds in DATASETS_ORDER:
        n_nan = matrix[ds].isna().sum()
        print(f"  {ds:24s} {n_nan:3d} ({100*n_nan/len(matrix):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
