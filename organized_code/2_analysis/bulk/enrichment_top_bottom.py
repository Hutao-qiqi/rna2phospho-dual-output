#!/usr/bin/env python
"""enrichment_top_bottom.py

输入：SCP682 pan-cancer (CPTAC_all) per-site Spearman ρ。
排序后取 top 100 best-predicted + bottom 100 worst-predicted 位点，
按 gene 去重作为查询集；以全部 18,413 评估位点对应的 unique gene 集合
作为背景。对 Hallmark (50) + KEGG Medicus (658) 两套基因集分别做
hypergeometric 富集 + BH-FDR 校正，输出 long-form TSV 供 R 画热图。

输出（追加 fig2 _scripts/）：
  enrichment_top_bottom_hallmark.tsv     — 50 行 × 富集字段
  enrichment_top_bottom_kegg_medicus.tsv — 658 行 × 富集字段
  enrichment_top_bottom_query.tsv        — 200 行 site → gene → 直方位置
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path("E:/data/gongke/TCGA-TCPA/paper_materials_SCP682")
PER_SITE_TSV = ROOT / "01_key_results" / "per_site_spearman_with_deep_learning.tsv"

GMT_HALLMARK = Path("E:/data/gongke/TCGA-TCPA/resources/msigdb/h.all.v2025.1.Hs.symbols.gmt")
GMT_KEGG_MED = Path("E:/data/gongke/rna-seq/c2.cp.kegg_medicus.v2025.1.Hs.symbols.gmt")

OUT_DIR = ROOT / "04_figure_source_data" / "fig2" / "_scripts"
OUT_HALL = OUT_DIR / "enrichment_top_bottom_hallmark.tsv"
OUT_KEGG = OUT_DIR / "enrichment_top_bottom_kegg_medicus.tsv"
OUT_QUERY = OUT_DIR / "enrichment_top_bottom_query.tsv"

TOPK = 1000  # 用户要求 top/bottom 1000 位点（v2 扩展）


def load_per_site() -> pd.DataFrame:
    """读 per-site Spearman，只保留 SCP682 × CPTAC_all 且 ρ 非 NaN。"""
    df = pd.read_csv(PER_SITE_TSV, sep="\t", low_memory=False)
    df = df[(df["method"] == "SCP682") & (df["dataset"] == "CPTAC_all")].copy()
    df["spearman"] = pd.to_numeric(df["spearman"], errors="coerce")
    df = df.dropna(subset=["spearman"])
    df["gene"] = df["target"].str.split("|", n=1).str[0]
    print(f"SCP682 × CPTAC_all 可用位点 n = {len(df):,}")
    print(f"  unique genes (background universe) = {df['gene'].nunique():,}")
    return df


def pick_top_bottom(df: pd.DataFrame, k: int = TOPK) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 ρ 升序，取尾部 k 为 top（best predicted），头部 k 为 bottom（worst predicted）。"""
    d = df.sort_values("spearman", ascending=True).reset_index(drop=True)
    bot = d.head(k).copy()
    top = d.tail(k).copy()
    bot["rank"] = bot.index + 1
    top["rank"] = top.index + 1  # 仍以排序后位置
    return top, bot


def parse_gmt(gmt_path: Path) -> dict[str, set[str]]:
    """读 GMT 文件，返回 dict[gene_set_name -> set[gene_symbol]]。"""
    gmt: dict[str, set[str]] = {}
    with gmt_path.open() as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            name = cols[0]
            genes = {g for g in cols[2:] if g}
            gmt[name] = genes
    print(f"loaded {gmt_path.name}: {len(gmt)} gene sets")
    return gmt


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR。pvals 是一维 array。返回相同长度 q-value array。"""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.arange(1, n + 1)
    q_sorted = pvals[order] * n / ranked
    # cumulative min from the right (BH 标准)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return np.clip(q, 0.0, 1.0)


def enrich(query_genes: set[str], universe: set[str],
           gmt: dict[str, set[str]]) -> pd.DataFrame:
    """
    单边 hypergeometric over-representation:
      N = |universe|
      K = |gene_set ∩ universe|
      n = |query_genes ∩ universe|
      k = |query_genes ∩ gene_set ∩ universe|
      p = P(X >= k) = hypergeom.sf(k-1, N, K, n)
      fold = (k/n) / (K/N)
    """
    q_in_univ = query_genes & universe
    n = len(q_in_univ)
    N = len(universe)
    rows = []
    for term, term_genes in gmt.items():
        K = len(term_genes & universe)
        if K == 0:
            continue
        overlap = q_in_univ & term_genes
        k = len(overlap)
        if k == 0:
            p = 1.0
            fold = 0.0
        else:
            p = float(hypergeom.sf(k - 1, N, K, n))
            fold = (k / n) / (K / N) if n > 0 and K > 0 else 0.0
        rows.append({
            "gene_set": term,
            "K_in_universe": K,
            "n_query": n,
            "k_overlap": k,
            "fold_enrichment": fold,
            "p_value": p,
            "overlap_genes": ";".join(sorted(overlap)),
        })
    out = pd.DataFrame(rows)
    out["q_value"] = bh_adjust(out["p_value"].values)
    out["neg_log10_q"] = -np.log10(out["q_value"].clip(lower=1e-300))
    return out.sort_values("q_value")


def join_top_bottom(top_df: pd.DataFrame, bot_df: pd.DataFrame) -> pd.DataFrame:
    """以 gene_set 为 key，左右合并 top / bottom 富集结果。"""
    top = top_df.rename(columns={
        "k_overlap": "top_k",
        "fold_enrichment": "top_fold",
        "p_value": "top_p",
        "q_value": "top_q",
        "neg_log10_q": "top_neg_log10_q",
        "overlap_genes": "top_genes",
        "n_query": "top_n",
    })[["gene_set", "K_in_universe", "top_n", "top_k", "top_fold",
        "top_p", "top_q", "top_neg_log10_q", "top_genes"]]
    bot = bot_df.rename(columns={
        "k_overlap": "bot_k",
        "fold_enrichment": "bot_fold",
        "p_value": "bot_p",
        "q_value": "bot_q",
        "neg_log10_q": "bot_neg_log10_q",
        "overlap_genes": "bot_genes",
        "n_query": "bot_n",
    })[["gene_set", "bot_n", "bot_k", "bot_fold",
        "bot_p", "bot_q", "bot_neg_log10_q", "bot_genes"]]
    merged = top.merge(bot, on="gene_set", how="outer")
    merged = merged.sort_values(
        merged[["top_q", "bot_q"]].min(axis=1).name
        if False else ["top_q", "bot_q"])
    return merged


def main() -> int:
    df = load_per_site()
    top, bot = pick_top_bottom(df, TOPK)

    # 查询集 = 唯一 gene
    top_genes = set(top["gene"].unique())
    bot_genes = set(bot["gene"].unique())
    universe = set(df["gene"].unique())

    print(f"\nTop {TOPK} 位点 → unique genes (query top) = {len(top_genes)}")
    print(f"Bottom {TOPK} 位点 → unique genes (query bot) = {len(bot_genes)}")
    print(f"Universe (评估过的所有 gene) = {len(universe)}")

    # 保存查询位点清单
    q_top = top[["target", "gene", "spearman"]].assign(direction="top")
    q_bot = bot[["target", "gene", "spearman"]].assign(direction="bottom")
    pd.concat([q_top, q_bot], ignore_index=True).to_csv(
        OUT_QUERY, sep="\t", index=False, float_format="%.6f")
    print(f"\nwrote {OUT_QUERY}")

    # Hallmark
    print("\n=== Hallmark ===")
    h_gmt = parse_gmt(GMT_HALLMARK)
    h_top = enrich(top_genes, universe, h_gmt)
    h_bot = enrich(bot_genes, universe, h_gmt)
    h_merged = join_top_bottom(h_top, h_bot)
    h_merged.to_csv(OUT_HALL, sep="\t", index=False, float_format="%.6g",
                    quoting=csv.QUOTE_MINIMAL)
    print(f"wrote {OUT_HALL}")
    print("top 10 terms by min(top_q, bot_q):")
    h_merged["min_q"] = np.minimum(h_merged["top_q"], h_merged["bot_q"])
    print(h_merged.sort_values("min_q").head(10)[
        ["gene_set", "top_k", "top_q", "bot_k", "bot_q", "top_fold", "bot_fold"]])

    # KEGG Medicus
    print("\n=== KEGG Medicus ===")
    k_gmt = parse_gmt(GMT_KEGG_MED)
    k_top = enrich(top_genes, universe, k_gmt)
    k_bot = enrich(bot_genes, universe, k_gmt)
    k_merged = join_top_bottom(k_top, k_bot)
    k_merged.to_csv(OUT_KEGG, sep="\t", index=False, float_format="%.6g",
                    quoting=csv.QUOTE_MINIMAL)
    print(f"wrote {OUT_KEGG}")
    k_merged["min_q"] = np.minimum(k_merged["top_q"], k_merged["bot_q"])
    print("top 10 terms by min(top_q, bot_q):")
    print(k_merged.sort_values("min_q").head(10)[
        ["gene_set", "top_k", "top_q", "bot_k", "bot_q", "top_fold", "bot_fold"]])

    return 0


if __name__ == "__main__":
    sys.exit(main())
