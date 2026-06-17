#!/usr/bin/env python
"""panel_f_v5_cluster_enrichment.py

按 panel f v5 聚类策略（top/bottom 各 k-means k=3 on raw ρ）算出每个簇的成员，
然后对每个簇做 Hallmark hypergeometric 富集，输出 top 显著通路。
作为 panel f v7 用：v5 视觉 + 簇 sidebar 标 pathway。

输出：
  panel_f_v5_cluster_assignment.tsv —— target, gene, direction, cluster_id (1..6)
  panel_f_v5_cluster_enrichment.tsv —— cluster_id, top_pathway, short_label, k, q, fold
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.cluster import KMeans

ROOT = Path("E:/data/gongke/TCGA-TCPA/paper_materials_SCP682")
SCRIPTS = ROOT / "04_figure_source_data" / "fig2" / "_scripts"

MATRIX_TSV   = SCRIPTS / "panel_f_heatmap_matrix_cancer_type.tsv"
GMT_HALLMARK = Path("E:/data/gongke/TCGA-TCPA/resources/msigdb/h.all.v2025.1.Hs.symbols.gmt")
PER_SITE_TSV = ROOT / "01_key_results" / "per_site_spearman_with_deep_learning.tsv"

OUT_ASSIGN = SCRIPTS / "panel_f_v5_cluster_assignment.tsv"
OUT_ENRICH = SCRIPTS / "panel_f_v5_cluster_enrichment.tsv"

CT = ["BRCA", "CCRCC", "ccPRCC", "COAD", "GBM",
      "HNSCC", "LSCC", "LUAD", "OV", "PDA", "STAD", "UCEC"]

K_SUB = 3        # 每个 direction 内子聚类数
SEED  = 42

# 通路简称（用于 cluster sidebar 显示）
SHORT_LABEL = {
    "HALLMARK_E2F_TARGETS":                       "E2F",
    "HALLMARK_G2M_CHECKPOINT":                    "G2M",
    "HALLMARK_ESTROGEN_RESPONSE_EARLY":           "ER (early)",
    "HALLMARK_ESTROGEN_RESPONSE_LATE":            "ER (late)",
    "HALLMARK_SPERMATOGENESIS":                   "Sperm.",
    "HALLMARK_MYC_TARGETS_V1":                    "MYC V1",
    "HALLMARK_MYC_TARGETS_V2":                    "MYC V2",
    "HALLMARK_MTORC1_SIGNALING":                  "mTORC1",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING":           "PI3K-AKT-mTOR",
    "HALLMARK_HEME_METABOLISM":                   "Heme",
    "HALLMARK_COAGULATION":                       "Coag.",
    "HALLMARK_COMPLEMENT":                        "Complement",
    "HALLMARK_IL2_STAT5_SIGNALING":               "IL2-STAT5",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE":         "IFNγ",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE":         "IFNα",
    "HALLMARK_INFLAMMATORY_RESPONSE":             "Inflamm.",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB":           "TNFα/NFKB",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": "EMT",
    "HALLMARK_MITOTIC_SPINDLE":                   "Mit. spindle",
    "HALLMARK_APOPTOSIS":                         "Apoptosis",
    "HALLMARK_HYPOXIA":                           "Hypoxia",
    "HALLMARK_GLYCOLYSIS":                        "Glycolysis",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION":         "OXPHOS",
    "HALLMARK_FATTY_ACID_METABOLISM":             "FA metab.",
    "HALLMARK_BILE_ACID_METABOLISM":              "Bile acid",
    "HALLMARK_XENOBIOTIC_METABOLISM":             "Xeno. metab.",
    "HALLMARK_ANGIOGENESIS":                      "Angiog.",
    "HALLMARK_APICAL_JUNCTION":                   "Apical junc.",
    "HALLMARK_APICAL_SURFACE":                    "Apical surf.",
    "HALLMARK_KRAS_SIGNALING_UP":                 "KRAS up",
    "HALLMARK_KRAS_SIGNALING_DN":                 "KRAS dn",
    "HALLMARK_NOTCH_SIGNALING":                   "Notch",
    "HALLMARK_HEDGEHOG_SIGNALING":                "Hedgehog",
    "HALLMARK_WNT_BETA_CATENIN_SIGNALING":        "Wnt/β-cat.",
    "HALLMARK_TGF_BETA_SIGNALING":                "TGFβ",
    "HALLMARK_UV_RESPONSE_UP":                    "UV up",
    "HALLMARK_UV_RESPONSE_DN":                    "UV dn",
    "HALLMARK_DNA_REPAIR":                        "DNA repair",
    "HALLMARK_P53_PATHWAY":                       "p53",
    "HALLMARK_CHOLESTEROL_HOMEOSTASIS":           "Cholest.",
    "HALLMARK_PROTEIN_SECRETION":                 "Prot. secr.",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE":         "UPR",
    "HALLMARK_PEROXISOME":                        "Peroxisome",
    "HALLMARK_ADIPOGENESIS":                      "Adipogen.",
    "HALLMARK_MYOGENESIS":                        "Myogen.",
    "HALLMARK_PANCREAS_BETA_CELLS":               "Panc. β-cell",
    "HALLMARK_ALLOGRAFT_REJECTION":               "Allograft",
    "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY":   "ROS",
    "HALLMARK_ANDROGEN_RESPONSE":                 "Androgen",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING":           "IL6/JAK/STAT3",
}


def parse_gmt(p: Path) -> dict[str, set[str]]:
    gmt: dict[str, set[str]] = {}
    with p.open() as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            gmt[cols[0]] = {g for g in cols[2:] if g}
    return gmt


def impute_rows(m: pd.DataFrame) -> pd.DataFrame:
    """同 R 版本：缺失值用行内均值，全行 NaN 用全矩阵均值兜底。"""
    row_means = m.mean(axis=1)
    out = m.copy()
    for c in m.columns:
        miss = out[c].isna()
        out.loc[miss, c] = row_means[miss]
    out = out.fillna(out.values.mean())
    return out


def kmeans_subcluster(m: pd.DataFrame, k: int, seed: int) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init=20, max_iter=50, random_state=seed)
    return km.fit_predict(m.values)


def reorder_by_pan(cluster_raw: np.ndarray, pan: np.ndarray,
                   base: int) -> np.ndarray:
    """簇按 pan-cancer ρ 中位降序重命名。base 用于位移 (top=0, bottom=3)。"""
    df = pd.DataFrame({"raw": cluster_raw, "pan": pan})
    med = df.groupby("raw")["pan"].median().sort_values(ascending=False)
    new_id_map = {old: i + 1 + base for i, old in enumerate(med.index)}
    return np.array([new_id_map[c] for c in cluster_raw])


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    o = np.argsort(p)
    q_sorted = p[o] * n / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(q_sorted)
    q[o] = q_sorted
    return np.clip(q, 0.0, 1.0)


def enrich(query_genes: set[str], universe: set[str],
           gmt: dict[str, set[str]]) -> pd.DataFrame:
    n = len(query_genes & universe)
    N = len(universe)
    rows = []
    for term, term_genes in gmt.items():
        K = len(term_genes & universe)
        if K == 0:
            continue
        overlap = query_genes & term_genes & universe
        k = len(overlap)
        if k == 0:
            continue
        p = float(hypergeom.sf(k - 1, N, K, n))
        fold = (k / n) / (K / N) if n and K else 0.0
        rows.append({
            "gene_set": term,
            "k": k, "K": K, "n_query": n, "N": N,
            "fold": fold, "p": p,
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["q"] = bh(out["p"].values)
    return out.sort_values("p")


def short(term: str) -> str:
    if term in SHORT_LABEL:
        return SHORT_LABEL[term]
    return (term.replace("HALLMARK_", "")
                .replace("_", " ")
                .lower()
                .capitalize())[:14]


def main() -> int:
    mat = pd.read_csv(MATRIX_TSV, sep="\t")
    m_imp = impute_rows(mat[CT])

    # ---- 复现 v5 聚类 ----
    cluster_top_raw = kmeans_subcluster(
        m_imp.loc[mat["direction"] == "top"], K_SUB, SEED)
    cluster_bot_raw = kmeans_subcluster(
        m_imp.loc[mat["direction"] == "bottom"], K_SUB, SEED)

    pan_top = mat.loc[mat["direction"] == "top", "CPTAC_all"].values
    pan_bot = mat.loc[mat["direction"] == "bottom", "CPTAC_all"].values
    cluster_top = reorder_by_pan(cluster_top_raw, pan_top, base=0)
    cluster_bot = reorder_by_pan(cluster_bot_raw, pan_bot, base=3)

    cluster_id = np.zeros(len(mat), dtype=int)
    cluster_id[mat["direction"] == "top"]    = cluster_top
    cluster_id[mat["direction"] == "bottom"] = cluster_bot

    assign = mat[["row_order", "target", "gene", "direction"]].copy()
    assign["cluster_id"] = cluster_id
    assign.to_csv(OUT_ASSIGN, sep="\t", index=False)
    print(f"wrote {OUT_ASSIGN}")
    print("\ncluster sizes:")
    print(pd.Series(cluster_id).value_counts().sort_index())

    # ---- Universe: SCP682 评估过的所有 unique gene ----
    raw = pd.read_csv(PER_SITE_TSV, sep="\t", low_memory=False)
    raw = raw[(raw["method"] == "SCP682") & (raw["dataset"] == "CPTAC_all")]
    raw = raw[pd.to_numeric(raw["spearman"], errors="coerce").notna()]
    universe = set(raw["target"].str.split("|", n=1).str[0].unique())
    print(f"\nuniverse size: {len(universe)}")

    # ---- 每个 cluster 做 Hallmark 富集 ----
    gmt = parse_gmt(GMT_HALLMARK)
    out_rows = []
    for cid in range(1, 7):
        members = assign[assign["cluster_id"] == cid]
        genes = set(members["gene"].unique())
        n_genes = len(genes & universe)
        enrich_df = enrich(genes, universe, gmt)
        if enrich_df.empty:
            print(f"cluster {cid}: n_query={n_genes} → no enrichment")
            continue
        # 取 top 3 by p
        top3 = enrich_df.head(3).copy()
        top3["cluster_id"] = cid
        top3["n_query_genes"] = n_genes
        top3["short_label"] = top3["gene_set"].map(short)
        # rank 1, 2, 3
        top3["rank"] = range(1, len(top3) + 1)
        out_rows.append(top3)
        print(f"\ncluster {cid} (n_genes={n_genes}) top hits:")
        print(top3[["rank", "gene_set", "short_label", "k", "K",
                    "fold", "p", "q"]].to_string(index=False))

    enrichment = pd.concat(out_rows, ignore_index=True)
    enrichment = enrichment[["cluster_id", "rank", "gene_set", "short_label",
                              "k", "K", "n_query_genes",
                              "fold", "p", "q"]]
    enrichment.to_csv(OUT_ENRICH, sep="\t", index=False, float_format="%.6g")
    print(f"\nwrote {OUT_ENRICH}")

    # 总结：cluster_id → top_pathway_short (rank 1)
    print("\n=== sidebar label assignment ===")
    print(enrichment[enrichment["rank"] == 1][
        ["cluster_id", "short_label", "k", "q"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
