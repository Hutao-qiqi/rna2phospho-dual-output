#!/usr/bin/env python3
"""Site-level cancer-specific clustering followed by KEGG enrichment."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests


ROOT = Path("E:/data/gongke/TCGA-TCPA")
TABLES = ROOT / "paper_final" / "fig5" / "source_data" / "tables"
MSIGDB_KEGG = ROOT / "resources" / "msigdb" / "c2.cp.v2025.1.Hs.symbols.gmt"

Q_CUTOFF = 0.05
MIN_SIGNAL_CANCERS = 1
N_SITE_CLUSTERS = 14
TOP_KEGG_PER_CLUSTER = 5
MIN_CLUSTER_GENES = 12


def bh_q(pvals: pd.Series | np.ndarray) -> np.ndarray:
    p = pd.to_numeric(pd.Series(pvals), errors="coerce").to_numpy(dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.sum() > 0:
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def clean_kegg_name(name: str) -> str:
    text = re.sub(r"^KEGG_", "", str(name))
    text = text.replace("_", " ").title()
    repl = {
        "Dna": "DNA",
        "Rna": "RNA",
        "P53": "p53",
        "Mtor": "mTOR",
        "Tgf": "TGF",
        "Jak": "JAK",
        "Stat": "STAT",
        "Mapk": "MAPK",
        "Erbb": "ERBB",
        "Vegf": "VEGF",
        "Fc Gamma": "Fc gamma",
        "Nf Kappa B": "NF-kappa B",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text


def read_kegg(path: Path) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if (
                len(fields) < 4
                or not fields[0].startswith("KEGG_")
                or fields[0].startswith("KEGG_MEDICUS_")
            ):
                continue
            genes = {g.upper() for g in fields[2:] if g}
            if 10 <= len(genes) <= 500:
                sets[fields[0]] = genes
    return sets


def top_join(values, n=4) -> str:
    vals = [str(v) for v in values if pd.notna(v) and str(v)]
    return "; ".join(vals[:n])


def main() -> None:
    z = pd.read_csv(TABLES / "panel_i_full_site_signed_cox_z_matrix.tsv", sep="\t")
    z["gene"] = z["gene_site"].str.split("|", regex=False).str[0].str.upper()
    cancers = [c for c in z.columns if c not in {"gene_site", "gene"}]
    z_mat = z.set_index("gene_site")[cancers].apply(pd.to_numeric, errors="coerce")

    lrt = pd.read_csv(TABLES / "panel_i_full_site_over_parent_lrt.tsv.gz", sep="\t")
    sig = lrt.loc[
        (pd.to_numeric(lrt["add_site_to_parent_mrna_lrt_q_bh"], errors="coerce") < Q_CUTOFF)
        & (pd.to_numeric(lrt["site_p_adjusted_for_parent_mrna_q_bh"], errors="coerce") < Q_CUTOFF),
        ["cancer", "gene_site", "gene", "add_site_to_parent_mrna_lrt_q_bh", "site_p_adjusted_for_parent_mrna_q_bh"],
    ].copy()
    sig["gene"] = sig["gene"].astype(str).str.upper()
    sig["gene_site"] = sig["gene_site"].astype(str)

    mask = pd.DataFrame(False, index=z_mat.index, columns=cancers)
    sig = sig.loc[sig["gene_site"].isin(mask.index) & sig["cancer"].isin(cancers)]
    for cancer, sub in sig.groupby("cancer"):
        mask.loc[sub["gene_site"].unique(), cancer] = True

    signal = z_mat.abs().where(mask, 0.0).clip(upper=6.0)
    signal["n_signal_cancers"] = (signal[cancers] > 0).sum(axis=1)
    keep_sites = signal.index[signal["n_signal_cancers"] >= MIN_SIGNAL_CANCERS].tolist()
    signal = signal.loc[keep_sites, cancers]
    z_keep = z.loc[z["gene_site"].isin(keep_sites), ["gene_site", "gene"]].drop_duplicates()

    row_sum = signal.sum(axis=1).replace(0, np.nan)
    profile = signal.div(row_sum, axis=0).fillna(0.0)
    dist = pdist(profile.to_numpy(dtype=float), metric="cosine")
    row_linkage = linkage(dist, method="average")
    cluster_raw = fcluster(row_linkage, N_SITE_CLUSTERS, criterion="maxclust")
    site_cluster = pd.DataFrame(
        {
            "gene_site": profile.index,
            "raw_cluster": cluster_raw,
        }
    ).merge(z_keep, on="gene_site", how="left")

    cluster_signal = []
    for cl, sub in site_cluster.groupby("raw_cluster"):
        mat = signal.loc[sub["gene_site"], cancers]
        frac = (mat > 0).mean(axis=0)
        mean_abs = mat.replace(0, np.nan).mean(axis=0).fillna(0.0)
        score = frac * mean_abs
        dom = score.sort_values(ascending=False).index[:4].tolist()
        cluster_signal.append(
            {
                "raw_cluster": cl,
                "n_sites": int(sub["gene_site"].nunique()),
                "n_genes": int(sub["gene"].nunique()),
                "dominant_cancers": top_join(dom, 4),
                **{c: float(score[c]) for c in cancers},
            }
        )
    cluster_signal = pd.DataFrame(cluster_signal)

    cluster_mat = cluster_signal.set_index("raw_cluster")[cancers]
    cluster_order = leaves_list(linkage(pdist(cluster_mat.to_numpy(dtype=float), metric="euclidean"), method="average"))
    ordered_raw = cluster_mat.index.to_numpy()[cluster_order].tolist()
    cluster_id_map = {raw: f"S{i + 1:02d}" for i, raw in enumerate(ordered_raw)}
    site_cluster["site_cluster"] = site_cluster["raw_cluster"].map(cluster_id_map)
    cluster_signal["site_cluster"] = cluster_signal["raw_cluster"].map(cluster_id_map)
    cluster_signal = cluster_signal.sort_values("site_cluster")

    kegg = read_kegg(MSIGDB_KEGG)
    background_genes = set(site_cluster["gene"].dropna().astype(str).str.upper())
    background_genes = {g for g in background_genes if g}
    M = len(background_genes)
    enrich_rows = []
    for raw, sub in site_cluster.groupby("raw_cluster"):
        cid = cluster_id_map[raw]
        genes = set(sub["gene"].dropna().astype(str).str.upper()) & background_genes
        N = len(genes)
        if N < MIN_CLUSTER_GENES:
            continue
        for pathway, geneset in kegg.items():
            gs = geneset & background_genes
            K = len(gs)
            x = len(genes & gs)
            if K < 5 or x < 3:
                continue
            p = float(hypergeom.sf(x - 1, M, K, N))
            enrich_rows.append(
                {
                    "site_cluster": cid,
                    "raw_cluster": raw,
                    "kegg_pathway": pathway,
                    "kegg_label": clean_kegg_name(pathway),
                    "overlap_genes": x,
                    "cluster_genes": N,
                    "pathway_genes_in_background": K,
                    "background_genes": M,
                    "p": p,
                    "gene_ratio": x / max(N, 1),
                    "overlap_gene_symbols": ";".join(sorted(genes & gs)),
                }
            )
    enrich = pd.DataFrame(enrich_rows)
    enrich["q"] = bh_q(enrich["p"])
    enrich["neglog10_q"] = -np.log10(enrich["q"].clip(lower=1e-50))
    enrich["neglog10_q"] = enrich["neglog10_q"].replace([np.inf, -np.inf], 50.0).fillna(0.0).clip(upper=50.0)

    top = (
        enrich.sort_values(["site_cluster", "q", "p", "overlap_genes"], ascending=[True, True, True, False])
        .groupby("site_cluster", as_index=False)
        .head(TOP_KEGG_PER_CLUSTER)
    )
    selected_pathways = (
        top.sort_values(["q", "overlap_genes"], ascending=[True, False])["kegg_pathway"]
        .drop_duplicates()
        .tolist()
    )
    enrich_mat = (
        enrich.loc[enrich["kegg_pathway"].isin(selected_pathways)]
        .pivot_table(index="kegg_pathway", columns="site_cluster", values="neglog10_q", aggfunc="max", fill_value=0.0)
        .reindex(columns=[cluster_id_map[raw] for raw in ordered_raw])
    )
    enrich_mat = enrich_mat.replace([np.inf, -np.inf], 50.0).fillna(0.0)
    if enrich_mat.shape[0] > 1:
        order = leaves_list(linkage(pdist(enrich_mat.to_numpy(dtype=float), metric="euclidean"), method="average"))
        enrich_mat = enrich_mat.iloc[order, :]
    pathway_labels = pd.DataFrame(
        {
            "kegg_pathway": enrich_mat.index,
            "kegg_label": [clean_kegg_name(x) for x in enrich_mat.index],
        }
    )

    cluster_matrix = cluster_signal.set_index("site_cluster")[cancers]
    cluster_matrix = cluster_matrix.reindex([cluster_id_map[raw] for raw in ordered_raw])
    cluster_meta = cluster_signal.set_index("site_cluster")[
        ["n_sites", "n_genes", "dominant_cancers"]
    ].reindex(cluster_matrix.index).reset_index()

    cluster_matrix.reset_index().to_csv(TABLES / "panel_i_site_specificity_cluster_cancer_matrix.tsv", sep="\t", index=False)
    cluster_meta.to_csv(TABLES / "panel_i_site_specificity_cluster_meta.tsv", sep="\t", index=False)
    site_cluster[["gene_site", "gene", "site_cluster", "raw_cluster"]].to_csv(
        TABLES / "panel_i_site_specificity_site_clusters.tsv", sep="\t", index=False
    )
    enrich.to_csv(TABLES / "panel_i_site_specificity_kegg_enrichment.tsv", sep="\t", index=False)
    enrich_mat.reset_index().to_csv(TABLES / "panel_i_site_specificity_kegg_matrix.tsv", sep="\t", index=False)
    pathway_labels.to_csv(TABLES / "panel_i_site_specificity_kegg_pathway_labels.tsv", sep="\t", index=False)
    summary = {
        "meaningful_site_cancer_rows": int(sig.shape[0]),
        "meaningful_unique_sites": int(len(keep_sites)),
        "meaningful_unique_genes": int(len(background_genes)),
        "site_clusters": int(cluster_matrix.shape[0]),
        "selected_kegg_pathways": int(enrich_mat.shape[0]),
        "q_cutoff": Q_CUTOFF,
        "signal_definition": "abs(signed Cox z) for site-cancer rows passing parent-mRNA LRT q and adjusted site q; zero otherwise",
        "direction_used": False,
    }
    (TABLES / "panel_i_site_specificity_kegg_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
