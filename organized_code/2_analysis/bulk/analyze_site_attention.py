#!/usr/bin/env python
"""analyze_site_attention.py

Test whether the LEARNED site-axis attention up-weights functionally coherent
edges, on the non-self-loop subset of the 420,102-edge phosphosite graph.

Functional-coherence labels per edge (i, j):
  - same_protein : gene_i == gene_j (neighbouring sites on the same protein)
  - same_hallmark: gene_i and gene_j share >=1 MSigDB Hallmark gene set
  (kinase-substrate label added separately if a KS table is available)

Output: enrichment of each label in the high-attention tail vs the rest,
with Fisher odds ratio + p, written as a small TSV the figure can read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

BASE = Path("E:/data/gongke/TCGA-TCPA/SCP682_MAIN/attention_export/20260612_site_attention_export_e160/tables")
ATT = BASE / "scp682_e160_site_attention.tsv"
HALLMARK = Path("E:/data/gongke/TCGA-TCPA/resources/msigdb/h.all.v2025.1.Hs.symbols.gmt")
OUT = Path("E:/data/gongke/TCGA-TCPA/SCP682_MAIN/attention_export")

TOP_QUANTILE = 0.95  # "high attention" = top 5% of non-self-loop edges


def load_hallmark_gene_to_sets(gmt: Path) -> dict[str, frozenset]:
    gene_sets: dict[str, set] = {}
    with gmt.open() as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            name = cols[0]
            for g in cols[2:]:
                if g:
                    gene_sets.setdefault(g, set()).add(name)
    return {g: frozenset(s) for g, s in gene_sets.items()}


def main() -> int:
    print("reading attention edges ...")
    df = pd.read_csv(ATT, sep="\t")
    n_all = len(df)
    # drop self-loops
    df = df[df["node_i"] != df["node_j"]].copy()
    n_edges = len(df)
    print(f"edges: {n_all} total, {n_all - n_edges} self-loops, {n_edges} non-self-loop")

    df["gene_i"] = df["name_i"].str.split("|", n=1).str[0]
    df["gene_j"] = df["name_j"].str.split("|", n=1).str[0]

    # ---- functional labels ----
    df["same_protein"] = (df["gene_i"] == df["gene_j"])

    g2s = load_hallmark_gene_to_sets(HALLMARK)
    print(f"hallmark genes mapped: {len(g2s)}")
    empty = frozenset()
    def shares_hallmark(gi, gj):
        si = g2s.get(gi, empty)
        if not si:
            return False
        return len(si & g2s.get(gj, empty)) > 0
    # vectorise-ish: only evaluate cross-protein edges (same-protein trivially share)
    df["same_hallmark"] = [
        shares_hallmark(gi, gj) for gi, gj in zip(df["gene_i"], df["gene_j"])
    ]

    # ---- high vs rest ----
    thr = df["attention"].quantile(TOP_QUANTILE)
    df["high"] = df["attention"] >= thr
    print(f"high-attention threshold (q{TOP_QUANTILE:.2f}) = {thr:.4g}; "
          f"{df['high'].sum()} high edges")

    rows = []
    for label in ["same_protein", "same_hallmark"]:
        a = int(((df["high"]) & (df[label])).sum())     # high & label
        b = int(((df["high"]) & (~df[label])).sum())    # high & not
        c = int(((~df["high"]) & (df[label])).sum())    # rest & label
        d = int(((~df["high"]) & (~df[label])).sum())   # rest & not
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        frac_high = a / (a + b) if (a + b) else float("nan")
        frac_rest = c / (c + d) if (c + d) else float("nan")
        rows.append({
            "label": label,
            "high_n": a + b, "high_frac_with_label": frac_high,
            "rest_n": c + d, "rest_frac_with_label": frac_rest,
            "enrichment": frac_high / frac_rest if frac_rest else float("nan"),
            "odds_ratio": odds, "fisher_p_greater": p,
        })
        print(f"  {label}: high {frac_high:.3f} vs rest {frac_rest:.3f} "
              f"(enrich {frac_high/frac_rest:.2f}x, OR {odds:.2f}, p {p:.2e})")

    enrich = pd.DataFrame(rows)
    enrich.to_csv(OUT / "site_attention_functional_enrichment.tsv", sep="\t",
                  index=False, float_format="%.6g")
    print(f"\nwrote {OUT / 'site_attention_functional_enrichment.tsv'}")

    # ---- attention by label, for a distribution view ----
    summ = []
    for label in ["same_protein", "same_hallmark"]:
        for val, name in [(True, "yes"), (False, "no")]:
            sub = df.loc[df[label] == val, "attention"]
            summ.append({
                "label": label, "value": name, "n": len(sub),
                "median_attention": float(sub.median()),
                "p95_attention": float(sub.quantile(0.95)),
            })
    pd.DataFrame(summ).to_csv(OUT / "site_attention_by_label.tsv", sep="\t",
                              index=False, float_format="%.6g")
    print(f"wrote {OUT / 'site_attention_by_label.tsv'}")

    # ---- persist the labelled non-self-loop edges (for figure / case study) ----
    keep = df[["name_i", "name_j", "gene_i", "gene_j", "attention",
               "same_protein", "same_hallmark", "high"]]
    keep.to_parquet(OUT / "site_attention_labelled_edges.parquet", index=False)
    print(f"wrote labelled edges parquet ({len(keep)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
