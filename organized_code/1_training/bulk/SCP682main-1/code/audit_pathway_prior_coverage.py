#!/usr/bin/env python3
"""Audit pathway coverage against the real training RNA and phosphosite axes."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from pathway_sample_graph import build_pathway_prior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna-path", required=True)
    parser.add_argument("--phosphosite-path", required=True)
    parser.add_argument("--hallmark-gmt", required=True)
    parser.add_argument("--canonical-pathway-gmt", required=True)
    parser.add_argument("--rna-context-genes", type=int, default=2048)
    parser.add_argument("--max-pathways", type=int, default=72)
    args = parser.parse_args()

    phosphosite = pd.read_parquet(args.phosphosite_path)
    rna = pd.read_parquet(args.rna_path)
    genes = rna.var(axis=0, skipna=True).sort_values(ascending=False).index[:args.rna_context_genes].tolist()
    prior = build_pathway_prior(
        genes,
        phosphosite.columns.tolist(),
        args.hallmark_gmt,
        args.canonical_pathway_gmt,
        max_pathways=args.max_pathways,
    )
    index = prior["site_pathway_index"]
    mask = prior["site_pathway_mask"]
    specific = ((index != 0) & mask).any(axis=1)
    result = {
        "n_targets": int(phosphosite.shape[1]),
        "n_rna_context_genes": len(genes),
        "n_pathways": len(prior["pathway_names"]),
        "specific_pathway_sites": int(specific.sum()),
        "global_only_sites": int((~specific).sum()),
        "site_pathway_memberships": int(mask.sum()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
