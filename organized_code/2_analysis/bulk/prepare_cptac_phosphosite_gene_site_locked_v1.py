#!/usr/bin/env python3
"""Build locked gene-site CPTAC phosphosite matrices with study-wise z-score."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/data/lsy/Infinite_Stream")
IN_DIR = ROOT / "01_data" / "multi_omics" / "processed" / "pancancer_phosphoproteome"
OUT_DIR = ROOT / "01_data" / "multi_omics" / "processed" / "pancancer_phosphoproteome_locked_v1"
RESULT_DIR = ROOT / "02_results" / "model_validation" / "20260426_cptac_phosphosite_locked_v1"


def canonical_mod_string(phosphosite: str) -> str:
    text = str(phosphosite)
    if ":" in text:
        text = text.split(":", 1)[1]
    mods = re.findall(r"[stySTY]\d+", text)
    if not mods:
        return text.lower()
    parsed = [(m[0].upper(), int(m[1:])) for m in mods]
    parsed.sort(key=lambda x: (x[1], x[0]))
    return "_".join(f"{aa}{pos}" for aa, pos in parsed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-coverage", type=float, default=0.20)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "logs").mkdir(parents=True, exist_ok=True)

    y_all = pd.read_parquet(IN_DIR / "phosphosite_logratio_paired_all_targets.parquet")
    x = pd.read_parquet(IN_DIR / "rna_log2_tpm_paired.parquet").loc[y_all.index]
    sample_manifest = pd.read_csv(IN_DIR / "sample_manifest.tsv", sep="\t").set_index("sample_id").loc[y_all.index]
    sample_manifest.index.name = "sample_id"
    target_manifest = pd.read_csv(IN_DIR / "target_manifest.tsv", sep="\t")

    tm = target_manifest.drop_duplicates("feature_id").copy()
    tm["gene"] = tm["gene"].astype(str)
    tm["site_canonical"] = tm["phosphosite"].map(canonical_mod_string)
    tm["gene_site_id"] = tm["gene"] + "|" + tm["site_canonical"]
    fmap = dict(zip(tm["feature_id"], tm["gene_site_id"]))
    known_cols = [c for c in y_all.columns if c in fmap]
    y = y_all[known_cols].copy()
    y.columns = [fmap[c] for c in known_cols]
    y_gene_site = y.T.groupby(level=0).mean().T

    coverage = y_gene_site.notna().mean(axis=0)
    keep = coverage >= args.min_coverage
    y_gene_site_filt = y_gene_site.loc[:, keep].copy()

    y_z = y_gene_site_filt.copy()
    z_stats_rows = []
    for study, idx in sample_manifest.groupby("pdc_study_id").groups.items():
        ids = list(idx)
        sub = y_z.loc[ids]
        mean = sub.mean(axis=0, skipna=True)
        std = sub.std(axis=0, skipna=True)
        std = std.mask((std < 1e-6) | std.isna(), 1.0)
        y_z.loc[ids] = (sub - mean) / std
        z_stats_rows.append(
            {
                "pdc_study_id": study,
                "n_samples": len(ids),
                "median_target_mean_before": float(np.nanmedian(mean.values)),
                "median_target_sd_before": float(np.nanmedian(std.values)),
            }
        )

    redundancy = (
        tm.groupby("gene_site_id", dropna=False)
        .agg(
            gene=("gene", "first"),
            site_canonical=("site_canonical", "first"),
            n_refseq_features=("feature_id", "nunique"),
            n_pdc_studies=("pdc_study_id", "nunique"),
            example_features=("feature_id", lambda s: ";".join(list(map(str, s.dropna().unique()[:5])))),
            example_peptides=("peptide", lambda s: ";".join(list(map(str, s.dropna().unique()[:5])))),
        )
        .reset_index()
    )
    target_locked = redundancy.merge(
        pd.DataFrame(
            {
                "gene_site_id": y_gene_site.columns,
                "sample_coverage_before_filter": coverage.reindex(y_gene_site.columns).values,
                "n_observed_before_filter": y_gene_site.notna().sum(axis=0).reindex(y_gene_site.columns).values,
                "kept_min_coverage": keep.reindex(y_gene_site.columns).values,
            }
        ),
        on="gene_site_id",
        how="right",
    )

    x.to_parquet(OUT_DIR / "rna_log2_tpm_paired.parquet")
    y_gene_site.to_parquet(OUT_DIR / "phosphosite_gene_site_logratio_all_targets.parquet")
    y_gene_site_filt.to_parquet(OUT_DIR / "phosphosite_gene_site_logratio_min20pct_targets.parquet")
    y_z.to_parquet(OUT_DIR / "phosphosite_gene_site_study_zscore_min20pct_targets.parquet")
    sample_manifest.reset_index().to_csv(OUT_DIR / "sample_manifest.tsv", sep="\t", index=False)
    target_locked.to_csv(OUT_DIR / "target_manifest_gene_site_locked_v1.tsv", sep="\t", index=False)
    redundancy.to_csv(RESULT_DIR / "tables" / "phosphosite_gene_site_redundancy.tsv", sep="\t", index=False)
    pd.DataFrame(z_stats_rows).to_csv(RESULT_DIR / "tables" / "studywise_zscore_stats.tsv", sep="\t", index=False)

    summary = {
        "version": "cptac_pancancer_phosphoproteome_locked_v1",
        "input_samples": int(y_all.shape[0]),
        "input_raw_targets": int(y_all.shape[1]),
        "gene_site_targets_all": int(y_gene_site.shape[1]),
        "gene_site_targets_min20pct": int(y_gene_site_filt.shape[1]),
        "rna_genes": int(x.shape[1]),
        "min_coverage": args.min_coverage,
        "collapse_rule": "gene + canonical residue positions parsed from phosphosite suffix; duplicate RefSeq/peptide features averaged per sample",
        "normalization_rule": "within each PDC study, each retained gene-site target is z-scored across samples; missing values remain missing",
    }
    (OUT_DIR / "LOCKED_DATA_CARD.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    hashes = []
    for path in [
        OUT_DIR / "rna_log2_tpm_paired.parquet",
        OUT_DIR / "phosphosite_gene_site_logratio_all_targets.parquet",
        OUT_DIR / "phosphosite_gene_site_logratio_min20pct_targets.parquet",
        OUT_DIR / "phosphosite_gene_site_study_zscore_min20pct_targets.parquet",
        OUT_DIR / "sample_manifest.tsv",
        OUT_DIR / "target_manifest_gene_site_locked_v1.tsv",
        OUT_DIR / "LOCKED_DATA_CARD.json",
    ]:
        hashes.append({"file": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    pd.DataFrame(hashes).to_csv(OUT_DIR / "LOCKED_FILE_HASHES.tsv", sep="\t", index=False)
    pd.DataFrame([summary]).to_csv(OUT_DIR / "locked_matrix_summary.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
