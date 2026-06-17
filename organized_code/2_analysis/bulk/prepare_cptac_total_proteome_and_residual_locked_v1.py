#!/usr/bin/env python3
"""Pair CPTAC total proteome and build phospho/total residual labels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/data/lsy/Infinite_Stream")
LOCKED_DIR = ROOT / "01_data" / "multi_omics" / "processed" / "pancancer_phosphoproteome_locked_v1"
PROTEOME_RAW = ROOT / "01_data" / "multi_omics" / "raw" / "cptac_pancancer_proteome_reports"
OUT_DIR = ROOT / "01_data" / "multi_omics" / "processed" / "pancancer_multi_task_locked_v1"
RESULT_DIR = ROOT / "02_results" / "model_validation" / "20260426_cptac_total_proteome_residual_locked_v1"


PHOSPHO_TO_PROTEOME = {
    "PDC000615": "PDC000614",
    "PDC000271": "PDC000270",
    "PDC000441": "PDC000439",
    "PDC000490": "PDC000489",
    "PDC000128": "PDC000127",
    "PDC000222": "PDC000221",
    "PDC000126": "PDC000125",
    "PDC000149": "PDC000153",
    "PDC000232": "PDC000234",
    "PDC000465": "PDC000464",
}


def sample_id_from_col(col: str, aliquots: set[str]) -> str | None:
    raw = str(col).strip()
    if "Unshared" in raw:
        return None
    if raw.endswith(" Log Ratio"):
        sample = raw.replace(" Log Ratio", "").strip()
    elif raw.startswith("Log ") and "/" in raw:
        sample = raw[4:].split("/", 1)[0].strip()
    else:
        return None
    if sample.startswith("Withdrawn:"):
        return None
    if sample in aliquots:
        return sample
    sample2 = re.sub(r"\.\d+$", "", sample)
    if sample2 in aliquots:
        return sample2
    return None


def find_proteome_file(proteome_study: str) -> Path | None:
    files = sorted((PROTEOME_RAW / proteome_study).glob("*Proteome.tmt*.tsv"))
    files = [p for p in files if ".peptide." not in p.name.lower() and ".peptides." not in p.name.lower()]
    return files[0] if files else None


def read_proteome_matrix(path: Path, aliquots: set[str]) -> pd.DataFrame:
    header = pd.read_csv(path, sep="\t", nrows=0)
    alias = {}
    for c in header.columns:
        sid = sample_id_from_col(c, aliquots)
        if sid is not None:
            alias[c] = sid
    cols = ["Gene"] + list(alias)
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in set(cols))
    df = df.rename(columns=alias)
    df["Gene"] = df["Gene"].astype(str)
    bad = {"Mean", "Median", "StdDev", "NumSpectra", "QValue", "Unshared Peptides", "Distinct Peptides"}
    df = df[~df["Gene"].isin(bad)].copy()
    value_cols = [c for c in df.columns if c != "Gene"]
    mat = df[["Gene"] + value_cols].groupby("Gene", as_index=True).mean(numeric_only=True)
    return mat.T.groupby(level=0).mean()


def study_zscore(df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for _, idx in manifest.groupby("pdc_study_id").groups.items():
        ids = list(idx)
        sub = out.loc[ids]
        mean = sub.mean(axis=0, skipna=True)
        std = sub.std(axis=0, skipna=True)
        std = std.mask((std < 1e-6) | std.isna(), 1.0)
        out.loc[ids] = (sub - mean) / std
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-protein-coverage", type=float, default=0.20)
    args = parser.parse_args()
    for d in [OUT_DIR, RESULT_DIR / "tables", RESULT_DIR / "logs"]:
        d.mkdir(parents=True, exist_ok=True)

    x = pd.read_parquet(LOCKED_DIR / "rna_log2_tpm_paired.parquet")
    y_phospho = pd.read_parquet(LOCKED_DIR / "phosphosite_gene_site_study_zscore_min20pct_targets.parquet").loc[x.index]
    manifest = pd.read_csv(LOCKED_DIR / "sample_manifest.tsv", sep="\t").set_index("sample_id").loc[x.index]
    manifest.index.name = "sample_id"
    target_manifest = pd.read_csv(LOCKED_DIR / "target_manifest_gene_site_locked_v1.tsv")

    protein_parts = []
    logs = []
    for phospho_study, proteome_study in PHOSPHO_TO_PROTEOME.items():
        ids = manifest.index[manifest["pdc_study_id"].eq(phospho_study)].tolist()
        sub_manifest = manifest.loc[ids]
        aliquots = set(sub_manifest["pdc_aliquot_submitter_id"].astype(str))
        path = find_proteome_file(proteome_study)
        if path is None:
            logs.append({"pdc_study_id": phospho_study, "proteome_study_id": proteome_study, "status": "missing_file"})
            continue
        mat = read_proteome_matrix(path, aliquots)
        aliquot_to_sample = dict(zip(sub_manifest["pdc_aliquot_submitter_id"].astype(str), sub_manifest.index.astype(str)))
        mat = mat.loc[mat.index.intersection(aliquot_to_sample)].copy()
        mat.index = [aliquot_to_sample[a] for a in mat.index]
        mat = mat.groupby(level=0).mean()
        protein_parts.append(mat)
        logs.append(
            {
                "pdc_study_id": phospho_study,
                "proteome_study_id": proteome_study,
                "file": str(path),
                "status": "paired",
                "n_samples": int(mat.shape[0]),
                "n_protein_genes": int(mat.shape[1]),
            }
        )
    protein = pd.concat(protein_parts, axis=0, join="outer").reindex(x.index)
    protein_cov = protein.notna().mean(axis=0)
    protein_keep = protein_cov >= args.min_protein_coverage
    protein_filt = protein.loc[:, protein_keep].copy()
    protein_z = study_zscore(protein_filt, manifest)

    target_gene = y_phospho.columns.to_series().str.split("|", regex=False).str[0]
    matched = target_gene[target_gene.isin(protein_z.columns)]
    y_resid = y_phospho.loc[:, matched.index].copy()
    for target, gene in matched.items():
        y_resid[target] = y_resid[target] - protein_z[gene]

    target_resid_manifest = pd.DataFrame(
        {
            "gene_site_id": matched.index,
            "total_protein_gene": matched.values,
            "phospho_sample_coverage": y_phospho.loc[:, matched.index].notna().mean(axis=0).values,
            "total_protein_sample_coverage": protein_z.loc[:, matched.values].notna().mean(axis=0).values,
        }
    )
    x.to_parquet(OUT_DIR / "rna_log2_tpm_paired.parquet")
    protein.to_parquet(OUT_DIR / "total_protein_gene_logratio_all.parquet")
    protein_z.to_parquet(OUT_DIR / "total_protein_gene_study_zscore_min20pct.parquet")
    y_phospho.to_parquet(OUT_DIR / "phosphosite_gene_site_study_zscore_min20pct_targets.parquet")
    y_resid.to_parquet(OUT_DIR / "phosphosite_gene_site_total_residual_targets.parquet")
    manifest.reset_index().to_csv(OUT_DIR / "sample_manifest.tsv", sep="\t", index=False)
    target_resid_manifest.to_csv(OUT_DIR / "residual_target_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(logs).to_csv(RESULT_DIR / "tables" / "total_proteome_pairing_log.tsv", sep="\t", index=False)

    summary = {
        "version": "cptac_pancancer_multi_task_locked_v1",
        "n_samples": int(x.shape[0]),
        "n_rna_genes": int(x.shape[1]),
        "n_total_protein_genes_all": int(protein.shape[1]),
        "n_total_protein_genes_min20pct": int(protein_z.shape[1]),
        "n_phosphosite_targets": int(y_phospho.shape[1]),
        "n_residual_targets_with_total_protein": int(y_resid.shape[1]),
        "min_protein_coverage": args.min_protein_coverage,
        "residual_rule": "study-zscored phosphosite gene-site minus study-zscored total protein abundance for the same gene",
    }
    (OUT_DIR / "MULTITASK_DATA_CARD.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(OUT_DIR / "multi_task_matrix_summary.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
