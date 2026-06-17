#!/usr/bin/env python3
"""Prepare paired CPTAC RNA and phosphoproteome matrices for pan-cancer modeling."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/data/lsy/Infinite_Stream")
META_DIR = ROOT / "01_data" / "multi_omics" / "metadata"
GDC_DIR = ROOT / "01_data" / "multi_omics" / "raw" / "gdc_cptac_open_star_counts"
PDC_REPORT_DIR = (
    ROOT
    / "01_data"
    / "multi_omics"
    / "raw"
    / "cptac_pancancer_phosphoproteome_reports"
)
OUT_DIR = (
    ROOT
    / "01_data"
    / "multi_omics"
    / "processed"
    / "pancancer_phosphoproteome"
)


STUDY_TO_CANCER = {
    "PDC000615": "STAD",
    "PDC000271": "PDA",
    "PDC000441": "UCEC_CONFIRM",
    "PDC000490": "LUAD_CONFIRM",
    "PDC000128": "CCRCC",
    "PDC000222": "HNSCC",
    "PDC000126": "UCEC",
    "PDC000149": "LUAD",
    "PDC000232": "LSCC",
    "PDC000465": "NON_CCRCC",
    "PDC000205": "GBM",
    "PDC000448": "GBM_CONFIRM",
}

DEFAULT_STUDIES = [
    "PDC000615",
    "PDC000271",
    "PDC000441",
    "PDC000490",
    "PDC000128",
    "PDC000222",
    "PDC000126",
    "PDC000149",
    "PDC000232",
    "PDC000465",
]


def strip_log_ratio(col: str) -> str:
    return col.replace(" Log Ratio", "").strip()


def phospho_column_sample_id(col: str, aliquots: set[str]) -> str | None:
    raw = str(col).strip()
    if raw.endswith(" Log Ratio"):
        sample = raw.replace(" Log Ratio", "").strip()
    elif raw.startswith("Log ") and "/" in raw:
        sample = raw[4:].split("/", 1)[0].strip()
    else:
        return None
    if sample in aliquots:
        return sample
    sample_no_suffix = re.sub(r"\.\d+$", "", sample)
    if sample_no_suffix in aliquots:
        return sample_no_suffix
    return None


def find_phosphosite_file(study: str) -> Path | None:
    candidates = sorted((PDC_REPORT_DIR / study).glob("*phosphosite*.tsv"))
    if not candidates and study == "PDC000205":
        candidates = sorted(
            (
                ROOT
                / "01_data"
                / "multi_omics"
                / "raw"
                / "cptac_gbm_phosphoproteome"
                / "pdc000205"
            ).glob("*phosphosite*.tsv")
        )
    return candidates[0] if candidates else None


def read_gdc_rna(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep="\t", comment="#")
    df = df[df["gene_type"].eq("protein_coding")].copy()
    vals = pd.to_numeric(df["tpm_unstranded"], errors="coerce")
    s = pd.Series(np.log2(vals.to_numpy(dtype=float) + 1.0), index=df["gene_name"].astype(str))
    return s.groupby(level=0).mean()


def read_phosphosite_matrix(path: Path, aliquots: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    header = pd.read_csv(path, sep="\t", nrows=0)
    alias_map = {}
    for c in header.columns:
        sample_id = phospho_column_sample_id(c, aliquots)
        if sample_id is not None:
            alias_map[c] = sample_id
    sample_cols = list(alias_map)
    usecols = ["Phosphosite", "Gene", "Peptide"] + sample_cols
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in set(usecols))
    df = df.dropna(subset=["Gene", "Phosphosite"])
    df["feature_id"] = df["Gene"].astype(str) + "|" + df["Phosphosite"].astype(str)
    value = df[["feature_id"] + sample_cols].copy()
    value = value.groupby("feature_id", as_index=True).mean(numeric_only=True)
    value = value.rename(columns=alias_map)
    value = value.T.groupby(level=0).mean().T
    target_manifest = (
        df[["feature_id", "Gene", "Phosphosite", "Peptide"]]
        .drop_duplicates("feature_id")
        .rename(columns={"Gene": "gene", "Phosphosite": "phosphosite", "Peptide": "peptide"})
    )
    return value.T, target_manifest


def is_tumor_record(sample_type: object, tissue_type: object = "") -> bool:
    text = f"{sample_type} {tissue_type}".lower()
    return "tumor" in text and "normal" not in text


def build_pairs_for_study(study: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pdc_meta_path = META_DIR / f"pdc_{study}_cases_samples_aliquots.tsv"
    gdc_meta_path = META_DIR / f"gdc_open_star_counts_for_{study}.tsv"
    phospho_path = find_phosphosite_file(study)
    if not pdc_meta_path.exists() or not gdc_meta_path.exists() or phospho_path is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    pdc = pd.read_csv(pdc_meta_path, sep="\t")
    gdc = pd.read_csv(gdc_meta_path, sep="\t")
    pdc_tumor = pdc[
        [
            is_tumor_record(sample_type, tissue_type)
            for sample_type, tissue_type in zip(pdc["sample_type"], pdc["tissue_type"])
        ]
    ].copy()
    gdc_tumor = gdc[
        [
            is_tumor_record(sample_type, tissue_type)
            for sample_type, tissue_type in zip(gdc["sample_type_gdc"], gdc["tissue_type_gdc"])
        ]
    ].copy()

    aliquots = set(pdc_tumor["aliquot_submitter_id"].astype(str))
    y_all, target_manifest = read_phosphosite_matrix(phospho_path, aliquots)
    y_all = y_all.loc[y_all.index.intersection(aliquots)]
    if y_all.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), target_manifest

    pdc_tumor = pdc_tumor[pdc_tumor["aliquot_submitter_id"].isin(y_all.index)].copy()
    pdc_tumor = pdc_tumor.sort_values(["case_submitter_id", "aliquot_submitter_id"])
    pdc_first = pdc_tumor.drop_duplicates("case_submitter_id", keep="first")
    gdc_tumor = gdc_tumor.sort_values(["case_submitter_id", "sample_submitter_id_gdc"])
    gdc_first = gdc_tumor.drop_duplicates("case_submitter_id", keep="first")

    paired = pdc_first.merge(
        gdc_first,
        on="case_submitter_id",
        suffixes=("_pdc", "_gdc"),
        how="inner",
    )
    if paired.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), target_manifest

    rna_rows = []
    y_rows = []
    manifest_rows = []
    for _, row in paired.iterrows():
        rna_path = GDC_DIR / study / row["file_name"]
        if not rna_path.exists():
            continue
        sample_id = f"{study}:{row['case_submitter_id']}"
        aliquot = str(row["aliquot_submitter_id"])
        if aliquot not in y_all.index:
            continue
        rna = read_gdc_rna(rna_path)
        rna.name = sample_id
        phospho = y_all.loc[aliquot].copy()
        phospho.name = sample_id
        rna_rows.append(rna)
        y_rows.append(phospho)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "pdc_study_id": study,
                "cancer_label": STUDY_TO_CANCER.get(study, study),
                "case_submitter_id": row["case_submitter_id"],
                "pdc_sample_submitter_id": row["sample_submitter_id"],
                "pdc_aliquot_submitter_id": aliquot,
                "gdc_sample_submitter_id": row["sample_submitter_id_gdc"],
                "gdc_file_name": row["file_name"],
                "phosphosite_file": str(phospho_path),
            }
        )

    if not rna_rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), target_manifest
    x = pd.DataFrame(rna_rows)
    y = pd.DataFrame(y_rows)
    manifest = pd.DataFrame(manifest_rows)
    return x, y, manifest, target_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", nargs="+", default=DEFAULT_STUDIES)
    parser.add_argument("--min-target-coverage", type=float, default=0.20)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_x, all_y, all_manifest, all_targets, summaries = [], [], [], [], []
    for study in args.studies:
        x, y, manifest, targets = build_pairs_for_study(study)
        all_targets.append(targets.assign(pdc_study_id=study) if not targets.empty else targets)
        summaries.append(
            {
                "pdc_study_id": study,
                "cancer_label": STUDY_TO_CANCER.get(study, study),
                "paired_samples": 0 if manifest.empty else len(manifest),
                "rna_genes": 0 if x.empty else x.shape[1],
                "phosphosite_targets_raw": 0 if y.empty else y.shape[1],
            }
        )
        print(study, "paired", 0 if manifest.empty else len(manifest), "targets", 0 if y.empty else y.shape[1])
        if not x.empty:
            all_x.append(x)
            all_y.append(y)
            all_manifest.append(manifest)

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT_DIR / "pairing_summary_by_study.tsv", sep="\t", index=False)
    if not all_x:
        print("No paired samples were created")
        return 1

    x_all = pd.concat(all_x, axis=0, join="outer").sort_index(axis=1)
    y_all = pd.concat(all_y, axis=0, join="outer").sort_index(axis=1)
    manifest_all = pd.concat(all_manifest, axis=0, ignore_index=True)
    target_all = pd.concat([t for t in all_targets if not t.empty], axis=0, ignore_index=True)
    target_all = target_all.drop_duplicates(["feature_id", "pdc_study_id"])

    keep_targets = y_all.notna().mean(axis=0) >= args.min_target_coverage
    y_filtered = y_all.loc[:, keep_targets]
    target_seen = pd.DataFrame(
        {
            "feature_id": y_all.columns,
            "sample_coverage": y_all.notna().mean(axis=0).to_numpy(),
            "n_observed": y_all.notna().sum(axis=0).to_numpy(),
            "kept_min_coverage": keep_targets.to_numpy(),
        }
    )
    target_manifest = target_all.merge(target_seen, on="feature_id", how="outer")

    x_all.to_parquet(OUT_DIR / "rna_log2_tpm_paired.parquet")
    y_all.to_parquet(OUT_DIR / "phosphosite_logratio_paired_all_targets.parquet")
    y_filtered.to_parquet(OUT_DIR / "phosphosite_logratio_paired_min20pct_targets.parquet")
    manifest_all.to_csv(OUT_DIR / "sample_manifest.tsv", sep="\t", index=False)
    target_manifest.to_csv(OUT_DIR / "target_manifest.tsv", sep="\t", index=False)

    final_summary = {
        "paired_samples": len(manifest_all),
        "rna_genes": x_all.shape[1],
        "phosphosite_targets_all": y_all.shape[1],
        "phosphosite_targets_min20pct": y_filtered.shape[1],
    }
    pd.DataFrame([final_summary]).to_csv(OUT_DIR / "paired_matrix_summary.tsv", sep="\t", index=False)
    print(final_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
