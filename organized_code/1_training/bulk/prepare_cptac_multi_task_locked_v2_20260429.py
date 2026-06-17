#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/data/lsy/Infinite_Stream")
META = ROOT / "01_data/multi_omics/metadata"
GDC = ROOT / "01_data/multi_omics/raw/gdc_cptac_open_star_counts"
PHOSPHO_DIRS = [
    ROOT / "01_data/multi_omics/raw/cptac_pancancer_phosphoproteome_reports",
    ROOT / "01_data/multi_omics/raw/cptac_requested_phosphoproteome_reports",
]
TOTAL_DIRS = [
    ROOT / "01_data/multi_omics/raw/cptac_pancancer_proteome_reports",
    ROOT / "01_data/multi_omics/raw/cptac_requested_proteome_reports",
]
OUT = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2"
RES = ROOT / "02_results/model_validation/20260429_cptac_multi_task_locked_v2"


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
    "PDC000121": "BRCA_PROSPECTIVE",
    "PDC000174": "BRCA_TCGA",
    "PDC000117": "COAD_PROSPECTIVE",
    "PDC000205": "GBM_DISCOVERY",
    "PDC000448": "GBM_CONFIRMATORY",
    "PDC000115": "OV_TCGA",
    "PDC000119": "OV_PROSPECTIVE",
}

PHOSPHO_TO_TOTAL = {
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
    "PDC000121": "PDC000120",
    "PDC000174": "PDC000173",
    "PDC000117": "PDC000116",
    "PDC000205": "PDC000204",
    "PDC000448": "PDC000446",
    "PDC000115": "PDC000114",
    "PDC000119": "PDC000118",
}

DEFAULT_STUDIES = list(STUDY_TO_CANCER)


def is_tumor(sample_type: object, tissue_type: object = "") -> bool:
    text = f"{sample_type} {tissue_type}".lower()
    return "tumor" in text and "normal" not in text


def clean_sample_token(col: str) -> str | None:
    raw = str(col).strip()
    if "Unshared" in raw:
        return None
    if raw.endswith(" Log Ratio"):
        raw = raw.replace(" Log Ratio", "").strip()
    elif raw.startswith("Log ") and "/" in raw:
        raw = raw[4:].split("/", 1)[0].strip()
    else:
        return None
    raw = re.sub(r"\.\d+$", "", raw)
    return raw


def aliases_for_row(row: pd.Series) -> set[str]:
    vals = {
        str(row.get("aliquot_submitter_id", "")),
        str(row.get("sample_submitter_id", "")),
        str(row.get("case_submitter_id", "")),
    }
    out = set()
    for val in vals:
        val = val.strip()
        if not val or val.lower() == "nan":
            continue
        out.add(val)
        if val.startswith("TCGA-"):
            out.add(val[5:])
        if len(val) >= 16 and val.startswith("TCGA-"):
            out.add(val[5:16])
        if len(val) >= 12:
            out.add(val[:12])
    return out


def read_gdc_rna(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep="\t", comment="#")
    df = df[df["gene_type"].eq("protein_coding")].copy()
    vals = pd.to_numeric(df["tpm_unstranded"], errors="coerce")
    out = pd.Series(np.log2(vals.to_numpy(dtype=float) + 1.0), index=df["gene_name"].astype(str))
    return out.groupby(level=0).mean()


def find_file(study: str, dirs: list[Path], layer: str) -> Path | None:
    files = []
    for root in dirs:
        files.extend(sorted((root / study).glob("*.tsv")))
        files.extend(sorted((root / study.lower()).glob("*.tsv")))
    keep = []
    for path in files:
        low = path.name.lower()
        if layer == "phospho":
            if "phosphosite" in low and "phosphopeptide" not in low and "peptides" not in low:
                keep.append(path)
        else:
            if ("proteome" in low or "protein" in low) and not re.search(
                r"phospho|glyco|ubiquit|acetyl|peptides|peptide|summary|qcmetrics|label|sample|metadata|protocol",
                low,
            ):
                keep.append(path)
    return keep[0] if keep else None


def find_summary_file(study: str) -> Path | None:
    files = []
    for root in PHOSPHO_DIRS:
        files.extend(sorted((root / study).glob("*summary*.tsv")))
        files.extend(sorted((root / study.lower()).glob("*summary*.tsv")))
    return files[0] if files else None


def refseq_to_gene_from_summary(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    header = pd.read_csv(path, sep="\t", nrows=0)
    if "Gene" not in header.columns or "Proteins" not in header.columns:
        return {}
    df = pd.read_csv(path, sep="\t", usecols=["Gene", "Proteins"])
    out = {}
    for _, row in df.dropna(subset=["Gene", "Proteins"]).iterrows():
        gene = str(row["Gene"])
        for token in str(row["Proteins"]).split(";"):
            token = token.strip()
            if token:
                out[token] = gene
    return out


def canonical_site(phosphosite: str) -> str:
    text = str(phosphosite)
    if ":" in text:
        text = text.split(":", 1)[1]
    mods = re.findall(r"[stySTY]\d+", text)
    if not mods:
        return text.lower()
    parsed = [(m[0].upper(), int(m[1:])) for m in mods]
    parsed.sort(key=lambda x: (x[1], x[0]))
    return "_".join(f"{aa}{pos}" for aa, pos in parsed)


def feature_ids(df: pd.DataFrame, study: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    phosphosite = df["Phosphosite"].astype(str)
    if "Gene" in df.columns:
        gene = df["Gene"].astype(str)
    else:
        refmap = refseq_to_gene_from_summary(find_summary_file(study))
        acc = phosphosite.str.split(":", n=1).str[0]
        gene = acc.map(refmap)
    peptide = df["Peptide"].astype(str) if "Peptide" in df.columns else pd.Series([""] * len(df), index=df.index)
    return gene, phosphosite, peptide


def paired_manifest(study: str) -> pd.DataFrame:
    pdc = pd.read_csv(META / f"pdc_{study}_cases_samples_aliquots.tsv", sep="\t")
    gdc = pd.read_csv(META / f"gdc_open_star_counts_for_{study}.tsv", sep="\t")
    pdc = pdc[[is_tumor(a, b) for a, b in zip(pdc["sample_type"], pdc["tissue_type"])]].copy()
    gdc = gdc[[is_tumor(a, b) for a, b in zip(gdc["sample_type_gdc"], gdc["tissue_type_gdc"])]].copy()
    pdc = pdc.sort_values(["case_submitter_id", "aliquot_submitter_id"]).drop_duplicates("case_submitter_id", keep="first")
    gdc = gdc.sort_values(["case_submitter_id", "sample_submitter_id_gdc"]).drop_duplicates("case_submitter_id", keep="first")
    paired = pdc.merge(gdc, on="case_submitter_id", how="inner", suffixes=("_pdc", "_gdc"))
    rows = []
    for _, row in paired.iterrows():
        sample_id = f"{study}:{row['case_submitter_id']}"
        rna_path = GDC / study / str(row["file_name"])
        if not rna_path.exists():
            continue
        rec = row.to_dict()
        rec["sample_id"] = sample_id
        rec["rna_path"] = str(rna_path)
        rows.append(rec)
    return pd.DataFrame(rows)


def read_phospho(path: Path, study: str, paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    header = pd.read_csv(path, sep="\t", nrows=0)
    alias = {}
    alias_to_sample = {}
    for _, row in paired.iterrows():
        for token in aliases_for_row(row):
            alias_to_sample[token] = row["sample_id"]
    for col in header.columns:
        token = clean_sample_token(col)
        if token in alias_to_sample:
            alias[col] = alias_to_sample[token]
    if not alias:
        return pd.DataFrame(), pd.DataFrame()
    meta_cols = [c for c in ["Phosphosite", "Gene", "Peptide"] if c in header.columns]
    df = pd.read_csv(path, sep="\t", usecols=meta_cols + list(alias))
    gene, phosphosite, peptide = feature_ids(df, study)
    keep = gene.notna() & phosphosite.notna()
    df = df.loc[keep].copy()
    gene = gene.loc[keep].astype(str)
    phosphosite = phosphosite.loc[keep].astype(str)
    peptide = peptide.loc[keep].astype(str)
    feature = gene + "|" + phosphosite
    value = df[list(alias)].copy()
    value.columns = [alias[c] for c in value.columns]
    value.index = feature
    mat = value.groupby(level=0).mean(numeric_only=True).T.groupby(level=0).mean()
    target = pd.DataFrame(
        {
            "feature_id": feature.values,
            "gene": gene.values,
            "phosphosite": phosphosite.values,
            "peptide": peptide.values,
            "pdc_study_id": study,
        }
    ).drop_duplicates(["feature_id", "pdc_study_id"])
    return mat, target


def read_total(path: Path, paired: pd.DataFrame) -> pd.DataFrame:
    header = pd.read_csv(path, sep="\t", nrows=0)
    alias = {}
    alias_to_sample = {}
    for _, row in paired.iterrows():
        for token in aliases_for_row(row):
            alias_to_sample[token] = row["sample_id"]
    for col in header.columns:
        token = clean_sample_token(col)
        if token in alias_to_sample:
            alias[col] = alias_to_sample[token]
    if not alias or "Gene" not in header.columns:
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t", usecols=["Gene"] + list(alias))
    df["Gene"] = df["Gene"].astype(str)
    bad = {"Mean", "Median", "StdDev", "NumSpectra", "QValue", "Unshared Peptides", "Distinct Peptides"}
    df = df[~df["Gene"].isin(bad)].copy()
    val = df[["Gene"] + list(alias)].rename(columns=alias).groupby("Gene", as_index=True).mean(numeric_only=True)
    return val.T.groupby(level=0).mean()


def study_zscore(df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for study, idx in manifest.groupby("pdc_study_id").groups.items():
        ids = list(idx)
        sub = out.loc[ids]
        mean = sub.mean(axis=0, skipna=True)
        std = sub.std(axis=0, skipna=True).mask(lambda s: (s < 1e-6) | s.isna(), 1.0)
        out.loc[ids] = (sub - mean) / std
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", nargs="+", default=DEFAULT_STUDIES)
    parser.add_argument("--min-phospho-coverage", type=float, default=0.20)
    parser.add_argument("--min-protein-coverage", type=float, default=0.20)
    args = parser.parse_args()
    for path in [OUT, RES / "tables", RES / "logs"]:
        path.mkdir(parents=True, exist_ok=True)

    x_parts, y_parts, total_parts, manifest_rows, targets, logs = [], [], [], [], [], []
    for study in args.studies:
        phospho_path = find_file(study, PHOSPHO_DIRS, "phospho")
        total_path = find_file(PHOSPHO_TO_TOTAL[study], TOTAL_DIRS, "total")
        paired = paired_manifest(study)
        status = "ok"
        if phospho_path is None:
            status = "missing_phosphosite_file"
        elif total_path is None:
            status = "missing_total_file"
        elif paired.empty:
            status = "no_paired_rna"
        if status != "ok":
            logs.append({"pdc_study_id": study, "status": status, "paired_rows": int(paired.shape[0])})
            continue
        y, target = read_phospho(phospho_path, study, paired)
        total = read_total(total_path, paired)
        if y.empty:
            status = "no_phospho_sample_columns_matched"
        elif total.empty:
            status = "no_total_sample_columns_matched"
        if status != "ok":
            logs.append({"pdc_study_id": study, "status": status, "paired_rows": int(paired.shape[0]), "phospho_file": str(phospho_path), "total_file": str(total_path)})
            continue
        ids = sorted(set(y.index) & set(total.index) & set(paired["sample_id"]))
        if not ids:
            logs.append({"pdc_study_id": study, "status": "no_common_samples_after_matrix_read", "paired_rows": int(paired.shape[0]), "phospho_file": str(phospho_path), "total_file": str(total_path)})
            continue
        rna_rows = []
        paired_index = paired.set_index("sample_id")
        for sid in ids:
            rna = read_gdc_rna(Path(paired_index.loc[sid, "rna_path"]))
            rna.name = sid
            rna_rows.append(rna)
            row = paired_index.loc[sid]
            manifest_rows.append(
                {
                    "sample_id": sid,
                    "pdc_study_id": study,
                    "cancer_label": STUDY_TO_CANCER[study],
                    "case_submitter_id": row["case_submitter_id"],
                    "pdc_sample_submitter_id": row["sample_submitter_id"],
                    "pdc_aliquot_submitter_id": row["aliquot_submitter_id"],
                    "gdc_sample_submitter_id": row["sample_submitter_id_gdc"],
                    "gdc_file_name": row["file_name"],
                    "phosphosite_file": str(phospho_path),
                    "total_protein_file": str(total_path),
                }
            )
        x_parts.append(pd.DataFrame(rna_rows))
        y_parts.append(y.loc[ids])
        total_parts.append(total.loc[ids])
        targets.append(target)
        logs.append(
            {
                "pdc_study_id": study,
                "cancer_label": STUDY_TO_CANCER[study],
                "status": "paired",
                "paired_samples": len(ids),
                "phospho_targets_raw": int(y.shape[1]),
                "total_genes_raw": int(total.shape[1]),
                "phospho_file": str(phospho_path),
                "total_file": str(total_path),
            }
        )
        print(study, STUDY_TO_CANCER[study], "paired", len(ids), "phospho", y.shape[1], "total", total.shape[1])

    if not x_parts:
        pd.DataFrame(logs).to_csv(RES / "tables/pairing_log.tsv", sep="\t", index=False)
        return 1

    x = pd.concat(x_parts, axis=0, join="outer").sort_index(axis=1)
    y_raw = pd.concat(y_parts, axis=0, join="outer").sort_index(axis=1)
    total_raw = pd.concat(total_parts, axis=0, join="outer").sort_index(axis=1)
    manifest = pd.DataFrame(manifest_rows).set_index("sample_id").loc[x.index]
    manifest.index.name = "sample_id"
    target_manifest = pd.concat(targets, axis=0, ignore_index=True).drop_duplicates(["feature_id", "pdc_study_id"])

    y_canon = y_raw.copy()
    y_canon.columns = [
        f"{str(c).split('|', 1)[0]}|{canonical_site(str(c).split('|', 1)[1])}" if "|" in str(c) else str(c)
        for c in y_canon.columns
    ]
    y_gene_site_all = y_canon.T.groupby(level=0).mean().T
    coverage = y_gene_site_all.notna().mean(axis=0)
    y_gene_site = y_gene_site_all.loc[:, coverage >= args.min_phospho_coverage].copy()
    y_z = study_zscore(y_gene_site, manifest)

    protein_cov = total_raw.notna().mean(axis=0)
    total_filt = total_raw.loc[:, protein_cov >= args.min_protein_coverage].copy()
    total_z = study_zscore(total_filt, manifest)

    target_gene = pd.Series(y_z.columns, index=y_z.columns).str.split("|", regex=False).str[0]
    matched = target_gene[target_gene.isin(total_z.columns)]
    residual = y_z.loc[:, matched.index].copy()
    for target, gene in matched.items():
        residual[target] = residual[target] - total_z[gene]

    target_seen = pd.DataFrame(
        {
            "gene_site_id": y_gene_site.columns,
            "sample_coverage": y_gene_site.notna().mean(axis=0).values,
            "n_observed": y_gene_site.notna().sum(axis=0).values,
        }
    )
    target_seen["gene"] = target_seen["gene_site_id"].str.split("|", regex=False).str[0]
    target_seen["site_canonical"] = target_seen["gene_site_id"].str.split("|", regex=False).str[1]
    residual_manifest = pd.DataFrame(
        {
            "gene_site_id": matched.index,
            "total_protein_gene": matched.values,
            "phospho_sample_coverage": y_z.loc[:, matched.index].notna().mean(axis=0).values,
            "total_protein_sample_coverage": total_z.loc[:, matched.values].notna().mean(axis=0).values,
        }
    )

    x.to_parquet(OUT / "rna_log2_tpm_paired.parquet")
    y_raw.to_parquet(OUT / "phosphosite_gene_refseq_logratio_all_targets.parquet")
    y_gene_site.to_parquet(OUT / "phosphosite_gene_site_logratio_min20pct_targets.parquet")
    y_z.to_parquet(OUT / "phosphosite_gene_site_study_zscore_min20pct_targets.parquet")
    total_raw.to_parquet(OUT / "total_protein_gene_logratio_all.parquet")
    total_z.to_parquet(OUT / "total_protein_gene_study_zscore_min20pct.parquet")
    residual.to_parquet(OUT / "phosphosite_gene_site_total_residual_targets.parquet")
    manifest.reset_index().to_csv(OUT / "sample_manifest.tsv", sep="\t", index=False)
    target_manifest.to_csv(OUT / "target_manifest_raw_feature_v2.tsv", sep="\t", index=False)
    target_seen.to_csv(OUT / "target_manifest_gene_site_locked_v2.tsv", sep="\t", index=False)
    residual_manifest.to_csv(OUT / "residual_target_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(logs).to_csv(RES / "tables/pairing_log.tsv", sep="\t", index=False)

    summary = {
        "version": "cptac_pancancer_multi_task_locked_v2",
        "n_samples": int(x.shape[0]),
        "n_studies": int(manifest["pdc_study_id"].nunique()),
        "n_cancer_contexts": int(manifest["cancer_label"].nunique()),
        "n_rna_genes": int(x.shape[1]),
        "n_total_protein_genes_all": int(total_raw.shape[1]),
        "n_total_protein_genes_min20pct": int(total_z.shape[1]),
        "n_phosphosite_targets_raw_refseq": int(y_raw.shape[1]),
        "n_phosphosite_gene_site_min20pct": int(y_z.shape[1]),
        "n_residual_targets_with_total_protein": int(residual.shape[1]),
        "min_phospho_coverage": args.min_phospho_coverage,
        "min_protein_coverage": args.min_protein_coverage,
        "residual_rule": "study-zscored phosphosite gene-site minus study-zscored total protein abundance for the same gene",
    }
    (OUT / "MULTITASK_DATA_CARD.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(OUT / "multi_task_matrix_summary.tsv", sep="\t", index=False)
    hashes = []
    for path in sorted(OUT.glob("*")):
        if path.is_file():
            hashes.append({"file": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    pd.DataFrame(hashes).to_csv(OUT / "LOCKED_FILE_HASHES.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
