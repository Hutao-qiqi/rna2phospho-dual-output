#!/usr/bin/env python3
"""Audit the desired 32-project TCGA RNA to TCPA RPPA data contract."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path("/data/lsy/Infinite_Stream")
OUT = ROOT / "02_results/model_validation/20260501_tcpa_32_project_data_contract_v1"
GDC_FILES = "https://api.gdc.cancer.gov/files"


def mkdirs() -> None:
    for sub in ["tables", "logs", "reports"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def read_rppa_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(ROOT / "01_data/tcga_tcpa/raw/gdc_tcga_rppa_manifest.tsv", sep="\t")
    manifest["sample_submitter_id"] = manifest["sample_submitter_id"].astype(str)
    manifest["sample_short"] = manifest["sample_submitter_id"].str[:16]
    manifest = manifest.drop_duplicates("sample_short", keep="first")
    return manifest[["sample_short", "case_submitter_id", "sample_submitter_id", "project_id"]].copy()


def read_rppa_l4_samples() -> pd.DataFrame:
    rppa = pd.read_csv(ROOT / "data/raw/tcpa/PANCAN_RPPA_L4.tsv", sep="\t", usecols=["sample_id"])
    rppa["sample_id"] = rppa["sample_id"].astype(str)
    rppa["sample_short"] = rppa["sample_id"].str[:16]
    return rppa.drop_duplicates("sample_short")


def read_current_rna_samples() -> pd.DataFrame:
    x = pd.read_parquet(ROOT / "data/processed/X_all.symbols.parquet")
    samples = x.columns.drop("gene_symbol") if "gene_symbol" in x.columns else x.columns
    out = pd.DataFrame({"rna_sample_id": samples.astype(str)})
    out["sample_short"] = out["rna_sample_id"].str[:16]
    return out.drop_duplicates("sample_short")


def read_current_master_samples() -> pd.DataFrame:
    master = pd.read_csv(ROOT / "data/interim/master_index.tsv", sep="\t")
    master["tcpa_sample_id"] = master["tcpa_sample_id"].astype(str)
    master["sample_short"] = master["tcpa_sample_id"].str[:16]
    return master.drop_duplicates("sample_short")


def make_gap_audit() -> tuple[pd.DataFrame, dict]:
    rppa_manifest = read_rppa_manifest()
    rppa_l4 = read_rppa_l4_samples()
    rna = read_current_rna_samples()
    master = read_current_master_samples()

    rppa = rppa_manifest.merge(rppa_l4[["sample_short"]], on="sample_short", how="inner")
    rppa["has_current_rna"] = rppa["sample_short"].isin(set(rna["sample_short"]))
    rppa["in_current_master"] = rppa["sample_short"].isin(set(master["sample_short"]))

    rows = []
    for project, sub in rppa.groupby("project_id"):
        rows.append(
            {
                "project_id": project,
                "n_rppa_l4_samples": int(sub.shape[0]),
                "n_current_rna_samples": int(sub["has_current_rna"].sum()),
                "n_current_master_samples": int(sub["in_current_master"].sum()),
                "has_any_current_rna": bool(sub["has_current_rna"].any()),
                "has_any_current_master": bool(sub["in_current_master"].any()),
                "needs_rna_download": bool(not sub["has_current_rna"].any()),
            }
        )
    audit = pd.DataFrame(rows).sort_values("project_id")
    summary = {
        "n_rppa_projects": int(audit.shape[0]),
        "n_rppa_l4_samples": int(rppa.shape[0]),
        "n_projects_with_current_rna": int((audit["n_current_rna_samples"] > 0).sum()),
        "n_projects_missing_current_rna": int((audit["n_current_rna_samples"] == 0).sum()),
        "missing_projects": audit.loc[audit["n_current_rna_samples"].eq(0), "project_id"].tolist(),
        "current_rna_projects": audit.loc[audit["n_current_rna_samples"].gt(0), "project_id"].tolist(),
    }
    return audit, summary


def gdc_filter(projects: list[str]) -> dict:
    return {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": projects}},
            {"op": "in", "content": {"field": "files.data_category", "value": ["Transcriptome Profiling"]}},
            {"op": "in", "content": {"field": "files.data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "files.analysis.workflow_type", "value": ["STAR - Counts"]}},
            {"op": "in", "content": {"field": "cases.samples.sample_type", "value": ["Primary Tumor"]}},
            {"op": "in", "content": {"field": "files.access", "value": ["open"]}},
        ],
    }


def query_gdc_star_counts(projects: list[str], size: int = 10000) -> pd.DataFrame:
    fields = [
        "file_id",
        "file_name",
        "cases.submitter_id",
        "cases.project.project_id",
        "cases.samples.submitter_id",
        "cases.samples.sample_type",
    ]
    params = {
        "filters": json.dumps(gdc_filter(projects)),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": str(size),
        "sort": "cases.project.project_id:asc",
    }
    url = GDC_FILES + "?" + urllib.parse.urlencode(params)
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(url, timeout=120) as handle:
                payload = json.loads(handle.read().decode("utf-8"))
            hits = payload.get("data", {}).get("hits", [])
            rows = []
            for hit in hits:
                file_id = hit.get("file_id")
                file_name = hit.get("file_name")
                for case in hit.get("cases", []) or []:
                    project_id = ((case.get("project") or {}).get("project_id"))
                    case_id = case.get("submitter_id")
                    for sample in case.get("samples", []) or []:
                        rows.append(
                            {
                                "file_id": file_id,
                                "file_name": file_name,
                                "case_submitter_id": case_id,
                                "sample_submitter_id": sample.get("submitter_id"),
                                "sample_short": str(sample.get("submitter_id"))[:16],
                                "sample_type": sample.get("sample_type"),
                                "project_id": project_id,
                            }
                        )
            return pd.DataFrame(rows).drop_duplicates(["file_id", "sample_short"])
        except Exception as exc:
            if attempt == 5:
                raise
            print(f"GDC query failed attempt {attempt}/5: {exc}", flush=True)
            time.sleep(5 * attempt)
    return pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-gdc-rna", action="store_true")
    args = parser.parse_args()

    mkdirs()
    audit, summary = make_gap_audit()
    audit.to_csv(OUT / "tables/tcpa_32_project_rppa_rna_gap_audit.tsv", sep="\t", index=False)
    audit[["project_id"]].to_csv(OUT / "tables/tcpa_32_project_list.tsv", sep="\t", index=False)
    audit.loc[audit["needs_rna_download"]].to_csv(OUT / "tables/tcpa_32_missing_rna_projects.tsv", sep="\t", index=False)

    if args.query_gdc_rna:
        missing = summary["missing_projects"]
        gdc = query_gdc_star_counts(missing)
        rppa_manifest = read_rppa_manifest()
        rppa_l4 = read_rppa_l4_samples()
        wanted = rppa_manifest.merge(rppa_l4[["sample_short"]], on="sample_short", how="inner")
        wanted = wanted.loc[wanted["project_id"].isin(missing)].copy()
        gdc["matches_rppa_l4"] = gdc["sample_short"].isin(set(wanted["sample_short"]))
        gdc.to_csv(OUT / "tables/gdc_star_counts_manifest_missing_13_projects.tsv", sep="\t", index=False)
        matched = gdc.loc[gdc["matches_rppa_l4"]].copy()
        matched.to_csv(OUT / "tables/gdc_star_counts_manifest_missing_13_projects_matched_to_rppa.tsv", sep="\t", index=False)
        summary["gdc_query"] = {
            "n_star_count_rows": int(gdc.shape[0]),
            "n_star_count_projects": int(gdc["project_id"].nunique()) if not gdc.empty else 0,
            "n_matched_rppa_star_count_rows": int(matched.shape[0]),
            "n_matched_rppa_star_count_projects": int(matched["project_id"].nunique()) if not matched.empty else 0,
        }

    (OUT / "logs/tcpa_32_project_data_contract_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = [
        "# TCGA 32-project RNA to TCPA RPPA data contract",
        "",
        "Final TCPA branch target: TCGA RNA to TCPA RPPA total and phospho antibody prediction across the 32 RPPA projects present in the GDC RPPA manifest.",
        "",
        f"RPPA projects: {summary['n_rppa_projects']}",
        f"RPPA L4 samples with project mapping: {summary['n_rppa_l4_samples']}",
        f"Projects already covered by current RNA matrix: {summary['n_projects_with_current_rna']}",
        f"Projects requiring RNA download: {summary['n_projects_missing_current_rna']}",
        "",
        "Missing RNA projects:",
        ", ".join(summary["missing_projects"]),
    ]
    (OUT / "reports/tcpa_32_project_data_contract.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
