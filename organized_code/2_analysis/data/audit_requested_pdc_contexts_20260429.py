#!/usr/bin/env python3
from pathlib import Path
import json
import time

import pandas as pd
import requests


ROOT = Path("/data/lsy/Infinite_Stream")
META = ROOT / "01_data/multi_omics/metadata"
OUT = ROOT / "02_results/model_validation/20260429_requested_cptac_context_expansion"
PDC = "https://pdc.cancer.gov/graphql"
GDC = "https://api.gdc.cancer.gov/files"


PHOSPHO_TO_TOTAL = {
    "PDC000121": "PDC000120",
    "PDC000174": "PDC000173",
    "PDC000583": "PDC000582",
    "PDC000117": "PDC000116",
    "PDC000205": "PDC000204",
    "PDC000448": "PDC000446",
    "PDC000515": "PDC000514",
    "PDC000115": "PDC000114",
    "PDC000119": "PDC000118",
}

CANCER_LABEL = {
    "PDC000121": "BRCA_PROSPECTIVE",
    "PDC000174": "BRCA_TCGA",
    "PDC000583": "BRCA_TRIAL",
    "PDC000117": "COAD_PROSPECTIVE",
    "PDC000205": "GBM_DISCOVERY",
    "PDC000448": "GBM_CONFIRMATORY",
    "PDC000515": "GBM_KNCC",
    "PDC000115": "OV_TCGA",
    "PDC000119": "OV_PROSPECTIVE",
}

QUERY_FILES = """
query Files($study:String!) {
  filesPerStudy(pdc_study_id: $study) {
    pdc_study_id
    study_name
    file_id
    file_name
    file_type
    file_size
    data_category
    file_format
    file_location
    md5sum
  }
}
"""

QUERY_CASES = """
query Cases($study:String!) {
  paginatedCasesSamplesAliquots(pdc_study_id:$study) {
    casesSamplesAliquots {
      case_submitter_id
      disease_type
      primary_site
      samples {
        sample_submitter_id
        sample_type
        tissue_type
        aliquots { aliquot_submitter_id }
      }
    }
  }
}
"""


def post_graphql(query: str, variables: dict) -> dict:
    for attempt in range(5):
        try:
            resp = requests.post(PDC, json={"query": query, "variables": variables}, timeout=120)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("errors"):
                raise RuntimeError(str(payload["errors"])[:1200])
            return payload["data"]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def collect_files(study: str) -> pd.DataFrame:
    path = META / f"pdc_{study}_files_per_study.tsv"
    if path.exists():
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            pass
    df = pd.DataFrame(post_graphql(QUERY_FILES, {"study": study}).get("filesPerStudy") or [])
    df.to_csv(path, sep="\t", index=False)
    return df


def collect_cases(study: str) -> pd.DataFrame:
    path = META / f"pdc_{study}_cases_samples_aliquots.tsv"
    if path.exists():
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            pass
    rows = []
    for case in post_graphql(QUERY_CASES, {"study": study})["paginatedCasesSamplesAliquots"]["casesSamplesAliquots"]:
        samples = case.get("samples") or [{}]
        for sample in samples:
            aliquots = sample.get("aliquots") or [{}]
            for aliquot in aliquots:
                rows.append(
                    {
                        "pdc_study_id": study,
                        "case_submitter_id": case.get("case_submitter_id"),
                        "disease_type": case.get("disease_type"),
                        "primary_site": case.get("primary_site"),
                        "sample_submitter_id": sample.get("sample_submitter_id"),
                        "sample_type": sample.get("sample_type"),
                        "tissue_type": sample.get("tissue_type"),
                        "aliquot_submitter_id": aliquot.get("aliquot_submitter_id"),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)
    return df


def gdc_open_star_counts(case_ids: list[str]) -> pd.DataFrame:
    if not case_ids:
        return pd.DataFrame()
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": case_ids}},
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["RNA-Seq"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }
    fields = [
        "file_id",
        "file_name",
        "access",
        "cases.submitter_id",
        "cases.project.project_id",
        "cases.samples.submitter_id",
        "cases.samples.sample_type",
        "cases.samples.tissue_type",
        "cases.primary_site",
        "cases.disease_type",
    ]
    params = {"filters": json.dumps(filters), "fields": ",".join(fields), "format": "JSON", "size": "5000"}
    resp = requests.get(GDC, params=params, timeout=120)
    resp.raise_for_status()
    rows = []
    for hit in resp.json()["data"]["hits"]:
        case = (hit.get("cases") or [{}])[0]
        sample = (case.get("samples") or [{}])[0]
        project = case.get("project") or {}
        rows.append(
            {
                "file_id": hit.get("file_id"),
                "file_name": hit.get("file_name"),
                "case_submitter_id": case.get("submitter_id"),
                "gdc_project_id": project.get("project_id"),
                "sample_submitter_id_gdc": sample.get("submitter_id"),
                "sample_type_gdc": sample.get("sample_type"),
                "tissue_type_gdc": sample.get("tissue_type"),
                "access": hit.get("access"),
            }
        )
    return pd.DataFrame(rows)


def count_selected_files(df: pd.DataFrame, kind: str) -> int:
    if df.empty:
        return 0
    name = df["file_name"].fillna("").astype(str).str.lower()
    cat = df["data_category"].fillna("").astype(str).str.lower()
    if kind == "phospho":
        keep = cat.eq("protein assembly") & name.str.endswith(".tsv") & (
            name.str.contains("phosphosite") | name.str.contains("phosphoproteome")
        )
    else:
        keep = (
            cat.eq("protein assembly")
            & name.str.endswith(".tsv")
            & name.str.contains("protein|proteome|tmt|itraq|global", regex=True)
            & ~name.str.contains("phospho|glyco|ubiquit|acetyl|qcmetrics|summary|label|metadata|protocol", regex=True)
        )
    return int(keep.sum())


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    rows = []
    for phospho, total in PHOSPHO_TO_TOTAL.items():
        phospho_files = collect_files(phospho)
        total_files = collect_files(total)
        phospho_cases = collect_cases(phospho)
        case_ids = sorted(phospho_cases["case_submitter_id"].dropna().astype(str).unique()) if not phospho_cases.empty else []
        gdc = gdc_open_star_counts(case_ids)
        gdc.to_csv(META / f"gdc_open_star_counts_for_{phospho}.tsv", sep="\t", index=False)
        tumor_cases = set()
        if not phospho_cases.empty:
            tumor_mask = phospho_cases["tissue_type"].fillna("").eq("Tumor") | phospho_cases["sample_type"].fillna("").str.contains("Tumor", case=False, regex=False)
            tumor_cases = set(phospho_cases.loc[tumor_mask, "case_submitter_id"].dropna().astype(str))
        gdc_tumor_cases = set()
        if not gdc.empty:
            gdc_tumor_mask = gdc["tissue_type_gdc"].fillna("").eq("Tumor") | gdc["sample_type_gdc"].fillna("").str.contains("Tumor", case=False, regex=False)
            gdc_tumor_cases = set(gdc.loc[gdc_tumor_mask, "case_submitter_id"].dropna().astype(str))
        rows.append(
            {
                "cancer_context": CANCER_LABEL[phospho],
                "phospho_study_id": phospho,
                "total_study_id": total,
                "phospho_cases": len(case_ids),
                "pdc_tumor_cases": len(tumor_cases),
                "open_gdc_tumor_rna_overlap": len(tumor_cases & gdc_tumor_cases),
                "gdc_projects": ";".join(sorted(gdc["gdc_project_id"].dropna().unique())) if not gdc.empty else "",
                "n_phospho_matrix_candidates": count_selected_files(phospho_files, "phospho"),
                "n_total_matrix_candidates": count_selected_files(total_files, "total"),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "tables/requested_context_trainability_audit.tsv", sep="\t", index=False)
    print(result.to_string(index=False))
    print("requested_without_pdc_phosphoproteome\tBLCA;SKCM;PRAD")


if __name__ == "__main__":
    main()
