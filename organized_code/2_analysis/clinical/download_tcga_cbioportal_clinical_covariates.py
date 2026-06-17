#!/usr/bin/env python
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


BASE = "https://www.cbioportal.org/api"


def api_json(path: str, timeout: int = 120):
    url = BASE + path
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_all_clinical(study_id: str, clinical_type: str) -> pd.DataFrame:
    rows = []
    page_size = 10000
    page = 0
    while True:
        qs = urllib.parse.urlencode({
            "clinicalDataType": clinical_type,
            "projection": "SUMMARY",
            "pageSize": page_size,
            "pageNumber": page,
        })
        chunk = api_json(f"/studies/{study_id}/clinical-data?{qs}")
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < page_size:
            break
        page += 1
        time.sleep(0.15)
    return pd.DataFrame(rows)


def pivot_patient(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    wide = (
        df.pivot_table(
            index=["studyId", "patientId"],
            columns="clinicalAttributeId",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    wide.columns = [c if c in {"studyId", "patientId"} else f"patient_{c}" for c in wide.columns]
    return wide


def pivot_sample(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    wide = (
        df.pivot_table(
            index=["studyId", "patientId", "sampleId"],
            columns="clinicalAttributeId",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    wide.columns = [c if c in {"studyId", "patientId", "sampleId"} else f"sample_{c}" for c in wide.columns]
    return wide


def stage_to_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).upper().replace("STAGE", "").strip()
    if not s or s in {"NAN", "NA", "NOT REPORTED", "UNKNOWN"}:
        return np.nan
    for roman, value in [("IV", 4), ("III", 3), ("II", 2), ("I", 1)]:
        if s.startswith(roman):
            return float(value)
    return np.nan


def grade_to_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).upper().strip()
    if not s or s in {"NAN", "NA", "NOT REPORTED", "UNKNOWN"}:
        return np.nan
    for token in ["G4", "GRADE 4", "HIGH GRADE"]:
        if token in s:
            return 4.0
    for token in ["G3", "GRADE 3"]:
        if token in s:
            return 3.0
    for token in ["G2", "GRADE 2", "INTERMEDIATE GRADE"]:
        if token in s:
            return 2.0
    for token in ["G1", "GRADE 1", "LOW GRADE"]:
        if token in s:
            return 1.0
    try:
        return float(s)
    except Exception:
        return np.nan


def main() -> int:
    out = Path("/data/lsy/Infinite_Stream/02_results/model_prediction/20260529_tcga_full_scp682_main_reprediction_v1/clinical_covariates")
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = Path("/data/lsy/Infinite_Stream/02_results/model_prediction/20260529_tcga_full_scp682_main_reprediction_v1/tables/tcga_scp682_prediction_sample_manifest.tsv")
    manifest = pd.read_csv(manifest_path, sep="\t")
    cancers = sorted(manifest["cancer"].dropna().unique())

    studies = api_json("/studies?projection=SUMMARY")
    all_study_ids = {s["studyId"] for s in studies}
    study_for_cancer = {}
    for cancer in cancers:
        candidate = f"{cancer.lower()}_tcga_pan_can_atlas_2018"
        if candidate in all_study_ids:
            study_for_cancer[cancer] = candidate
        elif cancer in {"COAD", "READ"} and "coadread_tcga_pan_can_atlas_2018" in all_study_ids:
            study_for_cancer[cancer] = "coadread_tcga_pan_can_atlas_2018"
        else:
            study_for_cancer[cancer] = None

    pd.DataFrame(
        [{"cancer": k, "cbioportal_study_id": v} for k, v in study_for_cancer.items()]
    ).to_csv(out / "tcga_cbioportal_study_map.tsv", sep="\t", index=False)

    patient_tables = []
    sample_tables = []
    attr_tables = []
    failed = []
    for cancer, study_id in study_for_cancer.items():
        if not study_id:
            failed.append({"cancer": cancer, "study_id": "", "reason": "study_not_found"})
            continue
        try:
            attrs = pd.DataFrame(api_json(f"/studies/{study_id}/clinical-attributes?projection=SUMMARY"))
            attrs["cancer"] = cancer
            attr_tables.append(attrs)
            pat = fetch_all_clinical(study_id, "PATIENT")
            samp = fetch_all_clinical(study_id, "SAMPLE")
            if not pat.empty:
                pat["cancer"] = cancer
                patient_tables.append(pat)
            if not samp.empty:
                samp["cancer"] = cancer
                sample_tables.append(samp)
            print(cancer, study_id, "patient_rows", len(pat), "sample_rows", len(samp), flush=True)
        except Exception as e:
            failed.append({"cancer": cancer, "study_id": study_id, "reason": repr(e)})
        time.sleep(0.2)

    attrs_all = pd.concat(attr_tables, ignore_index=True) if attr_tables else pd.DataFrame()
    patient_long = pd.concat(patient_tables, ignore_index=True) if patient_tables else pd.DataFrame()
    sample_long = pd.concat(sample_tables, ignore_index=True) if sample_tables else pd.DataFrame()
    attrs_all.to_csv(out / "tcga_cbioportal_clinical_attributes_long.tsv", sep="\t", index=False)
    patient_long.to_csv(out / "tcga_cbioportal_patient_clinical_long.tsv", sep="\t", index=False)
    sample_long.to_csv(out / "tcga_cbioportal_sample_clinical_long.tsv", sep="\t", index=False)
    pd.DataFrame(failed).to_csv(out / "tcga_cbioportal_clinical_download_failures.tsv", sep="\t", index=False)

    patient_wide = pivot_patient(patient_long)
    sample_wide = pivot_sample(sample_long)
    patient_wide.to_csv(out / "tcga_cbioportal_patient_clinical_wide_all33.tsv", sep="\t", index=False)
    sample_wide.to_csv(out / "tcga_cbioportal_sample_clinical_wide_all33.tsv", sep="\t", index=False)

    # Join onto the exact samples used by the SCP682 TCGA prediction matrix.
    joined = manifest.copy()
    joined["cbioportal_sample_id"] = joined["tcga_sample_type_id"]
    joined["cbioportal_study_id"] = joined["cancer"].map(study_for_cancer)
    joined = joined.merge(
        patient_wide,
        how="left",
        left_on=["cbioportal_study_id", "tcga_patient_id"],
        right_on=["studyId", "patientId"],
    )
    joined = joined.merge(
        sample_wide,
        how="left",
        left_on=["cbioportal_study_id", "tcga_patient_id", "cbioportal_sample_id"],
        right_on=["studyId", "patientId", "sampleId"],
        suffixes=("", "_sample"),
    )

    # Compact analysis-ready columns.
    def first_existing(cols):
        for c in cols:
            if c in joined.columns:
                return joined[c]
        return pd.Series([np.nan] * len(joined), index=joined.index)

    compact = pd.DataFrame({
        "sample_id": joined["sample_id"],
        "tcga_patient_id": joined["tcga_patient_id"],
        "project_id": joined["project_id"],
        "cancer": joined["cancer"],
        "sample_type": joined["sample_type"],
        "sex": first_existing(["patient_SEX", "patient_GENDER"]),
        "age": pd.to_numeric(first_existing(["patient_AGE", "patient_AGE_AT_DIAGNOSIS", "age_at_diagnosis"]), errors="coerce"),
        "stage": first_existing(["patient_AJCC_PATHOLOGIC_TUMOR_STAGE", "sample_AJCC_PATHOLOGIC_TUMOR_STAGE"]),
        "path_t_stage": first_existing(["patient_PATH_T_STAGE", "sample_PATH_T_STAGE"]),
        "path_n_stage": first_existing(["patient_PATH_N_STAGE", "sample_PATH_N_STAGE"]),
        "path_m_stage": first_existing(["patient_PATH_M_STAGE", "sample_PATH_M_STAGE"]),
        "grade": first_existing(["sample_GRADE", "patient_GRADE", "sample_TUMOR_GRADE", "patient_TUMOR_GRADE"]),
        "os_days": pd.to_numeric(joined["survival_time"], errors="coerce"),
        "os_event": pd.to_numeric(joined["survival_event"], errors="coerce"),
    })
    compact["stage_num"] = compact["stage"].map(stage_to_num)
    compact["grade_num"] = compact["grade"].map(grade_to_num)

    purity_candidates = [c for c in joined.columns if "PURITY" in c.upper() or "ABSOLUTE" in c.upper() or "ESTIMATE" in c.upper()]
    for c in purity_candidates:
        compact[c] = joined[c]
    if purity_candidates:
        numeric_purity = None
        for c in purity_candidates:
            vals = pd.to_numeric(joined[c], errors="coerce")
            if vals.notna().sum() > 0:
                numeric_purity = vals
                compact["tumor_purity"] = numeric_purity
                compact["tumor_purity_source"] = c
                break
        if numeric_purity is None:
            compact["tumor_purity"] = np.nan
            compact["tumor_purity_source"] = ""
    else:
        compact["tumor_purity"] = np.nan
        compact["tumor_purity_source"] = ""

    compact.to_csv(out / "tcga_scp682_sample_clinical_covariates_all33.tsv", sep="\t", index=False)
    joined.to_csv(out / "tcga_scp682_sample_clinical_full_joined_all33.tsv", sep="\t", index=False)

    coverage = []
    for col in ["sex", "age", "stage", "stage_num", "grade", "grade_num", "tumor_purity"]:
        coverage.append({
            "field": col,
            "n_nonmissing": int(compact[col].notna().sum()),
            "fraction_nonmissing": float(compact[col].notna().mean()),
        })
    pd.DataFrame(coverage).to_csv(out / "tcga_clinical_covariate_coverage.tsv", sep="\t", index=False)

    print("wrote", out)
    print(pd.DataFrame(coverage).to_string(index=False))
    if purity_candidates:
        print("purity_candidates", purity_candidates)
    else:
        print("purity_candidates NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
