#!/usr/bin/env python3
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from scipy.stats import false_discovery_control
from statsmodels.duration.hazard_regression import PHReg


ROOT = Path("/data/lsy/Infinite_Stream")
STRICT = ROOT / "_codex_inputs/strict_site_specific_candidates_measured_site_rna_total_null.tsv"
PHOS_CENTERED = ROOT / "02_results/public_bulk_phosphoproteome_atlas/20260508_scp682_v4_0_public_bulk_atlas/predictions/SCP682_v4_0_tcga_supported_phosphosite.parquet"
PHOS_RAW = ROOT / "02_results/public_bulk_phosphoproteome_atlas/20260508_scp682_v4_0_public_bulk_atlas/predictions/SCP682_v4_0_tcga_supported_phosphosite_raw_before_sample_median_centering.parquet"
TOTAL = ROOT / "02_results/model_prediction/20260428_cptac_total_proteome_film_vae_z_direct_residual_tcga_supported_v1/predictions/tcga_supported_predicted_cptac_total_proteome.parquet"
MANIFEST = ROOT / "02_results/public_bulk_phosphoproteome_atlas/20260508_scp682_v4_0_public_bulk_atlas/tables/tcga_supported_prediction_manifest.tsv"
OUT = ROOT / "02_results/model_validation/20260527_cptac_measured_tcga_predicted_site_survival_concordance_v1"

CBIOPORTAL_BASE = "https://www.cbioportal.org/api"
STUDY_BY_PROJECT = {
    "TCGA-HNSC": "hnsc_tcga_pan_can_atlas_2018",
    "TCGA-KIRC": "kirc_tcga_pan_can_atlas_2018",
    "TCGA-KIRP": "kirp_tcga_pan_can_atlas_2018",
    "TCGA-LUAD": "luad_tcga_pan_can_atlas_2018",
    "TCGA-LUSC": "lusc_tcga_pan_can_atlas_2018",
    "TCGA-PAAD": "paad_tcga_pan_can_atlas_2018",
    "TCGA-STAD": "stad_tcga_pan_can_atlas_2018",
    "TCGA-UCEC": "ucec_tcga_pan_can_atlas_2018",
}
CPTAC_TO_PROJECT = {
    "HNSCC": "TCGA-HNSC",
    "CCRCC": "TCGA-KIRC",
    "NON_CCRCC": "TCGA-KIRP",
    "LUAD": "TCGA-LUAD",
    "LUAD_CONFIRM": "TCGA-LUAD",
    "LSCC": "TCGA-LUSC",
    "PDA": "TCGA-PAAD",
    "STAD": "TCGA-STAD",
    "UCEC": "TCGA-UCEC",
    "UCEC_CONFIRM": "TCGA-UCEC",
}


def ensure_dirs() -> None:
    for sub in ["tables", "logs", "predictions"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def parquet_columns(path: Path) -> list[str]:
    return pq.read_schema(path).names


def read_prediction_subset(path: Path, targets: list[str]) -> pd.DataFrame:
    cols = parquet_columns(path)
    present = ["sample_id"] + [t for t in targets if t in cols]
    df = pd.read_parquet(path, columns=present)
    return df.set_index("sample_id")


def as_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("[^0-9eE+.-]", "", regex=True), errors="coerce")


def fetch_cbioportal_patient_clinical(projects: list[str], cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path, sep="\t")
    rows = []
    for project in projects:
        study = STUDY_BY_PROJECT.get(project)
        if study is None:
            continue
        response = requests.get(
            f"{CBIOPORTAL_BASE}/studies/{study}/clinical-data",
            params={"clinicalDataType": "PATIENT", "projection": "SUMMARY"},
            timeout=120,
        )
        response.raise_for_status()
        for item in response.json():
            rows.append(
                {
                    "project": project,
                    "study_id": study,
                    "patient_id": item.get("patientId"),
                    "clinicalAttributeId": item.get("clinicalAttributeId"),
                    "value": item.get("value"),
                }
            )
    long = pd.DataFrame(rows)
    if long.empty:
        raise RuntimeError("cBioPortal clinical table is empty.")
    wide = (
        long.pivot_table(
            index=["project", "study_id", "patient_id"],
            columns="clinicalAttributeId",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    wide.to_csv(cache_path, sep="\t", index=False)
    return wide


def standardize_clinical(clinical: pd.DataFrame) -> pd.DataFrame:
    out = clinical.copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out["os_days"] = as_number(out.get("OS_MONTHS", pd.Series(np.nan, index=out.index))) * 30.4375
    status = out.get("OS_STATUS", pd.Series("", index=out.index)).astype(str).str.upper()
    out["os_event"] = np.where(
        status.str.contains("1:|DECEASED|DEAD"),
        1,
        np.where(status.str.contains("0:|LIVING|ALIVE"), 0, np.nan),
    )
    out["age_at_diagnosis"] = as_number(out.get("AGE", pd.Series(np.nan, index=out.index)))
    return out[["project", "patient_id", "os_days", "os_event", "age_at_diagnosis"]]


def fit_cox(time: np.ndarray, event: np.ndarray, x: np.ndarray, age: np.ndarray | None = None) -> dict:
    ok = np.isfinite(time) & np.isfinite(event) & np.isfinite(x) & (time > 0)
    if age is not None:
        ok = ok & np.isfinite(age)
    n = int(ok.sum())
    events = int(event[ok].sum()) if n else 0
    if n < 10 or events < 3 or np.nanstd(x[ok]) < 1e-9:
        return {"n": n, "events": events, "beta": np.nan, "hr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}
    xx = x[ok].astype(float)
    xx = (xx - xx.mean()) / max(xx.std(ddof=0), 1e-9)
    exog_cols = [xx]
    if age is not None:
        aa = age[ok].astype(float)
        aa = (aa - aa.mean()) / max(aa.std(ddof=0), 1e-9)
        exog_cols.append(aa)
    exog = np.vstack(exog_cols).T
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = PHReg(time[ok].astype(float), exog, status=event[ok].astype(int), ties="efron").fit(disp=0)
        beta = float(res.params[0])
        se = float(res.bse[0])
        return {
            "n": n,
            "events": events,
            "beta": beta,
            "hr": float(np.exp(beta)),
            "ci_low": float(np.exp(beta - 1.96 * se)),
            "ci_high": float(np.exp(beta + 1.96 * se)),
            "p": float(res.pvalues[0]),
        }
    except Exception:
        return {"n": n, "events": events, "beta": np.nan, "hr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}


def project_residual(phos: pd.Series, total: pd.Series) -> pd.Series:
    df = pd.DataFrame({"phos": phos, "total": total}).dropna()
    out = pd.Series(np.nan, index=phos.index, dtype=float)
    if df.shape[0] < 20 or df["total"].std(ddof=0) < 1e-9:
        return out
    slope, intercept = np.polyfit(df["total"].to_numpy(dtype=float), df["phos"].to_numpy(dtype=float), 1)
    out.loc[df.index] = df["phos"] - (intercept + slope * df["total"])
    return out


def add_cox_prefix(rec: dict, prefix: str, cox: dict) -> None:
    rec[f"{prefix}_n"] = cox["n"]
    rec[f"{prefix}_events"] = cox["events"]
    rec[f"{prefix}_beta"] = cox["beta"]
    rec[f"{prefix}_hr_per_sd"] = cox["hr"]
    rec[f"{prefix}_ci95_low"] = cox["ci_low"]
    rec[f"{prefix}_ci95_high"] = cox["ci_high"]
    rec[f"{prefix}_p"] = cox["p"]


def main() -> int:
    ensure_dirs()
    strict = pd.read_csv(STRICT, sep="\t")
    targets = strict["target"].astype(str).drop_duplicates().tolist()
    available_centered = set(parquet_columns(PHOS_CENTERED))
    targets = [t for t in targets if t in available_centered]

    centered = read_prediction_subset(PHOS_CENTERED, targets)
    raw = read_prediction_subset(PHOS_RAW, targets)
    parent_genes = sorted({t.split("|", 1)[0] for t in targets})
    total_cols = parquet_columns(TOTAL)
    total = pd.read_parquet(TOTAL, columns=["sample_id"] + [g for g in parent_genes if g in total_cols]).set_index("sample_id")

    manifest = pd.read_csv(MANIFEST, sep="\t")
    manifest = manifest.loc[manifest["project"].isin(STUDY_BY_PROJECT)].copy()
    if "sample_type" in manifest.columns:
        manifest = manifest.loc[manifest["sample_type"].astype(str).str.contains("Primary", case=False, na=False)].copy()
    manifest = manifest.loc[manifest["sample_id"].isin(centered.index)].copy()

    clinical_raw = fetch_cbioportal_patient_clinical(sorted(manifest["project"].unique()), OUT / "tables/tcga_cbioportal_patient_clinical_wide.tsv")
    clinical = standardize_clinical(clinical_raw)
    clinical.to_csv(OUT / "tables/tcga_cbioportal_patient_clinical_standardized.tsv", sep="\t", index=False)
    meta = manifest.merge(clinical, on=["project", "patient_id"], how="left").set_index("sample_id")
    valid = (meta["os_days"] > 0) & meta["os_event"].isin([0, 1])
    meta = meta.loc[valid].copy()
    centered = centered.loc[centered.index.intersection(meta.index)]
    raw = raw.loc[centered.index]
    meta = meta.loc[centered.index]

    rows = []
    for _, row in strict.iterrows():
        target = str(row["target"])
        cancer = str(row["cancer_label"])
        project = CPTAC_TO_PROJECT.get(cancer)
        rec = row.to_dict()
        rec["tcga_project"] = project
        rec["prediction_model"] = "SCP682_v4_0_tcga_supported_phosphosite"
        if project is None or target not in centered.columns:
            add_cox_prefix(rec, "tcga_pred_site_centered", {"n": 0, "events": 0, "beta": np.nan, "hr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan})
            rows.append(rec)
            continue
        ids = meta.index[meta["project"].eq(project)]
        sub = meta.loc[ids]
        time = sub["os_days"].to_numpy(dtype=float)
        event = sub["os_event"].to_numpy(dtype=float)
        age = sub["age_at_diagnosis"].to_numpy(dtype=float)

        x_centered = centered.loc[ids, target].to_numpy(dtype=float)
        x_raw = raw.loc[ids, target].to_numpy(dtype=float) if target in raw.columns else np.full(len(ids), np.nan)
        centered_cox = fit_cox(time, event, x_centered)
        centered_age = fit_cox(time, event, x_centered, age)
        raw_cox = fit_cox(time, event, x_raw)

        add_cox_prefix(rec, "tcga_pred_site_centered", centered_cox)
        add_cox_prefix(rec, "tcga_pred_site_centered_age_adjusted", centered_age)
        add_cox_prefix(rec, "tcga_pred_site_raw", raw_cox)

        parent = target.split("|", 1)[0]
        if parent in total.columns:
            common_ids = [i for i in ids if i in total.index]
            rel = pd.Series(np.nan, index=ids, dtype=float)
            res = pd.Series(np.nan, index=ids, dtype=float)
            if common_ids:
                rel.loc[common_ids] = raw.loc[common_ids, target].astype(float) - total.loc[common_ids, parent].astype(float)
                res.loc[common_ids] = project_residual(raw.loc[common_ids, target].astype(float), total.loc[common_ids, parent].astype(float))
            rel_cox = fit_cox(time, event, rel.to_numpy(dtype=float))
            residual_cox = fit_cox(time, event, res.to_numpy(dtype=float))
        else:
            rel_cox = {"n": 0, "events": 0, "beta": np.nan, "hr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}
            residual_cox = rel_cox.copy()
        add_cox_prefix(rec, "tcga_pred_site_minus_parent_total", rel_cox)
        add_cox_prefix(rec, "tcga_pred_site_parent_residual", residual_cox)

        rec["direction_concordant_centered"] = bool(np.isfinite(centered_cox["beta"]) and np.sign(float(row["site_beta"])) == np.sign(centered_cox["beta"]))
        rows.append(rec)

    out = pd.DataFrame(rows)
    ok = out["tcga_pred_site_centered_p"].notna()
    out["tcga_pred_site_centered_fdr_within_strict_pool"] = np.nan
    if ok.any():
        out.loc[ok, "tcga_pred_site_centered_fdr_within_strict_pool"] = false_discovery_control(out.loc[ok, "tcga_pred_site_centered_p"].to_numpy(dtype=float), method="bh")

    out["tcga_pred_site_centered_significant"] = (
        (out["tcga_pred_site_centered_n"] >= 80)
        & (out["tcga_pred_site_centered_events"] >= 20)
        & (out["tcga_pred_site_centered_p"] < 0.05)
    )
    out["cptac_measured_and_tcga_predicted_concordant"] = (
        out["strict_site_specific_candidate"].astype(bool)
        & out["tcga_pred_site_centered_significant"]
        & out["direction_concordant_centered"].astype(bool)
    )
    out = out.sort_values(["cptac_measured_and_tcga_predicted_concordant", "tcga_pred_site_centered_p", "site_p"], ascending=[False, True, True])
    out.to_csv(OUT / "tables/cptac_strict_candidates_with_tcga_predicted_survival.tsv", sep="\t", index=False)
    confirmed = out.loc[out["cptac_measured_and_tcga_predicted_concordant"].astype(bool)].copy()
    confirmed.to_csv(OUT / "tables/tcga_confirmed_site_specific_candidates.tsv", sep="\t", index=False)

    summary_by_cancer = (
        out.groupby("cancer_label")
        .agg(
            n_cptac_strict=("target", "size"),
            n_tcga_p_lt_0_05=("tcga_pred_site_centered_significant", "sum"),
            n_direction_concordant_confirmed=("cptac_measured_and_tcga_predicted_concordant", "sum"),
            median_tcga_p=("tcga_pred_site_centered_p", "median"),
        )
        .reset_index()
    )
    summary_by_cancer.to_csv(OUT / "tables/tcga_predicted_survival_summary_by_cancer.tsv", sep="\t", index=False)

    audit = {
        "strict_candidate_file": str(STRICT),
        "phosphosite_centered_prediction": str(PHOS_CENTERED),
        "phosphosite_raw_prediction": str(PHOS_RAW),
        "total_prediction": str(TOTAL),
        "manifest": str(MANIFEST),
        "n_cptac_strict_candidates": int(strict.shape[0]),
        "n_targets_available_in_tcga_prediction": int(len(targets)),
        "n_tcga_survival_samples": int(meta.shape[0]),
        "projects": meta.groupby("project").size().sort_index().to_dict(),
        "events_by_project": meta.groupby("project")["os_event"].sum().astype(int).sort_index().to_dict(),
        "n_confirmed_direction_concordant": int(confirmed.shape[0]),
        "selection_rule": "CPTAC measured strict candidate plus TCGA sample-centered predicted phosphosite Cox p < 0.05, n >= 80, events >= 20, and same beta direction",
    }
    (OUT / "logs/run_summary.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(confirmed[["cancer_label", "target", "site_hr_per_sd", "site_p", "tcga_project", "tcga_pred_site_centered_hr_per_sd", "tcga_pred_site_centered_p", "tcga_pred_site_centered_fdr_within_strict_pool"]].head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
