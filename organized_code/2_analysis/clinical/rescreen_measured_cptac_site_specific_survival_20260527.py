#!/usr/bin/env python3
"""Measured CPTAC phosphosite survival screen with RNA and total-protein controls.

This script is intended to run on the remote Windows workstation where
``D:\\data\\lsy`` contains the full CPTAC multi-task matrices.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg


ROOT = Path(r"D:\data\lsy")
DATA_DIR = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2"
CLINICAL = ROOT / "_codex_inputs/gdc_clinical_os_for_cptac_model_samples.tsv"
OUT_DIR = ROOT / "02_results/model_validation/20260527_measured_cptac_site_specific_survival_rescreen_v1"
TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = OUT_DIR / "logs"

PHOSPHO = DATA_DIR / "phosphosite_gene_site_study_zscore_min20pct_targets.parquet"
RESIDUAL = DATA_DIR / "phosphosite_gene_site_total_residual_targets.parquet"
TOTAL = DATA_DIR / "total_protein_gene_study_zscore_min20pct.parquet"
RNA = DATA_DIR / "rna_log2_tpm_paired.parquet"
MANIFEST = DATA_DIR / "sample_manifest.tsv"
TARGET_MANIFEST = DATA_DIR / "target_manifest_gene_site_locked_v2.tsv"

MIN_N = 30
MIN_EVENTS = 8
SITE_SCORE_P_PREFILTER = 0.02
MAX_EXACT_PER_CANCER = 3000
SITE_EXACT_P_CANDIDATE = 0.01
CONTROL_P_NOT_SIGNIFICANT = 0.05


def bh_fdr(pvals: pd.Series) -> pd.Series:
    x = pd.to_numeric(pvals, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype=float)
    ok = x.notna() & np.isfinite(x)
    vals = x[ok].to_numpy()
    if vals.size == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    q = ranked * vals.size / (np.arange(vals.size) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(q)
    tmp[order] = q
    out.loc[ok] = tmp
    return out


def parse_target(target: str) -> tuple[str, str]:
    if "|" not in str(target):
        return str(target), ""
    gene, site = str(target).split("|", 1)
    return gene, site


def concordance_index(time: np.ndarray, event: np.ndarray, score: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0.0
    n = len(time)
    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if time[i] < time[j]:
                comparable += 1.0
                if score[i] > score[j]:
                    concordant += 1.0
                elif score[i] == score[j]:
                    concordant += 0.5
    if comparable == 0:
        return math.nan
    return concordant / comparable


def cox_exact(time_s: pd.Series, event_s: pd.Series, x_s: pd.Series) -> dict[str, float | int | bool]:
    df = pd.concat({"time": time_s, "event": event_s, "x": x_s}, axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df["time"] > 0) & df["event"].isin([0, 1])].copy()
    n = int(df.shape[0])
    n_events = int(df["event"].sum()) if n else 0
    if n < MIN_N or n_events < MIN_EVENTS or df["x"].std(ddof=0) < 1e-8:
        return {
            "n": n,
            "n_events": n_events,
            "hr_per_sd": math.nan,
            "beta": math.nan,
            "p": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "cindex": math.nan,
            "evaluable": False,
        }
    df["x"] = (df["x"] - df["x"].mean()) / df["x"].std(ddof=0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = PHReg(df["time"].to_numpy(float), df[["x"]].to_numpy(float), status=df["event"].to_numpy(int)).fit(
                disp=False
            )
        beta = float(fit.params[0])
        p = float(fit.pvalues[0])
        ci = fit.conf_int()
        ci_low = float(np.exp(ci[0, 0]))
        ci_high = float(np.exp(ci[0, 1]))
        cidx = concordance_index(df["time"].to_numpy(float), df["event"].to_numpy(int), df["x"].to_numpy(float) * beta)
        return {
            "n": n,
            "n_events": n_events,
            "hr_per_sd": float(np.exp(beta)),
            "beta": beta,
            "p": p,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "cindex": cidx,
            "evaluable": True,
        }
    except Exception:
        return {
            "n": n,
            "n_events": n_events,
            "hr_per_sd": math.nan,
            "beta": math.nan,
            "p": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "cindex": math.nan,
            "evaluable": False,
        }


def cox_score_screen(time_s: pd.Series, event_s: pd.Series, mat: pd.DataFrame) -> pd.DataFrame:
    meta = pd.concat({"time": time_s, "event": event_s}, axis=1)
    meta = meta[(meta["time"] > 0) & meta["event"].isin([0, 1])].dropna()
    xdf = mat.reindex(meta.index)
    if meta.empty or xdf.empty:
        return pd.DataFrame(columns=["target", "score_n", "score_events", "score_hr_per_sd", "score_p"])

    order = np.argsort(-meta["time"].to_numpy(float))
    event = meta["event"].to_numpy(int)[order]
    x = xdf.to_numpy(dtype=np.float32, copy=True)[order, :]
    observed = np.isfinite(x)

    n_obs = observed.sum(axis=0).astype(np.int32)
    event_obs = (observed & (event[:, None] == 1)).sum(axis=0).astype(np.int32)

    mean = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    valid = (n_obs >= MIN_N) & (event_obs >= MIN_EVENTS) & np.isfinite(sd) & (sd > 1e-8)
    if not valid.any():
        return pd.DataFrame(
            {
                "target": mat.columns.astype(str),
                "score_n": n_obs,
                "score_events": event_obs,
                "score_hr_per_sd": np.nan,
                "score_p": np.nan,
            }
        )

    x = (x - mean[None, :]) / sd[None, :]
    x[~observed] = 0.0
    m = observed.astype(np.float32)
    c_count = np.cumsum(m, axis=0)
    c_sum = np.cumsum(x, axis=0)
    c_sum2 = np.cumsum(x * x, axis=0)

    event_idx = np.flatnonzero(event == 1)
    event_m = m[event_idx, :]
    risk_count = c_count[event_idx, :]
    risk_mean = np.divide(c_sum[event_idx, :], risk_count, out=np.zeros_like(c_sum[event_idx, :]), where=risk_count > 0)
    risk_mean2 = np.divide(
        c_sum2[event_idx, :], risk_count, out=np.zeros_like(c_sum2[event_idx, :]), where=risk_count > 0
    )
    risk_var = np.maximum(risk_mean2 - risk_mean * risk_mean, 1e-12)

    u = (event_m * (x[event_idx, :] - risk_mean)).sum(axis=0)
    info = (event_m * risk_var).sum(axis=0)
    z2 = np.divide(u * u, info, out=np.full_like(u, np.nan, dtype=np.float32), where=info > 1e-12)
    p = stats.chi2.sf(z2.astype(float), 1)
    beta = np.divide(u, info, out=np.full_like(u, np.nan, dtype=np.float32), where=info > 1e-12)
    p[~valid] = np.nan
    beta[~valid] = np.nan

    return pd.DataFrame(
        {
            "target": mat.columns.astype(str),
            "score_n": n_obs,
            "score_events": event_obs,
            "score_hr_per_sd": np.exp(beta.astype(float)),
            "score_p": p,
        }
    )


def load_inputs() -> dict[str, pd.DataFrame]:
    missing = [p for p in [PHOSPHO, RESIDUAL, TOTAL, RNA, MANIFEST, CLINICAL] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + "; ".join(str(p) for p in missing))

    phos = pd.read_parquet(PHOSPHO)
    residual = pd.read_parquet(RESIDUAL)
    total = pd.read_parquet(TOTAL)
    rna = pd.read_parquet(RNA)
    manifest = pd.read_csv(MANIFEST, sep="\t").set_index("sample_id")
    clinical = pd.read_csv(CLINICAL, sep="\t")
    return {
        "phos": phos,
        "residual": residual,
        "total": total,
        "rna": rna,
        "manifest": manifest,
        "clinical": clinical,
    }


def make_meta(manifest: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    clinical = clinical.copy()
    clinical["case_submitter_id"] = clinical["case_submitter_id"].astype(str)
    meta = manifest.reset_index().merge(clinical, on="case_submitter_id", how="left").set_index("sample_id")
    meta["os_days"] = pd.to_numeric(meta["os_days"], errors="coerce")
    meta["os_event"] = pd.to_numeric(meta["os_event"], errors="coerce")
    meta["age_at_diagnosis"] = pd.to_numeric(meta.get("age_at_diagnosis"), errors="coerce")
    return meta


def add_target_annotations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["target"].map(parse_target)
    out["gene"] = [x[0] for x in parsed]
    out["site"] = [x[1] for x in parsed]
    if TARGET_MANIFEST.exists():
        tm = pd.read_csv(TARGET_MANIFEST, sep="\t", low_memory=False)
        key_cols = [c for c in ["gene_site", "parent_gene", "residue_site", "n_refseq_features", "n_pdc_studies"] if c in tm]
        if "gene_site" in tm.columns:
            tm = tm[key_cols].drop_duplicates("gene_site")
            out = out.merge(tm, left_on="target", right_on="gene_site", how="left")
    return out


def run_screen() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    data = load_inputs()
    phos = data["phos"]
    residual = data["residual"]
    total = data["total"]
    rna = data["rna"]
    meta = make_meta(data["manifest"], data["clinical"])

    common = meta.index.intersection(phos.index).intersection(total.index).intersection(rna.index)
    meta = meta.loc[common].copy()
    phos = phos.loc[common]
    residual = residual.reindex(index=common, columns=phos.columns)
    total = total.loc[common]
    rna = rna.loc[common]

    audit = [
        {"item": "phosphosite_matrix", "path": str(PHOSPHO), "rows": phos.shape[0], "columns": phos.shape[1]},
        {"item": "residual_matrix", "path": str(RESIDUAL), "rows": residual.shape[0], "columns": residual.shape[1]},
        {"item": "total_protein_matrix", "path": str(TOTAL), "rows": total.shape[0], "columns": total.shape[1]},
        {"item": "rna_matrix", "path": str(RNA), "rows": rna.shape[0], "columns": rna.shape[1]},
        {"item": "clinical", "path": str(CLINICAL), "rows": data["clinical"].shape[0], "columns": data["clinical"].shape[1]},
        {"item": "analysis_samples", "path": "intersection", "rows": meta.shape[0], "columns": meta.shape[1]},
    ]
    pd.DataFrame(audit).to_csv(TABLE_DIR / "input_data_audit.tsv", sep="\t", index=False)

    cancer_counts = (
        meta.groupby("cancer_label")
        .agg(n_samples=("os_days", "size"), n_with_os=("os_days", lambda x: int(x.notna().sum())), n_events=("os_event", "sum"))
        .reset_index()
    )
    cancer_counts["n_events"] = cancer_counts["n_events"].astype(int)
    cancer_counts.to_csv(TABLE_DIR / "cancer_os_sample_counts.tsv", sep="\t", index=False)

    all_rows: list[pd.DataFrame] = []
    exact_rows: list[dict[str, object]] = []
    control_cache: dict[tuple[str, str], dict[str, object]] = {}
    summary_rows: list[dict[str, object]] = []

    eligible_cancers = cancer_counts[
        (cancer_counts["n_with_os"] >= MIN_N) & (cancer_counts["n_events"] >= MIN_EVENTS)
    ]["cancer_label"].astype(str)

    for cancer in eligible_cancers:
        ids = meta.index[meta["cancer_label"].astype(str).eq(cancer)]
        ids = ids[meta.loc[ids, "os_days"].notna() & meta.loc[ids, "os_event"].notna()]
        time = meta.loc[ids, "os_days"]
        event = meta.loc[ids, "os_event"]

        score = cox_score_screen(time, event, phos.loc[ids])
        score["cancer_label"] = cancer
        score = add_target_annotations(score)
        score = score.sort_values("score_p", na_position="last")
        all_rows.append(score.head(MAX_EXACT_PER_CANCER).copy())

        exact_targets = score.loc[score["score_p"] < SITE_SCORE_P_PREFILTER, "target"].head(MAX_EXACT_PER_CANCER).tolist()
        for target in exact_targets:
            gene, site = parse_target(target)
            site_fit = cox_exact(time, event, phos.loc[ids, target])
            residual_fit = (
                cox_exact(time, event, residual.loc[ids, target])
                if target in residual.columns
                else {"n": 0, "n_events": 0, "hr_per_sd": math.nan, "beta": math.nan, "p": math.nan, "ci95_low": math.nan, "ci95_high": math.nan, "cindex": math.nan, "evaluable": False}
            )

            cache_key = (cancer, gene)
            if cache_key not in control_cache:
                rna_fit = (
                    cox_exact(time, event, rna.loc[ids, gene])
                    if gene in rna.columns
                    else {"n": 0, "n_events": 0, "hr_per_sd": math.nan, "beta": math.nan, "p": math.nan, "ci95_low": math.nan, "ci95_high": math.nan, "cindex": math.nan, "evaluable": False}
                )
                total_fit = (
                    cox_exact(time, event, total.loc[ids, gene])
                    if gene in total.columns
                    else {"n": 0, "n_events": 0, "hr_per_sd": math.nan, "beta": math.nan, "p": math.nan, "ci95_low": math.nan, "ci95_high": math.nan, "cindex": math.nan, "evaluable": False}
                )
                control_cache[cache_key] = {"rna": rna_fit, "total": total_fit}
            rna_fit = control_cache[cache_key]["rna"]
            total_fit = control_cache[cache_key]["total"]

            exact_rows.append(
                {
                    "cancer_label": cancer,
                    "target": target,
                    "gene": gene,
                    "site": site,
                    "site_n": site_fit["n"],
                    "site_events": site_fit["n_events"],
                    "site_hr_per_sd": site_fit["hr_per_sd"],
                    "site_beta": site_fit["beta"],
                    "site_p": site_fit["p"],
                    "site_ci95_low": site_fit["ci95_low"],
                    "site_ci95_high": site_fit["ci95_high"],
                    "site_cindex": site_fit["cindex"],
                    "residual_n": residual_fit["n"],
                    "residual_events": residual_fit["n_events"],
                    "residual_hr_per_sd": residual_fit["hr_per_sd"],
                    "residual_p": residual_fit["p"],
                    "rna_n": rna_fit["n"],
                    "rna_events": rna_fit["n_events"],
                    "rna_hr_per_sd": rna_fit["hr_per_sd"],
                    "rna_p": rna_fit["p"],
                    "rna_evaluable": rna_fit["evaluable"],
                    "total_n": total_fit["n"],
                    "total_events": total_fit["n_events"],
                    "total_hr_per_sd": total_fit["hr_per_sd"],
                    "total_p": total_fit["p"],
                    "total_evaluable": total_fit["evaluable"],
                }
            )

        summary_rows.append(
            {
                "cancer_label": cancer,
                "os_samples": int(len(ids)),
                "os_events": int(event.sum()),
                "score_tested_sites": int(score["score_p"].notna().sum()),
                "score_p_lt_0_05": int((score["score_p"] < 0.05).sum()),
                "score_p_lt_0_01": int((score["score_p"] < 0.01).sum()),
                "exact_targets_attempted": int(len(exact_targets)),
            }
        )
        print(json.dumps(summary_rows[-1], ensure_ascii=False), flush=True)

    score_top = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    score_top["score_fdr_within_cancer"] = score_top.groupby("cancer_label")["score_p"].transform(bh_fdr)
    score_top.to_csv(TABLE_DIR / "measured_phosphosite_score_screen_top_by_cancer.tsv", sep="\t", index=False)

    exact = pd.DataFrame(exact_rows)
    if not exact.empty:
        exact["site_fdr_within_exact_pool"] = exact.groupby("cancer_label")["site_p"].transform(bh_fdr)
        exact["site_significant"] = exact["site_p"] < SITE_EXACT_P_CANDIDATE
        exact["rna_not_significant"] = exact["rna_evaluable"] & (exact["rna_p"] >= CONTROL_P_NOT_SIGNIFICANT)
        exact["total_not_significant"] = exact["total_evaluable"] & (exact["total_p"] >= CONTROL_P_NOT_SIGNIFICANT)
        exact["strict_site_specific_candidate"] = (
            exact["site_significant"] & exact["rna_not_significant"] & exact["total_not_significant"]
        )
        exact["residual_support"] = exact["residual_p"] < CONTROL_P_NOT_SIGNIFICANT
        exact = exact.sort_values(
            ["strict_site_specific_candidate", "site_p", "residual_support"], ascending=[False, True, False]
        )
        exact.to_csv(TABLE_DIR / "measured_phosphosite_exact_cox_with_rna_total_controls.tsv", sep="\t", index=False)

        strict = exact[exact["strict_site_specific_candidate"]].copy()
        strict = strict.sort_values(["residual_support", "site_p"], ascending=[False, True])
        strict.to_csv(TABLE_DIR / "strict_site_specific_candidates_measured_site_rna_total_null.tsv", sep="\t", index=False)

        priority = strict.copy()
        priority["direction"] = np.where(priority["site_hr_per_sd"] > 1, "higher_site_worse_os", "higher_site_better_os")
        priority["selection_note"] = (
            "measured phosphosite Cox p<0.01; matched gene RNA p>=0.05; matched measured total protein p>=0.05"
        )
        priority.head(300).to_csv(TABLE_DIR / "priority_pool_for_antibody_novelty_biology_review.tsv", sep="\t", index=False)

    pd.DataFrame(summary_rows).to_csv(TABLE_DIR / "screen_summary_by_cancer.tsv", sep="\t", index=False)
    with open(LOG_DIR / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "root": str(ROOT),
                "data_dir": str(DATA_DIR),
                "min_n": MIN_N,
                "min_events": MIN_EVENTS,
                "site_score_p_prefilter": SITE_SCORE_P_PREFILTER,
                "site_exact_p_candidate": SITE_EXACT_P_CANDIDATE,
                "control_p_not_significant": CONTROL_P_NOT_SIGNIFICANT,
                "eligible_cancers": eligible_cancers.tolist(),
                "n_exact_rows": int(len(exact_rows)),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    run_screen()
