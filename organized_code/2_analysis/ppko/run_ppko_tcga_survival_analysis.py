from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "02_results"
    / "clinical_validation"
    / "20260531_tcga_tcpa_ppko_patient_response_v1_targeted_expanded"
)
DEFAULT_PRED = DEFAULT_RUN_DIR / "tables" / "v10b_300_patient_predictions.tsv"
DEFAULT_SURV = (
    PROJECT_ROOT
    / "01_data"
    / "tcga_survival"
    / "xena_toil"
    / "TCGA_survival_data.tsv"
)
DEFAULT_OUT = DEFAULT_RUN_DIR / "survival_v10b_300"


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["patient12"] = df["patient12"].astype(str).str[:12]
    df["Cancer"] = df["Cancer"].astype(str)
    for col in [
        "ppko_target_prior_abs_mean",
        "ppko_abs_delta_mean",
        "ppko_abs_delta_top200_mean",
        "ppko_observed_site_abs_mean",
        "ppko_signed_delta_mean",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def join_unique(values: pd.Series) -> str:
    vals = [str(v) for v in values.dropna().unique() if str(v) not in {"", "nan"}]
    return ";".join(sorted(vals))


def make_patient_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient, sub in df.groupby("patient12", sort=True):
        scores = pd.to_numeric(sub["ppko_target_prior_abs_mean"], errors="coerce")
        rows.append(
            {
                "patient12": patient,
                "Cancer": join_unique(sub["Cancer"]),
                "n_drug_records": int(len(sub)),
                "drug_name": join_unique(sub["drug_name"]),
                "target_genes": join_unique(sub["target_genes"]),
                "response_binary_any": int(pd.to_numeric(sub["response_binary"], errors="coerce").fillna(0).max()),
                "response_binary_mean": float(pd.to_numeric(sub["response_binary"], errors="coerce").mean()),
                "ppko_score_max": float(scores.max()),
                "ppko_score_mean": float(scores.mean()),
                "ppko_score_median": float(scores.median()),
            }
        )
    return pd.DataFrame(rows)


def read_survival(path: Path) -> pd.DataFrame:
    surv = pd.read_csv(path, sep="\t")
    surv["sample"] = surv["sample"].astype(str)
    surv["patient12"] = surv["sample"].str[:12]
    surv["sample_type"] = surv["sample"].str[13:15]
    numeric_cols = ["OS", "OS.time", "DSS", "DSS.time", "DFI", "DFI.time", "PFI", "PFI.time"]
    for col in numeric_cols:
        surv[col] = pd.to_numeric(surv[col], errors="coerce")
    surv["_primary_rank"] = np.where(surv["sample_type"].eq("01"), 0, 1)
    surv["_complete"] = surv[numeric_cols].notna().sum(axis=1)
    surv = surv.sort_values(["patient12", "_primary_rank", "_complete"], ascending=[True, True, False])
    surv = surv.drop_duplicates("patient12", keep="first").drop(columns=["_primary_rank", "_complete"])
    return surv


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def cox_one(sub: pd.DataFrame, time_col: str, event_col: str, covariate: str) -> dict:
    cols = [time_col, event_col, covariate]
    fit_df = sub[cols].dropna().copy()
    if fit_df[event_col].sum() < 3 or fit_df[covariate].nunique() < 2:
        return {"hr": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "p": np.nan, "status": "too_few_events_or_no_variation"}
    try:
        cph = CoxPHFitter()
        cph.fit(fit_df, duration_col=time_col, event_col=event_col)
        s = cph.summary.loc[covariate]
        return {
            "hr": safe_float(s["exp(coef)"]),
            "ci95_low": safe_float(s["exp(coef) lower 95%"]),
            "ci95_high": safe_float(s["exp(coef) upper 95%"]),
            "p": safe_float(s["p"]),
            "status": "ok",
        }
    except Exception as exc:
        return {"hr": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "p": np.nan, "status": f"failed: {exc}"}


def km_median(kmf: KaplanMeierFitter) -> float:
    val = kmf.median_survival_time_
    try:
        return float(val)
    except Exception:
        return np.nan


def plot_km(sub: pd.DataFrame, time_col: str, event_col: str, cohort: str, score_name: str, out_dir: Path) -> pd.DataFrame:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    curve_rows = []
    plt.figure(figsize=(4.7, 3.6))
    colors = {"Low": "#92B1D9", "High": "#F6C8B6"}
    for group in ["Low", "High"]:
        g = sub[sub["ppko_group"].eq(group)]
        kmf = KaplanMeierFitter(label=f"{group} PPKO")
        kmf.fit(g[time_col], event_observed=g[event_col])
        kmf.plot_survival_function(ci_show=True, color=colors[group], linewidth=2)
        sf = kmf.survival_function_.reset_index()
        sf.columns = ["time", "survival"]
        sf["group"] = group
        sf["cohort"] = cohort
        sf["endpoint"] = event_col.replace(".time", "")
        sf["score_summary"] = score_name
        curve_rows.append(sf)
    endpoint = event_col
    plt.xlabel("Days")
    plt.ylabel(f"{endpoint} probability")
    plt.title(f"{cohort} {endpoint}")
    plt.ylim(0, 1.03)
    plt.grid(axis="y", color="#D4D4D4", alpha=0.45, linewidth=0.7)
    plt.tight_layout()
    stem = f"km_{cohort}_{endpoint}_{score_name}".replace(" ", "_").replace("/", "_").lower()
    plt.savefig(fig_dir / f"{stem}.png", dpi=300)
    plt.savefig(fig_dir / f"{stem}.pdf")
    plt.close()
    return pd.concat(curve_rows, ignore_index=True)


def analyze_endpoint(df: pd.DataFrame, cohort: str, endpoint: str, score_col: str, score_name: str, out_dir: Path):
    time_col = f"{endpoint}.time"
    event_col = endpoint
    sub = df.dropna(subset=[time_col, event_col, score_col]).copy()
    sub = sub[(sub[time_col] > 0) & (sub[event_col].isin([0, 1]))]
    if sub.empty:
        return None, None, None, None
    cutoff = sub[score_col].median()
    sub["ppko_high"] = (sub[score_col] >= cutoff).astype(int)
    sub["ppko_group"] = np.where(sub["ppko_high"].eq(1), "High", "Low")
    sub["ppko_score_z"] = (sub[score_col] - sub[score_col].mean()) / sub[score_col].std(ddof=0)

    low = sub[sub["ppko_group"].eq("Low")]
    high = sub[sub["ppko_group"].eq("High")]
    lr = logrank_test(
        high[time_col],
        low[time_col],
        event_observed_A=high[event_col],
        event_observed_B=low[event_col],
    )

    kmf_low = KaplanMeierFitter().fit(low[time_col], event_observed=low[event_col])
    kmf_high = KaplanMeierFitter().fit(high[time_col], event_observed=high[event_col])

    cox_high = cox_one(sub, time_col, event_col, "ppko_high")
    cox_cont = cox_one(sub, time_col, event_col, "ppko_score_z")

    logrank_row = {
        "cohort": cohort,
        "endpoint": endpoint,
        "score_summary": score_name,
        "score_col": score_col,
        "n_patients": int(len(sub)),
        "n_events": int(sub[event_col].sum()),
        "score_cutoff_median": float(cutoff),
        "n_low": int(len(low)),
        "events_low": int(low[event_col].sum()),
        "n_high": int(len(high)),
        "events_high": int(high[event_col].sum()),
        "median_time_low": km_median(kmf_low),
        "median_time_high": km_median(kmf_high),
        "logrank_p": safe_float(lr.p_value),
    }
    cox_rows = [
        {
            "cohort": cohort,
            "endpoint": endpoint,
            "score_summary": score_name,
            "score_col": score_col,
            "model": "high_vs_low",
            "n_patients": int(len(sub)),
            "n_events": int(sub[event_col].sum()),
            **cox_high,
        },
        {
            "cohort": cohort,
            "endpoint": endpoint,
            "score_summary": score_name,
            "score_col": score_col,
            "model": "continuous_per_sd",
            "n_patients": int(len(sub)),
            "n_events": int(sub[event_col].sum()),
            **cox_cont,
        },
    ]
    curve = plot_km(sub, time_col, event_col, cohort, score_name, out_dir)
    sub_export = sub[
        [
            "patient12",
            "Cancer",
            "n_drug_records",
            "drug_name",
            "target_genes",
            "response_binary_any",
            score_col,
            "ppko_group",
            endpoint,
            time_col,
        ]
    ].copy()
    sub_export["cohort"] = cohort
    sub_export["endpoint"] = endpoint
    sub_export["score_summary"] = score_name
    return sub_export, logrank_row, cox_rows, curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--survival", type=Path, default=DEFAULT_SURV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables").mkdir(exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)

    pred = read_predictions(args.pred)
    patient_scores = make_patient_scores(pred)
    surv = read_survival(args.survival)
    merged = patient_scores.merge(surv, on="patient12", how="left", validate="one_to_one")

    patient_scores.to_csv(args.out / "tables" / "v10b_300_patient_level_ppko_scores.tsv", sep="\t", index=False)
    merged.to_csv(args.out / "tables" / "v10b_300_patient_level_survival_joined.tsv", sep="\t", index=False)

    cohorts = {
        "all": merged.copy(),
        "brca_only": merged[merged["Cancer"].str.contains("BRCA", na=False)].copy(),
        "no_brca": merged[~merged["Cancer"].str.contains("BRCA", na=False)].copy(),
    }
    score_defs = {
        "score_max": "ppko_score_max",
        "score_mean": "ppko_score_mean",
    }
    endpoints = ["DFI", "PFI"]

    endpoint_tables = []
    logrank_rows = []
    cox_rows = []
    curve_tables = []
    for cohort_name, cohort_df in cohorts.items():
        for score_name, score_col in score_defs.items():
            for endpoint in endpoints:
                sub_export, logrank_row, cox_endpoint_rows, curve = analyze_endpoint(
                    cohort_df, cohort_name, endpoint, score_col, score_name, args.out
                )
                if sub_export is None:
                    continue
                endpoint_tables.append(sub_export)
                logrank_rows.append(logrank_row)
                cox_rows.extend(cox_endpoint_rows)
                curve_tables.append(curve)

    endpoint_df = pd.concat(endpoint_tables, ignore_index=True) if endpoint_tables else pd.DataFrame()
    logrank_df = pd.DataFrame(logrank_rows)
    cox_df = pd.DataFrame(cox_rows)
    curve_df = pd.concat(curve_tables, ignore_index=True) if curve_tables else pd.DataFrame()

    endpoint_df.to_csv(args.out / "tables" / "v10b_300_survival_endpoint_patient_rows.tsv", sep="\t", index=False)
    logrank_df.to_csv(args.out / "tables" / "v10b_300_km_logrank_summary.tsv", sep="\t", index=False)
    cox_df.to_csv(args.out / "tables" / "v10b_300_cox_summary.tsv", sep="\t", index=False)
    curve_df.to_csv(args.out / "tables" / "v10b_300_km_curve_points.tsv", sep="\t", index=False)

    if not endpoint_df.empty:
        rows = []
        for keys, sub in endpoint_df.groupby(["cohort", "endpoint", "score_summary", "Cancer", "ppko_group"], dropna=False):
            cohort, endpoint, score_summary, cancer, group = keys
            rows.append(
                {
                    "cohort": cohort,
                    "endpoint": endpoint,
                    "score_summary": score_summary,
                    "Cancer": cancer,
                    "ppko_group": group,
                    "n_patients": int(sub["patient12"].nunique()),
                    "n_events": int(pd.to_numeric(sub[endpoint], errors="coerce").sum()),
                    "median_time": float(pd.to_numeric(sub[f"{endpoint}.time"], errors="coerce").median()),
                }
            )
        pd.DataFrame(rows).to_csv(
            args.out / "tables" / "v10b_300_survival_cancer_group_counts.tsv", sep="\t", index=False
        )

    audit = {
        "prediction_table": str(args.pred),
        "survival_table": str(args.survival),
        "output_dir": str(args.out),
        "drug_records": int(len(pred)),
        "unique_patients": int(patient_scores["patient12"].nunique()),
        "patients_with_survival_match": int(merged["sample"].notna().sum()),
        "endpoints": endpoints,
        "score_primary": "ppko_score_max",
        "score_sensitivity": "ppko_score_mean",
        "patient_deduplication": "one row per patient; max or mean ppko_target_prior_abs_mean across patient drug records",
    }
    (args.out / "survival_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(logrank_df.to_string(index=False))
    print(cox_df.to_string(index=False))


if __name__ == "__main__":
    main()
