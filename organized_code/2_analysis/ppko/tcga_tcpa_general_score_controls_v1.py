from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
PPKO_OUT = ROOT / "02_results" / "clinical_validation" / "20260527_tcga_tcpa_ppko_patient_response_v1"
PAN_RPPA = ROOT / "01_data" / "bulk_external" / "tcpa_rppa500_20260527" / "extracted" / "TCPA_TCGA_RPPA500.tsv"
OUT = ROOT / "02_results" / "clinical_validation" / "20260527_tcga_tcpa_ppko_patient_response_v1"

TARGET_TOTAL_ALIASES = {
    "EGFR": ["EGFR"],
    "ERBB2": ["HER2"],
    "MAP2K1": ["MEK1"],
    "MAP2K2": ["MEK2"],
    "BRAF": ["BRAF"],
    "RAF1": ["CRAF"],
    "KDR": [],
    "FLT1": [],
    "FLT3": [],
    "FLT4": [],
    "KIT": ["CKIT"],
    "PDGFRA": [],
    "PDGFRB": ["PDGFRB"],
    "MET": ["CMET"],
    "AXL": ["AXL"],
    "RET": [],
    "ABL1": ["CABL"],
    "SRC": ["SRC"],
    "LCK": ["LCK"],
    "YES1": [],
    "MTOR": ["MTOR"],
}


def auc_from_scores(y_true, scores):
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg)
        wins += 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


def score_stats(df, score_col, rng, n_boot=2000, n_perm=2000):
    y = df["response_binary"].to_numpy(float)
    score = df[score_col].to_numpy(float)
    auc = auc_from_scores(y, score)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        val = auc_from_scores(y[idx], score[idx])
        if np.isfinite(val):
            boot.append(val)
    perm = []
    for _ in range(n_perm):
        val = auc_from_scores(rng.permutation(y), score)
        if np.isfinite(val):
            perm.append(val)
    return {
        "score": score_col,
        "n": int(len(df)),
        "n_responder": int(np.sum(y == 1)),
        "n_non_responder": int(np.sum(y == 0)),
        "auc": auc,
        "bootstrap_ci_low": float(np.quantile(boot, 0.025)) if boot else np.nan,
        "bootstrap_ci_high": float(np.quantile(boot, 0.975)) if boot else np.nan,
        "permutation_p_right": float((np.sum(np.asarray(perm) >= auc) + 1) / (len(perm) + 1)) if perm else np.nan,
        "mean_responder": float(np.nanmean(df.loc[df["response_binary"].eq(1), score_col])),
        "mean_non_responder": float(np.nanmean(df.loc[df["response_binary"].eq(0), score_col])),
    }


def target_total_score(row, rppa_row):
    genes = str(row.get("target_genes", "")).split(";")
    cols = []
    for gene in genes:
        gene = gene.strip().upper()
        cols.extend([c for c in TARGET_TOTAL_ALIASES.get(gene, []) if c in rppa_row.index])
    cols = sorted(set(cols))
    if not cols:
        return np.nan
    vals = pd.to_numeric(rppa_row[cols], errors="coerce").to_numpy(float)
    return float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan


def main():
    rng = np.random.default_rng(20260527)
    rppa = pd.read_csv(PAN_RPPA, sep="\t").drop_duplicates("sample_id").set_index("sample_id")
    pred = pd.read_csv(PPKO_OUT / "tables" / "all_model_patient_predictions.tsv", sep="\t")
    base = pred[pred["model_name"].eq("v10b_300")].copy().reset_index(drop=True)

    phospho_cols = [c for c in rppa.columns if re.search(r"P[STY]\d", str(c).upper())]
    mapped_marker_cols = [c for c in [
        "EGFRPY1068", "HER2PY1248", "MAPKPT202Y204", "MEK1PS217S221", "AKTPS473", "AKTPT308",
        "MTORPS2448", "S6PS235S236", "S6PS240S244", "P70S6KPT389", "SRCPY416", "SRCPY527",
        "CABLPY412", "SHCPY317", "BRAFPS445", "P38PT180Y182", "SHP2PY542", "STAT3PY705",
        "PDK1PS241", "PRAS40PT246", "GSK3ALPHABETAPS21S9", "GSK3PS9", "TUBERINPT1462",
        "YAPPS127", "YB1PS102", "EPHA2PS897", "EPHA2PY588", "RBPS807S811", "CJUNPS73",
        "HSP27PS82", "EIF4EPS209", "X4EBP1PS65", "X4EBP1PT37T46", "X4EBP1PT70"
    ] if c in rppa.columns]

    rows = []
    for _, row in base.iterrows():
        r = rppa.loc[row["sample_id"]]
        rec = row.to_dict()
        phospho_vals = pd.to_numeric(r[phospho_cols], errors="coerce").to_numpy(float)
        mapped_vals = pd.to_numeric(r[mapped_marker_cols], errors="coerce").to_numpy(float)
        rec["control_global_phospho_mean"] = float(np.nanmean(phospho_vals))
        rec["control_global_phospho_abs_mean"] = float(np.nanmean(np.abs(phospho_vals)))
        rec["control_mapped_marker_mean"] = float(np.nanmean(mapped_vals))
        rec["control_mapped_marker_abs_mean"] = float(np.nanmean(np.abs(mapped_vals)))
        rec["control_target_total_mean"] = target_total_score(row, r)
        rec["control_hand_pathway_score"] = float(row["phospho_score"]) if "phospho_score" in row and np.isfinite(row["phospho_score"]) else np.nan
        rec["control_observed_marker_count"] = int(np.sum(np.isfinite(mapped_vals)))
        rows.append(rec)

    controls = pd.DataFrame(rows)
    controls.to_csv(OUT / "tables" / "v10b_general_score_controls.tsv", sep="\t", index=False)

    score_cols = [
        "ppko_target_prior_abs_mean",
        "ppko_abs_delta_top200_mean",
        "ppko_observed_site_abs_mean",
        "control_hand_pathway_score",
        "control_global_phospho_mean",
        "control_global_phospho_abs_mean",
        "control_mapped_marker_mean",
        "control_mapped_marker_abs_mean",
        "control_target_total_mean",
        "control_observed_marker_count",
    ]
    stats = [score_stats(controls, c, rng) for c in score_cols]

    random_rows = []
    y = controls["response_binary"].to_numpy(float)
    for k in [3, 5, 10, 20]:
        aucs = []
        for rep in range(1000):
            chosen = rng.choice(phospho_cols, size=min(k, len(phospho_cols)), replace=False)
            scores = []
            for sample_id in controls["sample_id"]:
                vals = pd.to_numeric(rppa.loc[sample_id, chosen], errors="coerce").to_numpy(float)
                scores.append(float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan)
            aucs.append(auc_from_scores(y, np.asarray(scores)))
        aucs = np.asarray([x for x in aucs if np.isfinite(x)])
        random_rows.append({
            "score": f"random_{k}_phospho_markers_mean",
            "n_repeats": int(len(aucs)),
            "auc_mean": float(np.mean(aucs)),
            "auc_ci_low": float(np.quantile(aucs, 0.025)),
            "auc_ci_high": float(np.quantile(aucs, 0.975)),
            "auc_ge_v10b_target_prior_fraction": float(np.mean(aucs >= controls["ppko_target_prior_abs_mean"].pipe(lambda s: auc_from_scores(y, s.to_numpy(float))))),
        })

    stats_df = pd.DataFrame(stats)
    random_df = pd.DataFrame(random_rows)
    stats_df.to_csv(OUT / "tables" / "v10b_general_score_control_auc.tsv", sep="\t", index=False)
    random_df.to_csv(OUT / "tables" / "v10b_random_marker_control_auc.tsv", sep="\t", index=False)

    report = {
        "n_rows": int(len(controls)),
        "n_responder": int(controls["response_binary"].eq(1).sum()),
        "n_non_responder": int(controls["response_binary"].eq(0).sum()),
        "n_phospho_columns_regex": int(len(phospho_cols)),
        "n_mapped_marker_columns": int(len(mapped_marker_cols)),
        "tables": [
            str(OUT / "tables" / "v10b_general_score_controls.tsv"),
            str(OUT / "tables" / "v10b_general_score_control_auc.tsv"),
            str(OUT / "tables" / "v10b_random_marker_control_auc.tsv"),
        ],
    }
    (OUT / "reports" / "general_score_control_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(stats_df.to_string(index=False))
    print(random_df.to_string(index=False))


if __name__ == "__main__":
    main()
