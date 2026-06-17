#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path("E:/data/gongke/TCGA-TCPA")
OUT = ROOT / "02_results/model_validation/20260530_fig5_rps6_ps6_panel_v1"
TAB = OUT / "tables"
FIG = OUT / "figures"
REPORT = OUT / "reports"
SAMPLE = ROOT / "02_results/model_validation/20260530_fig5_exact_site_anchor_search_v1/tables/tcga_kirc_rps6_s235_s236_sample_table.tsv.gz"
GRADE = TAB / "kirc_clinical_grade_stage.tsv"
PRED = TAB / "tcga_full_predicted_rps6_mtor_axis_sites.tsv"

BLUE = "#92B1D9"
WARM = "#F6C8B6"
GREY = "#D4D4D4"
INK = "#222222"


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return (x - x.mean()) / x.std(ddof=0)


def p_text(p: float) -> str:
    if not np.isfinite(p):
        return "p=NA"
    if p < 1e-4:
        return f"p={p:.1e}"
    if p < 0.001:
        return f"p={p:.2e}"
    return f"p={p:.3f}"


def clean_grade(x: object) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if s in {"", "GX", "G", "UNKNOWN", "NOT REPORTED", "NOT EVALUATED"}:
        return None
    if "G1" in s or s == "1" or "G2" in s or s == "2":
        return "G1/G2"
    if "G3" in s or s == "3" or "G4" in s or s == "4":
        return "G3/G4"
    return None


def half_violin(ax, values, xpos, color, width=0.16):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return
    kde = stats.gaussian_kde(values)
    yr = values.max() - values.min()
    y = np.linspace(values.min() - 0.08 * yr, values.max() + 0.08 * yr, 220)
    dens = kde(y)
    dens = dens / dens.max() * width
    ax.fill_betweenx(y, xpos, xpos - dens, color=color, alpha=0.34, lw=0)
    ax.plot(xpos - dens, y, color=color, lw=1.0)


def raincloud(ax, df, group_col, order, colors, title, ycol="predicted_rps6_s235_s236", show_ylabel=None):
    rng = np.random.default_rng(20260531)
    vals_list = []
    for i, (g, color) in enumerate(zip(order, colors)):
        vals = df.loc[df[group_col] == g, ycol].dropna().astype(float).values
        vals_list.append(vals)
        half_violin(ax, vals, i, color)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        lo = vals[vals >= q1 - 1.5 * (q3 - q1)].min()
        hi = vals[vals <= q3 + 1.5 * (q3 - q1)].max()
        ax.add_patch(plt.Rectangle((i - 0.050, q1), 0.10, q3 - q1, facecolor="white", edgecolor=INK, lw=0.8, zorder=5))
        ax.plot([i - 0.050, i + 0.050], [med, med], color=INK, lw=1.1, zorder=6)
        ax.plot([i, i], [lo, q1], color=INK, lw=0.75, zorder=5)
        ax.plot([i, i], [q3, hi], color=INK, lw=0.75, zorder=5)
        x = i + 0.070 + rng.normal(0, 0.017, len(vals))
        ax.scatter(x, vals, s=10 if len(vals) > 80 else 16, color=color, edgecolor="white", linewidth=0.22, alpha=0.72, zorder=4)
        ax.text(i, -0.16, f"n={len(vals)}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.4, color="#666666")
    p = stats.mannwhitneyu(vals_list[0], vals_list[1], alternative="two-sided").pvalue
    shift = float(np.median(vals_list[1]) - np.median(vals_list[0]))
    ax.text(0.98, 0.96, f"shift={shift:.2f}; {p_text(p)}", transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.6))
    ax.set_title(title, loc="left", fontsize=9.4, fontweight="bold")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=7.7)
    ax.grid(axis="y", color="#EFEFEF", lw=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.5)
    if show_ylabel:
        ax.set_ylabel(show_ylabel, fontsize=8.8)
    return {"panel": title, "group": group_col, "order": order, "n": [int(len(v)) for v in vals_list], "median_shift_second_minus_first": shift, "p": float(p)}


def ridge(ax, values, y, color, label):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    kde = stats.gaussian_kde(values)
    xs = np.linspace(-0.75, 0.85, 300)
    dens = kde(xs)
    dens = dens / dens.max() * 0.58
    ax.fill_between(xs, y, y + dens, color=color, alpha=0.36, lw=0)
    ax.plot(xs, y + dens, color=color, lw=1.0)
    rng = np.random.default_rng(20260531 + int(y * 17))
    ax.scatter(values, y - 0.035 + rng.uniform(-0.035, 0.035, len(values)), s=10, color=color, edgecolor="white", linewidth=0.22, alpha=0.70)
    ax.plot([np.median(values), np.median(values)], [y, y + 0.47], color=INK, lw=1.0)
    ax.text(-0.72, y + 0.25, f"{label}  n={len(values)}", fontsize=8.0, va="center")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "Arial",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
    })

    sample = pd.read_csv(SAMPLE, sep="\t")
    pred = pd.read_csv(PRED, sep="\t")
    pred = pred.rename(columns={"RPS6|S235_S236": "predicted_rps6_s235_s236_full"})
    axis_cols = ["MTOR|S2481", "EIF4EBP1|S65", "EIF4EBP1|T70", "RPS6KB1|T421_S424"]
    df = sample.merge(pred[["sample_id", "predicted_rps6_s235_s236_full"] + axis_cols], on="sample_id", how="left")
    df["predicted_rps6_s235_s236"] = df["predicted_rps6_s235_s236_full"].combine_first(df["predicted_rps6_s235_s236"])
    grade = pd.read_csv(GRADE, sep="\t")
    grade["_nonnull"] = grade[["histologic_grade", "pathologic_stage", "pathologic_T", "pathologic_N", "pathologic_M"]].notna().sum(axis=1)
    grade = grade.sort_values(["tcga_patient_id", "_nonnull"], ascending=[True, False]).drop_duplicates("tcga_patient_id")
    df = df.merge(grade.drop(columns=["_nonnull"]), on="tcga_patient_id", how="left")
    for c in axis_cols:
        df[c + "_z"] = zscore(df[c])
    df["predicted_mTOR_phospho_state"] = df[[c + "_z" for c in axis_cols]].mean(axis=1)
    cut = df["predicted_mTOR_phospho_state"].median()
    df["predicted mTOR phospho-state"] = np.where(df["predicted_mTOR_phospho_state"] >= cut, "High", "Low")
    df["OS status"] = np.where(df["survival_event"].astype(float) > 0, "Deceased", "Alive/censored")
    df["Grade"] = df["histologic_grade"].map(clean_grade)
    df.to_csv(TAB / "tcga_kirc_predicted_rps6_and_predicted_mtor_phospho_state.tsv", sep="\t", index=False)

    df["RPS6_mRNA_log2TPM"] = np.log2(pd.to_numeric(df["RPS6_mrna"], errors="coerce") + 1)

    fig, axes = plt.subplots(2, 3, figsize=(8.9, 5.7), sharey=False)
    s1 = raincloud(axes[0, 0], df, "OS status", ["Alive/censored", "Deceased"], [BLUE, WARM], "a  OS outcome", show_ylabel="Predicted RPS6 pS235/S236")
    s2 = raincloud(axes[0, 1], df.dropna(subset=["Grade"]), "Grade", ["G1/G2", "G3/G4"], [BLUE, WARM], "b  Histologic grade")
    s3 = raincloud(axes[0, 2], df.dropna(subset=["predicted_mTOR_phospho_state"]), "predicted mTOR phospho-state", ["Low", "High"], [BLUE, WARM], "c  Predicted mTOR phospho-state")
    s4 = raincloud(axes[1, 0], df, "OS status", ["Alive/censored", "Deceased"], [BLUE, WARM], "d  RPS6 mRNA control", ycol="RPS6_mRNA_log2TPM", show_ylabel="RPS6 mRNA log2(TPM+1)")
    s5 = raincloud(axes[1, 1], df.dropna(subset=["Grade"]), "Grade", ["G1/G2", "G3/G4"], [BLUE, WARM], "e  RPS6 mRNA control", ycol="RPS6_mRNA_log2TPM")
    s6 = raincloud(axes[1, 2], df.dropna(subset=["predicted_mTOR_phospho_state"]), "predicted mTOR phospho-state", ["Low", "High"], [BLUE, WARM], "f  RPS6 mRNA control", ycol="RPS6_mRNA_log2TPM")
    fig.suptitle("TCGA-KIRC predicted RPS6 pS235/S236 across patient states", x=0.02, y=1.03, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(w_pad=1.05, h_pad=1.25)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig5_tcga_predicted_rps6_ps6_clinical_mtor_phospho_raincloud.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)

    ridge_df = df.dropna(subset=["predicted mTOR phospho-state", "predicted_rps6_s235_s236"])
    fig, ax = plt.subplots(figsize=(4.9, 2.85))
    ridge(ax, ridge_df.loc[ridge_df["predicted mTOR phospho-state"] == "High", "predicted_rps6_s235_s236"], 1, WARM, "High predicted mTOR phospho-state")
    ridge(ax, ridge_df.loc[ridge_df["predicted mTOR phospho-state"] == "Low", "predicted_rps6_s235_s236"], 0, BLUE, "Low predicted mTOR phospho-state")
    ax.axvline(0, color="#AAAAAA", lw=0.8, ls="--")
    ax.set_ylim(-0.25, 1.75)
    ax.set_yticks([])
    ax.set_xlabel("Predicted RPS6 pS235/S236", fontsize=8.5)
    ax.set_title("Predicted pS6 stratified by predicted mTOR phosphorylation state", loc="left", fontsize=10.5, fontweight="bold")
    ax.grid(axis="x", color="#EFEFEF", lw=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig5_tcga_predicted_rps6_ps6_mtor_phospho_ridgeline.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)

    summary = {"raincloud": [s1, s2, s3, s4, s5, s6], "mTOR_phospho_state_features": axis_cols, "state_cutoff": float(cut)}
    (REPORT / "fig5_tcga_predicted_rps6_ps6_mtor_phospho_distribution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary["raincloud"]).to_csv(TAB / "tcga_kirc_predicted_rps6_ps6_clinical_mtor_phospho_raincloud_tests.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
