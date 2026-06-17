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
SRC = TAB / "tcga_kirc_rps6_ps6_with_mtor_axis_mutation.tsv"
GRADE = TAB / "kirc_clinical_grade_stage.tsv"

BLUE = "#92B1D9"
WARM = "#F6C8B6"
GREY = "#D4D4D4"
PURPLE = "#DBDDEF"
INK = "#222222"


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


def half_violin(ax, values, xpos, color, width=0.18):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return
    kde = stats.gaussian_kde(values)
    yr = values.max() - values.min()
    y = np.linspace(values.min() - 0.08 * yr, values.max() + 0.08 * yr, 220)
    dens = kde(y)
    dens = dens / dens.max() * width
    ax.fill_betweenx(y, xpos, xpos - dens, color=color, alpha=0.36, lw=0)
    ax.plot(xpos - dens, y, color=color, lw=1.0)


def raincloud(ax, df: pd.DataFrame, group_col: str, order: list[str], colors: list[str], title: str, test: str = "mw"):
    ycol = "predicted_rps6_s235_s236"
    rng = np.random.default_rng(20260531)
    vals_list = []
    for i, (g, color) in enumerate(zip(order, colors)):
        vals = df.loc[df[group_col] == g, ycol].dropna().astype(float).values
        vals_list.append(vals)
        half_violin(ax, vals, i, color, width=0.18)
        if len(vals):
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            lo = vals[vals >= q1 - 1.5 * (q3 - q1)].min()
            hi = vals[vals <= q3 + 1.5 * (q3 - q1)].max()
            ax.add_patch(plt.Rectangle((i - 0.055, q1), 0.11, q3 - q1, facecolor="white", edgecolor=INK, lw=0.8, zorder=5))
            ax.plot([i - 0.055, i + 0.055], [med, med], color=INK, lw=1.1, zorder=6)
            ax.plot([i, i], [lo, q1], color=INK, lw=0.75, zorder=5)
            ax.plot([i, i], [q3, hi], color=INK, lw=0.75, zorder=5)
            x = i + 0.075 + rng.normal(0, 0.018, len(vals))
            ax.scatter(x, vals, s=10 if len(vals) > 80 else 17, color=color, edgecolor="white", linewidth=0.22, alpha=0.72, zorder=4)
            ax.text(i, -0.16, f"n={len(vals)}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5, color="#666666")

    p = np.nan
    desc = ""
    if test == "mw" and len(vals_list) == 2:
        p = stats.mannwhitneyu(vals_list[0], vals_list[1], alternative="two-sided").pvalue
        desc = f"shift={np.median(vals_list[1]) - np.median(vals_list[0]):.2f}; {p_text(p)}"
    elif test == "kw":
        nonempty = [v for v in vals_list if len(v) > 1]
        if len(nonempty) > 1:
            p = stats.kruskal(*nonempty).pvalue
            desc = f"Kruskal {p_text(p)}"
    ax.set_title(title, loc="left", fontsize=9.4, fontweight="bold")
    if desc:
        ax.text(0.98, 0.96, desc, transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.6))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=7.5, rotation=0)
    ax.grid(axis="y", color="#EFEFEF", lw=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.5)
    return {"panel": title, "group": group_col, "order": order, "n": [int(len(v)) for v in vals_list], "p": float(p) if np.isfinite(p) else np.nan}


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

    df = pd.read_csv(SRC, sep="\t")
    grade = pd.read_csv(GRADE, sep="\t")
    grade["_nonnull"] = grade[["histologic_grade", "pathologic_stage", "pathologic_T", "pathologic_N", "pathologic_M"]].notna().sum(axis=1)
    grade = grade.sort_values(["tcga_patient_id", "_nonnull"], ascending=[True, False]).drop_duplicates("tcga_patient_id")
    df = df.drop(columns=[c for c in ["histologic_grade", "pathologic_stage", "pathologic_T", "pathologic_N", "pathologic_M"] if c in df.columns], errors="ignore")
    df = df.merge(grade.drop(columns=["_nonnull"]), on="tcga_patient_id", how="left")
    df["OS status"] = np.where(df["survival_event"].astype(float) > 0, "Deceased", "Alive/censored")
    df["Grade"] = df["histologic_grade"].map(clean_grade)
    df["mTOR axis"] = df["mTOR_axis_category"].replace({
        "PTEN/TSC-axis altered": "PTEN/TSC",
        "PI3K-AKT-axis altered": "PI3K-AKT",
        "mTOR-complex-axis altered": "mTOR complex",
    })
    plot_df = df[["sample_id", "tcga_patient_id", "predicted_rps6_s235_s236", "survival_event", "OS status", "Grade", "maf_covered", "mTOR axis", "mTOR_axis_category", "mutated_mTOR_axis_genes"]].copy()
    plot_df.to_csv(TAB / "tcga_kirc_predicted_rps6_ps6_combined_raincloud_data.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.35), sharey=True, gridspec_kw={"width_ratios": [1.0, 1.0, 1.48]})
    s1 = raincloud(axes[0], df, "OS status", ["Alive/censored", "Deceased"], [BLUE, WARM], "a  OS outcome", "mw")
    s2 = raincloud(axes[1], df.dropna(subset=["Grade"]), "Grade", ["G1/G2", "G3/G4"], [BLUE, WARM], "b  Histologic grade", "mw")
    mut_df = df.loc[df["maf_covered"].fillna(0).astype(int) == 1].copy()
    s3 = raincloud(axes[2], mut_df, "mTOR axis", ["WT", "PTEN/TSC", "PI3K-AKT", "mTOR complex"], [GREY, WARM, PURPLE, BLUE], "c  mTOR-axis mutation", "kw")
    axes[0].set_ylabel("Predicted RPS6 pS235/S236", fontsize=8.8)
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    axes[2].tick_params(axis="x", labelrotation=14)
    fig.suptitle("TCGA-KIRC predicted RPS6 pS235/S236 across patient states", x=0.02, y=1.03, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(w_pad=1.05)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig5_tcga_predicted_rps6_ps6_combined_raincloud.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)

    summary = {"panels": [s1, s2, s3], "source": str(SRC)}
    (REPORT / "fig5_tcga_predicted_rps6_ps6_combined_raincloud_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary["panels"]).to_csv(TAB / "tcga_kirc_predicted_rps6_ps6_combined_raincloud_tests.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
