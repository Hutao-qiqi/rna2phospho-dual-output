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
SRC = ROOT / "02_results/model_validation/20260530_fig5_exact_site_anchor_search_v1/tables/tcga_kirc_rps6_s235_s236_sample_table.tsv.gz"
GRADE = ROOT / "02_results/model_validation/20260530_fig5_rps6_ps6_panel_v1/tables/kirc_clinical_grade_stage.tsv"
OUT = ROOT / "02_results/model_validation/20260530_fig5_rps6_ps6_panel_v1"
FIG = OUT / "figures"
TAB = OUT / "tables"
REPORT = OUT / "reports"


BLUE = "#92B1D9"
WARM = "#F6C8B6"
GREY = "#D4D4D4"
INK = "#222222"


def p_text(p: float) -> str:
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
    if "G1" in s or s == "1":
        return "G1/G2"
    if "G2" in s or s == "2":
        return "G1/G2"
    if "G3" in s or s == "3":
        return "G3/G4"
    if "G4" in s or s == "4":
        return "G3/G4"
    return None


def half_violin(ax, values, xpos, color, width=0.28, side="left"):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return
    kde = stats.gaussian_kde(values)
    y = np.linspace(values.min() - 0.08 * values.ptp(), values.max() + 0.08 * values.ptp(), 240)
    dens = kde(y)
    dens = dens / dens.max() * width
    if side == "left":
        x = xpos - dens
    else:
        x = xpos + dens
    ax.fill_betweenx(y, xpos, x, color=color, alpha=0.55, lw=0)
    ax.plot(x, y, color=color, lw=1.2)


def raincloud(ax, data, group_col, order, colors, title):
    ycol = "predicted_rps6_s235_s236"
    rng = np.random.default_rng(20260530)
    rows = []
    for i, g in enumerate(order):
        vals = data.loc[data[group_col] == g, ycol].dropna().astype(float).values
        rows.append(vals)
        half_violin(ax, vals, i, colors[i], side="left")
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        lo = vals[vals >= q1 - 1.5 * (q3 - q1)].min()
        hi = vals[vals <= q3 + 1.5 * (q3 - q1)].max()
        ax.add_patch(plt.Rectangle((i - 0.10, q1), 0.20, q3 - q1, facecolor="white", edgecolor=INK, lw=1.0, zorder=4))
        ax.plot([i - 0.10, i + 0.10], [med, med], color=INK, lw=1.3, zorder=5)
        ax.plot([i, i], [lo, q1], color=INK, lw=0.9, zorder=4)
        ax.plot([i, i], [q3, hi], color=INK, lw=0.9, zorder=4)
        x = i + 0.09 + rng.normal(0, 0.024, len(vals))
        ax.scatter(x, vals, s=13, facecolor=colors[i], edgecolor="white", linewidth=0.25, alpha=0.72, zorder=3)
        ax.text(i, -0.10, f"n={len(vals)}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="#555555")

    stat, p = stats.mannwhitneyu(rows[0], rows[1], alternative="two-sided")
    delta = np.median(rows[1]) - np.median(rows[0])
    ymax = max(np.max(rows[0]), np.max(rows[1]))
    ymin = min(np.min(rows[0]), np.min(rows[1]))
    yr = ymax - ymin
    ax.plot([0, 0, 1, 1], [ymax + 0.07 * yr, ymax + 0.11 * yr, ymax + 0.11 * yr, ymax + 0.07 * yr], color=INK, lw=0.8)
    ax.text(0.5, ymax + 0.13 * yr, f"median shift={delta:.2f}; {p_text(p)}", ha="center", va="bottom", fontsize=8.5)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11, pad=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=9)
    ax.set_xlim(-0.75, len(order) - 0.25)
    ax.set_ylabel("Predicted RPS6 pS235/S236", fontsize=9)
    ax.grid(axis="y", color="#ECECEC", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    return {"group": group_col, "order": order, "n": [int(len(x)) for x in rows], "median": [float(np.median(x)) for x in rows], "median_shift_second_minus_first": float(delta), "mannwhitney_u": float(stat), "p": float(p)}


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
    grade = grade.sort_values(["tcga_patient_id", "_nonnull"], ascending=[True, False]).drop_duplicates("tcga_patient_id", keep="first")
    grade = grade.drop(columns=["_nonnull"])
    df = df.merge(grade, on="tcga_patient_id", how="left")
    df = df.loc[df["has_os_survival"].astype(bool)].copy()
    df["OS status"] = np.where(df["survival_event"].astype(float) > 0, "Deceased", "Alive/censored")
    df["Grade"] = df["histologic_grade"].map(clean_grade)

    plot_df = df[["sample_id", "tcga_patient_id", "survival_time", "survival_event", "predicted_rps6_s235_s236", "OS status", "histologic_grade", "Grade", "pathologic_stage"]].copy()
    plot_df.to_csv(TAB / "tcga_kirc_rps6_ps6_raincloud_plot_data.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), sharey=True)
    s1 = raincloud(axes[0], df, "OS status", ["Alive/censored", "Deceased"], [BLUE, WARM], "a  OS outcome")
    grade_df = df.dropna(subset=["Grade"]).copy()
    s2 = raincloud(axes[1], grade_df, "Grade", ["G1/G2", "G3/G4"], [BLUE, WARM], "b  Histologic grade")
    axes[1].set_ylabel("")
    fig.suptitle("Predicted RPS6 pS235/S236 is enriched in adverse ccRCC states", x=0.02, ha="left", y=1.02, fontsize=12.5, fontweight="bold")
    fig.tight_layout(w_pad=1.2)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig5_rps6_ps6_raincloud_clinical_states.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)

    summary = {"source": str(SRC), "grade_source": str(GRADE), "panels": [s1, s2], "n_total_with_os": int(df.shape[0]), "n_with_grade": int(grade_df.shape[0])}
    (REPORT / "fig5_rps6_ps6_raincloud_clinical_states_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary["panels"]).to_csv(TAB / "tcga_kirc_rps6_ps6_raincloud_group_tests.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
