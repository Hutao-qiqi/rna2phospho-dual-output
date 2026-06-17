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

BLUE = "#92B1D9"
WARM = "#F6C8B6"
GREY = "#D4D4D4"
PURPLE = "#DBDDEF"
INK = "#222222"


def ridge(ax, values, y, color, scale=0.72):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        ax.scatter(values, np.repeat(y, len(values)), s=8, color=color, alpha=0.7)
        return
    kde = stats.gaussian_kde(values)
    xs = np.linspace(-2.7, 3.2, 360)
    dens = kde(xs)
    dens = dens / dens.max() * scale
    ax.fill_between(xs, y, y + dens, facecolor=color, edgecolor=color, alpha=0.48, lw=1.0)
    ax.plot(xs, y + dens, color=color, lw=1.1)
    rng = np.random.default_rng(20260530 + int(y * 10))
    jitter = rng.uniform(-0.045, 0.045, len(values))
    ax.scatter(values, np.repeat(y - 0.035, len(values)) + jitter, s=8, color=color, edgecolor="white", linewidth=0.2, alpha=0.72)
    med = float(np.median(values))
    ax.plot([med, med], [y, y + scale * 0.82], color=INK, lw=1.0)


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
    df = df.loc[df["maf_covered"].fillna(0).astype(int) == 1].copy()
    order = ["WT", "PTEN/TSC-axis altered", "PI3K-AKT-axis altered", "mTOR-complex-axis altered"]
    colors = [GREY, WARM, PURPLE, BLUE]
    labels = ["WT", "PTEN/TSC axis", "PI3K-AKT axis", "mTOR complex axis"]

    plot_data = df.loc[df["mTOR_axis_category"].isin(order)].copy()
    plot_data.to_csv(TAB / "tcga_kirc_rps6_ps6_mtor_axis_mutation_ridgeline_data.tsv", sep="\t", index=False)

    rows = []
    wt = plot_data.loc[plot_data["mTOR_axis_category"] == "WT", "site_over_parent_residual"].dropna()
    for cat in order:
        vals = plot_data.loc[plot_data["mTOR_axis_category"] == cat, "site_over_parent_residual"].dropna()
        p = np.nan
        if cat != "WT" and len(vals) and len(wt):
            p = stats.mannwhitneyu(vals, wt, alternative="two-sided").pvalue
        rows.append({"category": cat, "n": int(len(vals)), "median_site_over_parent_residual": float(vals.median()), "vs_WT_p": float(p) if np.isfinite(p) else np.nan})
    pd.DataFrame(rows).to_csv(TAB / "tcga_kirc_rps6_ps6_mtor_axis_mutation_ridgeline_summary.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    y_positions = np.arange(len(order))[::-1]
    for y, cat, color, lab in zip(y_positions, order, colors, labels):
        vals = plot_data.loc[plot_data["mTOR_axis_category"] == cat, "site_over_parent_residual"].dropna()
        ridge(ax, vals, y, color)
        ax.text(-2.95, y + 0.28, f"{lab}  n={len(vals)}", ha="left", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color="#A8A8A8", lw=0.9, ls="--")
    ax.set_xlim(-3.0, 3.25)
    ax.set_ylim(-0.35, len(order) - 0.05)
    ax.set_yticks([])
    ax.set_xlabel("pS6 beyond RPS6 mRNA (z residual)", fontsize=9)
    ax.set_title("mTOR-axis mutations do not explain the pS6 residual distribution", loc="left", fontsize=11.5, fontweight="bold")
    ax.grid(axis="x", color="#EFEFEF", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    note = "MAF-covered TCGA-KIRC samples only; Unknown mutation status excluded."
    fig.text(0.02, 0.02, note, fontsize=7.2, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig5_rps6_ps6_mtor_axis_mutation_ridgeline.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)

    summary = {"n_maf_covered": int(plot_data.shape[0]), "categories": rows}
    (REPORT / "fig5_rps6_ps6_mtor_axis_mutation_ridgeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
