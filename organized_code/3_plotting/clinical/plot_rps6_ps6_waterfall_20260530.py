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
SRC = ROOT / "02_results/model_validation/20260530_fig5_exact_site_anchor_search_v1/tables/tcga_kirc_rps6_s235_s236_sample_table.tsv.gz"

BLUE = "#92B1D9"
WARM = "#F6C8B6"
GREY = "#D4D4D4"
INK = "#222222"


def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return (x - x.mean()) / x.std(ddof=0)


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
    df = df.loc[df["has_os_survival"].astype(bool)].copy()
    df["RPS6_mrna_log2"] = np.log2(pd.to_numeric(df["RPS6_mrna"], errors="coerce") + 1)
    df["predicted_pS6_z"] = zscore(df["predicted_rps6_s235_s236"])
    df["RPS6_mrna_z"] = zscore(df["RPS6_mrna_log2"])
    fit = df[["predicted_pS6_z", "RPS6_mrna_z"]].dropna()
    X = np.column_stack([np.ones(fit.shape[0]), fit["RPS6_mrna_z"].values])
    beta = np.linalg.lstsq(X, fit["predicted_pS6_z"].values, rcond=None)[0]
    pred = np.column_stack([np.ones(df.shape[0]), df["RPS6_mrna_z"].values]) @ beta
    df["site_over_parent_residual"] = df["predicted_pS6_z"].values - pred
    df["OS status"] = np.where(df["survival_event"].astype(float) > 0, "Deceased", "Alive/censored")
    df = df.sort_values("site_over_parent_residual").reset_index(drop=True)
    df["waterfall_rank"] = np.arange(1, df.shape[0] + 1)
    df.to_csv(TAB / "tcga_kirc_rps6_ps6_site_over_parent_waterfall_data.tsv", sep="\t", index=False)

    n = df.shape[0]
    high = df.tail(n // 4)
    low = df.head(n // 4)
    high_events = int(high["survival_event"].sum())
    low_events = int(low["survival_event"].sum())
    table = [[high_events, high.shape[0] - high_events], [low_events, low.shape[0] - low_events]]
    odds, fisher_p = stats.fisher_exact(table)
    rho, rho_p = stats.spearmanr(df["site_over_parent_residual"], df["survival_event"])

    fig = plt.figure(figsize=(9.0, 3.55))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4.2, 0.55], hspace=0.04)
    ax = fig.add_subplot(gs[0])
    ann = fig.add_subplot(gs[1], sharex=ax)

    colors = np.where(df["survival_event"].astype(float) > 0, WARM, BLUE)
    ax.bar(df["waterfall_rank"], df["site_over_parent_residual"], width=0.92, color=colors, edgecolor="none", alpha=0.94)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylabel("pS6 beyond RPS6 mRNA\n(z residual)", fontsize=9)
    ax.set_title("Patient-level waterfall of mRNA-independent predicted RPS6 pS235/S236 in TCGA-KIRC", loc="left", fontsize=12, fontweight="bold")
    ax.text(0.99, 0.98, f"n={n}; events={int(df['survival_event'].sum())}\nTop quartile events: {high_events}/{high.shape[0]}\nBottom quartile events: {low_events}/{low.shape[0]}\nFisher OR={odds:.2f}, p={fisher_p:.2e}", transform=ax.transAxes, ha="right", va="top", fontsize=8.2, bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=3))
    ax.grid(axis="y", color="#EFEFEF", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=8)

    ann.scatter(df["waterfall_rank"], np.zeros(n), c=colors, s=9, marker="s", linewidths=0)
    ann.set_ylim(-0.8, 0.8)
    ann.set_yticks([0])
    ann.set_yticklabels(["OS event"], fontsize=8)
    ann.set_xlabel("Patients sorted by site-over-parent residual", fontsize=9)
    ann.tick_params(axis="x", labelsize=8, length=2)
    for spine in ann.spines.values():
        spine.set_visible(False)
    ann.grid(False)

    handles = [
        mpl.patches.Patch(color=WARM, label="Deceased"),
        mpl.patches.Patch(color=BLUE, label="Alive/censored"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.92), fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig5_rps6_ps6_site_over_parent_waterfall.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n": int(n),
        "events": int(df["survival_event"].sum()),
        "residual_definition": "z(predicted RPS6 pS235/S236) residualized against z(log2(RPS6 mRNA + 1))",
        "top_quartile_events": high_events,
        "top_quartile_n": int(high.shape[0]),
        "bottom_quartile_events": low_events,
        "bottom_quartile_n": int(low.shape[0]),
        "fisher_or_top_vs_bottom": float(odds),
        "fisher_p": float(fisher_p),
        "spearman_residual_vs_event": {"rho": float(rho), "p": float(rho_p)},
    }
    (REPORT / "fig5_rps6_ps6_site_over_parent_waterfall_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
