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
BUBBLE_INPUT = TAB / "cptac_ccrcc_rps6_ps6_bubble_input_matrix.tsv"

BLUE = "#92B1D9"
WARM = "#F6C8B6"
GREY = "#D4D4D4"
INK = "#222222"


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return (x - x.mean()) / x.std(ddof=0)


def p_text(p: float) -> str:
    if p < 1e-4:
        return f"p={p:.1e}"
    if p < 0.001:
        return f"p={p:.2e}"
    return f"p={p:.3f}"


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

    df = pd.read_csv(BUBBLE_INPUT, sep="\t")
    axis_cols = ["MS p-mTOR S2481", "MS p-4EBP1 S65", "MS p-4EBP1 T70", "MS p-S6K1 T421/S424"]
    for c in axis_cols:
        df[c + " z"] = zscore(df[c])
    df["mTOR phospho-state"] = df[[c + " z" for c in axis_cols]].mean(axis=1, skipna=True)
    use = df.dropna(subset=["predicted_rps6_s235_s236", "measured_rps6_s235_s236", "mTOR phospho-state"]).copy()
    cut = use["mTOR phospho-state"].median()
    use["mTOR phospho-state group"] = np.where(use["mTOR phospho-state"] >= cut, "High mTOR phospho-state", "Low mTOR phospho-state")

    records = []
    for x in ["predicted_rps6_s235_s236", "measured_rps6_s235_s236"]:
        rho, p = stats.spearmanr(use[x], use["mTOR phospho-state"])
        records.append({"variable": x, "comparison": "mTOR phospho-state", "n": int(use[[x, "mTOR phospho-state"]].dropna().shape[0]), "spearman_rho": float(rho), "p": float(p)})
    for c in axis_cols:
        for x in ["predicted_rps6_s235_s236", "measured_rps6_s235_s236"]:
            d = df[[x, c]].dropna()
            if len(d) >= 8:
                rho, p = stats.spearmanr(d[x], d[c])
                records.append({"variable": x, "comparison": c, "n": int(len(d)), "spearman_rho": float(rho), "p": float(p)})
    pd.DataFrame(records).to_csv(TAB / "cptac_ccrcc_rps6_ps6_mtor_phospho_state_correlations.tsv", sep="\t", index=False)
    use.to_csv(TAB / "cptac_ccrcc_rps6_ps6_mtor_phospho_state_plot_data.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.15), gridspec_kw={"width_ratios": [1.05, 1.05, 1.1]})
    for ax, ycol, title in [
        (axes[0], "predicted_rps6_s235_s236", "Predicted pS6"),
        (axes[1], "measured_rps6_s235_s236", "Measured pS6"),
    ]:
        d = use[["mTOR phospho-state group", ycol]].dropna()
        groups = ["Low mTOR phospho-state", "High mTOR phospho-state"]
        vals = [d.loc[d["mTOR phospho-state group"] == g, ycol].values for g in groups]
        parts = ax.violinplot(vals, positions=[0, 1], widths=0.75, showmeans=False, showmedians=False, showextrema=False)
        for body, color in zip(parts["bodies"], [BLUE, WARM]):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.42)
        rng = np.random.default_rng(20260530)
        for i, (v, color) in enumerate(zip(vals, [BLUE, WARM])):
            ax.scatter(i + rng.normal(0, 0.055, len(v)), v, s=18, color=color, edgecolor="white", linewidth=0.35, alpha=0.85)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([i - 0.18, i + 0.18], [med, med], color=INK, lw=1.5)
            ax.plot([i, i], [q1, q3], color=INK, lw=2.5)
            ax.text(i, ax.get_ylim()[0], f"n={len(v)}", ha="center", va="bottom", fontsize=7, color="#666666")
        stat, p = stats.mannwhitneyu(vals[0], vals[1], alternative="two-sided")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.text(0.5, 0.98, p_text(p), transform=ax.transAxes, ha="center", va="top", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Low", "High"], fontsize=8)
        ax.set_xlabel("mTOR phospho-state", fontsize=8)
        ax.grid(axis="y", color="#EFEFEF", lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
    axes[0].set_ylabel("RPS6 pS235/S236 value", fontsize=9)
    axes[1].set_ylabel("")

    ax = axes[2]
    ax.scatter(use["mTOR phospho-state"], use["predicted_rps6_s235_s236"], s=26, color=WARM, edgecolor="white", linewidth=0.45, alpha=0.85, label="Predicted pS6")
    slope, intercept, r, p, se = stats.linregress(use["mTOR phospho-state"], use["predicted_rps6_s235_s236"])
    xx = np.linspace(use["mTOR phospho-state"].min(), use["mTOR phospho-state"].max(), 100)
    ax.plot(xx, intercept + slope * xx, color=INK, lw=1.2)
    rho, sp = stats.spearmanr(use["mTOR phospho-state"], use["predicted_rps6_s235_s236"])
    ax.text(0.03, 0.97, f"rho={rho:.2f}\n{p_text(sp)}", transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2))
    ax.set_title("Continuous state", fontsize=10, fontweight="bold")
    ax.set_xlabel("mTOR phospho-state score", fontsize=8)
    ax.set_ylabel("Predicted pS6", fontsize=8)
    ax.grid(color="#EFEFEF", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)

    fig.suptitle("Predicted RPS6 pS235/S236 tracks measured mTOR phosphorylation state", x=0.02, y=1.03, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig5_rps6_ps6_mtor_phospho_state.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_plot": int(use.shape[0]),
        "axis_features": axis_cols,
        "group_cutoff": float(cut),
    }
    (REPORT / "fig5_rps6_ps6_mtor_phospho_state_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
