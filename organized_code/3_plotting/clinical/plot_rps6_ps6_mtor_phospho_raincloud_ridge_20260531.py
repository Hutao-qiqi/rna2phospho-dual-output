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
SRC = TAB / "cptac_ccrcc_rps6_ps6_mtor_phospho_state_plot_data.tsv"

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


def half_violin(ax, values, xpos, color, width=0.24, side="left"):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return
    kde = stats.gaussian_kde(values)
    pad = 0.08 * (values.max() - values.min())
    y = np.linspace(values.min() - pad, values.max() + pad, 240)
    dens = kde(y)
    dens = dens / dens.max() * width
    x = xpos - dens if side == "left" else xpos + dens
    ax.fill_betweenx(y, xpos, x, color=color, alpha=0.42, lw=0)
    ax.plot(x, y, color=color, lw=1.1, alpha=0.95)


def raincloud_panel(ax, df, value_col, title):
    groups = ["Low mTOR phospho-state", "High mTOR phospho-state"]
    colors = [BLUE, WARM]
    rng = np.random.default_rng(20260531)
    vals_all = []
    for i, (g, color) in enumerate(zip(groups, colors)):
        vals = df.loc[df["mTOR phospho-state group"] == g, value_col].dropna().astype(float).values
        vals_all.append(vals)
        half_violin(ax, vals, i, color, width=0.24, side="left")
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        lo = vals[vals >= q1 - 1.5 * (q3 - q1)].min()
        hi = vals[vals <= q3 + 1.5 * (q3 - q1)].max()
        ax.add_patch(plt.Rectangle((i - 0.08, q1), 0.16, q3 - q1, facecolor="white", edgecolor=INK, lw=0.9, zorder=5))
        ax.plot([i - 0.08, i + 0.08], [med, med], color=INK, lw=1.25, zorder=6)
        ax.plot([i, i], [lo, q1], color=INK, lw=0.8, zorder=5)
        ax.plot([i, i], [q3, hi], color=INK, lw=0.8, zorder=5)
        x = i + 0.09 + rng.normal(0, 0.022, len(vals))
        ax.scatter(x, vals, s=22, color=color, edgecolor="white", linewidth=0.35, alpha=0.85, zorder=4)
        ax.text(i, -0.12, f"n={len(vals)}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.2, color="#666666")
    stat, p = stats.mannwhitneyu(vals_all[0], vals_all[1], alternative="two-sided")
    delta = float(np.median(vals_all[1]) - np.median(vals_all[0]))
    ymax = max(np.max(vals_all[0]), np.max(vals_all[1]))
    ymin = min(np.min(vals_all[0]), np.min(vals_all[1]))
    yr = ymax - ymin
    ax.plot([0, 0, 1, 1], [ymax + 0.07 * yr, ymax + 0.11 * yr, ymax + 0.11 * yr, ymax + 0.07 * yr], color=INK, lw=0.8)
    ax.text(0.5, ymax + 0.13 * yr, f"shift={delta:.2f}; {p_text(p)}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Low", "High"], fontsize=8)
    ax.set_xlabel("mTOR phospho-state", fontsize=8)
    ax.grid(axis="y", color="#EFEFEF", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    return {"variable": value_col, "n_low": int(len(vals_all[0])), "n_high": int(len(vals_all[1])), "median_shift_high_minus_low": delta, "p": float(p)}


def ridge(ax, values, y, color, label):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    kde = stats.gaussian_kde(values)
    xs = np.linspace(-2.6, 2.2, 300)
    dens = kde(xs)
    dens = dens / dens.max() * 0.62
    ax.fill_between(xs, y, y + dens, color=color, alpha=0.42, lw=0)
    ax.plot(xs, y + dens, color=color, lw=1.1)
    rng = np.random.default_rng(20260531 + int(y * 10))
    ax.scatter(values, y - 0.035 + rng.uniform(-0.04, 0.04, len(values)), s=13, color=color, edgecolor="white", linewidth=0.25, alpha=0.78)
    ax.plot([np.median(values), np.median(values)], [y, y + 0.50], color=INK, lw=1.0)
    ax.text(-2.55, y + 0.28, f"{label}  n={len(values)}", fontsize=8.5, va="center")


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
    df.to_csv(TAB / "cptac_ccrcc_rps6_ps6_mtor_phospho_raincloud_ridgeline_data.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.25), sharey=False)
    s1 = raincloud_panel(axes[0], df, "predicted_rps6_s235_s236", "a  Predicted pS6")
    s2 = raincloud_panel(axes[1], df, "measured_rps6_s235_s236", "b  Measured pS6")
    axes[0].set_ylabel("RPS6 pS235/S236", fontsize=9)
    fig.suptitle("RPS6 pS235/S236 across measured mTOR phosphorylation states", x=0.02, y=1.03, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(w_pad=1.0)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig5_rps6_ps6_mtor_phospho_state_raincloud.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), sharex=False)
    for ax, value_col, title in [
        (axes[0], "predicted_rps6_s235_s236", "Predicted pS6"),
        (axes[1], "measured_rps6_s235_s236", "Measured pS6"),
    ]:
        low = df.loc[df["mTOR phospho-state group"] == "Low mTOR phospho-state", value_col].dropna().values
        high = df.loc[df["mTOR phospho-state group"] == "High mTOR phospho-state", value_col].dropna().values
        ridge(ax, high, 1, WARM, "High mTOR phospho-state")
        ridge(ax, low, 0, BLUE, "Low mTOR phospho-state")
        ax.axvline(0, color="#AAAAAA", lw=0.8, ls="--")
        ax.set_ylim(-0.25, 1.8)
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("RPS6 pS235/S236", fontsize=8.5)
        ax.grid(axis="x", color="#EFEFEF", lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Ridgeline view of pS6 by mTOR phosphorylation state", x=0.02, y=1.03, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(w_pad=1.3)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig5_rps6_ps6_mtor_phospho_state_ridgeline.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)

    summary = {"raincloud": [s1, s2], "source": str(SRC)}
    (REPORT / "fig5_rps6_ps6_mtor_phospho_raincloud_ridgeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary["raincloud"]).to_csv(TAB / "cptac_ccrcc_rps6_ps6_mtor_phospho_raincloud_tests.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
