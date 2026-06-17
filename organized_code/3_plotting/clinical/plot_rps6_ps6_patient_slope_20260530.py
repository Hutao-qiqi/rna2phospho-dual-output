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
SRC = ROOT / "02_results/model_validation/20260530_fig5_exact_site_anchor_search_v1/tables/cptac_ccrcc_rps6_s235_s236_sample_table.tsv.gz"


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
    cols = {
        "Predicted pS6": "predicted_rps6_s235_s236",
        "Measured pS6": "measured_rps6_s235_s236",
        "RPS6 mRNA": "RPS6_mrna",
    }
    d = df[["case_submitter_id"] + list(cols.values())].dropna().copy()
    for label, col in cols.items():
        d[label] = zscore(d[col])
    d["pred_meas_gap"] = (d["Predicted pS6"] - d["Measured pS6"]).abs()
    d["mrna_gap"] = (0.5 * (d["Predicted pS6"] + d["Measured pS6"]) - d["RPS6 mRNA"]).abs()
    d["pattern"] = np.where((d["pred_meas_gap"] <= d["pred_meas_gap"].median()) & (d["mrna_gap"] >= d["mrna_gap"].median()), "pS6-aligned / mRNA-discordant", "other")
    d.to_csv(TAB / "cptac_ccrcc_rps6_ps6_patient_slope_plot_data.tsv", sep="\t", index=False)

    rho_pm, p_pm = stats.spearmanr(d["Predicted pS6"], d["Measured pS6"])
    rho_prna, p_prna = stats.spearmanr(d["Predicted pS6"], d["RPS6 mRNA"])
    rho_mrna, p_mrna = stats.spearmanr(d["Measured pS6"], d["RPS6 mRNA"])

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    x = np.array([0, 1, 2])
    labels = list(cols.keys())
    for _, r in d.sort_values("pattern").iterrows():
        y = np.array([r["Predicted pS6"], r["Measured pS6"], r["RPS6 mRNA"]], dtype=float)
        is_focus = r["pattern"] == "pS6-aligned / mRNA-discordant"
        ax.plot(x, y, color=WARM if is_focus else GREY, lw=1.4 if is_focus else 0.7, alpha=0.70 if is_focus else 0.36, zorder=2 if is_focus else 1)
        ax.scatter(x, y, s=20 if is_focus else 10, color=WARM if is_focus else GREY, edgecolor="white", linewidth=0.35, alpha=0.95 if is_focus else 0.55, zorder=3 if is_focus else 2)

    med = d[labels].median()
    ax.plot(x, med.values, color=INK, lw=2.2, marker="o", markersize=4.8, zorder=5)
    ax.axhline(0, color="#E8E8E8", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Within-cohort z score", fontsize=9)
    ax.set_title("Patient-level alignment between predicted and measured pS6", loc="left", fontsize=11.2, fontweight="bold")
    ax.text(0.02, 0.98, f"n={d.shape[0]}\nPredicted vs measured: rho={rho_pm:.2f}\nPredicted vs mRNA: rho={rho_prna:.2f}\nMeasured vs mRNA: rho={rho_mrna:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=3))
    ax.grid(axis="y", color="#EFEFEF", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig5_rps6_ps6_patient_slope_pred_measured_mrna.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_complete": int(d.shape[0]),
        "spearman_predicted_measured": {"rho": float(rho_pm), "p": float(p_pm)},
        "spearman_predicted_mrna": {"rho": float(rho_prna), "p": float(p_prna)},
        "spearman_measured_mrna": {"rho": float(rho_mrna), "p": float(p_mrna)},
        "focus_pattern_count": int((d["pattern"] == "pS6-aligned / mRNA-discordant").sum()),
    }
    (REPORT / "fig5_rps6_ps6_patient_slope_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
