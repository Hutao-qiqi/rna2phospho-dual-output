#!/usr/bin/env python
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.linear_model import LinearRegression


ROOT = Path("E:/data/gongke/TCGA-TCPA")
IN1 = ROOT / "02_results/model_validation/20260530_fig5_exact_site_anchor_search_v1/tables"
IN2 = ROOT / "02_results/model_validation/20260530_fig5_rps6_ps6_panel_v1/tables"
OUT = ROOT / "02_results/model_validation/20260530_fig5_rps6_ps6_panel_v1"
FIG = OUT / "figures"
TAB = OUT / "tables"

BLUE = "#92B1D9"
WARM = "#F6C8B6"
GREY = "#D4D4D4"
PURPLE = "#DBDDEF"
DARK = "#333333"


def ensure_dirs():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)


def set_style():
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#333333",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def km_curve(time, event):
    d = pd.DataFrame({"time": time, "event": event}).dropna().sort_values("time")
    d = d[d["time"] > 0]
    if d.empty:
        return np.array([0.0]), np.array([1.0])
    surv = 1.0
    xs = [0.0]
    ys = [1.0]
    for t, g in d.groupby("time"):
        n_risk = (d["time"] >= t).sum()
        n_event = g["event"].sum()
        xs.extend([t, t])
        ys.extend([surv, surv * (1 - n_event / n_risk)])
        surv = ys[-1]
    return np.asarray(xs), np.asarray(ys)


def logrank_p(df, group_col):
    d = df[["survival_time", "survival_event", group_col]].dropna()
    g1 = d[d[group_col] == "high"]
    g0 = d[d[group_col] == "low"]
    times = np.sort(d.loc[d["survival_event"].eq(1), "survival_time"].unique())
    obs = exp = var = 0.0
    for t in times:
        n1 = (g1["survival_time"] >= t).sum()
        n0 = (g0["survival_time"] >= t).sum()
        d1 = ((g1["survival_time"] == t) & (g1["survival_event"] == 1)).sum()
        d0 = ((g0["survival_time"] == t) & (g0["survival_event"] == 1)).sum()
        n = n1 + n0
        dd = d1 + d0
        if n <= 1:
            continue
        obs += d1
        exp += dd * n1 / n
        var += n1 * n0 * dd * (n - dd) / (n * n * (n - 1))
    z = (obs - exp) / math.sqrt(var) if var > 0 else np.nan
    p = float(stats.chi2.sf(z * z, 1)) if np.isfinite(z) else np.nan
    return z, p


def draw_km(ax, risk_ax):
    df = pd.read_csv(IN1 / "tcga_kirc_rps6_s235_s236_sample_table.tsv.gz", sep="\t")
    med = df["predicted_rps6_s235_s236"].median()
    df["ps6_group"] = np.where(df["predicted_rps6_s235_s236"] >= med, "high", "low")
    z, p = logrank_p(df, "ps6_group")
    for group, color, label in [("low", BLUE, "low predicted pS6"), ("high", WARM, "high predicted pS6")]:
        g = df[df["ps6_group"] == group]
        x, y = km_curve(g["survival_time"], g["survival_event"])
        ax.step(x / 365.25, y, where="post", color=color, lw=1.6, label=f"{label} (n={len(g)})")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Overall survival")
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    ax.text(0.02, 0.96, f"Median split; log-rank p={p:.2g}", transform=ax.transAxes, ha="left", va="top")
    ax.set_title("a  TCGA-KIRC clinical signal beyond RPS6 mRNA", loc="left", fontweight="bold", fontsize=10)

    ticks = np.array([0, 3, 6, 9, 12, 15])
    risk_ax.set_xlim(0, 15)
    risk_ax.set_ylim(-0.5, 1.5)
    risk_ax.axis("off")
    risk_ax.text(-0.3, 1.25, "Number at risk", ha="right", va="center", fontsize=7)
    for yi, (group, color) in enumerate([("high", WARM), ("low", BLUE)]):
        g = df[df["ps6_group"] == group]
        risk_ax.text(-0.3, yi, group, ha="right", va="center", fontsize=7, color=color)
        for t in ticks:
            n = (g["survival_time"] >= t * 365.25).sum()
            risk_ax.text(t, yi, str(int(n)), ha="center", va="center", fontsize=7)
    for t in ticks:
        risk_ax.text(t, -0.35, str(int(t)), ha="center", va="center", fontsize=7)
    return df


def draw_forest(ax):
    cox = pd.read_csv(IN1 / "tcga_kirc_rps6_s235_s236_parent_mrna_cox.tsv", sep="\t")
    rows = [
        ("RPS6 mRNA", cox.query("model == 'parent_mrna_only'").iloc[0], BLUE),
        ("predicted pS6", cox.query("model == 'site_only'").iloc[0], WARM),
        ("pS6 | mRNA", cox.query("model == 'site_plus_parent_mrna' and variable == 'predicted_rps6_s235_s236'").iloc[0], WARM),
    ]
    y = np.array([2.2, 1.1, 0.0])
    for yi, (label, row, color) in zip(y, rows):
        beta = row["beta_per_sd"]
        se = row["se"]
        hr = row["hr_per_sd"]
        lo = math.exp(beta - 1.96 * se)
        hi = math.exp(beta + 1.96 * se)
        ax.plot([lo, hi], [yi, yi], color=color, lw=1.5)
        ax.scatter([hr], [yi], s=45, color=color, edgecolor=DARK, linewidth=0.4, zorder=3)
        ax.text(2.12, yi, f"{hr:.2f} ({lo:.2f}-{hi:.2f})", va="center", fontsize=7)
    ax.axvline(1, color="#777777", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.65, 2.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Hazard ratio per SD")
    ax.set_ylim(-0.35, 2.55)
    ax.text(0.98, 0.05, "LRT p=8.26e-7", fontsize=7, ha="right", transform=ax.transAxes)
    ax.spines[["top", "right"]].set_visible(False)


def residualize(y, cov):
    d = pd.concat([y, cov], axis=1).dropna()
    Y = d.iloc[:, 0].to_numpy(float)
    X = d.iloc[:, 1:].to_numpy(float)
    if X.shape[1] == 0:
        return pd.Series(Y - Y.mean(), index=d.index)
    pred = LinearRegression().fit(X, Y).predict(X)
    return pd.Series(Y - pred, index=d.index)


def draw_measured(ax_raw, ax_resid, ax_bar):
    df = pd.read_csv(IN1 / "cptac_ccrcc_rps6_s235_s236_sample_table.tsv.gz", sep="\t")
    x = "predicted_rps6_s235_s236"
    y = "measured_rps6_s235_s236"
    d = df[[x, y]].dropna()
    rho, p = stats.spearmanr(d[x], d[y])
    ax_raw.scatter(d[x], d[y], s=28, color=WARM, edgecolor=DARK, linewidth=0.35, alpha=0.9)
    slope, intercept = np.polyfit(d[x], d[y], 1)
    xs = np.linspace(d[x].min(), d[x].max(), 100)
    ax_raw.plot(xs, slope * xs + intercept, color=DARK, lw=1)
    ax_raw.set_xlabel("Predicted RPS6 pS235/S236")
    ax_raw.set_ylabel("Measured RPS6 pS235/S236")
    ax_raw.set_title("b  Predicted pS6 tracks measured pS6", loc="left", fontweight="bold", fontsize=10)
    ax_raw.text(0.04, 0.96, f"CPTAC ccRCC, n={len(d)}\nSpearman rho={rho:.2f}, p={p:.1e}", transform=ax_raw.transAxes, ha="left", va="top")

    covars = df[["RPS6_mrna", "RPS6_total_protein"]]
    rx = residualize(df[x], covars)
    ry = residualize(df[y], covars)
    rd = pd.DataFrame({"pred_resid": rx, "meas_resid": ry}).dropna()
    rr, rp = stats.spearmanr(rd["pred_resid"], rd["meas_resid"])
    ax_resid.scatter(rd["pred_resid"], rd["meas_resid"], s=28, color=BLUE, edgecolor=DARK, linewidth=0.35, alpha=0.9)
    slope, intercept = np.polyfit(rd["pred_resid"], rd["meas_resid"], 1)
    xs = np.linspace(rd["pred_resid"].min(), rd["pred_resid"].max(), 100)
    ax_resid.plot(xs, slope * xs + intercept, color=DARK, lw=1)
    ax_resid.axhline(0, color=GREY, lw=0.6)
    ax_resid.axvline(0, color=GREY, lw=0.6)
    ax_resid.set_xlabel("Predicted residual")
    ax_resid.set_ylabel("Measured residual")
    ax_resid.text(0.04, 0.96, f"Adjusted for RPS6 mRNA + total protein\npartial rho-like r={rr:.2f}, p={rp:.1e}", transform=ax_resid.transAxes, ha="left", va="top")

    partial = pd.read_csv(IN1 / "cptac_ccrcc_rps6_s235_s236_partial_correlations.tsv", sep="\t")
    bar = pd.DataFrame(
        {
            "label": ["raw", "-mRNA", "-mRNA/protein"],
            "rho": [
                rho,
                partial.loc[partial["comparison"].eq("adjust_mrna"), "partial_spearman_like_r"].iloc[0],
                partial.loc[partial["comparison"].eq("adjust_mrna_total_protein"), "partial_spearman_like_r"].iloc[0],
            ],
        }
    )
    ax_bar.bar(bar["label"], bar["rho"], color=[GREY, PURPLE, BLUE], edgecolor=DARK, linewidth=0.4)
    ax_bar.set_ylim(0, 0.65)
    ax_bar.set_ylabel("Correlation")
    ax_bar.tick_params(axis="x", rotation=25)
    ax_bar.set_title("Abundance-adjusted consistency", loc="left", fontsize=8, fontweight="bold")


def draw_drug(ax):
    res = pd.read_csv(IN2 / "depmap_rcc_rps6_ps6_mtor_drug_correlations.tsv", sep="\t")
    keep = res[
        (res["feature"] == "predicted_rps6_s235_s236")
        & (res["group"] == "CCRCC_only")
        & (res["metric"] == "AUC")
        & (res["dataset"].isin(["REPURPOSING", "GDSC2"]))
    ].copy()
    keep["drug_label"] = keep["dataset"] + " " + keep["drug"].str.title()
    keep = keep.sort_values("spearman_rho")
    y = np.arange(len(keep))
    colors = [BLUE if v < 0 else GREY for v in keep["spearman_rho"]]
    ax.barh(y, keep["spearman_rho"], color=colors, edgecolor=DARK, linewidth=0.35)
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a} (n={int(n)})" for a, n in zip(keep["drug_label"], keep["n"])])
    ax.set_xlabel("Spearman correlation with AUC")
    ax.set_title("c  Drug sensitivity layer", loc="left", fontweight="bold", fontsize=10)
    ax.text(0.02, 0.04, "Lower AUC indicates higher sensitivity.\nNegative values support predicted pS6-high sensitivity.", transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
    return keep


def main():
    ensure_dirs()
    set_style()
    fig = plt.figure(figsize=(13.2, 7.2))
    gs = GridSpec(3, 5, figure=fig, width_ratios=[1.45, 1.25, 1.15, 1.15, 0.78], height_ratios=[1.0, 0.26, 1.0], wspace=0.78, hspace=0.55)
    ax_km = fig.add_subplot(gs[0, 0])
    ax_risk = fig.add_subplot(gs[1, 0], sharex=ax_km)
    ax_forest = fig.add_subplot(gs[0:2, 1])
    ax_raw = fig.add_subplot(gs[0:2, 2])
    ax_resid = fig.add_subplot(gs[0:2, 3])
    ax_bar = fig.add_subplot(gs[2, 3])
    ax_drug = fig.add_subplot(gs[2, 0:3])

    draw_km(ax_km, ax_risk)
    draw_forest(ax_forest)
    draw_measured(ax_raw, ax_resid, ax_bar)
    drug_keep = draw_drug(ax_drug)

    fig.suptitle("RPS6 pS235/S236 as an mRNA-independent, measured phosphosite anchor in ccRCC", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(FIG / "fig5_rps6_ps6_anchor_abc.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG / "fig5_rps6_ps6_anchor_abc.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig5_rps6_ps6_anchor_abc.svg", bbox_inches="tight")

    drug_keep.to_csv(TAB / "panel_c_plotted_drug_correlations.tsv", sep="\t", index=False)

    summary = {
        "figure": str(FIG / "fig5_rps6_ps6_anchor_abc.png"),
        "site": "RPS6|S235_S236",
        "tcga_kirc": json.load(open(IN1 / "tcga_kirc_rps6_s235_s236_summary.json")),
        "drug_panel_rows": drug_keep.to_dict(orient="records"),
    }
    with open(OUT / "reports" / "fig5_rps6_ps6_anchor_abc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
