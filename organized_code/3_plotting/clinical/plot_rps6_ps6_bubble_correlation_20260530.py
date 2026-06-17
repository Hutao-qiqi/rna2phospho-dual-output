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
CPTAC = ROOT / "02_results/model_validation/20260530_fig5_exact_site_anchor_search_v1/tables/cptac_ccrcc_rps6_s235_s236_sample_table.tsv.gz"
MTOR = TAB / "cptac_ccrcc_mtor_axis_phosphosite_features.tsv"


BLUE = "#92B1D9"
WARM = "#F6C8B6"
GREY = "#D4D4D4"
INK = "#222222"


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return (x - x.mean()) / x.std(ddof=0)


def residualize(y: pd.Series, cov: pd.DataFrame) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    cov = cov.apply(pd.to_numeric, errors="coerce")
    mat = pd.concat([y.rename("y"), cov], axis=1).dropna()
    out = pd.Series(np.nan, index=y.index, dtype=float)
    if mat.shape[0] < cov.shape[1] + 4:
        return out
    X = np.column_stack([np.ones(mat.shape[0]), mat[cov.columns].values])
    beta = np.linalg.lstsq(X, mat["y"].values, rcond=None)[0]
    out.loc[mat.index] = mat["y"].values - X @ beta
    return out


def partial_corr(df: pd.DataFrame, x: str, y: str, base_covars: list[str]) -> dict:
    if y in base_covars:
        d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        return {"r": 0.0, "p": np.nan, "n": int(d.shape[0]), "covars": ";".join(base_covars), "mode": "adjusted_out"}
    covars = [c for c in base_covars if c not in {x, y}]
    use_cols = [x, y] + covars
    d = df[use_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if d.shape[0] < max(8, len(covars) + 5):
        return {"r": np.nan, "p": np.nan, "n": int(d.shape[0]), "covars": ";".join(covars), "mode": "partial"}
    if covars:
        rx = residualize(d[x], d[covars])
        ry = residualize(d[y], d[covars])
    else:
        rx = d[x]
        ry = d[y]
    valid = pd.concat([rx.rename("x"), ry.rename("y")], axis=1).dropna()
    if valid.shape[0] < 8 or valid["x"].std(ddof=0) == 0 or valid["y"].std(ddof=0) == 0:
        return {"r": np.nan, "p": np.nan, "n": int(valid.shape[0]), "covars": ";".join(covars), "mode": "partial"}
    r, p = stats.pearsonr(valid["x"], valid["y"])
    return {"r": float(r), "p": float(p), "n": int(valid.shape[0]), "covars": ";".join(covars), "mode": "partial"}


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

    df = pd.read_csv(CPTAC, sep="\t")
    mtor = pd.read_csv(MTOR, sep="\t")
    df = df.merge(mtor, on="case_submitter_id", how="left")
    df["Proliferation score"] = pd.concat([zscore(df["MKI67_mrna"]), zscore(df["TOP2A_mrna"]), zscore(df["PCNA_mrna"])], axis=1).mean(axis=1)

    row_vars = [
        ("Predicted pS6", "predicted_rps6_s235_s236"),
        ("Measured pS6", "measured_rps6_s235_s236"),
    ]
    col_vars = [
        ("MS p-S6\nS235/S236", "MS p-S6 S235/S236"),
        ("MS p-S6K1\nT421/S424", "MS p-S6K1 T421/S424"),
        ("MS p-S6K1\nT389", "MS p-S6K1 T389"),
        ("MS p-4EBP1\nS65", "MS p-4EBP1 S65"),
        ("MS p-4EBP1\nT70", "MS p-4EBP1 T70"),
        ("MS p-mTOR\nS2481", "MS p-mTOR S2481"),
        ("RPS6 mRNA\nadjusted out", "RPS6_mrna"),
        ("RPS6 protein\nadjusted out", "RPS6_total_protein"),
        ("Proliferation\nscore", "Proliferation score"),
    ]
    base_covars = ["RPS6_mrna", "RPS6_total_protein"]

    records = []
    for rlab, rcol in row_vars:
        for clab, ccol in col_vars:
            stat = partial_corr(df, rcol, ccol, base_covars)
            records.append({"row_label": rlab, "row_variable": rcol, "column_label": clab.replace("\n", " "), "column_variable": ccol, **stat})
    res = pd.DataFrame(records)
    res.to_csv(TAB / "cptac_ccrcc_rps6_ps6_bubble_partial_correlations.tsv", sep="\t", index=False)
    df.to_csv(TAB / "cptac_ccrcc_rps6_ps6_bubble_input_matrix.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(7.8, 2.6))
    xs = np.arange(len(col_vars))
    ys = np.arange(len(row_vars))
    ax.set_xlim(-0.6, len(col_vars) - 0.4)
    ax.set_ylim(-0.55, len(row_vars) - 0.45)
    ax.invert_yaxis()
    ax.set_xticks(xs)
    ax.set_xticklabels([x[0] for x in col_vars], fontsize=8, rotation=45, ha="right")
    ax.set_yticks(ys)
    ax.set_yticklabels([x[0] for x in row_vars], fontsize=9)
    ax.set_title("Abundance-adjusted covariance structure of RPS6 pS235/S236 in CPTAC ccRCC", loc="left", fontsize=11.5, fontweight="bold", pad=8)
    ax.grid(color="#EFEFEF", lw=0.8)
    ax.set_axisbelow(True)

    cmap = mpl.colors.LinearSegmentedColormap.from_list("blue_warm", [BLUE, "#F7F7F7", WARM])
    norm = mpl.colors.TwoSlopeNorm(vmin=-0.75, vcenter=0, vmax=0.75)
    for _, rec in res.iterrows():
        xi = [v[1] for v in col_vars].index(rec["column_variable"])
        yi = [v[1] for v in row_vars].index(rec["row_variable"])
        r = rec["r"]
        if rec.get("mode", "") == "adjusted_out":
            ax.scatter(xi, yi, s=55, facecolor="white", edgecolor=GREY, linewidth=1.0)
            ax.text(xi, yi, "0", ha="center", va="center", fontsize=7, color="#777777")
            ax.text(xi, yi + 0.31, f"n={int(rec['n'])}", ha="center", va="center", fontsize=5.7, color="#666666")
            continue
        if not np.isfinite(r):
            ax.scatter(xi, yi, s=18, color=GREY, alpha=0.4, marker="x", lw=0.8)
            continue
        size = 70 + 830 * abs(r)
        ax.scatter(xi, yi, s=size, c=[cmap(norm(r))], edgecolor="white", linewidth=0.7)
        ax.text(xi, yi, f"{r:.2f}", ha="center", va="center", fontsize=7, color=INK)
        ax.text(xi, yi + 0.31, f"n={int(rec['n'])}", ha="center", va="center", fontsize=5.7, color="#666666")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    cax = fig.add_axes([0.86, 0.48, 0.012, 0.30])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label("partial r", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2)
    lax = fig.add_axes([0.84, 0.18, 0.10, 0.20])
    lax.axis("off")
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    for i, val in enumerate([0.2, 0.5, 0.8]):
        lax.scatter(0.25, 0.8 - i * 0.28, s=70 + 830 * val, color=GREY, edgecolor="white")
        lax.text(0.55, 0.8 - i * 0.28, f"|r|={val:.1f}", va="center", fontsize=6.5)
    fig.text(0.02, -0.02, "Partial correlations adjust for RPS6 mRNA and RPS6 total protein; abundance controls are shown as adjusted-out axes.", fontsize=7.2, color="#555555")
    fig.tight_layout(rect=[0.01, 0.04, 0.84, 0.95])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig5_rps6_ps6_bubble_partial_correlation.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input_rows": int(df.shape[0]),
        "row_variables": [x[0] for x in row_vars],
        "column_variables": [x[0].replace("\n", " ") for x in col_vars],
        "covariates": base_covars,
        "note": "RPS6 S240/S244 was not available in the CPTAC ccRCC phosphosite matrix used here.",
    }
    (REPORT / "fig5_rps6_ps6_bubble_partial_correlation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
