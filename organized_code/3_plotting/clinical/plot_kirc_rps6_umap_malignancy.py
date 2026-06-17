import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def clean_axes(ax):
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_all(fig, out_prefix: Path):
    fig.savefig(out_prefix.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def status_label(x):
    s = str(x)
    if "malignant" in s and "non_malignant" not in s:
        return "Malignant inferred"
    if "non_malignant" in s:
        return "Non-malignant inferred"
    return s


def group_stats(df, scope_name):
    rows = []
    x = df["predicted_RPS6_pS235_S236"].astype(float)
    is_mal = df["malignant_status"].astype(str).str.contains("malignant", na=False) & ~df[
        "malignant_status"
    ].astype(str).str.contains("non_malignant", na=False)
    for flag, label in [(False, "Non-malignant inferred"), (True, "Malignant inferred")]:
        vals = x[is_mal == flag].dropna()
        rows.append(
            {
                "scope": scope_name,
                "group": label,
                "n_cells": int(vals.shape[0]),
                "mean_predicted_RPS6_pS235_S236": float(vals.mean()),
                "median_predicted_RPS6_pS235_S236": float(vals.median()),
                "sd_predicted_RPS6_pS235_S236": float(vals.std(ddof=1)),
            }
        )
    a = x[is_mal].dropna()
    b = x[~is_mal].dropna()
    common = df[["predicted_RPS6_pS235_S236"]].copy()
    common["is_malignant"] = is_mal.astype(int)
    common = common.dropna()
    if len(a) > 2 and len(b) > 2:
        spearman = stats.spearmanr(common["is_malignant"], common["predicted_RPS6_pS235_S236"])
        pearson = stats.pearsonr(common["is_malignant"], common["predicted_RPS6_pS235_S236"])
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        welch = stats.ttest_ind(a, b, equal_var=False)
        rows.append(
            {
                "scope": scope_name,
                "group": "Malignant vs non-malignant",
                "n_cells": int(common.shape[0]),
                "mean_predicted_RPS6_pS235_S236": float(a.mean() - b.mean()),
                "median_predicted_RPS6_pS235_S236": float(a.median() - b.median()),
                "sd_predicted_RPS6_pS235_S236": np.nan,
                "spearman_r_malignant_binary": float(spearman.statistic),
                "spearman_p_malignant_binary": float(spearman.pvalue),
                "pearson_r_malignant_binary": float(pearson.statistic),
                "pearson_p_malignant_binary": float(pearson.pvalue),
                "mannwhitney_p": float(mw.pvalue),
                "welch_p": float(welch.pvalue),
            }
        )
    return rows


def continuous_corr(df, scope_name, variable):
    sub = df[["predicted_RPS6_pS235_S236", variable]].dropna()
    if sub.shape[0] < 3:
        return None
    sp = stats.spearmanr(sub[variable], sub["predicted_RPS6_pS235_S236"])
    pe = stats.pearsonr(sub[variable], sub["predicted_RPS6_pS235_S236"])
    return {
        "scope": scope_name,
        "variable": variable,
        "n_cells": int(sub.shape[0]),
        "spearman_r": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "pearson_r": float(pe.statistic),
        "pearson_p": float(pe.pvalue),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        default=r"D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260531_scp682_sc_kirc_rps6_validation_v1",
    )
    parser.add_argument("--input-table", default=None)
    parser.add_argument("--fig-dir", default=None)
    parser.add_argument("--font-family", default="Arial")
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": args.font_family,
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    result_dir = Path(args.result_dir)
    table = Path(args.input_table) if args.input_table else result_dir / "tables" / "kirc_cell_rps6_prediction.tsv"
    fig_dir = Path(args.fig_dir) if args.fig_dir else result_dir / "figures" / "umap_malignancy_rps6"
    table_dir = table.parent
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(table, sep="\t")
    df = df.dropna(subset=["UMAP1", "UMAP2", "predicted_RPS6_pS235_S236", "malignant_status"]).copy()
    df["malignant_label"] = df["malignant_status"].map(status_label)
    df["is_malignant"] = (
        df["malignant_status"].astype(str).str.contains("malignant", na=False)
        & ~df["malignant_status"].astype(str).str.contains("non_malignant", na=False)
    )

    rng = np.random.default_rng(7)
    df_plot = df.iloc[rng.permutation(len(df))].copy()

    colors = {
        "Non-malignant inferred": "#8DA0CB",
        "Malignant inferred": "#E64B35",
    }
    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    for lab in ["Non-malignant inferred", "Malignant inferred"]:
        sub = df_plot[df_plot["malignant_label"] == lab]
        ax.scatter(sub["UMAP1"], sub["UMAP2"], s=2.0, c=colors.get(lab, "#999999"), alpha=0.62, linewidths=0, label=f"{lab} (n={len(sub):,})")
    ax.set_title("KIRC malignant-cell annotation")
    ax.legend(frameon=False, loc="best", markerscale=4)
    clean_axes(ax)
    save_all(fig, fig_dir / "kirc_umap_malignant_status")

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    vals = df_plot["predicted_RPS6_pS235_S236"]
    lo, hi = np.nanpercentile(vals, [1, 99])
    sc = ax.scatter(
        df_plot["UMAP1"],
        df_plot["UMAP2"],
        s=2.0,
        c=vals,
        cmap="viridis",
        vmin=lo,
        vmax=hi,
        alpha=0.72,
        linewidths=0,
    )
    ax.set_title("Predicted RPS6 pS235/S236")
    clean_axes(ax)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.045)
    cb.set_label("Predicted RPS6 pS235/S236")
    save_all(fig, fig_dir / "kirc_umap_predicted_rps6_ps235_s236")

    mal = df_plot[df_plot["is_malignant"]].copy()
    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    vals = mal["predicted_RPS6_pS235_S236"]
    lo, hi = np.nanpercentile(vals, [1, 99])
    sc = ax.scatter(
        mal["UMAP1"],
        mal["UMAP2"],
        s=3.0,
        c=vals,
        cmap="magma",
        vmin=lo,
        vmax=hi,
        alpha=0.78,
        linewidths=0,
    )
    ax.set_title("Predicted RPS6 pS235/S236 in malignant cells")
    clean_axes(ax)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.045)
    cb.set_label("Predicted RPS6 pS235/S236")
    save_all(fig, fig_dir / "kirc_umap_malignant_only_predicted_rps6_ps235_s236")

    state_cols = [
        ("interferon_stress_score", "Interferon/stress score", "plasma"),
        ("angiogenesis_score", "Angiogenesis score", "viridis"),
        ("EMT_score", "EMT score", "cividis"),
    ]
    for col, label, cmap in state_cols:
        if col not in mal.columns:
            continue
        fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
        vals = mal[col].astype(float)
        lo, hi = np.nanpercentile(vals, [1, 99])
        sc = ax.scatter(
            mal["UMAP1"],
            mal["UMAP2"],
            s=3.0,
            c=vals,
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            alpha=0.78,
            linewidths=0,
        )
        ax.set_title(f"{label} in malignant cells")
        clean_axes(ax)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.045)
        cb.set_label(label)
        save_all(fig, fig_dir / f"kirc_umap_malignant_only_{col}")

    present_state_cols = [(c, l, cm) for c, l, cm in state_cols if c in mal.columns]
    if present_state_cols:
        n = len(present_state_cols)
        fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 5.0), squeeze=False, constrained_layout=True)
        for ax, (col, label, cmap) in zip(axes[0], present_state_cols):
            vals = mal[col].astype(float)
            lo, hi = np.nanpercentile(vals, [1, 99])
            sc = ax.scatter(
                mal["UMAP1"],
                mal["UMAP2"],
                s=2.8,
                c=vals,
                cmap=cmap,
                vmin=lo,
                vmax=hi,
                alpha=0.78,
                linewidths=0,
            )
            ax.set_title(label)
            clean_axes(ax)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.045)
            cb.set_label(label)
        save_all(fig, fig_dir / "kirc_umap_malignant_only_top3_cell_states")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    for lab in ["Non-malignant inferred", "Malignant inferred"]:
        sub = df_plot[df_plot["malignant_label"] == lab]
        axes[0].scatter(sub["UMAP1"], sub["UMAP2"], s=1.8, c=colors.get(lab, "#999999"), alpha=0.6, linewidths=0, label=lab)
    axes[0].set_title("Malignant annotation")
    axes[0].legend(frameon=False, loc="best", markerscale=4)
    clean_axes(axes[0])
    vals = df_plot["predicted_RPS6_pS235_S236"]
    lo, hi = np.nanpercentile(vals, [1, 99])
    sc = axes[1].scatter(df_plot["UMAP1"], df_plot["UMAP2"], s=1.8, c=vals, cmap="viridis", vmin=lo, vmax=hi, alpha=0.72, linewidths=0)
    axes[1].set_title("Predicted RPS6 pS235/S236")
    clean_axes(axes[1])
    cb = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.045)
    cb.set_label("Predicted RPS6 pS235/S236")
    save_all(fig, fig_dir / "kirc_umap_malignant_status_and_rps6")

    stat_rows = []
    stat_rows.extend(group_stats(df, "All cells"))
    if "tissue" in df.columns:
        stat_rows.extend(group_stats(df[df["tissue"].astype(str).eq("Tumor")].copy(), "Tumor tissue cells"))
    pd.DataFrame(stat_rows).to_csv(table_dir / "kirc_rps6_malignancy_umap_group_stats.tsv", sep="\t", index=False)

    corr_rows = []
    for scope, sub in [
        ("All cells", df),
        ("Tumor tissue cells", df[df["tissue"].astype(str).eq("Tumor")].copy() if "tissue" in df.columns else df.iloc[0:0]),
        ("Malignant inferred cells", df[df["is_malignant"]].copy()),
        ("Non-malignant inferred cells", df[~df["is_malignant"]].copy()),
    ]:
        for var in ["RPS6_mRNA", "mTOR_S6_score", "cell_cycle_score", "hypoxia_score", "angiogenesis_score"]:
            if var in sub.columns:
                row = continuous_corr(sub, scope, var)
                if row:
                    corr_rows.append(row)
    pd.DataFrame(corr_rows).to_csv(table_dir / "kirc_rps6_umap_continuous_correlations.tsv", sep="\t", index=False)

    print("done")
    print(f"n_cells={len(df)} n_malignant={int(df['is_malignant'].sum())} n_non_malignant={int((~df['is_malignant']).sum())}")
    print(fig_dir)


if __name__ == "__main__":
    main()
