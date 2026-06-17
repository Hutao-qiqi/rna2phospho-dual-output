#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(r"E:\data\gongke\TCGA-TCPA\paper_materials_SCP682")
OUT = ROOT / "04_figure_source_data" / "fig2_extensions"
OUT.mkdir(parents=True, exist_ok=True)

PERF_PATH = ROOT / "01_key_results" / "per_site_spearman_with_deep_learning.tsv"
OBS_PATH = ROOT / "02_data_tables" / "oof_branch_predictions" / "observed_phosphosite.parquet"
TARGET_META_PATH = ROOT / "02_data_tables" / "oof_branch_predictions" / "phosphosite_target_manifest.tsv"
RNA_EXPR_PATH = OUT / "parent_rna_mean_expression.tsv"

METHOD_MAP = {
    "parent_mRNA_linear": "Cognate mRNA",
    "MLP": "MLP",
    "VAE": "VAE",
    "DeepGxP_5fold": "DeepGxP",
    "SCP682": "SCP682",
}
METHOD_ORDER = ["Cognate mRNA", "DeepGxP", "MLP", "VAE", "SCP682"]
COLORS = {
    "Cognate mRNA": "#A8A8A8",
    "DeepGxP": "#D4D4D4",
    "MLP": "#92B1D9",
    "VAE": "#C1D8E9",
    "SCP682": "#1F3A5F",
}
BIN_ORDER = ["Low", "Middle", "High"]


def label_tertiles(values, low_q=0.25, high_q=0.75):
    lo = np.nanquantile(values, low_q)
    hi = np.nanquantile(values, high_q)
    out = np.full(len(values), "Middle", dtype=object)
    out[values <= lo] = "Low"
    out[values >= hi] = "High"
    return out, lo, hi


def add_panel_label(ax, label):
    ax.text(-0.11, 1.04, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold", family="Arial")


def grouped_boxplot(ax, df, group_col, ylabel, title):
    width = 0.13
    positions = np.arange(len(BIN_ORDER))
    offsets = np.linspace(-0.28, 0.28, len(METHOD_ORDER))
    for i, method in enumerate(METHOD_ORDER):
        data = []
        pos = []
        for j, group in enumerate(BIN_ORDER):
            vals = df[(df[group_col] == group) & (df["method_label"] == method)]["spearman"].dropna().to_numpy()
            data.append(vals)
            pos.append(positions[j] + offsets[i])
        bp = ax.boxplot(
            data,
            positions=pos,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#1A1A1A", "linewidth": 0.7},
            boxprops={"linewidth": 0.6, "color": "#555555"},
            whiskerprops={"linewidth": 0.55, "color": "#555555"},
            capprops={"linewidth": 0.55, "color": "#555555"},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.95)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Low", "Middle", "High"], fontsize=7, family="Arial")
    ax.set_ylabel(ylabel, fontsize=7, family="Arial")
    ax.set_title(title, fontsize=7.5, family="Arial", pad=5)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def scatter_with_bins(ax, df, x_col, x_label, title, panel_letter):
    sub = df[df["method_label"].eq("SCP682")].dropna(subset=[x_col, "spearman"]).copy()
    ax.scatter(sub[x_col], sub["spearman"], s=6, color="#1F3A5F", alpha=0.22, linewidths=0, rasterized=True)
    bins = pd.qcut(sub[x_col].rank(method="first"), 10, labels=False, duplicates="drop")
    binned = sub.assign(bin=bins).groupby("bin", as_index=False).agg(
        x_median=(x_col, "median"),
        y_median=("spearman", "median"),
        y_q25=("spearman", lambda x: np.nanquantile(x, 0.25)),
        y_q75=("spearman", lambda x: np.nanquantile(x, 0.75)),
        n=("spearman", "size"),
    )
    ax.plot(binned["x_median"], binned["y_median"], color="#D4A56B", linewidth=1.2)
    ax.fill_between(binned["x_median"], binned["y_q25"], binned["y_q75"], color="#D4A56B", alpha=0.18, linewidth=0)
    rho = spearmanr(sub[x_col], sub["spearman"], nan_policy="omit").statistic
    ax.text(0.03, 0.92, f"ρ = {rho:.2f}", transform=ax.transAxes, fontsize=7, family="Arial")
    ax.set_xlabel(x_label, fontsize=7, family="Arial")
    ax.set_ylabel("SCP682 per-site Spearman ρ", fontsize=7, family="Arial")
    ax.set_title(title, fontsize=7.5, family="Arial", pad=5)
    ax.grid(color="#E6E6E6", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_label(ax, panel_letter)
    return binned.assign(x_col=x_col)


def main():
    perf = pd.read_csv(PERF_PATH, sep="\t")
    perf = perf[perf["dataset"].eq("CPTAC_all") & perf["method"].isin(METHOD_MAP)].copy()
    perf["method_label"] = perf["method"].map(METHOD_MAP)

    obs = pd.read_parquet(OBS_PATH)
    coverage = pd.DataFrame({
        "target": obs.columns.astype(str),
        "coverage_count": obs.notna().sum(axis=0).astype(int).to_numpy(),
        "coverage_rate": obs.notna().mean(axis=0).astype(float).to_numpy(),
    })

    meta = pd.read_csv(TARGET_META_PATH, sep="\t")
    meta = meta.rename(columns={"scp682_site_id": "target"})
    meta["target"] = meta["target"].astype(str)
    meta["parent_gene"] = meta["parent_gene"].astype(str)

    expr = pd.read_csv(RNA_EXPR_PATH, sep="\t")
    expr["gene"] = expr["gene"].astype(str)

    site_annot = (
        meta[["target", "parent_gene", "residue", "position"]]
        .merge(coverage, on="target", how="left")
        .merge(expr.rename(columns={"gene": "parent_gene"}), on="parent_gene", how="left")
    )
    site_annot = site_annot[site_annot["target"].isin(perf["target"])].copy()

    cov_bin, cov_lo, cov_hi = label_tertiles(site_annot["coverage_rate"].to_numpy(dtype=float))
    site_annot["coverage_bin"] = cov_bin
    expr_values = site_annot["mean_log2_tpm"].to_numpy(dtype=float)
    expr_bin, expr_lo, expr_hi = label_tertiles(expr_values)
    site_annot["expression_bin"] = expr_bin

    merged = perf.merge(site_annot, on="target", how="left")
    merged.to_csv(OUT / "coverage_expression_strata_per_site.tsv", sep="\t", index=False)

    summary_rows = []
    for strat_col, strat_name in [("coverage_bin", "phosphosite coverage"), ("expression_bin", "parent RNA expression")]:
        for method in METHOD_ORDER:
            for group in BIN_ORDER:
                vals = merged[(merged[strat_col].eq(group)) & (merged["method_label"].eq(method))]["spearman"].dropna()
                summary_rows.append({
                    "stratification": strat_name,
                    "bin": group,
                    "method_label": method,
                    "n_sites": int(vals.shape[0]),
                    "median_spearman": float(vals.median()) if len(vals) else np.nan,
                    "q25_spearman": float(vals.quantile(0.25)) if len(vals) else np.nan,
                    "q75_spearman": float(vals.quantile(0.75)) if len(vals) else np.nan,
                })
    pd.DataFrame(summary_rows).to_csv(OUT / "coverage_expression_strata_summary.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.4), constrained_layout=False)
    grouped_boxplot(
        axes[0, 0],
        merged,
        "coverage_bin",
        "Per-site Spearman ρ",
        "Phosphosite coverage strata",
    )
    add_panel_label(axes[0, 0], "a")
    grouped_boxplot(
        axes[0, 1],
        merged,
        "expression_bin",
        "Per-site Spearman ρ",
        "Parent RNA expression strata",
    )
    add_panel_label(axes[0, 1], "b")
    b1 = scatter_with_bins(
        axes[1, 0],
        merged,
        "coverage_rate",
        "Phosphosite observed fraction",
        "SCP682 performance vs coverage",
        "c",
    )
    b2 = scatter_with_bins(
        axes[1, 1],
        merged,
        "mean_log2_tpm",
        "Parent gene mean log2(TPM + 1)",
        "SCP682 performance vs parent RNA expression",
        "d",
    )
    pd.concat([b1, b2], ignore_index=True).to_csv(OUT / "coverage_expression_binned_scatter.tsv", sep="\t", index=False)

    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS[m], markeredgecolor="#555555",
               markersize=6, label=m)
        for m in METHOD_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(METHOD_ORDER),
        frameon=False,
        bbox_to_anchor=(0.52, 1.00),
        prop={"family": "Arial", "size": 7},
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.90, wspace=0.26, hspace=0.34)
    for ext in ["pdf", "svg", "png"]:
        path = OUT / f"fig2_ext_coverage_expression_strata.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=300)
        else:
            fig.savefig(path)
    plt.close(fig)

    md = f"""# Coverage and expression strata panels

本图按 phosphosite 覆盖率和 parent RNA 表达量分层评估模型性能。

数据口径：

- 性能来自 `01_key_results/per_site_spearman_with_deep_learning.tsv` 的 `CPTAC_all`。
- 覆盖率来自 `observed_phosphosite.parquet`，定义为每个位点在 1,431 个训练样本中的非缺失比例。
- parent RNA 表达来自远端训练 RNA 矩阵 `rna_log2_tpm_paired.parquet`，这里只保存每个基因的均值、中央値和检测比例。
- Low/Middle/High 分别表示 Q1、Q2-Q3、Q4。覆盖率阈值：Low ≤ {cov_lo:.4f}，High ≥ {cov_hi:.4f}。parent RNA 表达阈值：Low ≤ {expr_lo:.4f}，High ≥ {expr_hi:.4f}。

输出：

- `fig2_ext_coverage_expression_strata.pdf/svg/png`
- `coverage_expression_strata_per_site.tsv`
- `coverage_expression_strata_summary.tsv`
- `coverage_expression_binned_scatter.tsv`
"""
    (OUT / "coverage_expression_strata.md").write_text(md, encoding="utf-8")

    print(pd.read_csv(OUT / "coverage_expression_strata_summary.tsv", sep="\t").to_string(index=False))


if __name__ == "__main__":
    main()
