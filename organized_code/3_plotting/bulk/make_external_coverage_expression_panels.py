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

PERF_PATH = ROOT / "01_key_results" / "per_site_spearman_external.tsv"
COHORT_SUMMARY_PATH = ROOT / "01_key_results" / "external_validation" / "per_cohort_summary.tsv"
TARGET_META_PATH = ROOT / "02_data_tables" / "oof_branch_predictions" / "phosphosite_target_manifest.tsv"
EXTERNAL_RNA_EXPR_PATH = OUT / "external_parent_rna_expression.tsv"

METHOD_MAP = {
    "parent_mRNA_linear": "Cognate mRNA",
    "masked_ridge_linear": "Ridge",
    "masked_elasticnet_linear": "Elastic net",
    "PCA_ridge": "PC ridge",
    "MLP": "MLP",
    "SCP682": "SCP682",
}
METHOD_ORDER = ["Cognate mRNA", "Ridge", "Elastic net", "PC ridge", "MLP", "SCP682"]
COLORS = {
    "Cognate mRNA": "#A8A8A8",
    "Ridge": "#D4D4D4",
    "Elastic net": "#DBDDEF",
    "PC ridge": "#C1D8E9",
    "MLP": "#92B1D9",
    "SCP682": "#1F3A5F",
}
DATASET_ORDER = ["fu_icca", "tu_sclc", "chcc_hbv_fpkm", "chcc_hbv_rsem"]
DATASET_LABEL = {
    "fu_icca": "FU-iCCA",
    "tu_sclc": "TU-SCLC",
    "chcc_hbv_fpkm": "CHCC-HBV FPKM",
    "chcc_hbv_rsem": "CHCC-HBV RSEM",
}
BIN_ORDER = ["Low", "Middle", "High"]


def label_quartile_strata(values):
    arr = np.asarray(values, dtype=float)
    lo = np.nanquantile(arr, 0.25)
    hi = np.nanquantile(arr, 0.75)
    out = np.full(len(arr), "Middle", dtype=object)
    out[arr <= lo] = "Low"
    out[arr >= hi] = "High"
    return out, lo, hi


def add_panel_label(ax, label):
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold", family="Arial")


def grouped_boxplot(ax, df, group_col):
    width = 0.11
    base = np.arange(len(BIN_ORDER))
    offsets = np.linspace(-0.30, 0.30, len(METHOD_ORDER))
    for i, method in enumerate(METHOD_ORDER):
        data = []
        pos = []
        for j, group in enumerate(BIN_ORDER):
            vals = df[(df[group_col].eq(group)) & (df["method_label"].eq(method))]["spearman"].dropna().to_numpy()
            data.append(vals)
            pos.append(base[j] + offsets[i])
        bp = ax.boxplot(
            data,
            positions=pos,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#1A1A1A", "linewidth": 0.65},
            boxprops={"linewidth": 0.55, "color": "#555555"},
            whiskerprops={"linewidth": 0.5, "color": "#555555"},
            capprops={"linewidth": 0.5, "color": "#555555"},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.95)
    ax.set_xticks(base)
    ax.set_xticklabels(BIN_ORDER, fontsize=6.5, family="Arial")
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def scatter_panel(ax, df, x_col, x_label):
    sub = df[df["method_label"].eq("SCP682")].dropna(subset=[x_col, "spearman"]).copy()
    ax.scatter(sub[x_col], sub["spearman"], s=5, color="#1F3A5F", alpha=0.24, linewidths=0, rasterized=True)
    if len(sub) >= 20:
        bins = pd.qcut(sub[x_col].rank(method="first"), 8, labels=False, duplicates="drop")
        binned = sub.assign(bin=bins).groupby("bin", as_index=False).agg(
            x_median=(x_col, "median"),
            y_median=("spearman", "median"),
            y_q25=("spearman", lambda x: np.nanquantile(x, 0.25)),
            y_q75=("spearman", lambda x: np.nanquantile(x, 0.75)),
        )
        ax.plot(binned["x_median"], binned["y_median"], color="#D4A56B", linewidth=1.0)
        ax.fill_between(binned["x_median"], binned["y_q25"], binned["y_q75"], color="#D4A56B", alpha=0.18, linewidth=0)
        rho = spearmanr(sub[x_col], sub["spearman"], nan_policy="omit").statistic
        ax.text(0.04, 0.88, f"ρ = {rho:.2f}", transform=ax.transAxes, fontsize=6.5, family="Arial")
    ax.set_xlabel(x_label, fontsize=6.5, family="Arial")
    ax.tick_params(axis="both", labelsize=6.5)
    ax.grid(color="#E6E6E6", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_external_strata():
    perf = pd.read_csv(PERF_PATH, sep="\t")
    perf = perf[perf["dataset"].isin(DATASET_ORDER) & perf["method"].isin(METHOD_MAP)].copy()
    perf["spearman"] = pd.to_numeric(perf["spearman"], errors="coerce")
    perf["n_samples_used"] = pd.to_numeric(perf["n_samples_used"], errors="coerce")
    perf["method_label"] = perf["method"].map(METHOD_MAP)

    cohort = pd.read_csv(COHORT_SUMMARY_PATH, sep="\t")
    cohort = cohort[cohort["model"].eq("scp682_general_graph_residual_sample_centered")].copy()
    denom = cohort.set_index("dataset")["n_matched_samples"].astype(float).to_dict()

    meta = pd.read_csv(TARGET_META_PATH, sep="\t")
    meta = meta.rename(columns={"scp682_site_id": "target"})
    meta["target"] = meta["target"].astype(str)
    meta["parent_gene"] = meta["parent_gene"].astype(str)
    meta["parent_gene_upper"] = meta["parent_gene"].str.upper()

    expr = pd.read_csv(EXTERNAL_RNA_EXPR_PATH, sep="\t")
    expr["dataset"] = expr["dataset"].astype(str)
    expr["gene_upper"] = expr["gene"].astype(str).str.upper()
    expr = expr.groupby(["dataset", "gene_upper"], as_index=False).agg(
        external_parent_mean_expression=("mean_expression", "mean"),
        external_parent_median_expression=("median_expression", "mean"),
        external_parent_detected_fraction=("detected_fraction", "mean"),
    )

    scp = perf[perf["method"].eq("SCP682")][["dataset", "target", "n_samples_used"]].drop_duplicates()
    scp["external_coverage_rate"] = scp.apply(
        lambda r: float(r["n_samples_used"]) / float(denom.get(r["dataset"], np.nan)), axis=1
    )
    annot = (
        scp.merge(meta[["target", "parent_gene", "parent_gene_upper", "residue", "position"]], on="target", how="left")
        .merge(expr, left_on=["dataset", "parent_gene_upper"], right_on=["dataset", "gene_upper"], how="left")
    )

    thresholds = []
    parts = []
    for ds in DATASET_ORDER:
        sub = annot[annot["dataset"].eq(ds)].copy()
        cov_bin, cov_lo, cov_hi = label_quartile_strata(sub["external_coverage_rate"].to_numpy(dtype=float))
        sub["external_coverage_bin"] = cov_bin
        expr_vals = sub["external_parent_mean_expression"].to_numpy(dtype=float)
        expr_bin, expr_lo, expr_hi = label_quartile_strata(expr_vals)
        sub["external_expression_bin"] = expr_bin
        thresholds.append({
            "dataset": ds,
            "dataset_label": DATASET_LABEL[ds],
            "n_matched_samples": int(denom[ds]),
            "coverage_q25": cov_lo,
            "coverage_q75": cov_hi,
            "parent_expression_q25": expr_lo,
            "parent_expression_q75": expr_hi,
            "parent_expression_match_rate": float(sub["external_parent_mean_expression"].notna().mean()),
        })
        parts.append(sub)
    annot = pd.concat(parts, ignore_index=True)

    merged = perf.merge(annot, on=["dataset", "target"], how="left")
    merged.to_csv(OUT / "external_coverage_expression_strata_per_site.tsv", sep="\t", index=False)
    pd.DataFrame(thresholds).to_csv(OUT / "external_coverage_expression_thresholds.tsv", sep="\t", index=False)

    summary = []
    for strat_col, strat_name in [
        ("external_coverage_bin", "external phosphosite coverage"),
        ("external_expression_bin", "external parent RNA expression"),
    ]:
        for ds in DATASET_ORDER:
            for method in METHOD_ORDER:
                for group in BIN_ORDER:
                    vals = merged[
                        merged["dataset"].eq(ds)
                        & merged["method_label"].eq(method)
                        & merged[strat_col].eq(group)
                    ]["spearman"].dropna()
                    summary.append({
                        "dataset": ds,
                        "dataset_label": DATASET_LABEL[ds],
                        "stratification": strat_name,
                        "bin": group,
                        "method_label": method,
                        "n_sites": int(len(vals)),
                        "median_spearman": float(vals.median()) if len(vals) else np.nan,
                        "q25_spearman": float(vals.quantile(0.25)) if len(vals) else np.nan,
                        "q75_spearman": float(vals.quantile(0.75)) if len(vals) else np.nan,
                    })
    pd.DataFrame(summary).to_csv(OUT / "external_coverage_expression_strata_summary.tsv", sep="\t", index=False)
    return merged, pd.DataFrame(thresholds), pd.DataFrame(summary)


def plot_boxplots(merged):
    fig, axes = plt.subplots(4, 2, figsize=(7.1, 8.7), sharey=True)
    letters = list("abcdefgh")
    k = 0
    for i, ds in enumerate(DATASET_ORDER):
        for j, (col, title) in enumerate([
            ("external_coverage_bin", "coverage strata"),
            ("external_expression_bin", "parent RNA expression strata"),
        ]):
            ax = axes[i, j]
            sub = merged[merged["dataset"].eq(ds)]
            grouped_boxplot(ax, sub, col)
            ax.set_title(f"{DATASET_LABEL[ds]}: {title}", fontsize=7.2, family="Arial", pad=4)
            if j == 0:
                ax.set_ylabel("Per-site Spearman ρ", fontsize=6.5, family="Arial")
            add_panel_label(ax, letters[k])
            k += 1
    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS[m], markeredgecolor="#555555",
               markersize=5.2, label=m)
        for m in METHOD_ORDER
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.52, 0.998), prop={"family": "Arial", "size": 6.5},
               handletextpad=0.35, columnspacing=0.9)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.93, wspace=0.16, hspace=0.44)
    for ext in ["pdf", "svg", "png"]:
        path = OUT / f"fig2_ext_external_coverage_expression_boxplots.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
    plt.close(fig)


def plot_scatter(merged):
    fig, axes = plt.subplots(4, 2, figsize=(7.1, 8.7), sharey=True)
    letters = list("abcdefgh")
    k = 0
    for i, ds in enumerate(DATASET_ORDER):
        sub = merged[merged["dataset"].eq(ds)]
        ax = axes[i, 0]
        scatter_panel(ax, sub, "external_coverage_rate", "External observed fraction")
        ax.set_title(f"{DATASET_LABEL[ds]}: coverage", fontsize=7.2, family="Arial", pad=4)
        ax.set_ylabel("SCP682 per-site Spearman ρ", fontsize=6.5, family="Arial")
        add_panel_label(ax, letters[k]); k += 1
        ax = axes[i, 1]
        scatter_panel(ax, sub, "external_parent_mean_expression", "Parent gene mean expression")
        ax.set_title(f"{DATASET_LABEL[ds]}: parent RNA expression", fontsize=7.2, family="Arial", pad=4)
        add_panel_label(ax, letters[k]); k += 1
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.96, wspace=0.20, hspace=0.46)
    for ext in ["pdf", "svg", "png"]:
        path = OUT / f"fig2_ext_external_coverage_expression_scatter.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
    plt.close(fig)


def main():
    merged, thresholds, summary = build_external_strata()
    plot_boxplots(merged)
    plot_scatter(merged)
    scp_summary = summary[summary["method_label"].eq("SCP682")].copy()
    md = "# External coverage and expression strata panels\n\n"
    md += "本图按四个外部队列分别定义 phosphosite 覆盖率和 parent RNA 表达量分层。\n\n"
    md += "覆盖率定义：该外部队列中每个位点有真实 phosphosite 实测值的样本数 / 该队列 matched samples。\n\n"
    md += "输出：\n\n"
    md += "- `fig2_ext_external_coverage_expression_boxplots.pdf/svg/png`\n"
    md += "- `fig2_ext_external_coverage_expression_scatter.pdf/svg/png`\n"
    md += "- `external_coverage_expression_strata_per_site.tsv`\n"
    md += "- `external_coverage_expression_strata_summary.tsv`\n"
    md += "- `external_coverage_expression_thresholds.tsv`\n\n"
    md += "SCP682 分层中位 Spearman：\n\n"
    for _, r in scp_summary.iterrows():
        md += f"- {r['dataset_label']} / {r['stratification']} / {r['bin']}: {r['median_spearman']:.4f} (n={int(r['n_sites'])})\n"
    (OUT / "external_coverage_expression_strata.md").write_text(md, encoding="utf-8")
    print(thresholds.to_string(index=False))
    print(scp_summary.to_string(index=False))


if __name__ == "__main__":
    main()
