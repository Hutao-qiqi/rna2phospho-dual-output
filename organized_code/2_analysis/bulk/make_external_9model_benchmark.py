#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\data\gongke\TCGA-TCPA\paper_materials_SCP682")
OUT = ROOT / "04_figure_source_data" / "fig2_extensions"
KEY = ROOT / "01_key_results"

BASE_EXTERNAL = KEY / "per_site_spearman_external.tsv"
DEEP_EXTERNAL = KEY / "per_site_spearman_external_deep_methods.tsv"

METHOD_RENAME = {
    "mean_pred": "Mean",
    "DeepGxP_5fold": "DeepGxP_5fold",
    "parent_mRNA_linear": "Cognate mRNA",
    "masked_ridge_linear": "Ridge",
    "PCA_ridge": "PC ridge",
    "VAE": "VAE",
    "MLP": "MLP",
    "masked_elasticnet_linear": "Elastic net",
    "SCP682": "SCP682",
}
METHOD_ORDER = [
    "Mean",
    "DeepGxP_5fold",
    "Cognate mRNA",
    "Ridge",
    "PC ridge",
    "VAE",
    "MLP",
    "Elastic net",
    "SCP682",
]
DATASET_ORDER = ["fu_icca", "tu_sclc", "chcc_hbv_fpkm", "chcc_hbv_rsem"]
DATASET_LABEL = {
    "fu_icca": "FU-iCCA",
    "tu_sclc": "TU-SCLC",
    "chcc_hbv_fpkm": "CHCC-HBV FPKM",
    "chcc_hbv_rsem": "CHCC-HBV RSEM",
}
COLORS = {
    "Mean": "#D4D4D4",
    "DeepGxP_5fold": "#BDBDBD",
    "Cognate mRNA": "#A8A8A8",
    "Ridge": "#DBDDEF",
    "PC ridge": "#C1D8E9",
    "VAE": "#B7D9D8",
    "MLP": "#92B1D9",
    "Elastic net": "#D7C2DF",
    "SCP682": "#1F3A5F",
}


def build_tables():
    base = pd.read_csv(BASE_EXTERNAL, sep="\t")
    deep = pd.read_csv(DEEP_EXTERNAL, sep="\t")
    df = pd.concat([base, deep], ignore_index=True)
    df = df[df["method"].isin(METHOD_RENAME) & df["dataset"].isin(DATASET_ORDER)].copy()
    df["method_label"] = df["method"].map(METHOD_RENAME)
    df["spearman"] = pd.to_numeric(df["spearman"], errors="coerce")
    df["rho_p_value"] = pd.to_numeric(df["rho_p_value"], errors="coerce")
    df["n_samples_used"] = pd.to_numeric(df["n_samples_used"], errors="coerce")
    df["dataset_label"] = df["dataset"].map(DATASET_LABEL)
    df.to_csv(KEY / "per_site_spearman_external_9models.tsv", sep="\t", index=False)
    df.to_csv(OUT / "external_9model_per_site_spearman.tsv", sep="\t", index=False)

    summary = df.groupby(["method_label", "dataset"], as_index=False).agg(
        median_spearman=("spearman", "median"),
        q25_spearman=("spearman", lambda x: np.nanquantile(x, 0.25) if np.isfinite(x).any() else np.nan),
        q75_spearman=("spearman", lambda x: np.nanquantile(x, 0.75) if np.isfinite(x).any() else np.nan),
        mean_spearman=("spearman", "mean"),
        n_site_rows=("target", "size"),
        n_evaluable=("spearman", lambda x: int(np.isfinite(x).sum())),
        na_ratio=("spearman", lambda x: float(pd.isna(x).mean())),
    )
    summary["method_label"] = pd.Categorical(summary["method_label"], METHOD_ORDER, ordered=True)
    summary["dataset"] = pd.Categorical(summary["dataset"], DATASET_ORDER, ordered=True)
    summary = summary.sort_values(["method_label", "dataset"])
    summary.to_csv(KEY / "median_spearman_external_9models.tsv", sep="\t", index=False)
    summary.to_csv(OUT / "external_9model_summary.tsv", sep="\t", index=False)

    med = summary.pivot(index="method_label", columns="dataset", values="median_spearman").reindex(METHOD_ORDER)
    med.columns = [DATASET_LABEL[c] for c in med.columns.astype(str)]
    med.to_csv(KEY / "median_spearman_external_9models_matrix.tsv", sep="\t")
    na = summary.pivot(index="method_label", columns="dataset", values="na_ratio").reindex(METHOD_ORDER)
    na.columns = [DATASET_LABEL[c] for c in na.columns.astype(str)]
    na.to_csv(KEY / "na_ratio_external_9models_matrix.tsv", sep="\t")
    return df, summary, med, na


def plot(summary):
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.55), sharey=True)
    for ax, ds in zip(axes, DATASET_ORDER):
        sub = summary[summary["dataset"].astype(str).eq(ds)].copy()
        sub["method_label"] = sub["method_label"].astype(str)
        sub = sub.set_index("method_label").reindex(METHOD_ORDER).reset_index()
        x = np.arange(len(METHOD_ORDER))
        y = sub["median_spearman"].to_numpy(dtype=float)
        y_plot = np.nan_to_num(y, nan=0.0)
        lower = y_plot - sub["q25_spearman"].fillna(0).to_numpy(dtype=float)
        upper = sub["q75_spearman"].fillna(0).to_numpy(dtype=float) - y_plot
        lower = np.where(np.isfinite(y), lower, 0)
        upper = np.where(np.isfinite(y), upper, 0)
        ax.bar(x, y_plot, color=[COLORS[m] for m in METHOD_ORDER], edgecolor="#4A4A4A", linewidth=0.35, width=0.72)
        ax.errorbar(x, y_plot, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#4A4A4A", elinewidth=0.45, capsize=1.4)
        for i, val in enumerate(y):
            if np.isfinite(val):
                ax.text(i, val + 0.018, f"{val:.2f}", ha="center", va="bottom", fontsize=5.2, rotation=90, family="Arial")
            else:
                ax.text(i, 0.018, "NA", ha="center", va="bottom", fontsize=5.2, rotation=90, family="Arial", color="#666666")
        ax.set_title(DATASET_LABEL[ds], fontsize=7.2, family="Arial", pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, rotation=90, ha="center", fontsize=5.7, family="Arial")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=6.3)
        ax.set_ylim(0, 0.43)
    axes[0].set_ylabel("Median per-site Spearman ρ", fontsize=6.7, family="Arial")
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.34, top=0.84, wspace=0.13)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"fig2_ext_external_9model_benchmark.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def write_md(med, na):
    md = "# External nine-model benchmark\n\n"
    md += "本表和图补齐四个外部队列的九模型对照：Mean、DeepGxP_5fold、Cognate mRNA、Ridge、PC ridge、VAE、MLP、Elastic net、SCP682。\n\n"
    md += "Mean 是每个位点训练集均值预测；在外部样本内为常数，Spearman 不定义，因此矩阵中为 NA，图中标注 NA。\n\n"
    md += "中位 Spearman 矩阵：\n\n"
    md += med.to_markdown() + "\n\n"
    md += "NA 比例矩阵：\n\n"
    md += na.to_markdown() + "\n"
    (OUT / "external_9model_benchmark.md").write_text(md, encoding="utf-8")


def main():
    _, summary, med, na = build_tables()
    plot(summary)
    write_md(med, na)
    print(med.to_string())
    print(na.to_string())


if __name__ == "__main__":
    main()
