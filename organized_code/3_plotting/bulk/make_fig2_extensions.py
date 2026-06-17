from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr


ROOT = Path(r"E:\data\gongke\TCGA-TCPA\paper_materials_SCP682")
KEY = ROOT / "01_key_results"
OOF = ROOT / "02_data_tables" / "oof_branch_predictions"
OUT = ROOT / "04_figure_source_data" / "fig2_extensions"
PRIOR = OUT / "prior_sources"

PALETTE = {
    "SCP682": "#1F3A5F",
    "cognate": "#D4A56B",
    "KSA annotated": "#6CBFB5",
    "CoPheeMap only": "#92B1D9",
    "orphan": "#A8A8A8",
    "rescued": "#D4A56B",
}

DATASET_LABEL = {
    "CPTAC_all": "CPTAC all",
    "fu_icca": "FU-iCCA",
    "tu_sclc": "TU-SCLC",
    "chcc_hbv_fpkm": "CHCC FPKM",
    "chcc_hbv_rsem": "CHCC RSEM",
}


def setup_style():
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })


def savefig(fig, stem: str, width_mm: float, height_mm: float):
    fig.set_size_inches(width_mm / 25.4, height_mm / 25.4)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_tsv(df: pd.DataFrame, path: Path, index: bool = False):
    df.to_csv(path, sep="\t", index=index, na_rep="NA")


def load_site_classes(targets: pd.Index) -> pd.DataFrame:
    manifest = pd.read_csv(OOF / "phosphosite_target_manifest.tsv", sep="\t")
    manifest = manifest.rename(columns={"scp682_site_id": "target"})
    manifest["target"] = manifest["target"].astype(str)
    ksa = pd.read_csv(PRIOR / "copheeksa_model_phosphosite_kinase_predictions.tsv", sep="\t", usecols=["gene_site_id"])
    ksa_sites = set(ksa["gene_site_id"].astype(str))
    edges = pd.read_csv(PRIOR / "copheemap_model_phosphosite_edges.tsv", sep="\t", usecols=["site_a", "site_b"])
    copheemap_sites = set(edges["site_a"].astype(str)).union(set(edges["site_b"].astype(str)))
    df = manifest[["target", "parent_gene", "residue_site", "residue", "position"]].copy()
    df["has_ksa"] = df["target"].isin(ksa_sites)
    df["has_copheemap"] = df["target"].isin(copheemap_sites)
    df["site_class"] = np.select(
        [df["has_ksa"], df["has_copheemap"]],
        ["KSA annotated", "CoPheeMap only"],
        default="orphan",
    )
    df = df[df["target"].isin(set(targets.astype(str)))]
    write_tsv(df, OUT / "site_class_annotation.tsv")
    return df


def paired_internal(site_class: pd.DataFrame) -> pd.DataFrame:
    perf = pd.read_csv(KEY / "per_site_spearman.tsv", sep="\t")
    sub = perf[(perf["dataset"] == "CPTAC_all") & (perf["method"].isin(["SCP682", "parent_mRNA_linear"]))]
    wide = sub.pivot_table(index="target", columns="method", values="spearman", aggfunc="first").reset_index()
    n = sub[sub["method"] == "SCP682"][["target", "n_samples_used"]].rename(columns={"n_samples_used": "n_samples"})
    wide = wide.merge(n, on="target", how="left")
    wide = wide.merge(site_class, on="target", how="left")
    wide = wide.rename(columns={"parent_mRNA_linear": "cognate_mrna", "SCP682": "scp682"})
    wide["delta"] = wide["scp682"] - wide["cognate_mrna"]
    wide["rescued"] = (wide["cognate_mrna"] < 0.1) & (wide["scp682"] >= 0.3)
    write_tsv(wide, OUT / "scp682_vs_cognate_internal.tsv")
    rescued = wide[wide["rescued"]].sort_values("delta", ascending=False)
    write_tsv(rescued, OUT / "rescued_sites_internal.tsv")
    return wide


def paired_external(site_class: pd.DataFrame) -> pd.DataFrame:
    perf = pd.read_csv(KEY / "per_site_spearman_external.tsv", sep="\t")
    sub = perf[perf["method"].isin(["SCP682", "parent_mRNA_linear"])]
    wide = sub.pivot_table(index=["dataset", "target"], columns="method", values="spearman", aggfunc="first").reset_index()
    n = sub[sub["method"] == "SCP682"][["dataset", "target", "n_samples_used"]].rename(columns={"n_samples_used": "n_samples"})
    wide = wide.merge(n, on=["dataset", "target"], how="left")
    wide = wide.merge(site_class, on="target", how="left")
    wide = wide.rename(columns={"parent_mRNA_linear": "cognate_mrna", "SCP682": "scp682"})
    wide["delta"] = wide["scp682"] - wide["cognate_mrna"]
    wide["rescued"] = (wide["cognate_mrna"] < 0.1) & (wide["scp682"] >= 0.3)
    write_tsv(wide, OUT / "scp682_vs_cognate_external.tsv")
    return wide


def plot_scatter_one(ax, df: pd.DataFrame, label: str, annotate: bool = True):
    d = df[["cognate_mrna", "scp682", "delta"]].dropna()
    ax.axline((0, 0), slope=1, color="#777777", linewidth=0.7, linestyle="--", zorder=1)
    ax.axhline(0.3, color="#A8A8A8", linewidth=0.5, linestyle=":", zorder=1)
    ax.axvline(0.1, color="#A8A8A8", linewidth=0.5, linestyle=":", zorder=1)
    rescued = d[(d["cognate_mrna"] < 0.1) & (d["scp682"] >= 0.3)]
    other = d.drop(index=rescued.index)
    ax.scatter(other["cognate_mrna"], other["scp682"], s=3, color="#1F3A5F", alpha=0.18, linewidths=0)
    ax.scatter(rescued["cognate_mrna"], rescued["scp682"], s=5, color=PALETTE["rescued"], alpha=0.6, linewidths=0)
    ax.set_xlim(-0.35, 0.9)
    ax.set_ylim(-0.35, 0.9)
    ax.set_title(label, fontsize=8, fontweight="bold")
    ax.set_xlabel("Cognate mRNA Spearman")
    ax.set_ylabel("SCP682 Spearman")
    if annotate and len(d):
        pct = float((d["delta"] > 0).mean() * 100)
        med = float(d["delta"].median())
        n_rescued = int(len(rescued))
        txt = f"SCP682 > cognate: {pct:.1f}%\nmedian delta: {med:.3f}\nrescued: {n_rescued}"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=6.5)


def plot_scatter_with_marginal(df: pd.DataFrame):
    fig = plt.figure(figsize=(90 / 25.4, 78 / 25.4))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05, figure=fig)
    ax_histx = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    ax_histy = fig.add_subplot(gs[1, 1])
    d = df[["cognate_mrna", "scp682", "delta"]].dropna()
    plot_scatter_one(ax, df, "CPTAC all", annotate=True)
    bins = np.linspace(-0.35, 0.9, 55)
    ax_histx.hist(d["cognate_mrna"], bins=bins, color="#D4A56B", alpha=0.75)
    ax_histy.hist(d["scp682"], bins=bins, orientation="horizontal", color="#1F3A5F", alpha=0.75)
    ax_histx.axis("off")
    ax_histy.axis("off")
    savefig(fig, "fig2_ext_scatter_scp682_vs_cognate_internal", 90, 78)


def plot_external_facets(df: pd.DataFrame):
    order = ["fu_icca", "tu_sclc", "chcc_hbv_fpkm", "chcc_hbv_rsem"]
    fig, axes = plt.subplots(1, 4, figsize=(180 / 25.4, 45 / 25.4), sharex=True, sharey=True)
    for ax, ds in zip(axes, order):
        plot_scatter_one(ax, df[df["dataset"] == ds], DATASET_LABEL[ds], annotate=True)
        if ax is not axes[0]:
            ax.set_ylabel("")
    savefig(fig, "fig2_ext_scatter_scp682_vs_cognate_external", 180, 45)


def plot_site_class(df: pd.DataFrame):
    order = ["KSA annotated", "CoPheeMap only", "orphan"]
    d = df[df["site_class"].isin(order)].dropna(subset=["scp682"])
    summary = d.groupby("site_class", dropna=False).agg(
        n=("target", "count"),
        median_scp682=("scp682", "median"),
        median_cognate=("cognate_mrna", "median"),
        median_delta=("delta", "median"),
        rescued_fraction=("rescued", "mean"),
    ).reindex(order)
    write_tsv(summary.reset_index(), OUT / "site_class_summary_internal.tsv")
    fig, ax = plt.subplots(figsize=(86 / 25.4, 58 / 25.4))
    parts = []
    vals = []
    for cls in order:
        arr = d.loc[d["site_class"] == cls, "delta"].dropna().to_numpy()
        vals.append(arr)
    parts = ax.violinplot(vals, positions=np.arange(len(order)), showmeans=False, showmedians=False, widths=0.75)
    for body, cls in zip(parts["bodies"], order):
        body.set_facecolor(PALETTE.get(cls, "#A8A8A8"))
        body.set_edgecolor("#333333")
        body.set_linewidth(0.4)
        body.set_alpha(0.85)
    for i, arr in enumerate(vals):
        q1, med, q3 = np.nanpercentile(arr, [25, 50, 75])
        ax.plot([i - 0.18, i + 0.18], [med, med], color="#111111", linewidth=1.0)
        ax.plot([i, i], [q1, q3], color="#111111", linewidth=0.8)
        ax.text(i, 0.84, f"n={len(arr)}", ha="center", va="top", fontsize=6.5)
    ax.axhline(0, color="#777777", linewidth=0.7, linestyle="--")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("SCP682 - cognate mRNA Spearman")
    ax.set_ylim(-0.55, 0.9)
    ax.set_title("Site class stratification", fontsize=8, fontweight="bold")
    savefig(fig, "fig2_ext_site_class_delta_violin", 86, 58)


def plot_anchor_cases():
    data = pd.read_csv(OUT / "anchor_case_data.tsv", sep="\t")
    perf = pd.read_csv(OUT / "scp682_vs_cognate_internal.tsv", sep="\t")
    anchors = ["CTNND1|S38", "MAPK1|T185_Y187", "STAT3|Y705", "RB1|S807"]
    selection = pd.DataFrame({
        "target": anchors,
        "selection_note": [
            "CTNND1 T310 was absent; selected highest-delta CTNND1 site",
            "requested MAPK1 anchor available as combined T185/Y187 site",
            "requested STAT3 Y705",
            "requested RB1 S807",
        ],
    }).merge(perf[["target", "scp682", "cognate_mrna", "delta", "n_samples", "site_class"]], on="target", how="left")
    write_tsv(selection, OUT / "anchor_site_selection.tsv")
    cmap = {
        "STAD": "#6CBFB5",
        "LUAD": "#92B1D9",
        "LSCC": "#1F3A5F",
        "CCRCC": "#D4A56B",
        "HNSCC": "#F6C8B6",
        "PDA": "#DBDDEF",
        "UCEC": "#A8A8A8",
    }
    fig, axes = plt.subplots(2, 2, figsize=(112 / 25.4, 92 / 25.4), sharex=False, sharey=False)
    for ax, target in zip(axes.ravel(), anchors):
        sub = data[data["target"] == target].copy()
        sub = sub[np.isfinite(sub["observed"]) & np.isfinite(sub["scp682_pred"])]
        for cancer, g in sub.groupby("cancer_label"):
            ax.scatter(g["observed"], g["scp682_pred"], s=7, alpha=0.5, linewidths=0, color=cmap.get(cancer, "#999999"), label=cancer)
        if len(sub) > 2:
            rho = spearmanr(sub["observed"], sub["scp682_pred"], nan_policy="omit").correlation
        else:
            rho = np.nan
        lo = np.nanpercentile(np.r_[sub["observed"], sub["scp682_pred"]], 1)
        hi = np.nanpercentile(np.r_[sub["observed"], sub["scp682_pred"]], 99)
        ax.plot([lo, hi], [lo, hi], color="#777777", linewidth=0.7, linestyle="--")
        ax.set_title(f"{target}  rho={rho:.2f}", fontsize=8, fontweight="bold")
        ax.set_xlabel("Observed phosphosite")
        ax.set_ylabel("SCP682 prediction")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles[:7], labels[:7], loc="upper center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.04))
    savefig(fig, "fig2_ext_anchor_case_studies", 112, 92)


def write_readme(internal: pd.DataFrame, external: pd.DataFrame):
    d = internal.dropna(subset=["scp682", "cognate_mrna"])
    ext = external.dropna(subset=["scp682", "cognate_mrna"])
    lines = [
        "# Fig2 extension source data",
        "",
        "Generated panels:",
        "- fig2_ext_scatter_scp682_vs_cognate_internal",
        "- fig2_ext_scatter_scp682_vs_cognate_external",
        "- fig2_ext_anchor_case_studies",
        "- fig2_ext_site_class_delta_violin",
        "",
        "Internal summary:",
        f"- paired sites: {len(d)}",
        f"- SCP682 > cognate mRNA: {(d['delta'] > 0).mean() * 100:.2f}%",
        f"- median delta: {d['delta'].median():.4f}",
        f"- rescued sites: {int(d['rescued'].sum())}",
        "",
        "External summary:",
        f"- paired dataset-site rows: {len(ext)}",
        f"- SCP682 > cognate mRNA: {(ext['delta'] > 0).mean() * 100:.2f}%",
        f"- median delta: {ext['delta'].median():.4f}",
        f"- rescued rows: {int(ext['rescued'].sum())}",
    ]
    (OUT / "README_fig2_extensions.md").write_text("\n".join(lines), encoding="utf-8")


def write_sidecar_docs():
    docs = {
        "scp682_vs_cognate_internal": "Internal CPTAC all per-site Spearman paired table. Columns: target, scp682, cognate_mrna, n_samples, phosphosite metadata, site_class, delta, rescued. rescued is cognate_mrna < 0.1 and scp682 >= 0.3.",
        "scp682_vs_cognate_external": "External per-site Spearman paired table across FU-iCCA, TU-SCLC, CHCC FPKM and CHCC RSEM. Columns match internal table with dataset added.",
        "rescued_sites_internal": "Subset of internal paired sites where cognate_mrna < 0.1 and scp682 >= 0.3, sorted by delta.",
        "site_class_annotation": "Per-site annotation used for stratification. KSA annotated is defined by CoPheeKSA membership; CoPheeMap only is defined by CoPheeMap edge membership and absence from KSA; orphan is neither.",
        "site_class_summary_internal": "Internal summary of SCP682 and cognate mRNA performance by site_class.",
        "anchor_case_data": "Long table for four anchor phosphosite case studies. Columns: sample_id, cancer_label, pdc_study_id, target, observed, scp682_pred, cognate_mrna_pred.",
        "anchor_site_selection": "Anchor site selection metadata and internal performance. CTNND1 T310 was absent from the target manifest and was replaced by CTNND1|S38.",
    }
    for stem, text in docs.items():
        (OUT / f"{stem}.md").write_text(text + "\n", encoding="utf-8")
    prior_docs = {
        "copheeksa_model_phosphosite_kinase_predictions": "Raw CoPheeKSA prior table copied from /data/lsy/Infinite_Stream/01_data/pathway_prior/processed/copheemap_v1/. Used only to define KSA annotated site membership.",
        "copheemap_model_phosphosite_edges": "Raw CoPheeMap site-site edge table copied from /data/lsy/Infinite_Stream/01_data/pathway_prior/processed/copheemap_v1/. Used only to define CoPheeMap-only site membership.",
    }
    for stem, text in prior_docs.items():
        (PRIOR / f"{stem}.md").write_text(text + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    all_targets = pd.read_csv(KEY / "per_site_spearman.tsv", sep="\t", usecols=["target"])["target"].drop_duplicates()
    site_class = load_site_classes(all_targets)
    internal = paired_internal(site_class)
    external = paired_external(site_class)
    plot_scatter_with_marginal(internal)
    plot_external_facets(external)
    plot_site_class(internal)
    plot_anchor_cases()
    write_readme(internal, external)
    write_sidecar_docs()


if __name__ == "__main__":
    main()
