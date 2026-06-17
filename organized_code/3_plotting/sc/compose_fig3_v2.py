"""
compose_fig3_v2 — Five-panel Figure 3 for Nature Methods, double-column width (180 mm).

Panels:
  a: Architecture schematic placeholder.
  b: Per-cohort Spearman bar chart (4 cohorts).
  c: Anchor phosphosite scatter (CTNND1 Thr310 in HeLa) + STAT3 Tyr705 cross-platform bars.
  d: Graph residual ablation (5 paired bars: 4 cohorts + macro).
  e: phospho-NMF × RNA-NMF cross-validation (HeLa):
     left hexbin of phospho-NMF3 vs rna_nmf01 + right hallmark enrichment bars.

Layout (in inches at 180 mm × 7.5 in tall):
  Row 1: a (full width, short)
  Row 2: b | c | d (equal widths)
  Row 3: e (full width, taller)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
PM = ROOT / "paper_materials_SCP682_SC11"
DT = PM / "02_data_tables"
KR = PM / "01_key_results"
SRC_CACHE = ROOT / "remote_scripts" / "_paper_extract_sources" / "sc11_result" / "tables"
RNA_SD = PM / "04_figure_source_data" / "sc11_validation_rna_nmf_v4" / "source_data"
FIG3 = PM / "04_figure_source_data" / "fig3"
OUT = PM / "04_figure_source_data" / "sc11_visualization_panels_v1" / "figures"

mpl.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.5,
})

COLORS = {
    "purple": "#9C8FC4",
    "blue": "#1F3A5F",
    "gold": "#D4A56B",
    "grey": "#A8A8A8",
    "green": "#6CBFB5",
    "red": "#C0392B",
}


def panel_a(ax):
    """Architecture placeholder."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    rect = FancyBboxPatch((1, 2), 98, 26,
                          boxstyle="round,pad=0.5,rounding_size=1.0",
                          facecolor="#F5F5F5", edgecolor="#BBB", linewidth=0.7,
                          linestyle="--")
    ax.add_patch(rect)
    ax.text(50, 18, "Panel a — Architecture schematic placeholder",
            ha="center", va="center", fontsize=10, color="#666", fontweight="bold")
    ax.text(50, 11, "scRNA → scFoundation → pathway attention → site queries → expanded ScNET GNN → fusion → multi-site prediction",
            ha="center", va="center", fontsize=6.5, color="#888", style="italic")
    ax.text(50, 6, "7,369 nodes (56 supervised + 7,313 auxiliary) · 882,959 edges · 121,847 training cells",
            ha="center", va="center", fontsize=6, color="#888")


def panel_b(ax):
    """Per-cohort Spearman bar chart. Site counts shown ABOVE the bar (next to value)."""
    cohorts = ["HeLa", "Blair", "GSE300551", "Vivo-Th17"]
    spearmans = [0.498, 0.351, 0.271, 0.311]
    site_counts = [5, 1, 18, 1]

    x = np.arange(len(cohorts))
    bars = ax.bar(x, spearmans, color=COLORS["purple"],
                  edgecolor="black", linewidth=0.5, width=0.65)
    for bar, val, sc in zip(bars, spearmans, site_counts):
        # Spearman value on first line
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)
        # n_sites compact, in the bar
        ax.text(bar.get_x() + bar.get_width() / 2, val / 2,
                f"n={sc}", ha="center", va="center", fontsize=5.5,
                color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, rotation=22, ha="right", fontsize=6.5)
    ax.set_ylabel("Spearman correlation")
    ax.set_ylim(0, 0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2)


def panel_c(ax_left, ax_right):
    """CTNND1 Thr310 in HeLa scatter + STAT3 Tyr705 cross-platform bars."""
    # Left: CTNND1 scatter
    fp = FIG3 / "panel_a_hela_ctnnd1_t310_scatter.tsv"
    if fp.exists():
        df = pd.read_csv(fp, sep="\t")
        ax_left.scatter(df["observed"], df["predicted"],
                        s=2, alpha=0.45, c=COLORS["blue"], linewidths=0)
        lo = min(df["observed"].min(), df["predicted"].min()) - 0.2
        hi = max(df["observed"].max(), df["predicted"].max()) + 0.2
        ax_left.plot([lo, hi], [lo, hi], ls="--", color="grey", linewidth=0.4)
        ax_left.set_xlim(lo, hi)
        ax_left.set_ylim(lo, hi)
        ax_left.set_xlabel("Observed")
        ax_left.set_ylabel("Predicted")
        ax_left.set_aspect("equal")
        ax_left.text(0.05, 0.95, "CTNND1 (Thr310) in HeLa\nρ = 0.631\nn = 1,143 cells",
                     transform=ax_left.transAxes, va="top", ha="left", fontsize=6)
        ax_left.spines["top"].set_visible(False)
        ax_left.spines["right"].set_visible(False)
        ax_left.tick_params(length=2)
    else:
        ax_left.text(0.5, 0.5, "missing data", ha="center", va="center")

    # Right: STAT3 Y705 cross-platform bars
    cohorts = ["GSE300551", "Vivo-seq Th17"]
    spearmans = [0.224, 0.311]
    ns = [55231, 1053]
    x = np.arange(2)
    bars = ax_right.bar(x, spearmans, color=COLORS["purple"],
                        edgecolor="black", linewidth=0.5, width=0.55)
    for bar, val, n in zip(bars, spearmans, ns):
        ax_right.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
                      f"{val:.3f}\n(n={n:,})", ha="center", va="bottom",
                      fontsize=5.5)
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(cohorts, rotation=20, ha="right", fontsize=6.5)
    ax_right.set_ylabel("Spearman")
    ax_right.set_ylim(0, 0.45)
    ax_right.set_title("STAT3 (Tyr705) cross-platform", fontsize=7, pad=3)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.tick_params(length=2)


def panel_d(ax):
    """Graph residual ablation paired bars."""
    groups = ["HeLa", "Blair", "GSE300551", "Vivo-Th17", "Macro"]
    with_graph = [0.498, 0.351, 0.271, 0.311, 0.331]
    no_graph = [0.480, 0.296, 0.222, 0.246, 0.271]
    deltas = [-0.018, -0.055, -0.049, -0.066, -0.052]
    x = np.arange(len(groups))
    w = 0.36
    ax.bar(x - w / 2, with_graph, w, color=COLORS["gold"],
           edgecolor="black", linewidth=0.5, label="with expanded GNN")
    ax.bar(x + w / 2, no_graph, w, color=COLORS["grey"],
           edgecolor="black", linewidth=0.5, label="no expanded GNN")
    for i, (a, b, d) in enumerate(zip(with_graph, no_graph, deltas)):
        my = max(a, b)
        ax.text(i, my + 0.018, f"{d:+.3f}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=18, ha="right", fontsize=6)
    ax.set_ylabel("Median Spearman")
    ax.set_ylim(0, 0.62)
    ax.legend(loc="upper right", frameon=False, fontsize=5.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2)
    # Wilcoxon p as inset
    ax.text(0.02, 0.94, "Wilcoxon P = 5.15×10⁻⁵\n(n = 25 paired)",
            transform=ax.transAxes, va="top", ha="left", fontsize=5.5,
            color="#444")


def panel_e_scatter(ax):
    """phospho-NMF3 vs RNA-NMF01 hexbin."""
    phospho = pd.read_csv(FIG3 / "fig9_hela_nmf_W_cell_scores.tsv", sep="\t")
    rna = pd.read_csv(
        RNA_SD / "signal_seq_gse256403_hela_2024_rna_nmf_program_scores.tsv.gz",
        sep="\t",
    )
    # Normalize phospho columns to uppercase
    ph_cols_lower = [c for c in phospho.columns if c.lower().startswith("nmf")]
    phospho = phospho.rename(columns={c: c.upper() for c in ph_cols_lower})
    df = phospho[["cell_id", "NMF3"]].merge(
        rna[["cell_id", "rna_nmf01"]], on="cell_id", how="inner"
    )
    x = df["NMF3"].values
    y = df["rna_nmf01"].values
    ax.hexbin(x, y, gridsize=28, cmap="Blues", mincnt=1, lw=0)
    rho, p = spearmanr(x, y)
    ax.text(0.04, 0.96,
            f"ρ = {rho:+.3f}\nP = {p:.1e}\nn = {len(df)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#666", linewidth=0.4, alpha=0.9))
    ax.set_xlabel("HeLa phospho-NMF3 score")
    ax.set_ylabel("HeLa RNA-NMF01 score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2)


def panel_e_hallmark(ax):
    """Top hallmark enrichments for rna_nmf01 in HeLa."""
    h = pd.read_csv(RNA_SD / "rna_nmf_hallmark_enrichment.tsv", sep="\t")
    h01 = h[(h["cohort_id"] == "signal_seq_gse256403_hela_2024")
            & (h["component"] == "rna_nmf01")].copy()
    h01 = h01.sort_values("neg_log10_q", ascending=False).head(7)
    h01["short"] = h01["gene_set"].str.replace("HALLMARK_", "", regex=False)
    yy = np.arange(len(h01))[::-1]
    ax.barh(yy, h01["neg_log10_q"], color=COLORS["red"],
            edgecolor="white", lw=0.4)
    for i, (_, row) in enumerate(h01.iterrows()):
        ax.text(row["neg_log10_q"] + 0.03, yy[i],
                f"overlap={int(row['overlap_count'])}",
                va="center", fontsize=5, color="#444")
    ax.set_yticks(yy)
    ax.set_yticklabels(h01["short"], fontsize=6)
    ax.set_xlabel("−log10(q)")
    ax.set_title("HeLa rna_nmf01 hallmark enrichment", fontsize=7, pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2)


def add_label(ax, label, x=-0.10, y=1.07):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


def main():
    # NM double column = 180 mm = 7.087 in
    fig = plt.figure(figsize=(7.087, 7.8))
    # 3-row layout: a (short) | b/c/d (medium) | e (taller)
    gs = GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.0, 2.5, 2.6],
        hspace=0.55,
        left=0.085, right=0.97, top=0.97, bottom=0.06,
    )

    # Panel a
    ax_a = fig.add_subplot(gs[0])
    panel_a(ax_a)
    add_label(ax_a, "a", x=-0.02, y=1.02)

    # Panel b/c/d row
    gs_mid = gs[1].subgridspec(1, 3, width_ratios=[1.0, 1.8, 1.5], wspace=0.65)

    ax_b = fig.add_subplot(gs_mid[0])
    panel_b(ax_b)
    add_label(ax_b, "b", x=-0.30, y=1.08)

    # Panel c has two sub-axes (CTNND1 scatter + STAT3 bars)
    gs_c = gs_mid[1].subgridspec(1, 2, width_ratios=[1, 0.85], wspace=0.50)
    ax_c1 = fig.add_subplot(gs_c[0])
    ax_c2 = fig.add_subplot(gs_c[1])
    panel_c(ax_c1, ax_c2)
    add_label(ax_c1, "c", x=-0.28, y=1.08)

    ax_d = fig.add_subplot(gs_mid[2])
    panel_d(ax_d)
    add_label(ax_d, "d", x=-0.22, y=1.08)

    # Panel e row: hexbin + hallmark bars
    gs_e = gs[2].subgridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.40)
    ax_e1 = fig.add_subplot(gs_e[0])
    panel_e_scatter(ax_e1)
    add_label(ax_e1, "e", x=-0.16, y=1.05)
    ax_e2 = fig.add_subplot(gs_e[1])
    panel_e_hallmark(ax_e2)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig3_main_v2.svg", bbox_inches="tight")
    fig.savefig(OUT / "fig3_main_v2.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT / "fig3_main_v2.pdf", bbox_inches="tight")
    plt.close(fig)

    # Copy to fig3/
    FIG3.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png", "pdf"):
        src_p = OUT / f"fig3_main_v2.{ext}"
        if src_p.exists():
            (FIG3 / src_p.name).write_bytes(src_p.read_bytes())

    print("fig3_main_v2 saved.")


if __name__ == "__main__":
    main()
