#!/usr/bin/env python
# 模型: SCP682-SC
# 作用: 生成单细胞磷酸化预测模型的论文候选图和对应图源表。
# 输入: paper_materials_SCP682_SC11 中的模型结果表、外部验证表、注意力表和图先验表。
# 输出: 04_figure_source_data/sc11_visualization_panels_v1 下的 png/svg/pdf 成图与 tsv 图源表。
# 依赖: pandas, numpy, matplotlib, seaborn, scipy, networkx, pillow。
# 原始路径: remote_scripts/generate_scp682_sc11_all_visualizations.py
# 原始版本: 2026-05-27 visualization panel v1

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle, Patch
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import wilcoxon


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "paper_materials_SCP682_SC11").exists():
            return candidate
        if candidate.name == "paper_materials_SCP682_SC11":
            return candidate.parent
    return Path.cwd()


def snake_case(name: str) -> str:
    text = str(name)
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or "column"


ROOT = find_project_root()
PM = ROOT / "paper_materials_SCP682_SC11"
DT = PM / "02_data_tables"
KR = PM / "01_key_results"
SRC_CACHE = ROOT / "remote_scripts" / "_paper_extract_sources" / "sc11_result" / "tables"
OUT = PM / "04_figure_source_data" / "sc11_visualization_panels_v1"
FIG = OUT / "figures"
SD = OUT / "source_data"
REP = OUT / "reports"

for d in [OUT, FIG, SD, REP]:
    d.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
    }
)

# Stable group ordering for sorting / separators
GROUP_ORDER = [
    "BCR/BTK",
    "JAK/STAT",
    "MAPK/stress",
    "NFkB",
    "AKT/mTOR/S6",
    "cell cycle/DNA",
    "adhesion",
    "other",
]

CMAP_DIV = LinearSegmentedColormap.from_list(
    "sc_div",
    [
        (0.00, "#08306B"),
        (0.25, "#2171B5"),
        (0.50, "#FFFFFF"),
        (0.75, "#CB181D"),
        (1.00, "#67000D"),
    ],
    N=1024,  # higher N so the 5 declared stops are sampled exactly (not quantized at ~256 steps)
)
CMAP_SEQ = LinearSegmentedColormap.from_list("sc_seq", ["#F5F7FA", "#BFD8D2", "#6CBFB5", "#1F3A5F"], N=256)
CMAP_ERR = LinearSegmentedColormap.from_list("err", ["#383C73", "#6CBFB5", "#F2D06B", "#C97064"], N=256)

PATHWAYS_8 = [
    "BCR_BTK_axis",
    "BTK_PLCG2_axis",
    "ERK_axis",
    "NFkB_axis",
    "AKT_mTOR_S6_axis",
    "cell_cycle",
    "stress_ifn",
    "ribosomal",
]

PATHWAY_COLORS = {
    "BCR/BTK": "#92B1D9",
    "JAK/STAT": "#C1D8E9",
    "MAPK/stress": "#D4A56B",
    "NFkB": "#C97064",
    "AKT/mTOR/S6": "#6CBFB5",
    "cell cycle/DNA": "#9C8FC4",
    "adhesion": "#A8A8A8",
    "other": "#D4D4D4",
}


def savefig(fig, stem: str):
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=350, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_table(df: pd.DataFrame, name: str, desc: str):
    path = SD / name
    out_df = df.copy()
    out_df.columns = [snake_case(c) for c in out_df.columns]
    out_df = out_df.fillna("NA")
    out_df.to_csv(path, sep="\t", index=False, na_rep="NA")
    path.with_suffix(".md").write_text(f"# {name}\n\n{desc}\n", encoding="utf-8")
    return path


def target_group(target: str) -> str:
    t = str(target).upper()
    if any(x in t for x in ["BTK", "SYK", "BLNK", "CD79", "PLCG", "LCK", "ZAP70", "LAT", "LCP2", "SRC"]):
        return "BCR/BTK"
    if any(x in t for x in ["STAT", "JAK"]):
        return "JAK/STAT"
    if any(x in t for x in ["MAPK", "MAP2K", "JUN", "FOS", "JNK", "P44_42"]):
        return "MAPK/stress"
    if any(x in t for x in ["RELA", "P-P65", "IKK", "IRAK", "NFKB"]):
        return "NFkB"
    if any(x in t for x in ["RPS6", "AKT", "TOR", "EIF4", "PDPK", "AMPK", "NDRG"]):
        return "AKT/mTOR/S6"
    if any(x in t for x in ["CDK", "RB", "HISTON", "HISTONE", "H2AFX", "H3"]):
        return "cell cycle/DNA"
    if any(x in t for x in ["CTNND"]):
        return "adhesion"
    return "other"


def short_label(x: str, n: int = 14) -> str:
    s = str(x).replace("_pSitePending", "").replace("MAPK1_MAPK3", "ERK1/2")
    s = s.replace("RPS6_pSitePending", "RPS6")
    s = s.replace("RELA_pSitePending_93H1", "RELA_93H1")
    return s if len(s) <= n else s[: n - 1] + "."


def median_internal_cv() -> pd.DataFrame:
    f = DT / "scp682_sc11_formal_internal_5fold_per_target.tsv"
    df = pd.read_csv(f, sep="\t")
    df["spearman"] = pd.to_numeric(df["spearman"], errors="coerce")
    sub = df[(df["evaluation"] == "internal_cv_reconstruction") & (df["test_dataset"] == "all")]
    out = (
        sub.groupby("target_id", as_index=False)
        .agg(internal_cv_spearman=("spearman", "median"), internal_cv_n=("n", "median"))
    )
    return out


def external_long() -> pd.DataFrame:
    files = {
        "GSE300551": KR / "external_validation" / "per_target_gse300551.tsv",
        "Blair": KR / "external_validation" / "per_target_blair.tsv",
        "HeLa": KR / "external_validation" / "per_target_signal_seq_hela.tsv",
        "PDO_CAF": KR / "external_validation" / "per_target_signal_seq_pdo_caf.tsv",
        "Vivo_Th17": KR / "external_validation" / "per_target_vivo_seq_th17.tsv",
    }
    rows = []
    for cohort, path in files.items():
        df = pd.read_csv(path, sep="\t")
        for _, r in df.iterrows():
            rows.append(
                {
                    "cohort": cohort,
                    "target_id": r["target_id"],
                    "spearman": pd.to_numeric(r["per_target_spearman"], errors="coerce"),
                    "n": pd.to_numeric(r["sample_size_used"], errors="coerce"),
                }
            )
    return pd.DataFrame(rows)


def fig1_attention_heatmap():
    """Nature-style attention heatmap: 8 pathway tokens (rows) × 56 readouts (cols).
    Each cell shows its attention value as a small number on a sequential warm color.
    Top bar: per-readout max attention. Right bar: per-pathway mean attention.
    """
    attn = pd.read_csv(DT / "pathway_attention_by_dataset_target.tsv", sep="\t")
    sub = attn[(attn["dataset"] == "all") & (attn["pathway"].isin(PATHWAYS_8))].copy()
    mat = sub.pivot(index="pathway", columns="target_id", values="mean_attention").reindex(index=PATHWAYS_8)
    mat = mat.dropna(axis=1, how="any")
    readouts = list(mat.columns)

    # Order readouts by biological group, then within-group cluster
    readout_group = {t: target_group(t) for t in readouts}
    df_order = pd.DataFrame({"target_id": readouts, "group": [readout_group[t] for t in readouts]})
    df_order["group_rank"] = df_order["group"].map({g: i for i, g in enumerate(GROUP_ORDER)})
    ordered = []
    for g in GROUP_ORDER:
        members = df_order[df_order["group"] == g]["target_id"].tolist()
        if len(members) >= 3:
            sub_attn = mat[members].T.values
            local = leaves_list(linkage(pdist(sub_attn, metric="euclidean"), method="average"))
            members = [members[i] for i in local]
        ordered.extend(members)
    mat = mat[ordered]
    groups = [readout_group[t] for t in ordered]
    col_boundaries = [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]]

    write_table(
        mat.reset_index(),
        "fig1_attention_heatmap_matrix.tsv",
        "8 个 pathway token 行 × 56 个监督 readout 列的 mean attention。列按生物通路分组排序，组内层次聚类。",
    )
    ann = pd.DataFrame({"target_id": ordered, "biological_group": groups})
    write_table(ann, "fig1_attention_heatmap_col_annotation.tsv", "通路注意力热图的列注释（读数生物分组）。")

    M = mat.values
    n_path, n_read = M.shape
    cmap_warm = LinearSegmentedColormap.from_list(
        "warm_seq", ["#FFF5EB", "#FDD49E", "#FDAE6B", "#F16913", "#A63603"], N=1024,
    )
    vmin = float(np.percentile(M, 2))
    vmax = float(np.percentile(M, 98))

    fig = plt.figure(figsize=(17.5, 5.2))
    gs = fig.add_gridspec(
        3, 3, figure=fig,
        height_ratios=[1.0, 0.30, 4.0],
        width_ratios=[1.6, 13.0, 1.8],
        hspace=0.05, wspace=0.04,
        left=0.055, right=0.95, top=0.92, bottom=0.20,
    )

    # Top bar — per-readout max attention
    ax_top = fig.add_subplot(gs[0, 1])
    per_readout_max = M.max(axis=0)
    ax_top.bar(np.arange(n_read), per_readout_max,
               color="#888", edgecolor="white", lw=0.3, width=0.85)
    ax_top.set_xlim(-0.5, n_read - 0.5)
    ax_top.set_ylim(0, max(0.25, per_readout_max.max() * 1.12))
    ax_top.set_xticks([])
    ax_top.set_ylabel("max attn", fontsize=7)
    ax_top.tick_params(axis="y", labelsize=6, length=2)
    for s in ("top", "right", "bottom"):
        ax_top.spines[s].set_visible(False)
    for b in col_boundaries:
        ax_top.axvline(b - 0.5, color="white", lw=1.6)

    # Top biological-group color strip
    ax_grp = fig.add_subplot(gs[1, 1])
    grp_cmap = mpl.colors.ListedColormap([PATHWAY_COLORS[g] for g in GROUP_ORDER])
    grp_codes = np.array([GROUP_ORDER.index(g) for g in groups])
    ax_grp.imshow(grp_codes[None, :], aspect="auto", cmap=grp_cmap,
                  vmin=0, vmax=len(GROUP_ORDER) - 1,
                  extent=(-0.5, n_read - 0.5, 0, 1), interpolation="nearest")
    ax_grp.set_xlim(-0.5, n_read - 0.5)
    ax_grp.set_ylim(0, 1)
    ax_grp.set_xticks([])
    ax_grp.set_yticks([0.5])
    ax_grp.set_yticklabels(["group"], fontsize=7)
    for s in ("top", "right", "bottom", "left"):
        ax_grp.spines[s].set_visible(False)
    ax_grp.tick_params(length=0)
    for b in col_boundaries:
        ax_grp.axvline(b - 0.5, color="white", lw=1.6)

    # Main heatmap with number-in-cell
    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(M, aspect="auto", cmap=cmap_warm, vmin=vmin, vmax=vmax,
              interpolation="nearest")
    ax.set_xlim(-0.5, n_read - 0.5)
    ax.set_ylim(n_path - 0.5, -0.5)
    ax.set_xticks(np.arange(n_read))
    ax.set_xticklabels([short_label(x, 18) for x in ordered],
                       rotation=45, ha="right", fontsize=5.6)
    ax.set_yticks(np.arange(n_path))
    ax.set_yticklabels([p.replace("_axis", "").replace("_", " ") for p in PATHWAYS_8],
                       fontsize=8)
    ax.tick_params(length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for b in col_boundaries:
        ax.axvline(b - 0.5, color="white", lw=1.6)
    threshold = vmin + 0.65 * (vmax - vmin)
    for i in range(n_path):
        for j in range(n_read):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=4.6, color="white" if v > threshold else "#1a1a1a")

    # Right bar — per-pathway mean attention
    ax_right = fig.add_subplot(gs[2, 2])
    per_path_mean = M.mean(axis=1)
    ax_right.barh(np.arange(n_path), per_path_mean,
                  color="#F16913", edgecolor="white", lw=0.3, height=0.75)
    ax_right.set_ylim(n_path - 0.5, -0.5)
    ax_right.set_xlim(0, max(0.18, per_path_mean.max() * 1.18))
    ax_right.set_yticks([])
    ax_right.set_xlabel("mean attn", fontsize=7)
    ax_right.tick_params(axis="x", labelsize=6, length=2)
    for s in ("top", "right", "left"):
        ax_right.spines[s].set_visible(False)
    for i, v in enumerate(per_path_mean):
        ax_right.text(v + 0.003, i, f"{v:.3f}", va="center", ha="left",
                      fontsize=6, color="#333")

    # Colorbar
    cax = fig.add_axes([0.955, 0.55, 0.008, 0.30])
    sm = mpl.cm.ScalarMappable(cmap=cmap_warm, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("attention", fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.outline.set_linewidth(0.4)

    # Biological group legend (top-right)
    ax_legend = fig.add_axes([0.78, 0.94, 0.18, 0.05])
    ax_legend.axis("off")
    present = [g for g in GROUP_ORDER if g in set(groups)]
    handles = [Patch(facecolor=PATHWAY_COLORS[g], edgecolor="none", label=g) for g in present]
    ax_legend.legend(
        handles=handles, loc="upper right", frameon=False,
        ncol=4, handlelength=0.9, handletextpad=0.35, columnspacing=0.8,
        fontsize=6, title="biological group", title_fontsize=7,
    )

    fig.suptitle("Pathway attention routing (per-pathway × per-readout, attention values shown in cells)",
                 fontsize=10, y=0.985)
    savefig(fig, "fig1_pathway_attention_heatmap")


def fig2_clone_dumbbell():
    gse = pd.read_csv(KR / "external_validation" / "per_target_gse300551.tsv", sep="\t")
    sens = pd.read_csv(KR / "sensitivity_scan.tsv", sep="\t")
    rows = []
    def add(gene, low_id, high_id, cohort, low, high):
        rows.append(
            {
                "gene": gene,
                "cohort_or_scope": cohort,
                "low_target_id": low_id,
                "high_target_id": high_id,
                "low_spearman": float(low),
                "high_spearman": float(high),
                "absolute_difference": float(high) - float(low),
                "fold_ratio_abs": abs(float(high)) / (abs(float(low)) + 1e-9),
            }
        )
    add(
        "MAPK14",
        "MAPK14_pSitePending",
        "MAPK14_pSitePending_D3F9",
        "GSE300551",
        gse.loc[gse.target_id.eq("MAPK14_pSitePending"), "per_target_spearman"].iloc[0],
        gse.loc[gse.target_id.eq("MAPK14_pSitePending_D3F9"), "per_target_spearman"].iloc[0],
    )
    for _, r in sens.iterrows():
        if r["protein_symbol"] != "MAPK14":
            add(r["protein_symbol"], r["low_target_id"], r["high_target_id"], "cross-cohort scan", r["low_spearman"], r["high_spearman"])
    df = pd.DataFrame(rows).sort_values("absolute_difference")
    write_table(df, "fig2_clone_sensitivity_dumbbell.tsv", "同基因/同 parent readout 的 Spearman 差异，用于抗体克隆敏感性哑铃图。")

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    y = np.arange(len(df))
    for i, r in enumerate(df.itertuples(index=False)):
        lw = 2.0 + 8.0 * min(float(r.absolute_difference), 0.45) / 0.45
        ax.plot([r.low_spearman, r.high_spearman], [i, i], color="#B8B8B8", lw=lw, solid_capstyle="round", zorder=1)
    ax.scatter(df["low_spearman"], y, s=70, color="#92B1D9", edgecolor="white", lw=0.6, zorder=2, label="lower readout")
    ax.scatter(df["high_spearman"], y, s=70, color="#D98973", edgecolor="white", lw=0.6, zorder=3, label="higher readout")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}\n{c}" for g, c in zip(df["gene"], df["cohort_or_scope"])])
    ax.set_xlabel("Spearman")
    ax.set_xlim(-0.02, 0.72)
    ax.grid(axis="x", color="#E8E8E8")
    for i, r in enumerate(df.itertuples(index=False)):
        ax.text(max(r.low_spearman, r.high_spearman) + 0.025, i, f"{r.fold_ratio_abs:.1f}x", va="center", fontsize=7)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Antibody/readout sensitivity")
    sns.despine(ax=ax)
    savefig(fig, "fig2_clone_sensitivity_dumbbell")


def fig3_hela_umap_error():
    umap = pd.read_csv(SRC_CACHE / "scp682_sc11_hela_scfoundation_umap.tsv", sep="\t")
    pred = pd.read_csv(DT / "external_predicted_observed" / "scp682_sc11_predicted_observed_signal_seq_gse256403_hela_2024.tsv", sep="\t")
    pred["abs_error"] = (pd.to_numeric(pred["predicted"], errors="coerce") - pd.to_numeric(pred["observed"], errors="coerce")).abs()
    merged = pred.merge(umap[["row_index", "cell_id", "umap1", "umap2"]], on=["row_index", "cell_id"], how="left")
    valid = merged.dropna(subset=["umap1", "umap2", "abs_error"]).copy()
    targets = (
        valid.groupby("target_id")["abs_error"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    chosen = ["CTNND1_T310"] + [t for t in targets if t != "CTNND1_T310"][:4]
    chosen = chosen[:5]
    write_table(valid[valid["target_id"].isin(chosen)], "fig3_hela_umap_error_long.tsv", "HeLa UMAP 坐标与多个 readout 的逐细胞预测误差。")

    fig = plt.figure(figsize=(8.2, 5.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 1.0, 1.0, 1.0], wspace=0.28, hspace=0.32)
    main = valid[valid["target_id"] == "CTNND1_T310"].copy()
    vmax = np.nanpercentile(main["abs_error"], 98)
    ax0 = fig.add_subplot(gs[:, :2])
    sc = ax0.scatter(main["umap1"], main["umap2"], c=main["abs_error"], cmap=CMAP_ERR, s=12, lw=0, vmin=0, vmax=vmax)
    ax0.set_title("HeLa CTNND1_T310 error")
    ax0.set_xlabel("UMAP1")
    ax0.set_ylabel("UMAP2")
    cbar = fig.colorbar(sc, ax=ax0, fraction=0.04, pad=0.01)
    cbar.set_label("|predicted - observed|")
    mini_positions = [(0, 2), (0, 3), (1, 2), (1, 3)]
    for pos, t in zip(mini_positions, chosen[1:5]):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        sub = valid[valid["target_id"] == t]
        vmax_t = np.nanpercentile(sub["abs_error"], 98)
        ax.scatter(sub["umap1"], sub["umap2"], c=sub["abs_error"], cmap=CMAP_ERR, s=8, lw=0, vmin=0, vmax=vmax_t)
        ax.set_title(short_label(t, 18), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    savefig(fig, "fig3_hela_umap_error")


def fig4_cross_cohort_matrix():
    internal = median_internal_cv()
    internal = internal.rename(columns={"internal_cv_spearman": "Internal_CV"})[["target_id", "Internal_CV"]]
    ext = external_long().pivot_table(index="target_id", columns="cohort", values="spearman", aggfunc="first").reset_index()
    all_targets = sorted(set(pd.read_csv(DT / "pathway_attention_by_dataset_target.tsv", sep="\t")["target_id"].unique()))
    wide = pd.DataFrame({"target_id": all_targets}).merge(internal, on="target_id", how="left").merge(ext, on="target_id", how="left")
    cols = ["Internal_CV", "GSE300551", "HeLa", "Blair", "Vivo_Th17", "PDO_CAF"]
    wide = wide[["target_id"] + [c for c in cols if c in wide.columns]]
    mat = wide.set_index("target_id")
    filled = mat.fillna(0.0)
    order = leaves_list(linkage(pdist(filled.values, metric="euclidean"), method="average"))
    mat = mat.iloc[order]
    write_table(mat.reset_index(), "fig4_cross_cohort_spearman_matrix.tsv", "内部五折和外部队列的逐 readout Spearman 宽表。缺失值表示该队列没有该 readout。")
    fig = plt.figure(figsize=(6.4, 10.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[8, 0.35], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(mat.values, aspect="auto", cmap=CMAP_DIV, norm=TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=1.5))
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels([short_label(x, 22) for x in mat.index], fontsize=5.5)
    ax.tick_params(length=0)
    ax.set_title("Cross-cohort Spearman matrix")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("Spearman")
    axb = fig.add_subplot(gs[0, 1])
    groups = [target_group(x) for x in mat.index]
    colors = [PATHWAY_COLORS[g] for g in groups]
    axb.imshow(np.arange(len(groups))[:, None], aspect="auto", cmap=mpl.colors.ListedColormap(colors))
    axb.set_xticks([])
    axb.set_yticks([])
    savefig(fig, "fig4_cross_cohort_spearman_matrix")


def fig5_gnn_contribution():
    ab = pd.read_csv(DT / "bulk_site_graph_matched_ablation_per_target.tsv", sep="\t")
    sub = ab[ab["test_dataset"] == "all"].copy()
    sub["spearman_expanded_site_graph"] = pd.to_numeric(sub["spearman_expanded_site_graph"], errors="coerce")
    sub["spearman_matched_no_site_graph"] = pd.to_numeric(sub["spearman_matched_no_site_graph"], errors="coerce")
    sub = sub.dropna(subset=["spearman_expanded_site_graph", "spearman_matched_no_site_graph"])
    sub["gnn_increment"] = sub["spearman_expanded_site_graph"] - sub["spearman_matched_no_site_graph"]
    sub = sub.sort_values("gnn_increment", ascending=True)
    write_table(sub, "fig5_gnn_residual_contribution.tsv", "严格同配置 expanded site graph 与 matched no site graph 的逐 readout Spearman 差值。")
    fig, ax = plt.subplots(figsize=(7.2, 8.5))
    y = np.arange(len(sub))
    ax.barh(y, sub["spearman_matched_no_site_graph"], color="#D4D4D4", edgecolor="white", label="SC backbone without expanded graph")
    inc = sub["gnn_increment"]
    ax.barh(y, inc.clip(lower=0), left=sub["spearman_matched_no_site_graph"], color="#D4A56B", edgecolor="white", label="expanded GNN gain")
    neg = inc.clip(upper=0)
    ax.barh(y, neg, left=sub["spearman_matched_no_site_graph"], color="#92B1D9", edgecolor="white", label="expanded GNN decrease")
    ax.set_yticks(y)
    ax.set_yticklabels([short_label(x, 20) for x in sub["target_id"]], fontsize=6)
    ax.axvline(0, color="#555555", lw=0.6)
    ax.set_xlabel("Spearman")
    ax.set_title("Expanded graph contribution")
    ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax)
    savefig(fig, "fig5_gnn_residual_contribution")


def fig6_qurie_delta_polar():
    df = pd.read_csv(DT / "qurie_ibrutinib_delta_per_target.tsv", sep="\t")
    sub = df[df["context"] == "time180"].copy()
    if sub.empty:
        sub = df[df["context"].astype(str).str.contains("pooled")].copy()
    sub["group"] = sub["target_id"].map(target_group)
    sub["abs_max"] = np.maximum(sub["real_delta"].abs(), sub["pred_delta"].abs())
    sub = sub.sort_values(["group", "target_index"]).reset_index(drop=True)
    write_table(sub, "fig6_qurie_ibrutinib_delta_polar.tsv", "QuRIE ibrutinib drug-control delta 的 time180 逐 readout 实测和预测值。该图用于训练拟合质量展示，不作为独立药物算子验证。")
    n = len(sub)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.38
    max_abs = np.nanmax(sub[["real_delta", "pred_delta"]].abs().values)
    fig = plt.figure(figsize=(7.0, 7.0))
    ax = fig.add_subplot(111, polar=True)
    for i, r in enumerate(sub.itertuples(index=False)):
        color = PATHWAY_COLORS.get(r.group, "#D4D4D4")
        mismatch = np.sign(r.real_delta) != np.sign(r.pred_delta)
        ax.bar(theta[i] - width / 2, r.real_delta, width=width, bottom=0, color=color, alpha=0.95, edgecolor="white", lw=0.4)
        ax.bar(theta[i] + width / 2, r.pred_delta, width=width, bottom=0, color="#C97064" if mismatch else color, alpha=0.45, edgecolor="white", lw=0.4)
    ax.set_ylim(-max_abs * 1.25, max_abs * 1.25)
    ax.set_xticks(theta)
    ax.set_xticklabels([short_label(x, 10) for x in sub["target_id"]], fontsize=5)
    ax.set_yticklabels([])
    ax.grid(color="#E5E5E5", lw=0.5)
    ax.set_title("QuRIE ibrutinib phospho delta fingerprint", y=1.08)
    handles = [
        Patch(facecolor="#777777", alpha=0.95, label="observed delta"),
        Patch(facecolor="#777777", alpha=0.45, label="predicted delta"),
        Patch(facecolor="#C97064", alpha=0.45, label="sign mismatch"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    savefig(fig, "fig6_qurie_ibrutinib_delta_polar")


def fig7_gnn_network():
    nodes = pd.read_csv(SRC_CACHE / "scp682_sc11_scnet_site_graph_nodes.tsv", sep="\t")
    edges = pd.read_csv(SRC_CACHE / "scp682_sc11_scnet_site_graph_edges.tsv", sep="\t")
    internal = median_internal_cv().set_index("target_id")["internal_cv_spearman"].to_dict()
    sc_nodes = set(nodes.loc[nodes["node_type"] == "sc_target", "node_index"].astype(int))
    candidate = edges[edges["source"] == "target_to_bulk_candidate"].copy()
    aux_seed = set(candidate["node_1"].astype(int)).union(set(candidate["node_2"].astype(int))) - sc_nodes
    onehop = edges[(edges["node_1"].isin(aux_seed)) | (edges["node_2"].isin(aux_seed))].copy()
    onehop["weight"] = pd.to_numeric(onehop["weight"], errors="coerce").fillna(0)
    selected = pd.concat(
        [
            candidate,
            onehop.sort_values("weight", ascending=False).groupby("source", group_keys=False).head(700),
        ],
        ignore_index=True,
    ).drop_duplicates(["source", "node_1", "node_2"])
    keep_nodes = set(selected["node_1"].astype(int)).union(set(selected["node_2"].astype(int)))
    node_sub = nodes[nodes["node_index"].astype(int).isin(keep_nodes)].copy()
    write_table(node_sub, "fig7_gnn_network_nodes.tsv", "扩展 ScNET 图可视化子图节点。包含全部 56 个监督节点及其候选一跳辅助节点。")
    write_table(selected, "fig7_gnn_network_edges.tsv", "扩展 ScNET 图可视化子图边。包含 target_to_bulk_candidate 和围绕候选辅助节点的高权重边。")
    G = nx.Graph()
    node_lookup = node_sub.set_index("node_index").to_dict("index")
    for _, r in node_sub.iterrows():
        idx = int(r["node_index"])
        G.add_node(idx, node_type=r["node_type"], label=r["label"])
    for _, r in selected.iterrows():
        a, b = int(r["node_1"]), int(r["node_2"])
        if a in G and b in G:
            G.add_edge(a, b, source=r["source"], weight=float(r["weight"]))
    pos = nx.spring_layout(G, seed=682, k=0.12, iterations=120, weight=None)
    fig, ax = plt.subplots(figsize=(8, 8))
    edge_colors = {
        "target_to_bulk_candidate": "#1F3A5F",
        "CoPheeMap_onehop": "#6CBFB5",
        "CoPheeKSA_onehop": "#D4A56B",
        "KSTAR_onehop": "#9C8FC4",
    }
    for source, color in edge_colors.items():
        e = [(int(r.node_1), int(r.node_2)) for r in selected[selected["source"] == source].itertuples(index=False) if int(r.node_1) in G and int(r.node_2) in G]
        nx.draw_networkx_edges(G, pos, edgelist=e, ax=ax, edge_color=color, alpha=0.18 if source != "target_to_bulk_candidate" else 0.45, width=0.5)
    aux = [n for n in G.nodes if n not in sc_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=aux, node_size=8, node_color="#D4D4D4", linewidths=0, alpha=0.75, ax=ax)
    sc_list = [n for n in G.nodes if n in sc_nodes]
    vals = []
    for n in sc_list:
        label = str(node_lookup[n]["label"])
        vals.append(internal.get(label, np.nan))
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=sc_list,
        node_size=95,
        node_color=vals,
        cmap=CMAP_SEQ,
        vmin=0,
        vmax=0.65,
        edgecolors="white",
        linewidths=0.6,
        ax=ax,
    )
    label_nodes = sorted(sc_list, key=lambda n: internal.get(str(node_lookup[n]["label"]), -1), reverse=True)[:10]
    # Push labels slightly off-node to reduce overlap
    label_offsets = {}
    for n in label_nodes:
        px, py = pos[n]
        label_offsets[n] = (px + 0.015, py + 0.015)
    for n in label_nodes:
        lx, ly = label_offsets[n]
        ax.text(
            lx, ly, short_label(node_lookup[n]["label"], 12),
            fontsize=6, ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
        )
    ax.set_axis_off()
    ax.set_title("Expanded phosphosite graph neighborhood")
    handles = [Patch(facecolor=c, label=k) for k, c in edge_colors.items()]
    # Legend placed outside the plot, below the title — no overlap with nodes
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=7,
        handlelength=1.2,
        columnspacing=1.2,
    )
    savefig(fig, "fig7_expanded_gnn_network")


def fig8_phospho_periodic_table():
    """Redesigned: rows = pathway groups; cells flow horizontally within row, no NA padding.
    Each cell = one readout. Two stacked color stripes (internal CV on top, external max on bottom)
    plus the target name centered. Cells with both values NA are dropped.
    """
    targets = sorted(pd.read_csv(DT / "pathway_attention_by_dataset_target.tsv", sep="\t")["target_id"].unique())
    internal = median_internal_cv()
    ext = external_long()
    ext_max = ext.groupby("target_id", as_index=False).agg(external_max_spearman=("spearman", "max"))
    df = pd.DataFrame({"target_id": targets})
    df["group"] = df["target_id"].map(target_group)
    df = df.merge(internal[["target_id", "internal_cv_spearman"]], on="target_id", how="left").merge(ext_max, on="target_id", how="left")
    # Drop readouts where BOTH metrics are missing (otherwise the cell is meaningless)
    df = df[~(df["internal_cv_spearman"].isna() & df["external_max_spearman"].isna())].copy()
    df["group_rank"] = df["group"].map({g: i for i, g in enumerate(GROUP_ORDER)}).fillna(99).astype(int)
    # Within each group, sort by max(internal, external) descending to put strong readouts first
    df["sort_metric"] = df[["internal_cv_spearman", "external_max_spearman"]].max(axis=1)
    df = df.sort_values(["group_rank", "sort_metric"], ascending=[True, False]).reset_index(drop=True)
    write_table(df.drop(columns=["sort_metric", "group_rank"]),
                "fig8_phospho_periodic_table.tsv",
                "保留至少一项 Spearman 非 NA 的 readout，按通路分组、组内按最高 Spearman 降序排，用于周期表式摘要图。")

    # Layout: one logical row per pathway group; within group cells flow horizontally.
    # Larger cells (~1.6 × 1.2 in plot units), generous spacing, no numeric overlay.
    # The two color stripes alone encode internal vs external Spearman; NA cells use diagonal hatch.
    grouped = {g: df[df["group"] == g]["target_id"].tolist() for g in GROUP_ORDER if g in set(df["group"])}
    max_per_row = max(len(v) for v in grouped.values())
    n_rows = len(grouped)

    cell_w = 1.6
    cell_h = 1.2
    label_w = 2.4
    gap = 0.08
    title_h = 1.0
    legend_h = 0.55
    fig_w_units = label_w + max_per_row * cell_w + 0.5
    fig_h_units = title_h + n_rows * cell_h + legend_h

    # Convert plot units to inches at a comfortable scale
    inch_per_unit = 0.55
    fig_w_in = fig_w_units * inch_per_unit
    fig_h_in = fig_h_units * inch_per_unit
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    ax.set_xlim(-0.1, label_w + max_per_row * cell_w + 0.3)
    ax.set_ylim(0, fig_h_units)
    ax.invert_yaxis()
    ax.axis("off")
    norm = Normalize(vmin=0, vmax=0.65)

    # Title at top
    ax.text(
        (label_w + max_per_row * cell_w) / 2, 0.35,
        "Phospho readout periodic table",
        ha="center", va="center", fontsize=12, fontweight="bold",
    )

    df_index = df.set_index("target_id")
    for row_i, (g, members) in enumerate(grouped.items()):
        y_top = title_h + row_i * cell_h
        group_color = PATHWAY_COLORS.get(g, "#D4D4D4")
        # Left-side group header block
        ax.add_patch(Rectangle(
            (0, y_top + gap), label_w - 0.15, cell_h - 2 * gap,
            facecolor=group_color, edgecolor="none", alpha=0.45,
        ))
        ax.text(
            label_w / 2 - 0.075, y_top + cell_h / 2, g,
            ha="center", va="center",
            fontsize=9.5, fontweight="bold",
            color="#1a1a1a",
        )
        # Cells flow within row
        for col_i, t in enumerate(members):
            x = label_w + col_i * cell_w
            r = df_index.loc[t]
            int_val = r["internal_cv_spearman"]
            ext_val = r["external_max_spearman"]
            top_color = CMAP_SEQ(norm(int_val)) if pd.notna(int_val) else "#EEEEEE"
            bot_color = CMAP_SEQ(norm(ext_val)) if pd.notna(ext_val) else "#EEEEEE"

            # Cell bounds (inside cell_w x cell_h, with gap padding)
            cx0 = x + gap
            cy0 = y_top + gap
            cw = cell_w - 2 * gap
            ch = cell_h - 2 * gap

            # Reserve top ~28% of cell for target name
            name_h = ch * 0.32
            stripe_h = (ch - name_h) / 2.0

            # Target name block (white background, group-colored border bottom)
            ax.add_patch(Rectangle(
                (cx0, cy0), cw, name_h,
                facecolor="white", edgecolor=group_color, lw=0.8,
            ))
            ax.text(
                cx0 + cw / 2, cy0 + name_h / 2,
                short_label(t, 13),
                ha="center", va="center",
                fontsize=6.8, color="#111111",
            )

            # Internal stripe
            top_rect = Rectangle(
                (cx0, cy0 + name_h), cw, stripe_h,
                facecolor=top_color, edgecolor=group_color, lw=0.6,
            )
            ax.add_patch(top_rect)
            if pd.isna(int_val):
                top_rect.set_hatch("////")
                top_rect.set_edgecolor("#999999")
            else:
                ax.text(
                    cx0 + cw / 2, cy0 + name_h + stripe_h / 2,
                    f"{int_val:.2f}",
                    ha="center", va="center",
                    fontsize=6.5,
                    color="#FFFFFF" if int_val > 0.42 else "#222222",
                )

            # External stripe
            bot_rect = Rectangle(
                (cx0, cy0 + name_h + stripe_h), cw, stripe_h,
                facecolor=bot_color, edgecolor=group_color, lw=0.6,
            )
            ax.add_patch(bot_rect)
            if pd.isna(ext_val):
                bot_rect.set_hatch("////")
                bot_rect.set_edgecolor("#999999")
            else:
                ax.text(
                    cx0 + cw / 2, cy0 + name_h + stripe_h + stripe_h / 2,
                    f"{ext_val:.2f}",
                    ha="center", va="center",
                    fontsize=6.5,
                    color="#FFFFFF" if ext_val > 0.42 else "#222222",
                )

    # Stripe legend below the grid
    legend_y = title_h + n_rows * cell_h + 0.18
    legend_x = label_w
    # Sample mini-cell for legend
    samp_w = 0.55
    samp_h = 0.32
    # Internal sample (top half color block)
    ax.add_patch(Rectangle(
        (legend_x, legend_y), samp_w, samp_h,
        facecolor=CMAP_SEQ(norm(0.35)), edgecolor="#666666", lw=0.5,
    ))
    ax.text(legend_x + samp_w + 0.1, legend_y + samp_h / 2, "internal CV (top stripe)",
            ha="left", va="center", fontsize=7, color="#333333")
    # External sample
    ax.add_patch(Rectangle(
        (legend_x + 3.0, legend_y), samp_w, samp_h,
        facecolor=CMAP_SEQ(norm(0.5)), edgecolor="#666666", lw=0.5,
    ))
    ax.text(legend_x + 3.0 + samp_w + 0.1, legend_y + samp_h / 2, "best external (bottom stripe)",
            ha="left", va="center", fontsize=7, color="#333333")
    # NA hatched sample
    hatched = Rectangle(
        (legend_x + 6.5, legend_y), samp_w, samp_h,
        facecolor="#EEEEEE", edgecolor="#999999", lw=0.5, hatch="////",
    )
    ax.add_patch(hatched)
    ax.text(legend_x + 6.5 + samp_w + 0.1, legend_y + samp_h / 2, "NA (not evaluated)",
            ha="left", va="center", fontsize=7, color="#333333")

    # Colorbar on the right (vertical, fixed height)
    sm = mpl.cm.ScalarMappable(cmap=CMAP_SEQ, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.01, shrink=0.55)
    cbar.set_label("Spearman", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    savefig(fig, "fig8_phospho_periodic_table")


def main():
    fig1_attention_heatmap()
    fig2_clone_dumbbell()
    fig3_hela_umap_error()
    fig4_cross_cohort_matrix()
    fig5_gnn_contribution()
    fig6_qurie_delta_polar()
    fig7_gnn_network()
    fig8_phospho_periodic_table()
    manifest = {
        "output_dir": str(OUT),
        "figures": sorted([p.name for p in FIG.glob("*.png")]),
        "source_tables": sorted([p.name for p in SD.glob("*.tsv")]),
        "notes": [
            "fig5 uses matched no-expanded-site-graph versus expanded-site-graph; true attention-only ablation is not available.",
            "fig7 draws a supervised-node neighborhood subgraph rather than all 882,959 edges to keep the graph interpretable.",
            "fig6 reports QuRIE ibrutinib training fit quality, not independent causal perturbation validation.",
        ],
    }
    (OUT / "VISUALIZATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# SCP682-SC11 visualization panels\n\n"
        "该目录包含 8 张 SCP682-SC11 论文候选图、对应 SVG/PNG/PDF 和可重绘 TSV 图源表。\n\n"
        "- fig1: 通路注意力路由热图。\n"
        "- fig2: 抗体/readout 敏感性哑铃图。\n"
        "- fig3: HeLa UMAP 预测误差图。\n"
        "- fig4: 跨队列 Spearman 矩阵。\n"
        "- fig5: 扩展 GNN 图贡献分解。\n"
        "- fig6: QuRIE ibrutinib delta 极坐标图。\n"
        "- fig7: 扩展 phosphosite 图网络子图。\n"
        "- fig8: phospho readout 周期表。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
