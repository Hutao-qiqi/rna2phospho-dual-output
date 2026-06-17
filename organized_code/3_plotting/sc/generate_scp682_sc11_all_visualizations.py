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
        "font.family": "Arial",
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

CMAP_DIV = LinearSegmentedColormap.from_list("sc_div", ["#92B1D9", "#F7F7F7", "#D98973"], N=256)
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
    attn = pd.read_csv(DT / "pathway_attention_by_dataset_target.tsv", sep="\t")
    sub = attn[(attn["dataset"] == "all") & (attn["pathway"].isin(PATHWAYS_8))].copy()
    mat = sub.pivot(index="target_id", columns="pathway", values="mean_attention").reindex(columns=PATHWAYS_8)
    mat = mat.dropna(how="any")
    z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, np.nan), axis=0).fillna(0)
    order = leaves_list(linkage(pdist(z.values, metric="euclidean"), method="average"))
    mat = mat.iloc[order]
    z = z.iloc[order]
    groups = [target_group(x) for x in mat.index]
    group_codes = pd.Categorical(groups, categories=list(PATHWAY_COLORS)).codes
    write_table(
        mat.reset_index(),
        "fig1_attention_heatmap_matrix.tsv",
        "56 个监督 readout 在 8 个 pathway token 上的 mean attention。行顺序为层次聚类结果，列排除 random_control。",
    )
    ann = pd.DataFrame({"target_id": mat.index, "biological_group": groups})
    write_table(ann, "fig1_attention_heatmap_row_annotation.tsv", "通路注意力热图的行注释。")

    fig = plt.figure(figsize=(7.2, 10.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[8.5, 0.35, 1.6], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(z.values, aspect="auto", cmap=CMAP_DIV, vmin=-2.2, vmax=2.2)
    ax.set_xticks(np.arange(len(PATHWAYS_8)))
    ax.set_xticklabels([x.replace("_axis", "").replace("_", "\n") for x in PATHWAYS_8], rotation=0)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([short_label(x, 22) for x in mat.index], fontsize=5.5)
    ax.tick_params(length=0)
    ax.set_title("Pathway attention routing")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("row z-score")

    axb = fig.add_subplot(gs[0, 1])
    colors = [PATHWAY_COLORS[g] for g in groups]
    axb.imshow(np.arange(len(groups))[:, None], aspect="auto", cmap=mpl.colors.ListedColormap(colors))
    axb.set_xticks([])
    axb.set_yticks([])
    axb.set_title("group", fontsize=7)

    axl = fig.add_subplot(gs[0, 2])
    axl.axis("off")
    handles = [Patch(facecolor=c, edgecolor="none", label=k) for k, c in PATHWAY_COLORS.items()]
    axl.legend(handles=handles, loc="upper left", frameon=False, borderaxespad=0)
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
    im = ax.imshow(mat.values, aspect="auto", cmap=CMAP_DIV, norm=TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.7))
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
    nx.draw_networkx_labels(G, pos, labels={n: short_label(node_lookup[n]["label"], 12) for n in label_nodes}, font_size=6, ax=ax)
    ax.set_axis_off()
    ax.set_title("Expanded phosphosite graph neighborhood")
    handles = [Patch(facecolor=c, label=k) for k, c in edge_colors.items()]
    ax.legend(handles=handles, frameon=False, loc="lower left")
    savefig(fig, "fig7_expanded_gnn_network")


def fig8_phospho_periodic_table():
    targets = sorted(pd.read_csv(DT / "pathway_attention_by_dataset_target.tsv", sep="\t")["target_id"].unique())
    internal = median_internal_cv()
    ext = external_long()
    ext_max = ext.groupby("target_id", as_index=False).agg(external_max_spearman=("spearman", "max"))
    df = pd.DataFrame({"target_id": targets})
    df["group"] = df["target_id"].map(target_group)
    df = df.merge(internal[["target_id", "internal_cv_spearman"]], on="target_id", how="left").merge(ext_max, on="target_id", how="left")
    df["order_group"] = pd.Categorical(df["group"], categories=list(PATHWAY_COLORS)).codes
    df = df.sort_values(["order_group", "target_id"]).reset_index(drop=True)
    write_table(df, "fig8_phospho_periodic_table.tsv", "56 个监督 readout 的内部五折 Spearman 与外部最高 Spearman，用于周期表式摘要图。")
    ncol = 8
    nrow = math.ceil(len(df) / ncol)
    fig, ax = plt.subplots(figsize=(10.5, 7.6))
    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)
    ax.invert_yaxis()
    ax.axis("off")
    norm = Normalize(vmin=0, vmax=0.65)
    for i, r in df.iterrows():
        row, col = divmod(i, ncol)
        x, y = col, row
        group_color = PATHWAY_COLORS.get(r["group"], "#D4D4D4")
        int_val = r["internal_cv_spearman"]
        ext_val = r["external_max_spearman"]
        top_color = CMAP_SEQ(norm(int_val)) if pd.notna(int_val) else (0.9, 0.9, 0.9, 1)
        bot_color = CMAP_SEQ(norm(ext_val)) if pd.notna(ext_val) else (0.96, 0.96, 0.96, 1)
        ax.add_patch(Rectangle((x + 0.04, y + 0.06), 0.92, 0.42, facecolor=top_color, edgecolor="white", lw=0.4))
        ax.add_patch(Rectangle((x + 0.04, y + 0.48), 0.92, 0.42, facecolor=bot_color, edgecolor="white", lw=0.4))
        ax.add_patch(Rectangle((x + 0.04, y + 0.06), 0.92, 0.84, facecolor="none", edgecolor=group_color, lw=1.1))
        ax.text(x + 0.50, y + 0.34, short_label(r["target_id"], 11), ha="center", va="center", fontsize=5.8, color="#222222")
        ax.text(x + 0.50, y + 0.72, f"{int_val:.2f}" if pd.notna(int_val) else "NA", ha="center", va="center", fontsize=5.2, color="#222222")
    ax.set_title("Phospho readout periodic table", pad=10)
    sm = mpl.cm.ScalarMappable(cmap=CMAP_SEQ, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Spearman")
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
