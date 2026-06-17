from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score


ROOT = Path(r"E:\data\gongke\TCGA-TCPA")
RUN = ROOT / "02_results" / "figure_sources" / "20260531_fig4_ppko_v10b_mechanism" / "20260531_scp682_ppko_v10b_p100_sitelevel_all125"
SITE_TABLE = RUN / "tables" / "p100_v10b_all125_unique_sitelevel_delta_long.tsv"
SOURCE = ROOT / "04_figure_source_data" / "fig4_ppko_v10b_mechanism"
FIG = ROOT / "04_figures" / "fig4_ppko_v10b_mechanism"
SOURCE.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

for font_path in [
    Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
]:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PALETTE = {
    "EGFR_HER2": "#92B1D9",
    "ABL_SRC": "#6FA8C9",
    "HDAC": "#C1D8E9",
    "MEK": "#F6C8B6",
    "mTOR": "#DBDDEF",
    "proteasome": "#D4D4D4",
    "other": "#B8B8B8",
}

MODULE_COLORS = {
    "MAPK": "#92B1D9",
    "PI3K-AKT": "#C1D8E9",
    "mTOR": "#DBDDEF",
    "HDAC": "#F6C8B6",
    "proteasome": "#D4D4D4",
    "DNA damage": "#A7C4A0",
    "cell cycle": "#E3C48F",
    "RTK/SRC": "#B7A0C8",
    "other": "#C9C9C9",
}


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def normalize_site(site: str) -> str:
    return str(site).replace("p", "").replace("_", "").upper()


def site_contains(site: str, tokens: set[str]) -> bool:
    s = normalize_site(site)
    return any(tok.upper() in s for tok in tokens)


def assign_module(gene: str, site: str) -> tuple[str, str, str]:
    gene = str(gene).upper()
    site = normalize_site(site)

    if gene in {"MAPK1", "MAPK3"} and site_contains(site, {"T202", "Y204", "T185", "Y187"}):
        return "MAPK", "exact_site", "MAPK1/MAPK3 activation loop"
    if gene in {"MAPK14"} and site_contains(site, {"T180", "Y182"}):
        return "MAPK", "exact_site", "MAPK14 activation loop"
    if gene in {"RPS6KA1", "RPS6KA3", "DUSP6", "ELK1", "FOS", "JUN", "BRAF", "RAF1", "ARAF", "MAP2K1", "MAP2K2"}:
        return "MAPK", "gene_rule", gene

    if gene in {"AKT1", "AKT2", "AKT3"} and site_contains(site, {"T308", "S473"}):
        return "PI3K-AKT", "exact_site", "AKT activation site"
    if gene in {"PDPK1", "GSK3A", "GSK3B", "FOXO1", "FOXO3", "PIK3CA", "PIK3CB", "PIK3CD", "PIK3R1", "PTEN"}:
        return "PI3K-AKT", "gene_rule", gene

    if gene == "RPS6" and site_contains(site, {"S235", "S236", "S240", "S244"}):
        return "mTOR", "exact_site", "RPS6 mTOR-axis site"
    if gene in {"RPS6KB1", "EIF4EBP1", "AKT1S1", "MTOR", "RPTOR", "RICTOR", "RHEB", "TSC1", "TSC2", "RPS6"}:
        return "mTOR", "gene_rule", gene

    if gene in {"HDAC1", "HDAC2", "HDAC3", "HDAC6", "EP300", "CREBBP", "BRD4", "KAT2A", "KAT5"}:
        return "HDAC", "gene_rule", gene

    if gene.startswith(("PSMA", "PSMB", "PSMC", "PSMD")) or gene in {"NFE2L2", "HSPA1A", "DDIT3"}:
        return "proteasome", "gene_rule", gene

    if gene == "H2AFX" and site_contains(site, {"S139"}):
        return "DNA damage", "exact_site", "H2AFX S139"
    if gene in {"ATM", "ATR", "CHEK1", "CHEK2", "BRCA1", "BRCA2", "MDC1", "NBN", "MRE11", "RAD51", "PARP1", "PRKDC", "TP53"}:
        return "DNA damage", "gene_rule", gene

    if gene.startswith(("CCNA", "CCNB", "CCND", "CCNE", "MCM", "E2F")) or gene in {
        "CDK1",
        "CDK2",
        "CDK4",
        "CDK6",
        "RB1",
        "CDC25A",
        "CDC25C",
        "AURKA",
        "AURKB",
        "PLK1",
        "BUB1",
        "BUB1B",
        "TOP2A",
    }:
        return "cell cycle", "gene_rule", gene

    if gene in {
        "EGFR",
        "ERBB2",
        "ERBB3",
        "ERBB4",
        "MET",
        "AXL",
        "RET",
        "FGFR1",
        "FGFR2",
        "FGFR3",
        "KIT",
        "PDGFRA",
        "PDGFRB",
        "KDR",
        "SRC",
        "YES1",
        "FYN",
        "LCK",
        "LYN",
        "HCK",
        "ABL1",
        "ABL2",
    }:
        return "RTK/SRC", "gene_rule", gene

    return "other", "fallback", "unassigned"


def sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def ci_low(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(x.mean() - 1.96 * sem(x))


def ci_high(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(x.mean() + 1.96 * sem(x))


def make_module_map(site_df: pd.DataFrame) -> pd.DataFrame:
    site_cols = ["phosphosite", "gene", "site", "uniprot", "modified_peptide"]
    rows = site_df[site_cols].drop_duplicates("phosphosite").copy()
    assigned = rows.apply(lambda r: assign_module(r["gene"], r["site"]), axis=1)
    rows["pathway_module"] = [x[0] for x in assigned]
    rows["evidence_level"] = [x[1] for x in assigned]
    rows["matched_rule"] = [x[2] for x in assigned]
    rows["secondary_module"] = ""
    rows["evidence_source"] = "curated_gene_site_rules"
    return rows


def make_embedding(site_df: pd.DataFrame, prefix: str, value_col: str) -> pd.DataFrame:
    mat = site_df.pivot_table(index="comparison_id", columns="phosphosite", values=value_col, aggfunc="mean")
    meta = (
        site_df.drop_duplicates("comparison_id")
        .set_index("comparison_id")[["perturbation", "drug_class", "cell_line", "target_genes", "dose_molar", "time_hours"]]
        .reindex(mat.index)
    )
    mat = mat.fillna(0.0)
    reducer = PCA(n_components=2, random_state=682)
    emb = reducer.fit_transform(mat.to_numpy())
    method = "PCA"
    out = meta.reset_index()
    out["embedding_source"] = prefix
    out["embedding_method"] = method
    out["dim1"] = emb[:, 0]
    out["dim2"] = emb[:, 1]
    return out


def plot_embedding(emb: pd.DataFrame, name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.7, 4.4))
    for cls, sub in emb.groupby("drug_class"):
        ax.scatter(
            sub["dim1"],
            sub["dim2"],
            s=28,
            color=PALETTE.get(cls, "#B8B8B8"),
            alpha=0.86,
            edgecolor="white",
            linewidth=0.35,
            label=cls,
        )
    ax.set_xlabel("维度 1")
    ax.set_ylabel("维度 2")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, name)


def plot_pathway_heatmap(matrix: pd.DataFrame, name: str, title: str) -> None:
    order_rows = matrix.abs().mean(axis=1).sort_values(ascending=False).index
    order_cols = ["MAPK", "PI3K-AKT", "mTOR", "HDAC", "proteasome", "DNA damage", "cell cycle", "RTK/SRC", "other"]
    mat = matrix.reindex(index=order_rows, columns=[c for c in order_cols if c in matrix.columns]).fillna(0.0)
    vmax = float(np.nanmax(np.abs(mat.to_numpy()))) if mat.size else 1.0
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=8)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("模块平均 Δp")
    save_fig(fig, name)


def plot_scatter(df: pd.DataFrame, name: str, title: str) -> None:
    r = stats.pearsonr(df["observed_delta"], df["predicted_delta"]).statistic if len(df) >= 3 else np.nan
    rho = stats.spearmanr(df["observed_delta"], df["predicted_delta"]).statistic if len(df) >= 3 else np.nan
    r2 = r2_score(df["observed_delta"], df["predicted_delta"]) if len(df) >= 3 else np.nan
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for mod, sub in df.groupby("pathway_module"):
        ax.scatter(
            sub["observed_delta"],
            sub["predicted_delta"],
            s=34,
            color=MODULE_COLORS.get(mod, "#C9C9C9"),
            alpha=0.88,
            edgecolor="white",
            linewidth=0.35,
            label=mod,
        )
    lim = float(np.nanmax(np.abs(df[["observed_delta", "predicted_delta"]].to_numpy())))
    lim = max(lim, 1e-3)
    ax.plot([-lim, lim], [-lim, lim], color="#555555", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="#D0D0D0", linewidth=0.7)
    ax.axvline(0, color="#D0D0D0", linewidth=0.7)
    ax.set_xlim(-lim * 1.05, lim * 1.05)
    ax.set_ylim(-lim * 1.05, lim * 1.05)
    ax.set_xlabel("真实 Δp")
    ax.set_ylabel("预测 Δp")
    ax.set_title(title)
    ax.text(0.03, 0.97, f"Pearson={r:.2f}\nSpearman={rho:.2f}\nR²={r2:.2f}", transform=ax.transAxes, va="top", fontsize=8)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, name)


def plot_regulated_counts(counts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for cls, sub in counts.groupby("drug_class"):
        ax.scatter(
            sub["n_regulated_obs"],
            sub["n_regulated_pred"],
            s=44,
            color=PALETTE.get(cls, "#B8B8B8"),
            alpha=0.88,
            edgecolor="white",
            linewidth=0.35,
            label=cls,
        )
    maxv = int(max(counts["n_regulated_obs"].max(), counts["n_regulated_pred"].max(), 1))
    ax.plot([0, maxv], [0, maxv], color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("真实超过阈值位点数")
    ax.set_ylabel("预测超过阈值位点数")
    ax.set_title("P100 regulated-site count")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "regulated_site_count_scatter")


def main() -> None:
    site = pd.read_csv(SITE_TABLE, sep="\t")
    module_map = make_module_map(site)
    module_map.to_csv(SOURCE / "phosphosite_pathway_module_map.tsv", sep="\t", index=False)
    merged = site.merge(module_map[["phosphosite", "pathway_module", "evidence_level", "matched_rule"]], on="phosphosite", how="left")
    merged.to_csv(SOURCE / "p100_v10b_sitelevel_delta_with_modules.tsv", sep="\t", index=False)

    pred_emb = make_embedding(merged, "predicted_delta", "predicted_delta")
    obs_emb = make_embedding(merged, "observed_delta", "observed_delta")
    emb = pd.concat([pred_emb, obs_emb], ignore_index=True)
    emb.to_csv(SOURCE / "p100_delta_embedding_coordinates.tsv", sep="\t", index=False)
    plot_embedding(pred_emb, "p100_predicted_delta_embedding_by_drug_class", "预测 Δp 嵌入")
    plot_embedding(obs_emb, "p100_observed_delta_embedding_by_drug_class", "真实 Δp 嵌入")

    pathway_long = (
        merged.groupby(["perturbation", "drug_class", "pathway_module"], as_index=False)
        .agg(
            n_sites=("phosphosite", "nunique"),
            predicted_module_delta=("predicted_delta", "mean"),
            observed_module_delta=("observed_delta", "mean"),
            predicted_module_abs=("predicted_delta", lambda x: float(np.mean(np.abs(x)))),
            observed_module_abs=("observed_delta", lambda x: float(np.mean(np.abs(x)))),
        )
    )
    pathway_long.to_csv(SOURCE / "drug_pathway_module_delta_long.tsv", sep="\t", index=False)
    pred_mat = pathway_long.pivot(index="perturbation", columns="pathway_module", values="predicted_module_delta")
    obs_mat = pathway_long.pivot(index="perturbation", columns="pathway_module", values="observed_module_delta")
    pred_mat.to_csv(SOURCE / "drug_pathway_predicted_delta_matrix.tsv", sep="\t")
    obs_mat.to_csv(SOURCE / "drug_pathway_observed_delta_matrix.tsv", sep="\t")
    plot_pathway_heatmap(pred_mat, "drug_pathway_predicted_delta_heatmap", "预测 drug × pathway")
    plot_pathway_heatmap(obs_mat, "drug_pathway_observed_delta_heatmap", "真实 drug × pathway")

    # 代表比较项：优先选 responsive20 cosine 高且共享位点数不低的 P100 药物。
    comp_rows = []
    for cid, sub in merged.groupby("comparison_id"):
        if len(sub) < 10:
            continue
        comp_rows.append(
            {
                "comparison_id": cid,
                "perturbation": sub["perturbation"].iloc[0],
                "cell_line": sub["cell_line"].iloc[0],
                "drug_class": sub["drug_class"].iloc[0],
                "n_sites": int(len(sub)),
                "pearson": stats.pearsonr(sub["observed_delta"], sub["predicted_delta"]).statistic if len(sub) >= 3 else np.nan,
                "spearman": stats.spearmanr(sub["observed_delta"], sub["predicted_delta"]).statistic if len(sub) >= 3 else np.nan,
                "cosine": float(np.dot(sub["observed_delta"], sub["predicted_delta"]) / (np.linalg.norm(sub["observed_delta"]) * np.linalg.norm(sub["predicted_delta"]))),
            }
        )
    comp = pd.DataFrame(comp_rows).sort_values(["cosine", "n_sites"], ascending=[False, False])
    comp.to_csv(SOURCE / "representative_comparison_candidates.tsv", sep="\t", index=False)
    rep = comp.iloc[0]
    rep_df = merged[merged["comparison_id"].eq(rep["comparison_id"])].copy()
    rep_df.to_csv(SOURCE / "representative_predicted_vs_observed_scatter.tsv", sep="\t", index=False)
    plot_scatter(rep_df, "representative_predicted_vs_observed_scatter", f"{rep['perturbation']} / {rep['cell_line']}")

    cutoff = float(merged["observed_delta"].abs().quantile(0.80))
    count_rows = []
    for (perturbation, cls), sub in merged.groupby(["perturbation", "drug_class"]):
        count_rows.append(
            {
                "perturbation": perturbation,
                "drug_class": cls,
                "n_comparisons": int(sub["comparison_id"].nunique()),
                "n_rows": int(len(sub)),
                "cutoff_abs_delta": cutoff,
                "n_regulated_obs": int((sub["observed_delta"].abs() >= cutoff).sum()),
                "n_regulated_pred": int((sub["predicted_delta"].abs() >= cutoff).sum()),
            }
        )
    counts = pd.DataFrame(count_rows)
    counts.to_csv(SOURCE / "regulated_site_count_by_drug_fixed_cutoff.tsv", sep="\t", index=False)
    plot_regulated_counts(counts)

    stats_tables = {
        "n_site_rows": int(len(merged)),
        "n_comparisons": int(merged["comparison_id"].nunique()),
        "n_unique_phosphosites": int(merged["phosphosite"].nunique()),
        "module_counts": module_map["pathway_module"].value_counts().to_dict(),
        "representative_comparison": rep.to_dict(),
        "regulated_abs_delta_cutoff": cutoff,
        "embedding_method_predicted": str(pred_emb["embedding_method"].iloc[0]),
        "embedding_method_observed": str(obs_emb["embedding_method"].iloc[0]),
    }
    (SOURCE / "mechanism_panel_summary.json").write_text(json.dumps(stats_tables, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = """# Fig4 PPKO V10B 机制图源数据

输入表: `02_results/figure_sources/20260531_fig4_ppko_v10b_mechanism/20260531_scp682_ppko_v10b_p100_sitelevel_all125/tables/p100_v10b_all125_unique_sitelevel_delta_long.tsv`

新增源表:
- `phosphosite_pathway_module_map.tsv`
- `p100_v10b_sitelevel_delta_with_modules.tsv`
- `p100_delta_embedding_coordinates.tsv`
- `drug_pathway_module_delta_long.tsv`
- `drug_pathway_predicted_delta_matrix.tsv`
- `drug_pathway_observed_delta_matrix.tsv`
- `representative_comparison_candidates.tsv`
- `representative_predicted_vs_observed_scatter.tsv`
- `regulated_site_count_by_drug_fixed_cutoff.tsv`
- `mechanism_panel_summary.json`

新增图:
- `p100_predicted_delta_embedding_by_drug_class.png/pdf`
- `p100_observed_delta_embedding_by_drug_class.png/pdf`
- `drug_pathway_predicted_delta_heatmap.png/pdf`
- `drug_pathway_observed_delta_heatmap.png/pdf`
- `representative_predicted_vs_observed_scatter.png/pdf`
- `regulated_site_count_scatter.png/pdf`

说明:
- 通路模块映射使用基因和经典位点规则，未命中者归为 `other`。
- regulated-site 阈值使用 P100 全部真实 |Δp| 的第 80 百分位数，避免 top20% 固定计数。
- 嵌入图使用 PCA；P100 当前是 125×41 的小矩阵，PCA 更适合保持可复现和快速重画。
"""
    (SOURCE / "MECHANISM_MANIFEST.md").write_text(manifest, encoding="utf-8")


if __name__ == "__main__":
    main()
