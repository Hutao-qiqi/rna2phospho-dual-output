from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


ROOT = Path(r"E:\data\gongke\TCGA-TCPA")
RUN = (
    ROOT
    / "02_results"
    / "figure_sources"
    / "20260531_fig4_ppko_v10b_mechanism"
    / "20260531_scp682_ppko_v10b_pxd063604_sitelevel"
)
TABLE_DIR = RUN / "tables"
REPORT_DIR = RUN / "reports"
SITE_TABLE = TABLE_DIR / "pxd063604_v10b_sitelevel_delta_long.tsv"
METRICS_TABLE = TABLE_DIR / "pxd063604_v10b_comparison_metrics.tsv"
DRUG_SUMMARY_TABLE = TABLE_DIR / "pxd063604_v10b_drug_summary.tsv"
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

NODE_ORDER = ["KRAS", "SOS1", "SHP2", "RAF", "MEK", "ERK", "RSK", "S6", "DUSP", "JUN/FOS", "other_MAPK"]
MODULE_ORDER = [
    "RAS-MAPK",
    "RTK-SHP2-SOS",
    "PI3K-AKT",
    "mTOR",
    "cell cycle",
    "DNA damage",
    "proteasome/stress",
    "splicing/RNA",
    "other",
]
TARGET_ORDER = {"KRAS": 0, "SOS1": 1, "SHP2": 2, "MEK": 3, "ERK": 4}
SELECTED_AXIS_PAIRS = [
    ("ASPC1", "SOTORASIB"),
    ("ASPC1", "TRAMETINIB"),
    ("MiaPaCa2", "MRTX1257"),
    ("MiaPaCa2", "RMC4630"),
]
DIVERGING = LinearSegmentedColormap.from_list("scp682_diverging", ["#92B1D9", "#F1F1F1", "#F6C8B6"], N=256)


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def split_genes(value: str) -> set[str]:
    if pd.isna(value):
        return set()
    raw = str(value).replace(",", ";").replace("|", ";").replace("/", ";")
    return {x.strip().upper() for x in raw.split(";") if x.strip()}


def axis_node(genes: str) -> str | None:
    gene_set = split_genes(genes)
    if "KRAS" in gene_set:
        return "KRAS"
    if "SOS1" in gene_set:
        return "SOS1"
    if "PTPN11" in gene_set:
        return "SHP2"
    if gene_set & {"ARAF", "BRAF", "RAF1"}:
        return "RAF"
    if gene_set & {"MAP2K1", "MAP2K2"}:
        return "MEK"
    if gene_set & {"MAPK1", "MAPK3"}:
        return "ERK"
    if gene_set & {"RPS6KA1", "RPS6KA2", "RPS6KA3", "RPS6KA4"}:
        return "RSK"
    if "RPS6" in gene_set:
        return "S6"
    if gene_set & {"DUSP5", "DUSP6", "DUSP7", "DUSP9"}:
        return "DUSP"
    if gene_set & {"JUN", "FOS", "ELK1"}:
        return "JUN/FOS"
    if gene_set & {"SHC1", "GRB2", "SPRY2", "SPRY4", "KSR1", "KSR2", "NF1", "RASA1", "PEA15"}:
        return "other_MAPK"
    return None


def assign_module(genes: str) -> tuple[str, str]:
    gene_set = split_genes(genes)
    if axis_node(genes) is not None or gene_set & {"MAPKAPK2", "MAPKAPK3", "MAPK14", "MAPK8", "MAPK9", "MAPK10"}:
        return "RAS-MAPK", "ras_mapk_gene_rule"
    if gene_set & {
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
        "SOS1",
        "PTPN11",
    }:
        return "RTK-SHP2-SOS", "rtk_shp2_sos_gene_rule"
    if gene_set & {
        "PIK3CA",
        "PIK3CB",
        "PIK3CD",
        "PIK3R1",
        "PTEN",
        "PDPK1",
        "AKT1",
        "AKT2",
        "AKT3",
        "GSK3A",
        "GSK3B",
        "FOXO1",
        "FOXO3",
    }:
        return "PI3K-AKT", "pi3k_akt_gene_rule"
    if gene_set & {"MTOR", "RPTOR", "RICTOR", "RHEB", "TSC1", "TSC2", "RPS6", "RPS6KB1", "RPS6KB2", "EIF4EBP1", "AKT1S1", "EIF4B"}:
        return "mTOR", "mtor_gene_rule"
    if any(g.startswith(("CDK", "CCNA", "CCNB", "CCND", "CCNE", "MCM", "E2F")) for g in gene_set) or gene_set & {
        "RB1",
        "CDC25A",
        "CDC25C",
        "AURKA",
        "AURKB",
        "PLK1",
        "BUB1",
        "BUB1B",
        "TOP2A",
        "MKI67",
    }:
        return "cell cycle", "cell_cycle_gene_rule"
    if gene_set & {
        "ATM",
        "ATR",
        "CHEK1",
        "CHEK2",
        "BRCA1",
        "BRCA2",
        "MDC1",
        "NBN",
        "MRE11",
        "RAD51",
        "PARP1",
        "PRKDC",
        "TP53",
        "H2AFX",
    }:
        return "DNA damage", "dna_damage_gene_rule"
    if any(g.startswith(("PSMA", "PSMB", "PSMC", "PSMD")) for g in gene_set) or gene_set & {"NFE2L2", "HSPA1A", "DDIT3", "SQSTM1"}:
        return "proteasome/stress", "proteasome_stress_gene_rule"
    if any(g.startswith(("SRSF", "HNRNP")) for g in gene_set) or gene_set & {
        "SRRM1",
        "SRRM2",
        "BCLAF1",
        "THRAP3",
        "SF3B1",
        "LARP1",
        "RALY",
        "ZC3H18",
        "ZC3H13",
        "ACIN1",
    }:
        return "splicing/RNA", "splicing_rna_gene_rule"
    return "other", "unassigned"


def safe_cosine(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if len(x) == 0 or denom == 0:
        return np.nan
    return float(np.dot(x, y) / denom)


def safe_spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan
    if x[ok].nunique() <= 1 or y[ok].nunique() <= 1:
        return np.nan
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def direction_accuracy(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna() & (x != 0) & (y != 0)
    if ok.sum() == 0:
        return np.nan
    return float((np.sign(x[ok]) == np.sign(y[ok])).mean())


def add_subset_rows(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    subset_masks = {
        "all": pd.Series(True, index=df.index),
        "regulated": df["regulated"].astype(bool),
        "responsive20": df["is_responsive20"].astype(bool),
        "predicted20": df["is_predicted20"].astype(bool),
    }
    for subset, mask in subset_masks.items():
        sub = df.loc[mask].copy()
        sub["subset"] = subset
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def grouped_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_rows": int(len(sub)),
                "n_sites": int(sub["phosphosite"].nunique()),
                "n_regulated": int(sub["regulated"].astype(bool).sum()),
                "observed_delta_mean": float(sub["observed_delta"].mean()),
                "predicted_delta_mean": float(sub["predicted_delta"].mean()),
                "observed_abs_mean": float(sub["observed_delta"].abs().mean()),
                "predicted_abs_mean": float(sub["predicted_delta"].abs().mean()),
                "site_cosine": safe_cosine(sub["observed_delta"], sub["predicted_delta"]),
                "spearman": safe_spearman(sub["observed_delta"], sub["predicted_delta"]),
                "direction_accuracy": direction_accuracy(sub["observed_delta"], sub["predicted_delta"]),
                "observed_positive_fraction": float((sub["observed_delta"] > 0).mean()),
                "predicted_positive_fraction": float((sub["predicted_delta"] > 0).mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_heatmap(matrix: pd.DataFrame, name: str, title: str, label: str = "平均 Δp") -> None:
    mat = matrix.copy()
    if mat.empty:
        return
    vals = mat.to_numpy(dtype=float)
    vmax = float(np.nanpercentile(np.abs(vals), 98)) if np.isfinite(vals).any() else 1.0
    vmax = max(vmax, 1e-6)
    masked = np.ma.masked_invalid(vals)
    cmap = DIVERGING.copy()
    cmap.set_bad("#E6E6E6")
    fig, ax = plt.subplots(figsize=(8.4, max(3.6, 0.34 * len(mat) + 1.2)))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=7)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label(label)
    ax.tick_params(length=0)
    save_fig(fig, name)


def row_label(meta: pd.DataFrame) -> pd.Series:
    return meta["cell_line"].astype(str) + " " + meta["drug_code"].astype(str)


def sorted_drug_cell_rows(summary: pd.DataFrame) -> list[str]:
    meta = summary[["drug_cell", "drug_class", "cell_line", "drug_code"]].drop_duplicates().copy()
    meta["target_order"] = meta["drug_class"].map(TARGET_ORDER).fillna(9)
    meta = meta.sort_values(["target_order", "cell_line", "drug_code"])
    return meta["drug_cell"].tolist()


def make_drug_cell_pathway_matrices(pathway_summary: pd.DataFrame, subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = pathway_summary[pathway_summary["subset"].eq(subset)].copy()
    sub["drug_cell"] = row_label(sub)
    row_order = sorted_drug_cell_rows(sub)
    columns = [c for c in MODULE_ORDER if c in set(sub["pathway_module"])]
    pred = sub.pivot_table(index="drug_cell", columns="pathway_module", values="predicted_delta_mean", aggfunc="mean").reindex(index=row_order, columns=columns)
    obs = sub.pivot_table(index="drug_cell", columns="pathway_module", values="observed_delta_mean", aggfunc="mean").reindex(index=row_order, columns=columns)
    return pred, obs


def make_axis_matrix(axis_summary: pd.DataFrame, subset: str) -> pd.DataFrame:
    sub = axis_summary[axis_summary["subset"].eq(subset)].copy()
    pair_mask = pd.Series(False, index=sub.index)
    for cell_line, drug_code in SELECTED_AXIS_PAIRS:
        pair_mask |= sub["cell_line"].eq(cell_line) & sub["drug_code"].eq(drug_code)
    selected = sub.loc[pair_mask].copy()
    if selected.empty:
        selected = sub.copy()
    matrices = []
    row_order = []
    for comparison in selected["comparison"].drop_duplicates():
        one = selected[selected["comparison"].eq(comparison)].copy()
        label_base = f"{one['cell_line'].iloc[0]} {one['drug_code'].iloc[0]}"
        obs = one.pivot_table(index="comparison", columns="axis_node", values="observed_delta_mean", aggfunc="mean", observed=True).reindex(columns=NODE_ORDER)
        obs.index = [f"{label_base} 真实"]
        pred = one.pivot_table(index="comparison", columns="axis_node", values="predicted_delta_mean", aggfunc="mean", observed=True).reindex(columns=NODE_ORDER)
        pred.index = [f"{label_base} 预测"]
        matrices.extend([obs, pred])
        row_order.extend([obs.index[0], pred.index[0]])
    mat = pd.concat(matrices, axis=0)
    return mat.reindex(index=row_order, columns=NODE_ORDER)


def main() -> None:
    site = pd.read_csv(SITE_TABLE, sep="\t")
    metrics = pd.read_csv(METRICS_TABLE, sep="\t")
    drug_summary = pd.read_csv(DRUG_SUMMARY_TABLE, sep="\t")

    site["phosphosite"] = site["genes"].astype(str) + "|" + site["norm_sequence"].astype(str)
    site["axis_node"] = site["genes"].map(axis_node)
    module_assignment = site["genes"].map(assign_module)
    site["pathway_module"] = [x[0] for x in module_assignment]
    site["pathway_evidence"] = [x[1] for x in module_assignment]
    site["regulated"] = site["regulated"].astype(bool)
    site["is_responsive20"] = site["is_responsive20"].astype(bool)
    site["is_predicted20"] = site["is_predicted20"].astype(bool)
    site["direction_match"] = site["direction_match"].astype(bool)

    shutil.copy2(SITE_TABLE, SOURCE / "pxd063604_v10b_original_sitelevel_delta_long.tsv")
    annotated_path = SOURCE / "pxd063604_v10b_sitelevel_delta_long.tsv"
    site.to_csv(annotated_path, sep="\t", index=False)

    module_map = (
        site[["phosphosite", "genes", "proteins", "norm_sequence", "modified_sequence", "pathway_module", "pathway_evidence", "axis_node"]]
        .drop_duplicates("phosphosite")
        .sort_values(["pathway_module", "genes", "norm_sequence"])
    )
    module_map.to_csv(SOURCE / "pxd063604_v10b_phosphosite_pathway_module_map.tsv", sep="\t", index=False)

    subset_site = add_subset_rows(site)

    pathway_summary = grouped_metrics(
        subset_site,
        ["subset", "comparison", "cell_line", "drug_code", "drug", "drug_class", "target_genes", "pathway_module"],
    )
    pathway_summary.to_csv(SOURCE / "pxd063604_v10b_drug_cell_pathway_module_delta_long.tsv", sep="\t", index=False)

    drug_pathway_summary = grouped_metrics(
        subset_site,
        ["subset", "drug_code", "drug", "drug_class", "target_genes", "pathway_module"],
    )
    drug_pathway_summary.to_csv(SOURCE / "pxd063604_v10b_drug_pathway_module_delta_long.tsv", sep="\t", index=False)

    for subset in ["all", "regulated", "responsive20", "predicted20"]:
        pred, obs = make_drug_cell_pathway_matrices(pathway_summary, subset)
        pred.to_csv(SOURCE / f"pxd063604_v10b_drug_cell_pathway_predicted_{subset}_matrix.tsv", sep="\t")
        obs.to_csv(SOURCE / f"pxd063604_v10b_drug_cell_pathway_observed_{subset}_matrix.tsv", sep="\t")
        if subset in {"responsive20", "regulated"}:
            plot_heatmap(
                pred,
                f"pxd063604_v10b_drug_cell_pathway_predicted_{subset}_heatmap",
                f"PXD063604 预测 药物-细胞系 × 通路（{subset}）",
            )
            plot_heatmap(
                obs,
                f"pxd063604_v10b_drug_cell_pathway_observed_{subset}_heatmap",
                f"PXD063604 真实 药物-细胞系 × 通路（{subset}）",
            )

    ras = subset_site[subset_site["axis_node"].notna()].copy()
    ras["axis_node"] = pd.Categorical(ras["axis_node"], categories=NODE_ORDER, ordered=True)
    ras.to_csv(SOURCE / "pxd063604_v10b_ras_mapk_projection_site_long.tsv", sep="\t", index=False)

    axis_summary = grouped_metrics(
        ras,
        ["subset", "comparison", "cell_line", "drug_code", "drug", "drug_class", "target_genes", "axis_node"],
    )
    axis_summary["axis_node"] = pd.Categorical(axis_summary["axis_node"], categories=NODE_ORDER, ordered=True)
    axis_summary = axis_summary.sort_values(["subset", "comparison", "axis_node"])
    axis_summary.to_csv(SOURCE / "pxd063604_v10b_ras_mapk_projection_node_summary.tsv", sep="\t", index=False)

    for subset in ["responsive20", "regulated", "all"]:
        axis_mat = make_axis_matrix(axis_summary, subset)
        axis_mat.to_csv(SOURCE / f"pxd063604_v10b_ras_mapk_projection_{subset}_heatmap_matrix.tsv", sep="\t")
        if subset in {"responsive20", "regulated"}:
            plot_heatmap(
                axis_mat,
                f"pxd063604_v10b_ras_mapk_projection_{subset}_heatmap",
                f"PXD063604 KRAS-MAPK 投影（{subset}）",
            )

    comparison_metrics_out = SOURCE / "pxd063604_v10b_comparison_metrics.tsv"
    metrics.to_csv(comparison_metrics_out, sep="\t", index=False)
    drug_summary.to_csv(SOURCE / "pxd063604_v10b_drug_summary.tsv", sep="\t", index=False)
    if (REPORT_DIR / "pxd063604_v10b_sitelevel_export_summary.json").exists():
        shutil.copy2(REPORT_DIR / "pxd063604_v10b_sitelevel_export_summary.json", SOURCE / "pxd063604_v10b_sitelevel_export_summary.json")

    metric_means = (
        metrics.groupby("subset", as_index=False)
        .agg(
            n_comparisons=("comparison", "nunique"),
            mean_site_cosine=("site_cosine", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_direction_accuracy=("direction_accuracy", "mean"),
            mean_n_overlap_sites=("n_overlap_sites", "mean"),
        )
        .sort_values("subset")
    )
    metric_means.to_csv(SOURCE / "pxd063604_v10b_subset_metric_means.tsv", sep="\t", index=False)

    report = {
        "analysis": "PXD063604 frozen V10B mechanism tables",
        "input_site_table": str(SITE_TABLE),
        "site_rows": int(len(site)),
        "comparisons": int(site["comparison"].nunique()),
        "unique_phosphosites": int(site["phosphosite"].nunique()),
        "module_counts": site.drop_duplicates("phosphosite")["pathway_module"].value_counts().to_dict(),
        "ras_mapk_axis_rows_all": int(site["axis_node"].notna().sum()),
        "ras_mapk_axis_unique_sites": int(site.loc[site["axis_node"].notna(), "phosphosite"].nunique()),
        "metric_means": metric_means.to_dict(orient="records"),
        "selected_axis_pairs": [{"cell_line": c, "drug_code": d} for c, d in SELECTED_AXIS_PAIRS],
    }
    (SOURCE / "pxd063604_v10b_mechanism_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = """# PXD063604 V10B 机制图源数据

输入来自冻结正式模型 V10B 对 PXD063604 的逐位点导出：

- `02_results/figure_sources/20260531_fig4_ppko_v10b_mechanism/20260531_scp682_ppko_v10b_pxd063604_sitelevel/tables/pxd063604_v10b_sitelevel_delta_long.tsv`

已写入源表：

- `pxd063604_v10b_original_sitelevel_delta_long.tsv`
- `pxd063604_v10b_sitelevel_delta_long.tsv`
- `pxd063604_v10b_phosphosite_pathway_module_map.tsv`
- `pxd063604_v10b_drug_cell_pathway_module_delta_long.tsv`
- `pxd063604_v10b_drug_pathway_module_delta_long.tsv`
- `pxd063604_v10b_drug_cell_pathway_predicted_all_matrix.tsv`
- `pxd063604_v10b_drug_cell_pathway_observed_all_matrix.tsv`
- `pxd063604_v10b_drug_cell_pathway_predicted_regulated_matrix.tsv`
- `pxd063604_v10b_drug_cell_pathway_observed_regulated_matrix.tsv`
- `pxd063604_v10b_drug_cell_pathway_predicted_responsive20_matrix.tsv`
- `pxd063604_v10b_drug_cell_pathway_observed_responsive20_matrix.tsv`
- `pxd063604_v10b_drug_cell_pathway_predicted_predicted20_matrix.tsv`
- `pxd063604_v10b_drug_cell_pathway_observed_predicted20_matrix.tsv`
- `pxd063604_v10b_ras_mapk_projection_site_long.tsv`
- `pxd063604_v10b_ras_mapk_projection_node_summary.tsv`
- `pxd063604_v10b_ras_mapk_projection_all_heatmap_matrix.tsv`
- `pxd063604_v10b_ras_mapk_projection_regulated_heatmap_matrix.tsv`
- `pxd063604_v10b_ras_mapk_projection_responsive20_heatmap_matrix.tsv`
- `pxd063604_v10b_comparison_metrics.tsv`
- `pxd063604_v10b_drug_summary.tsv`
- `pxd063604_v10b_subset_metric_means.tsv`
- `pxd063604_v10b_sitelevel_export_summary.json`
- `pxd063604_v10b_mechanism_summary.json`

已写入图片：

- `pxd063604_v10b_drug_cell_pathway_predicted_responsive20_heatmap.png/pdf`
- `pxd063604_v10b_drug_cell_pathway_observed_responsive20_heatmap.png/pdf`
- `pxd063604_v10b_drug_cell_pathway_predicted_regulated_heatmap.png/pdf`
- `pxd063604_v10b_drug_cell_pathway_observed_regulated_heatmap.png/pdf`
- `pxd063604_v10b_ras_mapk_projection_responsive20_heatmap.png/pdf`
- `pxd063604_v10b_ras_mapk_projection_regulated_heatmap.png/pdf`

口径：

- drug-cell × pathway 使用 12 个 PXD063604 drug-cell line pair，不合并细胞系。
- KRAS-MAPK 投影使用基因规则把位点归到 KRAS、SOS1、SHP2、RAF、MEK、ERK、RSK、S6、DUSP、JUN/FOS 和 other_MAPK。
- 主图优先使用 `responsive20`，补图可用 `regulated`；`all` 和 `predicted20` 已保留矩阵供复查。
- 旧 `pxd063604_v6_*` 文件只作为历史候选表；当前可用图源以 `pxd063604_v10b_*` 为准。
"""
    (SOURCE / "PXD063604_V10B_MECHANISM_MANIFEST.md").write_text(manifest, encoding="utf-8")


if __name__ == "__main__":
    main()
