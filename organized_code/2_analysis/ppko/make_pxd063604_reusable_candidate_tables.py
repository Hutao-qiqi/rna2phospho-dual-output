from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(r"E:\data\gongke\TCGA-TCPA")
BASE = ROOT / "02_results" / "figure_sources" / "20260531_fig4_ppko_v10b_mechanism" / "server_reusable_tables" / "pxd063604_v6_cophee_atlas"
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


def axis_node(genes: str) -> str | None:
    gene_set = {g.strip().upper() for g in str(genes).replace(",", ";").split(";") if g.strip()}
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
    if gene_set & {
        "SHC1",
        "GRB2",
        "SPRY2",
        "SPRY4",
        "KSR1",
        "KSR2",
        "NF1",
        "RASA1",
        "PEA15",
    }:
        return "other_MAPK"
    return None


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    site = pd.read_csv(BASE / "pxd063604_site_level_predictions.tsv", sep="\t")
    metrics = pd.read_csv(BASE / "pxd063604_comparison_metrics.tsv", sep="\t")
    desc = pd.read_csv(BASE / "pxd063604_drug_descriptors.tsv", sep="\t")

    inventory = pd.DataFrame(
        [
            {
                "dataset": "P100",
                "path": "02_results/figure_sources/20260531_fig4_ppko_v10b_mechanism/20260531_scp682_ppko_v10b_p100_sitelevel_all125/tables/p100_v10b_all125_unique_sitelevel_delta_long.tsv",
                "model_version": "V10B strong300",
                "rows": 3839,
                "status": "used_for_current_fig4",
                "note": "P100 125 个比较项逐位点表，当前主图使用",
            },
            {
                "dataset": "PXD063604",
                "path": str(BASE / "pxd063604_site_level_predictions.tsv"),
                "model_version": "external_bulk_validation_v6_cophee_atlas",
                "rows": int(len(site)),
                "status": "reusable_candidate_not_mixed_into_p100_main",
                "note": "12 个 drug-cell line pair；不是冻结 V10B 主包导出，适合作为候选补图或重跑 V10B 前参考",
            },
            {
                "dataset": "TCGA-TCPA",
                "path": "02_results/figure_sources/20260528_fig4_locked_p100_v10b_cosine_direction/tcga_validation/tables/v10b_300_patient_predictions.tsv",
                "model_version": "V10B strong300",
                "rows": 64,
                "status": "existing_tcga_validation_tables",
                "note": "患者药物响应表，已经有 AUC 和随机标志物对照",
            },
        ]
    )
    inventory.to_csv(SOURCE / "server_reusable_tables_inventory.tsv", sep="\t", index=False)

    site["axis_node"] = site["genes"].map(axis_node)
    ras = site[site["axis_node"].notna()].copy()
    keep_drugs = {"sotorasib", "mrtx1257", "rmc4630", "trametinib", "adagrasib", "mrtx1133", "ars1620", "bi3406", "temuterkib"}
    ras = ras[ras["drug"].str.lower().isin(keep_drugs)].copy()
    ras["axis_node"] = pd.Categorical(ras["axis_node"], categories=NODE_ORDER, ordered=True)
    ras.to_csv(SOURCE / "pxd063604_v6_ras_mapk_projection_site_long.tsv", sep="\t", index=False)

    node_summary = (
        ras.groupby(["comparison", "cell_line", "drug", "drug_code", "axis_node"], observed=True)
        .agg(
            n_sites=("norm_sequence", "count"),
            n_regulated=("regulated", "sum"),
            observed_delta_mean=("observed_delta", "mean"),
            predicted_delta_mean=("predicted_delta", "mean"),
            observed_abs_mean=("observed_delta", lambda x: float(np.mean(np.abs(x)))),
            predicted_abs_mean=("predicted_delta", lambda x: float(np.mean(np.abs(x)))),
            direction_accuracy=("direction_match", "mean"),
        )
        .reset_index()
    )
    node_summary.to_csv(SOURCE / "pxd063604_v6_ras_mapk_projection_node_summary.tsv", sep="\t", index=False)

    selected = node_summary[node_summary["drug"].isin(["sotorasib", "mrtx1257", "rmc4630", "trametinib"])].copy()
    if selected.empty:
        selected = node_summary.copy()
    selected["panel_row"] = selected["comparison"] + " observed"
    obs = selected.pivot(index="panel_row", columns="axis_node", values="observed_delta_mean")
    pred = selected.copy()
    pred["panel_row"] = pred["comparison"] + " predicted"
    pred_mat = pred.pivot(index="panel_row", columns="axis_node", values="predicted_delta_mean")
    mat = pd.concat([obs, pred], axis=0).reindex(columns=NODE_ORDER)
    row_order = []
    for comp in selected["comparison"].drop_duplicates():
        row_order.extend([f"{comp} observed", f"{comp} predicted"])
    mat = mat.reindex(row_order)
    mat.to_csv(SOURCE / "pxd063604_v6_ras_mapk_projection_heatmap_matrix.tsv", sep="\t")

    vmax = float(np.nanmax(np.abs(mat.to_numpy()))) if mat.size else 1.0
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(8.2, max(3.8, 0.32 * len(mat))))
    im = ax.imshow(mat.fillna(0).to_numpy(), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right", fontsize=8)
    labels = [idx.replace("PXD063604_", "").replace("_", " ") for idx in mat.index]
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_title("PXD063604 RAS-MAPK 候选投影")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("平均 Δp")
    save_fig(fig, "pxd063604_v6_ras_mapk_projection_candidate_heatmap")

    report = {
        "source": str(BASE),
        "model_version_note": "PXD063604 表来自 external_bulk_validation_v6_cophee_atlas，不是冻结 V10B strong300 导出。",
        "site_rows": int(len(site)),
        "ras_mapk_rows": int(len(ras)),
        "node_summary_rows": int(len(node_summary)),
        "comparisons": sorted(site["comparison"].dropna().unique().tolist()),
        "mean_site_cosine_all": float(metrics[metrics["subset"].eq("all")]["site_cosine"].mean()),
        "mean_site_cosine_regulated": float(metrics[metrics["subset"].eq("regulated")]["site_cosine"].mean()),
    }
    (SOURCE / "pxd063604_v6_candidate_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = """# PXD063604 可复用候选表

这些表来自服务器 `20260519_scp682_ppko_1_external_bulk_validation_v6_cophee_atlas`，不是冻结 V10B strong300 主包导出的结果。

已生成:
- `server_reusable_tables_inventory.tsv`
- `pxd063604_v6_ras_mapk_projection_site_long.tsv`
- `pxd063604_v6_ras_mapk_projection_node_summary.tsv`
- `pxd063604_v6_ras_mapk_projection_heatmap_matrix.tsv`
- `pxd063604_v6_candidate_summary.json`

候选图:
- `pxd063604_v6_ras_mapk_projection_candidate_heatmap.png/pdf`

使用限制:
- 可以作为“服务器已有未用表”的整理结果。
- 若要放进与 V10B P100 同一主图，需要用冻结 V10B strong300 对 PXD063604 重新导出。
"""
    (SOURCE / "PXD063604_REUSABLE_TABLES_MANIFEST.md").write_text(manifest, encoding="utf-8")


if __name__ == "__main__":
    main()
