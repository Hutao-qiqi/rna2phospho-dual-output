import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:/data/gongke/TCGA-TCPA")
SRC = ROOT / "04_figures/20260528_fig5_v2/tables/full_tcga_scp682_main_20260529"
OUT = ROOT / "paper_final/fig5/source_data/tables"
OUT.mkdir(parents=True, exist_ok=True)


def cancer_short(x: str) -> str:
    return x.replace("TCGA-", "")


def count_by_direction(df: pd.DataFrame, flag_col: str, beta_col: str) -> pd.DataFrame:
    d = df[df[flag_col].fillna(False)].copy()
    out = []
    if d.empty:
        return pd.DataFrame(columns=["cancer", "risk", "protective", "total"])
    for cancer, g in d.groupby("cancer", sort=False):
        beta = pd.to_numeric(g[beta_col], errors="coerce")
        out.append(
            {
                "cancer": cancer,
                "risk": int((beta > 0).sum()),
                "protective": int((beta < 0).sum()),
                "total": int(beta.notna().sum()),
            }
        )
    return pd.DataFrame(out)


def add_matrix_row(rows, row_id, row_label, row_group, cancer, value, direction="total", source=""):
    rows.append(
        {
            "row_id": row_id,
            "row_label": row_label,
            "row_group": row_group,
            "cancer": cancer,
            "cancer_short": cancer_short(cancer),
            "direction": direction,
            "value": int(value) if pd.notna(value) else 0,
            "source": source,
        }
    )


def main():
    summary = pd.read_csv(SRC / "full_tcga_scp682_main_project_effect_summary.tsv", sep="\t")
    summary_q = pd.read_csv(SRC / "full_tcga_scp682_main_project_effect_summary_with_q.tsv", sep="\t")
    fig5a_old = pd.read_csv(SRC / "fig5a_tcga_cancer_effect_classification.tsv", sep="\t")
    arch = pd.read_csv(SRC / "full_tcga_scp682_main_architecture_effect_matrix.tsv.gz", sep="\t")
    modules = pd.read_csv(SRC / "full_tcga_scp682_main_residual_module_nmf_input.tsv", sep="\t")
    anchor_dir = (
        ROOT
        / "02_results/model_validation/20260527_cptac_measured_tcga_predicted_site_survival_concordance_v1/tables"
    )
    strict_anchor_file = anchor_dir / "cptac_strict_candidates_with_tcga_predicted_survival.tsv"
    confirmed_anchor_file = anchor_dir / "tcga_confirmed_site_specific_candidates.tsv"
    strict_anchors = pd.read_csv(strict_anchor_file, sep="\t")
    confirmed_anchors = pd.read_csv(confirmed_anchor_file, sep="\t")

    arch = arch[arch["analysis_evaluable"].fillna(False)].copy()
    arch["cancer"] = arch["cancer"].astype(str)

    # === 多重检验校正 (BH-FDR q<0.05) ===
    # 上游 flag (clinical_significant / parent_mrna_independent / graph_residual_significant)
    # 用的是未校正 nominal p<0.05; 在 ~153k 位点 x 28 癌种规模下必须做 FDR 校正,
    # 否则海量假阳性。这里用上游已算好的 BH q 列重定义三个 flag(公式/方向口径不变, 仅 p->q):
    #   clinical           = cox_q_full_bh < 0.05
    #   beyond parent mRNA = clinical AND add_site_to_parent_mrna_lrt_q_bh < 0.05  (联合 Cox 增量 LRT)
    #   graph residual     = clinical AND gain_q_bh < 0.05
    QTHR = 0.05
    _q = lambda c: pd.to_numeric(arch[c], errors="coerce") < QTHR
    arch["clinical_significant"] = _q("cox_q_full_bh")
    arch["parent_mrna_independent"] = arch["clinical_significant"] & _q("add_site_to_parent_mrna_lrt_q_bh")
    arch["graph_residual_significant"] = arch["clinical_significant"] & _q("gain_q_bh")
    cancers = (
        fig5a_old.sort_values("classification_score", ascending=False)["cancer"]
        .astype(str)
        .tolist()
    )

    # 只保留有常规 TCGA 生存分析覆盖的癌种。ACC 在旧表里 manuscript_evaluable=False，
    # 但保留在列注释中，绘图时可按 manuscript_evaluable 决定是否展示。
    col_meta = summary.merge(summary_q, on="project_id", how="left", suffixes=("", "_q"))
    col_meta["cancer"] = col_meta["project_id"]
    col_meta = col_meta[col_meta["cancer"].isin(cancers)].copy()
    col_meta["cancer_short"] = col_meta["cancer"].map(cancer_short)
    order = {c: i + 1 for i, c in enumerate(cancers)}
    col_meta["cancer_order"] = col_meta["cancer"].map(order)
    col_meta = col_meta.sort_values("cancer_order")

    clinical_dir = count_by_direction(arch, "clinical_significant", "cox_beta_full")
    site_over_mrna_dir = count_by_direction(
        arch, "parent_mrna_independent", "site_beta_adjusted_for_parent_mrna"
    )
    graph_dir = count_by_direction(arch, "graph_residual_significant", "graph_delta_beta")
    both = arch[
        arch["parent_mrna_independent"].fillna(False)
        & arch["graph_residual_significant"].fillna(False)
    ].copy()
    both_dir = count_by_direction(
        both.assign(flag=True), "flag", "site_beta_adjusted_for_parent_mrna"
    )

    parent_mrna = (
        arch[arch["parent_mrna_present"].fillna(False)]
        .assign(parent_mrna_clinical=lambda x: pd.to_numeric(x["parent_mrna_p"], errors="coerce") < 0.05)
        .query("parent_mrna_clinical")
        .groupby("cancer")["gene"]
        .nunique()
        .reset_index(name="parent_mrna_clinical_genes")
    )

    # 模块层读出。旧 NMF 输入表里模块读出量是 0-1 归一化比例；
    # 为了用于癌种入口图，取 >=0.10 作为“模块内有足够位点支持”的展示阈值。
    module_threshold = 0.10
    sample_modules = (
        modules[modules["nmf_n_site_over_mrna_sites"] >= module_threshold]
        .groupby("cancer")["gene"]
        .nunique()
        .reset_index(name="sample_graph_modules")
    )
    site_modules = (
        modules[modules["nmf_n_graph_residual_sites"] >= module_threshold]
        .groupby("cancer")["gene"]
        .nunique()
        .reset_index(name="site_graph_modules")
    )
    multisite_modules = (
        modules[
            (modules["n_sites_total"] >= 3)
            & (modules["nmf_n_site_over_mrna_sites"] > 0)
        ]
        .groupby("cancer")["gene"]
        .nunique()
        .reset_index(name="multi_site_genes")
    )

    strict_external = (
        strict_anchors[strict_anchors["tcga_project"].notna()]
        .groupby("tcga_project")["target"]
        .nunique()
        .reset_index(name="strict_external_anchors")
        .rename(columns={"tcga_project": "cancer"})
    )
    confirmed_external = (
        confirmed_anchors[confirmed_anchors["tcga_project"].notna()]
        .groupby("tcga_project")["target"]
        .nunique()
        .reset_index(name="confirmed_external_anchors")
        .rename(columns={"tcga_project": "cancer"})
    )

    merged = fig5a_old[["cancer", "n", "events", "classification_score"]].copy()
    for d in [
        clinical_dir.add_prefix("clinical_").rename(columns={"clinical_cancer": "cancer"}),
        site_over_mrna_dir.add_prefix("site_over_mrna_").rename(columns={"site_over_mrna_cancer": "cancer"}),
        graph_dir.add_prefix("graph_").rename(columns={"graph_cancer": "cancer"}),
        both_dir.add_prefix("both_").rename(columns={"both_cancer": "cancer"}),
        parent_mrna,
        sample_modules,
        site_modules,
        multisite_modules,
        strict_external,
        confirmed_external,
    ]:
        merged = merged.merge(d, on="cancer", how="left")
    count_cols = [c for c in merged.columns if c not in ["cancer", "n", "events", "classification_score"]]
    merged[count_cols] = merged[count_cols].fillna(0).astype(int)

    # 顶部柱图：灰色为所有单变量临床预测位点；暖色/蓝色为超过 parent mRNA 的风险/保护位点。
    top_bars = merged[
        [
            "cancer",
            "clinical_total",
            "site_over_mrna_risk",
            "site_over_mrna_protective",
            "site_over_mrna_total",
            "n",
            "events",
            "classification_score",
        ]
    ].copy()
    top_bars["cancer_short"] = top_bars["cancer"].map(cancer_short)
    top_bars["cancer_order"] = top_bars["cancer"].map(order)
    top_bars = top_bars.sort_values("cancer_order")

    row_defs = [
        ("r01", "Parent mRNA clinical genes", "mRNA baseline", "parent_mrna_clinical_genes", "total"),
        ("r02", "All clinical predicted sites", "Predicted phosphosite", "clinical_total", "total"),
        ("r03", "Sites beyond parent mRNA, risk", "Beyond parent mRNA", "site_over_mrna_risk", "risk"),
        ("r04", "Sites beyond parent mRNA, protective", "Beyond parent mRNA", "site_over_mrna_protective", "protective"),
        ("r05", "Sites with graph-residual effect, risk", "Graph residual", "graph_risk", "risk"),
        ("r06", "Sites with graph-residual effect, protective", "Graph residual", "graph_protective", "protective"),
        ("r07", "Sites beyond parent mRNA and graph residual", "Intersection", "both_total", "total"),
        ("r08", "Modules with sample-graph structure", "Architecture module", "sample_graph_modules", "total"),
        ("r09", "Modules with site-graph coherence", "Architecture module", "site_graph_modules", "total"),
        ("r10", "Genes with multi-site residual effect", "Architecture module", "multi_site_genes", "total"),
        ("r11", "Strict external anchors", "External anchor", "strict_external_anchors", "total"),
        ("r12", "Confirmed external anchors", "External anchor", "confirmed_external_anchors", "total"),
    ]

    matrix_rows = []
    for _, row in merged.iterrows():
        for row_id, label, group, col, direction in row_defs:
            add_matrix_row(
                matrix_rows,
                row_id,
                label,
                group,
                row["cancer"],
                row.get(col, 0),
                direction=direction,
                source="full_tcga_scp682_main_20260529",
            )
    matrix = pd.DataFrame(matrix_rows)
    matrix["row_order"] = matrix["row_id"].str.extract(r"r(\d+)").astype(int)
    matrix["cancer_order"] = matrix["cancer"].map(order)
    matrix = matrix.sort_values(["row_order", "cancer_order"])

    row_totals = (
        matrix.groupby(["row_id", "row_label", "row_group", "direction", "row_order"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "total_value"})
        .sort_values("row_order")
    )
    row_totals["fraction_of_all_clinical_sites"] = (
        row_totals["total_value"]
        / max(1, int(row_totals.loc[row_totals["row_id"].eq("r02"), "total_value"].iloc[0]))
    )

    merged_out = merged.copy()
    merged_out["cancer_short"] = merged_out["cancer"].map(cancer_short)
    merged_out["cancer_order"] = merged_out["cancer"].map(order)
    merged_out = merged_out.sort_values("cancer_order")

    top_bars.to_csv(OUT / "panel_a_cancer_entry_top_bars.tsv", sep="\t", index=False)
    matrix.to_csv(OUT / "panel_a_cancer_entry_matrix_long.tsv", sep="\t", index=False)
    row_totals.to_csv(OUT / "panel_a_cancer_entry_row_totals.tsv", sep="\t", index=False)
    col_meta.to_csv(OUT / "panel_a_cancer_entry_column_meta.tsv", sep="\t", index=False)
    merged_out.to_csv(OUT / "panel_a_cancer_entry_wide_counts.tsv", sep="\t", index=False)

    audit = {
        "source_dir": str(SRC),
        "output_dir": str(OUT),
        "n_cancers": int(len(cancers)),
        "module_threshold": module_threshold,
        "multiple_testing": "BH-FDR q<0.05, recomputed here from upstream BH q columns (cox_q_full_bh / add_site_to_parent_mrna_lrt_q_bh / gain_q_bh)",
        "clinical_site_definition": "analysis_evaluable and cox_q_full_bh<0.05",
        "site_over_parent_mrna_definition": "clinical(q) and add_site_to_parent_mrna_lrt_q_bh<0.05 (joint Cox incremental LRT, BH-FDR)",
        "graph_residual_definition": "clinical(q) and gain_q_bh<0.05 (BH-FDR)",
        "beyond_mrna_and_graph_residual_definition": "parent_mrna_independent(q) and graph_residual_significant(q)",
        "baseline_parent_mrna_note": "r01 kept at nominal parent_mrna_p<0.05 (baseline reference row; no upstream q column)",
        "module_note": "r08-r10 modules from NMF readouts on the nominal-p site set (not recomputed under FDR)",
        "sample_graph_module_definition": "gene modules with nmf_n_site_over_mrna_sites >= 0.10",
        "site_graph_module_definition": "gene modules with nmf_n_graph_residual_sites >= 0.10",
        "multi_site_definition": "gene modules with n_sites_total >= 3 and nmf_n_site_over_mrna_sites > 0",
        "strict_external_anchor_definition": "unique CPTAC strict candidate targets with TCGA project mapping",
        "confirmed_external_anchor_definition": "unique targets in tcga_confirmed_site_specific_candidates.tsv by TCGA project",
        "strict_external_anchor_source": str(strict_anchor_file),
        "confirmed_external_anchor_source": str(confirmed_anchor_file),
        "files": {
            "top_bars": "panel_a_cancer_entry_top_bars.tsv",
            "matrix_long": "panel_a_cancer_entry_matrix_long.tsv",
            "row_totals": "panel_a_cancer_entry_row_totals.tsv",
            "column_meta": "panel_a_cancer_entry_column_meta.tsv",
            "wide_counts": "panel_a_cancer_entry_wide_counts.tsv",
        },
    }
    with open(OUT / "panel_a_cancer_entry_summary.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, ensure_ascii=False, indent=2)

    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
