#!/usr/bin/env python3
"""Audit the SCP682 v1.0 model contract and TCPA matched RNA/RPPA coverage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/data/lsy/Infinite_Stream")
OUT = ROOT / "02_results/model_validation/20260501_scp682_final_model_contract_v1"

CPTAC_DATA = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2"
CPTAC_RUN = ROOT / "02_results/model_validation/20260429_cptac_joint_total_phosphosite_residual_nas_v2"
TCPA_RUN = ROOT / "02_results/model_validation/20260428_tcpa_pancancer_rppa_film_vae_z_direct_residual_v1"
PUBLIC_ATLAS = ROOT / "02_results/public_bulk_phosphoproteome_atlas/20260430_fixed_v2_bulk_atlas_v1"
TCPA_ALL_TCGA = ROOT / "02_results/model_prediction/20260428_tcpa_rppa_film_vae_z_direct_residual_all_tcga_predictions_v1"


def clean_antibody(name: str) -> str:
    name = str(name)
    if name.startswith("X") and len(name) > 1 and name[1].isdigit():
        return name[1:]
    return name


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tcpa_matched() -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    x_raw = pd.read_parquet(ROOT / "data/processed/X_all.symbols.parquet")
    x_raw = x_raw.drop_duplicates("gene_symbol", keep="first").set_index("gene_symbol")
    rna_samples = set(x_raw.columns.astype(str))

    rppa = pd.read_csv(ROOT / "data/raw/tcpa/PANCAN_RPPA_L4.tsv", sep="\t")
    rppa["sample_id"] = rppa["sample_id"].astype(str)
    rppa = rppa.rename(columns={c: clean_antibody(c) for c in rppa.columns if c != "sample_id"})
    antibody_cols = [c for c in rppa.columns if c != "sample_id"]

    master = pd.read_csv(ROOT / "data/interim/master_index.tsv", sep="\t")
    master["tcga_barcode_full"] = master["tcga_barcode_full"].astype(str)
    master["tcpa_sample_id"] = master["tcpa_sample_id"].astype(str)

    barcode = pd.read_csv(ROOT / "metadata/tcga_barcodes_pancan.tsv", sep="\t")
    barcode["barcode"] = barcode["barcode"].astype(str)
    meta = master.merge(
        barcode[["barcode", "project", "sample_type"]],
        left_on="tcga_barcode_full",
        right_on="barcode",
        how="left",
    )

    merged = rppa[["sample_id"] + antibody_cols].merge(
        meta,
        left_on="sample_id",
        right_on="tcpa_sample_id",
        how="inner",
    )
    merged = merged.loc[merged["tcga_barcode_full"].isin(rna_samples)].copy()
    merged = merged.drop_duplicates("tcga_barcode_full", keep="first")
    merged = merged.sort_values("tcga_barcode_full").set_index("tcga_barcode_full")

    phospho = set((ROOT / "metadata/phospho_proteins.txt").read_text(encoding="utf-8").split())
    target_type = {a: ("phospho" if a in phospho else "total") for a in antibody_cols}
    return merged, antibody_cols, target_type


def load_manifest_projects(path: Path, columns: list[str]) -> set[str]:
    if not path.exists():
        return set()
    tab = pd.read_csv(path, sep="\t")
    for column in columns:
        if column in tab.columns:
            return set(tab[column].dropna().astype(str))
    return set()


def audit_tcpa() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    matched, antibody_cols, target_type = load_tcpa_matched()
    numeric = matched[antibody_cols].apply(pd.to_numeric, errors="coerce")
    total_cols = [a for a in antibody_cols if target_type[a] == "total"]
    phospho_cols = [a for a in antibody_cols if target_type[a] == "phospho"]

    public_tcga_projects = load_manifest_projects(
        PUBLIC_ATLAS / "tables/tcga_supported_tcpa_prediction_manifest.tsv",
        ["tcpa_project", "project"],
    )
    public_cbio_tcpa_projects = load_manifest_projects(
        PUBLIC_ATLAS / "tables/cbioportal_supported_tcpa_prediction_manifest.tsv",
        ["tcpa_project", "project"],
    )
    tcpa_all_projects = load_manifest_projects(
        TCPA_ALL_TCGA / "tables/tcga_all_prediction_manifest.tsv",
        ["tcpa_project", "project"],
    )

    rows = []
    for project, idx in matched.groupby("project", dropna=False).groups.items():
        project_key = "" if pd.isna(project) else str(project)
        sub = numeric.loc[idx]
        coverage = sub.notna().mean(axis=0)
        rows.append(
            {
                "project": project_key,
                "n_matched_rppa_rna_samples": int(sub.shape[0]),
                "n_patients": int(matched.loc[idx, "patient_id"].astype(str).nunique()),
                "n_antibodies_total": int(len(antibody_cols)),
                "n_antibodies_observed_any": int((coverage > 0).sum()),
                "n_total_antibodies_observed_any": int((coverage[total_cols] > 0).sum()),
                "n_phospho_antibodies_observed_any": int((coverage[phospho_cols] > 0).sum()),
                "n_antibodies_observed_ge_20pct_samples": int((coverage >= 0.20).sum()),
                "n_antibodies_observed_ge_50pct_samples": int((coverage >= 0.50).sum()),
                "median_nonmissing_fraction": float(coverage.median()),
                "trainable_min20_samples": bool(sub.shape[0] >= 20),
                "trainable_min30_samples": bool(sub.shape[0] >= 30),
                "in_current_tcpa_training": True,
                "in_gdc_tcga_cptac_supported_manifest": project_key in public_tcga_projects,
                "in_public_cbioportal_supported_manifest": project_key in public_cbio_tcpa_projects,
                "in_tcga_all_prediction_manifest": project_key in tcpa_all_projects,
            }
        )
    project_tab = pd.DataFrame(rows).sort_values(["n_matched_rppa_rna_samples", "project"], ascending=[False, True])

    manifest_cols = ["rna_sample_id", "patient_id", "tcpa_sample_id", "project", "sample_type", "rppa_sample_id"]
    sample_manifest = matched.reset_index().rename(
        columns={
            "tcga_barcode_full": "rna_sample_id",
            "sample_id": "rppa_sample_id",
        }
    )
    keep_cols = [c for c in manifest_cols if c in sample_manifest.columns]
    sample_manifest = sample_manifest[keep_cols].sort_values("rna_sample_id")

    summary = {
        "source": {
            "rppa": str(ROOT / "data/raw/tcpa/PANCAN_RPPA_L4.tsv"),
            "rna": str(ROOT / "data/processed/X_all.symbols.parquet"),
            "master_index": str(ROOT / "data/interim/master_index.tsv"),
            "barcode_metadata": str(ROOT / "metadata/tcga_barcodes_pancan.tsv"),
        },
        "n_matched_samples": int(matched.shape[0]),
        "n_projects": int(project_tab["project"].nunique()),
        "n_trainable_projects_min20": int(project_tab["trainable_min20_samples"].sum()),
        "n_trainable_projects_min30": int(project_tab["trainable_min30_samples"].sum()),
        "n_antibodies": int(len(antibody_cols)),
        "n_total_antibodies": int(len(total_cols)),
        "n_phospho_antibodies": int(len(phospho_cols)),
        "projects": project_tab["project"].tolist(),
    }
    return project_tab, sample_manifest, summary


def write_cptac_contexts() -> tuple[pd.DataFrame, dict]:
    card = read_json(CPTAC_DATA / "MULTITASK_DATA_CARD.json")
    sample_meta = pd.read_csv(CPTAC_DATA / "sample_manifest.tsv", sep="\t")
    rename = {"cancer_label": "cptac_pdc_context", "pdc_study_id": "pdc_study_id"}
    sample_meta = sample_meta.rename(columns=rename)
    group_cols = [c for c in ["cptac_pdc_context", "pdc_study_id"] if c in sample_meta.columns]
    if group_cols:
        contexts = sample_meta.groupby(group_cols, dropna=False).size().reset_index(name="n_samples")
        contexts = contexts.sort_values(group_cols)
    else:
        contexts = pd.DataFrame()
    return contexts, card


def cptac_performance() -> dict:
    tab = pd.read_csv(CPTAC_RUN / "tables/best_config_full5fold_metrics.tsv", sep="\t")
    return {
        "config_id": str(tab["config_id"].iloc[0]),
        "n_folds": int(tab["fold"].nunique()),
        "total_median_spearman_mean_across_folds": float(tab["total_median_spearman"].mean()),
        "total_median_spearman_median_across_folds": float(tab["total_median_spearman"].median()),
        "phosphosite_median_spearman_mean_across_folds": float(tab["phospho_median_spearman"].mean()),
        "phosphosite_median_spearman_median_across_folds": float(tab["phospho_median_spearman"].median()),
        "fold_table": str(CPTAC_RUN / "tables/best_config_full5fold_metrics.tsv"),
    }


def tcpa_performance() -> dict:
    summary = read_json(TCPA_RUN / "logs/tcpa_pancancer_rppa_film_vae_z_direct_residual_v1_summary.json")
    by_panel = {x["panel"]: x for x in summary["target_summary"]}
    return {
        "n_samples": int(summary["n_samples"]),
        "n_antibodies": int(summary["n_antibodies"]),
        "n_total_antibodies": int(summary["n_total_antibodies"]),
        "n_phospho_antibodies": int(summary["n_phospho_antibodies"]),
        "all_antibody_median_spearman": float(by_panel["all"]["median_spearman"]),
        "total_antibody_median_spearman": float(by_panel["total"]["median_spearman"]),
        "phospho_antibody_median_spearman": float(by_panel["phospho"]["median_spearman"]),
        "sample_spearman_median": float(summary["sample_spearman_median"]),
    }


def write_contract(
    cptac_card: dict,
    cptac_perf: dict,
    tcpa_audit: dict,
    tcpa_perf: dict,
) -> tuple[pd.DataFrame, dict]:
    rows = [
        {
            "layer": "CPTAC/PDC total protein",
            "branch": "CPTAC/PDC v2 branch",
            "training_data_contract": "pancancer_multi_task_locked_v2",
            "source_run": str(CPTAC_RUN),
            "n_training_samples": int(cptac_card["n_samples"]),
            "n_training_contexts_or_projects": int(cptac_card["n_cancer_contexts"]),
            "n_outputs": int(cptac_card["n_total_protein_genes_min20pct"]),
            "validation_metric": "full 5-fold mean of fold median Spearman",
            "validation_value": cptac_perf["total_median_spearman_mean_across_folds"],
            "deployment_role": "final",
        },
        {
            "layer": "CPTAC/PDC phosphosite",
            "branch": "CPTAC/PDC v2 branch",
            "training_data_contract": "pancancer_multi_task_locked_v2",
            "source_run": str(CPTAC_RUN),
            "n_training_samples": int(cptac_card["n_samples"]),
            "n_training_contexts_or_projects": int(cptac_card["n_cancer_contexts"]),
            "n_outputs": int(cptac_card["n_phosphosite_gene_site_min20pct"]),
            "validation_metric": "full 5-fold mean of fold median Spearman",
            "validation_value": cptac_perf["phosphosite_median_spearman_mean_across_folds"],
            "deployment_role": "final",
        },
        {
            "layer": "TCPA total antibody",
            "branch": "TCPA RPPA branch",
            "training_data_contract": "full matched TCGA RNA + TCPA RPPA audit 20260501",
            "source_run": str(TCPA_RUN),
            "n_training_samples": int(tcpa_audit["n_matched_samples"]),
            "n_training_contexts_or_projects": int(tcpa_audit["n_projects"]),
            "n_outputs": int(tcpa_perf["n_total_antibodies"]),
            "validation_metric": "full matched 5-fold antibody median Spearman",
            "validation_value": tcpa_perf["total_antibody_median_spearman"],
            "deployment_role": "final",
        },
        {
            "layer": "TCPA phospho antibody",
            "branch": "TCPA RPPA branch",
            "training_data_contract": "full matched TCGA RNA + TCPA RPPA audit 20260501",
            "source_run": str(TCPA_RUN),
            "n_training_samples": int(tcpa_audit["n_matched_samples"]),
            "n_training_contexts_or_projects": int(tcpa_audit["n_projects"]),
            "n_outputs": int(tcpa_perf["n_phospho_antibodies"]),
            "validation_metric": "full matched 5-fold antibody median Spearman",
            "validation_value": tcpa_perf["phospho_antibody_median_spearman"],
            "deployment_role": "final",
        },
    ]
    contract = pd.DataFrame(rows)
    payload = {
        "model_name": "SCP682",
        "model_version": "v1.0",
        "definition": "pan-cancer bulk RNA to four-layer protein-state prediction model",
        "model_contract_sentence": "SCP682 uses all curated trainable CPTAC/PDC and TCPA matched RNA-protein cohorts available under the current data contract.",
        "model_contract_sentence_zh": "SCP682 使用当前数据合同下所有可配对、可训练的 CPTAC/PDC 与 TCPA RNA-蛋白质标签队列。",
        "cptac_pdc_branch": {
            "data_contract": "pancancer_multi_task_locked_v2",
            "source_run": str(CPTAC_RUN),
            "n_samples": int(cptac_card["n_samples"]),
            "n_cancer_contexts": int(cptac_card["n_cancer_contexts"]),
            "n_total_outputs": int(cptac_card["n_total_protein_genes_min20pct"]),
            "n_phosphosite_outputs": int(cptac_card["n_phosphosite_gene_site_min20pct"]),
            "performance": cptac_perf,
        },
        "tcpa_rppa_branch": {
            "data_contract": "full matched TCGA RNA + TCPA RPPA audit 20260501",
            "source_run": str(TCPA_RUN),
            "audit": tcpa_audit,
            "performance": tcpa_perf,
        },
        "historical_v1_policy": "historical benchmark only; not a replacement component for the final SCP682 v1.0 website model",
    }
    return contract, payload


def main() -> int:
    for sub in ["tables", "logs", "reports"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    tcpa_project_tab, tcpa_sample_manifest, tcpa_audit_summary = audit_tcpa()
    tcpa_project_tab.to_csv(OUT / "tables/tcpa_full_matched_project_coverage.tsv", sep="\t", index=False)
    tcpa_sample_manifest.to_csv(OUT / "tables/tcpa_full_matched_sample_manifest.tsv", sep="\t", index=False)

    cptac_contexts, cptac_card = write_cptac_contexts()
    cptac_contexts.to_csv(OUT / "tables/cptac_pdc_v2_contexts.tsv", sep="\t", index=False)

    cptac_perf = cptac_performance()
    tcpa_perf = tcpa_performance()
    contract, contract_json = write_contract(cptac_card, cptac_perf, tcpa_audit_summary, tcpa_perf)
    contract.to_csv(OUT / "tables/final_model_contract.tsv", sep="\t", index=False)

    (OUT / "logs/tcpa_full_matched_audit_summary.json").write_text(
        json.dumps(tcpa_audit_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUT / "logs/scp682_final_model_contract.json").write_text(
        json.dumps(contract_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = [
        "# SCP682 v1.0 final model contract",
        "",
        "SCP682 uses all curated trainable CPTAC/PDC and TCPA matched RNA-protein cohorts available under the current data contract.",
        "",
        f"CPTAC/PDC branch: pancancer_multi_task_locked_v2, {cptac_card['n_samples']} samples, {cptac_card['n_cancer_contexts']} contexts.",
        f"TCPA branch audit: {tcpa_audit_summary['n_matched_samples']} matched RNA/RPPA samples, {tcpa_audit_summary['n_projects']} projects.",
        "",
        "Layer performance uses the final website model components only; historical v1 components are benchmarks, not replacement heads.",
        "",
        "## Methods wording",
        "",
        "SCP682 v1.0 was defined as a pan-cancer bulk RNA-to-protein-state prediction model with two independent supervised branches. The CPTAC/PDC branch was trained under the pancancer_multi_task_locked_v2 data contract and predicts mass-spectrometry total-protein and phosphosite matrices. The TCPA branch was trained from the audited matched TCGA RNA and TCPA RPPA data contract and predicts total-antibody and phospho-antibody RPPA matrices. All four layers are exposed through the same website/API, but their measurement semantics and output spaces are kept separate.",
        "",
        "中文：SCP682 v1.0 定义为泛癌 bulk RNA 到蛋白质状态预测模型，包含两个独立监督分支。CPTAC/PDC 分支使用 pancancer_multi_task_locked_v2 数据合同，预测质谱总蛋白和质谱磷酸化位点矩阵。TCPA 分支使用重新审计后的 matched TCGA RNA 与 TCPA RPPA 数据合同，预测 total antibody 和 phospho antibody RPPA 矩阵。四层结果通过同一个网站/API 输出，但测量语义和输出空间保持分离。",
    ]
    (OUT / "reports/scp682_final_model_contract.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps(contract_json, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
