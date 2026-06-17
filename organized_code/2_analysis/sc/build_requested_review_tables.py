from pathlib import Path
import shutil
import math

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


ROOT = Path(__file__).resolve().parents[2]
KEY = ROOT / "01_key_results"
DT = ROOT / "02_data_tables"
FIG3 = ROOT / "04_figure_source_data" / "fig3"
OUT = ROOT / "04_figure_source_data" / "reviewer_requested_tables_v1"
OUT.mkdir(parents=True, exist_ok=True)


def copy_table(src: Path, dst_name: str):
    if src.exists():
        dst = OUT / dst_name
        shutil.copy2(src, dst)
        return dst
    return None


def safe_wilcoxon(x):
    x = pd.Series(x).dropna()
    x = x[x != 0]
    if len(x) < 2 or wilcoxon is None:
        return np.nan
    try:
        return float(wilcoxon(x).pvalue)
    except Exception:
        return np.nan


def build_random_control_attention():
    src = DT / "pathway_attention_by_dataset_target.tsv"
    att = pd.read_csv(src, sep="\t")
    bio = att[att["pathway"] != "random_control"].copy()
    idx = bio.groupby(["dataset", "target_id"])["mean_attention"].idxmax()
    max_bio = bio.loc[idx, ["dataset", "target_id", "pathway", "mean_attention"]].rename(
        columns={"pathway": "max_biological_pathway", "mean_attention": "max_biological_attention"}
    )
    rc = att[att["pathway"] == "random_control"][["dataset", "target_id", "mean_attention"]].rename(
        columns={"mean_attention": "random_control_attention"}
    )
    out = max_bio.merge(rc, on=["dataset", "target_id"], how="left")
    out["delta_biological_minus_random"] = out["max_biological_attention"] - out["random_control_attention"]
    out["ratio_biological_to_random"] = out["max_biological_attention"] / out["random_control_attention"].replace(0, np.nan)
    out.to_csv(OUT / "random_control_attention_contrast.tsv", sep="\t", index=False)

    summ = out.groupby("dataset").agg(
        n_readouts=("target_id", "nunique"),
        median_max_biological_attention=("max_biological_attention", "median"),
        median_random_control_attention=("random_control_attention", "median"),
        median_delta=("delta_biological_minus_random", "median"),
    ).reset_index()
    pvals = []
    for ds, g in out.groupby("dataset"):
        pvals.append({"dataset": ds, "paired_wilcoxon_p": safe_wilcoxon(g["delta_biological_minus_random"])})
    summ = summ.merge(pd.DataFrame(pvals), on="dataset", how="left")
    summ.to_csv(OUT / "random_control_attention_summary.tsv", sep="\t", index=False)


def iter_predicted_observed():
    pred_dir = DT / "external_predicted_observed"
    for p in sorted(pred_dir.glob("scp682_sc11_predicted_observed_*.tsv")):
        if "manifest" in p.name or "per_target" in p.name:
            continue
        yield p, pd.read_csv(p, sep="\t")


def build_output_collapse_tables():
    rows = []
    for p, df in iter_predicted_observed():
        for (cohort, target), g in df.groupby(["cohort_id", "target_id"]):
            if len(g) < 3:
                continue
            pred = pd.to_numeric(g["predicted"], errors="coerce")
            obs = pd.to_numeric(g["observed"], errors="coerce")
            valid = pred.notna() & obs.notna()
            pred = pred[valid]
            obs = obs[valid]
            if len(pred) < 3:
                continue
            pred_sd = float(pred.std(ddof=1))
            obs_sd = float(obs.std(ddof=1))
            rows.append({
                "cohort_id": cohort,
                "target_id": target,
                "n": int(len(pred)),
                "predicted_mean": float(pred.mean()),
                "observed_mean": float(obs.mean()),
                "predicted_sd": pred_sd,
                "observed_sd": obs_sd,
                "sd_ratio_predicted_over_observed": pred_sd / obs_sd if obs_sd else np.nan,
                "spearman": float(pred.corr(obs, method="spearman")),
                "pearson": float(pred.corr(obs, method="pearson")),
                "source_file": str(p),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "output_collapse_check_by_cohort_target.tsv", sep="\t", index=False)
    summ = out.groupby("cohort_id").agg(
        n_readouts=("target_id", "nunique"),
        median_predicted_sd=("predicted_sd", "median"),
        median_observed_sd=("observed_sd", "median"),
        median_sd_ratio=("sd_ratio_predicted_over_observed", "median"),
        median_spearman=("spearman", "median"),
    ).reset_index()
    summ.to_csv(OUT / "output_collapse_check_by_cohort_summary.tsv", sep="\t", index=False)


def build_calibration_tables(n_bins=10):
    rows = []
    summ_rows = []
    for p, df in iter_predicted_observed():
        for (cohort, target), g in df.groupby(["cohort_id", "target_id"]):
            g = g[["predicted", "observed"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(g) < n_bins * 5 or g["predicted"].nunique() < n_bins:
                continue
            try:
                g["prediction_bin"] = pd.qcut(g["predicted"], q=n_bins, labels=False, duplicates="drop") + 1
            except Exception:
                continue
            per_bin = g.groupby("prediction_bin").agg(
                n=("predicted", "size"),
                predicted_mean=("predicted", "mean"),
                predicted_min=("predicted", "min"),
                predicted_max=("predicted", "max"),
                observed_mean=("observed", "mean"),
                observed_sd=("observed", "std"),
            ).reset_index()
            per_bin["cohort_id"] = cohort
            per_bin["target_id"] = target
            per_bin["calibration_error_observed_minus_predicted"] = per_bin["observed_mean"] - per_bin["predicted_mean"]
            rows.append(per_bin)
            summ_rows.append({
                "cohort_id": cohort,
                "target_id": target,
                "n": int(len(g)),
                "n_bins": int(per_bin["prediction_bin"].nunique()),
                "mean_abs_calibration_error": float(per_bin["calibration_error_observed_minus_predicted"].abs().mean()),
                "median_abs_calibration_error": float(per_bin["calibration_error_observed_minus_predicted"].abs().median()),
                "spearman_bin_mean": float(per_bin["predicted_mean"].corr(per_bin["observed_mean"], method="spearman")),
                "source_file": str(p),
            })
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(OUT / "calibration_bins_by_cohort_target.tsv", sep="\t", index=False)
    pd.DataFrame(summ_rows).to_csv(OUT / "calibration_summary_by_cohort_target.tsv", sep="\t", index=False)


def build_gnn_vs_mlp_tables():
    formal = pd.read_csv(DT / "scp682_sc11_formal_internal_5fold_per_target.tsv", sep="\t")
    mlp = pd.read_csv(DT / "internal_5fold_scfoundation_site_aware_mlp_per_target.tsv", sep="\t")
    formal = formal[(formal["evaluation"] == "internal_cv_reconstruction") & (formal["test_dataset"].isin(["iccite_seq_tcell_2025", "qurie_seq_bjab_2021"]))]
    formal["fold"] = formal["fold"].astype(str)
    formal = formal[["fold", "test_dataset", "target_id", "n", "spearman", "pearson"]].rename(
        columns={"n": "n_scp682_sc", "spearman": "scp682_sc_spearman", "pearson": "scp682_sc_pearson"}
    )
    mlp = mlp[(mlp["evaluation"] == "internal_5fold") & (mlp["test_dataset"].isin(["iccite_seq_tcell_2025", "qurie_seq_bjab_2021"]))]
    mlp["fold"] = mlp["fold"].astype(str)
    mlp = mlp[["fold", "test_dataset", "target_id", "n", "spearman", "pearson", "method"]].rename(
        columns={"n": "n_site_aware_mlp", "spearman": "site_aware_mlp_spearman", "pearson": "site_aware_mlp_pearson"}
    )
    paired = formal.merge(mlp, on=["fold", "test_dataset", "target_id"], how="inner")
    paired["delta_scp682_sc_minus_mlp"] = paired["scp682_sc_spearman"] - paired["site_aware_mlp_spearman"]
    paired.to_csv(OUT / "gnn_vs_site_aware_mlp_paired_per_fold_target.tsv", sep="\t", index=False)

    summ = paired.groupby("test_dataset").agg(
        n_pairs=("target_id", "count"),
        n_targets=("target_id", "nunique"),
        median_scp682_sc_spearman=("scp682_sc_spearman", "median"),
        median_site_aware_mlp_spearman=("site_aware_mlp_spearman", "median"),
        median_delta=("delta_scp682_sc_minus_mlp", "median"),
        mean_delta=("delta_scp682_sc_minus_mlp", "mean"),
    ).reset_index()
    pvals = []
    for ds, g in paired.groupby("test_dataset"):
        pvals.append({"test_dataset": ds, "paired_wilcoxon_p": safe_wilcoxon(g["delta_scp682_sc_minus_mlp"])})
    summ = summ.merge(pd.DataFrame(pvals), on="test_dataset", how="left")
    summ.to_csv(OUT / "gnn_vs_site_aware_mlp_summary.tsv", sep="\t", index=False)


def build_fivefold_stability():
    formal = pd.read_csv(DT / "scp682_sc11_formal_internal_5fold_per_target.tsv", sep="\t")
    formal = formal[formal["evaluation"] == "internal_cv_reconstruction"].copy()
    out = formal.groupby(["test_dataset", "target_id"]).agg(
        n_folds=("fold", "nunique"),
        median_spearman=("spearman", "median"),
        mean_spearman=("spearman", "mean"),
        sd_spearman=("spearman", "std"),
        min_spearman=("spearman", "min"),
        max_spearman=("spearman", "max"),
        median_n=("n", "median"),
    ).reset_index()
    out.to_csv(OUT / "fivefold_stability_by_readout.tsv", sep="\t", index=False)


def build_training_curve_summary():
    curve = pd.read_csv(DT / "training_curves.tsv", sep="\t")
    out = curve.groupby(["split", "loss_name"]).agg(
        n_epochs=("epoch", "nunique"),
        first_epoch=("epoch", "min"),
        last_epoch=("epoch", "max"),
        first_value=("loss_value", "first"),
        last_value=("loss_value", "last"),
        min_value=("loss_value", "min"),
        max_value=("loss_value", "max"),
    ).reset_index()
    out.to_csv(OUT / "training_curve_summary.tsv", sep="\t", index=False)


def build_manifest():
    existing = [
        ("benchmark_table_available.tsv", "existing", "Benchmark summary currently present in 01_key_results."),
        ("ablation_results_available.tsv", "existing", "Component/variant ablation summary currently present in 01_key_results."),
        ("internal_5fold_scfoundation_site_aware_mlp_per_target.tsv", "existing", "Site-aware MLP comparison source."),
        ("scp682_sc11_formal_internal_5fold_per_target.tsv", "existing", "Formal SCP682-SC five-fold per-target source."),
        ("scp682_sc11_formal_internal_5fold_summary_by_fold.tsv", "existing", "Formal five-fold fold-level source."),
        ("training_curves.tsv", "existing", "Training curve source."),
        ("pathway_attention_by_dataset_target.tsv", "existing", "Pathway attention source with random_control rows."),
        ("bulk_site_graph_matched_ablation_per_target.tsv", "existing", "Graph residual ablation per target."),
        ("qurie_ibrutinib_delta_per_target.tsv", "existing", "QuRIE ibrutinib training-distribution delta summary."),
        ("sensitivity_scan.tsv", "existing", "Antibody clone sensitivity source."),
        ("fig19_per_readout_difficulty.tsv", "existing", "Per-readout difficulty source."),
    ]
    derived = [
        ("random_control_attention_contrast.tsv", "derived", "Per-readout biological-token maximum versus random_control attention."),
        ("random_control_attention_summary.tsv", "derived", "Dataset-level paired attention summary."),
        ("output_collapse_check_by_cohort_target.tsv", "derived", "Predicted SD versus observed SD per cohort and readout."),
        ("output_collapse_check_by_cohort_summary.tsv", "derived", "Dataset-level output-collapse summary."),
        ("calibration_bins_by_cohort_target.tsv", "derived", "Prediction-bin calibration table."),
        ("calibration_summary_by_cohort_target.tsv", "derived", "Per-readout calibration summary."),
        ("gnn_vs_site_aware_mlp_paired_per_fold_target.tsv", "derived", "Formal model versus site-aware MLP paired rows."),
        ("gnn_vs_site_aware_mlp_summary.tsv", "derived", "Formal model versus site-aware MLP summary with paired Wilcoxon where available."),
        ("fivefold_stability_by_readout.tsv", "derived", "Fold-to-fold stability per readout."),
        ("training_curve_summary.tsv", "derived", "Training curve compact summary."),
    ]
    missing = [
        ("mean_baseline_per_target", "missing", "No per-target mean baseline source table found in SCP682-SC package."),
        ("cognate_mRNA_per_target", "missing", "No per-target cognate-mRNA baseline table found in SCP682-SC package."),
        ("pathway_attention_removed_ablation", "missing", "No strict matched run with pathway-attention removed found."),
        ("scFoundation_removed_raw_expression_ablation", "missing", "No strict matched run replacing scFoundation with raw-expression-only input found."),
        ("drug_delta_removed_ablation", "not_applicable_current_formal", "Formal SC reconstruction model used delta weight 0 in the archived main result; no delta-removal contrast applies to this result."),
        ("high_error_cell_qc_metadata", "missing", "Predicted/observed tables do not contain UMI depth, detected genes, or mitochondrial fraction."),
        ("SC_to_pseudobulk_bulk_match", "missing", "No matched single-cell pseudobulk and bulk phospho table found in the SC paper package."),
    ]
    rows = []
    for fname, status, note in existing + derived + missing:
        path = OUT / fname
        rows.append({
            "table_or_item": fname,
            "status": status if status.startswith("missing") or status.startswith("not_") else ("available" if path.exists() else status),
            "path": str(path) if path.exists() else "",
            "note": note,
        })
    pd.DataFrame(rows).to_csv(OUT / "TABLE_MANIFEST.tsv", sep="\t", index=False)
    with open(OUT / "TABLE_MANIFEST.md", "w", encoding="utf-8") as f:
        f.write("# reviewer_requested_tables_v1\n\n")
        f.write("This folder collects source tables for reviewer-facing SCP682-SC controls and extended data panels.\n\n")
        f.write("| table_or_item | status | note |\n|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['table_or_item']} | {r['status']} | {r['note']} |\n")


def main():
    copy_table(KEY / "benchmark_table.tsv", "benchmark_table_available.tsv")
    copy_table(KEY / "ablation_results.tsv", "ablation_results_available.tsv")
    copy_table(DT / "internal_5fold_scfoundation_site_aware_mlp_per_target.tsv", "internal_5fold_scfoundation_site_aware_mlp_per_target.tsv")
    copy_table(DT / "scp682_sc11_formal_internal_5fold_per_target.tsv", "scp682_sc11_formal_internal_5fold_per_target.tsv")
    copy_table(DT / "scp682_sc11_formal_internal_5fold_summary_by_fold.tsv", "scp682_sc11_formal_internal_5fold_summary_by_fold.tsv")
    copy_table(DT / "training_curves.tsv", "training_curves.tsv")
    copy_table(DT / "pathway_attention_by_dataset_target.tsv", "pathway_attention_by_dataset_target.tsv")
    copy_table(DT / "bulk_site_graph_matched_ablation_per_target.tsv", "bulk_site_graph_matched_ablation_per_target.tsv")
    copy_table(DT / "bulk_site_graph_matched_ablation_by_cohort.tsv", "bulk_site_graph_matched_ablation_by_cohort.tsv")
    copy_table(DT / "qurie_ibrutinib_delta_per_target.tsv", "qurie_ibrutinib_delta_per_target.tsv")
    copy_table(KEY / "sensitivity_scan.tsv", "sensitivity_scan.tsv")
    copy_table(FIG3 / "fig19_per_readout_difficulty.tsv", "fig19_per_readout_difficulty.tsv")
    copy_table(FIG3 / "fig13_nmf_gse300551_H.tsv", "fig13_nmf_gse300551_H.tsv")
    copy_table(FIG3 / "fig13_nmf_hela_H.tsv", "fig13_nmf_hela_H.tsv")
    copy_table(FIG3 / "fig13_nmf_pdo_caf_H.tsv", "fig13_nmf_pdo_caf_H.tsv")
    copy_table(FIG3 / "fig13_nmf_vivo_seq_th17_H.tsv", "fig13_nmf_vivo_seq_th17_H.tsv")
    copy_table(FIG3 / "fig15_cross_cohort_hallmark_matrix.tsv", "fig15_cross_cohort_hallmark_matrix.tsv")
    copy_table(FIG3 / "fig16_cell_cycle_marker_hits.tsv", "fig16_cell_cycle_marker_hits.tsv")
    copy_table(FIG3 / "fig17_cross_cohort_phospho_rna.tsv", "fig17_cross_cohort_phospho_rna.tsv")
    copy_table(FIG3 / "fig18_landscape_summary.tsv", "fig18_landscape_summary.tsv")
    copy_table(FIG3 / "panel_external_external_predicted_observed_manifest.tsv", "external_predicted_observed_manifest.tsv")
    copy_table(FIG3 / "panel_external_external_per_target_from_predicted_observed.tsv", "external_per_target_from_predicted_observed.tsv")

    build_random_control_attention()
    build_output_collapse_tables()
    build_calibration_tables()
    build_gnn_vs_mlp_tables()
    build_fivefold_stability()
    build_training_curve_summary()
    build_manifest()
    print(OUT)


if __name__ == "__main__":
    main()
