#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/data/lsy/Infinite_Stream")
PYTHON = Path("/data/lsy/conda-envs/3ef0fd2916e30222c6bfbc5c753696fe_/bin/python")
PACKAGE = ROOT / "SCP682_PORTABLE"
OUT = ROOT / "SCP682-main/results/20260530_fig2c_additional_external_scp682_main"

APOLLO_SCRIPT = ROOT / "03_code/external_validation/proteogenomics/deploy_v38_apollo_luad_and_evaluate_true_phosphosite_20260503.py"
LUAD_SCRIPT = ROOT / "03_code/external_validation/phosphoproteomics/deploy_v38_luad_cas_and_evaluate_true_phosphosite_20260502.py"
OLD_APOLLO = ROOT / "02_results/external_validation/20260503_v38_apollo_luad_predicted_vs_true_phosphosite"
OLD_LUAD = ROOT / "02_results/external_validation/20260502_v38_luad_cas_fpkm_to_tpm_predicted_vs_true_phosphosite"
ORIGINAL_EXTERNAL = ROOT / "SCP682-main/results/20260523_general_graph_external_fixed_anchor/tables"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sample_key(x: object) -> str:
    return str(x).split("::", 1)[-1]


def write_matrix(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "sample_id"
    out.reset_index().to_csv(path, sep="\t", index=False)


def make_manifest(index: pd.Index, cancer: str, study: str, source: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": index.astype(str),
            "source_dataset": source,
            "source_sample_id": [sample_key(x) for x in index],
            "cptac_cancer_label": cancer,
            "cptac_study_id": study,
        }
    )


def run_predict(label: str, rna: pd.DataFrame, manifest: pd.DataFrame, force: bool) -> Path:
    pred_dir = OUT / "predictions" / label
    pred_file = pred_dir / "predictions" / "scp682_main_phosphosite.parquet"
    if pred_file.exists() and not force:
        return pred_file
    rna_path = OUT / "inputs" / f"{label}_rna_log2tpm.tsv"
    manifest_path = OUT / "inputs" / f"{label}_manifest.tsv"
    write_matrix(rna, rna_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, sep="\t", index=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PACKAGE}:{ROOT / '.python_targets/pyg_torch251'}:" + env.get("PYTHONPATH", "")
    cmd = [
        str(PYTHON),
        str(PACKAGE / "predict_scp682.py"),
        "--rna",
        str(rna_path),
        "--manifest",
        str(manifest_path),
        "--rna-scale",
        "log2tpm",
        "--outdir",
        str(pred_dir),
        "--device",
        "cuda:0",
        "--v4-batch-size",
        "32",
        "--graph-batch-size",
        "2",
    ]
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{label}.stdout.log").open("w", encoding="utf-8") as stdout, (
        log_dir / f"{label}.stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        subprocess.run(cmd, check=True, env=env, stdout=stdout, stderr=stderr)
    return pred_file


def spearman_rows(y: pd.DataFrame, p: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in y.columns:
        yy = y[target].to_numpy(dtype=float)
        pp = p[target].to_numpy(dtype=float)
        ok = np.isfinite(yy) & np.isfinite(pp)
        n = int(ok.sum())
        rho = np.nan
        pval = np.nan
        if n >= min_n:
            res = spearmanr(yy[ok], pp[ok])
            rho = float(res.correlation) if np.isfinite(res.correlation) else np.nan
            pval = float(res.pvalue) if np.isfinite(res.pvalue) else np.nan
        rows.append({"target": target, "n": n, "spearman": rho, "rho_p_value": pval})
    return pd.DataFrame(rows)


def sample_rows(y: pd.DataFrame, p: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample in y.index:
        yy = y.loc[sample].to_numpy(dtype=float)
        pp = p.loc[sample].to_numpy(dtype=float)
        ok = np.isfinite(yy) & np.isfinite(pp)
        n = int(ok.sum())
        rho = np.nan
        pval = np.nan
        if n >= min_n:
            res = spearmanr(yy[ok], pp[ok])
            rho = float(res.correlation) if np.isfinite(res.correlation) else np.nan
            pval = float(res.pvalue) if np.isfinite(res.pvalue) else np.nan
        rows.append({"sample_id": sample, "n": n, "spearman": rho, "rho_p_value": pval})
    return pd.DataFrame(rows)


def bootstrap_median(vals: pd.Series, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    arr = vals.dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    med = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        med[i] = np.median(rng.choice(arr, size=arr.size, replace=True))
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def evaluate(label: str, pred_path: Path, true_path: Path, min_site_n: int, min_sample_n: int) -> dict[str, object]:
    pred = pd.read_parquet(pred_path)
    true = pd.read_parquet(true_path)
    pred2 = pred.copy()
    pred2.index = [sample_key(x) for x in pred2.index]
    true.index = true.index.astype(str)
    samples = [s for s in true.index if s in pred2.index]
    targets = [t for t in pred2.columns if t in true.columns]
    y = true.loc[samples, targets]
    p = pred2.loc[samples, targets]
    per_site = spearman_rows(y, p, min_site_n)
    per_sample = sample_rows(y, p, min_sample_n)
    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    per_site.to_csv(tables / f"{label}_scp682_main_phosphosite_per_target.tsv", sep="\t", index=False)
    per_sample.to_csv(tables / f"{label}_scp682_main_phosphosite_per_sample.tsv", sep="\t", index=False)
    lo, hi = bootstrap_median(per_site["spearman"])
    summary = {
        "dataset": label,
        "layer": "phosphosite",
        "model": "SCP682_main",
        "n_rna_samples": int(pred.shape[0]),
        "n_matched_samples": int(len(samples)),
        "n_predicted_targets": int(pred.shape[1]),
        "n_true_targets": int(true.shape[1]),
        "n_matched_targets": int(len(targets)),
        "targets_tested": int(per_site["spearman"].notna().sum()),
        "median_spearman": float(per_site["spearman"].median(skipna=True)),
        "mean_spearman": float(per_site["spearman"].mean(skipna=True)),
        "ci95_low": lo,
        "ci95_high": hi,
        "ge_0_1": int((per_site["spearman"] >= 0.1).sum()),
        "ge_0_2": int((per_site["spearman"] >= 0.2).sum()),
        "ge_0_3": int((per_site["spearman"] >= 0.3).sum()),
        "ge_0_5": int((per_site["spearman"] >= 0.5).sum()),
        "sample_median_spearman": float(per_sample["spearman"].median(skipna=True)),
        "per_target_file": str(tables / f"{label}_scp682_main_phosphosite_per_target.tsv"),
    }
    return summary


def collect_original_plot_rows() -> list[dict[str, object]]:
    summary = pd.read_csv(ORIGINAL_EXTERNAL / "scp682_general_graph_external_summary.tsv", sep="\t")
    rows: list[dict[str, object]] = []
    labels = {
        "fu_icca": ("FU-iCCA", "RNA-seq", "FU-iCCA"),
        "tu_sclc": ("TU-SCLC", "RNA-seq", "TU-SCLC"),
        "chcc_hbv_fpkm": ("CHCC-HBV FPKM", "FPKM", "CHCC-HBV"),
        "chcc_hbv_rsem": ("CHCC-HBV RSEM", "RSEM", "CHCC-HBV"),
    }
    for key, (cohort_label, input_setting, independent) in labels.items():
        row = summary.loc[
            (summary["dataset"].eq(key))
            & (summary["model"].eq("scp682_general_graph_residual_sample_centered"))
        ].iloc[0]
        per_file = ORIGINAL_EXTERNAL / f"{key}_scp682_general_graph_residual_sample_centered_phosphosite_per_target.tsv"
        per = pd.read_csv(per_file, sep="\t")
        lo, hi = bootstrap_median(per["spearman"])
        rows.append(
            {
                "cohort_key": key,
                "cohort_label": cohort_label,
                "independent_cohort": independent,
                "input_setting": input_setting,
                "n_samples": int(row["n_matched_samples"]),
                "n_sites": int(row["n_matched_targets"]),
                "targets_tested": int(row["targets_tested"]),
                "median_spearman": float(row["median_spearman"]),
                "mean_spearman": float(row["mean_spearman"]),
                "ci95_low": lo,
                "ci95_high": hi,
                "model": "SCP682",
                "plot_include": True,
                "status": "complete",
                "source_file": str(per_file),
                "note": "existing SCP682 main external result; CHCC-HBV FPKM/RSEM are two RNA input settings of one independent cohort",
            }
        )
    return rows


def unavailable_audit_rows() -> list[dict[str, object]]:
    rows = []
    cnhpp_phospho = ROOT / "01_data/external_phosphoproteomics/processed/cnhpp_gc_2023/cnhpp_gc_2023_phosphosite_gene_site_matrix.parquet"
    cnhpp_tfre = ROOT / "01_data/external_phosphoproteomics/processed/cnhpp_gc_2023/cnhpp_gc_2023_tfre_gene_matrix.parquet"
    hnscc_phospho = ROOT / "01_data/external_phosphoproteomics/processed/hnscc_pxd030343/hnscc_pxd030343_phosphosite_gene_site_matrix.parquet"
    hnscc_proteome = ROOT / "01_data/external_phosphoproteomics/processed/hnscc_pxd030343/hnscc_pxd030343_proteome_gene_matrix.parquet"
    for key, label, phospho_path, candidate_path, reason in [
        (
            "cnhpp_gc_2023",
            "CNHPP GC",
            cnhpp_phospho,
            cnhpp_tfre,
            "paired bulk RNA matrix not present; available matrix is TF activity, not RNA expression",
        ),
        (
            "pxd030343_hnscc",
            "PXD030343 HNSCC",
            hnscc_phospho,
            hnscc_proteome,
            "paired bulk RNA matrix not present; available matrix is proteome, not RNA expression",
        ),
    ]:
        phospho_shape = [None, None]
        candidate_shape = [None, None]
        if phospho_path.exists():
            x = pd.read_parquet(phospho_path)
            phospho_shape = [int(x.shape[0]), int(x.shape[1])]
        if candidate_path.exists():
            x = pd.read_parquet(candidate_path)
            candidate_shape = [int(x.shape[0]), int(x.shape[1])]
        rows.append(
            {
                "cohort_key": key,
                "cohort_label": label,
                "plot_include": False,
                "status": "not_rerun",
                "reason": reason,
                "phosphosite_matrix_path": str(phospho_path),
                "phosphosite_n_samples": phospho_shape[0],
                "phosphosite_n_sites": phospho_shape[1],
                "candidate_non_rna_matrix_path": str(candidate_path),
                "candidate_non_rna_n_samples": candidate_shape[0],
                "candidate_non_rna_n_features": candidate_shape[1],
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-site-n", type=int, default=8)
    ap.add_argument("--min-sample-n", type=int, default=50)
    args = ap.parse_args()
    for sub in ["inputs", "predictions", "tables", "logs"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    (OUT / "run_status.txt").write_text("running\n", encoding="utf-8")

    apollo = import_module(APOLLO_SCRIPT, "apollo_reader")
    luad = import_module(LUAD_SCRIPT, "luad_reader")

    apollo_rna = apollo.read_apollo_expression()
    apollo_manifest = make_manifest(apollo_rna.index, cancer="LUAD", study="PDC000149", source="apollo_luad_2022")
    apollo_pred = run_predict("apollo_luad", apollo_rna, apollo_manifest, args.force)
    apollo_summary = evaluate(
        "apollo_luad",
        apollo_pred,
        OLD_APOLLO / "predictions/apollo_luad_true_phospho_all_exact.parquet",
        args.min_site_n,
        args.min_sample_n,
    )

    luad_rna = luad.read_luad_expression()
    luad_manifest = make_manifest(luad_rna.index, cancer="LUAD", study="PDC000149", source="luad_cas_2020")
    luad_pred = run_predict("luad_cas_phoscancer", luad_rna, luad_manifest, args.force)
    luad_summary = evaluate(
        "luad_cas_phoscancer",
        luad_pred,
        OLD_LUAD / "predictions/luad_cas_true_phoscancer_phosphoproteomics_mapped.parquet",
        args.min_site_n,
        args.min_sample_n,
    )

    summary = pd.DataFrame([apollo_summary, luad_summary])
    summary.to_csv(OUT / "tables/additional_external_scp682_main_summary.tsv", sep="\t", index=False)

    plot_rows = collect_original_plot_rows()
    extra_labels = {
        "apollo_luad": ("APOLLO LUAD", "TPM", "APOLLO LUAD"),
        "luad_cas_phoscancer": ("LUAD-CAS / PhosCancer", "FPKM to TPM", "LUAD-CAS / PhosCancer"),
    }
    for s in [apollo_summary, luad_summary]:
        label, input_setting, independent = extra_labels[str(s["dataset"])]
        plot_rows.append(
            {
                "cohort_key": s["dataset"],
                "cohort_label": label,
                "independent_cohort": independent,
                "input_setting": input_setting,
                "n_samples": int(s["n_matched_samples"]),
                "n_sites": int(s["n_matched_targets"]),
                "targets_tested": int(s["targets_tested"]),
                "median_spearman": float(s["median_spearman"]),
                "mean_spearman": float(s["mean_spearman"]),
                "ci95_low": float(s["ci95_low"]),
                "ci95_high": float(s["ci95_high"]),
                "model": "SCP682",
                "plot_include": True,
                "status": "complete",
                "source_file": s["per_target_file"],
                "note": "rerun with SCP682 portable main model",
            }
        )
    fig2c = pd.DataFrame(plot_rows)
    fig2c.to_csv(OUT / "tables/fig2_panel_c_data_expanded.tsv", sep="\t", index=False)
    audit = pd.DataFrame(unavailable_audit_rows())
    audit.to_csv(OUT / "tables/fig2_panel_c_external_rerun_audit.tsv", sep="\t", index=False)

    manifest = {
        "output_dir": str(OUT),
        "completed_plot_rows": int(fig2c.shape[0]),
        "additional_summary": str(OUT / "tables/additional_external_scp682_main_summary.tsv"),
        "fig2c_table": str(OUT / "tables/fig2_panel_c_data_expanded.tsv"),
        "audit_table": str(OUT / "tables/fig2_panel_c_external_rerun_audit.tsv"),
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "done.txt").write_text("done\n", encoding="utf-8")
    (OUT / "run_status.txt").write_text("done\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "fatal.log").write_text(repr(exc) + "\n", encoding="utf-8")
        raise
