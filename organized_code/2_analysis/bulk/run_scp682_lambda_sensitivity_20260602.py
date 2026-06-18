#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/data/lsy/Infinite_Stream")
OUT = ROOT / "SCP682-main/results/20260602_lambda_sensitivity_single_parameter"
PORTABLE = ROOT / "SCP682_PORTABLE"
FIXED_EXTERNAL = ROOT / "SCP682-main/results/20260523_general_graph_external_fixed_anchor"
ADDITIONAL = ROOT / "SCP682-main/results/20260530_fig2c_additional_external_scp682_main"

LAMBDA_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
REFERENCE_LAMBDA = 0.3
MIN_SITE_N = 8
MIN_SAMPLE_N = 50


def sample_key(x: object) -> str:
    return str(x).split("::", 1)[-1]


def sample_center(df: pd.DataFrame) -> pd.DataFrame:
    med = df.median(axis=1, skipna=True)
    return df.sub(med, axis=0)


def align_matrices(pred: pd.DataFrame, obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pred.copy()
    y = obs.copy()
    p.index = [sample_key(x) for x in p.index]
    y.index = [sample_key(x) for x in y.index]
    samples = [s for s in y.index if s in set(p.index)]
    targets = [t for t in p.columns if t in set(y.columns)]
    return p.loc[samples, targets], y.loc[samples, targets]


def per_site_spearman(pred: pd.DataFrame, obs: pd.DataFrame, min_n: int = MIN_SITE_N) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in obs.columns:
        yy = obs[target].to_numpy(dtype=float)
        pp = pred[target].to_numpy(dtype=float)
        ok = np.isfinite(yy) & np.isfinite(pp)
        n = int(ok.sum())
        rho = np.nan
        pval = np.nan
        if n >= min_n:
            res = spearmanr(yy[ok], pp[ok])
            rho = float(res.correlation) if np.isfinite(res.correlation) else np.nan
            pval = float(res.pvalue) if np.isfinite(res.pvalue) else np.nan
        rows.append({"target": target, "n_samples_used": n, "spearman": rho, "rho_p_value": pval})
    return pd.DataFrame(rows)


def per_sample_spearman(pred: pd.DataFrame, obs: pd.DataFrame, min_n: int = MIN_SAMPLE_N) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample in obs.index:
        yy = obs.loc[sample].to_numpy(dtype=float)
        pp = pred.loc[sample].to_numpy(dtype=float)
        ok = np.isfinite(yy) & np.isfinite(pp)
        n = int(ok.sum())
        rho = np.nan
        if n >= min_n:
            res = spearmanr(yy[ok], pp[ok])
            rho = float(res.correlation) if np.isfinite(res.correlation) else np.nan
        rows.append({"sample_id": sample, "n_targets_used": n, "spearman": rho})
    return pd.DataFrame(rows)


def bootstrap_median(values: pd.Series, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    x = values.dropna().to_numpy(dtype=float)
    if x.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = np.median(rng.choice(x, size=x.size, replace=True))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def evaluate_dataset(
    dataset: str,
    group: str,
    baseline: pd.DataFrame,
    delta: pd.DataFrame,
    observed: pd.DataFrame,
    sample_center_prediction: bool,
    per_site_rows: list[pd.DataFrame],
    summary_rows: list[dict[str, object]],
) -> None:
    targets = [t for t in baseline.columns if t in delta.columns]
    baseline = baseline[targets]
    delta = delta[targets]
    for lam in LAMBDA_GRID:
        pred = baseline + float(lam) * delta
        if sample_center_prediction:
            pred = sample_center(pred)
        pred_a, obs_a = align_matrices(pred, observed)
        site = per_site_spearman(pred_a, obs_a)
        sample = per_sample_spearman(pred_a, obs_a)
        site.insert(0, "lambda", float(lam))
        site.insert(0, "evaluation_group", group)
        site.insert(0, "dataset", dataset)
        per_site_rows.append(site)
        lo, hi = bootstrap_median(site["spearman"])
        summary_rows.append(
            {
                "dataset": dataset,
                "evaluation_group": group,
                "lambda": float(lam),
                "n_samples": int(pred_a.shape[0]),
                "n_targets": int(pred_a.shape[1]),
                "targets_tested": int(site["spearman"].notna().sum()),
                "median_spearman": float(site["spearman"].median(skipna=True)),
                "mean_spearman": float(site["spearman"].mean(skipna=True)),
                "ci95_low": lo,
                "ci95_high": hi,
                "ge_0_3": int((site["spearman"] >= 0.3).sum()),
                "ge_0_5": int((site["spearman"] >= 0.5).sum()),
                "sample_median_spearman": float(sample["spearman"].median(skipna=True)),
                "is_released_lambda": bool(abs(float(lam) - REFERENCE_LAMBDA) < 1e-9),
            }
        )
        print(f"{dataset} lambda={lam} median={summary_rows[-1]['median_spearman']:.4f}", flush=True)


def load_internal() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_parquet(PORTABLE / "training_set/v4_phosphosite_baseline.parquet")
    main = pd.read_parquet(PORTABLE / "predictions/scp682_main_oof_phosphosite.parquet")
    observed = pd.read_parquet(PORTABLE / "training_set/observed_phosphosite.parquet")
    delta = (main.reindex_like(baseline) - baseline) / REFERENCE_LAMBDA
    return baseline, delta, observed


def load_original_external(key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_name = {
        "fu_icca": "SCP682_v4_0_fu_icca_phosphosite.parquet",
        "tu_sclc": "SCP682_v4_0_tu_sclc_phosphosite.parquet",
        "chcc_hbv_fpkm": "SCP682_v4_0_chcc_hbv_fpkm_phosphosite.parquet",
        "chcc_hbv_rsem": "SCP682_v4_0_chcc_hbv_rsem_phosphosite.parquet",
    }[key]
    true_dir = {
        "fu_icca": "20260503_fu_icca_v38_v39_predicted_vs_true_phosphosite",
        "tu_sclc": "20260503_tu_sclc_v38_v39_predicted_vs_true_phosphosite",
        "chcc_hbv_fpkm": "20260503_chcc_hbv_fpkm_v38_v39_predicted_vs_true_phosphosite",
        "chcc_hbv_rsem": "20260503_chcc_hbv_rsem_v38_v39_predicted_vs_true_phosphosite",
    }[key]
    baseline = pd.read_parquet(PORTABLE / "v4_baseline_release/predictions" / baseline_name)
    full = pd.read_parquet(FIXED_EXTERNAL / "predictions" / f"{key}_scp682_general_graph_residual_sample_centered.parquet")
    observed = pd.read_parquet(ROOT / "02_results/external_validation" / true_dir / "predictions" / f"{key}_true_phosphosite.parquet")
    delta = (full.reindex_like(baseline) - baseline) / REFERENCE_LAMBDA
    return baseline, delta, observed


def load_additional_external(key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = ADDITIONAL / "predictions" / key / "predictions"
    baseline = pd.read_parquet(base / "scp682_v4_baseline.parquet")
    delta = pd.read_parquet(base / "scp682_exact_scnet_gnn_delta.parquet")
    true_path = {
        "apollo_luad": ROOT / "02_results/external_validation/20260503_v38_apollo_luad_predicted_vs_true_phosphosite/predictions/apollo_luad_true_phospho_all_exact.parquet",
        "luad_cas_phoscancer": ROOT / "02_results/external_validation/20260502_v38_luad_cas_fpkm_to_tpm_predicted_vs_true_phosphosite/predictions/luad_cas_true_phoscancer_phosphoproteomics_mapped.parquet",
    }[key]
    observed = pd.read_parquet(true_path)
    return baseline, delta, observed


def main() -> int:
    for sub in ["tables", "reports"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    (OUT / "run_status.txt").write_text("running\n", encoding="utf-8")
    per_site_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    baseline, delta, observed = load_internal()
    evaluate_dataset("CPTAC_internal_OOF", "internal", baseline, delta, observed, False, per_site_rows, summary_rows)

    for key in ["fu_icca", "tu_sclc", "chcc_hbv_fpkm", "chcc_hbv_rsem"]:
        baseline, delta, observed = load_original_external(key)
        evaluate_dataset(key, "external", baseline, delta, observed, False, per_site_rows, summary_rows)

    for key in ["apollo_luad", "luad_cas_phoscancer"]:
        baseline, delta, observed = load_additional_external(key)
        evaluate_dataset(key, "external_additional", baseline, delta, observed, False, per_site_rows, summary_rows)

    summary = pd.DataFrame(summary_rows)
    per_site = pd.concat(per_site_rows, ignore_index=True)
    summary.to_csv(OUT / "tables/lambda_sensitivity_summary.tsv", sep="\t", index=False)
    per_site.to_csv(OUT / "tables/lambda_sensitivity_per_site.tsv", sep="\t", index=False)

    released = summary.loc[summary["is_released_lambda"]].copy()
    pivot = summary.pivot_table(index="dataset", columns="lambda", values="median_spearman", aggfunc="first")
    pivot.to_csv(OUT / "tables/lambda_sensitivity_median_matrix.tsv", sep="\t")
    released.to_csv(OUT / "tables/lambda_sensitivity_released_lambda_rows.tsv", sep="\t", index=False)

    report = {
        "formula": "phosphosite_hat(lambda) = S_phi + lambda * Delta_theta",
        "released_lambda": REFERENCE_LAMBDA,
        "lambda_grid": LAMBDA_GRID,
        "summary": str(OUT / "tables/lambda_sensitivity_summary.tsv"),
        "per_site": str(OUT / "tables/lambda_sensitivity_per_site.tsv"),
    }
    (OUT / "reports/run_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "done.txt").write_text("done\n", encoding="utf-8")
    (OUT / "run_status.txt").write_text("done\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "fatal.log").write_text(repr(exc) + "\n", encoding="utf-8")
        raise
