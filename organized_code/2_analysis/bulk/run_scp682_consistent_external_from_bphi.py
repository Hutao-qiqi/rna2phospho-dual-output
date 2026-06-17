#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/data/lsy/Infinite_Stream")
PACKAGE = ROOT / "SCP682_PORTABLE"
OUT = ROOT / "SCP682-main/results/20260603_consistent_external_validation"


sys.path.insert(0, str(PACKAGE))
from scp682_graph_runtime import SCP682GraphRuntime  # noqa: E402


DATASETS = {
    "fu_icca": {
        "label": "FU-iCCA",
        "independent_cohort": "FU-iCCA",
        "baseline": PACKAGE / "v4_baseline_release/predictions/SCP682_v4_0_fu_icca_phosphosite.parquet",
        "observed": ROOT / "02_results/external_validation/20260503_fu_icca_v38_v39_predicted_vs_true_phosphosite/predictions/fu_icca_true_phosphosite.parquet",
        "main_figure": True,
    },
    "tu_sclc": {
        "label": "TU-SCLC",
        "independent_cohort": "TU-SCLC",
        "baseline": PACKAGE / "v4_baseline_release/predictions/SCP682_v4_0_tu_sclc_phosphosite.parquet",
        "observed": ROOT / "02_results/external_validation/20260503_tu_sclc_v38_v39_predicted_vs_true_phosphosite/predictions/tu_sclc_true_phosphosite.parquet",
        "main_figure": True,
    },
    "chcc_hbv_fpkm": {
        "label": "CHCC-HBV FPKM",
        "independent_cohort": "CHCC-HBV",
        "baseline": PACKAGE / "v4_baseline_release/predictions/SCP682_v4_0_chcc_hbv_fpkm_phosphosite.parquet",
        "observed": ROOT / "02_results/external_validation/20260503_chcc_hbv_fpkm_v38_v39_predicted_vs_true_phosphosite/predictions/chcc_hbv_fpkm_true_phosphosite.parquet",
        "main_figure": True,
    },
    "chcc_hbv_rsem": {
        "label": "CHCC-HBV RSEM",
        "independent_cohort": "CHCC-HBV",
        "baseline": PACKAGE / "v4_baseline_release/predictions/SCP682_v4_0_chcc_hbv_rsem_phosphosite.parquet",
        "observed": ROOT / "02_results/external_validation/20260503_chcc_hbv_rsem_v38_v39_predicted_vs_true_phosphosite/predictions/chcc_hbv_rsem_true_phosphosite.parquet",
        "main_figure": False,
    },
    "apollo_luad": {
        "label": "APOLLO LUAD",
        "independent_cohort": "APOLLO LUAD",
        "baseline": ROOT / "SCP682-main/results/20260530_fig2c_additional_external_scp682_main/predictions/apollo_luad/predictions/scp682_v4_baseline.parquet",
        "observed": ROOT / "02_results/external_validation/20260503_v38_apollo_luad_predicted_vs_true_phosphosite/predictions/apollo_luad_true_phospho_all_exact.parquet",
        "main_figure": True,
    },
    "luad_cas_phoscancer": {
        "label": "LUAD-CAS PhosCancer",
        "independent_cohort": "LUAD-CAS",
        "baseline": ROOT / "SCP682-main/results/20260530_fig2c_additional_external_scp682_main/predictions/luad_cas_phoscancer/predictions/scp682_v4_baseline.parquet",
        "observed": ROOT / "02_results/external_validation/20260502_v38_luad_cas_fpkm_to_tpm_predicted_vs_true_phosphosite/predictions/luad_cas_true_phoscancer_phosphoproteomics_mapped.parquet",
        "main_figure": False,
    },
}


def read_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = [str(x).split("::", 1)[-1] for x in df.index]
    df.columns = df.columns.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def per_site_spearman(pred: pd.DataFrame, obs: pd.DataFrame, min_n: int = 8) -> pd.DataFrame:
    samples = obs.index.intersection(pred.index)
    targets = [c for c in pred.columns if c in obs.columns]
    pred = pred.loc[samples, targets]
    obs = obs.loc[samples, targets]
    rows = []
    for target in targets:
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
    return pd.DataFrame(rows), len(samples), len(targets)


def per_sample_spearman(pred: pd.DataFrame, obs: pd.DataFrame, min_n: int = 50) -> pd.DataFrame:
    samples = obs.index.intersection(pred.index)
    targets = [c for c in pred.columns if c in obs.columns]
    pred = pred.loc[samples, targets]
    obs = obs.loc[samples, targets]
    rows = []
    for sample in samples:
        yy = obs.loc[sample].to_numpy(dtype=float)
        pp = pred.loc[sample].to_numpy(dtype=float)
        ok = np.isfinite(yy) & np.isfinite(pp)
        n = int(ok.sum())
        rho = np.nan
        pval = np.nan
        if n >= min_n:
            res = spearmanr(yy[ok], pp[ok])
            rho = float(res.correlation) if np.isfinite(res.correlation) else np.nan
            pval = float(res.pvalue) if np.isfinite(res.pvalue) else np.nan
        rows.append({"sample_id": sample, "n_targets_used": n, "spearman": rho, "rho_p_value": pval})
    return pd.DataFrame(rows)


def bootstrap_median(values: pd.Series, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    arr = values.dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = np.median(rng.choice(arr, size=arr.size, replace=True))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> int:
    for sub in ["predictions", "tables", "logs", "reports"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    (OUT / "run_status.txt").write_text("running\n", encoding="utf-8")
    if (OUT / "fatal.log").exists():
        (OUT / "fatal.log").unlink()

    runtime = SCP682GraphRuntime(package_dir=PACKAGE, device="cuda:0", knn=25, temperature=0.08, batch_size=4)
    summary_rows = []
    all_site_rows = []
    for key, cfg in DATASETS.items():
        print(f"start {key}", flush=True)
        baseline = read_matrix(cfg["baseline"])
        observed = read_matrix(cfg["observed"])
        graph_out = runtime.predict(baseline)
        pred = graph_out["scp682"]
        delta = graph_out["graph_delta"]
        pred.to_parquet(OUT / "predictions" / f"{key}_scp682_consistent_phosphosite.parquet")
        baseline.to_parquet(OUT / "predictions" / f"{key}_B_phi.parquet")
        delta.to_parquet(OUT / "predictions" / f"{key}_U_theta.parquet")

        per_site, n_samples, n_targets = per_site_spearman(pred, observed)
        per_sample = per_sample_spearman(pred, observed)
        per_site.insert(0, "dataset", key)
        per_site.insert(1, "cohort_label", cfg["label"])
        all_site_rows.append(per_site)
        per_site.to_csv(OUT / "tables" / f"{key}_per_site_spearman.tsv", sep="\t", index=False)
        per_sample.to_csv(OUT / "tables" / f"{key}_per_sample_spearman.tsv", sep="\t", index=False)
        lo, hi = bootstrap_median(per_site["spearman"])
        summary_rows.append(
            {
                "dataset": key,
                "cohort_label": cfg["label"],
                "independent_cohort": cfg["independent_cohort"],
                "main_figure": bool(cfg["main_figure"]),
                "n_matched_samples": int(n_samples),
                "n_matched_targets": int(n_targets),
                "targets_tested": int(per_site["spearman"].notna().sum()),
                "median_spearman": float(per_site["spearman"].median(skipna=True)),
                "mean_spearman": float(per_site["spearman"].mean(skipna=True)),
                "ci95_low": lo,
                "ci95_high": hi,
                "ge_0_3": int((per_site["spearman"] >= 0.3).sum()),
                "ge_0_5": int((per_site["spearman"] >= 0.5).sum()),
                "sample_median_spearman": float(per_sample["spearman"].median(skipna=True)),
                "baseline_source": str(cfg["baseline"]),
                "observed_source": str(cfg["observed"]),
            }
        )
        pd.DataFrame(summary_rows).to_csv(OUT / "tables" / "external_summary.tsv", sep="\t", index=False)
        print(f"done {key} median={summary_rows[-1]['median_spearman']:.4f}", flush=True)

    summary = pd.DataFrame(summary_rows)
    all_site = pd.concat(all_site_rows, ignore_index=True)
    summary.to_csv(OUT / "tables" / "external_summary.tsv", sep="\t", index=False)
    all_site.to_csv(OUT / "tables" / "external_per_site_spearman.tsv", sep="\t", index=False)
    main_fig = summary.loc[summary["main_figure"]].copy()
    main_fig.to_csv(OUT / "tables" / "fig2_panel_c_data.tsv", sep="\t", index=False)
    (OUT / "reports" / "run_summary.json").write_text(
        json.dumps(
            {
                "formula": "Y_hat=B_phi+0.3*U_theta",
                "runtime_state": str(PACKAGE / "models/scp682_graph_runtime_state.pt"),
                "summary": str(OUT / "tables/external_summary.tsv"),
                "fig2_panel_c_data": str(OUT / "tables/fig2_panel_c_data.tsv"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "done.txt").write_text("done\n", encoding="utf-8")
    (OUT / "run_status.txt").write_text("done\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "fatal.log").write_text(repr(exc) + "\n", encoding="utf-8")
        raise
