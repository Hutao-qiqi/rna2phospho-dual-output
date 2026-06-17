#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep="\t")
        if "sample_id" in df.columns:
            df = df.set_index("sample_id")
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def per_site_spearman(pred: pd.DataFrame, obs: pd.DataFrame, min_n: int) -> pd.DataFrame:
    samples = pred.index.intersection(obs.index)
    targets = [c for c in pred.columns if c in obs.columns]
    pred = pred.loc[samples, targets]
    obs = obs.loc[samples, targets]
    rows: list[dict[str, object]] = []
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--main-prediction", default=None)
    ap.add_argument("--observed", default=None)
    ap.add_argument("--dataset", default="CPTAC_internal_OOF")
    ap.add_argument("--min-n", type=int, default=10)
    args = ap.parse_args()

    package_dir = Path(args.package_dir)
    out = Path(args.output_dir)
    for sub in ["tables", "reports"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "run_status.txt").write_text("running\n", encoding="utf-8")

    baseline_path = Path(args.baseline) if args.baseline else package_dir / "training_set" / "v4_phosphosite_baseline.parquet"
    main_path = Path(args.main_prediction) if args.main_prediction else package_dir / "predictions" / "scp682_main_oof_phosphosite.parquet"
    observed_path = Path(args.observed) if args.observed else package_dir / "training_set" / "observed_phosphosite.parquet"

    baseline = read_table(baseline_path)
    main_pred = read_table(main_path)
    observed = read_table(observed_path)
    samples = baseline.index.intersection(main_pred.index).intersection(observed.index)
    targets = [c for c in baseline.columns if c in main_pred.columns and c in observed.columns]
    baseline = baseline.loc[samples, targets]
    main_pred = main_pred.loc[samples, targets]
    observed = observed.loc[samples, targets]

    release_lambda = 0.3
    proposal = (main_pred - baseline) / release_lambda
    gamma_grid = [0.0, 0.05, 0.1, 0.15, 0.3, 0.45, 0.5, 0.75, 1.0, 1.5]
    lambda_grid = [0.1, 0.3, 0.5, 1.0]
    alpha_grid = [0.5, 1.0, 1.5]

    per_site_frames = []
    summary_rows = []
    for gamma in gamma_grid:
        pred = baseline + float(gamma) * proposal
        site = per_site_spearman(pred, observed, min_n=args.min_n)
        site.insert(0, "gamma", float(gamma))
        site.insert(0, "dataset", args.dataset)
        per_site_frames.append(site)
        lo, hi = bootstrap_median(site["spearman"])
        summary_rows.append(
            {
                "dataset": args.dataset,
                "gamma": float(gamma),
                "effective_formula": "Y_hat(gamma)=B_phi+gamma*U_theta",
                "n_samples": int(len(samples)),
                "n_targets": int(len(targets)),
                "targets_tested": int(site["spearman"].notna().sum()),
                "median_spearman": float(site["spearman"].median(skipna=True)),
                "mean_spearman": float(site["spearman"].mean(skipna=True)),
                "ci95_low": lo,
                "ci95_high": hi,
                "is_release_point": bool(abs(float(gamma) - release_lambda) < 1e-9),
            }
        )
        print(f"{args.dataset} gamma={gamma} median={summary_rows[-1]['median_spearman']:.4f}", flush=True)

    sensitivity_rows = []
    summary_df = pd.DataFrame(summary_rows)
    for lam in lambda_grid:
        for alpha in alpha_grid:
            gamma = float(lam) * float(alpha)
            closest = summary_df.iloc[(summary_df["gamma"] - gamma).abs().argsort()].iloc[0]
            sensitivity_rows.append(
                {
                    "lambda": float(lam),
                    "alpha": float(alpha),
                    "gamma": gamma,
                    "nearest_evaluated_gamma": float(closest["gamma"]),
                    "nearest_median_spearman": float(closest["median_spearman"]),
                    "note": "exact value shown when gamma is in the evaluated grid; otherwise nearest evaluated point",
                }
            )

    per_site = pd.concat(per_site_frames, ignore_index=True)
    summary_df.to_csv(out / "tables" / "gamma_sensitivity_summary.tsv", sep="\t", index=False)
    per_site.to_csv(out / "tables" / "gamma_sensitivity_per_site.tsv", sep="\t", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(out / "tables" / "lambda_alpha_grid_projection.tsv", sep="\t", index=False)
    summary_df.pivot_table(index="dataset", columns="gamma", values="median_spearman", aggfunc="first").to_csv(
        out / "tables" / "gamma_sensitivity_median_matrix.tsv", sep="\t"
    )
    report = {
        "dataset": args.dataset,
        "formula": "Y_hat(gamma)=B_phi+gamma*U_theta",
        "release_gamma": release_lambda,
        "source_baseline": str(baseline_path),
        "source_main_prediction": str(main_path),
        "source_observed": str(observed_path),
    }
    (out / "reports" / "run_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "done.txt").write_text("done\n", encoding="utf-8")
    (out / "run_status.txt").write_text("done\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Keep the failure local to this result folder for monitoring.
        raise
