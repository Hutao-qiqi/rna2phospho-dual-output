#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/data/lsy/Infinite_Stream")
DEFAULT_PACKAGE = ROOT / "SCP682-22/frozen_release/SCP682_22_paper_package_20260520"
DEFAULT_BASELINE = ROOT / "SCP682-main/inputs/general_baseline_predictions/general_baseline_internal_cptac_pdc_phosphosite.parquet"
DEFAULT_MAIN = ROOT / "SCP682-main/results/20260523_general_graph_residual_e160"
DEFAULT_OUT = ROOT / "SCP682-main/results/20260523_shrinkage_sensitivity_grid"


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def spearman_cols(y: np.ndarray, pred: np.ndarray, mask: np.ndarray, targets: list[str]) -> pd.DataFrame:
    rows = []
    for j, t in enumerate(targets):
        ok = mask[:, j] & np.isfinite(y[:, j]) & np.isfinite(pred[:, j])
        if int(ok.sum()) < 3:
            sp = np.nan
        else:
            sp = pd.Series(y[ok, j]).corr(pd.Series(pred[ok, j]), method="spearman")
        rows.append({"target": t, "n": int(ok.sum()), "spearman": sp})
    return pd.DataFrame(rows)


def summarize(per: pd.DataFrame, model: str, shrinkage: float, prediction_source: str) -> dict:
    v = pd.to_numeric(per["spearman"], errors="coerce").dropna()
    return {
        "prediction_source": prediction_source,
        "model": model,
        "shrinkage": shrinkage,
        "n_targets": int(v.shape[0]),
        "median_spearman": float(v.median()),
        "mean_spearman": float(v.mean()),
        "ge_0_3": int((v >= 0.3).sum()),
        "ge_0_5": int((v >= 0.5).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-dir", default=str(DEFAULT_PACKAGE))
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--main-result", default=str(DEFAULT_MAIN))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--coefficients", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    args = ap.parse_args()

    out = Path(args.output_dir)
    for sub in ["tables", "reports"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    train_dir = Path(args.package_dir) / "training_set"
    y_df = clean_numeric(pd.read_parquet(train_dir / "observed_phosphosite.parquet"))
    base_df = clean_numeric(pd.read_parquet(args.baseline))
    sources = {
        "trainmode": Path(args.main_result) / "predictions/scp682_general_graph_residual_trainmode_phosphosite_best.parquet",
        "pseudo_external": Path(args.main_result) / "predictions/scp682_general_graph_residual_pseudo_external_phosphosite_best.parquet",
    }

    summaries = []
    per_site_frames = []
    for source_name, pred_path in sources.items():
        pred_df = clean_numeric(pd.read_parquet(pred_path))
        samples = y_df.index.intersection(base_df.index).intersection(pred_df.index)
        targets = [c for c in y_df.columns if c in base_df.columns and c in pred_df.columns]
        y = y_df.loc[samples, targets].to_numpy(np.float32)
        base = base_df.loc[samples, targets].to_numpy(np.float32)
        full = pred_df.loc[samples, targets].to_numpy(np.float32)
        mask = np.isfinite(y) & np.isfinite(base) & np.isfinite(full)
        base = np.nan_to_num(base, nan=0.0)
        full = np.nan_to_num(full, nan=0.0)
        delta = full - base
        for c in args.coefficients:
            pred = base + float(c) * delta
            per = spearman_cols(y, pred, mask, targets)
            per["prediction_source"] = source_name
            per["shrinkage"] = float(c)
            per_site_frames.append(per)
            summaries.append(summarize(per, "general_baseline_plus_scaled_graph_delta", float(c), source_name))

    pd.DataFrame(summaries).to_csv(out / "tables/shrinkage_sensitivity_grid.tsv", sep="\t", index=False)
    pd.concat(per_site_frames, axis=0, ignore_index=True).to_csv(out / "tables/shrinkage_sensitivity_per_site.tsv", sep="\t", index=False)
    (out / "reports/run_summary.json").write_text(
        json.dumps(
            {
                "coefficients": args.coefficients,
                "baseline": str(args.baseline),
                "main_result": str(args.main_result),
                "formula": "prediction = general_baseline + shrinkage * (graph_prediction - general_baseline)",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out / "done.txt").write_text("done\n", encoding="utf-8")


if __name__ == "__main__":
    main()
