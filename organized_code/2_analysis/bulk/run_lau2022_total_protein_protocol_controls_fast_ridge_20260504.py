#!/usr/bin/env python3
"""Fast Lau-style controls using matrix ridge models.

This is a quick diagnostic for label-normalization effects. It uses the same
random 80:20 split for logratio and study-zscore total-protein labels, then fits
a multi-output ridge model and evaluates each protein on the held-out test set.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path("/data/lsy/Infinite_Stream")
DATA = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2"
OUT = ROOT / "02_results/model_validation/20260504_lau2022_total_protein_protocol_controls_fast_ridge"


@dataclass(frozen=True)
class Config:
    seed: int = 682104
    test_size: float = 0.2
    min_obs: int = 50
    alpha: float = 100.0


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, int]:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(ok.sum()) < 10:
        return np.nan, np.nan, int(ok.sum())
    if np.nanstd(y_true[ok]) < 1e-8 or np.nanstd(y_pred[ok]) < 1e-8:
        return 0.0, 0.0, int(ok.sum())
    return float(pearsonr(y_true[ok], y_pred[ok]).statistic), float(spearmanr(y_true[ok], y_pred[ok]).correlation), int(ok.sum())


def fit_eval(label_name: str, x_df: pd.DataFrame, y_df: pd.DataFrame, train_pos: np.ndarray, test_pos: np.ndarray, cfg: Config) -> pd.DataFrame:
    x = x_df.to_numpy(dtype=np.float32)
    y = y_df.to_numpy(dtype=np.float32)
    y_train = y[train_pos].copy()
    y_test = y[test_pos].copy()
    n_obs = np.isfinite(y).sum(axis=0)
    train_obs = np.isfinite(y_train).sum(axis=0)
    keep = (n_obs >= cfg.min_obs) & (train_obs >= cfg.min_obs)
    kept_cols = np.where(keep)[0]
    y_train_keep = y_train[:, kept_cols]
    y_mean = np.nanmean(y_train_keep, axis=0)
    y_mean = np.where(np.isfinite(y_mean), y_mean, 0.0).astype(np.float32)
    y_train_filled = np.where(np.isfinite(y_train_keep), y_train_keep, y_mean[None, :])
    x_imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_imp.fit_transform(x[train_pos]))
    x_test = scaler.transform(x_imp.transform(x[test_pos]))
    model = Ridge(alpha=cfg.alpha)
    model.fit(x_train, y_train_filled)
    pred = model.predict(x_test)
    rows = []
    for out_j, col_j in enumerate(kept_cols):
        p, s, n_test = safe_corr(y_test[:, col_j], pred[:, out_j])
        rows.append({
            "label": label_name,
            "target": str(y_df.columns[col_j]),
            "n_obs": int(n_obs[col_j]),
            "n_train": int(train_obs[col_j]),
            "n_test": n_test,
            "ridge_pearson": p,
            "ridge_spearman": s,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, sub in df.groupby("label"):
        row = {"label": label, "n_targets": int(len(sub))}
        for metric in ["ridge_pearson", "ridge_spearman"]:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_median"] = float(vals.median())
            row[f"{metric}_q25"] = float(vals.quantile(0.25))
            row[f"{metric}_q75"] = float(vals.quantile(0.75))
            row[f"{metric}_gt_0_5_n"] = int((vals > 0.5).sum())
            row[f"{metric}_gt_0_3_n"] = int((vals > 0.3).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = Config()
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(DATA / "sample_manifest.tsv", sep="\t").set_index("sample_id")
    x_df = pd.read_parquet(DATA / "rna_log2_tpm_paired.parquet")
    y_log = pd.read_parquet(DATA / "total_protein_gene_logratio_all.parquet")
    y_z = pd.read_parquet(DATA / "total_protein_gene_study_zscore_min20pct.parquet")
    common = manifest.index.intersection(x_df.index).intersection(y_log.index).intersection(y_z.index)
    manifest = manifest.loc[common]
    x_df = x_df.loc[common]
    y_log = y_log.loc[common]
    y_z = y_z.loc[common]
    split_label = manifest["cancer_label"].astype(str) + "__" + manifest["pdc_study_id"].astype(str)
    vc = split_label.value_counts()
    strat = split_label.where(split_label.map(vc) >= 2, "RARE")
    train_pos, test_pos = train_test_split(np.arange(len(common)), test_size=cfg.test_size, random_state=cfg.seed, stratify=strat)
    with (OUT / "logs" / "config.json").open("w") as fh:
        json.dump({**asdict(cfg), "n_samples": len(common), "n_train": len(train_pos), "n_test": len(test_pos)}, fh, indent=2)
    out_log = fit_eval("logratio_all_random80_20_fast_ridge", x_df, y_log, train_pos, test_pos, cfg)
    out_z = fit_eval("study_zscore_random80_20_fast_ridge", x_df, y_z, train_pos, test_pos, cfg)
    all_df = pd.concat([out_log, out_z], ignore_index=True)
    all_df.to_csv(OUT / "tables" / "fast_ridge_per_protein.tsv", sep="\t", index=False)
    summarize(all_df).to_csv(OUT / "tables" / "fast_ridge_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
