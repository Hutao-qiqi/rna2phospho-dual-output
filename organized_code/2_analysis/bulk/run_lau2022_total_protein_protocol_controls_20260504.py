#!/usr/bin/env python3
"""Run Lau-style total protein protocol controls.

Two paired controls:
1. total_protein_gene_logratio_all with random 80:20 split.
2. total_protein_gene_study_zscore_min20pct with the same random 80:20 split.

The goal is to separate label-normalization effects from model effects.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ROOT = Path("/data/lsy/Infinite_Stream")
DATA = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2"
OUT = ROOT / "02_results/model_validation/20260504_lau2022_total_protein_protocol_controls"


@dataclass(frozen=True)
class Config:
    seed: int = 682104
    test_size: float = 0.2
    min_obs: int = 50
    max_iter: int = 5000
    n_jobs: int = 32
    alphas: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    l1_ratio: tuple[float, ...] = (0.1, 0.5, 0.9, 0.95)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(ok.sum()) < 10:
        return np.nan, np.nan
    if np.nanstd(y_true[ok]) < 1e-8 or np.nanstd(y_pred[ok]) < 1e-8:
        return 0.0, 0.0
    return float(pearsonr(y_true[ok], y_pred[ok]).statistic), float(spearmanr(y_true[ok], y_pred[ok]).correlation)


def run_one_label(label_name: str, y_df: pd.DataFrame, x_df: pd.DataFrame, split_label: pd.Series, cfg: Config) -> pd.DataFrame:
    common = x_df.index.intersection(y_df.index)
    x_df = x_df.loc[common]
    y_df = y_df.loc[common]
    split_label = split_label.loc[common].astype(str)
    vc = split_label.value_counts()
    strat = split_label.where(split_label.map(vc) >= 2, "RARE")
    train_ids, test_ids = train_test_split(
        np.arange(len(common)),
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=strat,
    )
    x = x_df.to_numpy(dtype=np.float32)
    rows = []
    for j, target in enumerate(y_df.columns.astype(str)):
        y = y_df.iloc[:, j].to_numpy(dtype=np.float32)
        n_obs = int(np.isfinite(y).sum())
        if n_obs < cfg.min_obs:
            rows.append({"label": label_name, "target": target, "n_obs": n_obs, "skipped": True})
            continue
        train_ok = train_ids[np.isfinite(y[train_ids])]
        test_ok = test_ids[np.isfinite(y[test_ids])]
        if len(train_ok) < cfg.min_obs or len(test_ok) < 10:
            rows.append({"label": label_name, "target": target, "n_obs": n_obs, "n_train": len(train_ok), "n_test": len(test_ok), "skipped": True})
            continue
        x_train = x[train_ok]
        x_test = x[test_ok]
        y_train = y[train_ok]
        y_test = y[test_ok]
        ridge = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeCV(alphas=np.array(cfg.alphas)),
        )
        ridge.fit(x_train, y_train)
        pred_ridge = ridge.predict(x_test)
        rp, rs = safe_corr(y_test, pred_ridge)
        enet_p = np.nan
        enet_s = np.nan
        try:
            enet = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                ElasticNetCV(
                    l1_ratio=list(cfg.l1_ratio),
                    cv=5,
                    max_iter=cfg.max_iter,
                    n_jobs=1,
                    random_state=cfg.seed,
                    selection="random",
                ),
            )
            enet.fit(x_train, y_train)
            pred_enet = enet.predict(x_test)
            enet_p, enet_s = safe_corr(y_test, pred_enet)
        except Exception as exc:
            rows.append({
                "label": label_name,
                "target": target,
                "n_obs": n_obs,
                "n_train": len(train_ok),
                "n_test": len(test_ok),
                "ridge_pearson": rp,
                "ridge_spearman": rs,
                "elasticnet_error": str(exc)[:300],
                "skipped": False,
            })
            continue
        rows.append({
            "label": label_name,
            "target": target,
            "n_obs": n_obs,
            "n_train": len(train_ok),
            "n_test": len(test_ok),
            "ridge_pearson": rp,
            "ridge_spearman": rs,
            "elasticnet_pearson": enet_p,
            "elasticnet_spearman": enet_s,
            "skipped": False,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, sub0 in df.groupby("label"):
        sub = sub0[sub0["skipped"] == False].copy()  # noqa: E712
        row = {"label": label, "n_targets": int(len(sub))}
        for metric in ["ridge_pearson", "ridge_spearman", "elasticnet_pearson", "elasticnet_spearman"]:
            if metric in sub:
                vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
                row[f"{metric}_median"] = float(vals.median()) if len(vals) else np.nan
                row[f"{metric}_q25"] = float(vals.quantile(0.25)) if len(vals) else np.nan
                row[f"{metric}_q75"] = float(vals.quantile(0.75)) if len(vals) else np.nan
                row[f"{metric}_gt_0_5_n"] = int((vals > 0.5).sum()) if len(vals) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = Config()
    seed_all(cfg.seed)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(DATA / "sample_manifest.tsv", sep="\t").set_index("sample_id")
    x_df = pd.read_parquet(DATA / "rna_log2_tpm_paired.parquet")
    y_logratio = pd.read_parquet(DATA / "total_protein_gene_logratio_all.parquet")
    y_z = pd.read_parquet(DATA / "total_protein_gene_study_zscore_min20pct.parquet")
    split_label = manifest["cancer_label"].astype(str) + "__" + manifest["pdc_study_id"].astype(str)
    with (OUT / "logs" / "config.json").open("w") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    out1 = run_one_label("logratio_all_random80_20", y_logratio, x_df, split_label, cfg)
    out1.to_csv(OUT / "tables" / "logratio_all_random80_20_per_protein.tsv", sep="\t", index=False)
    out2 = run_one_label("study_zscore_random80_20", y_z, x_df, split_label, cfg)
    out2.to_csv(OUT / "tables" / "study_zscore_random80_20_per_protein.tsv", sep="\t", index=False)
    all_df = pd.concat([out1, out2], ignore_index=True)
    all_df.to_csv(OUT / "tables" / "lau2022_protocol_controls_per_protein.tsv", sep="\t", index=False)
    summarize(all_df).to_csv(OUT / "tables" / "lau2022_protocol_controls_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
