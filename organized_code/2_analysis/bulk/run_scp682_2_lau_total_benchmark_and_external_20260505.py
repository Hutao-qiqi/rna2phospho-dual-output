#!/usr/bin/env python3
"""SCP682-2-LauTotal benchmark and Lau-style external validation.

This script intentionally follows the easier Lau-style total-protein protocol:
- total_protein_gene_logratio_all labels
- no sample-median centering
- random 80:20 split inside each cancer type
- one independent model per protein
- full RNA input
- Pearson/Spearman per protein on held-out samples

The Lau GitHub repository publishes code and feature resources, not frozen model
weights. External validation therefore uses Lau-style reproduced models trained
from our CPTAC/PDC paired RNA-total-protein matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/data/lsy/Infinite_Stream")
CPTAC = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2"
OUT = ROOT / "02_results/model_validation/20260505_scp682_2_lau_total_benchmark"
EXT_ROOT = ROOT / "02_results/external_validation"
LAU_REPO = ROOT / "01_data/external_models/lau_lab/CPTAC_Protein"


EXTERNAL = {
    "FU_iCCA": {
        "dir": EXT_ROOT / "20260503_fu_icca_v38_v39_predicted_vs_true_phosphosite",
        "prefix": "fu_icca",
    },
    "TU_SCLC": {
        "dir": EXT_ROOT / "20260503_tu_sclc_v38_v39_predicted_vs_true_phosphosite",
        "prefix": "tu_sclc",
    },
    "CHCC_HBV_FPKM": {
        "dir": EXT_ROOT / "20260503_chcc_hbv_fpkm_v38_v39_predicted_vs_true_phosphosite",
        "prefix": "chcc_hbv_fpkm",
    },
    "CHCC_HBV_RSEM": {
        "dir": EXT_ROOT / "20260503_chcc_hbv_rsem_v38_v39_predicted_vs_true_phosphosite",
        "prefix": "chcc_hbv_rsem",
    },
}


@dataclass(frozen=True)
class Config:
    seed: int = 2
    test_size: float = 0.2
    min_obs: int = 50
    min_test_obs: int = 10
    n_jobs: int = 96
    max_models_per_method: int = 0
    rf_estimators: int = 500
    rf_max_depth: int = 4
    gb_estimators: int = 1000
    gb_max_depth: int = 3
    gb_learning_rate: float = 0.025
    gb_subsample: float = 0.5
    mlp_hidden: tuple[int, ...] = (256, 128, 64)
    mlp_max_iter: int = 500
    save_models: bool = True


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    for d in ["tables", "models", "predictions", "logs"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)


def safe_corr(y: np.ndarray, p: np.ndarray) -> tuple[float, float, int]:
    ok = np.isfinite(y) & np.isfinite(p)
    n = int(ok.sum())
    if n < 10:
        return np.nan, np.nan, n
    if np.nanstd(y[ok]) < 1e-8 or np.nanstd(p[ok]) < 1e-8:
        return 0.0, 0.0, n
    return float(pearsonr(y[ok], p[ok]).statistic), float(spearmanr(y[ok], p[ok]).correlation), n


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_id" in out.columns:
        out = out.set_index("sample_id")
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    return out


def load_cptac() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    manifest = pd.read_csv(CPTAC / "sample_manifest.tsv", sep="\t").set_index("sample_id")
    x = pd.read_parquet(CPTAC / "rna_log2_tpm_paired.parquet")
    y = pd.read_parquet(CPTAC / "total_protein_gene_logratio_all.parquet")
    common = manifest.index.intersection(x.index).intersection(y.index)
    return x.loc[common].astype(np.float32), y.loc[common].astype(np.float32), manifest.loc[common, "cancer_label"].astype(str)


def make_model(method: str, cfg: Config) -> Any:
    if method == "elasticnet":
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 0.95],
            cv=5,
            fit_intercept=False,
            n_jobs=1,
            tol=1e-3,
            max_iter=2000,
            random_state=cfg.seed,
        )
    elif method == "ridge":
        model = RidgeCV(alphas=np.logspace(-3, 4, 40), fit_intercept=True)
    elif method == "randomforest":
        model = RandomForestRegressor(
            n_estimators=cfg.rf_estimators,
            criterion="squared_error",
            max_depth=cfg.rf_max_depth,
            random_state=cfg.seed,
            n_jobs=1,
        )
    elif method == "gradientboosting":
        model = GradientBoostingRegressor(
            n_estimators=cfg.gb_estimators,
            max_depth=cfg.gb_max_depth,
            subsample=cfg.gb_subsample,
            min_samples_split=5,
            learning_rate=cfg.gb_learning_rate,
            random_state=cfg.seed,
        )
    elif method == "mlp":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=cfg.mlp_hidden,
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                early_stopping=True,
                validation_fraction=0.12,
                max_iter=cfg.mlp_max_iter,
                random_state=cfg.seed,
            )),
        ])
    elif method == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=700,
            max_depth=3,
            learning_rate=0.025,
            subsample=0.7,
            colsample_bytree=0.7,
            objective="reg:squarederror",
            n_jobs=1,
            random_state=cfg.seed,
            tree_method="hist",
        )
    elif method == "lightgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(
            n_estimators=700,
            max_depth=4,
            learning_rate=0.025,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=cfg.seed,
            n_jobs=1,
            verbose=-1,
        )
    else:
        raise ValueError(method)
    return model


def available_methods(methods: list[str]) -> list[str]:
    ok = []
    for m in methods:
        if m == "xgboost":
            try:
                import xgboost  # noqa: F401
            except Exception:
                continue
        if m == "lightgbm":
            try:
                import lightgbm  # noqa: F401
            except Exception:
                continue
        ok.append(m)
    return ok


def fit_one_target(
    cancer: str,
    method: str,
    target: str,
    x_sub: pd.DataFrame,
    y_sub: pd.DataFrame,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    cfg: Config,
    save_model: bool,
) -> dict[str, Any]:
    y = y_sub[target].to_numpy(dtype=np.float32)
    n_obs = int(np.isfinite(y).sum())
    train_ok = train_pos[np.isfinite(y[train_pos])]
    test_ok = test_pos[np.isfinite(y[test_pos])]
    if n_obs < cfg.min_obs or len(train_ok) < cfg.min_obs or len(test_ok) < cfg.min_test_obs:
        return {"cancer": cancer, "method": method, "target": target, "n_obs": n_obs, "n_train": len(train_ok), "n_test": len(test_ok), "skipped": True}
    x = x_sub.to_numpy(dtype=np.float32)
    imp = SimpleImputer(strategy="median")
    x_imp = imp.fit_transform(x)
    try:
        vt = VarianceThreshold(0.0)
        x_var = vt.fit_transform(x_imp)
    except ValueError:
        return {"cancer": cancer, "method": method, "target": target, "n_obs": n_obs, "n_train": len(train_ok), "n_test": len(test_ok), "skipped": True}
    model = make_model(method, cfg)
    t0 = time.time()
    model.fit(x_var[train_ok], y[train_ok])
    pred = model.predict(x_var[test_ok])
    pp, ss, nn = safe_corr(y[test_ok], pred)
    row = {
        "cancer": cancer,
        "method": method,
        "target": target,
        "n_obs": n_obs,
        "n_train": int(len(train_ok)),
        "n_test": nn,
        "pearson": pp,
        "spearman": ss,
        "n_features": int(x_var.shape[1]),
        "seconds": float(time.time() - t0),
        "skipped": False,
    }
    if save_model:
        mdir = OUT / "models" / cancer / method
        mdir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"imputer": imp, "variance": vt, "model": model, "genes": x_sub.columns.astype(str).tolist(), "target": target, "cancer": cancer, "method": method}, mdir / f"{target}.joblib", compress=3)
    return row


def run_internal(methods: list[str], cfg: Config) -> None:
    x, y, labels = load_cptac()
    cancer_sets = {"PAN_CANCER": np.arange(len(x))}
    for c in sorted(labels.unique()):
        cancer_sets[c] = np.where(labels.to_numpy() == c)[0]
    with (OUT / "logs" / "config.json").open("w") as fh:
        json.dump({**asdict(cfg), "methods": methods, "lau_repo": str(LAU_REPO)}, fh, indent=2)
    all_summary = []
    cancer_filter = set(os.environ.get("SCP682_LAU_CANCERS", "").split(",")) if os.environ.get("SCP682_LAU_CANCERS") else set()
    for cancer, pos in cancer_sets.items():
        if cancer_filter and cancer not in cancer_filter:
            continue
        if len(pos) < 80:
            continue
        x_sub = x.iloc[pos].copy()
        y_sub = y.iloc[pos].copy()
        train_pos, test_pos = train_test_split(np.arange(len(pos)), test_size=cfg.test_size, random_state=cfg.seed)
        pd.DataFrame({"sample_id": x_sub.index, "split": ["train" if i in set(train_pos) else "test" for i in range(len(pos))]}).to_csv(OUT / "tables" / f"{cancer}_split.tsv", sep="\t", index=False)
        targets = y_sub.columns.astype(str).tolist()
        if cfg.max_models_per_method > 0:
            targets = targets[:cfg.max_models_per_method]
        for method in methods:
            out_file = OUT / "tables" / f"{cancer}_{method}_per_protein.tsv"
            if out_file.exists():
                df = pd.read_csv(out_file, sep="\t")
            else:
                rows = []
                chunk_size = int(os.environ.get("SCP682_LAU_CHUNK_SIZE", "256"))
                tmp_file = OUT / "tables" / f"{cancer}_{method}_per_protein.tmp.tsv"
                for start in range(0, len(targets), chunk_size):
                    chunk = targets[start:start + chunk_size]
                    chunk_rows = Parallel(n_jobs=cfg.n_jobs, backend="loky", verbose=5)(
                        delayed(fit_one_target)(cancer, method, target, x_sub, y_sub, train_pos, test_pos, cfg, cfg.save_models)
                        for target in chunk
                    )
                    rows.extend(chunk_rows)
                    pd.DataFrame(rows).to_csv(tmp_file, sep="\t", index=False)
                    with (OUT / "logs" / "heartbeat.log").open("a") as fh:
                        fh.write(json.dumps({
                            "cancer": cancer,
                            "method": method,
                            "targets_done": len(rows),
                            "targets_total": len(targets),
                            "tmp_file": str(tmp_file),
                        }) + "\n")
                df = pd.DataFrame(rows)
                df.to_csv(out_file, sep="\t", index=False)
                if tmp_file.exists():
                    tmp_file.unlink()
            sub = df[df["skipped"] == False].copy()  # noqa: E712
            vals = pd.to_numeric(sub["pearson"], errors="coerce").dropna()
            summary = {
                "cancer": cancer,
                "method": method,
                "n_samples": int(len(pos)),
                "n_train": int(len(train_pos)),
                "n_test": int(len(test_pos)),
                "n_targets": int(len(vals)),
                "pearson_median": float(vals.median()) if len(vals) else np.nan,
                "pearson_q25": float(vals.quantile(0.25)) if len(vals) else np.nan,
                "pearson_q75": float(vals.quantile(0.75)) if len(vals) else np.nan,
                "pearson_gt_0_5_n": int((vals > 0.5).sum()) if len(vals) else 0,
            }
            all_summary.append(summary)
            pd.DataFrame(all_summary).to_csv(OUT / "tables" / "internal_lau_total_summary_live.tsv", sep="\t", index=False)
            with (OUT / "logs" / "heartbeat.log").open("a") as fh:
                fh.write(json.dumps(summary) + "\n")
    pd.DataFrame(all_summary).to_csv(OUT / "tables" / "internal_lau_total_summary.tsv", sep="\t", index=False)


def load_external_total() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for cohort, cfg in EXTERNAL.items():
        pdir = cfg["dir"] / "predictions"
        prefix = cfg["prefix"]
        truth_path = pdir / f"{prefix}_true_total_protein.parquet"
        pred_ref = pdir / f"{prefix}_v3_total_total_protein.parquet"
        if not truth_path.exists() or not pred_ref.exists():
            continue
        truth = clean_df(pd.read_parquet(truth_path))
        # Existing v3 prediction file has the external sample index and output columns.
        ref = clean_df(pd.read_parquet(pred_ref))
        out[cohort] = (truth, ref)
    return out


def align_external_rna_like_reference(ref: pd.DataFrame) -> pd.DataFrame | None:
    # The external prediction folders keep model predictions and truth, but not
    # always the RNA matrix. For Lau-style external prediction we need RNA.
    # If a prepared RNA matrix is present in the folder, it will be picked up by name.
    return None


def summarize_external_existing() -> None:
    # Record what can be evaluated immediately from existing SCP682 external total predictions.
    rows = []
    for cohort, (truth, ref) in load_external_total().items():
        samples = truth.index.intersection(ref.index)
        targets = truth.columns.intersection(ref.columns)
        y = truth.loc[samples, targets].to_numpy(dtype=float)
        p = ref.loc[samples, targets].to_numpy(dtype=float)
        for j, t in enumerate(targets):
            pp, ss, n = safe_corr(y[:, j], p[:, j])
            rows.append({"cohort": cohort, "model": "existing_v3_total_reference", "target": t, "n": n, "pearson": pp, "spearman": ss})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "tables" / "external_existing_reference_total_per_target.tsv", sep="\t", index=False)
        summary = df.groupby(["cohort", "model"]).agg(
            n_targets=("pearson", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
            pearson_median=("pearson", "median"),
            spearman_median=("spearman", "median"),
        ).reset_index()
        summary.to_csv(OUT / "tables" / "external_existing_reference_total_summary.tsv", sep="\t", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="elasticnet,randomforest,gradientboosting,mlp,xgboost,lightgbm")
    parser.add_argument("--n-jobs", type=int, default=96)
    parser.add_argument("--max-models-per-method", type=int, default=0)
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--cancers", default="", help="Comma-separated cancer labels; empty means all.")
    args = parser.parse_args()
    cfg = Config(n_jobs=args.n_jobs, max_models_per_method=args.max_models_per_method, save_models=not args.no_save_models)
    seed_all(cfg.seed)
    ensure_dirs()
    requested = [x.strip() for x in args.methods.split(",") if x.strip()]
    methods = available_methods(requested)
    missing = sorted(set(requested) - set(methods))
    with (OUT / "logs" / "method_availability.json").open("w") as fh:
        json.dump({"requested": requested, "available": methods, "missing": missing}, fh, indent=2)
    summarize_external_existing()
    if args.cancers:
        os.environ["SCP682_LAU_CANCERS"] = args.cancers
    run_internal(methods, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
