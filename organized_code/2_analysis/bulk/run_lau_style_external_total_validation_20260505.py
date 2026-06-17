#!/usr/bin/env python3
"""Apply reproduced Lau-style total-protein models to four external cohorts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer


ROOT = Path("/data/lsy/Infinite_Stream")
CPTAC = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2"
OUT = ROOT / "02_results/model_validation/20260505_lau_style_external_total_validation"
MODEL_VALIDATION_DIR = ROOT / "03_code/model_validation"
LAU_BENCH_SCRIPT = MODEL_VALIDATION_DIR / "run_scp682_2_lau_total_benchmark_and_external_20260505.py"
FU_SCRIPT = ROOT / "03_code/external_validation/proteogenomics/deploy_v38_v39_fu_icca_external_validation_20260503.py"
TU_SCRIPT = ROOT / "03_code/external_validation/proteogenomics/deploy_v38_v39_tu_sclc_external_validation_20260503.py"
CHCC_SCRIPT = ROOT / "03_code/external_validation/proteogenomics/deploy_v38_v39_chcc_hbv_external_validation_20260503.py"


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(MODEL_VALIDATION_DIR))
import run_scp682_2_lau_total_benchmark_and_external_20260505 as bench  # noqa: E402


EXTERNAL = {
    "FU_iCCA": {"script": FU_SCRIPT, "expr": "read_fu_expression", "truth": "read_fu_true_total", "sources": ["PAN_CANCER", "STAD"]},
    "TU_SCLC": {"script": TU_SCRIPT, "expr": "read_tu_sclc_expression", "truth": "read_tu_sclc_true_total", "sources": ["PAN_CANCER", "LSCC"]},
    "CHCC_HBV_FPKM": {"script": CHCC_SCRIPT, "expr": "read_chcc_expression", "expr_arg": "fpkm", "truth": "read_chcc_true_total", "sources": ["PAN_CANCER", "STAD"]},
    "CHCC_HBV_RSEM": {"script": CHCC_SCRIPT, "expr": "read_chcc_expression", "expr_arg": "rsem", "truth": "read_chcc_true_total", "sources": ["PAN_CANCER", "STAD"]},
}


def ensure_dirs() -> None:
    for d in ["tables", "predictions", "logs"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)


def load_cptac() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    manifest = pd.read_csv(CPTAC / "sample_manifest.tsv", sep="\t").set_index("sample_id")
    x = pd.read_parquet(CPTAC / "rna_log2_tpm_paired.parquet")
    y = pd.read_parquet(CPTAC / "total_protein_gene_logratio_all.parquet")
    common = manifest.index.intersection(x.index).intersection(y.index)
    return x.loc[common].astype(np.float32), y.loc[common].astype(np.float32), manifest.loc[common, "cancer_label"].astype(str)


def load_external(cohort: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = EXTERNAL[cohort]
    mod = import_file(f"external_reader_{cohort}", cfg["script"])
    if "expr_arg" in cfg:
        x = getattr(mod, cfg["expr"])(cfg["expr_arg"])
    else:
        x = getattr(mod, cfg["expr"])()
    y = getattr(mod, cfg["truth"])()
    x = bench.clean_df(x).astype(np.float32)
    y = bench.clean_df(y).astype(np.float32)
    samples = x.index.intersection(y.index)
    return x.loc[samples], y.loc[samples]


def fit_predict_one(
    cohort: str,
    source: str,
    method: str,
    target: str,
    x_train: pd.DataFrame,
    y_train_df: pd.DataFrame,
    x_ext: pd.DataFrame,
    y_ext: pd.DataFrame,
    cfg: Any,
) -> dict[str, Any]:
    y = y_train_df[target].to_numpy(dtype=np.float32)
    ok = np.isfinite(y)
    n_train = int(ok.sum())
    if n_train < cfg.min_obs or target not in y_ext.columns:
        return {"cohort": cohort, "source": source, "method": method, "target": target, "skipped": True, "n_train": n_train}
    x0 = x_train.to_numpy(dtype=np.float32)
    xe0 = x_ext.reindex(columns=x_train.columns).to_numpy(dtype=np.float32)
    imp = SimpleImputer(strategy="median")
    x_imp = imp.fit_transform(x0)
    xe_imp = imp.transform(xe0)
    try:
        vt = VarianceThreshold(0.0)
        x_var = vt.fit_transform(x_imp)
        xe_var = vt.transform(xe_imp)
    except ValueError:
        return {"cohort": cohort, "source": source, "method": method, "target": target, "skipped": True, "n_train": n_train}
    model = bench.make_model(method, cfg)
    t0 = time.time()
    model.fit(x_var[ok], y[ok])
    pred = model.predict(xe_var)
    truth = y_ext[target].to_numpy(dtype=float)
    pp, ss, nn = bench.safe_corr(truth, pred)
    return {
        "cohort": cohort,
        "source": source,
        "method": method,
        "target": target,
        "n_train": n_train,
        "n_external": nn,
        "pearson": pp,
        "spearman": ss,
        "n_features": int(x_var.shape[1]),
        "seconds": float(time.time() - t0),
        "skipped": False,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub0 in df.groupby(["cohort", "source", "method"]):
        sub = sub0[sub0["skipped"] == False].copy()  # noqa: E712
        vals = pd.to_numeric(sub["pearson"], errors="coerce").dropna()
        sp = pd.to_numeric(sub["spearman"], errors="coerce").dropna()
        rows.append({
            "cohort": keys[0],
            "source": keys[1],
            "method": keys[2],
            "n_targets": int(len(vals)),
            "pearson_median": float(vals.median()) if len(vals) else np.nan,
            "pearson_q25": float(vals.quantile(0.25)) if len(vals) else np.nan,
            "pearson_q75": float(vals.quantile(0.75)) if len(vals) else np.nan,
            "spearman_median": float(sp.median()) if len(sp) else np.nan,
            "pearson_gt_0_5_n": int((vals > 0.5).sum()) if len(vals) else 0,
            "pearson_gt_0_3_n": int((vals > 0.3).sum()) if len(vals) else 0,
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="elasticnet,randomforest,gradientboosting,mlp,xgboost,lightgbm")
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--max-targets", type=int, default=0)
    args = parser.parse_args()
    ensure_dirs()
    cfg = bench.Config(n_jobs=args.n_jobs, save_models=False)
    methods = bench.available_methods([m.strip() for m in args.methods.split(",") if m.strip()])
    x_train_all, y_train_all, labels = load_cptac()
    all_rows = []
    for cohort, ecfg in EXTERNAL.items():
        x_ext, y_ext = load_external(cohort)
        for source in ecfg["sources"]:
            if source == "PAN_CANCER":
                pos = np.arange(len(x_train_all))
            else:
                pos = np.where(labels.to_numpy() == source)[0]
            if len(pos) < 50:
                continue
            x_train = x_train_all.iloc[pos].copy()
            y_train = y_train_all.iloc[pos].copy()
            targets = [t for t in y_ext.columns.astype(str) if t in set(y_train.columns.astype(str))]
            if args.max_targets > 0:
                targets = targets[:args.max_targets]
            for method in methods:
                out_file = OUT / "tables" / f"{cohort}_{source}_{method}_per_target.tsv"
                if out_file.exists():
                    df = pd.read_csv(out_file, sep="\t")
                else:
                    rows = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
                        delayed(fit_predict_one)(cohort, source, method, target, x_train, y_train, x_ext, y_ext, cfg)
                        for target in targets
                    )
                    df = pd.DataFrame(rows)
                    df.to_csv(out_file, sep="\t", index=False)
                all_rows.append(df)
                summary = summarize(pd.concat(all_rows, ignore_index=True))
                summary.to_csv(OUT / "tables" / "lau_style_external_total_summary_live.tsv", sep="\t", index=False)
                with (OUT / "logs" / "heartbeat.log").open("a") as fh:
                    fh.write(json.dumps({"cohort": cohort, "source": source, "method": method, "targets": len(df), "time": time.ctime()}) + "\n")
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(OUT / "tables" / "lau_style_external_total_per_target.tsv", sep="\t", index=False)
    summarize(all_df).to_csv(OUT / "tables" / "lau_style_external_total_summary.tsv", sep="\t", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
