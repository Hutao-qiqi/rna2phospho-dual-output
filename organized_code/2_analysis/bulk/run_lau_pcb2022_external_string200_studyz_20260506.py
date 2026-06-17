#!/usr/bin/env python3
"""External validation for Lau PCB2022 reproduction: STRING200 vs Transcriptome."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV


ROOT = Path("/data/lsy/Infinite_Stream")
CODE = ROOT / "03_code/model_validation"
OUT = ROOT / "02_results/model_validation/20260506_lau_pcb2022_external_string200_studyz"


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lau5 = import_file("lau5_external", CODE / "run_lau_pcb2022_fivefold_reproduction_20260505.py")
reader = import_file("lau_external_reader_string200", CODE / "run_lau_style_external_total_validation_20260505.py")


def ensure_dirs() -> None:
    for d in [OUT, OUT / "tables", OUT / "logs", OUT / "predictions"]:
        d.mkdir(parents=True, exist_ok=True)
    (reader.OUT / "tables").mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"{time.ctime()} {msg}"
    print(line, flush=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def safe_corr(y: np.ndarray, p: np.ndarray):
    ok = np.isfinite(y) & np.isfinite(p)
    n = int(ok.sum())
    if n < 10:
        return np.nan, np.nan, n
    if np.nanstd(y[ok]) < 1e-8 or np.nanstd(p[ok]) < 1e-8:
        return 0.0, 0.0, n
    return float(pearsonr(y[ok], p[ok]).statistic), float(spearmanr(y[ok], p[ok]).correlation), n


def load_external(cohort: str):
    ecfg = reader.EXTERNAL[cohort]
    mod = import_file(f"external_reader_lau_{cohort}", ecfg["script"])
    if "expr_arg" in ecfg:
        x = getattr(mod, ecfg["expr"])(ecfg["expr_arg"])
    else:
        x = getattr(mod, ecfg["expr"])()
    y = getattr(mod, ecfg["truth"])()
    return reader.align_external_samples(cohort, x, y)


def fit_predict_one(feature_set: str, target: str, target_idx: int, x_train: pd.DataFrame, y_train: pd.DataFrame, x_ext: pd.DataFrame, y_ext: pd.DataFrame, feat_idx: list[int], cfg) -> dict:
    if target not in y_ext.columns or not feat_idx:
        return {"feature_set": feature_set, "target": target, "target_index": target_idx, "skipped": True}
    y = y_train[target].to_numpy(dtype=np.float32)
    ok = np.isfinite(y)
    if ok.sum() < cfg.min_obs:
        return {"feature_set": feature_set, "target": target, "target_index": target_idx, "skipped": True, "n_train": int(ok.sum())}
    x0 = x_train.to_numpy(dtype=np.float32)[:, feat_idx]
    xe0 = x_ext.reindex(columns=x_train.columns).to_numpy(dtype=np.float32)[:, feat_idx]
    imp = SimpleImputer(strategy="median")
    vt = VarianceThreshold(0.0)
    try:
        xt = vt.fit_transform(imp.fit_transform(x0))
        xe = vt.transform(imp.transform(xe0))
    except ValueError:
        return {"feature_set": feature_set, "target": target, "target_index": target_idx, "skipped": True, "n_train": int(ok.sum())}
    model = RidgeCV(alphas=np.logspace(-3, 4, 40), fit_intercept=True)
    t0 = time.time()
    model.fit(xt[ok], y[ok])
    pred = model.predict(xe)
    truth = y_ext[target].to_numpy(dtype=float)
    pp, ss, nn = safe_corr(truth, pred)
    return {
        "feature_set": feature_set,
        "target": target,
        "target_index": target_idx,
        "skipped": False,
        "n_train": int(ok.sum()),
        "n_external": nn,
        "n_features": int(xt.shape[1]),
        "pearson": pp,
        "spearman": ss,
        "seconds": float(time.time() - t0),
    }


def summarize(per: pd.DataFrame, meta: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for (cohort, feature_set), sub0 in per.groupby(["cohort", "feature_set"]):
        sub = sub0[sub0["skipped"] == False].copy()  # noqa: E712
        vals = pd.to_numeric(sub["pearson"], errors="coerce").dropna()
        sp = pd.to_numeric(sub["spearman"], errors="coerce").dropna()
        high = sub[pd.to_numeric(sub["n_train"], errors="coerce") >= 800]
        hv = pd.to_numeric(high["pearson"], errors="coerce").dropna()
        rows.append({
            "cohort": cohort,
            "feature_set": feature_set,
            "n_samples": meta[cohort]["n_samples"],
            "n_measured_proteins": meta[cohort]["n_measured_proteins"],
            "n_overlap_proteins": meta[cohort]["n_overlap_proteins"],
            "n_evaluable_proteins": int(len(vals)),
            "Pearson median": float(vals.median()) if len(vals) else np.nan,
            "Spearman median": float(sp.median()) if len(sp) else np.nan,
            "high-coverage Pearson median": float(hv.median()) if len(hv) else np.nan,
            "sample-level Pearson median": np.nan,
            "input RNA unit": meta[cohort]["input RNA unit"],
            "RNA conversion method": meta[cohort]["RNA conversion method"],
        })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    cfg = lau5.Config(n_jobs=args.n_jobs, label_file=args.label_file)
    x_train, y_train, manifest = lau5.load_data(cfg)
    rna_cols = list(x_train.columns.astype(str))
    targets = list(y_train.columns.astype(str))
    rna_genes = {g.upper() for g in rna_cols}
    target_genes = {g.upper() for g in targets}
    maps = {
        "corum": lau5.parse_corum(rna_genes, target_genes),
        "string800": lau5.parse_string_edges(rna_genes, target_genes, cfg.string800_min_score),
        "string200": lau5.parse_string_edges(rna_genes, target_genes, cfg.string200_min_score),
    }
    feature_sets = [x.strip() for x in args.feature_sets.split(",") if x.strip()]
    feat_map = {
        fs: {i: lau5.feature_indices(fs, t, rna_cols, maps) for i, t in enumerate(targets)}
        for fs in feature_sets
    }
    rows = []
    meta = {}
    for cohort in args.cohorts.split(","):
        cohort = cohort.strip()
        x_ext, y_ext = load_external(cohort)
        overlap = [t for t in y_ext.columns.astype(str) if t in set(targets)]
        if args.max_targets > 0:
            overlap = overlap[: args.max_targets]
        meta[cohort] = {
            "n_samples": int(len(x_ext)),
            "n_measured_proteins": int(y_ext.shape[1]),
            "n_overlap_proteins": int(len(overlap)),
            "input RNA unit": "reader-defined",
            "RNA conversion method": "existing external reader + clean_df + reindex CPTAC genes",
        }
        for fs in feature_sets:
            log(json.dumps({"cohort": cohort, "feature_set": fs, "targets": len(overlap)}, ensure_ascii=False))
            part = Parallel(n_jobs=cfg.n_jobs, backend="threading", verbose=5)(
                delayed(fit_predict_one)(fs, t, targets.index(t), x_train, y_train, x_ext, y_ext, feat_map[fs][targets.index(t)], cfg)
                for t in overlap
            )
            rows.extend([dict(r, cohort=cohort) for r in part])
            per = pd.DataFrame(rows)
            per.to_csv(OUT / "external_per_protein.tsv", sep="\t", index=False)
            summarize(per, meta).to_csv(OUT / "external_summary.tsv", sep="\t", index=False)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "external_per_protein.tsv", sep="\t", index=False)
    sm = summarize(per, meta)
    sm.to_csv(OUT / "external_summary.tsv", sep="\t", index=False)
    (OUT / "config.json").write_text(json.dumps({"feature_sets": feature_sets, "label_file": args.label_file, "cohorts": args.cohorts, "max_targets": args.max_targets}, indent=2, ensure_ascii=False), encoding="utf-8")
    log(sm.to_json(orient="records", force_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-sets", default="string200,transcriptome")
    parser.add_argument("--cohorts", default="FU_iCCA,TU_SCLC,CHCC_HBV_FPKM,CHCC_HBV_RSEM")
    parser.add_argument("--label-file", default="total_protein_gene_logratio_all.parquet")
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=16)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
