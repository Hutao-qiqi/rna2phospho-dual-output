#!/usr/bin/env python3
"""Reproduce Srivastava/Lau PLOS Comput Biol 2022 models under SCP682 five-fold CV.

Paper model family:
- one model per protein
- RNA feature groups: Single, CORUM, STRING 800, STRING 200, Transcriptome
- estimators: linear/ridge, elastic net, random forest, gradient boosting

This implementation keeps the model idea but replaces their single 80:20 split
with one fixed stratified five-fold CPTAC/PDC split.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/data/lsy/Infinite_Stream")
CPTAC = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2"
PRIOR = ROOT / "01_data/pathway_prior/raw"
OUT = ROOT / "02_results/model_validation/20260505_lau_pcb2022_fivefold_reproduction_full_ridge"


@dataclass(frozen=True)
class Config:
    seed: int = 6822022
    n_splits: int = 5
    label_file: str = "total_protein_gene_logratio_all.parquet"
    min_obs: int = 50
    min_fold_obs: int = 5
    string200_min_score: int = 200
    string800_min_score: int = 800
    n_jobs: int = 16
    rf_estimators: int = 500
    rf_max_depth: int = 4
    gb_estimators: int = 700
    gb_learning_rate: float = 0.025
    gb_max_depth: int = 3
    gb_subsample: float = 0.7


def ensure_dirs() -> None:
    for d in [OUT, OUT / "tables", OUT / "logs", OUT / "predictions"]:
        d.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"{time.ctime()} {msg}"
    print(line, flush=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def safe_corr(y: np.ndarray, p: np.ndarray) -> tuple[float, float, int]:
    ok = np.isfinite(y) & np.isfinite(p)
    n = int(ok.sum())
    if n < 10:
        return np.nan, np.nan, n
    if np.nanstd(y[ok]) < 1e-8 or np.nanstd(p[ok]) < 1e-8:
        return 0.0, 0.0, n
    return float(pearsonr(y[ok], p[ok]).statistic), float(spearmanr(y[ok], p[ok]).correlation), n


def load_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(CPTAC / "sample_manifest.tsv", sep="\t").set_index("sample_id")
    x = pd.read_parquet(CPTAC / "rna_log2_tpm_paired.parquet")
    y = pd.read_parquet(CPTAC / cfg.label_file)
    common = manifest.index.intersection(x.index).intersection(y.index)
    return x.loc[common].astype(np.float32), y.loc[common].astype(np.float32), manifest.loc[common].copy()


def build_folds(manifest: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    path = OUT / "sample_split_5fold.tsv"
    if path.exists():
        return pd.read_csv(path, sep="\t")
    labels = manifest["cancer_label"].astype(str).to_numpy()
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    fold = np.full(len(manifest), -1, dtype=int)
    for k, (_, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        fold[test_idx] = k
    out = pd.DataFrame({
        "sample_id": manifest.index.astype(str),
        "fold": fold,
        "cancer_label": manifest["cancer_label"].astype(str).to_numpy(),
        "pdc_study_id": manifest["pdc_study_id"].astype(str).to_numpy(),
    })
    out.to_csv(path, sep="\t", index=False)
    return out


def parse_string_edges(rna_genes: set[str], targets: set[str], min_score: int) -> dict[str, list[str]]:
    cache = OUT / "tables" / f"string_neighbors_score{min_score}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    info_path = PRIOR / "string/9606.protein.info.v12.0.txt.gz"
    links_path = PRIOR / "string/9606.protein.links.v12.0.txt.gz"
    id_to_gene: dict[str, str] = {}
    with gzip.open(info_path, "rt", encoding="utf-8", errors="ignore") as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                g = p[1].upper()
                if g in rna_genes or g in targets:
                    id_to_gene[p[0]] = g
    score_map: dict[str, dict[str, int]] = {}
    with gzip.open(links_path, "rt", encoding="utf-8", errors="ignore") as fh:
        next(fh, None)
        for line in fh:
            p = line.strip().split()
            if len(p) < 3:
                continue
            score = int(p[2])
            if score < min_score:
                continue
            a = id_to_gene.get(p[0])
            b = id_to_gene.get(p[1])
            if a is None or b is None or a == b:
                continue
            if a in targets and b in rna_genes:
                score_map.setdefault(a, {})[b] = max(score_map.setdefault(a, {}).get(b, 0), score)
            if b in targets and a in rna_genes:
                score_map.setdefault(b, {})[a] = max(score_map.setdefault(b, {}).get(a, 0), score)
    out = {k: [g for g, _ in sorted(v.items(), key=lambda x: -x[1])] for k, v in score_map.items()}
    cache.write_text(json.dumps(out))
    return out


def parse_corum(rna_genes: set[str], targets: set[str]) -> dict[str, list[str]]:
    cache = OUT / "tables" / "corum_neighbors.json"
    if cache.exists():
        return json.loads(cache.read_text())
    p = PRIOR / "corum/humanComplexes.txt"
    out_map: dict[str, set[str]] = {}
    df = pd.read_csv(p, sep="\t")
    for _, row in df.iterrows():
        genes = [str(x).strip().upper() for x in str(row.get("subunits_gene_name", "")).split(";")]
        genes = [g for g in genes if g and g != "NAN"]
        gset = set(genes)
        for t in gset & targets:
            members = [g for g in genes if g != t and g in rna_genes]
            if members:
                out_map.setdefault(t, set()).update(members)
    out = {k: sorted(v) for k, v in out_map.items()}
    cache.write_text(json.dumps(out))
    return out


def feature_indices(feature_set: str, target: str, rna_cols: list[str], maps: dict[str, dict[str, list[str]]]) -> list[int]:
    gene_to_idx = {g.upper(): i for i, g in enumerate(rna_cols)}
    tu = target.upper()
    genes: list[str] = []
    if feature_set == "single":
        genes = [tu] if tu in gene_to_idx else []
    elif feature_set == "corum":
        genes = ([tu] if tu in gene_to_idx else []) + maps["corum"].get(tu, [])
    elif feature_set == "string800":
        genes = ([tu] if tu in gene_to_idx else []) + maps["string800"].get(tu, [])
    elif feature_set == "string200":
        genes = ([tu] if tu in gene_to_idx else []) + maps["string200"].get(tu, [])
    elif feature_set == "transcriptome":
        return list(range(len(rna_cols)))
    else:
        raise ValueError(feature_set)
    out, seen = [], set()
    for g in genes:
        gu = g.upper()
        if gu in gene_to_idx and gu not in seen:
            out.append(gene_to_idx[gu])
            seen.add(gu)
    return out


def make_model(method: str, cfg: Config) -> Any:
    if method == "linear":
        return RidgeCV(alphas=[0.0, 1e-6, 1e-4, 1e-2, 1.0], fit_intercept=True)
    if method == "ridge":
        return RidgeCV(alphas=np.logspace(-3, 4, 40), fit_intercept=True)
    if method == "elasticnet":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 0.95], cv=5, max_iter=3000, tol=1e-3, random_state=cfg.seed, n_jobs=1)),
        ])
    if method == "randomforest":
        return RandomForestRegressor(n_estimators=cfg.rf_estimators, max_depth=cfg.rf_max_depth, random_state=cfg.seed, n_jobs=1)
    if method == "gradientboosting":
        return GradientBoostingRegressor(n_estimators=cfg.gb_estimators, learning_rate=cfg.gb_learning_rate, max_depth=cfg.gb_max_depth, subsample=cfg.gb_subsample, random_state=cfg.seed)
    raise ValueError(method)


def fit_target(method: str, feature_set: str, target_idx: int, target: str, x: np.ndarray, y_mat: np.ndarray, folds: np.ndarray, feat_idx: list[int], cfg: Config) -> tuple[dict[str, Any], pd.DataFrame]:
    y = y_mat[:, target_idx]
    if np.isfinite(y).sum() < cfg.min_obs or len(feat_idx) == 0:
        return {"method": method, "feature_set": feature_set, "target": target, "target_index": target_idx, "skipped": True, "n_features": len(feat_idx), "n_obs": int(np.isfinite(y).sum())}, pd.DataFrame()
    pred = np.full(len(y), np.nan, dtype=np.float32)
    rows = []
    x0 = x[:, feat_idx]
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        ok_train = train & np.isfinite(y)
        ok_test = test & np.isfinite(y)
        if ok_train.sum() < cfg.min_obs or ok_test.sum() < cfg.min_fold_obs:
            continue
        imp = SimpleImputer(strategy="median")
        vt = VarianceThreshold(0.0)
        try:
            xt = vt.fit_transform(imp.fit_transform(x0[train]))
            xv = vt.transform(imp.transform(x0[test]))
        except ValueError:
            continue
        model = make_model(method, cfg)
        model.fit(xt[ok_train[train]], y[ok_train])
        pred[test] = model.predict(xv)
        pp, ss, nn = safe_corr(y[test], pred[test])
        rows.append({"method": method, "feature_set": feature_set, "target": target, "target_index": target_idx, "fold": int(fold), "n_train": int(ok_train.sum()), "n_test": nn, "pearson": pp, "spearman": ss, "n_features": int(xt.shape[1])})
    pp, ss, nn = safe_corr(y, pred)
    return {"method": method, "feature_set": feature_set, "target": target, "target_index": target_idx, "skipped": False, "n_obs": int(np.isfinite(y).sum()), "n_oof": nn, "n_features": int(len(feat_idx)), "pearson": pp, "spearman": ss}, pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, feature_set), sub0 in df.groupby(["method", "feature_set"]):
        sub = sub0[sub0["skipped"] == False].copy()  # noqa: E712
        vals = pd.to_numeric(sub["pearson"], errors="coerce").dropna()
        sp = pd.to_numeric(sub["spearman"], errors="coerce").dropna()
        rows.append({"method": method, "feature_set": feature_set, "n_targets": int(len(vals)), "Pearson median": float(vals.median()) if len(vals) else np.nan, "Spearman median": float(sp.median()) if len(sp) else np.nan, "r > 0.5 count": int((vals > 0.5).sum()) if len(vals) else 0, "r > 0.3 count": int((vals > 0.3).sum()) if len(vals) else 0})
    return pd.DataFrame(rows).sort_values("Pearson median", ascending=False)


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    cfg = Config(n_jobs=args.n_jobs, label_file=args.label_file)
    (OUT / "config.json").write_text(json.dumps({**asdict(cfg), "methods": args.methods, "feature_sets": args.feature_sets}, indent=2, ensure_ascii=False), encoding="utf-8")
    x_df, y_df, manifest = load_data(cfg)
    split = build_folds(manifest, cfg)
    folds = split["fold"].to_numpy(dtype=int)
    rna_cols = list(x_df.columns.astype(str))
    target_cols = list(y_df.columns.astype(str))
    rna_genes = {g.upper() for g in rna_cols}
    targets = {g.upper() for g in target_cols}
    maps = {
        "corum": parse_corum(rna_genes, targets),
        "string800": parse_string_edges(rna_genes, targets, cfg.string800_min_score),
        "string200": parse_string_edges(rna_genes, targets, cfg.string200_min_score),
    }
    selected = list(enumerate(target_cols))
    if args.max_targets > 0:
        selected = selected[: args.max_targets]
    x = x_df.to_numpy(dtype=np.float32)
    y = y_df.to_numpy(dtype=np.float32)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    feature_sets = [f.strip() for f in args.feature_sets.split(",") if f.strip()]
    all_rows = []
    fold_rows = []
    manifest_rows = []
    for feature_set in feature_sets:
        feat_map = {i: feature_indices(feature_set, t, rna_cols, maps) for i, t in selected}
        for i, t in selected:
            manifest_rows.append({"target": t, "target_index": i, "feature_set": feature_set, "n_features": len(feat_map[i])})
        pd.DataFrame(manifest_rows).to_csv(OUT / "feature_manifest.tsv", sep="\t", index=False)
        for method in methods:
            log(f"start method={method} feature_set={feature_set} targets={len(selected)}")
            results = Parallel(n_jobs=cfg.n_jobs, backend="threading", verbose=5)(
                delayed(fit_target)(method, feature_set, i, t, x, y, folds, feat_map[i], cfg)
                for i, t in selected
            )
            for row, fdf in results:
                all_rows.append(row)
                if not fdf.empty:
                    fold_rows.append(fdf)
            per = pd.DataFrame(all_rows)
            per.to_csv(OUT / "per_protein_metrics.tsv", sep="\t", index=False)
            if fold_rows:
                pd.concat(fold_rows, ignore_index=True).to_csv(OUT / "per_fold_metrics.tsv", sep="\t", index=False)
            sm = summarize(per)
            sm.to_csv(OUT / "summary_metrics.tsv", sep="\t", index=False)
            log(sm.head(20).to_json(orient="records", force_ascii=False))
    report = [
        "# Lau PLOS Computational Biology 2022 reproduction under SCP682 five-fold CV",
        "",
        "Protocol: fixed stratified five-fold CPTAC/PDC CV; one model per protein; OOF per-protein Pearson/Spearman.",
        "",
    ]
    if (OUT / "summary_metrics.tsv").exists():
        report.append(pd.read_csv(OUT / "summary_metrics.tsv", sep="\t").to_markdown(index=False))
    (OUT / "model_selection_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="ridge,elasticnet,randomforest,gradientboosting")
    parser.add_argument("--feature-sets", default="single,corum,string800,string200,transcriptome")
    parser.add_argument("--label-file", default="total_protein_gene_logratio_all.parquet")
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=16)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
