#!/usr/bin/env python
"""Adapt Srivastava/Lau 2022 RNA-to-protein baselines to TCGA RPPA targets."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def normalize_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def split_gene_field(value: object) -> list[str]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    parts: list[str] = []
    for raw in re.split(r"[,;/]", text):
        raw = raw.strip()
        if not raw:
            continue
        if re.fullmatch(r"\d+", raw) and parts:
            prefix = re.sub(r"\d+$", "", parts[-1])
            parts.append(prefix + raw)
        else:
            parts.append(raw)
    return [p for p in parts if p]


def read_antibody_mapping(project_root: Path, proteins: list[str], rna_genes: set[str]) -> pd.DataFrame:
    mapping_paths = [
        project_root / "01_data/tcga_tcpa/raw/rppa_antibody_reliability/RPPA_Expanded_Ab_List_Updated.xlsx",
        project_root / "01_data/tcga_tcpa/raw/rppa_antibody_reliability/7_RPPA_Standard_Ab_List.xlsx",
    ]
    frames = []
    for path in mapping_paths:
        if not path.exists():
            continue
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            for header in range(0, 10):
                try:
                    df = pd.read_excel(path, sheet_name=sheet, header=header)
                except Exception:
                    continue
                df.columns = [str(c).strip() for c in df.columns]
                needed = {"Official Ab Name", "Ab Name Reported on Dataset", "Gene Name"}
                if needed.issubset(set(df.columns)):
                    sub = df[["Official Ab Name", "Ab Name Reported on Dataset", "Gene Name"]].copy()
                    sub["source_file"] = path.name
                    sub["sheet"] = sheet
                    frames.append(sub)
                    break
    if not frames:
        raise RuntimeError("No RPPA antibody mapping table could be parsed.")

    raw = pd.concat(frames, ignore_index=True).dropna(subset=["Gene Name"], how="any")
    for col in ["Official Ab Name", "Ab Name Reported on Dataset", "Gene Name"]:
        raw[col] = raw[col].astype(str).str.strip()
    raw = raw[(raw["Gene Name"] != "Gene Name") & (raw["Gene Name"].str.lower() != "nan")]

    lookup: dict[str, list[str]] = {}
    for _, row in raw.iterrows():
        genes = split_gene_field(row["Gene Name"])
        for key_col in ["Official Ab Name", "Ab Name Reported on Dataset"]:
            key = normalize_name(row[key_col])
            if key and key not in lookup:
                lookup[key] = genes

    rows = []
    for protein in proteins:
        genes = lookup.get(normalize_name(protein), [])
        available = [g for g in genes if g in rna_genes]
        rows.append(
            {
                "protein": protein,
                "mapped_genes_raw": ",".join(genes),
                "mapped_genes_available": ",".join(available),
                "n_available_genes": len(available),
            }
        )
    return pd.DataFrame(rows)


def clean_gene_matrix(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["gene_symbol"] = x["gene_symbol"].astype(str)
    x = x[(x["gene_symbol"] != "") & (x["gene_symbol"].str.lower() != "nan")]
    x = x.groupby("gene_symbol", sort=False).mean(numeric_only=True)
    return x.T


def metric_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5:
        return math.nan, math.nan
    yt = y_true[mask]
    yp = y_pred[mask]
    if np.std(yt) == 0 or np.std(yp) == 0:
        return 0.0, 0.0
    return float(pearsonr(yt, yp)[0]), float(spearmanr(yt, yp)[0])


def make_model(method: str):
    if method == "linreg":
        return LinearRegression()
    if method == "elastic":
        return ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 0.95],
            cv=5,
            fit_intercept=False,
            n_jobs=1,
            tol=1e-3,
            max_iter=2000,
            random_state=2,
        )
    if method == "forest":
        return RandomForestRegressor(
            n_estimators=500,
            criterion="squared_error",
            max_depth=4,
            random_state=2,
            n_jobs=1,
        )
    if method == "boosting":
        return GradientBoostingRegressor(
            n_estimators=1000,
            max_depth=3,
            subsample=0.5,
            min_samples_split=5,
            learning_rate=0.025,
            random_state=2,
        )
    raise ValueError(method)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, feature_set), sub in detail.groupby(["method", "feature_set"], dropna=False):
        valid = sub.dropna(subset=["spearman"])
        rows.append(
            {
                "method": method,
                "feature_set": feature_set,
                "n_targets": int(len(valid)),
                "median_pearson": float(valid["pearson"].median()) if len(valid) else math.nan,
                "mean_pearson": float(valid["pearson"].mean()) if len(valid) else math.nan,
                "median_spearman": float(valid["spearman"].median()) if len(valid) else math.nan,
                "mean_spearman": float(valid["spearman"].mean()) if len(valid) else math.nan,
                "spearman_q25": float(valid["spearman"].quantile(0.25)) if len(valid) else math.nan,
                "spearman_q75": float(valid["spearman"].quantile(0.75)) if len(valid) else math.nan,
                "n_spearman_ge_0_5": int((valid["spearman"] >= 0.5).sum()) if len(valid) else 0,
                "n_spearman_ge_0_7": int((valid["spearman"] >= 0.7).sum()) if len(valid) else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["median_spearman", "median_pearson"], ascending=False)


def run(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root)
    out_dir = Path(args.out_dir)
    table_dir = out_dir / "tables"
    log_dir = out_dir / "logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Loading RNA symbol matrix")
    x_raw = pd.read_parquet(project_root / "data/processed/X_all.symbols.parquet")
    x = clean_gene_matrix(x_raw)
    del x_raw
    print("RNA samples x genes", x.shape)

    print("Loading RPPA target matrix")
    y_raw = pd.read_parquet(project_root / "data/processed/Y.rppa.imputed.parquet")
    y = y_raw.set_index("protein").T
    y.index = y.index.astype(str)
    y.columns = y.columns.astype(str)
    print("RPPA samples x antibodies", y.shape)

    print("Loading active model OOF predictions")
    active = pd.read_parquet(project_root / "data/interim/oof_predictions.parquet")
    active = active.set_index("sample")
    active.index = active.index.astype(str)
    active.columns = active.columns.astype(str)

    samples = sorted(set(x.index) & set(y.index) & set(active.index))
    proteins = [p for p in y.columns if p in active.columns]
    x = x.loc[samples]
    y = y.loc[samples, proteins]
    active = active.loc[samples, proteins]
    print("Aligned", len(samples), "samples and", len(proteins), "targets")

    train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=2)
    train_samples = list(train_samples)
    test_samples = list(test_samples)

    mapping = read_antibody_mapping(project_root, proteins, set(x.columns))
    mapping.to_csv(table_dir / "rppa_antibody_to_gene_mapping.tsv", sep="\t", index=False)

    details = []
    # Active VAE+MLP evaluated on the same Srivastava/Lau 20% holdout samples.
    for protein in proteins:
        pear, spear = metric_pair(y.loc[test_samples, protein].to_numpy(float), active.loc[test_samples, protein].to_numpy(float))
        details.append(
            {
                "method": "active_vae_mlp_oof",
                "feature_set": "vae_latent",
                "protein": protein,
                "n_features": 16,
                "n_test": len(test_samples),
                "pearson": pear,
                "spearman": spear,
            }
        )

    # Faithful adaptation of Srivastava/Lau single-gene feature setting.
    protein_to_genes = dict(zip(mapping["protein"], mapping["mapped_genes_available"]))
    for method in ["linreg", "elastic", "forest", "boosting"]:
        print("Running single-gene", method)
        for i, protein in enumerate(proteins, start=1):
            gene_text = protein_to_genes.get(protein, "")
            genes = [g for g in gene_text.split(",") if g]
            if not genes:
                details.append(
                    {
                        "method": method,
                        "feature_set": "single_gene_mapped",
                        "protein": protein,
                        "n_features": 0,
                        "n_test": len(test_samples),
                        "pearson": math.nan,
                        "spearman": math.nan,
                    }
                )
                continue
            model = make_model(method)
            x_train = x.loc[train_samples, genes].to_numpy(float)
            x_test = x.loc[test_samples, genes].to_numpy(float)
            y_train = y.loc[train_samples, protein].to_numpy(float)
            y_test = y.loc[test_samples, protein].to_numpy(float)
            imputer = SimpleImputer(missing_values=np.nan, strategy="median")
            x_train = imputer.fit_transform(x_train)
            x_test = imputer.transform(x_test)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            pear, spear = metric_pair(y_test, pred)
            details.append(
                {
                    "method": method,
                    "feature_set": "single_gene_mapped",
                    "protein": protein,
                    "n_features": len(genes),
                    "n_test": len(test_samples),
                    "pearson": pear,
                    "spearman": spear,
                }
            )
            if i % 100 == 0:
                print(method, i, "targets")

    # A stronger linear sanity baseline using the most variable RNA genes.
    print("Running all-RNA Ridge top variance baseline")
    variances = x.loc[train_samples].var(axis=0).sort_values(ascending=False)
    top_genes = variances.head(args.top_genes).index.tolist()
    imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    x_train = imputer.fit_transform(x.loc[train_samples, top_genes].to_numpy(float))
    x_test = imputer.transform(x.loc[test_samples, top_genes].to_numpy(float))
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    ridge = Ridge(alpha=args.ridge_alpha, random_state=2)
    ridge.fit(x_train, y.loc[train_samples, proteins].to_numpy(float))
    ridge_pred = ridge.predict(x_test)
    for j, protein in enumerate(proteins):
        pear, spear = metric_pair(y.loc[test_samples, protein].to_numpy(float), ridge_pred[:, j])
        details.append(
            {
                "method": f"ridge_alpha_{args.ridge_alpha:g}",
                "feature_set": f"all_rna_top{args.top_genes}",
                "protein": protein,
                "n_features": len(top_genes),
                "n_test": len(test_samples),
                "pearson": pear,
                "spearman": spear,
            }
        )

    detail_df = pd.DataFrame(details)
    detail_path = table_dir / "srivastava_lau2022_rppa_adapted_baseline_detail.tsv"
    summary_path = table_dir / "srivastava_lau2022_rppa_adapted_baseline_summary.tsv"
    detail_df.to_csv(detail_path, sep="\t", index=False)
    summarize(detail_df).to_csv(summary_path, sep="\t", index=False)

    run_log = {
        "project_root": str(project_root),
        "n_samples_aligned": len(samples),
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "n_targets": len(proteins),
        "n_mapped_targets_with_available_rna_gene": int((mapping["n_available_genes"] > 0).sum()),
        "top_genes": args.top_genes,
        "ridge_alpha": args.ridge_alpha,
        "outputs": {
            "mapping": str(table_dir / "rppa_antibody_to_gene_mapping.tsv"),
            "detail": str(detail_path),
            "summary": str(summary_path),
        },
    }
    (log_dir / "run_srivastava_lau2022_rppa_baseline.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    print(json.dumps(run_log, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/data/lsy/Infinite_Stream")
    parser.add_argument(
        "--out-dir",
        default="/data/lsy/Infinite_Stream/02_results/model_validation/20260428_srivastava_lau2022_baseline_review",
    )
    parser.add_argument("--top-genes", type=int, default=5000)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
