#!/usr/bin/env python3
"""Build the 32-project TCGA RNA to TCPA RPPA training contract."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/data/lsy/Infinite_Stream")
AUDIT = ROOT / "02_results/model_validation/20260501_tcpa_32_project_data_contract_v1"
OUT = ROOT / "01_data/tcga_tcpa/processed/tcpa_32_project_rna_rppa_20260501"
RAW = ROOT / "01_data/tcga_tcpa/raw/gdc_star_counts_missing_13_projects"
GDC_DATA = "https://api.gdc.cancer.gov/data"


def mkdirs() -> None:
    for sub in ["matrices", "tables", "logs"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)


def clean_antibody(name: str) -> str:
    name = str(name)
    if name.startswith("X") and len(name) > 1 and name[1].isdigit():
        return name[1:]
    return name


def read_missing_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(
        AUDIT / "tables/gdc_star_counts_manifest_missing_13_projects_matched_to_rppa.tsv",
        sep="\t",
    )
    manifest = manifest.loc[manifest["matches_rppa_l4"].astype(bool)].copy()
    manifest["sample_short"] = manifest["sample_short"].astype(str)
    manifest["project_id"] = manifest["project_id"].astype(str)
    manifest = manifest.drop_duplicates("sample_short", keep="first")
    return manifest.sort_values(["project_id", "sample_short"]).reset_index(drop=True)


def read_rppa_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(ROOT / "01_data/tcga_tcpa/raw/gdc_tcga_rppa_manifest.tsv", sep="\t")
    manifest["sample_submitter_id"] = manifest["sample_submitter_id"].astype(str)
    manifest["sample_short"] = manifest["sample_submitter_id"].str[:16]
    manifest = manifest.drop_duplicates("sample_short", keep="first")
    return manifest[["sample_short", "case_submitter_id", "sample_submitter_id", "project_id"]].copy()


def target_path(row: pd.Series) -> Path:
    return RAW / str(row["project_id"]) / str(row["file_id"]) / str(row["file_name"])


def download_one(row: pd.Series, retries: int = 5, sleep: float = 2.0) -> dict:
    path = target_path(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        return {
            "sample_short": row["sample_short"],
            "project_id": row["project_id"],
            "file_id": row["file_id"],
            "path": str(path),
            "status": "exists",
            "bytes": path.stat().st_size,
        }

    url = f"{GDC_DATA}/{row['file_id']}"
    tmp = path.with_suffix(path.suffix + ".tmp")
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=180) as response, tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if tmp.stat().st_size <= 1_000_000:
                raise RuntimeError(f"downloaded file is too small: {tmp.stat().st_size}")
            tmp.replace(path)
            return {
                "sample_short": row["sample_short"],
                "project_id": row["project_id"],
                "file_id": row["file_id"],
                "path": str(path),
                "status": "downloaded",
                "bytes": path.stat().st_size,
            }
        except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
            last_error = str(exc)
            if tmp.exists():
                tmp.unlink()
            time.sleep(sleep * attempt)

    return {
        "sample_short": row["sample_short"],
        "project_id": row["project_id"],
        "file_id": row["file_id"],
        "path": str(path),
        "status": "failed",
        "bytes": 0,
        "error": last_error,
    }


def download_missing(manifest: pd.DataFrame, threads: int) -> pd.DataFrame:
    rows = manifest.to_dict(orient="records")
    results = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(download_one, pd.Series(row)) for row in rows]
        for i, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            results.append(result)
            if i == 1 or i % 25 == 0 or i == len(futures):
                elapsed = time.time() - started
                print(f"download {i}/{len(futures)} elapsed={elapsed:.1f}s status={result['status']}", flush=True)
    tab = pd.DataFrame(results)
    tab.to_csv(OUT / "tables/gdc_star_counts_missing_13_download_status.tsv", sep="\t", index=False)
    failed = tab.loc[tab["status"].eq("failed")]
    if not failed.empty:
        raise SystemExit(f"{failed.shape[0]} downloads failed; see {OUT / 'tables/gdc_star_counts_missing_13_download_status.tsv'}")
    return tab


def load_current_19_rna(rppa_manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_raw = pd.read_parquet(ROOT / "data/processed/X_all.symbols.parquet")
    x_raw = x_raw.drop_duplicates("gene_symbol", keep="first").set_index("gene_symbol")
    x_raw.index = x_raw.index.astype(str)

    cols = [c for c in x_raw.columns if str(c) != "gene_symbol"]
    samples = pd.DataFrame({"rna_sample_id": [str(c) for c in cols]})
    samples["sample_short"] = samples["rna_sample_id"].str[:16]
    samples = samples.merge(rppa_manifest[["sample_short", "project_id"]], on="sample_short", how="inner")
    samples = samples.sort_values(["project_id", "sample_short", "rna_sample_id"])
    samples = samples.drop_duplicates("sample_short", keep="first").reset_index(drop=True)

    x = x_raw.loc[:, samples["rna_sample_id"].tolist()].copy()
    x = x.apply(pd.to_numeric, errors="coerce")
    return x, samples


def parse_star_tpm(path: Path, gene_order: list[str]) -> pd.Series:
    tab = pd.read_csv(path, sep="\t", comment="#", usecols=["gene_name", "tpm_unstranded"])
    tab = tab.dropna(subset=["gene_name"])
    tab["gene_name"] = tab["gene_name"].astype(str)
    tab = tab.loc[~tab["gene_name"].str.startswith("N_")]
    tab["value"] = np.log2(pd.to_numeric(tab["tpm_unstranded"], errors="coerce").fillna(0.0) + 1.0)
    tab = tab.groupby("gene_name", sort=False)["value"].mean()
    return tab.reindex(gene_order).astype("float32")


def build_missing_13_rna(missing_manifest: pd.DataFrame, gene_order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    series = []
    samples = []
    started = time.time()
    for i, row in missing_manifest.iterrows():
        path = target_path(row)
        if not path.exists():
            raise FileNotFoundError(path)
        values = parse_star_tpm(path, gene_order)
        rna_sample_id = str(row["sample_short"])
        values.name = rna_sample_id
        series.append(values)
        samples.append(
            {
                "rna_sample_id": rna_sample_id,
                "sample_short": str(row["sample_short"]),
                "project_id": str(row["project_id"]),
                "source": "gdc_star_counts_downloaded_20260501",
                "file_id": str(row["file_id"]),
                "file_name": str(row["file_name"]),
                "raw_path": str(path),
            }
        )
        if (i + 1) == 1 or (i + 1) % 50 == 0 or (i + 1) == missing_manifest.shape[0]:
            print(f"parse missing RNA {i + 1}/{missing_manifest.shape[0]} elapsed={time.time() - started:.1f}s", flush=True)
    x = pd.concat(series, axis=1)
    return x, pd.DataFrame(samples)


def build_labels(sample_manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rppa = pd.read_csv(ROOT / "data/raw/tcpa/PANCAN_RPPA_L4.tsv", sep="\t")
    rppa["sample_id"] = rppa["sample_id"].astype(str)
    rppa["sample_short"] = rppa["sample_id"].str[:16]
    rppa = rppa.rename(columns={c: clean_antibody(c) for c in rppa.columns})
    rppa = rppa.drop_duplicates("sample_short", keep="first")
    antibody_cols = [c for c in rppa.columns if c not in {"sample_id", "sample_short"}]

    y = sample_manifest[["rna_sample_id", "sample_short"]].merge(
        rppa[["sample_short"] + antibody_cols],
        on="sample_short",
        how="left",
    )
    y = y.set_index("rna_sample_id")[antibody_cols].apply(pd.to_numeric, errors="coerce")

    phos = set((ROOT / "metadata/phospho_proteins.txt").read_text().split())
    panel = pd.DataFrame(
        {
            "antibody": antibody_cols,
            "type": ["phospho" if c in phos else "total" for c in antibody_cols],
            "n_observed": [int(y[c].notna().sum()) for c in antibody_cols],
        }
    )
    return y, panel


def write_parquet(df: pd.DataFrame, path: Path, index_name: str | None = None) -> None:
    out = df.copy()
    if index_name is not None:
        out.index.name = index_name
        out = out.reset_index()
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), path)


def build_contract() -> dict:
    rppa_manifest = read_rppa_manifest()
    missing_manifest = read_missing_manifest()
    current_x, current_samples = load_current_19_rna(rppa_manifest)
    current_samples["source"] = "existing_tcga_star_log2_tpm_matrix"
    current_samples["file_id"] = ""
    current_samples["file_name"] = ""
    current_samples["raw_path"] = ""

    gene_order = list(current_x.index)
    missing_x, missing_samples = build_missing_13_rna(missing_manifest, gene_order)
    sample_manifest = pd.concat([current_samples, missing_samples], ignore_index=True)
    sample_manifest["patient_id"] = sample_manifest["sample_short"].str[:12]
    sample_manifest["sample_type"] = "Primary Tumor"
    sample_manifest = sample_manifest.sort_values(["project_id", "sample_short", "rna_sample_id"]).reset_index(drop=True)

    x_all = pd.concat([current_x, missing_x], axis=1)
    x_all = x_all.loc[:, sample_manifest["rna_sample_id"].tolist()]
    y_all, antibody_panel = build_labels(sample_manifest)
    y_all = y_all.loc[sample_manifest["rna_sample_id"]]

    observed = y_all.notna().sum(axis=1)
    keep = observed > 0
    sample_manifest = sample_manifest.loc[keep.to_numpy()].reset_index(drop=True)
    x_all = x_all.loc[:, sample_manifest["rna_sample_id"].tolist()]
    y_all = y_all.loc[sample_manifest["rna_sample_id"]]

    project_counts = (
        sample_manifest.groupby("project_id", dropna=False)
        .agg(n_samples=("rna_sample_id", "count"))
        .reset_index()
        .sort_values("project_id")
    )
    project_counts.to_csv(OUT / "tables/tcpa_32_project_sample_counts.tsv", sep="\t", index=False)
    sample_manifest.to_csv(OUT / "tables/tcpa_32_sample_manifest.tsv", sep="\t", index=False)
    antibody_panel.to_csv(OUT / "tables/tcpa_32_antibody_panel.tsv", sep="\t", index=False)
    write_parquet(x_all, OUT / "matrices/X_tcpa_32.symbols.parquet", index_name="gene_symbol")
    write_parquet(y_all, OUT / "matrices/Y_tcpa_32.rppa.parquet", index_name="rna_sample_id")

    total = antibody_panel.loc[antibody_panel["type"].eq("total")]
    phospho = antibody_panel.loc[antibody_panel["type"].eq("phospho")]
    summary = {
        "data_contract": "tcpa_32_project_rna_rppa_20260501",
        "n_projects": int(project_counts.shape[0]),
        "projects": project_counts["project_id"].tolist(),
        "n_samples": int(sample_manifest.shape[0]),
        "n_genes": int(x_all.shape[0]),
        "n_antibodies": int(antibody_panel.shape[0]),
        "n_total_antibodies": int(total.shape[0]),
        "n_phospho_antibodies": int(phospho.shape[0]),
        "n_current_19_samples": int(sample_manifest["source"].eq("existing_tcga_star_log2_tpm_matrix").sum()),
        "n_downloaded_13_samples": int(sample_manifest["source"].eq("gdc_star_counts_downloaded_20260501").sum()),
        "matrices": {
            "rna": str(OUT / "matrices/X_tcpa_32.symbols.parquet"),
            "rppa": str(OUT / "matrices/Y_tcpa_32.rppa.parquet"),
        },
        "tables": {
            "sample_manifest": str(OUT / "tables/tcpa_32_sample_manifest.tsv"),
            "antibody_panel": str(OUT / "tables/tcpa_32_antibody_panel.tsv"),
            "project_counts": str(OUT / "tables/tcpa_32_project_sample_counts.tsv"),
        },
    }
    (OUT / "logs/tcpa_32_project_rna_rppa_contract_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    mkdirs()
    manifest = read_missing_manifest()
    if args.download:
        download_missing(manifest, threads=max(1, args.threads))
    if args.build:
        summary = build_contract()
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if not args.download and not args.build:
        raise SystemExit("use --download and/or --build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
