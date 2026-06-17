#!/usr/bin/env python3
"""Download PDC quantDataMatrix outputs for CPTAC phosphoproteome studies."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path("/data/lsy/Infinite_Stream")
OVERLAP_PATH = PROJECT_ROOT / "02_results/model_validation/20260426_pdc_pancancer_phosphoproteome_feasibility/tables/pdc_cptac_phosphoproteome_studies_with_gdc_open_rna_overlap.tsv"
RAW_ROOT = PROJECT_ROOT / "01_data/multi_omics/raw/cptac_pancancer_phosphoproteome_quant_matrix"
OUT_DIR = PROJECT_ROOT / "02_results/model_validation/20260426_pdc_pancancer_phosphoproteome_feasibility"
LOG_DIR = OUT_DIR / "logs"
TABLE_DIR = OUT_DIR / "tables"

PDC_URL = "https://pdc.cancer.gov/graphql"
EXCLUDE_KEYWORDS = ["CompRef", "Kinase Inhibition", "Pediatric", "T-ALL"]


def should_keep(name: str, tumor_rna_n: int) -> bool:
    if tumor_rna_n < 20:
        return False
    return not any(k.lower() in name.lower() for k in EXCLUDE_KEYWORDS)


def download_matrix(study_id: str, out_path: Path, timeout: int) -> dict:
    if out_path.exists() and out_path.stat().st_size > 0:
        return {"status": "existing", "bytes": out_path.stat().st_size, "elapsed_sec": 0.0, "error": ""}

    query = f'{{ quantDataMatrix(pdc_study_id:"{study_id}" data_type:"log2_ratio" acceptDUA:true) }}'
    started = time.time()
    try:
        resp = requests.post(PDC_URL, json={"query": query}, timeout=timeout)
        elapsed = time.time() - started
        if not resp.ok:
            return {"status": "http_failed", "bytes": 0, "elapsed_sec": elapsed, "error": resp.text[:500]}
        data = resp.json()
        if data.get("errors"):
            return {"status": "graphql_failed", "bytes": 0, "elapsed_sec": elapsed, "error": json.dumps(data["errors"])[:800]}
        matrix = data.get("data", {}).get("quantDataMatrix")
        if not matrix:
            return {"status": "empty", "bytes": 0, "elapsed_sec": elapsed, "error": resp.text[:800]}
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(matrix, encoding="utf-8")
        tmp.rename(out_path)
        return {"status": "downloaded", "bytes": out_path.stat().st_size, "elapsed_sec": elapsed, "error": ""}
    except Exception as exc:
        return {"status": "exception", "bytes": 0, "elapsed_sec": time.time() - started, "error": str(exc)[:800]}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--studies", default="", help="Comma-separated PDC IDs. Empty means all selected studies.")
    args = parser.parse_args()

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    studies = pd.read_csv(OVERLAP_PATH, sep="\t")
    studies = studies[
        studies.apply(
            lambda r: should_keep(str(r["submitter_id_name"]), int(r.get("n_pdc_tumor_cases_with_open_gdc_tumor_rna", 0) or 0)),
            axis=1,
        )
    ].copy()
    if args.studies.strip():
        keep = {x.strip() for x in args.studies.split(",") if x.strip()}
        studies = studies[studies["pdc_study_id"].isin(keep)].copy()
    studies = studies.sort_values("n_pdc_tumor_cases_with_open_gdc_tumor_rna", ascending=False)
    if args.limit > 0:
        studies = studies.head(args.limit)

    rows = []
    for _, row in studies.iterrows():
        study_id = str(row["pdc_study_id"])
        out_dir = RAW_ROOT / study_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{study_id}_quantDataMatrix_log2_ratio.tsv"
        result = download_matrix(study_id, out_path, args.timeout)
        rows.append(
            {
                "pdc_study_id": study_id,
                "submitter_id_name": row["submitter_id_name"],
                "n_open_tumor_rna": row.get("n_pdc_tumor_cases_with_open_gdc_tumor_rna"),
                "local_path": str(out_path),
                **result,
            }
        )
        pd.DataFrame(rows).to_csv(TABLE_DIR / "pdc_quant_matrix_download_log.tsv", sep="\t", index=False)
        print(json.dumps(rows[-1], ensure_ascii=False))

    summary = {
        "n_attempted": len(rows),
        "n_downloaded_or_existing": int(sum(r["status"] in {"downloaded", "existing"} for r in rows)),
        "n_failed": int(sum(r["status"] not in {"downloaded", "existing"} for r in rows)),
        "log_path": str(TABLE_DIR / "pdc_quant_matrix_download_log.tsv"),
    }
    (LOG_DIR / "pdc_quant_matrix_download_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
