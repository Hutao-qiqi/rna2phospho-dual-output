#!/usr/bin/env python3
"""Download open GDC STAR count files for selected CPTAC phosphoproteome studies."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd


PROJECT_ROOT = Path("/data/lsy/Infinite_Stream")
OVERLAP_PATH = PROJECT_ROOT / "02_results/model_validation/20260426_pdc_pancancer_phosphoproteome_feasibility/tables/pdc_cptac_phosphoproteome_studies_with_gdc_open_rna_overlap.tsv"
META_DIR = PROJECT_ROOT / "01_data/multi_omics/metadata"
RAW_ROOT = PROJECT_ROOT / "01_data/multi_omics/raw/gdc_cptac_open_star_counts"
OUT_DIR = PROJECT_ROOT / "02_results/model_validation/20260426_pdc_pancancer_phosphoproteome_feasibility"
LOG_DIR = OUT_DIR / "logs"
TABLE_DIR = OUT_DIR / "tables"

GDC_DATA_URL = "https://api.gdc.cancer.gov/data/{file_id}"


EXCLUDE_KEYWORDS = ["CompRef", "Kinase Inhibition", "Pediatric", "T-ALL"]


def should_keep(name: str, tumor_rna_n: int) -> bool:
    if tumor_rna_n < 20:
        return False
    return not any(k.lower() in name.lower() for k in EXCLUDE_KEYWORDS)


def download_file(file_id: str, file_name: str, out_path: Path) -> str:
    if out_path.exists() and out_path.stat().st_size > 0:
        return "existing"
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    cmd = [
        "curl",
        "--fail",
        "--location",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "--output",
        str(tmp),
        GDC_DATA_URL.format(file_id=file_id),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        return "failed"
    tmp.rename(out_path)
    return "downloaded"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
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

    tasks = []
    for _, study in studies.iterrows():
        study_id = str(study["pdc_study_id"])
        manifest = META_DIR / f"gdc_open_star_counts_for_{study_id}.tsv"
        if not manifest.exists():
            rows.append({"pdc_study_id": study_id, "status": "missing_manifest"})
            continue
        gdc = pd.read_csv(manifest, sep="\t")
        tumor = gdc[
            gdc["tissue_type_gdc"].eq("Tumor")
            | gdc["sample_type_gdc"].fillna("").str.contains("Tumor", case=False, regex=False)
        ].copy()
        study_dir = RAW_ROOT / study_id
        study_dir.mkdir(parents=True, exist_ok=True)
        for rec in tumor.to_dict(orient="records"):
            out_path = study_dir / str(rec["file_name"])
            tasks.append((study_id, study["submitter_id_name"], rec, out_path))

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_task = {
            pool.submit(download_file, str(rec["file_id"]), str(rec["file_name"]), out_path): (
                study_id,
                submitter_id_name,
                rec,
                out_path,
            )
            for study_id, submitter_id_name, rec, out_path in tasks
        }
        for i, future in enumerate(as_completed(future_to_task), start=1):
            study_id, submitter_id_name, rec, out_path = future_to_task[future]
            status = future.result()
            rows.append(
                {
                    "pdc_study_id": study_id,
                    "submitter_id_name": submitter_id_name,
                    "case_submitter_id": rec.get("case_submitter_id"),
                    "sample_submitter_id_gdc": rec.get("sample_submitter_id_gdc"),
                    "file_id": rec.get("file_id"),
                    "file_name": rec.get("file_name"),
                    "local_path": str(out_path),
                    "status": status,
                    "bytes": out_path.stat().st_size if out_path.exists() else 0,
                }
            )
            if i % 25 == 0:
                pd.DataFrame(rows).to_csv(TABLE_DIR / "gdc_open_star_counts_download_log.tsv", sep="\t", index=False)
                print(f"completed {i}/{len(tasks)}")

    log = pd.DataFrame(rows)
    log_path = TABLE_DIR / "gdc_open_star_counts_download_log.tsv"
    log.to_csv(log_path, sep="\t", index=False)
    summary = {
        "n_rows": int(log.shape[0]),
        "n_downloaded": int((log["status"] == "downloaded").sum()) if not log.empty else 0,
        "n_existing": int((log["status"] == "existing").sum()) if not log.empty else 0,
        "n_failed": int((log["status"] == "failed").sum()) if not log.empty else 0,
        "n_studies": int(log["pdc_study_id"].nunique()) if not log.empty else 0,
        "log_path": str(log_path),
    }
    (LOG_DIR / "gdc_open_star_counts_download_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
