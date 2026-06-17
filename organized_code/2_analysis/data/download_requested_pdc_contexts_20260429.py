#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


ROOT = Path("/data/lsy/Infinite_Stream")
META = ROOT / "01_data/multi_omics/metadata"
PDC_PHOSPHO_RAW = ROOT / "01_data/multi_omics/raw/cptac_requested_phosphoproteome_reports"
PDC_TOTAL_RAW = ROOT / "01_data/multi_omics/raw/cptac_requested_proteome_reports"
GDC_RAW = ROOT / "01_data/multi_omics/raw/gdc_cptac_open_star_counts"
OUT = ROOT / "02_results/model_validation/20260429_requested_cptac_context_expansion"
PDC = "https://pdc.cancer.gov/graphql"
GDC_DATA = "https://api.gdc.cancer.gov/data/{file_id}"


PHOSPHO_TO_TOTAL = {
    "PDC000121": "PDC000120",
    "PDC000174": "PDC000173",
    "PDC000583": "PDC000582",
    "PDC000117": "PDC000116",
    "PDC000205": "PDC000204",
    "PDC000448": "PDC000446",
    "PDC000515": "PDC000514",
    "PDC000115": "PDC000114",
    "PDC000119": "PDC000118",
}

TRAINABLE_PHOSPHO_STUDIES = [
    "PDC000121",
    "PDC000174",
    "PDC000117",
    "PDC000205",
    "PDC000448",
    "PDC000115",
    "PDC000119",
]


def post_graphql(query: str, variables: dict | None = None) -> dict:
    for attempt in range(5):
        try:
            resp = requests.post(PDC, json={"query": query, "variables": variables or {}}, timeout=120)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("errors"):
                raise RuntimeError(str(payload["errors"])[:1200])
            return payload["data"]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def study_uuid_map() -> dict[str, str]:
    query = """
    query {
      getPaginatedUIStudy(offset: 0, limit: 1000) {
        uiStudies { study_id pdc_study_id submitter_id_name }
      }
    }
    """
    rows = post_graphql(query)["getPaginatedUIStudy"]["uiStudies"]
    return {row["pdc_study_id"]: row["study_id"] for row in rows}


def signed_url(study_uuid: str, file_name: str) -> str:
    query = """
    query FilesDataQuery($file_name: String!, $study_id: String!) {
      uiFilesPerStudy(file_name: $file_name, study_id: $study_id) {
        file_id
        file_name
        signedUrl { url }
      }
    }
    """
    rows = post_graphql(query, {"file_name": file_name, "study_id": study_uuid}).get("uiFilesPerStudy") or []
    if not rows:
        raise RuntimeError("no uiFilesPerStudy row")
    url = ((rows[0].get("signedUrl") or {}).get("url") or "").strip()
    if not url:
        raise RuntimeError("empty signedUrl")
    return url


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_file_table(study: str) -> pd.DataFrame:
    path = META / f"pdc_{study}_files_per_study.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def select_phospho_files(study: str) -> pd.DataFrame:
    df = read_file_table(study)
    name = df["file_name"].fillna("").astype(str)
    low = name.str.lower()
    cat = df["data_category"].fillna("").astype(str).str.lower()
    keep = cat.eq("protein assembly")
    keep &= low.str.endswith((".tsv", ".txt", ".xlsx", ".tar.gz", ".zip"))
    keep &= (
        low.str.contains("phosphosite")
        | low.str.contains("phosphoproteome")
        | low.str.contains("sample")
        | low.str.contains("label")
        | low.str.contains("clinical")
        | low.str.contains("reporterion")
    )
    keep &= ~low.str.contains("phosphopeptide|peptides|\\.raw|mzml|mzid|psm", regex=True)
    out = df.loc[keep].copy()
    out["omic_layer"] = "phosphoproteome"
    out["study_role"] = "phospho"
    return out


def select_total_files(study: str) -> pd.DataFrame:
    df = read_file_table(study)
    name = df["file_name"].fillna("").astype(str)
    low = name.str.lower()
    cat = df["data_category"].fillna("").astype(str).str.lower()
    keep = cat.eq("protein assembly")
    keep &= low.str.endswith((".tsv", ".txt", ".xlsx", ".tar.gz", ".zip"))
    keep &= low.str.contains("protein|proteome|tmt|itraq|sample|label|clinical|reporterion", regex=True)
    keep &= ~low.str.contains("phospho|glyco|ubiquit|acetyl|qcmetrics|summary|protocol|peptides|\\.raw|mzml|mzid|psm", regex=True)
    out = df.loc[keep].copy()
    out["omic_layer"] = "total_proteome"
    out["study_role"] = "total"
    return out


def download_pdc_one(row: dict, uuid_map: dict[str, str], overwrite: bool) -> dict:
    study = row["pdc_study_id"]
    role = row["study_role"]
    fname = row["file_name"]
    root = PDC_PHOSPHO_RAW if role == "phospho" else PDC_TOTAL_RAW
    dest = root / study / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(row.get("file_size") or 0)
    expected_md5 = str(row.get("md5sum") or "").strip()
    log = {
        "pdc_study_id": study,
        "study_role": role,
        "file_name": fname,
        "file_size": expected_size,
        "expected_md5": expected_md5,
        "path": str(dest),
        "status": "",
        "message": "",
    }
    if dest.exists() and not overwrite:
        size_ok = expected_size <= 0 or dest.stat().st_size == expected_size
        md5_ok = True
        if expected_md5 and size_ok:
            md5_ok = md5sum(dest) == expected_md5
        if size_ok and md5_ok:
            log["status"] = "exists"
            return log
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        uuid = uuid_map.get(study)
        if not uuid:
            raise RuntimeError("missing study uuid")
        url = signed_url(uuid, fname)
        with requests.get(url, stream=True, timeout=(30, 900)) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(1024 * 1024):
                    if chunk:
                        fh.write(chunk)
        if expected_size and tmp.stat().st_size != expected_size:
            raise RuntimeError(f"size mismatch {tmp.stat().st_size} != {expected_size}")
        if expected_md5:
            got = md5sum(tmp)
            if got != expected_md5:
                raise RuntimeError(f"md5 mismatch {got} != {expected_md5}")
        os.replace(tmp, dest)
        log["status"] = "downloaded"
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        log["status"] = "failed"
        log["message"] = str(exc)[:1000]
    return log


def download_gdc_one(row: dict, study: str) -> dict:
    study_dir = GDC_RAW / study
    study_dir.mkdir(parents=True, exist_ok=True)
    dest = study_dir / str(row["file_name"])
    log = {
        "pdc_study_id": study,
        "case_submitter_id": row.get("case_submitter_id"),
        "file_id": row.get("file_id"),
        "file_name": row.get("file_name"),
        "path": str(dest),
        "status": "",
        "bytes": 0,
    }
    if dest.exists() and dest.stat().st_size > 0:
        log["status"] = "exists"
        log["bytes"] = dest.stat().st_size
        return log
    tmp = dest.with_suffix(dest.suffix + ".tmp")
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
        GDC_DATA.format(file_id=row["file_id"]),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        log["status"] = "failed"
        log["message"] = proc.stderr[-1000:]
        return log
    os.replace(tmp, dest)
    log["status"] = "downloaded"
    log["bytes"] = dest.stat().st_size
    return log


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-gdc", action="store_true")
    args = parser.parse_args()

    for path in [PDC_PHOSPHO_RAW, PDC_TOTAL_RAW, GDC_RAW, OUT / "tables", OUT / "logs"]:
        path.mkdir(parents=True, exist_ok=True)

    selected = []
    for phospho, total in PHOSPHO_TO_TOTAL.items():
        p = select_phospho_files(phospho)
        p["paired_total_study_id"] = total
        selected.append(p)
        t = select_total_files(total)
        t["paired_phospho_study_id"] = phospho
        selected.append(t)
    selected_df = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    selected_df.to_csv(OUT / "tables/requested_pdc_report_file_candidates.tsv", sep="\t", index=False)

    uuid_map = study_uuid_map()
    logs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_pdc_one, row, uuid_map, args.overwrite) for row in selected_df.to_dict("records")]
        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            logs.append(rec)
            print(f"PDC {i}/{len(futures)} {rec['pdc_study_id']} {rec['file_name']} {rec['status']}")
    pd.DataFrame(logs).to_csv(OUT / "tables/requested_pdc_report_download_log.tsv", sep="\t", index=False)

    gdc_logs = []
    if not args.skip_gdc:
        tasks = []
        for study in TRAINABLE_PHOSPHO_STUDIES:
            manifest = META / f"gdc_open_star_counts_for_{study}.tsv"
            if not manifest.exists():
                continue
            gdc = pd.read_csv(manifest, sep="\t")
            if gdc.empty:
                continue
            tumor = gdc[
                gdc["tissue_type_gdc"].fillna("").eq("Tumor")
                | gdc["sample_type_gdc"].fillna("").str.contains("Tumor", case=False, regex=False)
            ].copy()
            for row in tumor.to_dict("records"):
                tasks.append((study, row))
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(download_gdc_one, row, study) for study, row in tasks]
            for i, fut in enumerate(as_completed(futures), start=1):
                rec = fut.result()
                gdc_logs.append(rec)
                if i % 25 == 0 or i == len(futures):
                    print(f"GDC {i}/{len(futures)}")
        pd.DataFrame(gdc_logs).to_csv(OUT / "tables/requested_gdc_star_counts_download_log.tsv", sep="\t", index=False)

    summary = {
        "n_selected_pdc_files": int(selected_df.shape[0]),
        "n_pdc_downloaded": int(sum(1 for r in logs if r["status"] == "downloaded")),
        "n_pdc_exists": int(sum(1 for r in logs if r["status"] == "exists")),
        "n_pdc_failed": int(sum(1 for r in logs if r["status"] == "failed")),
        "n_gdc_files": int(len(gdc_logs)),
        "n_gdc_downloaded": int(sum(1 for r in gdc_logs if r["status"] == "downloaded")),
        "n_gdc_exists": int(sum(1 for r in gdc_logs if r["status"] == "exists")),
        "n_gdc_failed": int(sum(1 for r in gdc_logs if r["status"] == "failed")),
    }
    (OUT / "logs/download_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["n_pdc_failed"] == 0 and summary["n_gdc_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
