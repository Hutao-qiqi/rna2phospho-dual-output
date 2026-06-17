#!/usr/bin/env python3
"""Download CPTAC phosphoproteome report files from PDC signed URLs.

This uses the same GraphQL route as the PDC web UI:
uiFilesPerStudy(file_name, study_id) -> signedUrl.url.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


ROOT = Path("/data/lsy/Infinite_Stream")
METADATA_DIR = ROOT / "01_data" / "multi_omics" / "metadata"
OUT_DIR = (
    ROOT
    / "01_data"
    / "multi_omics"
    / "raw"
    / "cptac_pancancer_phosphoproteome_reports"
)
RESULT_DIR = (
    ROOT
    / "02_results"
    / "model_validation"
    / "20260426_pdc_pancancer_phosphoproteome_feasibility"
)
LOG_PATH = RESULT_DIR / "tables" / "pdc_report_file_download_log.tsv"
GRAPHQL_URL = "https://pdc.cancer.gov/graphql"


DEFAULT_SOLID_STUDIES = [
    "PDC000615",  # STAD
    "PDC000271",  # PDA discovery
    "PDC000441",  # UCEC confirmatory
    "PDC000490",  # LUAD confirmatory
    "PDC000128",  # CCRCC discovery
    "PDC000222",  # HNSCC discovery
    "PDC000126",  # UCEC discovery
    "PDC000149",  # LUAD discovery
    "PDC000232",  # LSCC discovery
    "PDC000412",  # CCRCC confirmatory DIA, may not have report files
    "PDC000465",  # non-ccRCC
]


def post_graphql(query: str, variables: dict | None = None, timeout: int = 120) -> dict:
    for attempt in range(1, 6):
        try:
            r = requests.post(
                GRAPHQL_URL,
                json={"query": query, "variables": variables or {}},
                timeout=timeout,
            )
            r.raise_for_status()
            payload = r.json()
            if payload.get("errors"):
                raise RuntimeError(str(payload["errors"])[:1000])
            return payload["data"]
        except Exception:
            if attempt == 5:
                raise
            time.sleep(3 * attempt)
    raise RuntimeError("unreachable")


def get_study_uuid_map() -> dict[str, str]:
    query = """
    query {
      getPaginatedUIStudy(offset: 0, limit: 1000) {
        uiStudies { study_id pdc_study_id submitter_id_name }
      }
    }
    """
    data = post_graphql(query)
    studies = data["getPaginatedUIStudy"]["uiStudies"]
    return {row["pdc_study_id"]: row["study_id"] for row in studies}


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
    data = post_graphql(query, {"file_name": file_name, "study_id": study_uuid})
    rows = data.get("uiFilesPerStudy") or []
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


def download_one(row: dict, study_uuid: str, overwrite: bool = False) -> dict:
    study = row["pdc_study_id"]
    fname = row["file_name"]
    expected_size = int(row.get("file_size") or 0)
    expected_md5 = str(row.get("md5sum") or "").strip()
    dest_dir = OUT_DIR / study
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / fname
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    log = {
        "pdc_study_id": study,
        "file_name": fname,
        "file_size": expected_size,
        "expected_md5": expected_md5,
        "path": str(dest),
        "status": "unknown",
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

    try:
        url = signed_url(study_uuid, fname)
        with requests.get(url, stream=True, timeout=(30, 600)) as r:
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
        if expected_size and tmp.stat().st_size != expected_size:
            raise RuntimeError(f"size mismatch: got {tmp.stat().st_size}, expected {expected_size}")
        if expected_md5:
            got = md5sum(tmp)
            if got != expected_md5:
                raise RuntimeError(f"md5 mismatch: got {got}, expected {expected_md5}")
        os.replace(tmp, dest)
        log["status"] = "downloaded"
    except Exception as exc:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        log["status"] = "failed"
        log["message"] = str(exc)[:1000]
    return log


def select_report_files(studies: list[str]) -> pd.DataFrame:
    rows = []
    for study in studies:
        path = METADATA_DIR / f"pdc_{study}_files_per_study.tsv"
        if not path.exists():
            rows.append({"pdc_study_id": study, "file_name": "", "status": "missing_metadata"})
            continue
        df = pd.read_csv(path, sep="\t")
        name = df["file_name"].astype(str)
        cat = df["data_category"].astype(str).str.lower()
        keep = cat.eq("protein assembly") & (
            name.str.contains("phosphosite", case=False, regex=False)
            | name.str.contains("summary", case=False, regex=False)
            | name.str.contains("label", case=False, regex=False)
        )
        sub = df.loc[keep].copy()
        if sub.empty:
            rows.append({"pdc_study_id": study, "file_name": "", "status": "no_report_file"})
        else:
            sub["status"] = "selected"
            rows.extend(sub.to_dict("records"))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", nargs="+", default=DEFAULT_SOLID_STUDIES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    candidates = select_report_files(args.studies)
    candidates.to_csv(
        RESULT_DIR / "tables" / "pdc_report_file_candidates.tsv",
        sep="\t",
        index=False,
    )

    selected = candidates[candidates.get("status").eq("selected")].copy()
    if selected.empty:
        candidates.to_csv(LOG_PATH, sep="\t", index=False)
        print("No report files selected", file=sys.stderr)
        return 1

    uuid_map = get_study_uuid_map()
    jobs = []
    logs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in selected.to_dict("records"):
            study = row["pdc_study_id"]
            study_uuid = uuid_map.get(study)
            if not study_uuid:
                row["status"] = "failed"
                row["message"] = "study uuid not found"
                logs.append(row)
                continue
            jobs.append(pool.submit(download_one, row, study_uuid, args.overwrite))
        for fut in as_completed(jobs):
            rec = fut.result()
            logs.append(rec)
            print(rec["pdc_study_id"], rec["file_name"], rec["status"], rec.get("message", ""))

    extra = candidates[~candidates.get("status").eq("selected")].copy()
    if not extra.empty:
        logs.extend(extra.to_dict("records"))
    pd.DataFrame(logs).to_csv(LOG_PATH, sep="\t", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
