#!/usr/bin/env python3
"""Download companion CPTAC global proteome report files from PDC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


ROOT = Path("/data/lsy/Infinite_Stream")
META_DIR = ROOT / "01_data" / "multi_omics" / "metadata"
OUT_DIR = ROOT / "01_data" / "multi_omics" / "raw" / "cptac_pancancer_proteome_reports"
RESULT_DIR = ROOT / "02_results" / "model_validation" / "20260426_cptac_pancancer_proteome_download"
GRAPHQL_URL = "https://pdc.cancer.gov/graphql"


COMPANION_PROTEOME = {
    "PDC000615": "PDC000614",
    "PDC000271": "PDC000270",
    "PDC000441": "PDC000439",
    "PDC000490": "PDC000489",
    "PDC000128": "PDC000127",
    "PDC000222": "PDC000221",
    "PDC000126": "PDC000125",
    "PDC000149": "PDC000153",
    "PDC000232": "PDC000234",
    "PDC000465": "PDC000464",
}


QUERY_FILES = """
query Files($study:String!) {
  filesPerStudy(pdc_study_id: $study) {
    pdc_study_id
    study_name
    file_id
    file_name
    file_type
    file_size
    data_category
    file_format
    file_location
    md5sum
  }
}
"""


def post_graphql(query: str, variables: dict | None = None, timeout: int = 120) -> dict:
    for attempt in range(1, 6):
        try:
            r = requests.post(GRAPHQL_URL, json={"query": query, "variables": variables or {}}, timeout=timeout)
            r.raise_for_status()
            payload = r.json()
            if payload.get("errors"):
                raise RuntimeError(str(payload["errors"])[:1200])
            return payload["data"]
        except Exception:
            if attempt == 5:
                raise
            time.sleep(3 * attempt)
    raise RuntimeError("unreachable")


def study_uuid_map() -> dict[str, str]:
    q = """
    query {
      getPaginatedUIStudy(offset: 0, limit: 1000) {
        uiStudies { study_id pdc_study_id submitter_id_name }
      }
    }
    """
    rows = post_graphql(q)["getPaginatedUIStudy"]["uiStudies"]
    return {r["pdc_study_id"]: r["study_id"] for r in rows}


def signed_url(study_uuid: str, file_name: str) -> str:
    q = """
    query FilesDataQuery($file_name: String!, $study_id: String!) {
      uiFilesPerStudy(file_name: $file_name, study_id: $study_id) {
        file_id
        file_name
        signedUrl { url }
      }
    }
    """
    rows = post_graphql(q, {"file_name": file_name, "study_id": study_uuid}).get("uiFilesPerStudy") or []
    if not rows:
        raise RuntimeError("no uiFilesPerStudy row")
    url = ((rows[0].get("signedUrl") or {}).get("url") or "").strip()
    if not url:
        raise RuntimeError("empty signedUrl")
    return url


def collect_files(study: str) -> pd.DataFrame:
    path = META_DIR / f"pdc_{study}_files_per_study.tsv"
    if path.exists():
        return pd.read_csv(path, sep="\t")
    df = pd.DataFrame(post_graphql(QUERY_FILES, {"study": study}).get("filesPerStudy") or [])
    df.to_csv(path, sep="\t", index=False)
    return df


def select_files(studies: list[str]) -> pd.DataFrame:
    rows = []
    for phospho, proteome in [(s, COMPANION_PROTEOME[s]) for s in studies]:
        df = collect_files(proteome)
        if df.empty:
            rows.append({"phospho_study_id": phospho, "proteome_study_id": proteome, "status": "no_files"})
            continue
        name = df["file_name"].fillna("").astype(str)
        cat = df["data_category"].fillna("").astype(str).str.lower()
        low = name.str.lower()
        keep = cat.eq("protein assembly") & low.str.endswith(".tsv")
        keep &= ~low.str.contains("phospho|glyco|ubiquit|acetyl|qcmetrics|summary|label|metadata|protocol")
        keep &= low.str.contains("protein|proteome|log2|tmt|itraq|report|global")
        sub = df.loc[keep].copy()
        if sub.empty:
            rows.append({"phospho_study_id": phospho, "proteome_study_id": proteome, "status": "no_matrix_candidate"})
        else:
            sub["phospho_study_id"] = phospho
            sub["proteome_study_id"] = proteome
            sub["status"] = "selected"
            rows.extend(sub.to_dict("records"))
    return pd.DataFrame(rows)


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_one(row: dict, uuid: str, overwrite: bool = False) -> dict:
    proteome = row["proteome_study_id"]
    fname = row["file_name"]
    dest = OUT_DIR / proteome / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    expected_size = int(row.get("file_size") or 0)
    expected_md5 = str(row.get("md5sum") or "").strip()
    log = {
        "phospho_study_id": row.get("phospho_study_id", ""),
        "proteome_study_id": proteome,
        "file_name": fname,
        "file_size": expected_size,
        "expected_md5": expected_md5,
        "path": str(dest),
        "status": "",
        "message": "",
    }
    if dest.exists() and not overwrite:
        if expected_size <= 0 or dest.stat().st_size == expected_size:
            log["status"] = "exists"
            return log
    try:
        url = signed_url(uuid, fname)
        with requests.get(url, stream=True, timeout=(30, 900)) as r:
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(1024 * 1024):
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", nargs="+", default=list(COMPANION_PROTEOME))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for d in [OUT_DIR, RESULT_DIR / "tables", RESULT_DIR / "logs"]:
        d.mkdir(parents=True, exist_ok=True)
    selected = select_files(args.studies)
    selected.to_csv(RESULT_DIR / "tables" / "proteome_report_file_candidates.tsv", sep="\t", index=False)
    todo = selected[selected.get("status").eq("selected")].copy()
    if todo.empty:
        print("no selected files")
        return 1
    uuid_map = study_uuid_map()
    logs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for row in todo.to_dict("records"):
            uuid = uuid_map.get(row["proteome_study_id"])
            if not uuid:
                row["status"] = "failed"
                row["message"] = "missing study uuid"
                logs.append(row)
                continue
            futures.append(pool.submit(download_one, row, uuid, args.overwrite))
        for fut in as_completed(futures):
            rec = fut.result()
            logs.append(rec)
            print(rec["proteome_study_id"], rec["file_name"], rec["status"], rec["message"])
    pd.DataFrame(logs).to_csv(RESULT_DIR / "tables" / "proteome_report_download_log.tsv", sep="\t", index=False)
    (RESULT_DIR / "logs" / "companion_proteome_map.json").write_text(json.dumps(COMPANION_PROTEOME, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
