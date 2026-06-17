import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests


DEFAULT_ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
DEFAULT_OUT = DEFAULT_ROOT / r"01_data\single_cell\raw\decryptm_pxd037285_v1"
PRIDE_FILES_API = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects/PXD037285/files"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def https_location(file_record):
    for loc in file_record.get("publicFileLocations", []):
        value = str(loc.get("value", ""))
        if value.startswith("ftp://ftp.pride.ebi.ac.uk/"):
            return value.replace("ftp://ftp.pride.ebi.ac.uk/", "https://ftp.pride.ebi.ac.uk/", 1)
        if value.startswith("https://"):
            return value
    return f"https://ftp.pride.ebi.ac.uk/pride/data/archive/2023/03/PXD037285/{file_record['fileName']}"


def list_pride_files():
    records = []
    page = 0
    while True:
        response = requests.get(PRIDE_FILES_API, params={"page": page, "pageSize": 100}, timeout=120)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        records.extend(batch)
        page += 1
    return records


def wanted_record(record, include_search_zips=False):
    name = str(record.get("fileName", ""))
    if name == "Experiment_summary.zip":
        return True
    if name.endswith("_Curves.zip"):
        return True
    if include_search_zips and name.endswith(".zip") and "Phosphoproteome" in name:
        return True
    return False


def download_one(url, out_path, expected_size, retries=4):
    out_path = Path(out_path)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if out_path.exists() and expected_size and out_path.stat().st_size == expected_size:
        return "present"
    if out_path.exists() and not expected_size:
        return "present"
    if out_path.exists() and expected_size and out_path.stat().st_size != expected_size:
        out_path.replace(tmp_path)

    for attempt in range(1, retries + 1):
        mode = "ab" if tmp_path.exists() else "wb"
        headers = {}
        start = tmp_path.stat().st_size if tmp_path.exists() else 0
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            with requests.get(url, stream=True, headers=headers, timeout=180) as response:
                if response.status_code == 416 and expected_size and start == expected_size:
                    tmp_path.replace(out_path)
                    return "downloaded"
                response.raise_for_status()
                if start and response.status_code == 200:
                    mode = "wb"
                with tmp_path.open(mode + "") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if expected_size and tmp_path.stat().st_size != expected_size:
                raise RuntimeError(f"size mismatch {tmp_path.stat().st_size} != {expected_size}")
            tmp_path.replace(out_path)
            return "downloaded"
        except BaseException:
            if attempt == retries:
                raise
            time.sleep(10 * attempt)
    return "failed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--include-search-zips", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    records = [r for r in list_pride_files() if wanted_record(r, args.include_search_zips)]
    rows = []
    for record in sorted(records, key=lambda x: str(x.get("fileName", ""))):
        url = https_location(record)
        size = int(record.get("fileSizeBytes") or 0)
        rows.append(
            {
                "file_name": record.get("fileName"),
                "file_category": (record.get("fileCategory") or {}).get("value"),
                "file_size_bytes": size,
                "url": url,
                "local_path": str(out_dir / record.get("fileName")),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "decryptm_pxd037285_download_manifest.tsv", sep="\t", index=False)
    with (out_dir / "decryptm_pxd037285_download_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    if args.manifest_only:
        print(f"wrote manifest files={len(rows)} out={out_dir}", flush=True)
        return

    done_rows = []
    for row in rows:
        status = download_one(row["url"], out_dir / row["file_name"], int(row["file_size_bytes"]))
        row = dict(row)
        row["download_status"] = status
        row["downloaded_size_bytes"] = Path(row["local_path"]).stat().st_size if Path(row["local_path"]).exists() else 0
        done_rows.append(row)
        print(f"{status}\t{row['file_name']}\t{row['downloaded_size_bytes']}", flush=True)
    pd.DataFrame(done_rows).to_csv(out_dir / "decryptm_pxd037285_download_status.tsv", sep="\t", index=False)
    print(f"done files={len(done_rows)} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
