import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
OUT_ROOT = ROOT / r"01_data\single_cell\raw\external_bulk_phospho_validation_v1"
LOG_DIR = OUT_ROOT / "_logs"
MANIFEST = OUT_ROOT / "external_bulk_file_manifest.tsv"
DOWNLOAD_STATUS = OUT_ROOT / "external_bulk_download_status.tsv"
PROJECT_TABLE = OUT_ROOT / "external_bulk_project_table.tsv"


PRIDE_ACCESSIONS = [
    "PXD008032",
    "PXD021607",
    "PXD021608",
    "PXD021609",
    "PXD021611",
    "PXD058009",
    "PXD001440",
]


MASSIVE_PROJECTS = {
    "PXD063604": {
        "title": "Illuminating oncogenic KRAS signaling by multi-dimensional proteomics",
        "repository": "MassIVE",
        "massive_id": "MSV000097797",
        "dataset_url": "https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD063604",
        "ftp_url": "ftp://massive-ftp.ucsd.edu/v09/MSV000097797/",
        "note": "MassIVE-hosted ProteomeXchange dataset; PRIDE project endpoint returns 404.",
    }
}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(message):
    ensure_dir(LOG_DIR)
    with (LOG_DIR / "download_external_bulk.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {message}\n")
    print(f"[{now()}] {message}", flush=True)


def safe_name(text):
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in str(text))


def fetch_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "SCP682-PPKO-1"})
    with urllib.request.urlopen(req, timeout=90) as handle:
        return json.load(handle)


def public_url(file_row):
    locations = file_row.get("publicFileLocations") or []
    for loc in locations:
        value = loc.get("value", "")
        if value.startswith("https://"):
            return value
    for loc in locations:
        value = loc.get("value", "")
        if value.startswith("ftp://ftp.pride.ebi.ac.uk/"):
            return value.replace("ftp://ftp.pride.ebi.ac.uk/", "https://ftp.pride.ebi.ac.uk/")
    for loc in locations:
        value = loc.get("value", "")
        if value.startswith("ftp://"):
            return value
    return ""


def category_value(file_row):
    return (file_row.get("fileCategory") or {}).get("value", "")


def should_download(accession, file_row):
    name = str(file_row.get("fileName", ""))
    low = name.lower()
    cat = category_value(file_row)
    size = int(file_row.get("fileSizeBytes") or 0)
    if accession == "PXD008032":
        return cat in {"OTHER", "SEARCH"} and size < 250_000_000
    if accession == "PXD001440":
        return name in {"txt_phos.zip", "txtbcrIP.zip", "txt_gg.zip"} and size < 800_000_000
    if accession == "PXD058009":
        return cat == "OTHER" and size < 5_000_000
    if accession in {"PXD021607", "PXD021608", "PXD021609", "PXD021611"}:
        return False
    return cat not in {"RAW", "PEAK"} and any(x in low for x in ["phospho", "site", "quant", "ratio", "txt", "xlsx", "csv", "tsv"])


def write_table(path, rows, cols):
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(cols) + "\n")
        for row in rows:
            values = [str(row.get(c, "")) for c in cols]
            handle.write("\t".join(v.replace("\t", " ").replace("\n", " ") for v in values) + "\n")


def collect_pride():
    ensure_dir(OUT_ROOT)
    project_rows = []
    file_rows = []
    for accession in PRIDE_ACCESSIONS:
        project_dir = ensure_dir(OUT_ROOT / accession)
        try:
            project = fetch_json(f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{accession}")
            (project_dir / f"{accession}_project.json").write_text(json.dumps(project, indent=2), encoding="utf-8")
            project_rows.append(
                {
                    "accession": accession,
                    "repository": "PRIDE",
                    "title": project.get("title", ""),
                    "publication_date": project.get("publicationDate", ""),
                    "submission_date": project.get("submissionDate", ""),
                    "status": "metadata_ok",
                    "url": f"https://www.ebi.ac.uk/pride/archive/projects/{accession}",
                    "note": "",
                }
            )
            files = fetch_json(f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{accession}/files")
            (project_dir / f"{accession}_files.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
            for item in files:
                name = str(item.get("fileName", ""))
                url = public_url(item)
                download = should_download(accession, item)
                file_rows.append(
                    {
                        "accession": accession,
                        "repository": "PRIDE",
                        "file_name": name,
                        "category": category_value(item),
                        "file_size_bytes": int(item.get("fileSizeBytes") or 0),
                        "download": "yes" if download else "no",
                        "url": url,
                        "rel_dir": accession,
                        "note": "selected_processed_file" if download else "",
                    }
                )
            log(f"metadata ok: {accession} files={len(files)} selected={sum(1 for r in file_rows if r['accession']==accession and r['download']=='yes')}")
        except Exception as exc:
            project_rows.append(
                {
                    "accession": accession,
                    "repository": "PRIDE",
                    "title": "",
                    "publication_date": "",
                    "submission_date": "",
                    "status": "metadata_failed",
                    "url": f"https://www.ebi.ac.uk/pride/archive/projects/{accession}",
                    "note": str(exc),
                }
            )
            log(f"metadata failed: {accession} {exc}")
    for accession, item in MASSIVE_PROJECTS.items():
        project_dir = ensure_dir(OUT_ROOT / accession)
        (project_dir / f"{accession}_project.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        project_rows.append(
            {
                "accession": accession,
                "repository": item["repository"],
                "title": item["title"],
                "publication_date": "",
                "submission_date": "",
                "status": "metadata_ok",
                "url": item["dataset_url"],
                "note": item["note"],
            }
        )
        file_rows.append(
            {
                "accession": accession,
                "repository": item["repository"],
                "file_name": "",
                "category": "",
                "file_size_bytes": "",
                "download": "no",
                "url": item["ftp_url"],
                "rel_dir": accession,
                "note": item["note"],
            }
        )
    write_table(PROJECT_TABLE, project_rows, ["accession", "repository", "title", "publication_date", "submission_date", "status", "url", "note"])
    write_table(MANIFEST, file_rows, ["accession", "repository", "file_name", "category", "file_size_bytes", "download", "url", "rel_dir", "note"])
    return file_rows


def download_one(row):
    out_dir = ensure_dir(OUT_ROOT / row["rel_dir"])
    out_path = out_dir / row["file_name"]
    if out_path.exists() and out_path.stat().st_size > 0:
        status = "exists"
        code = 0
    else:
        url = row["url"]
        aria2 = shutil.which("aria2c.exe") or shutil.which("aria2c")
        if aria2:
            cmd = [
                aria2,
                "--continue=true",
                "--max-connection-per-server=8",
                "--split=8",
                "--min-split-size=1M",
                "--max-tries=8",
                "--retry-wait=10",
                "--connect-timeout=60",
                "--timeout=60",
                "--allow-overwrite=true",
                "--auto-file-renaming=false",
                "--dir",
                str(out_dir),
                "--out",
                row["file_name"],
                url,
            ]
        else:
            cmd = [
                "curl.exe",
                "-L",
                "-C",
                "-",
                "--retry",
                "8",
                "--retry-delay",
                "10",
                "--connect-timeout",
                "60",
                "--output",
                str(out_path),
                url,
            ]
        log(f"download start: {row['accession']} {row['file_name']}")
        with (LOG_DIR / f"{safe_name(row['accession'])}__{safe_name(row['file_name'])}.log").open("ab") as handle:
            proc = subprocess.run(cmd, stdout=handle, stderr=handle)
        code = proc.returncode
        status = "done" if code == 0 and out_path.exists() and out_path.stat().st_size > 0 else "failed"
    size = out_path.stat().st_size if out_path.exists() else 0
    out = dict(row)
    out.update({"status": status, "exit_code": code, "bytes": size, "path": str(out_path), "time": now()})
    log(f"download {status}: {row['accession']} {row['file_name']} bytes={size} code={code}")
    return out


def main():
    ensure_dir(OUT_ROOT)
    ensure_dir(LOG_DIR)
    rows = collect_pride()
    selected = [r for r in rows if r.get("download") == "yes" and r.get("url")]
    completed = []
    for row in selected:
        try:
            completed.append(download_one(row))
        except Exception as exc:
            bad = dict(row)
            bad.update({"status": "failed_exception", "exit_code": "", "bytes": 0, "path": "", "time": now(), "note": f"{row.get('note','')} {exc}"})
            completed.append(bad)
            log(f"download exception: {row['accession']} {row['file_name']} {exc}")
        write_table(
            DOWNLOAD_STATUS,
            completed,
            ["accession", "repository", "file_name", "category", "file_size_bytes", "download", "url", "rel_dir", "note", "status", "exit_code", "bytes", "path", "time"],
        )
    (LOG_DIR / "done.txt").write_text(now() + "\n", encoding="utf-8")
    log("external bulk metadata and selected downloads complete")


if __name__ == "__main__":
    main()
