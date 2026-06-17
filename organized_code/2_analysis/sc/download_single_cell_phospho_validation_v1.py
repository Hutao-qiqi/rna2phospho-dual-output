import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
OUT_ROOT = ROOT / r"01_data\single_cell\raw\external_single_cell_phospho_validation_v1"
LOG_DIR = OUT_ROOT / "_logs"
MANIFEST = OUT_ROOT / "download_manifest.tsv"
STATUS = OUT_ROOT / "download_status.tsv"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "download.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {message}\n")
    print(f"[{now()}] {message}", flush=True)


def safe_name(text):
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in text)


def add_download(rows, dataset, label, url, rel_dir, filename=None, note=""):
    if filename is None:
        filename = url.split("?")[0].rstrip("/").split("/")[-1] or safe_name(label)
    rows.append(
        {
            "dataset": dataset,
            "label": label,
            "url": url,
            "rel_dir": rel_dir,
            "filename": filename,
            "note": note,
        }
    )


def zenodo_files(record_id):
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url, timeout=60) as handle:
        data = json.load(handle)
    rows = []
    for item in data.get("files", []):
        key = item.get("key") or item.get("filename")
        link = item.get("links", {}).get("self") or item.get("links", {}).get("download")
        if key and link:
            rows.append((key, link, item.get("size")))
    return data, rows


def build_manifest():
    rows = []
    geo_sets = [
        ("icCITE-plex_GSE300551", "GSE300551", "GSE300nnn", "PRJNA1346700"),
        ("Vivo-seq_Th17_GSE297075", "GSE297075", "GSE297nnn", None),
        ("Phospho-seq_Blair_GSE285561", "GSE285561", "GSE285nnn", None),
    ]
    for dataset, gse, series, bioproject in geo_sets:
        base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{series}/{gse}"
        add_download(rows, dataset, "GEO family SOFT", f"{base}/soft/{gse}_family.soft.gz", dataset)
        add_download(rows, dataset, "GEO family MINiML", f"{base}/miniml/{gse}_family.xml.tgz", dataset)
        add_download(rows, dataset, "GEO supplementary RAW tar", f"{base}/suppl/{gse}_RAW.tar", dataset)
        add_download(rows, dataset, "GEO page html", f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}", dataset, f"{gse}_geo_page.html")
        if bioproject:
            add_download(
                rows,
                dataset,
                "SRA runinfo",
                f"https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc={bioproject}",
                dataset,
                f"{bioproject}_SraRunInfo.csv",
            )

    try:
        data, files = zenodo_files("7754315")
        zdir = "Phospho-seq_Blair_Zenodo_7754315"
        (OUT_ROOT / zdir).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / zdir / "zenodo_record_7754315.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        for key, link, size in files:
            add_download(rows, "Phospho-seq_Blair_Zenodo_7754315", key, link, zdir, key, f"size={size}")
    except Exception as exc:
        log(f"Zenodo 7754315 metadata failed: {exc}")

    for record_id, dataset in [
        ("17174371", "icCITE-seq_TCR100_icCITE_Zenodo_17174371"),
        ("17174337", "icCITE-seq_TCR100_ASAP_Zenodo_17174337"),
        ("17174369", "icCITE-seq_TCR100_CITE_Zenodo_17174369"),
    ]:
        zdir = dataset
        try:
            data, files = zenodo_files(record_id)
            (OUT_ROOT / zdir).mkdir(parents=True, exist_ok=True)
            (OUT_ROOT / zdir / f"zenodo_record_{record_id}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            if not files:
                (OUT_ROOT / zdir / "README_status.txt").write_text(
                    "Zenodo record metadata was saved, but no downloadable files were exposed by the API. "
                    "The TCR100 icCITE record appears embargoed or restricted.\n",
                    encoding="utf-8",
                )
            for key, link, size in files:
                add_download(rows, dataset, key, link, zdir, key, f"size={size}")
        except Exception as exc:
            (OUT_ROOT / zdir).mkdir(parents=True, exist_ok=True)
            (OUT_ROOT / zdir / "README_status.txt").write_text(f"Zenodo metadata failed: {exc}\n", encoding="utf-8")

    eh_prefix = "https://mghp.osn.xsede.org/bir190004-bucket01/ExperimentHub/"
    add_download(
        rows,
        "Bodenmiller_2012_HDCytoData",
        "Bodenmiller_BCR_XL_SE.rda",
        eh_prefix + "HDCytoData/Bodenmiller_BCR_XL/Bodenmiller_BCR_XL_SE.rda",
        "Bodenmiller_2012_HDCytoData",
    )
    add_download(
        rows,
        "Bodenmiller_2012_HDCytoData",
        "Bodenmiller_BCR_XL_flowSet.rda",
        eh_prefix + "HDCytoData/Bodenmiller_BCR_XL/Bodenmiller_BCR_XL_flowSet.rda",
        "Bodenmiller_2012_HDCytoData",
    )
    add_download(
        rows,
        "Sachs_2005_phospho_flow",
        "sachs.data.txt.gz",
        "https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz",
        "Sachs_2005_phospho_flow",
    )
    add_download(
        rows,
        "Sachs_2005_phospho_flow",
        "sachs.interventional.txt.gz",
        "https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz",
        "Sachs_2005_phospho_flow",
    )
    return rows


def write_tsv(path, rows, extra_status=None):
    cols = ["dataset", "label", "url", "rel_dir", "filename", "note"]
    if extra_status:
        cols = cols + ["status", "exit_code", "bytes", "path", "time"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(cols) + "\n")
        for row in rows:
            values = [str(row.get(c, "")) for c in cols]
            handle.write("\t".join(v.replace("\t", " ") for v in values) + "\n")


def download_one(row):
    out_dir = OUT_ROOT / row["rel_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / row["filename"]
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
            row["filename"],
            row["url"],
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
            row["url"],
        ]
    log(f"download start: {row['dataset']} | {row['filename']}")
    with (LOG_DIR / f"{safe_name(row['dataset'])}__{safe_name(row['filename'])}.curl.log").open("ab") as ferr:
        proc = subprocess.run(cmd, stdout=ferr, stderr=ferr)
    size = out_path.stat().st_size if out_path.exists() else 0
    row["status"] = "done" if proc.returncode == 0 and size > 0 else "failed"
    row["exit_code"] = str(proc.returncode)
    row["bytes"] = str(size)
    row["path"] = str(out_path)
    row["time"] = now()
    log(f"download {row['status']}: {row['dataset']} | {row['filename']} | bytes={size} | code={proc.returncode}")
    return row


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_manifest()
    write_tsv(MANIFEST, rows)
    completed = []
    for row in rows:
        try:
            completed.append(download_one(dict(row)))
        except Exception as exc:
            bad = dict(row)
            bad.update({"status": "failed_exception", "exit_code": "", "bytes": "0", "path": "", "time": now(), "note": f"{row.get('note','')} {exc}"})
            completed.append(bad)
            log(f"download exception: {row['dataset']} | {row['filename']} | {exc}")
        write_tsv(STATUS, completed, extra_status=True)
    (LOG_DIR / "done.txt").write_text(now() + "\n", encoding="utf-8")
    log("all downloads attempted")


if __name__ == "__main__":
    main()
