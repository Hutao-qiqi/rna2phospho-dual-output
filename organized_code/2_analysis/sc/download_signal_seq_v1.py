import argparse
import csv
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
OUT_ROOT = ROOT / r"01_data\single_cell\raw\external_single_cell_phospho_validation_v1\SIGNAL-seq_GSE256405"
LOG_DIR = OUT_ROOT / "_logs"

GEO_SERIES = [
    ("GSE256405", "GSE256nnn", "superseries"),
    ("GSE256403", "GSE256nnn", "part_I"),
    ("GSE256404", "GSE256nnn", "part_II"),
]

BIOPROJECTS = [
    ("PRJNA1079350", "superseries"),
    ("PRJNA1079357", "part_I"),
    ("PRJNA1079356", "part_II"),
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(msg: str) -> None:
    ensure_dir(LOG_DIR)
    text = f"[{now()}] {msg}"
    print(text, flush=True)
    with (LOG_DIR / "download_signal_seq.log").open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in text)


def download(url: str, out_path: Path, resume: bool = True) -> dict:
    ensure_dir(out_path.parent)
    if out_path.exists() and out_path.stat().st_size > 0:
        return {"status": "exists", "bytes": out_path.stat().st_size, "exit_code": 0, "path": str(out_path)}
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
            str(out_path.parent),
            "--out",
            out_path.name,
            url,
        ]
    else:
        cmd = [
            "curl.exe",
            "-L",
            "-C",
            "-" if resume else "0",
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
    log(f"download start: {out_path.name}")
    with (LOG_DIR / f"{safe_name(out_path.name)}.log").open("ab") as ferr:
        proc = subprocess.run(cmd, stdout=ferr, stderr=ferr)
    size = out_path.stat().st_size if out_path.exists() else 0
    status = "done" if proc.returncode == 0 and size > 0 else "failed"
    log(f"download {status}: {out_path.name} bytes={size} code={proc.returncode}")
    return {"status": status, "bytes": size, "exit_code": proc.returncode, "path": str(out_path)}


def geo_rows() -> list[dict]:
    rows = []
    for gse, series, label in GEO_SERIES:
        base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{series}/{gse}"
        rows.extend(
            [
                {
                    "group": label,
                    "accession": gse,
                    "kind": "soft",
                    "url": f"{base}/soft/{gse}_family.soft.gz",
                    "rel_path": f"{label}/{gse}_family.soft.gz",
                },
                {
                    "group": label,
                    "accession": gse,
                    "kind": "miniml",
                    "url": f"{base}/miniml/{gse}_family.xml.tgz",
                    "rel_path": f"{label}/{gse}_family.xml.tgz",
                },
                {
                    "group": label,
                    "accession": gse,
                    "kind": "series_matrix",
                    "url": f"{base}/matrix/{gse}_series_matrix.txt.gz",
                    "rel_path": f"{label}/{gse}_series_matrix.txt.gz",
                },
                {
                    "group": label,
                    "accession": gse,
                    "kind": "geo_page",
                    "url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}",
                    "rel_path": f"{label}/{gse}_geo_page.html",
                },
            ]
        )
    return rows


def ena_report_url(accession: str) -> str:
    fields = ",".join(
        [
            "study_accession",
            "sample_accession",
            "secondary_sample_accession",
            "experiment_accession",
            "run_accession",
            "library_strategy",
            "library_layout",
            "instrument_platform",
            "instrument_model",
            "fastq_ftp",
            "fastq_bytes",
            "submitted_ftp",
            "submitted_bytes",
        ]
    )
    return (
        "https://www.ebi.ac.uk/ena/portal/api/filereport"
        f"?accession={accession}&result=read_run&fields={fields}&format=tsv"
    )


def sra_runinfo_url(accession: str) -> str:
    return f"https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc={accession}"


def read_ena_report(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fastq_rows_from_ena(reports: list[Path]) -> list[dict]:
    rows = []
    seen = set()
    for report in reports:
        for row in read_ena_report(report):
            run = row.get("run_accession", "")
            ftp = row.get("fastq_ftp", "")
            sizes = row.get("fastq_bytes", "")
            for i, item in enumerate([x for x in ftp.split(";") if x]):
                url = "https://" + item if item.startswith("ftp.sra.ebi.ac.uk/") else item
                filename = item.rstrip("/").split("/")[-1]
                key = (run, filename)
                if key in seen:
                    continue
                seen.add(key)
                size = sizes.split(";")[i] if i < len(sizes.split(";")) else ""
                rows.append(
                    {
                        "group": "fastq",
                        "accession": run,
                        "kind": "fastq",
                        "url": url,
                        "rel_path": f"fastq/{filename}",
                        "bytes_expected": size,
                    }
                )
    return rows


def write_tsv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--download-fastq", action="store_true")
    args = ap.parse_args()

    global OUT_ROOT, LOG_DIR
    root = Path(args.root)
    OUT_ROOT = root / r"01_data\single_cell\raw\external_single_cell_phospho_validation_v1\SIGNAL-seq_GSE256405"
    LOG_DIR = OUT_ROOT / "_logs"
    ensure_dir(OUT_ROOT)
    ensure_dir(LOG_DIR)

    rows = geo_rows()
    report_paths = []
    for accession, label in BIOPROJECTS:
        ena_path = OUT_ROOT / label / f"{accession}_ENA_run_report.tsv"
        sra_path = OUT_ROOT / label / f"{accession}_SraRunInfo.csv"
        rows.append(
            {
                "group": label,
                "accession": accession,
                "kind": "ena_run_report",
                "url": ena_report_url(accession),
                "rel_path": f"{label}/{accession}_ENA_run_report.tsv",
            }
        )
        rows.append(
            {
                "group": label,
                "accession": accession,
                "kind": "sra_runinfo",
                "url": sra_runinfo_url(accession),
                "rel_path": f"{label}/{accession}_SraRunInfo.csv",
            }
        )
        report_paths.append(ena_path)

    write_tsv(OUT_ROOT / "signal_seq_download_manifest.tsv", rows)
    status = []
    for row in rows:
        out_path = OUT_ROOT / row["rel_path"]
        result = download(row["url"], out_path)
        status.append({**row, **result, "time": now()})
        write_tsv(OUT_ROOT / "signal_seq_download_status.tsv", status)

    fastq_rows = fastq_rows_from_ena(report_paths)
    write_tsv(OUT_ROOT / "signal_seq_fastq_manifest.tsv", fastq_rows)
    if args.download_fastq:
        for row in fastq_rows:
            out_path = OUT_ROOT / row["rel_path"]
            result = download(row["url"], out_path)
            status.append({**row, **result, "time": now()})
            write_tsv(OUT_ROOT / "signal_seq_download_status.tsv", status)

    summary = {
        "out_root": str(OUT_ROOT),
        "n_metadata_rows": len(rows),
        "n_fastq_files": len(fastq_rows),
        "fastq_download_requested": bool(args.download_fastq),
    }
    (OUT_ROOT / "signal_seq_download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
