import csv
import json
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
OUT = ROOT / r"01_data\single_cell\raw\external_single_cell_phospho_validation_v1\SIGNAL-seq_GSE256405\processed_h5ad"
LOG = OUT / "_logs"

FILES = [
    (
        "GSE256403",
        "HeLa RNA",
        "GSE256403_ex0003_hela_rna_adata.h5ad.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE256nnn/GSE256403/suppl/GSE256403_ex0003_hela_rna_adata.h5ad.gz",
    ),
    (
        "GSE256403",
        "HeLa ADT",
        "GSE256403_ex0003_hela_adt_adata.h5ad.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE256nnn/GSE256403/suppl/GSE256403_ex0003_hela_adt_adata.h5ad.gz",
    ),
    (
        "GSE256403",
        "HeLa ADT feature reference",
        "GSE256403_ex0003_adt_feature_reference.csv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE256nnn/GSE256403/suppl/GSE256403_ex0003_adt_feature_reference.csv.gz",
    ),
    (
        "GSE256404",
        "PDO CAF RNA",
        "GSE256404_ex0015_pdo_rna_adata.h5ad.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE256nnn/GSE256404/suppl/GSE256404_ex0015_pdo_rna_adata.h5ad.gz",
    ),
    (
        "GSE256404",
        "PDO CAF ADT",
        "GSE256404_ex0015_pdo_adt_adata.h5ad.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE256nnn/GSE256404/suppl/GSE256404_ex0015_pdo_adt_adata.h5ad.gz",
    ),
    (
        "GSE256404",
        "PDO CAF feature reference",
        "GSE256404_ex0015_feature_reference.csv.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE256nnn/GSE256404/suppl/GSE256404_ex0015_feature_reference.csv.gz",
    ),
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in text)


def log(msg: str) -> None:
    ensure_dir(LOG)
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with (LOG / "download_processed_h5ad.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def download_one(row: dict) -> dict:
    out_path = OUT / row["filename"]
    if out_path.exists() and out_path.stat().st_size > 0:
        return {**row, "status": "exists", "bytes": out_path.stat().st_size, "exit_code": 0, "path": str(out_path), "time": now()}
    aria2 = shutil.which("aria2c.exe") or shutil.which("aria2c")
    if aria2:
        cmd = [
            aria2,
            "--continue=true",
            "--max-connection-per-server=8",
            "--split=8",
            "--min-split-size=1M",
            "--max-tries=10",
            "--retry-wait=10",
            "--connect-timeout=60",
            "--timeout=60",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--dir",
            str(OUT),
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
            "10",
            "--retry-delay",
            "10",
            "--connect-timeout",
            "60",
            "--output",
            str(out_path),
            row["url"],
        ]
    log(f"download start: {row['filename']}")
    with (LOG / f"{safe_name(row['filename'])}.log").open("ab") as ferr:
        proc = subprocess.run(cmd, stdout=ferr, stderr=ferr)
    size = out_path.stat().st_size if out_path.exists() else 0
    status = "done" if proc.returncode == 0 and size > 0 else "failed"
    log(f"download {status}: {row['filename']} bytes={size} code={proc.returncode}")
    return {**row, "status": status, "bytes": size, "exit_code": proc.returncode, "path": str(out_path), "time": now()}


def write_tsv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_dir(OUT)
    ensure_dir(LOG)
    rows = [
        {"series": series, "label": label, "filename": filename, "url": url}
        for series, label, filename, url in FILES
    ]
    write_tsv(OUT / "signal_seq_processed_h5ad_manifest.tsv", rows)
    done = []
    for row in rows:
        done.append(download_one(row))
        write_tsv(OUT / "signal_seq_processed_h5ad_status.tsv", done)
    summary = {
        "out_dir": str(OUT),
        "n_files": len(rows),
        "n_done": sum(1 for x in done if x["status"] in {"done", "exists"}),
        "n_failed": sum(1 for x in done if x["status"] == "failed"),
        "bytes": sum(int(x.get("bytes") or 0) for x in done),
    }
    (OUT / "signal_seq_processed_h5ad_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
