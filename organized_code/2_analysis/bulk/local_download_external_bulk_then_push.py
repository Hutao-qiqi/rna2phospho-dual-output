import argparse
import csv
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from ftplib import FTP_TLS
from pathlib import Path


PROJECT_ROOT = Path(r"E:\data\gongke\TCGA-TCPA")
LOCAL_ROOT = PROJECT_ROOT / r"01_data\single_cell\raw\external_bulk_phospho_validation_v1"
LOG_DIR = LOCAL_ROOT / "_logs" / "local_then_push_v1"
REMOTE_HOST = "admin@REMOTE_HOST_REDACTED"
REMOTE_ROOT_WIN = r"D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\raw\external_bulk_phospho_validation_v1"
REMOTE_ROOT_SCP = "D:/data/lsy/vm_lsy_parent/lsy/01_data/single_cell/raw/external_bulk_phospho_validation_v1"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(message):
    ensure_dir(LOG_DIR)
    text = f"[{now()}] {message}"
    print(text, flush=True)
    with (LOG_DIR / "local_download_then_push.log").open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def read_table(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_status(path, row):
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    cols = [
        "source",
        "accession",
        "category",
        "rel_path",
        "file_size_bytes",
        "local_path",
        "remote_path",
        "download_status",
        "push_status",
        "bytes",
        "time",
    ]
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def file_size(path):
    path = Path(path)
    return path.stat().st_size if path.exists() else 0


def run_cmd(cmd, timeout=None):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)


def remote_win_dir(rel_dir):
    rel_dir = str(rel_dir).replace("/", "\\").strip("\\")
    return REMOTE_ROOT_WIN + ("\\" + rel_dir if rel_dir else "")


def remote_scp_path(rel_path):
    rel_path = str(rel_path).replace("\\", "/").strip("/")
    return f"{REMOTE_HOST}:{REMOTE_ROOT_SCP}/{rel_path}"


def make_remote_dir(rel_path):
    rel_dir = str(Path(rel_path).parent).replace(".", "").replace("/", "\\").strip("\\")
    target = remote_win_dir(rel_dir)
    cmd = [
        "ssh",
        REMOTE_HOST,
        "powershell",
        "-NoProfile",
        "-Command",
        f"New-Item -ItemType Directory -Force -Path '{target}' | Out-Null",
    ]
    return run_cmd(cmd, timeout=120).returncode == 0


def push_file(local_path, rel_path, expected_size):
    local_path = Path(local_path)
    if not local_path.exists():
        return "missing_local"
    make_remote_dir(rel_path)
    cmd = ["scp", "-q", str(local_path), remote_scp_path(rel_path)]
    res = run_cmd(cmd, timeout=3600)
    if res.returncode != 0:
        log(f"push failed rel={rel_path} code={res.returncode} output={res.stdout[-500:]}")
        return "failed"
    return "pushed"


def pride_rows(args):
    manifest = LOCAL_ROOT / "_logs" / "large_assets_v2" / "pride_large_asset_manifest.tsv"
    rows = read_table(manifest)
    selected = []
    max_bytes = int(args.pride_max_file_gb * 1024**3) if args.pride_max_file_gb > 0 else 0
    for row in rows:
        size = int(float(row.get("file_size_bytes") or 0))
        category = str(row.get("category", ""))
        if args.skip_raw and category == "RAW":
            continue
        if max_bytes and size > max_bytes:
            continue
        accession = str(row["accession"])
        name = str(row["file_name"])
        local_path = LOCAL_ROOT / accession / name
        selected.append(
            {
                "source": "PRIDE",
                "accession": accession,
                "category": category,
                "rel_path": f"{accession}/{name}",
                "url": row["url"],
                "file_size_bytes": size,
                "local_path": str(local_path),
            }
        )
    return selected


def massive_rows(args):
    manifest = LOCAL_ROOT / "_logs" / "large_assets_v2" / "pxd063604_massive_file_manifest.tsv"
    rows = read_table(manifest)
    selected = []
    max_bytes = int(args.massive_max_file_gb * 1024**3) if args.massive_max_file_gb > 0 else 0
    skip = {x.replace("\\", "/").strip("/") for x in args.skip_massive_rel_path}
    categories = set(args.massive_category)
    for row in rows:
        rel = str(row["rel_path"]).replace("\\", "/").strip("/")
        category = str(row.get("category", ""))
        size = int(float(row.get("file_size_bytes") or 0))
        if categories and category not in categories:
            continue
        if not args.include_massive_raw and category == "raw":
            continue
        if max_bytes and size > max_bytes:
            continue
        if rel in skip:
            continue
        local_path = LOCAL_ROOT / "PXD063604" / rel
        selected.append(
            {
                "source": "MassIVE",
                "accession": "PXD063604",
                "category": category,
                "rel_path": f"PXD063604/{rel}",
                "remote_ftp_path": row.get("remote_path") or f"/v09/MSV000097797/{rel}",
                "file_size_bytes": size,
                "local_path": str(local_path),
            }
        )
    return selected


def download_pride_one(row, args):
    local_path = Path(row["local_path"])
    ensure_dir(local_path.parent)
    complete_marker = local_path.with_suffix(local_path.suffix + ".complete")
    expected = int(row["file_size_bytes"])
    if complete_marker.exists() and local_path.exists() and file_size(local_path) > 0:
        return "exists", file_size(local_path)
    if expected and file_size(local_path) == expected:
        complete_marker.write_text(f"complete {now()}\n", encoding="utf-8")
        return "exists", expected
    curl = shutil.which("curl.exe") or shutil.which("curl")
    cmd = [
        curl,
        "--fail",
        "--location",
        "--continue-at",
        "-",
        "--retry",
        str(args.retries),
        "--retry-delay",
        str(args.retry_delay),
        "--connect-timeout",
        "60",
        "--speed-limit",
        str(args.speed_limit),
        "--speed-time",
        str(args.speed_time),
        "--output",
        str(local_path),
        row["url"],
    ]
    res = run_cmd(cmd, timeout=args.file_timeout)
    size = file_size(local_path)
    if res.returncode == 0 and size > 0:
        complete_marker.write_text(f"complete {now()}\n", encoding="utf-8")
        return "done", size
    log(f"PRIDE download incomplete rel={row['rel_path']} code={res.returncode} bytes={size} expected={expected} out={res.stdout[-300:]}")
    return "partial" if size > 0 else "failed", size


def open_massive(args):
    last = None
    for attempt in range(1, args.retries + 1):
        try:
            ftp = FTP_TLS("massive-ftp.ucsd.edu", timeout=args.massive_timeout)
            ftp.login()
            ftp.prot_p()
            return ftp
        except Exception as exc:
            last = exc
            log(f"MassIVE connect retry attempt={attempt} error={exc}")
            time.sleep(args.retry_delay)
    raise last


def download_massive_one(row, args):
    local_path = Path(row["local_path"])
    ensure_dir(local_path.parent)
    expected = int(row["file_size_bytes"])
    if expected and file_size(local_path) == expected:
        return "exists", expected
    for attempt in range(1, args.retries + 1):
        start = file_size(local_path)
        mode = "ab" if start else "wb"
        try:
            ftp = open_massive(args)
            ftp.voidcmd("TYPE I")
            with local_path.open(mode) as handle:
                rest = str(start) if start else None
                with ftp.transfercmd(f"RETR {row['remote_ftp_path']}", rest=rest) as conn:
                    conn.settimeout(args.massive_read_timeout)
                    while True:
                        block = conn.recv(args.massive_block_size)
                        if not block:
                            break
                        handle.write(block)
                ftp.voidresp()
            ftp.quit()
            size = file_size(local_path)
            if expected and size == expected:
                return "done", size
            log(f"MassIVE partial rel={row['rel_path']} attempt={attempt} bytes={size} expected={expected}")
        except Exception as exc:
            size = file_size(local_path)
            log(f"MassIVE retry rel={row['rel_path']} attempt={attempt} bytes={size} error={exc}")
            time.sleep(args.retry_delay)
    size = file_size(local_path)
    return "partial" if size > 0 else "failed", size


def handle_one(row, args):
    expected = int(row["file_size_bytes"])
    if row["source"] == "PRIDE":
        d_status, size = download_pride_one(row, args)
    else:
        d_status, size = download_massive_one(row, args)
    push_status = "not_pushed"
    complete_enough = (row["source"] == "PRIDE" and d_status in {"done", "exists"} and size > 0) or (expected and size == expected)
    if args.push_complete and complete_enough:
        push_status = push_file(row["local_path"], row["rel_path"], expected)
    out = {
        **row,
        "download_status": d_status,
        "push_status": push_status,
        "bytes": size,
        "remote_path": remote_scp_path(row["rel_path"]),
        "time": now(),
    }
    append_status(LOG_DIR / "local_download_then_push_status.tsv", out)
    log(f"{row['source']} {d_status} push={push_status} rel={row['rel_path']} bytes={size}/{expected}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["pride", "massive", "all"], default="pride")
    parser.add_argument("--push-complete", action="store_true", default=True)
    parser.add_argument("--no-push", action="store_false", dest="push_complete")
    parser.add_argument("--skip-raw", action="store_true", default=True)
    parser.add_argument("--include-raw", action="store_false", dest="skip_raw")
    parser.add_argument("--pride-max-file-gb", type=float, default=5.0)
    parser.add_argument("--massive-max-file-gb", type=float, default=5.0)
    parser.add_argument("--massive-category", action="append", default=["ccms_parameters", "metadata", "quant", "search"])
    parser.add_argument("--include-massive-raw", action="store_true", default=False)
    parser.add_argument("--skip-massive-rel-path", action="append", default=[])
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--massive-parallel", type=int, default=1)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--retry-delay", type=int, default=10)
    parser.add_argument("--speed-limit", type=int, default=1024)
    parser.add_argument("--speed-time", type=int, default=180)
    parser.add_argument("--file-timeout", type=int, default=7200)
    parser.add_argument("--massive-timeout", type=int, default=90)
    parser.add_argument("--massive-read-timeout", type=int, default=90)
    parser.add_argument("--massive-block-size", type=int, default=1048576)
    args = parser.parse_args()

    ensure_dir(LOG_DIR)
    rows = []
    if args.source in {"pride", "all"}:
        rows.extend(pride_rows(args))
    if args.source in {"massive", "all"}:
        rows.extend(massive_rows(args))
    total = sum(int(r["file_size_bytes"]) for r in rows)
    log(f"selected source={args.source} files={len(rows)} gb={total/1024**3:.2f} push={args.push_complete}")
    (LOG_DIR / f"selected_{args.source}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if args.source == "massive":
        parallel = args.massive_parallel
    elif args.source == "all":
        parallel = min(args.parallel, 4)
    else:
        parallel = args.parallel
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futs = [pool.submit(handle_one, row, args) for row in rows]
        for fut in as_completed(futs):
            fut.result()
    log(f"complete source={args.source}")


if __name__ == "__main__":
    main()
