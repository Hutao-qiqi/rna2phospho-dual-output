import argparse
import json
import shutil
import subprocess
import sys
import time
from ftplib import FTP_TLS
from pathlib import Path


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
OUT_ROOT = ROOT / r"01_data\single_cell\raw\external_bulk_phospho_validation_v1"
LOG_DIR = OUT_ROOT / "_logs" / "large_assets_v2"

PRIDE_ACCESSIONS = [
    "PXD021607",
    "PXD021608",
    "PXD021609",
    "PXD021611",
    "PXD058009",
]

MASSIVE_ACCESSION = "PXD063604"
MASSIVE_ID = "MSV000097797"
MASSIVE_ROOT = f"/v09/{MASSIVE_ID}"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(text):
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in str(text))


def log(message):
    ensure_dir(LOG_DIR)
    text = f"[{now()}] {message}"
    print(text, flush=True)
    with (LOG_DIR / "download_large_assets_v2.log").open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def write_table(path, rows, cols):
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(cols) + "\n")
        for row in rows:
            values = [str(row.get(c, "")) for c in cols]
            handle.write("\t".join(v.replace("\t", " ").replace("\n", " ") for v in values) + "\n")


def read_table(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    cols = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line:
            continue
        values = line.split("\t")
        row = {col: values[i] if i < len(values) else "" for i, col in enumerate(cols)}
        rows.append(row)
    return rows


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


def pride_priority(category):
    if category in {"SEARCH", "RESULT", "OTHER"}:
        return 0
    if category == "PEAK":
        return 1
    if category == "RAW":
        return 2
    return 3


def collect_pride_rows(include_raw, max_file_gb=0.0):
    rows = []
    max_file_bytes = int(float(max_file_gb) * 1024**3) if max_file_gb and max_file_gb > 0 else 0
    for accession in PRIDE_ACCESSIONS:
        json_path = OUT_ROOT / accession / f"{accession}_files.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing PRIDE file metadata: {json_path}")
        files = json.loads(json_path.read_text(encoding="utf-8"))
        for item in files:
            category = category_value(item)
            name = str(item.get("fileName", ""))
            size = int(item.get("fileSizeBytes") or 0)
            if not include_raw and category == "RAW":
                continue
            if max_file_bytes > 0 and size > max_file_bytes:
                continue
            url = public_url(item)
            rows.append(
                {
                    "accession": accession,
                    "repository": "PRIDE",
                    "file_name": name,
                    "category": category,
                    "file_size_bytes": size,
                    "url": url,
                    "rel_path": name,
                    "local_path": str(OUT_ROOT / accession / name),
                    "priority": pride_priority(category),
                    "download": "yes" if url else "no_url",
                }
            )
    rows.sort(key=lambda row: (row["priority"], row["accession"], row["file_name"]))
    return rows


def write_aria2_input(rows, path):
    lines = []
    for row in rows:
        if row.get("download") != "yes":
            continue
        out_dir = ensure_dir(OUT_ROOT / row["accession"])
        lines.append(row["url"])
        lines.append(f"  dir={out_dir}")
        lines.append(f"  out={row['file_name']}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def start_pride_download(rows, args):
    aria2 = shutil.which("aria2c.exe") or shutil.which("aria2c")
    if aria2 is None:
        raise RuntimeError("aria2c not found on remote host.")
    input_path = LOG_DIR / "pride_aria2_input.txt"
    session_path = LOG_DIR / "pride_aria2_session.txt"
    aria2_log = LOG_DIR / "pride_aria2.log"
    write_aria2_input(rows, input_path)
    cmd = [
        aria2,
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--file-allocation=none",
        "--max-connection-per-server",
        str(args.pride_connections),
        "--split",
        str(args.pride_connections),
        "--min-split-size",
        "10M",
        "--max-concurrent-downloads",
        str(args.pride_parallel),
        "--max-tries",
        str(args.max_tries),
        "--retry-wait",
        str(args.retry_wait),
        "--connect-timeout",
        "60",
        "--timeout",
        "120",
        "--summary-interval",
        "60",
        "--save-session",
        str(session_path),
        "--save-session-interval",
        "60",
        "--input-file",
        str(input_path),
    ]
    log(f"PRIDE aria2 start rows={sum(1 for r in rows if r.get('download') == 'yes')} input={input_path}")
    handle = aria2_log.open("ab")
    return subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT), handle


def open_massive(max_attempts=8, timeout=300):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            ftp = FTP_TLS("massive-ftp.ucsd.edu", timeout=timeout)
            ftp.login()
            ftp.prot_p()
            return ftp
        except Exception as exc:
            last_error = exc
            log(f"MassIVE connect retry attempt={attempt} error={exc}")
            time.sleep(10 * attempt)
    raise last_error


def list_massive_files():
    rows = []
    log(f"MassIVE recursive list start host=massive-ftp.ucsd.edu root={MASSIVE_ROOT}")
    ftp = open_massive()

    def reconnect():
        nonlocal ftp
        try:
            ftp.close()
        except Exception:
            pass
        ftp = open_massive()

    def mlsd_with_retry(remote_dir, max_attempts=6):
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                return list(ftp.mlsd(remote_dir))
            except Exception as exc:
                last_error = exc
                log(f"MassIVE list retry attempt={attempt} dir={remote_dir} error={exc}")
                time.sleep(5 * attempt)
                reconnect()
        raise last_error

    def massive_priority(category, rel_path):
        low = rel_path.lower()
        if low.startswith("metadata/") or low.startswith("ccms_parameters/"):
            return 0
        if low.startswith("quant/"):
            return 1
        if low.startswith("search/"):
            return 2
        if low.startswith("sequence/"):
            return 3
        if "correctionfactors" in low:
            return 4
        if low.startswith("raw/"):
            return 6
        if "maxquant" in low:
            return 8
        return 5

    def walk(remote_dir):
        log(f"MassIVE list dir {remote_dir}")
        for name, facts in mlsd_with_retry(remote_dir):
            if name in {".", ".."}:
                continue
            remote_path = remote_dir.rstrip("/") + "/" + name
            if facts.get("type") == "dir":
                walk(remote_path)
            elif facts.get("type") == "file":
                rel = remote_path.split(f"{MASSIVE_ID}/", 1)[1]
                category = rel.split("/", 1)[0]
                size = int(facts.get("size") or 0)
                rows.append(
                    {
                        "accession": MASSIVE_ACCESSION,
                        "repository": "MassIVE",
                        "file_name": Path(rel).name,
                        "category": category,
                        "file_size_bytes": size,
                        "url": f"ftps://massive-ftp.ucsd.edu{remote_path}",
                        "remote_path": remote_path,
                        "rel_path": rel,
                        "local_path": str(OUT_ROOT / MASSIVE_ACCESSION / rel),
                        "priority": massive_priority(category, rel),
                        "download": "yes",
                    }
                )

    walk(MASSIVE_ROOT)
    ftp.quit()
    rows.sort(key=lambda row: (row["priority"], row["category"], row["rel_path"]))
    log(f"MassIVE recursive list complete files={len(rows)} gb={sum(int(r['file_size_bytes']) for r in rows) / 1024**3:.2f}")
    return rows


def load_massive_manifest(path):
    rows = read_table(path)
    out = []
    for row in rows:
        rel = row.get("rel_path", "")
        if not rel:
            continue
        item = dict(row)
        item["file_size_bytes"] = int(float(item.get("file_size_bytes") or 0))
        item["priority"] = int(float(item.get("priority") or 0))
        item["remote_path"] = item.get("remote_path") or f"{MASSIVE_ROOT}/{rel}"
        item["local_path"] = item.get("local_path") or str(OUT_ROOT / MASSIVE_ACCESSION / rel)
        item["url"] = item.get("url") or f"ftps://massive-ftp.ucsd.edu{item['remote_path']}"
        out.append(item)
    out.sort(key=lambda row: (int(row.get("priority", 0)), row.get("category", ""), row.get("rel_path", "")))
    return out


def retrieve_massive_binary(ftp, row, handle, args, rest=None):
    ftp.voidcmd("TYPE I")
    with ftp.transfercmd(f"RETR {row['remote_path']}", rest=rest) as conn:
        conn.settimeout(args.massive_read_timeout)
        while True:
            block = conn.recv(args.massive_block_size)
            if not block:
                break
            handle.write(block)
    return ftp.voidresp()


def download_massive_file_direct(row, args):
    out_path = Path(row["local_path"])
    ensure_dir(out_path.parent)
    expected = int(row["file_size_bytes"])
    if out_path.exists() and expected > 0 and out_path.stat().st_size == expected:
        return "exists", 0, out_path.stat().st_size
    max_file_tries = args.massive_file_tries if args.massive_file_tries is not None else args.max_tries
    for attempt in range(1, max_file_tries + 1):
        current = out_path.stat().st_size if out_path.exists() else 0
        mode = "ab" if current > 0 else "wb"
        try:
            ftp = open_massive(max_attempts=args.massive_connect_tries, timeout=args.massive_ftp_timeout)
            with out_path.open(mode) as handle:
                rest = str(current) if current > 0 else None
                retrieve_massive_binary(ftp, row, handle, args, rest=rest)
            ftp.quit()
            size = out_path.stat().st_size if out_path.exists() else 0
            if expected <= 0 or size == expected:
                return "done", 0, size
            log(f"MassIVE partial size mismatch attempt={attempt} file={row['rel_path']} size={size} expected={expected}")
        except Exception as exc:
            log(f"MassIVE retry attempt={attempt} file={row['rel_path']} error={exc}")
            time.sleep(args.retry_wait)
    size = out_path.stat().st_size if out_path.exists() else 0
    log(f"MassIVE skip after retries file={row['rel_path']} tries={max_file_tries} bytes={size} expected={expected}")
    return "failed", 1, size


def download_massive_file_with_watchdog(row, args):
    status_dir = ensure_dir(LOG_DIR / "massive_single_status")
    status_path = status_dir / f"{safe_name(row['rel_path'])}.json"
    if status_path.exists():
        status_path.unlink()
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--source",
        "massive",
        "--reuse-massive-manifest",
        "--massive-single-rel-path",
        row["rel_path"],
        "--massive-single-output-json",
        str(status_path),
        "--max-tries",
        str(args.max_tries),
        "--retry-wait",
        str(args.retry_wait),
        "--massive-block-size",
        str(args.massive_block_size),
        "--massive-connect-tries",
        str(args.massive_connect_tries),
        "--massive-ftp-timeout",
        str(args.massive_ftp_timeout),
        "--massive-read-timeout",
        str(args.massive_read_timeout),
    ]
    if args.include_raw:
        cmd.append("--include-raw")
    if args.include_massive_tools:
        cmd.append("--include-massive-tools")
    if args.massive_file_tries is not None:
        cmd.extend(["--massive-file-tries", str(args.massive_file_tries)])

    out_path = Path(row["local_path"])
    last_size = out_path.stat().st_size if out_path.exists() else 0
    last_change = time.time()
    log(f"MassIVE child start file={row['rel_path']} stall_timeout={args.massive_process_stall_timeout}s bytes={last_size}")
    proc = subprocess.Popen(cmd)
    while proc.poll() is None:
        time.sleep(args.massive_process_poll)
        size = out_path.stat().st_size if out_path.exists() else 0
        if size != last_size:
            last_size = size
            last_change = time.time()
            log(f"MassIVE child progress file={row['rel_path']} bytes={size}")
        if time.time() - last_change > args.massive_process_stall_timeout:
            log(f"MassIVE child stalled file={row['rel_path']} bytes={size}; terminating")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            size = out_path.stat().st_size if out_path.exists() else 0
            return "failed", 124, size

    size = out_path.stat().st_size if out_path.exists() else 0
    if status_path.exists():
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        return payload.get("status", "failed"), int(payload.get("exit_code", proc.returncode or 0)), int(payload.get("bytes", size))
    expected = int(row.get("file_size_bytes") or 0)
    if expected > 0 and size == expected:
        return "done", int(proc.returncode or 0), size
    return "failed", int(proc.returncode or 1), size


def download_massive_file(row, args):
    if args.massive_process_stall_timeout and not args.massive_single_rel_path:
        return download_massive_file_with_watchdog(row, args)
    return download_massive_file_direct(row, args)


def download_massive_rows(rows, args):
    status_rows = []
    skip = {str(x).replace("\\", "/").strip("/") for x in args.skip_massive_rel_path}
    selected = [
        row
        for row in rows
        if (args.include_raw or row["category"] != "raw")
        and (args.include_massive_tools or "maxquant" not in row["rel_path"].lower())
        and str(row["rel_path"]).replace("\\", "/").strip("/") not in skip
    ]
    log(f"MassIVE download start rows={len(selected)} include_raw={args.include_raw}")
    for i, row in enumerate(selected, 1):
        log(f"MassIVE file {i}/{len(selected)} start {row['rel_path']} size_gb={int(row['file_size_bytes']) / 1024**3:.3f}")
        status, code, size = download_massive_file(row, args)
        out = dict(row)
        out.update({"status": status, "exit_code": code, "bytes": size, "time": now()})
        status_rows.append(out)
        write_table(
            LOG_DIR / "pxd063604_massive_download_status.tsv",
            status_rows,
            ["accession", "repository", "category", "rel_path", "file_size_bytes", "url", "local_path", "status", "exit_code", "bytes", "time"],
        )
        log(f"MassIVE file {i}/{len(selected)} {status} {row['rel_path']} bytes={size}")
    return status_rows


def summarize_local_files(rows):
    out = []
    for row in rows:
        path = Path(row["local_path"])
        expected = int(row.get("file_size_bytes") or 0)
        size = path.stat().st_size if path.exists() else 0
        if expected > 0 and size == expected:
            status = "complete"
        elif size > 0:
            status = "partial"
        else:
            status = "missing"
        item = dict(row)
        item.update({"status": status, "bytes": size, "time": now()})
        out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-raw", action="store_true", default=False)
    parser.add_argument("--skip-raw", action="store_false", dest="include_raw")
    parser.add_argument("--pride-parallel", type=int, default=4)
    parser.add_argument("--pride-connections", type=int, default=8)
    parser.add_argument("--pride-max-file-gb", type=float, default=0.0)
    parser.add_argument("--max-tries", type=int, default=20)
    parser.add_argument("--retry-wait", type=int, default=20)
    parser.add_argument("--massive-block-size", type=int, default=1048576)
    parser.add_argument("--massive-file-tries", type=int, default=None)
    parser.add_argument("--massive-connect-tries", type=int, default=4)
    parser.add_argument("--massive-ftp-timeout", type=int, default=120)
    parser.add_argument("--massive-read-timeout", type=int, default=60)
    parser.add_argument("--massive-process-stall-timeout", type=int, default=180)
    parser.add_argument("--massive-process-poll", type=int, default=10)
    parser.add_argument("--massive-single-rel-path", default="")
    parser.add_argument("--massive-single-output-json", default="")
    parser.add_argument("--manifest-only", action="store_true", default=False)
    parser.add_argument("--source", choices=["all", "pride", "massive"], default="all")
    parser.add_argument("--include-massive-tools", action="store_true", default=False)
    parser.add_argument("--reuse-massive-manifest", action="store_true", default=True)
    parser.add_argument("--refresh-massive-manifest", action="store_false", dest="reuse_massive_manifest")
    parser.add_argument("--skip-massive-rel-path", action="append", default=[])
    args = parser.parse_args()

    ensure_dir(LOG_DIR)
    ensure_dir(OUT_ROOT / MASSIVE_ACCESSION)
    if args.massive_single_rel_path:
        massive_manifest = LOG_DIR / "pxd063604_massive_file_manifest.tsv"
        massive_rows = load_massive_manifest(massive_manifest)
        wanted = str(args.massive_single_rel_path).replace("\\", "/").strip("/")
        row = next((item for item in massive_rows if str(item.get("rel_path", "")).replace("\\", "/").strip("/") == wanted), None)
        if row is None:
            raise FileNotFoundError(f"Missing MassIVE manifest row: {wanted}")
        status, code, size = download_massive_file_direct(row, args)
        payload = {"status": status, "exit_code": code, "bytes": size, "rel_path": row["rel_path"], "time": now()}
        if args.massive_single_output_json:
            Path(args.massive_single_output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise SystemExit(code)

    pride_rows = collect_pride_rows(args.include_raw, args.pride_max_file_gb) if args.source in {"all", "pride"} else []
    massive_manifest = LOG_DIR / "pxd063604_massive_file_manifest.tsv"
    massive_rows = []
    if args.source in {"all", "massive"}:
        if args.reuse_massive_manifest and massive_manifest.exists():
            massive_rows = load_massive_manifest(massive_manifest)
            log(f"MassIVE manifest reused path={massive_manifest} files={len(massive_rows)}")
        else:
            massive_rows = list_massive_files()
    if pride_rows:
        write_table(
            LOG_DIR / "pride_large_asset_manifest.tsv",
            pride_rows,
            ["accession", "repository", "category", "file_name", "file_size_bytes", "url", "local_path", "priority", "download"],
        )
    if massive_rows:
        write_table(
            LOG_DIR / "pxd063604_massive_file_manifest.tsv",
            massive_rows,
            ["accession", "repository", "category", "rel_path", "remote_path", "file_size_bytes", "url", "local_path", "priority", "download"],
        )
    log(
        "manifest ready "
        f"pride_files={len(pride_rows)} pride_gb={sum(int(r['file_size_bytes']) for r in pride_rows)/1024**3:.2f} "
        f"massive_files={len(massive_rows)} massive_gb={sum(int(r['file_size_bytes']) for r in massive_rows if args.include_raw or r['category'] != 'raw')/1024**3:.2f}"
    )
    if args.manifest_only:
        log("manifest-only requested; no downloads started")
        return

    pride_code = 0
    if pride_rows and massive_rows:
        pride_proc = None
        pride_handle = None
        try:
            pride_proc, pride_handle = start_pride_download(pride_rows, args)
            download_massive_rows(massive_rows, args)
            pride_code = pride_proc.wait()
            log(f"PRIDE aria2 exit_code={pride_code}")
        finally:
            if pride_proc is not None and pride_proc.poll() is None:
                pride_proc.terminate()
            if pride_handle is not None:
                pride_handle.close()
    elif pride_rows:
        pride_proc, pride_handle = start_pride_download(pride_rows, args)
        pride_code = pride_proc.wait()
        pride_handle.close()
        log(f"PRIDE aria2 exit_code={pride_code}")
    elif massive_rows:
        download_massive_rows(massive_rows, args)

    pride_status = summarize_local_files(pride_rows)
    skip = {str(x).replace("\\", "/").strip("/") for x in args.skip_massive_rel_path}
    massive_final = summarize_local_files(
        [
            row
            for row in massive_rows
            if (args.include_raw or row["category"] != "raw")
            and (args.include_massive_tools or "maxquant" not in row["rel_path"].lower())
            and str(row["rel_path"]).replace("\\", "/").strip("/") not in skip
        ]
    )
    write_table(
        LOG_DIR / "pride_large_asset_download_status.tsv",
        pride_status,
        ["accession", "repository", "category", "file_name", "file_size_bytes", "url", "local_path", "status", "bytes", "time"],
    )
    write_table(
        LOG_DIR / "pxd063604_massive_download_status.final.tsv",
        massive_final,
        ["accession", "repository", "category", "rel_path", "file_size_bytes", "url", "local_path", "status", "bytes", "time"],
    )
    summary = {
        "created_at": now(),
        "include_raw": bool(args.include_raw),
        "pride_files": len(pride_rows),
        "pride_complete": sum(1 for row in pride_status if row["status"] == "complete"),
        "pride_partial": sum(1 for row in pride_status if row["status"] == "partial"),
        "massive_files": len(massive_final),
        "massive_complete": sum(1 for row in massive_final if row["status"] == "complete"),
        "massive_partial": sum(1 for row in massive_final if row["status"] == "partial"),
        "pride_total_gb": sum(int(r["file_size_bytes"]) for r in pride_rows) / 1024**3,
        "massive_total_gb": sum(int(r["file_size_bytes"]) for r in massive_final) / 1024**3,
    }
    (LOG_DIR / "large_asset_download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"large asset download complete {json.dumps(summary)}")


if __name__ == "__main__":
    main()
