import gzip
import json
import re
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
RAW_DIR = ROOT / r"01_data\single_cell\raw\external_single_cell_phospho_validation_v1\icCITE-plex_GSE300551"
OUT_DIR = ROOT / r"02_results\single_cell\20260518_gse300551_iccite_plex_inspect"


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return "exists"
    cmd = ["curl.exe", "-L", "--retry", "5", "--retry-delay", "5", "--output", str(out_path), url]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace"))
    return "downloaded"


def parse_soft_supplementary():
    text = gzip.open(RAW_DIR / "GSE300551_family.soft.gz", "rt", encoding="utf-8", errors="replace").read()
    urls = []
    for line in text.splitlines():
        if line.startswith("!Series_supplementary_file = "):
            url = line.split("=", 1)[1].strip()
            if url.endswith("GSE300551_RAW.tar"):
                continue
            urls.append(url.replace("ftp://", "https://"))
    return urls


def read_csv_gz(path):
    try:
        return pd.read_csv(path, compression="gzip")
    except Exception:
        return pd.read_csv(path, compression="gzip", sep=None, engine="python")


def preview_excel_gz(path):
    tmp = OUT_DIR / "intermediate" / path.name[:-3]
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "rb") as src, tmp.open("wb") as dst:
        dst.write(src.read())
    out = {"file": str(path), "decompressed": str(tmp), "bytes": path.stat().st_size}
    try:
        xls = pd.ExcelFile(tmp)
        out["sheet_names"] = xls.sheet_names
        for sheet in xls.sheet_names:
            df = pd.read_excel(tmp, sheet_name=sheet)
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sheet).strip("_") or "sheet"
            df.head(2000).to_csv(OUT_DIR / "tables" / f"additional_metadata_{safe}_preview.tsv", sep="\t", index=False)
            out.setdefault("sheets", {})[sheet] = {
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "columns": [str(c) for c in df.columns],
                "head": df.head(5).astype(str).to_dict(orient="records"),
            }
    except Exception as exc:
        out["excel_error"] = str(exc)
        try:
            raw = tmp.read_bytes()[:2000]
            out["raw_preview"] = raw.decode("utf-8", "replace")
        except Exception:
            pass
    return out


def main():
    ensure_dir(OUT_DIR / "tables")
    ensure_dir(OUT_DIR / "reports")
    urls = parse_soft_supplementary()
    download_rows = []
    table_summaries = {}
    for url in urls:
        name = url.rsplit("/", 1)[-1]
        out_path = RAW_DIR / name
        status = download(url, out_path)
        download_rows.append({"url": url, "file": str(out_path), "bytes": out_path.stat().st_size, "status": status})
        if name.endswith(".csv.gz"):
            df = read_csv_gz(out_path)
            df.to_csv(OUT_DIR / "tables" / name.replace(".csv.gz", ".tsv"), sep="\t", index=False)
            table_summaries[name] = {
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "columns": [str(c) for c in df.columns],
                "head": df.head(8).astype(str).to_dict(orient="records"),
            }
        elif name.endswith(".xls.gz"):
            table_summaries[name] = preview_excel_gz(out_path)
    pd.DataFrame(download_rows).to_csv(OUT_DIR / "tables" / "gse300551_supplementary_downloads.tsv", sep="\t", index=False)
    (OUT_DIR / "reports" / "gse300551_metadata_summary.json").write_text(
        json.dumps(table_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"downloads": download_rows, "tables": table_summaries}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
