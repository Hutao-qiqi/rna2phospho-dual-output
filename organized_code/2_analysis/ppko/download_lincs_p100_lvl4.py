import argparse
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
DEFAULT_OUT = DEFAULT_ROOT / r"01_data\single_cell\raw\lincs_p100_lvl4_v1"
P100_PAGE = "https://panoramaweb.org/LINCS/Publications/Touchstone_2/P100/project-begin.view"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_text(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read().decode("utf-8", errors="replace")


def download(url, dest, timeout=240):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as src, open(dest, "wb") as out:
        out.write(src.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--page-url", default=P100_PAGE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    out = ensure_dir(args.output_dir)
    html = fetch_text(args.page_url)
    (out / "p100_project_page.html").write_text(html, encoding="utf-8")
    urls = sorted(
        set(
            re.findall(
                r"https://panoramaweb\.org/_webdav/LINCS/Publications/Touchstone_2/P100/%40files/GCT/[^\"<>]+_LVL4\.gct",
                html,
            )
        )
    )
    if args.limit and args.limit > 0:
        urls = urls[: args.limit]

    rows = []
    for url in urls:
        name = url.rsplit("/", 1)[-1].replace("%40", "@")
        dest = out / name
        status = "exists"
        if not dest.exists() or dest.stat().st_size == 0:
            status = "downloaded"
            download(url, dest)
            time.sleep(args.sleep)
        rows.append({"file": name, "url": url, "bytes": int(dest.stat().st_size), "status": status})
        print(f"{status} {name} {dest.stat().st_size}", flush=True)

    pd.DataFrame(rows).to_csv(out / "lincs_p100_lvl4_manifest.tsv", sep="\t", index=False)
    print(f"done files={len(rows)} out={out}", flush=True)


if __name__ == "__main__":
    main()
