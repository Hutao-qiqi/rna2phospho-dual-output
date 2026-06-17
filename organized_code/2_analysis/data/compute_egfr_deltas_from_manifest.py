#!/usr/bin/env python3
"""Compute EGFR mRNA deltas for paired samples in metadata/external_cellline_manifest.tsv.

This is an MVP sanity check: uses sample titles (from series_matrix) to map GSMs to
expression-matrix column names.

Output:
  reports/external_validation/external_cellline_egfr_deltas.tsv
"""

from __future__ import annotations

import csv
import gzip
import re
from pathlib import Path

import pandas as pd


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if str(path).lower().endswith(".gz") else open(path, "rt", encoding="utf-8", errors="replace")


def guess_delimiter(line: str) -> str:
    tabs = line.count("\t")
    commas = line.count(",")
    if tabs == 0 and commas == 0:
        return "\t"
    return "\t" if tabs >= commas else ","


def strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def norm_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(s)).lower()


def load_series_titles() -> dict[tuple[str, str], str]:
    # map (gse,gsm) -> title
    p = Path("reports/external_validation/series_matrix_samples.tsv")
    df = pd.read_csv(p, sep="\t", dtype=str).fillna("")
    return {(r["gse"], r["gsm"]): r["title"] for _, r in df.iterrows()}


def read_egfr_row(matrix_path: Path) -> tuple[list[str], list[str]] | None:
    with open_text(matrix_path) as fh:
        first = fh.readline()
        if not first:
            return None
        delim = guess_delimiter(first)
    with open_text(matrix_path) as fh2:
        reader = csv.reader(fh2, delimiter=delim)
        header = next(reader, None)
        if not header or len(header) < 2:
            return None
        cols = [strip_quotes(c) for c in header[1:]]
        for row in reader:
            if not row:
                continue
            gene = strip_quotes(row[0]).strip().upper()
            if gene == "EGFR" or gene.startswith("EGFR"):
                vals = [strip_quotes(x) for x in row[1 : 1 + len(cols)]]
                return cols, vals
    return None


def to_float(x: str) -> float | None:
    x = str(x).strip()
    if x == "" or x.lower() in {"na", "nan", "null"}:
        return None
    try:
        return float(x)
    except Exception:
        return None


def main() -> None:
    manifest = pd.read_csv("metadata/external_cellline_manifest.tsv", sep="\t", dtype=str).fillna("")
    titles = load_series_titles()

    out_rows = []
    for _, r in manifest.iterrows():
        gse = r["gse"]
        gsm0 = r["gsm_baseline"]
        gsm1 = r["gsm_perturbed"]
        expr = r["expression_matrix"]
        if not gse or not gsm0 or not gsm1 or not expr:
            continue
        matrix_path = Path(expr)
        if not matrix_path.exists():
            continue

        t0 = titles.get((gse, gsm0), "")
        t1 = titles.get((gse, gsm1), "")
        egfr = read_egfr_row(matrix_path)
        if egfr is None:
            continue
        cols, vals = egfr
        col_to_val = {c: v for c, v in zip(cols, vals)}

        def lookup(title: str) -> float | None:
            if title in col_to_val:
                return to_float(col_to_val.get(title, ""))
            nk = norm_key(title)
            if not nk:
                return None
            # Try normalized match
            norm_map = {}
            for c in col_to_val:
                norm_map.setdefault(norm_key(c), []).append(c)
            hits = norm_map.get(nk, [])
            if len(hits) == 1:
                return to_float(col_to_val.get(hits[0], ""))
            return None

        v0 = lookup(t0)
        v1 = lookup(t1)

        out_rows.append(
            {
                "gse": gse,
                "cell_line": r.get("cell_line", ""),
                "pairing_type": r.get("pairing_type", ""),
                "gsm_baseline": gsm0,
                "gsm_perturbed": gsm1,
                "baseline_title": t0,
                "perturbed_title": t1,
                "egfr_baseline": v0,
                "egfr_perturbed": v1,
                "delta_pert_minus_base": (v1 - v0) if (v0 is not None and v1 is not None) else None,
                "expression_matrix": str(matrix_path),
            }
        )

    out = Path("reports/external_validation/external_cellline_egfr_deltas.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out, sep="\t", index=False)
    print("WROTE", out, "rows=", len(out_rows))


if __name__ == "__main__":
    main()
