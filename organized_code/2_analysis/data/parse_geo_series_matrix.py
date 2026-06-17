#!/usr/bin/env python3
"""Parse GEO series_matrix files into sample metadata tables.

Input:
  data/raw/geo_supplementary/<GSE>/<GSE>_series_matrix.txt.gz

Outputs:
  reports/external_validation/series_matrix_samples.tsv
  reports/external_validation/series_matrix_characteristics_long.tsv

Notes:
- Only parses header metadata; ignores the expression table section.
- Extracts !Sample_* lines including repeated !Sample_characteristics_ch1.
"""

from __future__ import annotations

import csv
import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SampleRec:
    gse: str
    gsm: str
    title: str = ""
    source_name_ch1: str = ""
    organism_ch1: str = ""
    characteristics: list[str] = field(default_factory=list)


def strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def parse_series_matrix(path: Path) -> tuple[str, dict[str, SampleRec]]:
    gse = path.parent.name
    # SOFT files may hard-wrap long lines. Continuation lines do NOT start with '!'.
    # We merge continuation lines into the previous record before splitting fields.
    merged_lines: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        prev: str | None = None
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            if prev is None:
                prev = line
                continue
            if line.startswith("!"):
                merged_lines.append(prev)
                prev = line
            else:
                # Continuation: append as-is (no extra separator) to preserve quoted tokens.
                prev += line
        if prev is not None:
            merged_lines.append(prev)

    def split_fields(line: str) -> list[str]:
        if "\t" in line:
            return line.split("\t")
        # Fallback: SOFT sometimes uses multiple spaces as separators.
        return re.split(r"\s{2,}", line.strip())

    gsm_order: list[str] = []
    samples: dict[str, SampleRec] = {}
    pending_single: dict[str, list[str]] = {}
    pending_characteristics: list[list[str]] = []

    def apply_row(tag: str, values: list[str]) -> None:
        nonlocal gsm_order, samples
        if not gsm_order:
            return
        for gsm, val in zip(gsm_order, values):
            if gsm not in samples:
                samples[gsm] = SampleRec(gse=gse, gsm=gsm)
            rec = samples[gsm]
            if tag == "!Sample_title":
                rec.title = val
            elif tag == "!Sample_source_name_ch1":
                rec.source_name_ch1 = val
            elif tag == "!Sample_organism_ch1":
                rec.organism_ch1 = val
            elif tag.startswith("!Sample_characteristics_ch1"):
                if val:
                    rec.characteristics.append(val)

    for line in merged_lines:
        row = split_fields(line)
        if not row:
            continue
        tag = row[0]
        if tag == "!series_matrix_table_begin":
            break
        if not tag.startswith("!Sample_"):
            continue

        values = [strip_quotes(v) for v in row[1:]]

        if tag == "!Sample_geo_accession":
            gsm_order = values
            for gsm in gsm_order:
                if gsm and gsm not in samples:
                    samples[gsm] = SampleRec(gse=gse, gsm=gsm)
            # Apply any buffered sample-level rows that appeared before geo_accession.
            for ptag, pvals in pending_single.items():
                apply_row(ptag, pvals)
            for pvals in pending_characteristics:
                apply_row("!Sample_characteristics_ch1", pvals)
            pending_single.clear()
            pending_characteristics.clear()
            continue

        # Buffer or apply.
        if not gsm_order:
            if tag in {"!Sample_title", "!Sample_source_name_ch1", "!Sample_organism_ch1"}:
                pending_single[tag] = values
            elif tag.startswith("!Sample_characteristics_ch1"):
                pending_characteristics.append(values)
            continue

        apply_row(tag, values)

    return gse, samples


def parse_characteristics(characteristics: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in characteristics:
        if ":" in item:
            k, v = item.split(":", 1)
            out.append((k.strip(), v.strip()))
        else:
            out.append(("characteristic", item.strip()))
    return out


def main() -> None:
    inputs = sorted(Path("data/raw/geo_supplementary").glob("GSE*/GSE*_series_matrix.txt.gz"))
    out_dir = Path("reports/external_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_out = out_dir / "series_matrix_samples.tsv"
    long_out = out_dir / "series_matrix_characteristics_long.tsv"

    all_samples: list[SampleRec] = []
    long_rows: list[dict[str, str]] = []

    for path in inputs:
        gse, samples = parse_series_matrix(path)
        for gsm, rec in sorted(samples.items()):
            all_samples.append(rec)
            for k, v in parse_characteristics(rec.characteristics):
                long_rows.append({"gse": gse, "gsm": gsm, "key": k, "value": v})

    # Write samples table
    with samples_out.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["gse", "gsm", "title", "source_name_ch1", "organism_ch1", "characteristics"])
        for rec in sorted(all_samples, key=lambda r: (r.gse, r.gsm)):
            w.writerow(
                [
                    rec.gse,
                    rec.gsm,
                    rec.title,
                    rec.source_name_ch1,
                    rec.organism_ch1,
                    " | ".join(rec.characteristics),
                ]
            )

    # Write long characteristics
    with long_out.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["gse", "gsm", "key", "value"])
        for r in sorted(long_rows, key=lambda d: (d["gse"], d["gsm"], d["key"], d["value"])):
            w.writerow([r["gse"], r["gsm"], r["key"], r["value"]])

    print("WROTE", samples_out, "rows=", len(all_samples))
    print("WROTE", long_out, "rows=", len(long_rows))


if __name__ == "__main__":
    main()
