#!/usr/bin/env python3
"""Case study: extract EGFR expression and resistant-vs-parental deltas for GSE156054.

Inputs (downloaded already):
- data/raw/geo_supplementary/GSE156054/GSE156054_Raw_gene_counts_TPM.txt.gz

Outputs:
- reports/external_validation/gse156054_egfr_values.tsv
- reports/external_validation/gse156054_egfr_deltas.tsv

Notes:
- This dataset contains parental NSCLC lines and EGFR-TKI-selected resistant derivatives.
- Values are TPM (as provided by the submitter).
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

INPUT = Path("data/raw/geo_supplementary/GSE156054/GSE156054_Raw_gene_counts_TPM.txt.gz")
OUT_VALUES = Path("reports/external_validation/gse156054_egfr_values.tsv")
OUT_DELTAS = Path("reports/external_validation/gse156054_egfr_deltas.tsv")


def read_egfr_row(path: Path) -> tuple[list[str], list[str]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        # File is tab-delimited.
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        for row in reader:
            if not row:
                continue
            gene = row[0].strip().upper()
            if gene == "EGFR":
                return header, row
    raise RuntimeError("EGFR row not found")


def to_float(x: str) -> float | None:
    x = (x or "").strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def main() -> None:
    OUT_VALUES.parent.mkdir(parents=True, exist_ok=True)

    header, row = read_egfr_row(INPUT)
    samples = header[1:]
    values = row[1 : 1 + len(samples)]

    # Values table
    with OUT_VALUES.open("w", encoding="utf-8") as fo:
        fo.write("sample\tvalue\n")
        for sample, value in zip(samples, values):
            fo.write(f"{sample}\t{value}\n")

    # Simple pairing rules for this specific dataset
    pairs = [
        ("HCC827", "HCC827GR"),
        ("PC9", "PC9GR"),
        ("H1975", "H1975OR"),
    ]
    sample_to_value = {s: to_float(v) for s, v in zip(samples, values)}

    with OUT_DELTAS.open("w", encoding="utf-8") as fo:
        fo.write("parental\tresistant\tvalue_parental\tvalue_resistant\tdelta_res_minus_parent\n")
        for parental, resistant in pairs:
            vp = sample_to_value.get(parental)
            vr = sample_to_value.get(resistant)
            delta = (vr - vp) if (vp is not None and vr is not None) else None
            fo.write(
                f"{parental}\t{resistant}\t"
                f"{'' if vp is None else vp}\t{'' if vr is None else vr}\t{'' if delta is None else delta}\n"
            )


if __name__ == "__main__":
    main()
