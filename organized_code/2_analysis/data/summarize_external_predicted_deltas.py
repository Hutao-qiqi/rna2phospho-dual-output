#!/usr/bin/env python3
"""Summarize external predicted delta results.

Reads:
  reports/external_validation/external_cellline_predicted_deltas.tsv

Writes:
  reports/external_validation/external_cellline_predicted_deltas_summary.tsv

The summary is grouped by (gse, pairing_type, cell_line) and reports:
- n_pairs
- mean/median delta for EGFR mRNA and predicted proteins
- counts of positive/negative/zero deltas

Intended as a lightweight QC + reporting helper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_signed(series: pd.Series) -> dict[str, float | int]:
    s = pd.to_numeric(series, errors="coerce")
    out: dict[str, float | int] = {
        "n": int(s.notna().sum()),
        "mean": float(s.mean()) if s.notna().any() else np.nan,
        "median": float(s.median()) if s.notna().any() else np.nan,
        "n_pos": int((s > 0).sum()),
        "n_neg": int((s < 0).sum()),
        "n_zero": int((s == 0).sum()),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="inp",
        type=Path,
        default=Path("reports/external_validation/external_cellline_predicted_deltas.tsv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("reports/external_validation/external_cellline_predicted_deltas_summary.tsv"),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.inp, sep="\t")
    df["error"] = df.get("error", "").fillna("")
    ok = df[df["error"] == ""].copy()

    if ok.empty:
        raise SystemExit(f"No valid rows (error=='') in {args.inp}")

    group_cols = [c for c in ["gse", "pairing_type", "cell_line"] if c in ok.columns]

    metric_cols = {
        "egfr_mrna_delta": "egfr_mrna",
        "pred_EGFR_delta": "pred_EGFR",
        "pred_EGFRPY1068_delta": "pred_EGFRPY1068",
    }

    rows: list[dict[str, object]] = []
    for keys, g in ok.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec: dict[str, object] = dict(zip(group_cols, keys))
        rec["n_pairs"] = int(len(g))

        for col, prefix in metric_cols.items():
            if col not in g.columns:
                continue
            s = summarize_signed(g[col])
            rec[f"{prefix}_n"] = s["n"]
            rec[f"{prefix}_mean"] = s["mean"]
            rec[f"{prefix}_median"] = s["median"]
            rec[f"{prefix}_n_pos"] = s["n_pos"]
            rec[f"{prefix}_n_neg"] = s["n_neg"]
            rec[f"{prefix}_n_zero"] = s["n_zero"]

        rows.append(rec)

    out = pd.DataFrame(rows).sort_values(group_cols + ["n_pairs"], ascending=[True] * len(group_cols) + [False])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    print("WROTE", args.out, "rows=", len(out))


if __name__ == "__main__":
    main()
