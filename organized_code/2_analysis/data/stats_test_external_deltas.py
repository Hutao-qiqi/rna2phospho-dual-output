#!/usr/bin/env python3
"""Statistical tests for external predicted deltas.

Primary use-case: test whether predicted pEGFR-like (e.g., EGFRPY1068) decreases
under EGFR-TKI relative to vehicle, while total EGFR stays closer to unchanged.

This script reads:
  reports/external_validation/external_cellline_predicted_deltas.tsv

and writes:
  reports/external_validation/external_cellline_predicted_deltas_stats.tsv

Tests (per group):
- Wilcoxon signed-rank test of delta vs 0 (one-sided or two-sided as appropriate)
- Binomial sign test of delta direction vs 0 (ignoring zeros)
- Paired Wilcoxon test of (pSite_delta - EGFR_delta) vs 0 (one-sided)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def infer_drug(label: str) -> str:
    t = str(label).upper()
    # Keep this intentionally simple and conservative.
    for key in ["OSI", "OSIM", "ERL", "ERLOT", "GEF", "GEFIT", "WZ", "WZ4002", "AFA", "AFAT"]:
        if key in t:
            if key.startswith("OS"):  # OSI/OSIM/OSIMERTINIB
                return "OSI"
            if key.startswith("ERL"):
                return "ERL"
            if key.startswith("GEF"):
                return "GEF"
            if key.startswith("WZ"):
                return "WZ"
            if key.startswith("AFA"):
                return "AFA"
    return "UNK"


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg FDR adjustment (returns q-values in original order)."""
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    ranked = np.array(pvals, dtype=float)[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out.tolist()


def safe_wilcoxon(x: np.ndarray, *, alternative: str) -> tuple[float, int] | tuple[float, int]:
    """Return (pvalue, n_nonzero_used)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x_nz = x[x != 0]
    if x_nz.size == 0:
        return (np.nan, 0)
    # SciPy: use 'wilcox' zero method; deterministic.
    res = wilcoxon(x, zero_method="wilcox", alternative=alternative, mode="auto")
    return (float(res.pvalue), int(x_nz.size))


def sign_test_p(x: np.ndarray, *, alternative: str) -> tuple[float, int, int, int]:
    """Binomial sign test, ignoring zeros.

    Returns: (pvalue, n_eff, n_neg, n_pos)
    alternative: 'less' tests P(negative) > 0.5 (i.e., more negatives)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    neg = int(np.sum(x < 0))
    pos = int(np.sum(x > 0))
    n = neg + pos
    if n == 0:
        return (np.nan, 0, neg, pos)
    if alternative == "less":
        # more negatives than expected under p=0.5
        p = float(binomtest(neg, n, 0.5, alternative="greater").pvalue)
    elif alternative == "greater":
        p = float(binomtest(pos, n, 0.5, alternative="greater").pvalue)
    else:
        p = float(binomtest(neg, n, 0.5, alternative="two-sided").pvalue)
    return (p, n, neg, pos)


def summarize_vector(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "frac_neg": np.nan,
            "frac_pos": np.nan,
            "n_zero": 0,
        }
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "frac_neg": float(np.mean(x < 0)),
        "frac_pos": float(np.mean(x > 0)),
        "n_zero": int(np.sum(x == 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="in_tsv",
        default="reports/external_validation/external_cellline_predicted_deltas.tsv",
        help="Input per-pair deltas TSV",
    )
    ap.add_argument(
        "--out",
        dest="out_tsv",
        default="reports/external_validation/external_cellline_predicted_deltas_stats.tsv",
        help="Output stats TSV",
    )
    ap.add_argument("--gse", type=str, default="", help="If set, only analyze one GSE")
    ap.add_argument(
        "--psite",
        type=str,
        default="EGFRPY1068",
        help="Single phospho-site model name (legacy). Prefer --psites.",
    )
    ap.add_argument(
        "--psites",
        type=str,
        default="",
        help="Comma-separated phospho-site model names to sum (e.g., 'EGFRPY1068,EGFRPY1173').",
    )
    args = ap.parse_args()

    in_tsv = Path(args.in_tsv)
    out_tsv = Path(args.out_tsv)
    psite = str(args.psite).strip()
    psites = [p.strip() for p in str(args.psites).split(",") if p.strip()] if str(args.psites).strip() else []
    if not psites:
        psites = [psite]

    df = pd.read_csv(in_tsv, sep="\t")
    df = df.fillna("")

    # Only keep valid rows with numeric deltas.
    df = df[df["error"] == ""].copy()
    if args.gse:
        df = df[df["gse"] == args.gse].copy()

    need_cols = ["gse", "baseline_label", "perturbed_label", "pred_EGFR_delta"] + [f"pred_{p}_delta" for p in psites]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    for c in ["pred_EGFR_delta", "egfr_mrna_delta", "gene_coverage_frac"] + [f"pred_{p}_delta" for p in psites]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["drug"] = df["perturbed_label"].map(infer_drug)

    groups = []
    # Always include an overall group per GSE, plus per-drug subgroup for interpretability.
    for (gse, drug), g in df.groupby(["gse", "drug"], dropna=False):
        groups.append((str(gse), str(drug), g))
    for gse, g in df.groupby("gse", dropna=False):
        groups.append((str(gse), "ALL", g))

    rows: list[dict] = []
    raw_pvals: list[float] = []
    raw_pvals_key: list[tuple[int, str]] = []

    def add_row(row: dict, p_for_fdr: float | None):
        idx = len(rows)
        rows.append(row)
        if p_for_fdr is not None and np.isfinite(p_for_fdr):
            raw_pvals.append(float(p_for_fdr))
            raw_pvals_key.append((idx, "wilcoxon_p"))

    for gse, drug, g in groups:
        x_egfr = g["pred_EGFR_delta"].to_numpy(dtype=float)
        x_ps_by_site = {p: g[f"pred_{p}_delta"].to_numpy(dtype=float) for p in psites}
        x_ps_sum = np.zeros_like(x_egfr, dtype=float)
        for p in psites:
            x_ps_sum = x_ps_sum + x_ps_by_site[p]
        x_diff_sum = x_ps_sum - x_egfr

        # 1) Per-site delta < 0 (one-sided)
        for p in psites:
            x_ps = x_ps_by_site[p]
            s = summarize_vector(x_ps)
            w_p, w_n = safe_wilcoxon(x_ps, alternative="less")
            s_p, s_n, s_neg, s_pos = sign_test_p(x_ps, alternative="less")
            add_row(
                {
                    "gse": gse,
                    "group": drug,
                    "test": f"pred_{p}_delta < 0",
                    "n": s["n"],
                    "median": s["median"],
                    "mean": s["mean"],
                    "frac_neg": s["frac_neg"],
                    "wilcoxon_p": w_p,
                    "wilcoxon_n_nonzero": w_n,
                    "sign_p": s_p,
                    "sign_n_nonzero": s_n,
                    "sign_n_neg": s_neg,
                    "sign_n_pos": s_pos,
                },
                w_p,
            )

        # 1b) Sum of phospho-site deltas < 0
        psum_label = "+".join(psites)
        s = summarize_vector(x_ps_sum)
        w_p, w_n = safe_wilcoxon(x_ps_sum, alternative="less")
        s_p, s_n, s_neg, s_pos = sign_test_p(x_ps_sum, alternative="less")
        add_row(
            {
                "gse": gse,
                "group": drug,
                "test": f"pred_phospho_sum({psum_label})_delta < 0",
                "n": s["n"],
                "median": s["median"],
                "mean": s["mean"],
                "frac_neg": s["frac_neg"],
                "wilcoxon_p": w_p,
                "wilcoxon_n_nonzero": w_n,
                "sign_p": s_p,
                "sign_n_nonzero": s_n,
                "sign_n_neg": s_neg,
                "sign_n_pos": s_pos,
            },
            w_p,
        )

        # 2) EGFR delta != 0 (two-sided), because expectation is "no change".
        s = summarize_vector(x_egfr)
        w_p2, w_n2 = safe_wilcoxon(x_egfr, alternative="two-sided")
        s_p2, s_n2, s_neg2, s_pos2 = sign_test_p(x_egfr, alternative="two-sided")
        add_row(
            {
                "gse": gse,
                "group": drug,
                "test": "pred_EGFR_delta != 0",
                "n": s["n"],
                "median": s["median"],
                "mean": s["mean"],
                "frac_neg": s["frac_neg"],
                "wilcoxon_p": w_p2,
                "wilcoxon_n_nonzero": w_n2,
                "sign_p": s_p2,
                "sign_n_nonzero": s_n2,
                "sign_n_neg": s_neg2,
                "sign_n_pos": s_pos2,
            },
            w_p2,
        )

        # 3) Sum phospho decreases more than EGFR: (sum_phospho - EGFR) < 0 (one-sided)
        s = summarize_vector(x_diff_sum)
        w_p3, w_n3 = safe_wilcoxon(x_diff_sum, alternative="less")
        s_p3, s_n3, s_neg3, s_pos3 = sign_test_p(x_diff_sum, alternative="less")
        add_row(
            {
                "gse": gse,
                "group": drug,
                "test": f"(pred_phospho_sum({psum_label})_delta - pred_EGFR_delta) < 0",
                "n": s["n"],
                "median": s["median"],
                "mean": s["mean"],
                "frac_neg": s["frac_neg"],
                "wilcoxon_p": w_p3,
                "wilcoxon_n_nonzero": w_n3,
                "sign_p": s_p3,
                "sign_n_nonzero": s_n3,
                "sign_n_neg": s_neg3,
                "sign_n_pos": s_pos3,
            },
            w_p3,
        )

        # 4) Optionally report EGFR mRNA deltas (two-sided), to contextualize why predicted EGFR moves.
        if "egfr_mrna_delta" in g.columns:
            x_mrna = g["egfr_mrna_delta"].to_numpy(dtype=float)
            s = summarize_vector(x_mrna)
            w_pm, w_nm = safe_wilcoxon(x_mrna, alternative="two-sided")
            s_pm, s_nm, s_negm, s_posm = sign_test_p(x_mrna, alternative="two-sided")
            add_row(
                {
                    "gse": gse,
                    "group": drug,
                    "test": "egfr_mrna_delta != 0",
                    "n": s["n"],
                    "median": s["median"],
                    "mean": s["mean"],
                    "frac_neg": s["frac_neg"],
                    "wilcoxon_p": w_pm,
                    "wilcoxon_n_nonzero": w_nm,
                    "sign_p": s_pm,
                    "sign_n_nonzero": s_nm,
                    "sign_n_neg": s_negm,
                    "sign_n_pos": s_posm,
                },
                w_pm,
            )

    # Attach BH-FDR to wilcoxon p-values (across all rows in this run).
    qvals = bh_fdr(raw_pvals)
    for (idx, _), q in zip(raw_pvals_key, qvals, strict=True):
        rows[idx]["wilcoxon_q_bh"] = float(q)
    for r in rows:
        if "wilcoxon_q_bh" not in r:
            r["wilcoxon_q_bh"] = np.nan

    out = pd.DataFrame(rows)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_tsv, sep="\t", index=False)
    print(f"WROTE {out_tsv} rows={len(out)}")


if __name__ == "__main__":
    main()
