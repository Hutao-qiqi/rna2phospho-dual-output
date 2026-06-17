#!/usr/bin/env python3
"""Create violin plots for external predicted deltas with statistical annotations.

Default: for GSE75602, plot
- phospho_sum_delta = pred_EGFRPY1068_delta + pred_EGFRPY1173_delta
- pred_EGFR_delta

Outputs two PNGs under reports/external_validation/figures/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon


def wilcoxon_p(x: np.ndarray, *, alternative: str) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x != 0]
    if x.size == 0:
        return float("nan")
    return float(wilcoxon(x, alternative=alternative, zero_method="wilcox").pvalue)


def make_violin(
    values: np.ndarray,
    *,
    title: str,
    subtitle: str,
    ylabel: str,
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sns.set(style="whitegrid", context="talk")

    df = pd.DataFrame({"delta": values})
    df = df[np.isfinite(df["delta"])].copy()

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    sns.violinplot(data=df, y="delta", inner=None, cut=0, color="#7aa6c2", ax=ax)
    sns.stripplot(data=df, y="delta", color="black", alpha=0.65, size=5, jitter=0.18, ax=ax)
    ax.axhline(0.0, color="#b00020", linestyle="--", linewidth=1.6)

    ax.set_title(title, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.text(
        0.02,
        0.98,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="in_tsv",
        type=Path,
        default=Path("reports/external_validation/external_cellline_predicted_deltas.egfrphospho.v2.tsv"),
        help="Input deltas TSV (must contain EGFR and phospho-site delta columns)",
    )
    ap.add_argument("--gse", type=str, default="GSE75602")
    ap.add_argument(
        "--psites",
        type=str,
        default="EGFRPY1068,EGFRPY1173",
        help="Comma-separated phospho-site model names to sum",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/external_validation/figures"),
        help="Output directory for figures",
    )
    args = ap.parse_args()

    psites = [p.strip() for p in str(args.psites).split(",") if p.strip()]
    if not psites:
        raise SystemExit("--psites is empty")

    df = pd.read_csv(args.in_tsv, sep="\t").fillna("")
    df = df[(df["gse"] == args.gse) & (df["error"] == "")].copy()
    if df.empty:
        raise SystemExit(f"No valid rows for gse={args.gse} in {args.in_tsv}")

    for c in ["pred_EGFR_delta"] + [f"pred_{p}_delta" for p in psites]:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["phospho_sum_delta"] = 0.0
    for p in psites:
        df["phospho_sum_delta"] = df["phospho_sum_delta"] + df[f"pred_{p}_delta"]

    x_sum = df["phospho_sum_delta"].to_numpy(dtype=float)
    x_egfr = df["pred_EGFR_delta"].to_numpy(dtype=float)

    p_sum = wilcoxon_p(x_sum, alternative="less")
    p_egfr = wilcoxon_p(x_egfr, alternative="two-sided")

    sum_label = "+".join(psites)
    sub_sum = f"Wilcoxon signed-rank: H1 delta < 0\nN={int(np.sum(np.isfinite(x_sum)))}  median={np.nanmedian(x_sum):.3g}  p={p_sum:.3g}"
    sub_egfr = f"Wilcoxon signed-rank: H1 delta != 0\nN={int(np.sum(np.isfinite(x_egfr)))}  median={np.nanmedian(x_egfr):.3g}  p={p_egfr:.3g}"

    make_violin(
        x_sum,
        title=f"{args.gse}: Predicted EGFR phospho sum Δ",
        subtitle=f"Sum = {sum_label}\n{sub_sum}",
        ylabel="Δ (treated − baseline)",
        out_png=args.out_dir / f"{args.gse}.phospho_sum_{sum_label}.violin.png",
    )

    make_violin(
        x_egfr,
        title=f"{args.gse}: Predicted total EGFR Δ",
        subtitle=sub_egfr,
        ylabel="Δ (treated − baseline)",
        out_png=args.out_dir / f"{args.gse}.EGFR.violin.png",
    )

    print("WROTE figures to", args.out_dir)


if __name__ == "__main__":
    main()
