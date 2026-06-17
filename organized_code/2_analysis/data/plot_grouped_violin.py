#!/usr/bin/env python3
"""Grouped violin plots for external predicted values.

This script supports two modes:
1. --mode subline: Group by subline (x-axis) and hue by treatment. (Good for comparing responses within specific sublines)
2. --mode treatment: Group by treatment (x-axis) and pool all sublines. (Good for overall population distribution)

It calculates the 'phospho sum' (EGFRPY1068 + EGFRPY1173) and 'total EGFR' and plots them.
Statistical significance is calculated using paired Wilcoxon signed-rank tests on the deltas.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon


def parse_subline_and_treatment(label: str) -> tuple[str, str]:
    """Parse something like 'PC9_GR3_WZ.rep1' -> ('PC9_GR3', 'WZ')."""
    t = str(label).strip()
    if not t:
        return ("", "")
    # Remove .repX suffix
    t2 = re.sub(r"\brep\s*[0-9]+\b", "", t, flags=re.I)
    t2 = re.sub(r"[\s\-]+", "_", t2)
    t2 = re.sub(r"\.+", "_", t2)
    t2 = re.sub(r"_+", "_", t2).strip("_")

    parts = [p for p in t2.split("_") if p]
    if len(parts) < 2:
        return (t2, "")
    treatment = parts[-1].upper()
    subline = "_".join(parts[:-1]).upper()
    return (subline, treatment)


def safe_wilcoxon_p(x: np.ndarray, *, alternative: str) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x != 0]
    if x.size == 0:
        return float("nan")
    # wilcoxon requires at least one difference
    if x.size < 1:
        return float("nan")
    try:
        return float(wilcoxon(x, alternative=alternative, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="in_tsv",
        type=Path,
        default=Path("reports/external_validation/external_cellline_predicted_deltas.egfrphospho.v2.tsv"),
    )
    ap.add_argument("--gse", type=str, default="GSE75602")
    ap.add_argument(
        "--psites",
        type=str,
        default="EGFRPY1068,EGFRPY1173",
        help="Comma-separated phospho-site model names to sum",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["subline", "treatment"],
        default="treatment",
        help="Plotting mode: 'subline' (split by subline) or 'treatment' (pool all sublines by treatment)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. If not set, defaults to a name based on mode.",
    )
    args = ap.parse_args()

    # Determine default output if not provided
    if args.out is None:
        agg_str = "pooled" if args.mode == "treatment" else "grouped"
        args.out = Path(f"reports/external_validation/figures/{args.gse}.{agg_str}_violin.EGFR_phospho_sum.png")

    psites = [p.strip() for p in str(args.psites).split(",") if p.strip()]
    if not psites:
        raise SystemExit("--psites is empty")

    df = pd.read_csv(args.in_tsv, sep="\t").fillna("")
    df = df[(df["gse"] == args.gse) & (df["error"] == "")].copy()
    if df.empty:
        raise SystemExit(f"No valid rows for gse={args.gse} in {args.in_tsv}")

    # Ensure numeric
    cols_to_numeric = ["pred_EGFR_baseline", "pred_EGFR_perturbed", "pred_EGFR_delta"]
    for p in psites:
        cols_to_numeric += [f"pred_{p}_baseline", f"pred_{p}_perturbed", f"pred_{p}_delta"]
    for c in cols_to_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # --- Prepare Sample-Level Data (Long Format) ---
    # We reconstruct the absolute values from "baseline" and "perturbed" columns
    base = df[["baseline_label", "pred_EGFR_baseline"] + [f"pred_{p}_baseline" for p in psites]].copy()
    base = base.rename(columns={"baseline_label": "sample_label", "pred_EGFR_baseline": "pred_EGFR"})
    for p in psites:
        base = base.rename(columns={f"pred_{p}_baseline": f"pred_{p}"})
    base["stage"] = "BASELINE"

    pert = df[["perturbed_label", "pred_EGFR_perturbed"] + [f"pred_{p}_perturbed" for p in psites]].copy()
    pert = pert.rename(columns={"perturbed_label": "sample_label", "pred_EGFR_perturbed": "pred_EGFR"})
    for p in psites:
        pert = pert.rename(columns={f"pred_{p}_perturbed": f"pred_{p}"})
    pert["stage"] = "PERTURBED"

    samples = pd.concat([base, pert], ignore_index=True)
    samples["subline"], samples["treatment"] = zip(*samples["sample_label"].map(parse_subline_and_treatment))

    # Calculate Phospho Sum
    samples["pred_phospho_sum"] = 0.0
    for p in psites:
        samples["pred_phospho_sum"] = samples["pred_phospho_sum"] + pd.to_numeric(samples[f"pred_{p}"], errors="coerce")
    
    # Dedup: Identical samples might appear in multiple pairs (e.g. one control used for multiple treatments)
    # We want the distribution of UNIQUE samples.
    samples = samples.drop_duplicates(subset=["sample_label"])

    # Define Treatment Order
    # Standard order of likely drugs
    hue_order = ["VEH", "DMSO", "GEF", "WZ", "OSI", "ERL", "AFA"]
    present_treatments = [h for h in hue_order if h in set(samples["treatment"].tolist())]
    # Add any others found
    others = sorted([t for t in set(samples["treatment"].tolist()) if t not in hue_order])
    present_treatments += others

    # --- Prepare Stats Data (Delta/Pair Level) ---
    df["subline_base"], df["treat_base"] = zip(*df["baseline_label"].map(parse_subline_and_treatment))
    df["subline_pert"], df["treat_pert"] = zip(*df["perturbed_label"].map(parse_subline_and_treatment))
    
    # We only care about pairs where subline is consistent (paired within cell line)
    # Our manifest logic typically ensures this, but let's be safe
    df_pairs = df[df["subline_base"] == df["subline_pert"]].copy()
    df_pairs["subline"] = df_pairs["subline_base"]
    df_pairs["drug"] = df_pairs["treat_pert"] # The treatment applied
    
    # Calculate Sum Delta
    df_pairs["phospho_sum_delta"] = 0.0
    for p in psites:
        df_pairs["phospho_sum_delta"] = df_pairs["phospho_sum_delta"] + df_pairs[f"pred_{p}_delta"].fillna(0.0)

    # --- Configuration based on Mode ---
    
    if args.mode == "subline":
        # SPLIT BY SUBLINE
        x_col = "subline"
        hue_col = "treatment"
        x_order = sorted(set(samples["subline"].tolist()))
        hue_order_plot = present_treatments
        palette = None # default
        
        # Stats per (subline, drug) for Non-VEH drugs
        stats_lines_ph = []
        stats_lines_eg = []
        
        for subline in x_order:
            for drug in present_treatments:
                if drug in ["VEH", "DMSO"]: continue # Skip controls
                
                # Find pairs consistent with this subline and drug
                # Note: The 'control' in the pair logic is implied by the manifest generation (baseline=VEH usually)
                sub_pairs = df_pairs[(df_pairs["subline"] == subline) & (df_pairs["drug"] == drug)]
                
                if sub_pairs.empty: continue
                
                x_ph = sub_pairs["phospho_sum_delta"].to_numpy()
                x_eg = sub_pairs["pred_EGFR_delta"].to_numpy()
                
                p_ph = safe_wilcoxon_p(x_ph, alternative="less") # Expect decrease
                p_eg = safe_wilcoxon_p(x_eg, alternative="two-sided") # Expect no change
                
                n = len(x_ph)
                stats_lines_ph.append(f"{subline} {drug}: n={n}, p={format_p(p_ph)}")
                stats_lines_eg.append(f"{subline} {drug}: n={n}, p={format_p(p_eg)}")

    else:
        # POOLED / TREATMENT MODE
        # X-axis is Treatment. No hue.
        x_col = "treatment"
        hue_col = None
        x_order = present_treatments
        hue_order_plot = None
        palette = "viridis" # Just to make them look distinct
        
        # Global stats per Drug vs Control (paired delta)
        # We aggregate all pairs for a given drug across all sublines
        stats_lines_ph = []
        stats_lines_eg = []
        
        for drug in present_treatments:
            if drug in ["VEH", "DMSO"]: continue
            
            drug_pairs = df_pairs[df_pairs["drug"] == drug]
            if drug_pairs.empty: continue
            
            x_ph = drug_pairs["phospho_sum_delta"].to_numpy()
            x_eg = drug_pairs["pred_EGFR_delta"].to_numpy()
            
            p_ph = safe_wilcoxon_p(x_ph, alternative="less")
            p_eg = safe_wilcoxon_p(x_eg, alternative="two-sided")
            
            n = len(x_ph)
            stats_lines_ph.append(f"Global {drug} (vs Ctrl): N={n}, p={format_p(p_ph)}")
            stats_lines_eg.append(f"Global {drug} (vs Ctrl): N={n}, p={format_p(p_eg)}")


    # --- Plotting ---
    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 1, figsize=(8, 12), dpi=150) # Taller for stacked panels

    def plot_panel(ax, y_col, title, stats_text_lines):
        # VIOLIN
        sns.violinplot(
            data=samples,
            x=x_col,
            y=y_col,
            hue=hue_col,
            order=x_order,
            hue_order=hue_order_plot,
            palette=palette,
            cut=0, # Do not extend past data range
            scale="width", # Even width for all violins
            linewidth=1.2,
            inner=None, # We will add stripplot
            ax=ax
        )
        
        # STRIPPLOT (Points)
        sns.stripplot(
            data=samples,
            x=x_col,
            y=y_col,
            hue=hue_col,
            order=x_order,
            hue_order=hue_order_plot,
            dodge=True if hue_col else False,
            color="black", # Make points black
            alpha=0.6,
            size=5,
            jitter=True,
            ax=ax
        )
        
        # Cleaning up logic for hue vs no-hue
        if hue_col is None:
             # If no hue, stripplot might have colored points if we didn't force color='black'
             # but we did. 
             pass
        else:
             # Remove legend from stripplot/violin to avoid duplication, handle global legend later if needed
             if ax.get_legend():
                 ax.get_legend().remove()

        ax.set_title(title, pad=20)
        ax.set_ylabel("Predicted Value")
        ax.set_xlabel("")
        
        # Add stats text box
        if stats_text_lines:
            stats_str = "Wilcoxon (Paired Δ):\n" + "\n".join(stats_text_lines)
            ax.text(
                1.02, 1.0, stats_str,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=10,
                bbox=dict(boxstyle="round, pad=0.5", fc="white", ec="gray", alpha=0.9)
            )

    # Panel 1: Phospho Sum
    plot_panel(
        axes[0], 
        "pred_phospho_sum", 
        f"EGFR Phosphorylation (Sum {len(psites)} sites)\n{args.gse}", 
        stats_lines_ph
    )
    
    # Panel 2: Total EGFR
    plot_panel(
        axes[1], 
        "pred_EGFR", 
        f"Total EGFR Expression\n{args.gse}", 
        stats_lines_eg
    )

    # Handle Legend if needed (only for 'subline' mode where hue is used)
    if args.mode == "subline":
        handles, labels = axes[0].get_legend_handles_labels()
        # Filter duplicates if any
        if handles:
             fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(present_treatments))

    plt.tight_layout()
    # Adjust for legend or text box space
    plt.subplots_adjust(right=0.65) # Make room for the side stats box
    
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"WROTE plot to: {args.out}")

if __name__ == "__main__":
    main()
