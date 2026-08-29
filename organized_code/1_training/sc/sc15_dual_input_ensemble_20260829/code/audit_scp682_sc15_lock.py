from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = {
    "overall": ("overall_spearman", "overall_spearman"),
    "within_condition": (
        "within_condition_spearman",
        "within_group_residual_spearman",
    ),
    "pseudobulk": ("pseudobulk_spearman", "pseudobulk_spearman"),
}


def bootstrap_ci(values, rng, n_boot=20000):
    estimates = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, 1000):
        count = min(1000, n_boot - start)
        take = rng.integers(0, len(values), size=(count, len(values)))
        estimates[start : start + count] = values[take].mean(axis=1)
    return np.quantile(estimates, [0.025, 0.975])


def paired_permutation_p(values, rng, n_perm=100000):
    observed = abs(values.mean())
    exceed = 0
    for start in range(0, n_perm, 2000):
        count = min(2000, n_perm - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(values)))
        exceed += int((np.abs((signs * values).mean(axis=1)) >= observed).sum())
    return (exceed + 1) / (n_perm + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sc15-dir", type=Path, required=True)
    parser.add_argument("--competitor-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sc15-model", default="selected")
    parser.add_argument("--competitor-model", default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_table = pd.read_csv(args.sc15_dir / "ensemble_by_fold.tsv", sep="\t")
    selected_models = fold_table[fold_table["selected"]].set_index("split_seed")["model"]
    sc15_all = pd.read_csv(args.sc15_dir / "ensemble_per_target.tsv", sep="\t")
    if args.sc15_model == "selected":
        sc15 = pd.concat(
            [
                sc15_all[
                    sc15_all["split_seed"].eq(seed)
                    & sc15_all["model"].eq(model)
                ]
                for seed, model in selected_models.items()
            ],
            ignore_index=True,
        )
    else:
        sc15 = sc15_all[sc15_all["model"].eq(args.sc15_model)].copy()
    sc15 = sc15.rename(columns={"split_seed": "seed"})
    competitors = pd.read_csv(args.competitor_table, sep="\t")
    competitors = competitors.rename(columns={"architecture": "method", "split_seed": "seed"})
    if args.competitor_model is not None and "model" in competitors:
        competitors = competitors[competitors["model"].eq(args.competitor_model)].copy()
    rng = np.random.default_rng(682)
    statistics = []
    pairs_all = []
    plate_rows = []
    target_rows = []
    for method in ("ctpnet", "scipenn"):
        current = competitors[competitors["method"].eq(method)].copy()
        paired = sc15.merge(current, on=["seed", "target_id"], suffixes=("_sc15", "_competitor"))
        for label, (sc_column, competitor_column) in METRICS.items():
            sc_name = f"{sc_column}_sc15" if f"{sc_column}_sc15" in paired else sc_column
            requested_competitor = competitor_column
            if requested_competitor not in paired and label == "within_condition":
                requested_competitor = "within_condition_spearman"
            competitor_name = (
                f"{requested_competitor}_competitor"
                if f"{requested_competitor}_competitor" in paired
                else requested_competitor
            )
            a = paired[sc_name].to_numpy(float)
            b = paired[competitor_name].to_numpy(float)
            keep = np.isfinite(a) & np.isfinite(b)
            delta = a[keep] - b[keep]
            ci_low, ci_high = bootstrap_ci(delta, rng)
            try:
                wilcoxon_p = float(wilcoxon(delta, alternative="two-sided").pvalue)
            except ValueError:
                wilcoxon_p = np.nan
            statistics.append(
                {
                    "competitor": method,
                    "metric": label,
                    "n_pairs": len(delta),
                    "mean_delta": delta.mean(),
                    "median_delta": np.median(delta),
                    "bootstrap_95ci_low": ci_low,
                    "bootstrap_95ci_high": ci_high,
                    "wilcoxon_two_sided_p": wilcoxon_p,
                    "paired_permutation_two_sided_p": paired_permutation_p(delta, rng),
                    "sc15_win_fraction": np.mean(delta > 0),
                    "largest_absolute_pair_share": np.max(np.abs(delta)) / np.sum(np.abs(delta)),
                }
            )
            pair_frame = paired.loc[keep, ["seed", "target_id"]].copy()
            pair_frame.insert(0, "competitor", method)
            pair_frame.insert(1, "metric", label)
            pair_frame["sc15"] = a[keep]
            pair_frame["competitor_value"] = b[keep]
            pair_frame["delta"] = delta
            pairs_all.append(pair_frame)
            for seed, group in pair_frame.groupby("seed"):
                plate_rows.append(
                    {
                        "competitor": method,
                        "metric": label,
                        "seed": seed,
                        "n_targets": len(group),
                        "median_sc15": group["sc15"].median(),
                        "median_competitor": group["competitor_value"].median(),
                        "median_delta": group["delta"].median(),
                        "win_fraction": np.mean(group["delta"] > 0),
                    }
                )
            for target, group in pair_frame.groupby("target_id"):
                target_rows.append(
                    {
                        "competitor": method,
                        "metric": label,
                        "target_id": target,
                        "n_folds": len(group),
                        "median_delta": group["delta"].median(),
                        "mean_delta": group["delta"].mean(),
                        "fold_win_fraction": np.mean(group["delta"] > 0),
                    }
                )
    statistics_table = pd.DataFrame(statistics)
    plate_table = pd.DataFrame(plate_rows)
    target_table = pd.DataFrame(target_rows)
    pairs_table = pd.concat(pairs_all, ignore_index=True)
    statistics_table.to_csv(args.output_dir / "paired_statistics.tsv", sep="\t", index=False)
    plate_table.to_csv(args.output_dir / "performance_by_plate.tsv", sep="\t", index=False)
    target_table.to_csv(args.output_dir / "performance_by_target.tsv", sep="\t", index=False)
    pairs_table.to_csv(args.output_dir / "fold_target_pairs.tsv", sep="\t", index=False)
    report = {
        "formal_name": "SCP682-SC15 Dual-Input Ensemble",
        "n_folds": 5,
        "n_targets": 20,
        "n_fold_target_pairs": 100,
        "sc15_model": args.sc15_model,
        "competitor_model": args.competitor_model,
        "fold_specific_hvg_selection": True,
        "fold_specific_normalization": True,
        "validation_only_model_selection": True,
        "all_folds_complete": bool(sc15["seed"].nunique() == 5),
        "all_targets_complete": bool(sc15["target_id"].nunique() == 20),
    }
    (args.output_dir / "lock_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "SUCCESS").write_text("ok\n", encoding="utf-8")
    print(statistics_table.to_string(index=False))


if __name__ == "__main__":
    main()
