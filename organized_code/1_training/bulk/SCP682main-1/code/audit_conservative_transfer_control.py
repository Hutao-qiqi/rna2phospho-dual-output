"""Compare the paired fixed-229 conservative-transfer experiment arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--phosphosite-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def interval(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def main() -> int:
    args = arguments()
    names = ("cptac916_control", "combined2067")
    roots = {name: args.result_root / name for name in names}
    for root in roots.values():
        if not (root / "SUCCESS").is_file():
            raise FileNotFoundError(f"incomplete experiment arm: {root}")

    summaries = {
        name: json.loads((root / "reports" / "run_summary.json").read_text("utf-8"))
        for name, root in roots.items()
    }
    sample = {
        name: pd.read_csv(
            root / "tables" / "validation_per_sample_best.tsv", sep="\t"
        ).set_index("sample_id")
        for name, root in roots.items()
    }
    site = {
        name: pd.read_csv(
            root / "tables" / "validation_per_site_best.tsv", sep="\t"
        ).set_index("target")
        for name, root in roots.items()
    }
    if not sample[names[0]].index.equals(sample[names[1]].index):
        raise ValueError("validation patient order differs between arms")
    if not site[names[0]].index.equals(site[names[1]].index):
        raise ValueError("validation site order differs between arms")

    paired_sample = pd.DataFrame(index=sample[names[0]].index)
    for metric in ("cosine", "within_sample_spearman", "mse", "mae"):
        paired_sample[f"control_{metric}"] = sample[names[0]][metric]
        paired_sample[f"combined_{metric}"] = sample[names[1]][metric]
        paired_sample[f"delta_{metric}"] = (
            sample[names[1]][metric] - sample[names[0]][metric]
        )
    paired_sample.reset_index().to_csv(
        args.result_root / "paired_per_sample.tsv", sep="\t", index=False
    )

    manifest = pd.read_csv(args.phosphosite_manifest, sep="\t")
    target_column = next(
        name
        for name in ("scp682_site_id", "gene_site", "gene_site_id")
        if name in manifest
    )
    parent_column = next(
        name for name in ("parent_gene", "total_protein_gene") if name in manifest
    )
    parent = manifest.set_index(target_column)[parent_column].astype(str)
    paired_site = pd.DataFrame(index=site[names[0]].index)
    paired_site["parent_gene"] = parent.reindex(paired_site.index)
    for metric in (
        "spearman",
        "pearson",
        "mse",
        "prediction_to_target_sd_ratio",
    ):
        paired_site[f"control_{metric}"] = site[names[0]][metric]
        paired_site[f"combined_{metric}"] = site[names[1]][metric]
        paired_site[f"delta_{metric}"] = site[names[1]][metric] - site[names[0]][metric]
    paired_site.reset_index().to_csv(
        args.result_root / "paired_per_site.tsv", sep="\t", index=False
    )

    rng = np.random.default_rng(args.seed)
    sample_draws = {
        metric: np.empty(args.bootstrap_replicates, dtype=np.float64)
        for metric in ("cosine", "within_sample_spearman")
    }
    for replicate in range(args.bootstrap_replicates):
        rows = rng.integers(0, len(paired_sample), size=len(paired_sample))
        for metric in sample_draws:
            sample_draws[metric][replicate] = float(
                paired_sample[f"delta_{metric}"].iloc[rows].mean()
            )

    parent_groups = [
        group.index.to_numpy(np.int64)
        for _, group in paired_site.reset_index(drop=True).groupby(
            "parent_gene", dropna=False
        )
    ]
    site_draws = np.empty(args.bootstrap_replicates, dtype=np.float64)
    site_delta = paired_site["delta_spearman"].to_numpy(np.float64)
    for replicate in range(args.bootstrap_replicates):
        selected_groups = rng.integers(
            0, len(parent_groups), size=len(parent_groups)
        )
        rows = np.concatenate([parent_groups[index] for index in selected_groups])
        site_draws[replicate] = float(np.nanmedian(site_delta[rows]))

    report = {
        "status": "complete",
        "fixed_validation_patients": int(len(paired_sample)),
        "sealed_labels_loaded": False,
        "arms": summaries,
        "paired_point_estimates": {
            "batch8_flattened_cosine": (
                summaries[names[1]]["best_validation_batch8_flattened_cosine"]
                - summaries[names[0]]["best_validation_batch8_flattened_cosine"]
            ),
            "mean_per_sample_cosine": float(
                paired_sample["delta_cosine"].mean()
            ),
            "median_per_sample_spearman": float(
                sample[names[1]]["within_sample_spearman"].median()
                - sample[names[0]]["within_sample_spearman"].median()
            ),
            "median_per_site_spearman": float(
                site[names[1]]["spearman"].median()
                - site[names[0]]["spearman"].median()
            ),
            "fraction_samples_cosine_improved": float(
                (paired_sample["delta_cosine"] > 0).mean()
            ),
            "fraction_sites_spearman_improved": float(
                (paired_site["delta_spearman"] > 0).mean()
            ),
        },
        "paired_bootstrap": {
            "mean_per_sample_cosine_delta": interval(sample_draws["cosine"]),
            "mean_per_sample_spearman_delta": interval(
                sample_draws["within_sample_spearman"]
            ),
            "parent_cluster_median_site_spearman_delta": interval(site_draws),
        },
    }
    (args.result_root / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
