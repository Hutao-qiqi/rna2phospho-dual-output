from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from train_scp682_sc15_direct import metrics


EXTRAS = {
    "ctpnet": {
        69001: (88011, 88012),
        69002: (88021, 88022),
        69003: (88031, 88032),
        69004: (88041, 88042),
        69005: (88051, 88052),
    },
    "scipenn": {
        69001: (89011, 89012),
        69002: (89021, 89022),
        69003: (89031, 89032),
        69004: (89041, 89042),
        69005: (89051, 89052),
    },
}


def rank_ensemble(predictions):
    values = []
    for prediction in predictions:
        ranked = np.column_stack(
            [rankdata(prediction[:, j], method="average") for j in range(prediction.shape[1])]
        )
        quantile = (ranked - 0.5) / len(prediction)
        values.append(norm.ppf(np.clip(quantile, 1e-5, 1 - 1e-5)))
    return np.mean(values, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--extra-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(args.contract_dir / "full20_panel.tsv", sep="\t")[
        "target_id"
    ].astype(str).tolist()
    metadata = pd.read_csv(args.metadata, sep="\t", low_memory=False).set_index(
        "dataset_cell_index"
    )
    rows = []
    per_target = []
    for architecture, split_map in EXTRAS.items():
        for split_seed, training_seeds in split_map.items():
            original = np.load(
                args.original_root
                / architecture
                / "predictions"
                / f"seed_{split_seed}_test_predictions.npz",
                allow_pickle=False,
            )
            cell_index = original["dataset_cell_index"].astype(np.int64)
            truth = original["truth_native"].astype(np.float32)
            predictions = [original["prediction_native"].astype(np.float32)]
            for training_seed in training_seeds:
                current = np.load(
                    args.extra_root
                    / architecture
                    / f"split_{split_seed}"
                    / f"seed_{training_seed}"
                    / "predictions"
                    / f"seed_{training_seed}_test_predictions.npz",
                    allow_pickle=False,
                )
                if not np.array_equal(cell_index, current["dataset_cell_index"].astype(np.int64)):
                    raise RuntimeError(f"Cell mismatch: {architecture} {split_seed}")
                predictions.append(current["prediction_native"].astype(np.float32))
            candidates = {
                "single_original": predictions[0],
                "mean3": np.mean(predictions, axis=0),
                "rank3": rank_ensemble(predictions),
            }
            groups = metadata.loc[cell_index, "condition_group"].astype(str).to_numpy()
            for model, prediction in candidates.items():
                summary, target_table = metrics(prediction, truth, groups, targets)
                rows.append(
                    {
                        "architecture": architecture,
                        "model": model,
                        "split_seed": split_seed,
                        **summary,
                    }
                )
                target_table.insert(0, "architecture", architecture)
                target_table.insert(1, "model", model)
                target_table.insert(2, "split_seed", split_seed)
                per_target.append(target_table)
    by_fold = pd.DataFrame(rows)
    by_fold.to_csv(args.output_dir / "equal_budget_by_fold.tsv", sep="\t", index=False)
    pd.concat(per_target, ignore_index=True).to_csv(
        args.output_dir / "equal_budget_per_target.tsv", sep="\t", index=False
    )
    summary = (
        by_fold.groupby(["architecture", "model"], as_index=False)
        .agg(
            mean_overall_spearman=("median_overall_spearman", "mean"),
            median_overall_spearman=("median_overall_spearman", "median"),
            mean_within_condition_spearman=("median_within_condition_spearman", "mean"),
            median_within_condition_spearman=("median_within_condition_spearman", "median"),
            mean_pseudobulk_spearman=("median_pseudobulk_spearman", "mean"),
            median_pseudobulk_spearman=("median_pseudobulk_spearman", "median"),
        )
    )
    summary.to_csv(args.output_dir / "equal_budget_summary.tsv", sep="\t", index=False)
    (args.output_dir / "SUCCESS").write_text("ok\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
