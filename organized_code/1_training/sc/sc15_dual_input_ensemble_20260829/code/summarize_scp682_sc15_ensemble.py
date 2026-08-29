from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from train_scp682_sc15_direct import metrics


SPLITS = {
    69001: (79011, 79012),
    69002: (79021, 79022),
    69003: (79031, 79032),
    69004: (79041, 79042),
    69005: (79051, 79052),
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


def load_pair(direct_root, ensemble_root, split_seed, training_seeds, split):
    original_name = (
        f"{split}_predictions_complete.npz" if split in {"validation", "test"} else ""
    )
    original = np.load(direct_root / f"A2_seed{split_seed}" / original_name, allow_pickle=False)
    predictions = [original["prediction"].astype(np.float32)]
    truth = original["truth"].astype(np.float32)
    rows = original["dataset_cell_index"].astype(np.int64)
    for training_seed in training_seeds:
        name = "validation_predictions.npz" if split == "validation" else "test_predictions.npz"
        cache = np.load(
            ensemble_root / f"split_{split_seed}" / f"seed_{training_seed}" / name,
            allow_pickle=False,
        )
        if not np.array_equal(rows, cache["dataset_cell_index"].astype(np.int64)):
            raise RuntimeError(f"Row mismatch: {split_seed} {training_seed} {split}")
        predictions.append(cache["prediction"].astype(np.float32))
    return rows, truth, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.contract_dir / "full20_panel.tsv", sep="\t")
    targets = panel["target_id"].astype(str).tolist()
    metadata = pd.read_csv(args.metadata, sep="\t", low_memory=False).set_index(
        "dataset_cell_index"
    )
    rows = []
    per_target = []
    for split_seed, training_seeds in SPLITS.items():
        validation_rows, validation_truth, validation_predictions = load_pair(
            args.direct_root,
            args.ensemble_root,
            split_seed,
            training_seeds,
            "validation",
        )
        test_rows, test_truth, test_predictions = load_pair(
            args.direct_root,
            args.ensemble_root,
            split_seed,
            training_seeds,
            "test",
        )
        candidates_validation = {
            "original": validation_predictions[0],
            "member2": validation_predictions[1],
            "member3": validation_predictions[2],
            "mean3": np.mean(validation_predictions, axis=0),
            "rank3": rank_ensemble(validation_predictions),
        }
        candidates_test = {
            "original": test_predictions[0],
            "member2": test_predictions[1],
            "member3": test_predictions[2],
            "mean3": np.mean(test_predictions, axis=0),
            "rank3": rank_ensemble(test_predictions),
        }
        validation_groups = metadata.loc[validation_rows, "condition_group"].astype(str).to_numpy()
        test_groups = metadata.loc[test_rows, "condition_group"].astype(str).to_numpy()
        validation_scores = {}
        for name, prediction in candidates_validation.items():
            summary, _ = metrics(prediction, validation_truth, validation_groups, targets)
            validation_scores[name] = summary["median_overall_spearman"]
        selectable = {name: validation_scores[name] for name in ("original", "mean3", "rank3")}
        winner = max(selectable, key=selectable.get)
        if validation_scores[winner] < validation_scores["original"] + 0.005:
            winner = "original"
        for name, prediction in candidates_test.items():
            summary, table = metrics(prediction, test_truth, test_groups, targets)
            rows.append(
                {
                    "split_seed": split_seed,
                    "model": name,
                    "selected": name == winner,
                    "validation_spearman": validation_scores[name],
                    **summary,
                }
            )
            table.insert(0, "split_seed", split_seed)
            table.insert(1, "model", name)
            per_target.append(table)
        np.savez_compressed(
            args.output_dir / f"split_{split_seed}_selected_predictions.npz",
            dataset_cell_index=test_rows,
            prediction=candidates_test[winner],
            truth=test_truth,
            selected_model=np.asarray(winner),
        )
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "ensemble_by_fold.tsv", sep="\t", index=False)
    pd.concat(per_target, ignore_index=True).to_csv(
        args.output_dir / "ensemble_per_target.tsv", sep="\t", index=False
    )
    summary_rows = []
    for model, group in table.groupby("model"):
        summary_rows.append(
            {
                "model": model,
                "mean_overall_spearman": group["median_overall_spearman"].mean(),
                "median_overall_spearman": group["median_overall_spearman"].median(),
                "mean_within_condition_spearman": group["median_within_condition_spearman"].mean(),
                "median_within_condition_spearman": group["median_within_condition_spearman"].median(),
                "mean_pseudobulk_spearman": group["median_pseudobulk_spearman"].mean(),
                "median_pseudobulk_spearman": group["median_pseudobulk_spearman"].median(),
            }
        )
    selected = table[table["selected"]]
    summary_rows.append(
        {
            "model": "validation_selected",
            "mean_overall_spearman": selected["median_overall_spearman"].mean(),
            "median_overall_spearman": selected["median_overall_spearman"].median(),
            "mean_within_condition_spearman": selected["median_within_condition_spearman"].mean(),
            "median_within_condition_spearman": selected["median_within_condition_spearman"].median(),
            "mean_pseudobulk_spearman": selected["median_pseudobulk_spearman"].mean(),
            "median_pseudobulk_spearman": selected["median_pseudobulk_spearman"].median(),
        }
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "ensemble_summary.tsv", sep="\t", index=False)
    (args.output_dir / "SUCCESS").write_text("ok\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
