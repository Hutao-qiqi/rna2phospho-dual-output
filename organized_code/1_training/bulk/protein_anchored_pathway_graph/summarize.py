from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import discover_input_paths, per_site_spearman  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine fold-level strict-inductive predictions.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction_paths = sorted(args.result_dir.glob("fold_*/predictions/strict_inductive_oof.parquet"))
    baseline_paths = sorted(args.result_dir.glob("fold_*/predictions/parent_protein_baseline_oof.parquet"))
    if len(prediction_paths) != 5 or len(baseline_paths) != 5:
        raise RuntimeError(
            f"Expected five completed folds, found prediction={len(prediction_paths)}, baseline={len(baseline_paths)}"
        )
    prediction = pd.concat([pd.read_parquet(path) for path in prediction_paths], axis=0)
    baseline = pd.concat([pd.read_parquet(path) for path in baseline_paths], axis=0)
    if prediction.index.has_duplicates or baseline.index.has_duplicates:
        raise ValueError("Fold predictions contain duplicate sample identifiers")
    prediction = prediction.sort_index()
    baseline = baseline.reindex(prediction.index)
    observed = pd.read_parquet(discover_input_paths(args.project_root).phosphosite)
    observed = observed.loc[prediction.index, prediction.columns]
    sample_index = pd.RangeIndex(len(prediction)).to_numpy()
    model_metrics = per_site_spearman(
        observed.to_numpy(), prediction.to_numpy(), prediction.columns.tolist(), sample_index
    )
    baseline_metrics = per_site_spearman(
        observed.to_numpy(), baseline.to_numpy(), baseline.columns.tolist(), sample_index
    )
    output_dir = args.result_dir / "combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction.to_parquet(output_dir / "strict_inductive_oof.parquet")
    baseline.to_parquet(output_dir / "parent_protein_baseline_oof.parquet")
    model_metrics.to_csv(output_dir / "per_site_spearman.tsv", sep="\t", index=False)
    baseline_metrics.to_csv(output_dir / "parent_baseline_per_site_spearman.tsv", sep="\t", index=False)
    summary = {
        "n_samples": len(prediction),
        "n_sites": prediction.shape[1],
        "median_spearman": float(model_metrics["spearman"].median(skipna=True)),
        "parent_baseline_median_spearman": float(baseline_metrics["spearman"].median(skipna=True)),
        "evaluation": "five-fold strict inductive; test samples aggregate only training reference nodes",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
