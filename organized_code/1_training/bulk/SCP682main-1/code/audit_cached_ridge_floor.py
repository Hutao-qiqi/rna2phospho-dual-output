"""Audit a cached ridge floor on the locked development validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "base_snapshot" / "code"
for path in (HERE, BASE):
    sys.path.insert(0, str(path))

from axial_dynamic_data import (
    apply_study_site_standardization,
    fit_parent_calibration,
    fit_study_site_standardization,
    parent_baseline,
    per_site_metrics,
    read_parquet_rows,
    read_prediction_matrix,
    read_sample_studies,
    validate_split_manifest,
)
from chunked_centering import fit_training_residual_scale, full_panel_center_prediction
from train_decoder_retrieval import _fixed_target, _targets


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "protein-prediction", "protein-reliability", "phosphosite",
        "phosphosite-manifest", "split-manifest", "sample-metadata",
        "ridge-cache", "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--study-column", default="pdc_study_id")
    args = parser.parse_args()

    split = validate_split_manifest(args.split_manifest)
    development_ids = split.development_ids.tolist()
    n_train = len(split.train_ids)
    train = np.arange(n_train, dtype=np.int64)
    validation = np.arange(n_train, len(development_ids), dtype=np.int64)
    studies = read_sample_studies(
        args.sample_metadata, development_ids,
        study_column=args.study_column, sample_id_column=args.sample_id_column,
    )
    protein_frame = read_prediction_matrix(args.protein_prediction, split)
    reliability_frame = read_parquet_rows(
        args.protein_reliability, development_ids,
        columns=protein_frame.columns.astype(str).tolist(),
    )
    manifest = pd.read_csv(args.phosphosite_manifest, sep="\t")
    targets, parents = _targets(manifest)
    phosphosite_frame = read_parquet_rows(args.phosphosite, development_ids, columns=targets)
    phosphosite_raw = phosphosite_frame.to_numpy(np.float32)
    study_fit = fit_study_site_standardization(
        phosphosite_raw, studies, train, minimum_observations=8
    )
    phosphosite, observed = apply_study_site_standardization(
        phosphosite_raw, studies, study_fit
    )
    protein_raw = protein_frame.to_numpy(np.float32)
    reliability = reliability_frame.to_numpy(np.float32)
    protein_vocabulary = {str(value).upper(): index for index, value in enumerate(protein_frame.columns)}
    parent_index = np.asarray([protein_vocabulary.get(parent, 0) for parent in parents], np.int64)
    parent_mask = np.asarray([parent in protein_vocabulary for parent in parents], bool)
    intercept, slope, _ = fit_parent_calibration(
        protein_raw, phosphosite, parent_index, parent_mask, train,
        ridge=1.0, minimum_observations=8,
    )
    unadjusted = parent_baseline(protein_raw, intercept, slope, parent_index, parent_mask)
    site_reliability = reliability[:, parent_index] * parent_mask[None, :]
    baseline = intercept[None, :] + site_reliability * (unadjusted - intercept[None, :])
    coverage = observed[train].mean(0)
    eligible = np.flatnonzero(coverage >= 0.9)
    panel = eligible[np.argsort(-coverage[eligible])[:1000]]
    residual, model_target, _, _ = _fixed_target(
        phosphosite, observed, baseline, panel, train
    )
    residual_scale = fit_training_residual_scale(
        residual, observed, train, minimum_observations=8
    )
    with np.load(args.ridge_cache, allow_pickle=False) as archive:
        ridge_standardized = np.asarray(archive["prediction"], np.float32)
    ridge_residual, _ = full_panel_center_prediction(
        ridge_standardized[validation] * residual_scale[None, :], panel
    )
    ridge_prediction = baseline[validation] + ridge_residual
    metrics = per_site_metrics(
        model_target[validation], ridge_prediction, observed[validation], targets
    )
    summary = {
        "validation_ridge_median_spearman": float(
            pd.to_numeric(metrics["spearman"], errors="coerce").median()
        ),
        "validation_ridge_median_pearson": float(
            pd.to_numeric(metrics["pearson"], errors="coerce").median()
        ),
        "validation_ridge_median_mse": float(
            pd.to_numeric(metrics["mse"], errors="coerce").median()
        ),
        "sealed_phosphosite_rows_loaded": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "validation_ridge_per_site.tsv", sep="\t", index=False)
    (args.output_dir / "ridge_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
