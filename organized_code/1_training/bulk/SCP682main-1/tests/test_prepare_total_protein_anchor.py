from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

from prepare_total_protein_anchor import prepare_anchor_package  # noqa: E402


def _inputs(tmp_path: Path):
    sample_ids = [f"S{i}" for i in range(1431)]
    manifest = tmp_path / "sample_manifest.tsv"
    pd.DataFrame({"sample_id": sample_ids}).to_csv(manifest, sep="\t", index=False)
    proteins = np.asarray(["A", "B", "C"])
    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    np.savez_compressed(
        train,
        prediction=np.ones((916, 3), dtype=np.float32),
        train_indices=np.arange(916),
        protein_names=proteins,
    )
    np.savez_compressed(
        validation,
        prediction=np.full((229, 3), 2.0, dtype=np.float32),
        validation_indices=np.arange(916, 1145),
        protein_names=proteins,
    )
    train_summary = tmp_path / "train_summary.json"
    train_summary.write_text(
        json.dumps(
            {
                "split_sizes": [916, 229, 286],
                "selection_validation_229_evaluated": False,
                "outer_test_286_evaluated": False,
                "cross_validation_scope": "selection_train_916_only",
                "study_calibration_refit_within_each_audit_fold": True,
            }
        ),
        encoding="utf-8",
    )
    validation_summary = tmp_path / "validation_summary.json"
    validation_summary.write_text(
        json.dumps(
            {
                "split_sizes": [916, 229, 286],
                "outer_test_evaluated": False,
                "calibration_fit_source": "selection_train_916_out_of_fold_predictions",
                "selected_by_locked_229_median_spearman": {"median_spearman_all": 0.62},
            }
        ),
        encoding="utf-8",
    )
    return manifest, train, validation, train_summary, validation_summary


def test_prepare_anchor_package_keeps_sealed_samples_out(tmp_path: Path):
    inputs = _inputs(tmp_path)
    output = tmp_path / "out"
    report = prepare_anchor_package(
        sample_manifest_path=inputs[0],
        train_prediction_path=inputs[1],
        validation_prediction_path=inputs[2],
        train_summary_path=inputs[3],
        validation_summary_path=inputs[4],
        output_dir=output,
    )
    prediction = pd.read_parquet(output / "total_protein_development_predictions.parquet")
    provenance = pd.read_csv(output / "total_protein_prediction_provenance.tsv", sep="\t")
    split = pd.read_csv(output / "locked_916_229_286.tsv", sep="\t")
    assert prediction.shape == (1145, 3)
    assert provenance.shape[0] == 1145
    assert (split["role"] == "sealed_test").sum() == 286
    assert set(split.loc[split["role"] == "sealed_test", "sample_id"]).isdisjoint(prediction.index)
    assert report["development_sealed_overlap"] == 0


def test_prepare_anchor_package_rejects_protein_order_change(tmp_path: Path):
    inputs = list(_inputs(tmp_path))
    np.savez_compressed(
        inputs[2],
        prediction=np.ones((229, 3), dtype=np.float32),
        validation_indices=np.arange(916, 1145),
        protein_names=np.asarray(["B", "A", "C"]),
    )
    with pytest.raises(ValueError, match="protein vocabularies"):
        prepare_anchor_package(
            sample_manifest_path=inputs[0],
            train_prediction_path=inputs[1],
            validation_prediction_path=inputs[2],
            train_summary_path=inputs[3],
            validation_summary_path=inputs[4],
            output_dir=tmp_path / "out",
        )
