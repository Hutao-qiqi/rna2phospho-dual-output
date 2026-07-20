"""Package locked total-protein predictions for phosphosite training.

The input prediction files are immutable results of the total-protein audit.
This program only validates, labels, and combines them.  Sealed samples are
recorded in the split table and are never written to the prediction matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_SIZES = (916, 229, 286)
TRAIN_ROLE = "selection_train"
VALIDATION_ROLE = "selection_validation"
SEALED_ROLE = "sealed_test"


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _load_prediction(
    path: str | Path,
    *,
    prediction_key: str,
    index_key: str,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        required = {prediction_key, index_key, "protein_names"}
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"prediction archive lacks keys: {sorted(missing)}")
        prediction = np.asarray(saved[prediction_key], dtype=np.float32)
        indices = np.asarray(saved[index_key], dtype=np.int64)
        proteins = np.asarray(saved["protein_names"]).astype(str)
    if prediction.ndim != 2 or prediction.shape[0] != expected_rows:
        raise ValueError(
            f"prediction row count differs: observed={prediction.shape}, expected_rows={expected_rows}"
        )
    if indices.shape != (expected_rows,) or np.unique(indices).size != expected_rows:
        raise ValueError("prediction indices are not a unique row vector")
    if proteins.shape != (prediction.shape[1],) or np.unique(proteins).size != proteins.size:
        raise ValueError("protein vocabulary differs from the prediction columns")
    if not np.isfinite(prediction).all():
        raise ValueError("prediction matrix contains missing or non-finite values")
    return prediction, indices, proteins


def _read_summary(path: str | Path, required_false: tuple[str, ...]) -> dict[str, Any]:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in required_false:
        if summary.get(key) is not False:
            raise ValueError(f"source summary does not seal {key}")
    if tuple(summary.get("split_sizes", ())) != EXPECTED_SIZES:
        raise ValueError("source summary split sizes differ from 916/229/286")
    return summary


def prepare_anchor_package(
    *,
    sample_manifest_path: str | Path,
    train_prediction_path: str | Path,
    validation_prediction_path: str | Path,
    train_summary_path: str | Path,
    validation_summary_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    sample_manifest_path = Path(sample_manifest_path)
    train_prediction_path = Path(train_prediction_path)
    validation_prediction_path = Path(validation_prediction_path)
    train_summary_path = Path(train_summary_path)
    validation_summary_path = Path(validation_summary_path)
    output_dir = Path(output_dir)

    manifest = pd.read_csv(sample_manifest_path, sep="\t")
    if "sample_id" not in manifest.columns:
        raise ValueError("sample manifest lacks sample_id")
    sample_ids = manifest["sample_id"].astype(str).to_numpy()
    if sample_ids.size != sum(EXPECTED_SIZES) or np.unique(sample_ids).size != sample_ids.size:
        raise ValueError("sample manifest must contain 1,431 unique samples")

    train_prediction, train_indices, train_proteins = _load_prediction(
        train_prediction_path,
        prediction_key="prediction",
        index_key="train_indices",
        expected_rows=EXPECTED_SIZES[0],
    )
    validation_prediction, validation_indices, validation_proteins = _load_prediction(
        validation_prediction_path,
        prediction_key="prediction",
        index_key="validation_indices",
        expected_rows=EXPECTED_SIZES[1],
    )
    if not np.array_equal(train_proteins, validation_proteins):
        raise ValueError("training and validation protein vocabularies differ")
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("training and validation indices overlap")
    development_indices = np.concatenate([train_indices, validation_indices])
    if development_indices.min() < 0 or development_indices.max() >= sample_ids.size:
        raise IndexError("prediction indices exceed the sample manifest")
    sealed_indices = np.setdiff1d(
        np.arange(sample_ids.size, dtype=np.int64), development_indices, assume_unique=False
    )
    if sealed_indices.size != EXPECTED_SIZES[2]:
        raise ValueError("sealed complement does not contain 286 samples")

    train_summary = _read_summary(
        train_summary_path,
        required_false=("selection_validation_229_evaluated", "outer_test_286_evaluated"),
    )
    validation_summary = _read_summary(
        validation_summary_path,
        required_false=("outer_test_evaluated",),
    )
    if train_summary.get("cross_validation_scope") != "selection_train_916_only":
        raise ValueError("training prediction summary has an unexpected scope")
    if train_summary.get("study_calibration_refit_within_each_audit_fold") is not True:
        raise ValueError("training study calibration is not cross-fitted")
    if validation_summary.get("calibration_fit_source") != "selection_train_916_out_of_fold_predictions":
        raise ValueError("validation calibration source is not the 916-sample training fold")

    split_roles = np.full(sample_ids.size, SEALED_ROLE, dtype=object)
    split_roles[train_indices] = TRAIN_ROLE
    split_roles[validation_indices] = VALIDATION_ROLE
    split = pd.DataFrame({"sample_id": sample_ids, "role": split_roles})

    development_ids = np.concatenate([sample_ids[train_indices], sample_ids[validation_indices]])
    combined = np.concatenate([train_prediction, validation_prediction], axis=0)
    prediction_frame = pd.DataFrame(combined, index=development_ids, columns=train_proteins)
    prediction_frame.index.name = "sample_id"
    provenance = pd.DataFrame(
        {
            "sample_id": development_ids,
            "prediction_role": np.concatenate(
                [
                    np.repeat("cross_fitted", EXPECTED_SIZES[0]),
                    np.repeat("selection_train_only", EXPECTED_SIZES[1]),
                ]
            ),
            "phosphosite_labels_used": False,
            "study_calibration_cross_fitted": np.concatenate(
                [
                    np.repeat(True, EXPECTED_SIZES[0]),
                    np.repeat(False, EXPECTED_SIZES[1]),
                ]
            ),
            "upstream_hyperparameters_nested_within_fold": False,
            "source_archive": np.concatenate(
                [
                    np.repeat(str(train_prediction_path), EXPECTED_SIZES[0]),
                    np.repeat(str(validation_prediction_path), EXPECTED_SIZES[1]),
                ]
            ),
            "source_row_index": np.concatenate([train_indices, validation_indices]),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_output = output_dir / "total_protein_development_predictions.parquet"
    provenance_output = output_dir / "total_protein_prediction_provenance.tsv"
    split_output = output_dir / "locked_916_229_286.tsv"
    prediction_frame.to_parquet(prediction_output, compression="zstd", index=True)
    provenance.to_csv(provenance_output, sep="\t", index=False)
    split.to_csv(split_output, sep="\t", index=False)

    report = {
        "status": "complete",
        "split_sizes": {
            TRAIN_ROLE: int((split_roles == TRAIN_ROLE).sum()),
            VALIDATION_ROLE: int((split_roles == VALIDATION_ROLE).sum()),
            SEALED_ROLE: int((split_roles == SEALED_ROLE).sum()),
        },
        "prediction_shape": list(prediction_frame.shape),
        "prediction_finite": bool(np.isfinite(combined).all()),
        "protein_order_equal": True,
        "development_sealed_overlap": int(
            np.intersect1d(development_indices, sealed_indices).size
        ),
        "training_prediction_scope": {
            "upstream_predictions": train_summary.get("upstream_predictions"),
            "study_calibration_refit_within_each_fold": train_summary.get(
                "study_calibration_refit_within_each_audit_fold"
            ),
            "upstream_hyperparameters_refit_within_each_fold": train_summary.get(
                "upstream_hyperparameters_refit_within_each_audit_fold"
            ),
        },
        "selected_validation_model": validation_summary.get(
            "selected_by_locked_229_median_spearman"
        ),
        "source_files": {
            "sample_manifest": {
                "path": str(sample_manifest_path),
                "sha256": sha256_file(sample_manifest_path),
            },
            "train_crossfit_prediction": {
                "path": str(train_prediction_path),
                "sha256": sha256_file(train_prediction_path),
            },
            "validation_prediction": {
                "path": str(validation_prediction_path),
                "sha256": sha256_file(validation_prediction_path),
            },
            "train_summary": {
                "path": str(train_summary_path),
                "sha256": sha256_file(train_summary_path),
            },
            "validation_summary": {
                "path": str(validation_summary_path),
                "sha256": sha256_file(validation_summary_path),
            },
        },
        "outputs": {},
    }
    for name, path in {
        "prediction": prediction_output,
        "provenance": provenance_output,
        "split": split_output,
    }.items():
        report["outputs"][name] = {"path": str(path), "sha256": sha256_file(path)}
    report_path = output_dir / "anchor_package_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--train-prediction", type=Path, required=True)
    parser.add_argument("--validation-prediction", type=Path, required=True)
    parser.add_argument("--train-summary", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = prepare_anchor_package(
        sample_manifest_path=args.sample_manifest,
        train_prediction_path=args.train_prediction,
        validation_prediction_path=args.validation_prediction,
        train_summary_path=args.train_summary,
        validation_summary_path=args.validation_summary,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
