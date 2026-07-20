from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CODE = Path(__file__).parents[1] / "code"
sys.path.insert(0, str(CODE))

from axial_dynamic_data import (  # noqa: E402
    build_biological_prior,
    build_query_reference_knn,
    fit_parent_calibration,
    masked_row_median_center,
    validate_protein_prediction_provenance,
    validate_split_manifest,
)


def write_split(tmp_path: Path) -> Path:
    path = tmp_path / "split.tsv"
    pd.DataFrame(
        {
            "sample_id": ["T0", "T1", "V0", "S0"],
            "role": ["selection_train", "selection_train", "selection_validation", "sealed_test"],
        }
    ).to_csv(path, sep="\t", index=False)
    return path


def test_split_and_total_protein_provenance_contract(tmp_path):
    split = validate_split_manifest(write_split(tmp_path), expected_sizes=(2, 1, 1))
    provenance = tmp_path / "protein.tsv"
    pd.DataFrame(
        {
            "sample_id": ["T0", "T1", "V0"],
            "prediction_role": ["cross_fitted", "cross_fitted", "selection_train_only"],
            "phosphosite_labels_used": [False, False, False],
        }
    ).to_csv(provenance, sep="\t", index=False)
    table = validate_protein_prediction_provenance(provenance, split)
    assert set(table["sample_id"]) == {"T0", "T1", "V0"}


def test_provenance_rejects_sealed_or_training_fitted_predictions(tmp_path):
    split = validate_split_manifest(write_split(tmp_path), expected_sizes=(2, 1, 1))
    provenance = tmp_path / "bad.tsv"
    pd.DataFrame(
        {
            "sample_id": ["T0", "T1", "S0"],
            "prediction_role": ["selection_train_only", "cross_fitted", "selection_train_only"],
            "phosphosite_labels_used": [False, False, False],
        }
    ).to_csv(provenance, sep="\t", index=False)
    with pytest.raises(ValueError):
        validate_protein_prediction_provenance(provenance, split)


def test_parent_calibration_uses_only_declared_training_rows():
    protein = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    site = np.asarray([[1.0], [3.0], [5.0], [7.0]], dtype=np.float32)
    first = fit_parent_calibration(
        protein,
        site,
        np.asarray([0]),
        np.asarray([True]),
        np.asarray([0, 1]),
        ridge=0.0,
        minimum_observations=2,
    )
    changed = site.copy()
    changed[2:] = 10000.0
    second = fit_parent_calibration(
        protein,
        changed,
        np.asarray([0]),
        np.asarray([True]),
        np.asarray([0, 1]),
        ridge=0.0,
        minimum_observations=2,
    )
    assert np.allclose(first[0], second[0])
    assert np.allclose(first[1], second[1])


def test_candidate_graph_excludes_the_same_training_sample():
    features = np.eye(4, dtype=np.float32)
    index, _ = build_query_reference_knn(
        features,
        features,
        2,
        query_ids=["A", "B", "C", "D"],
        reference_ids=["A", "B", "C", "D"],
    )
    assert all(row not in index[row].tolist() for row in range(4))


def test_masked_centering_ignores_missing_sites():
    values = np.asarray([[1.0, 3.0, 99.0, 5.0]], dtype=np.float32)
    mask = np.asarray([[True, True, False, True]])
    centered, offset = masked_row_median_center(values, mask)
    assert np.allclose(offset, [3.0])
    assert np.allclose(centered[0, mask[0]], [-2.0, 0.0, 2.0])


def test_site_pathway_mapping_uses_full_gene_set():
    prior = build_biological_prior(
        {"GLOBAL_CONTEXT": ["A", "B"], "SIGNAL": ["A", "B"]},
        {"GLOBAL_CONTEXT": ["A", "B"], "SIGNAL": ["A", "B", "PARENT"]},
        ["A", "B"],
        ["PARENT"],
        ["PARENT|S10"],
        ["PARENT"],
        {},
        np.asarray([2]),
        2,
        max_rna_members=2,
        max_protein_members=2,
        max_site_pathways=2,
        max_site_kinases=2,
    )
    assert prior.site_pathway_mask[0, 0]
    assert prior.site_pathway_index[0, 0] == 1
