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
    apply_study_site_standardization,
    fit_study_site_standardization,
    fit_parent_calibration,
    load_site_kinases,
    masked_row_median_center,
    read_sample_studies,
    read_parquet_rows,
    select_pathways,
    validate_protein_prediction_provenance,
    validate_case_split_disjointness,
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
            "source_model_id": ["fold0", "fold1", "train_full"],
            "source_fold": ["0", "1", "selection_train"],
            "sample_in_source_training": [False, False, False],
            "source_archive": ["a.npz", "a.npz", "b.npz"],
            "source_row_index": [0, 1, 0],
        }
    ).to_csv(provenance, sep="\t", index=False)
    table = validate_protein_prediction_provenance(provenance, split)
    assert set(table["sample_id"]) == {"T0", "T1", "V0"}


def test_single_global_pathway_selection_is_explicitly_supported(tmp_path):
    genes = [f"G{index}" for index in range(8)]
    hallmark = tmp_path / "hallmark.gmt"
    canonical = tmp_path / "canonical.gmt"
    hallmark.write_text(
        "HALLMARK_SIGNAL\tna\t" + "\t".join(genes) + "\n", encoding="utf-8"
    )
    canonical.write_text(
        "CANONICAL_SIGNAL\tna\t" + "\t".join(genes) + "\n", encoding="utf-8"
    )
    selection = select_pathways(
        hallmark,
        canonical,
        genes,
        ["G0"],
        np.arange(len(genes), dtype=np.float32),
        max_pathways=1,
        max_members=8,
        minimum_members=8,
    )
    assert list(selection.members) == ["GLOBAL_CONTEXT"]


def test_provenance_rejects_sealed_or_training_fitted_predictions(tmp_path):
    split = validate_split_manifest(write_split(tmp_path), expected_sizes=(2, 1, 1))
    provenance = tmp_path / "bad.tsv"
    pd.DataFrame(
        {
            "sample_id": ["T0", "T1", "S0"],
            "prediction_role": ["selection_train_only", "cross_fitted", "selection_train_only"],
            "phosphosite_labels_used": [False, False, False],
            "source_model_id": ["bad", "fold1", "train_full"],
            "source_fold": ["all", "1", "selection_train"],
            "sample_in_source_training": [True, False, False],
            "source_archive": ["a.npz", "a.npz", "b.npz"],
            "source_row_index": [0, 1, 0],
        }
    ).to_csv(provenance, sep="\t", index=False)
    with pytest.raises(ValueError):
        validate_protein_prediction_provenance(provenance, split)


def test_legacy_crossfit_provenance_is_explicitly_downgraded(tmp_path):
    split = validate_split_manifest(write_split(tmp_path), expected_sizes=(2, 1, 1))
    provenance = tmp_path / "legacy.tsv"
    pd.DataFrame(
        {
            "sample_id": ["T0", "T1", "V0"],
            "prediction_role": ["cross_fitted", "cross_fitted", "selection_train_only"],
            "phosphosite_labels_used": [False, False, False],
            "source_archive": ["a.npz", "a.npz", "b.npz"],
            "source_row_index": [0, 1, 0],
        }
    ).to_csv(provenance, sep="\t", index=False)
    table = validate_protein_prediction_provenance(provenance, split)
    assert table.attrs["crossfit_evidence_level"] == "audited_archive_and_role_only"


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


def test_candidate_graph_excludes_self_when_k_reaches_reference_count():
    features = np.eye(4, dtype=np.float32)
    index, _ = build_query_reference_knn(
        features,
        features,
        99,
        query_ids=["A", "B", "C", "D"],
        reference_ids=["A", "B", "C", "D"],
    )
    assert index.shape == (4, 3)
    assert all(row not in index[row].tolist() for row in range(4))


def test_study_site_standardization_uses_training_rows_only():
    values = np.asarray(
        [[1.0, 10.0], [3.0, 14.0], [100.0, 1000.0]], dtype=np.float32
    )
    studies = np.asarray(["A", "A", "A"])
    fitted = fit_study_site_standardization(values, studies, np.asarray([0, 1]))
    changed = values.copy()
    changed[2] = -10000.0
    refitted = fit_study_site_standardization(changed, studies, np.asarray([0, 1]))
    assert np.allclose(fitted.mean, refitted.mean)
    assert np.allclose(fitted.scale, refitted.scale)
    transformed, support = apply_study_site_standardization(values, studies, fitted)
    assert support.all()
    assert np.allclose(transformed[:2].mean(axis=0), 0.0)


def test_missing_study_metadata_fails_without_fallback(tmp_path):
    path = tmp_path / "metadata.tsv"
    pd.DataFrame({"aliquot": ["A"], "study": ["S1"]}).to_csv(
        path, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="lacks"):
        read_sample_studies(
            path,
            ["A", "B"],
            study_column="study",
            sample_id_column="aliquot",
        )


def test_sealed_phosphosite_rows_are_not_requested(monkeypatch, tmp_path):
    captured = {}

    def fake_read_parquet(path, *, columns, filters, engine):
        captured["filters"] = filters
        return pd.DataFrame({"SITE": [1.0, 2.0]}, index=["T0", "V0"])

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    result = read_parquet_rows(tmp_path / "labels.parquet", ["T0", "V0"], columns=["SITE"])
    assert result.index.tolist() == ["T0", "V0"]
    assert "S0" not in captured["filters"][0][2]


def test_case_identifier_cannot_cross_locked_roles(tmp_path):
    split = validate_split_manifest(write_split(tmp_path), expected_sizes=(2, 1, 1))
    metadata = tmp_path / "metadata.tsv"
    pd.DataFrame(
        {
            "aliquot": ["T0", "T1", "V0", "S0"],
            "case_submitter_id": ["C0", "C1", "C0", "C2"],
        }
    ).to_csv(metadata, sep="\t", index=False)
    with pytest.raises(ValueError, match="overlap"):
        validate_case_split_disjointness(
            metadata,
            split,
            sample_id_column="aliquot",
        )


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


def test_kinase_loader_supports_copheeksa_and_substrate_gene_site(tmp_path):
    copheeksa = tmp_path / "copheeksa.tsv"
    pd.DataFrame(
        {"gene_site_id": ["AAK1|S624"], "kinase": ["CDK8"]}
    ).to_csv(copheeksa, sep="\t", index=False)
    substrate = tmp_path / "substrate.tsv"
    pd.DataFrame(
        {
            "kinase_gene": ["AAK1", "MAPK1"],
            "substrate_gene": ["ATP1A3", "AAK1"],
            "substrate_site": ["T705", "S624"],
        }
    ).to_csv(substrate, sep="\t", index=False)
    mapping = load_site_kinases(
        [copheeksa, substrate], ["AAK1|S624", "ATP1A3|T705"]
    )
    assert mapping["AAK1|S624"] == {"CDK8", "MAPK1"}
    assert mapping["ATP1A3|T705"] == {"AAK1"}


def test_kinase_loader_supports_parent_gene_and_site(tmp_path):
    prior_path = tmp_path / "site_prior.tsv"
    pd.DataFrame(
        {"kinase": ["AKT1"], "parent_gene": ["EIF4EBP1"], "site": ["T37"]}
    ).to_csv(prior_path, sep="\t", index=False)
    mapping = load_site_kinases([prior_path], ["EIF4EBP1|T37"])
    assert mapping["EIF4EBP1|T37"] == {"AKT1"}
