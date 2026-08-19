"""Configuration, split-isolation and reporting helpers for the experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from decoder_retrieval_model import CONFIGURATIONS
from reference_retrieval import REFERENCE_COUNT_BINS


def configuration_flags(name: str) -> dict[str, bool]:
    if name not in CONFIGURATIONS:
        raise ValueError(f"unknown configuration {name!r}")
    hgt, reference = CONFIGURATIONS[name]
    return {"hgt_enabled": hgt, "reference_enabled": reference}


def padded_kinase_mapping(
    kinase_site_edge_index: np.ndarray,
    site_index: np.ndarray,
    *,
    maximum_kinases: int,
) -> tuple[np.ndarray, np.ndarray]:
    edge = np.asarray(kinase_site_edge_index, dtype=np.int64)
    sites = np.asarray(site_index, dtype=np.int64)
    output = np.zeros((sites.size, maximum_kinases), dtype=np.int64)
    mask = np.zeros_like(output, dtype=bool)
    for local, global_site in enumerate(sites.tolist()):
        kinases = np.unique(edge[0, edge[1] == global_site])
        if kinases.size > maximum_kinases:
            raise ValueError(
                "kinase capacity would truncate prior edges; increase maximum_kinases"
            )
        output[local, : kinases.size] = kinases
        mask[local, : kinases.size] = True
    return output, mask


def assert_development_only_phosphosite_rows(
    requested_sample_ids: list[str],
    train_ids: list[str],
    validation_ids: list[str],
    sealed_ids: list[str],
) -> None:
    requested = set(map(str, requested_sample_ids))
    allowed = set(map(str, train_ids)) | set(map(str, validation_ids))
    sealed = set(map(str, sealed_ids))
    if requested & sealed:
        raise ValueError("sealed phosphosite labels were requested")
    if requested != allowed:
        raise ValueError("phosphosite loader must request exactly train and validation rows")


def validation_metric_strata(
    metrics: pd.DataFrame,
    coverage: np.ndarray,
    reference_effective_sample_size: np.ndarray | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = metrics.copy()
    table["training_coverage"] = np.asarray(coverage, dtype=float)
    table["coverage_quartile"] = pd.qcut(
        table["training_coverage"], 4, labels=False, duplicates="drop"
    )
    numeric = [name for name in ("spearman", "pearson", "mse", "prediction_to_target_sd_ratio") if name in table]
    coverage_summary = table.groupby("coverage_quartile", dropna=False)[numeric].median().reset_index()
    if reference_effective_sample_size is None:
        reference_summary = pd.DataFrame(
            columns=["effective_reference_group", "n_sites", *numeric]
        )
    else:
        median_count = np.median(np.asarray(reference_effective_sample_size), axis=0)
        labels = np.full(median_count.shape, "outside", dtype=object)
        for low, high in REFERENCE_COUNT_BINS:
            labels[(median_count >= low) & (median_count <= high)] = f"{low}-{high}"
        table["effective_reference_group"] = labels
        reference_summary = table.groupby("effective_reference_group", dropna=False)[numeric].median()
        reference_summary.insert(0, "n_sites", table.groupby("effective_reference_group").size())
        reference_summary = reference_summary.reset_index()
    return coverage_summary, reference_summary


def write_run_contract(
    path: Path,
    *,
    configuration: str,
    arguments: dict[str, object],
    parameter_count: int,
) -> None:
    flags = configuration_flags(configuration)
    payload = {
        "architecture_status": "experimental_unvalidated",
        "configuration": configuration,
        **flags,
        "parameter_count": int(parameter_count),
        "maximum_sites_per_forward": 1000,
        "sealed_phosphosite_rows_loaded": False,
        "reference_role": "selection_train_only" if flags["reference_enabled"] else None,
        "arguments": arguments,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
