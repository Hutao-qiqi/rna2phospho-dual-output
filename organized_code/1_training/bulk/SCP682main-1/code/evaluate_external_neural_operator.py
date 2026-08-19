"""Evaluate a frozen site-scaled neural operator on external cohorts.

External phosphosite labels are transformed with the training-site min-max
parameters stored in the checkpoint. No model parameter, threshold,
calibration coefficient, or input transform is fitted from an external label.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "base_snapshot" / "code"
for path in (HERE, BASE):
    sys.path.insert(0, str(path))

from axial_dynamic_data import (  # noqa: E402
    apply_feature_zscore,
    read_parquet_rows,
    sample_rank_encode,
    sha256_file,
    validate_split_manifest,
)
from chunk_manifest import build_site_chunks  # noqa: E402
from cophee_prior_bundle import load_cophee_prior_bundle  # noqa: E402
from decoder_retrieval_model import (  # noqa: E402
    DecoderRetrievalConfig,
    DecoderRetrievalModel,
)
from experiment_contract import padded_kinase_mapping  # noqa: E402
from train_decoder_retrieval import (  # noqa: E402
    apply_site_profile_scaler,
    load_esm2_site_prior,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--external-h5", type=Path, required=True)
    parser.add_argument("--external-protein-anchor", type=Path, required=True)
    parser.add_argument("--training-rna", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cophee-prior-bundle", type=Path, required=True)
    parser.add_argument("--esm2-prior-npz", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--site-chunk-size", type=int, default=1000)
    parser.add_argument("--minimum-site-observations", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def decode_text(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in values
        ],
        dtype=str,
    )


def observed_row_ranks(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Use the same observed-only RNA rank transform as the development export."""
    output = np.full(values.shape, 0.5, dtype=np.float32)
    for row in range(values.shape[0]):
        use = np.flatnonzero(observed[row])
        if not len(use):
            continue
        ranks = rankdata(values[row, use], method="average")
        output[row, use] = ((ranks - 1.0) / max(len(use) - 1, 1)).astype(
            np.float32
        )
    return output


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return math.nan
    return float(np.dot(left, right) / denominator)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3 or np.ptp(left) <= 0 or np.ptp(right) <= 0:
        return math.nan
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def sample_table(
    sample_ids: np.ndarray,
    cohorts: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row, sample_id in enumerate(sample_ids):
        valid = mask[row] & np.isfinite(target[row]) & np.isfinite(prediction[row])
        truth = target[row, valid].astype(np.float64)
        estimate = prediction[row, valid].astype(np.float64)
        baseline = template[valid].astype(np.float64)
        model_cosine = cosine(truth, estimate)
        template_cosine = cosine(truth, baseline)
        truth_residual = truth - baseline
        prediction_residual = estimate - baseline
        rows.append(
            {
                "sample_id": sample_id,
                "cohort": cohorts[row],
                "observed_site_count": int(valid.sum()),
                "model_cosine": model_cosine,
                "template_cosine": template_cosine,
                "cosine_gain_over_template": model_cosine - template_cosine,
                "model_spearman": spearman(truth, estimate),
                "template_spearman": spearman(truth, baseline),
                "spearman_gain_over_template": spearman(truth, estimate)
                - spearman(truth, baseline),
                "residual_cosine_after_training_template": cosine(
                    truth_residual, prediction_residual
                ),
                "mse": float(np.mean(np.square(estimate - truth))),
                "mae": float(np.mean(np.abs(estimate - truth))),
            }
        )
    return pd.DataFrame(rows)


def batch8_cosine(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    batch_size: int = 8,
) -> tuple[float, int]:
    values: list[float] = []
    for start in range(0, len(target), batch_size):
        stop = min(start + batch_size, len(target))
        valid = mask[start:stop]
        truth = target[start:stop][valid].astype(np.float64)
        estimate = prediction[start:stop][valid].astype(np.float64)
        value = cosine(truth, estimate)
        if np.isfinite(value):
            values.append(value)
    return (float(np.mean(values)) if values else math.nan, len(values))


def site_table(
    targets: np.ndarray,
    cohorts: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    minimum_observations: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cohort_levels = ["ALL", *sorted(set(cohorts.tolist()))]
    for cohort in cohort_levels:
        sample_mask = np.ones(len(cohorts), dtype=bool) if cohort == "ALL" else cohorts == cohort
        for column, site_id in enumerate(targets):
            valid = sample_mask & mask[:, column]
            count = int(valid.sum())
            if count < minimum_observations:
                continue
            truth = target[valid, column].astype(np.float64)
            estimate = prediction[valid, column].astype(np.float64)
            rows.append(
                {
                    "cohort": cohort,
                    "target": site_id,
                    "observed_count": count,
                    "spearman": spearman(truth, estimate),
                    "pearson": (
                        float(np.corrcoef(truth, estimate)[0, 1])
                        if np.ptp(truth) > 0 and np.ptp(estimate) > 0
                        else math.nan
                    ),
                    "mse": float(np.mean(np.square(estimate - truth))),
                    "mae": float(np.mean(np.abs(estimate - truth))),
                }
            )
    return pd.DataFrame(rows)


def paired_stratified_bootstrap(
    table: pd.DataFrame,
    column: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    groups = [
        frame[column].to_numpy(np.float64)
        for _, frame in table.groupby("cohort", sort=True)
    ]
    observed = float(table[column].mean())
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = [
            values[rng.integers(0, len(values), size=len(values))] for values in groups
        ]
        draws[replicate] = float(np.concatenate(sampled).mean())
    return {
        "mean": observed,
        "ci_lower_95": float(np.quantile(draws, 0.025)),
        "ci_upper_95": float(np.quantile(draws, 0.975)),
    }


def main() -> int:
    args = arguments()
    if args.sample_batch_size < 1:
        raise ValueError("sample batch size must be positive")
    if not 1 <= args.site_chunk_size <= 1000:
        raise ValueError("site chunk size must lie in [1, 1000]")

    output = args.output_dir
    for name in ("predictions", "tables", "reports", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("architecture") != "latent_transformer_biology_operator":
        raise ValueError("checkpoint architecture is incompatible")
    if checkpoint.get("sealed_phosphosite_rows_loaded") is not False:
        raise ValueError("checkpoint does not document sealed-label protection")

    targets = np.asarray(checkpoint["targets"], dtype=str)
    rna_vocabulary = np.asarray(checkpoint["rna_vocabulary"], dtype=str)
    protein_vocabulary = np.asarray(checkpoint["protein_vocabulary"], dtype=str)

    with h5py.File(args.external_h5, "r") as handle:
        sample_ids = decode_text(handle["sample_id"][:])
        cohorts = decode_text(handle["cohort"][:])
        external_rna_vocabulary = decode_text(handle["rna_vocabulary"][:])
        external_protein_vocabulary = decode_text(handle["protein_vocabulary"][:])
        external_target_vocabulary = decode_text(handle["target_vocabulary"][:])
        rna_raw = handle["rna"][:].astype(np.float32)
        rna_observed = handle["rna_observed"][:].astype(bool)
        phosphosite_raw = handle["phosphosite"][:].astype(np.float32)
        phosphosite_observed = handle["phosphosite_observed"][:].astype(bool)

    rna_case_only_difference_count = int(
        np.sum(rna_vocabulary != external_rna_vocabulary)
    )
    if not np.array_equal(
        np.char.upper(rna_vocabulary), np.char.upper(external_rna_vocabulary)
    ):
        raise ValueError("external RNA vocabulary differs from checkpoint")
    if not np.array_equal(
        np.char.upper(protein_vocabulary), np.char.upper(external_protein_vocabulary)
    ):
        raise ValueError("external protein vocabulary differs from checkpoint")
    if not np.array_equal(targets, external_target_vocabulary):
        raise ValueError("external phosphosite vocabulary differs from checkpoint")
    if not np.array_equal(np.isfinite(rna_raw), rna_observed):
        raise ValueError("external RNA mask differs from finite values")
    if not np.array_equal(np.isfinite(phosphosite_raw), phosphosite_observed):
        raise ValueError("external phosphosite mask differs from finite values")

    anchor = np.load(args.external_protein_anchor, allow_pickle=True)
    anchor_samples = np.asarray(anchor["sample_id"], dtype=str)
    anchor_targets = np.asarray(anchor["target"], dtype=str)
    if not np.array_equal(sample_ids, anchor_samples):
        raise ValueError("external protein anchor sample order differs from HDF5")
    if not np.array_equal(np.char.upper(protein_vocabulary), np.char.upper(anchor_targets)):
        raise ValueError("external protein anchor vocabulary differs from checkpoint")
    usable = np.asarray(anchor["external_sample_usable"], dtype=bool)
    exclusion_reason = np.asarray(anchor["external_sample_exclusion_reason"], dtype=str)
    protein_raw = np.asarray(anchor["prediction"], dtype=np.float32)
    reliability = np.asarray(anchor["reliability"], dtype=np.float32)
    if protein_raw.shape != (len(sample_ids), len(protein_vocabulary)):
        raise ValueError("external protein prediction shape is invalid")
    if reliability.shape != protein_raw.shape:
        raise ValueError("external protein reliability shape is invalid")
    if not np.isfinite(protein_raw[usable]).all():
        raise ValueError("usable external protein predictions contain non-finite values")
    if not np.isfinite(reliability).all() or bool(
        ((reliability < 0) | (reliability > 1)).any()
    ):
        raise ValueError("external protein reliability lies outside [0, 1]")

    split_table = pd.read_csv(args.split_manifest, sep="\t", dtype=str)
    if list(split_table.columns) != ["sample_id", "role"]:
        raise ValueError("split manifest must contain sample_id and role")
    training_ids = split_table.loc[
        split_table["role"] == "selection_train", "sample_id"
    ].to_numpy(str)
    if len(training_ids) != 1796 or len(np.unique(training_ids)) != 1796:
        raise ValueError("external evaluation requires the locked 1,796 training IDs")
    training_rna = read_parquet_rows(
        args.training_rna,
        training_ids.tolist(),
        columns=rna_vocabulary.tolist(),
    ).apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    if not np.isfinite(training_rna).all():
        raise ValueError("selection-training RNA contains non-finite values")
    rna_rank = sample_rank_encode(observed_row_ranks(rna_raw, rna_observed))
    protein = apply_feature_zscore(
        protein_raw,
        np.asarray(checkpoint["protein_mean"], dtype=np.float32),
        np.asarray(checkpoint["protein_scale"], dtype=np.float32),
    )
    normalization = checkpoint.get("normalization", {})
    if normalization.get("method") != "training_site_minmax":
        raise ValueError("checkpoint does not contain a training-site scaler")
    normalized_target, normalized_mask = apply_site_profile_scaler(
        phosphosite_raw,
        np.asarray(normalization["site_minimum"], dtype=np.float32),
        np.asarray(normalization["site_maximum"], dtype=np.float32),
    )
    normalized_mask &= phosphosite_observed

    config = DecoderRetrievalConfig(**metadata["config"])
    checkpoint_state = checkpoint["model_state"]
    (
        esm2_embedding,
        esm2_mask,
        esm2_local_embedding,
        esm2_local_mask,
        _,
    ) = load_esm2_site_prior(args.esm2_prior_npz, targets.tolist())
    if config.esm2_dimension and esm2_embedding is None:
        raise ValueError("ESM-2 checkpoint evaluation requires --esm2-prior-npz")
    model = DecoderRetrievalModel(
        config,
        configuration=metadata["configuration"],
        parent_protein_index=torch.from_numpy(
            np.asarray(checkpoint["parent_protein_index"], dtype=np.int64)
        ),
        parent_protein_mask=torch.from_numpy(
            np.asarray(checkpoint["parent_protein_mask"], dtype=bool)
        ),
        parent_rna_index=torch.from_numpy(
            np.asarray(checkpoint["parent_rna_index"], dtype=np.int64)
        ),
        parent_rna_mask=torch.from_numpy(
            np.asarray(checkpoint["parent_rna_mask"], dtype=bool)
        ),
        kinase_rna_index=torch.from_numpy(
            np.asarray(checkpoint["kinase_rna_index"], dtype=np.int64)
        ),
        kinase_rna_mask=torch.from_numpy(
            np.asarray(checkpoint["kinase_rna_mask"], dtype=bool)
        ),
        kinase_protein_index=torch.from_numpy(
            np.asarray(checkpoint["kinase_protein_index"], dtype=np.int64)
        ),
        kinase_protein_mask=torch.from_numpy(
            np.asarray(checkpoint["kinase_protein_mask"], dtype=bool)
        ),
        site_residue_index=torch.from_numpy(
            np.asarray(checkpoint["site_residue_index"], dtype=np.int64)
        ),
        site_position=torch.from_numpy(
            np.asarray(checkpoint["site_position"], dtype=np.float32)
        ),
        site_esm2_residue_embedding=(
            None if esm2_embedding is None else torch.from_numpy(esm2_embedding)
        ),
        site_esm2_residue_mask=(
            None if esm2_mask is None else torch.from_numpy(esm2_mask)
        ),
        site_esm2_local_embedding=(
            None
            if esm2_local_embedding is None
            else torch.from_numpy(esm2_local_embedding)
        ),
        site_esm2_local_mask=(
            None if esm2_local_mask is None else torch.from_numpy(esm2_local_mask)
        ),
    )
    model.load_state_dict(checkpoint_state, strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(device).eval()

    bundle = load_cophee_prior_bundle(
        args.cophee_prior_bundle,
        targets.tolist(),
        include_site_site=False,
        include_kinase_site=True,
    )
    if list(bundle.kinase_vocabulary) != list(checkpoint["kinase_vocabulary"]):
        raise ValueError("CoPhee kinase vocabulary differs from checkpoint")
    chunks = build_site_chunks(
        np.asarray(checkpoint["parent_protein_index"], dtype=np.int64),
        np.asarray(checkpoint["parent_protein_mask"], dtype=bool),
        bundle.kinase_site_edge_index,
        chunk_size=args.site_chunk_size,
        maximum_kinases_per_site=config.maximum_kinases_per_site,
    )
    chunk_runtime = []
    for chunk in chunks:
        site_array = np.asarray(chunk.site_index, dtype=np.int64)
        kinase_index, kinase_mask = padded_kinase_mapping(
            bundle.kinase_site_edge_index,
            site_array,
            maximum_kinases=config.maximum_kinases_per_site,
        )
        chunk_runtime.append((site_array, kinase_index, kinase_mask))

    selected = np.flatnonzero(usable)
    prediction = np.full((len(sample_ids), len(targets)), np.nan, dtype=np.float32)
    autocast_enabled = args.precision == "bfloat16" and device.type == "cuda"
    with torch.no_grad():
        for start in range(0, len(selected), args.sample_batch_size):
            rows = selected[start : start + args.sample_batch_size]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                context = model.encode_context(
                    torch.from_numpy(rna_rank[rows]).to(device),
                    torch.from_numpy(protein[rows]).to(device),
                    torch.from_numpy(reliability[rows]).to(device),
                )
                for site_array, kinase_index, kinase_mask in chunk_runtime:
                    estimate = model.decode_sites(
                        context,
                        torch.from_numpy(site_array).to(device),
                        torch.from_numpy(kinase_index).to(device),
                        torch.from_numpy(kinase_mask).to(device),
                        enable_cancer_adapter=False,
                    )["normalized_profile"]
                    prediction[np.ix_(rows, site_array)] = estimate.float().cpu().numpy()

    template = np.asarray(checkpoint["training_template"], dtype=np.float32)
    selected_target = normalized_target[selected]
    selected_prediction = prediction[selected]
    selected_mask = normalized_mask[selected]
    selected_template = np.broadcast_to(
        template[None, :], selected_target.shape
    ).copy()
    per_sample = sample_table(
        sample_ids[selected],
        cohorts[selected],
        selected_target,
        selected_prediction,
        template,
        selected_mask,
    )
    per_site = site_table(
        targets,
        cohorts[selected],
        selected_target,
        selected_prediction,
        selected_mask,
        minimum_observations=args.minimum_site_observations,
    )

    summary_rows: list[dict[str, object]] = []
    for cohort in ["ALL", *sorted(set(cohorts[selected].tolist()))]:
        rows = (
            np.arange(len(selected), dtype=np.int64)
            if cohort == "ALL"
            else np.flatnonzero(cohorts[selected] == cohort)
        )
        model_batch_cosine, batches = batch8_cosine(
            selected_target[rows],
            selected_prediction[rows],
            selected_mask[rows],
        )
        template_batch_cosine, _ = batch8_cosine(
            selected_target[rows],
            selected_template[rows],
            selected_mask[rows],
        )
        sample_subset = (
            per_sample if cohort == "ALL" else per_sample.loc[per_sample["cohort"] == cohort]
        )
        site_subset = per_site.loc[per_site["cohort"] == cohort]
        summary_rows.append(
            {
                "cohort": cohort,
                "sample_count": int(len(rows)),
                "batch8_count": batches,
                "model_batch8_flattened_cosine": model_batch_cosine,
                "template_batch8_flattened_cosine": template_batch_cosine,
                "batch8_cosine_gain_over_template": model_batch_cosine
                - template_batch_cosine,
                "model_per_sample_cosine_median": float(
                    sample_subset["model_cosine"].median()
                ),
                "template_per_sample_cosine_median": float(
                    sample_subset["template_cosine"].median()
                ),
                "per_sample_cosine_gain_median": float(
                    sample_subset["cosine_gain_over_template"].median()
                ),
                "per_sample_cosine_gain_mean": float(
                    sample_subset["cosine_gain_over_template"].mean()
                ),
                "model_per_sample_spearman_median": float(
                    sample_subset["model_spearman"].median()
                ),
                "template_per_sample_spearman_median": float(
                    sample_subset["template_spearman"].median()
                ),
                "per_sample_spearman_gain_median": float(
                    sample_subset["spearman_gain_over_template"].median()
                ),
                "residual_cosine_median": float(
                    sample_subset["residual_cosine_after_training_template"].median()
                ),
                "per_site_spearman_median": float(site_subset["spearman"].median()),
                "evaluated_site_count": int(site_subset["spearman"].notna().sum()),
                "mse": float(
                    np.mean(
                        np.square(
                            selected_prediction[rows][selected_mask[rows]]
                            - selected_target[rows][selected_mask[rows]]
                        )
                    )
                ),
                "mae": float(
                    np.mean(
                        np.abs(
                            selected_prediction[rows][selected_mask[rows]]
                            - selected_target[rows][selected_mask[rows]]
                        )
                    )
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    bootstrap = paired_stratified_bootstrap(
        per_sample,
        "cosine_gain_over_template",
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )

    pd.DataFrame(prediction, index=sample_ids, columns=targets).to_parquet(
        output / "predictions" / "external_site_scaled_prediction.parquet"
    )
    per_sample.to_csv(
        output / "tables" / "external_per_sample_metrics.tsv", sep="\t", index=False
    )
    per_site.to_csv(
        output / "tables" / "external_per_site_metrics.tsv", sep="\t", index=False
    )
    summary.to_csv(
        output / "tables" / "external_cohort_summary.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "cohort": cohorts,
            "usable": usable,
            "exclusion_reason": exclusion_reason,
            "rna_observed_count": rna_observed.sum(axis=1),
            "phosphosite_observed_count": phosphosite_observed.sum(axis=1),
        }
    ).to_csv(output / "tables" / "external_sample_audit.tsv", sep="\t", index=False)

    report = {
        "status": "complete",
        "architecture": metadata["architecture"],
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "external_samples_total": int(len(sample_ids)),
        "external_samples_usable": int(usable.sum()),
        "external_samples_excluded": int((~usable).sum()),
        "external_phosphosite_labels_used_for_fit": False,
        "external_measured_protein_used": False,
        "external_phosphosite_labels_used_for_evaluation": True,
        "external_target_transform": "training-site min-max without clipping",
        "external_prediction_rescaled": False,
        "training_template_frozen": True,
        "rna_vocabulary_case_only_difference_count": (
            rna_case_only_difference_count
        ),
        "paired_stratified_bootstrap_cosine_gain": bootstrap,
        "sources": {
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": sha256_file(args.checkpoint),
            },
            "external_h5": {
                "path": str(args.external_h5),
                "sha256": sha256_file(args.external_h5),
            },
            "external_protein_anchor": {
                "path": str(args.external_protein_anchor),
                "sha256": sha256_file(args.external_protein_anchor),
            },
        },
    }
    (output / "reports" / "external_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_csv(sep="\t", index=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
