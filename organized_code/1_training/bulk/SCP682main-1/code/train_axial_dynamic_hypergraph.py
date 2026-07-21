"""Train the locked protein-anchored axial phosphosite residual model."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .axial_dynamic_data import (
        apply_feature_zscore,
        apply_study_site_standardization,
        build_biological_prior,
        build_query_reference_knn,
        fit_feature_zscore,
        fit_parent_calibration,
        fit_study_site_standardization,
        load_site_kinases,
        masked_row_median_center,
        parent_anchor_statistics,
        parent_baseline,
        per_site_metrics,
        prior_metadata,
        read_parquet_rows,
        read_prediction_matrix,
        read_sample_studies,
        sample_rank_encode,
        select_pathways,
        sha256_file,
        validate_protein_prediction_provenance,
        validate_case_split_disjointness,
        validate_split_manifest,
        write_json,
    )
    from .axial_dynamic_hypergraph import (
        AxialHypergraphConfig,
        ProteinAnchoredAxialDynamicHypergraph,
        compose_training_objective,
        masked_sample_intercept_site_equal_mse,
        masked_site_equal_pearson_loss,
        masked_site_equal_variance_loss,
    )
except ImportError:
    from axial_dynamic_data import (
        apply_feature_zscore,
        apply_study_site_standardization,
        build_biological_prior,
        build_query_reference_knn,
        fit_feature_zscore,
        fit_parent_calibration,
        fit_study_site_standardization,
        load_site_kinases,
        masked_row_median_center,
        parent_anchor_statistics,
        parent_baseline,
        per_site_metrics,
        prior_metadata,
        read_parquet_rows,
        read_prediction_matrix,
        read_sample_studies,
        sample_rank_encode,
        select_pathways,
        sha256_file,
        validate_protein_prediction_provenance,
        validate_case_split_disjointness,
        validate_split_manifest,
        write_json,
    )
    from axial_dynamic_hypergraph import (
        AxialHypergraphConfig,
        ProteinAnchoredAxialDynamicHypergraph,
        compose_training_objective,
        masked_sample_intercept_site_equal_mse,
        masked_site_equal_pearson_loss,
        masked_site_equal_variance_loss,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--protein-prediction", type=Path, required=True)
    parser.add_argument("--protein-provenance", type=Path, required=True)
    parser.add_argument("--phosphosite", type=Path, required=True)
    parser.add_argument("--phosphosite-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--sample-metadata", type=Path, required=True)
    parser.add_argument("--sample-id-column", required=True)
    parser.add_argument("--study-column", required=True)
    parser.add_argument("--case-id-column", default="case_submitter_id")
    parser.add_argument("--hallmark-gmt", type=Path, required=True)
    parser.add_argument("--canonical-gmt", type=Path, required=True)
    parser.add_argument("--kinase-prior", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--sample-knn", type=int, default=16)
    parser.add_argument("--candidate-rna-genes", type=int, default=2048)
    parser.add_argument("--max-pathways", type=int, default=128)
    parser.add_argument("--max-rna-members", type=int, default=128)
    parser.add_argument("--max-protein-members", type=int, default=128)
    parser.add_argument("--max-site-pathways", type=int, default=8)
    parser.add_argument("--max-site-kinases", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--axial-layers", type=int, default=2)
    parser.add_argument("--pathway-adapter-rank", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--site-chunk-size", type=int, default=512)
    parser.add_argument("--initial-site-shrinkage", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--pearson-loss-weight", type=float, default=0.20)
    parser.add_argument("--variance-loss-weight", type=float, default=0.10)
    parser.add_argument("--shrinkage-regularization", type=float, default=0.01)
    parser.add_argument("--parent-calibration-ridge", type=float, default=1.0)
    parser.add_argument("--minimum-site-observations", type=int, default=8)
    parser.add_argument("--minimum-study-site-observations", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-inputs-only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _target_columns(manifest: pd.DataFrame) -> tuple[list[str], list[str]]:
    target_column = next(
        (name for name in ["scp682_site_id", "gene_site", "gene_site_id"] if name in manifest.columns),
        None,
    )
    parent_column = next(
        (name for name in ["parent_gene", "total_protein_gene"] if name in manifest.columns),
        None,
    )
    if target_column is None or parent_column is None:
        raise ValueError("phosphosite manifest lacks target or parent-gene columns")
    targets = manifest[target_column].astype(str).tolist()
    parents = manifest[parent_column].astype(str).str.upper().tolist()
    if len(set(targets)) != len(targets):
        raise ValueError("phosphosite manifest contains duplicate targets")
    return targets, parents


def _torch(array: np.ndarray, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.as_tensor(array, device=device, dtype=dtype)


def _slice_query_neighbours(
    values: torch.Tensor, start: int, end: int
) -> torch.Tensor:
    """Slice the query axis for shared or pathway-expanded neighbour tensors."""
    if values.ndim == 2:
        return values[start:end]
    if values.ndim == 3:
        return values[:, start:end]
    raise ValueError("neighbour tensor must have shape [query, k] or [pathway, query, k]")


def _require_finite(name: str, value: torch.Tensor) -> None:
    finite = torch.isfinite(value)
    if not bool(finite.all()):
        count = int((~finite).sum().detach().cpu())
        raise FloatingPointError(f"{name} contains {count} non-finite values")


def build_model(
    config: AxialHypergraphConfig,
    prior,
    site_anchor_quality: np.ndarray,
    site_anchor_coverage: np.ndarray,
    device: torch.device,
):
    return ProteinAnchoredAxialDynamicHypergraph(
        config,
        rna_pathway_index=torch.as_tensor(prior.rna_pathway_index),
        rna_pathway_mask=torch.as_tensor(prior.rna_pathway_mask),
        protein_pathway_index=torch.as_tensor(prior.protein_pathway_index),
        protein_pathway_mask=torch.as_tensor(prior.protein_pathway_mask),
        pathway_relation_mask=torch.as_tensor(prior.pathway_relation_mask),
        site_pathway_index=torch.as_tensor(prior.site_pathway_index),
        site_pathway_mask=torch.as_tensor(prior.site_pathway_mask),
        site_pathway_weight=torch.as_tensor(prior.site_pathway_weight),
        parent_protein_index=torch.as_tensor(prior.parent_protein_index),
        parent_protein_mask=torch.as_tensor(prior.parent_protein_mask),
        site_kinase_index=torch.as_tensor(prior.site_kinase_index),
        site_kinase_mask=torch.as_tensor(prior.site_kinase_mask),
        site_coverage=torch.as_tensor(prior.site_coverage),
        site_anchor_quality=torch.as_tensor(site_anchor_quality),
        site_anchor_coverage=torch.as_tensor(site_anchor_coverage),
    ).to(device)


@torch.no_grad()
def predict(
    model: ProteinAnchoredAxialDynamicHypergraph,
    query_index: np.ndarray,
    rna_rank: torch.Tensor,
    protein_z: torch.Tensor,
    baseline: torch.Tensor,
    reference_cache: tuple[torch.Tensor, ...],
    neighbour_index: torch.Tensor,
    neighbour_similarity: torch.Tensor,
    reference_residual: torch.Tensor,
    reference_residual_mask: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts = []
    for start in range(0, len(query_index), batch_size):
        end = min(len(query_index), start + batch_size)
        ids = torch.as_tensor(query_index[start:end], device=rna_rank.device)
        output = model(
            rna_rank.index_select(0, ids),
            protein_z.index_select(0, ids),
            baseline.index_select(0, ids),
            reference_cache,
            _slice_query_neighbours(neighbour_index, start, end),
            _slice_query_neighbours(neighbour_similarity, start, end),
            reference_residual,
            reference_residual_mask,
        )
        parts.append(output["prediction"].float().cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def median_spearman(table: pd.DataFrame) -> float:
    values = pd.to_numeric(table["spearman"], errors="coerce").dropna()
    return float(values.median()) if len(values) else float("nan")


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 1)
        args.max_pathways = min(args.max_pathways, 8)
        args.max_rna_members = min(args.max_rna_members, 24)
        args.max_protein_members = min(args.max_protein_members, 24)
        args.site_chunk_size = min(args.site_chunk_size, 128)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    output = args.output_dir
    for name in ["models", "predictions", "tables", "logs", "reports"]:
        (output / name).mkdir(parents=True, exist_ok=True)

    started = time.time()
    split = validate_split_manifest(args.split_manifest)
    validate_case_split_disjointness(
        args.sample_metadata,
        split,
        sample_id_column=args.sample_id_column,
        case_id_column=args.case_id_column,
    )
    provenance = validate_protein_prediction_provenance(args.protein_provenance, split)
    protein_crossfit_evidence = provenance.attrs["crossfit_evidence_level"]
    development_ids = split.development_ids.tolist()
    studies = read_sample_studies(
        args.sample_metadata,
        development_ids,
        study_column=args.study_column,
        sample_id_column=args.sample_id_column,
    )
    protein_frame = read_prediction_matrix(args.protein_prediction, split)
    phosphosite_manifest = pd.read_csv(args.phosphosite_manifest, sep="\t")
    targets, parent_genes = _target_columns(phosphosite_manifest)
    rna_frame = read_parquet_rows(args.rna, development_ids)
    phosphosite_frame = read_parquet_rows(args.phosphosite, development_ids, columns=targets)
    rna_frame.columns = rna_frame.columns.astype(str)
    phosphosite_frame.columns = phosphosite_frame.columns.astype(str)
    if list(phosphosite_frame.columns) != targets:
        raise ValueError("phosphosite matrix order differs from the target manifest")

    n_train = len(split.train_ids)
    train_index = np.arange(n_train, dtype=np.int64)
    validation_index = np.arange(n_train, len(development_ids), dtype=np.int64)
    rna_raw = rna_frame.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    phosphosite_raw = phosphosite_frame.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    protein_prediction = protein_frame.to_numpy(np.float32)
    rna_rank = sample_rank_encode(rna_raw)
    protein_mean, protein_scale = fit_feature_zscore(protein_prediction, train_index)
    protein_z = apply_feature_zscore(protein_prediction, protein_mean, protein_scale)
    study_standardization = fit_study_site_standardization(
        phosphosite_raw,
        studies,
        train_index,
        minimum_observations=args.minimum_study_site_observations,
    )
    phosphosite, observed = apply_study_site_standardization(
        phosphosite_raw,
        studies,
        study_standardization,
    )

    train_variance = np.nanvar(rna_raw[train_index], axis=0)
    pathways = select_pathways(
        args.hallmark_gmt,
        args.canonical_gmt,
        rna_frame.columns,
        parent_genes,
        train_variance,
        max_pathways=args.max_pathways,
        max_members=args.max_rna_members,
    )
    site_kinases = load_site_kinases(args.kinase_prior, targets)
    prior = build_biological_prior(
        pathways.members,
        pathways.full_genes,
        rna_frame.columns,
        protein_frame.columns,
        targets,
        parent_genes,
        site_kinases,
        observed[train_index].sum(axis=0),
        len(train_index),
        max_rna_members=args.max_rna_members,
        max_protein_members=args.max_protein_members,
        max_site_pathways=args.max_site_pathways,
        max_site_kinases=args.max_site_kinases,
    )
    intercept, slope, calibration_count = fit_parent_calibration(
        protein_prediction,
        phosphosite,
        prior.parent_protein_index,
        prior.parent_protein_mask,
        train_index,
        ridge=args.parent_calibration_ridge,
        minimum_observations=args.minimum_site_observations,
    )
    anchor_quality, anchor_coverage, anchor_count = parent_anchor_statistics(
        protein_prediction,
        phosphosite,
        prior.parent_protein_index,
        prior.parent_protein_mask,
        train_index,
    )
    baseline_raw = parent_baseline(
        protein_prediction,
        intercept,
        slope,
        prior.parent_protein_index,
        prior.parent_protein_mask,
    )
    baseline_mask = np.broadcast_to(prior.parent_protein_mask[None, :], baseline_raw.shape)
    baseline_centered, baseline_offset = masked_row_median_center(baseline_raw, baseline_mask)
    target_centered, target_offset = masked_row_median_center(phosphosite, observed)
    residual_target = np.where(
        observed, target_centered - baseline_centered, 0.0
    ).astype(np.float32)

    candidate_count = min(args.candidate_rna_genes, rna_rank.shape[1])
    candidate_columns = np.argsort(-train_variance)[:candidate_count]
    candidate_features = rna_rank[:, candidate_columns]
    train_neighbour, train_similarity = build_query_reference_knn(
        candidate_features[train_index],
        candidate_features[train_index],
        args.sample_knn,
        query_ids=split.train_ids,
        reference_ids=split.train_ids,
    )
    validation_neighbour, validation_similarity = build_query_reference_knn(
        candidate_features[validation_index],
        candidate_features[train_index],
        args.sample_knn,
    )

    config = AxialHypergraphConfig(
        n_rna=rna_rank.shape[1],
        n_proteins=protein_prediction.shape[1],
        n_sites=len(targets),
        n_pathways=len(prior.pathway_names),
        n_kinases=len(prior.kinase_names),
        hidden=args.hidden,
        heads=args.heads,
        axial_layers=args.axial_layers,
        pathway_adapter_rank=args.pathway_adapter_rank,
        dropout=args.dropout,
        site_chunk_size=args.site_chunk_size,
        initial_site_shrinkage=args.initial_site_shrinkage,
    )
    model = build_model(config, prior, anchor_quality, anchor_coverage, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rna_tensor = _torch(rna_rank, device, torch.float32)
    protein_tensor = _torch(protein_z, device, torch.float32)
    baseline_tensor = _torch(baseline_centered, device, torch.float32)
    target_tensor = _torch(target_centered, device, torch.float32)
    residual_tensor = _torch(residual_target, device, torch.float32)
    mask_tensor = _torch(observed, device, torch.bool)
    train_index_tensor = _torch(train_index, device, torch.long)
    reference_residual_tensor = residual_tensor.index_select(0, train_index_tensor).detach()
    reference_residual_mask_tensor = mask_tensor.index_select(0, train_index_tensor).detach()
    train_neighbour_tensor = _torch(train_neighbour, device, torch.long)
    train_similarity_tensor = _torch(train_similarity, device, torch.float32)
    validation_neighbour_tensor = _torch(validation_neighbour, device, torch.long)
    validation_similarity_tensor = _torch(validation_similarity, device, torch.float32)
    use_bfloat16 = args.precision == "bfloat16"
    if use_bfloat16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bfloat16 precision was requested on an unsupported device")

    input_manifest = {
        "rna": {"path": str(args.rna), "sha256": sha256_file(args.rna)},
        "protein_prediction": {"path": str(args.protein_prediction), "sha256": sha256_file(args.protein_prediction)},
        "protein_provenance": {"path": str(args.protein_provenance), "sha256": sha256_file(args.protein_provenance)},
        "protein_crossfit_evidence": protein_crossfit_evidence,
        "phosphosite": {"path": str(args.phosphosite), "sha256": sha256_file(args.phosphosite)},
        "phosphosite_manifest": {"path": str(args.phosphosite_manifest), "sha256": sha256_file(args.phosphosite_manifest)},
        "split_manifest": {"path": str(args.split_manifest), "sha256": sha256_file(args.split_manifest)},
        "sample_metadata": {"path": str(args.sample_metadata), "sha256": sha256_file(args.sample_metadata)},
        "study_column": args.study_column,
        "sample_id_column": args.sample_id_column,
        "case_id_column": args.case_id_column,
        "study_site_standardization_fit_role": "selection_train_only",
        "sealed_phosphosite_rows_loaded": False,
        "split_sizes": {"train": len(train_index), "validation": len(validation_index), "sealed": len(split.sealed_ids)},
    }
    write_json(output / "reports/input_manifest.json", input_manifest)
    write_json(output / "reports/architecture.json", model.checkpoint_metadata())
    write_json(output / "reports/prior_summary.json", prior_metadata(prior))
    provenance.to_csv(output / "tables/protein_prediction_provenance.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "target": targets,
            "parent_gene": parent_genes,
            "parent_protein_mapped": prior.parent_protein_mask,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "calibration_n": calibration_count,
            "anchor_quality_abs_pearson": anchor_quality,
            "anchor_coverage": anchor_coverage,
            "anchor_n": anchor_count,
            "training_coverage": prior.site_coverage,
        }
    ).to_csv(output / "tables/parent_protein_calibration.tsv", sep="\t", index=False)
    standardization_rows = []
    for study_index, study_name in enumerate(study_standardization.study_names):
        for site_index, target in enumerate(targets):
            standardization_rows.append(
                {
                    "study_id": study_name,
                    "target": target,
                    "training_mean": study_standardization.mean[study_index, site_index],
                    "training_scale": study_standardization.scale[study_index, site_index],
                    "training_n": study_standardization.count[study_index, site_index],
                    "available": study_standardization.available[study_index, site_index],
                }
            )
    pd.DataFrame(standardization_rows).to_csv(
        output / "tables/study_site_standardization.tsv", sep="\t", index=False
    )

    if args.validate_inputs_only:
        write_json(
            output / "reports/input_validation_summary.json",
            {
                "status": "complete",
                "architecture": model.checkpoint_metadata()["architecture"],
                "n_development_samples": len(development_ids),
                "n_training_samples": len(train_index),
                "n_validation_samples": len(validation_index),
                "n_sealed_samples": len(split.sealed_ids),
                "n_rna_genes": rna_rank.shape[1],
                "n_total_proteins": protein_prediction.shape[1],
                "n_phosphosites": len(targets),
                "n_pathways": len(prior.pathway_names),
                "n_kinases": len(prior.kinase_names),
                "mapped_parent_sites": int(prior.parent_protein_mask.sum()),
                "mapped_kinase_sites": int(prior.site_kinase_mask.any(axis=1).sum()),
                "mapped_specific_pathway_sites": int(
                    ((prior.site_pathway_index != 0) & prior.site_pathway_mask)
                    .any(axis=1)
                    .sum()
                ),
                "sealed_phosphosite_rows_loaded": False,
                "total_protein_trained_in_this_run": False,
            },
        )
        (output / "INPUT_VALIDATION_SUCCESS").write_text("complete\n", encoding="utf-8")
        return 0

    history: list[dict[str, float | int]] = []
    best_score = -np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    local_train = np.arange(len(train_index), dtype=np.int64)
    for epoch in range(1, args.epochs + 1):
        model.eval()
        reference_cache = model.build_reference_cache(
            rna_tensor[train_index],
            protein_tensor[train_index],
            train_neighbour_tensor,
            train_similarity_tensor,
            chunk_size=args.cache_batch_size,
        )
        for layer_index, cached_state in enumerate(reference_cache):
            _require_finite(f"reference_cache[{layer_index}]", cached_state)
        model.train()
        generator = np.random.default_rng(args.seed + epoch)
        generator.shuffle(local_train)
        epoch_losses = []
        epoch_value = []
        epoch_pearson = []
        epoch_variance = []
        for start in range(0, len(local_train), args.batch_size):
            local = local_train[start : start + args.batch_size]
            global_index = train_index[local]
            ids = torch.as_tensor(global_index, device=device)
            local_ids = torch.as_tensor(local, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                result = model(
                    rna_tensor.index_select(0, ids),
                    protein_tensor.index_select(0, ids),
                    baseline_tensor.index_select(0, ids),
                    reference_cache,
                    train_neighbour_tensor.index_select(0, local_ids),
                    train_similarity_tensor.index_select(0, local_ids),
                    reference_residual_tensor,
                    reference_residual_mask_tensor,
                )
                prediction = result["prediction"]
                target_batch = target_tensor.index_select(0, ids)
                mask_batch = mask_tensor.index_select(0, ids)
                value_loss = masked_sample_intercept_site_equal_mse(
                    prediction, target_batch, mask_batch
                )
                pearson_loss = masked_site_equal_pearson_loss(
                    prediction,
                    target_batch,
                    mask_batch,
                    minimum_observations=min(args.minimum_site_observations, len(local)),
                )
                variance_loss = masked_site_equal_variance_loss(
                    prediction,
                    target_batch,
                    mask_batch,
                    minimum_observations=min(args.minimum_site_observations, len(local)),
                )
                _require_finite("value_loss", value_loss)
                _require_finite("pearson_loss", pearson_loss)
                _require_finite("variance_loss", variance_loss)
                loss, _ = compose_training_objective(
                    value_loss,
                    pearson_loss,
                    variance_loss,
                    model.shrinkage_regularization(),
                    pearson_weight=args.pearson_loss_weight,
                    variance_weight=args.variance_loss_weight,
                    shrinkage_weight=args.shrinkage_regularization,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_value.append(float(value_loss.detach().cpu()))
            epoch_pearson.append(float(pearson_loss.detach().cpu()))
            epoch_variance.append(float(variance_loss.detach().cpu()))

        score = float("nan")
        if epoch % args.validation_interval == 0 or epoch == args.epochs:
            model.eval()
            reference_cache = model.build_reference_cache(
                rna_tensor[train_index],
                protein_tensor[train_index],
                train_neighbour_tensor,
                train_similarity_tensor,
                chunk_size=args.cache_batch_size,
            )
            validation_prediction = predict(
                model,
                validation_index,
                rna_tensor,
                protein_tensor,
                baseline_tensor,
                reference_cache,
                validation_neighbour_tensor,
                validation_similarity_tensor,
                reference_residual_tensor,
                reference_residual_mask_tensor,
                args.batch_size,
            )
            validation_metrics = per_site_metrics(
                target_centered[validation_index],
                validation_prediction,
                observed[validation_index],
                targets,
            )
            score = median_spearman(validation_metrics)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
                validation_metrics.to_csv(
                    output / "tables/validation_per_site_metrics_best.tsv", sep="\t", index=False
                )
                pd.DataFrame(
                    validation_prediction,
                    index=split.validation_ids,
                    columns=targets,
                ).to_parquet(output / "predictions/validation_prediction_best.parquet")
                stale = 0
            else:
                stale += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "value_mse": float(np.mean(epoch_value)),
                "pearson_loss": float(np.mean(epoch_pearson)),
                "variance_loss": float(np.mean(epoch_variance)),
                "validation_median_spearman": score,
                "mean_site_shrinkage": float(
                    torch.sigmoid(model.site_decoder.site_shrinkage_logit).mean().detach().cpu()
                ),
            }
        )
        pd.DataFrame(history).to_csv(output / "logs/training_history.tsv", sep="\t", index=False)
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("training produced no finite validation Spearman checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    reference_cache = model.build_reference_cache(
        rna_tensor[train_index],
        protein_tensor[train_index],
        train_neighbour_tensor,
        train_similarity_tensor,
        chunk_size=args.cache_batch_size,
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_metadata": model.checkpoint_metadata(),
        "rna_genes": rna_frame.columns.tolist(),
        "protein_genes": protein_frame.columns.tolist(),
        "phosphosite_targets": targets,
        "parent_genes": parent_genes,
        "pathway_names": prior.pathway_names,
        "pathway_genes": prior.pathway_genes,
        "pathway_full_genes": prior.pathway_full_genes,
        "kinase_names": prior.kinase_names,
        "prior_tensors": {
            "rna_pathway_index": prior.rna_pathway_index,
            "rna_pathway_mask": prior.rna_pathway_mask,
            "protein_pathway_index": prior.protein_pathway_index,
            "protein_pathway_mask": prior.protein_pathway_mask,
            "pathway_relation_mask": prior.pathway_relation_mask,
            "site_pathway_index": prior.site_pathway_index,
            "site_pathway_mask": prior.site_pathway_mask,
            "site_pathway_weight": prior.site_pathway_weight,
            "parent_protein_index": prior.parent_protein_index,
            "parent_protein_mask": prior.parent_protein_mask,
            "site_kinase_index": prior.site_kinase_index,
            "site_kinase_mask": prior.site_kinase_mask,
            "site_coverage": prior.site_coverage,
        },
        "protein_mean": protein_mean,
        "protein_scale": protein_scale,
        "parent_intercept": intercept,
        "parent_slope": slope,
        "site_anchor_quality": anchor_quality,
        "site_anchor_coverage": anchor_coverage,
        "study_site_standardization": {
            "sample_id_column": args.sample_id_column,
            "case_id_column": args.case_id_column,
            "study_column": args.study_column,
            "study_names": study_standardization.study_names,
            "mean": study_standardization.mean,
            "scale": study_standardization.scale,
            "count": study_standardization.count,
            "available": study_standardization.available,
            "fit_role": "selection_train_only",
            "minimum_observations": args.minimum_study_site_observations,
            "validation_parameters_refit": False,
        },
        "candidate_rna_columns": candidate_columns,
        "reference_candidate_features": candidate_features[train_index],
        "reference_sample_ids": split.train_ids.tolist(),
        "reference_cache": tuple(value.float().cpu() for value in reference_cache),
        "reference_residual": reference_residual_tensor.float().cpu(),
        "reference_residual_mask": reference_residual_mask_tensor.cpu(),
        "sample_knn": int(args.sample_knn),
        "best_epoch": best_epoch,
        "best_validation_median_spearman": best_score,
        "sealed_phosphosite_rows_loaded": False,
        "training_objective_terms": [
            "site_equal_mse",
            "site_equal_pearson",
            "site_equal_variance",
            "site_shrinkage",
        ],
        "protein_crossfit_evidence": protein_crossfit_evidence,
    }
    torch.save(checkpoint, output / "models/axial_dynamic_hypergraph_best.pt")
    pd.DataFrame(
        {
            "sample_id": development_ids,
            "target_median_offset": target_offset,
            "parent_baseline_median_offset": baseline_offset,
        }
    ).to_csv(output / "tables/sample_centering_offsets.tsv", sep="\t", index=False)
    summary = {
        "architecture": model.checkpoint_metadata()["architecture"],
        "best_epoch": best_epoch,
        "validation_median_spearman": best_score,
        "epochs_completed": len(history),
        "runtime_seconds": time.time() - started,
        "device": str(device),
        "precision": args.precision,
        "total_protein_trained_in_this_run": False,
        "sealed_samples_evaluated": False,
        "sealed_phosphosite_rows_loaded": False,
    }
    write_json(output / "reports/run_summary.json", summary)
    (output / "SUCCESS").write_text("complete\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
