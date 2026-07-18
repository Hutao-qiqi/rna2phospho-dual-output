from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .data import (
        InputPaths,
        PathwayTensors,
        apply_zscore,
        build_pathway_knn,
        build_pathway_tensors,
        discover_input_paths,
        fit_parent_calibration,
        fit_zscore,
        load_site_kinases,
        masked_huber,
        masked_row_median_center,
        parent_baseline,
        pathway_summary_features,
        per_site_spearman,
        sample_rank_encode,
        select_pathways,
        write_input_manifest,
        write_pathway_manifest,
    )
    from .model import ModelConfig, ProteinAnchoredPathwayGraphModel
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import (  # type: ignore
        InputPaths,
        PathwayTensors,
        apply_zscore,
        build_pathway_knn,
        build_pathway_tensors,
        discover_input_paths,
        fit_parent_calibration,
        fit_zscore,
        load_site_kinases,
        masked_huber,
        masked_row_median_center,
        parent_baseline,
        pathway_summary_features,
        per_site_spearman,
        sample_rank_encode,
        select_pathways,
        write_input_manifest,
        write_pathway_manifest,
    )
    from model import ModelConfig, ProteinAnchoredPathwayGraphModel  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the protein-anchored pathway-specific sample-graph residual model."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--worker-tag", default="worker")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--group-column", default="cancer_label")
    parser.add_argument("--max-pathways", type=int, default=72)
    parser.add_argument("--max-rna-members", type=int, default=128)
    parser.add_argument("--max-protein-members", type=int, default=128)
    parser.add_argument("--max-site-pathways", type=int, default=8)
    parser.add_argument("--max-site-kinases", type=int, default=4)
    parser.add_argument("--sample-knn", type=int, default=15)
    parser.add_argument("--protein-hidden", type=int, default=384)
    parser.add_argument("--pathway-hidden", type=int, default=64)
    parser.add_argument("--pathway-adapter-rank", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--initial-site-shrinkage", type=float, default=0.30)
    parser.add_argument("--protein-epochs", type=int, default=120)
    parser.add_argument("--protein-patience", type=int, default=15)
    parser.add_argument("--residual-epochs", type=int, default=100)
    parser.add_argument("--protein-batch-size", type=int, default=32)
    parser.add_argument("--residual-batch-size", type=int, default=4)
    parser.add_argument("--protein-lr", type=float, default=5.0e-4)
    parser.add_argument("--residual-lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--residual-loss-weight", type=float, default=0.30)
    parser.add_argument("--shrinkage-prior-weight", type=float, default=0.005)
    parser.add_argument("--site-chunk-size", type=int, default=1024)
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rna-path", type=Path)
    parser.add_argument("--total-protein-path", type=Path)
    parser.add_argument("--phosphosite-path", type=Path)
    parser.add_argument("--sample-manifest-path", type=Path)
    parser.add_argument("--phosphosite-manifest-path", type=Path)
    parser.add_argument("--total-protein-manifest-path", type=Path)
    parser.add_argument("--hallmark-gmt", type=Path)
    parser.add_argument("--c2-gmt", type=Path)
    parser.add_argument("--kinase-edges", type=Path)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def override_paths(defaults: InputPaths, args: argparse.Namespace) -> InputPaths:
    replacements = {
        "rna": args.rna_path,
        "total_protein": args.total_protein_path,
        "phosphosite": args.phosphosite_path,
        "sample_manifest": args.sample_manifest_path,
        "phosphosite_manifest": args.phosphosite_manifest_path,
        "total_protein_manifest": args.total_protein_manifest_path,
        "hallmark_gmt": args.hallmark_gmt,
        "c2_gmt": args.c2_gmt,
        "kinase_edges": args.kinase_edges,
    }
    return replace(defaults, **{key: value for key, value in replacements.items() if value is not None})


def read_inputs(paths: InputPaths, max_samples: int | None = None) -> dict[str, object]:
    rna = pd.read_parquet(paths.rna)
    protein = pd.read_parquet(paths.total_protein)
    phosphosite = pd.read_parquet(paths.phosphosite)
    sample_manifest = pd.read_csv(paths.sample_manifest, sep="\t")
    site_manifest = pd.read_csv(paths.phosphosite_manifest, sep="\t")
    protein_manifest = (
        pd.read_csv(paths.total_protein_manifest, sep="\t")
        if paths.total_protein_manifest is not None
        else None
    )

    sample_column = "sample_id" if "sample_id" in sample_manifest.columns else sample_manifest.columns[0]
    sample_manifest[sample_column] = sample_manifest[sample_column].astype(str)
    common = set(rna.index.astype(str)) & set(protein.index.astype(str)) & set(phosphosite.index.astype(str))
    samples = [sample for sample in sample_manifest[sample_column] if sample in common]
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError("No samples are shared by RNA, protein, phosphosite and manifest inputs")

    site_id_column = next(
        column for column in ["gene_site", "scp682_site_id", "gene_site_id"] if column in site_manifest.columns
    )
    if "phosphosite_index" in site_manifest.columns:
        site_manifest = site_manifest.sort_values("phosphosite_index")
    targets = site_manifest[site_id_column].astype(str).tolist()
    parent_column = next(
        column for column in ["parent_gene", "total_protein_gene", "protein_gene"] if column in site_manifest.columns
    )
    parent_genes = site_manifest[parent_column].astype(str).tolist()

    if protein_manifest is not None:
        if "protein_index" in protein_manifest.columns:
            protein_manifest = protein_manifest.sort_values("protein_index")
        protein_column = next(
            column for column in ["protein_gene", "gene", "target"] if column in protein_manifest.columns
        )
        protein_genes = protein_manifest[protein_column].astype(str).tolist()
    else:
        protein_genes = protein.columns.astype(str).tolist()
    missing_sites = sorted(set(targets) - set(phosphosite.columns.astype(str)))
    missing_proteins = sorted(set(protein_genes) - set(protein.columns.astype(str)))
    if missing_sites or missing_proteins:
        raise ValueError(
            f"Target alignment failed: missing_sites={len(missing_sites)}, missing_proteins={len(missing_proteins)}"
        )

    rna = rna.loc[samples].apply(pd.to_numeric, errors="coerce")
    protein = protein.loc[samples, protein_genes].apply(pd.to_numeric, errors="coerce")
    phosphosite = phosphosite.loc[samples, targets].apply(pd.to_numeric, errors="coerce")
    metadata = sample_manifest.set_index(sample_column).loc[samples].copy()
    return {
        "rna": rna,
        "protein": protein,
        "phosphosite": phosphosite,
        "metadata": metadata,
        "targets": targets,
        "parent_genes": parent_genes,
        "protein_genes": protein_genes,
    }


def split_indices(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros(len(groups), dtype=np.float32)
    return list(splitter.split(dummy, groups))


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(prediction.dtype)
    return ((prediction - target).square() * weight).sum() / weight.sum().clamp_min(1.0)


def train_protein_anchor(
    model: ProteinAnchoredPathwayGraphModel,
    rna_z: np.ndarray,
    protein: np.ndarray,
    outer_train: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed)
    inner_train_local, inner_val_local = next(splitter.split(np.zeros(len(outer_train)), groups[outer_train]))
    inner_train = outer_train[inner_train_local]
    inner_val = outer_train[inner_val_local]
    initial_state = copy.deepcopy(model.protein_predictor.state_dict())

    def fit_epochs(train_index: np.ndarray, epochs: int, monitor_index: np.ndarray | None) -> tuple[int, float]:
        optimizer = torch.optim.AdamW(
            model.protein_predictor.parameters(), lr=args.protein_lr, weight_decay=args.weight_decay
        )
        dataset = TensorDataset(torch.as_tensor(train_index, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=args.protein_batch_size, shuffle=True)
        best_state = copy.deepcopy(model.protein_predictor.state_dict())
        best_epoch = 1
        best_loss = float("inf")
        stale = 0
        for epoch in range(1, epochs + 1):
            model.protein_predictor.train()
            for (sample_index,) in loader:
                sample_np = sample_index.numpy()
                x = torch.as_tensor(rna_z[sample_np], device=device)
                y_np = protein[sample_np]
                mask = torch.as_tensor(np.isfinite(y_np), device=device)
                y = torch.as_tensor(np.nan_to_num(y_np, nan=0.0), device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = masked_mse(model.predict_protein(x), y, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.protein_predictor.parameters(), 5.0)
                optimizer.step()
            if monitor_index is None:
                best_epoch = epoch
                continue
            model.protein_predictor.eval()
            with torch.no_grad():
                x = torch.as_tensor(rna_z[monitor_index], device=device)
                y_np = protein[monitor_index]
                mask = torch.as_tensor(np.isfinite(y_np), device=device)
                y = torch.as_tensor(np.nan_to_num(y_np, nan=0.0), device=device)
                validation = float(masked_mse(model.predict_protein(x), y, mask).cpu())
            if validation < best_loss - 1.0e-5:
                best_loss = validation
                best_epoch = epoch
                best_state = copy.deepcopy(model.protein_predictor.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= args.protein_patience:
                break
        if monitor_index is not None:
            model.protein_predictor.load_state_dict(best_state)
        return best_epoch, best_loss

    selected_epoch, validation_loss = fit_epochs(inner_train, args.protein_epochs, inner_val)
    model.protein_predictor.load_state_dict(initial_state)
    fit_epochs(outer_train, selected_epoch, None)
    return {"selected_epoch": selected_epoch, "inner_validation_loss": validation_loss}


def predict_protein_all(
    model: ProteinAnchoredPathwayGraphModel,
    rna_z: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.protein_predictor.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(rna_z), batch_size):
            x = torch.as_tensor(rna_z[start : start + batch_size], device=device)
            output.append(model.predict_protein(x).float().cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def encode_pathway_cache(
    model: ProteinAnchoredPathwayGraphModel,
    rna_rank: np.ndarray,
    protein_hat: np.ndarray,
    sample_index: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model.pathway_encoder.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(sample_index), batch_size):
            ids = sample_index[start : start + batch_size]
            x = torch.as_tensor(rna_rank[ids], device=device)
            p = torch.as_tensor(protein_hat[ids], device=device)
            chunks.append(model.encode_pathways(x, p).detach())
    return torch.cat(chunks, dim=0)


def train_residual_operator(
    model: ProteinAnchoredPathwayGraphModel,
    rna_rank: np.ndarray,
    protein_hat: np.ndarray,
    phosphosite: np.ndarray,
    baseline: np.ndarray,
    centered_target: np.ndarray,
    outer_train: np.ndarray,
    graph_index: np.ndarray,
    graph_similarity: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    history_path: Path,
) -> None:
    model.freeze_protein_anchor()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.residual_lr, weight_decay=args.weight_decay)
    local_ids = np.arange(len(outer_train), dtype=np.int64)
    history: list[dict[str, float]] = []
    initial_gate = torch.tensor(args.initial_site_shrinkage, device=device)

    for epoch in range(1, args.residual_epochs + 1):
        cache = encode_pathway_cache(
            model, rna_rank, protein_hat, outer_train, device, args.cache_batch_size
        )
        model.pathway_encoder.eval()
        model.sample_graph.train()
        model.site_decoder.train()
        np.random.default_rng(args.seed + epoch).shuffle(local_ids)
        epoch_loss = 0.0
        epoch_batches = 0
        for start in range(0, len(local_ids), args.residual_batch_size):
            batch_local = local_ids[start : start + args.residual_batch_size]
            batch_global = outer_train[batch_local]
            rna_batch = torch.as_tensor(rna_rank[batch_global], device=device)
            protein_batch = torch.as_tensor(protein_hat[batch_global], device=device)
            neighbour_index = torch.as_tensor(graph_index[:, batch_local], device=device)
            neighbour_similarity = torch.as_tensor(graph_similarity[:, batch_local], device=device)
            y_np = phosphosite[batch_global]
            y = torch.as_tensor(np.nan_to_num(y_np, nan=0.0), device=device)
            mask = torch.as_tensor(np.isfinite(y_np), device=device)
            residual_target = torch.as_tensor(centered_target[batch_global], device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model.residual_forward(
                    rna_batch,
                    protein_batch,
                    cache,
                    neighbour_index,
                    neighbour_similarity,
                )
                prediction = output["prediction"]
                correction = output["correction"]
                shrinkage = output["site_shrinkage"]
                prediction_loss = masked_huber(prediction, y, mask)
                residual_loss = masked_huber(correction, residual_target, mask)
                shrinkage_loss = (shrinkage.mean() - initial_gate).square()
                loss = (
                    prediction_loss
                    + args.residual_loss_weight * residual_loss
                    + args.shrinkage_prior_weight * shrinkage_loss
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            epoch_batches += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(epoch_batches, 1),
                "mean_site_shrinkage": float(torch.sigmoid(model.site_decoder.site_shrinkage_logit).mean().detach().cpu()),
            }
        )
        pd.DataFrame(history).to_csv(history_path, sep="\t", index=False)


def predict_residual(
    model: ProteinAnchoredPathwayGraphModel,
    rna_rank: np.ndarray,
    protein_hat: np.ndarray,
    source_index: np.ndarray,
    query_index: np.ndarray,
    graph_index: np.ndarray,
    graph_similarity: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    source_cache = encode_pathway_cache(
        model, rna_rank, protein_hat, source_index, device, args.cache_batch_size
    )
    predictions = []
    with torch.no_grad():
        for start in range(0, len(query_index), args.residual_batch_size):
            end = min(start + args.residual_batch_size, len(query_index))
            ids = query_index[start:end]
            output = model.residual_forward(
                torch.as_tensor(rna_rank[ids], device=device),
                torch.as_tensor(protein_hat[ids], device=device),
                source_cache,
                torch.as_tensor(graph_index[:, start:end], device=device),
                torch.as_tensor(graph_similarity[:, start:end], device=device),
            )
            predictions.append(output["prediction"].float().cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32)


def build_model(
    args: argparse.Namespace,
    n_rna: int,
    n_proteins: int,
    n_sites: int,
    pathways: PathwayTensors,
    device: torch.device,
) -> ProteinAnchoredPathwayGraphModel:
    config = ModelConfig(
        n_rna=n_rna,
        n_proteins=n_proteins,
        n_sites=n_sites,
        n_pathways=len(pathways.names),
        n_kinases=len(pathways.kinase_names),
        protein_hidden=args.protein_hidden,
        pathway_hidden=args.pathway_hidden,
        pathway_adapter_rank=args.pathway_adapter_rank,
        dropout=args.dropout,
        initial_site_shrinkage=args.initial_site_shrinkage,
        site_chunk_size=args.site_chunk_size,
    )
    return ProteinAnchoredPathwayGraphModel(
        config,
        torch.as_tensor(pathways.rna_index),
        torch.as_tensor(pathways.rna_mask),
        torch.as_tensor(pathways.protein_index),
        torch.as_tensor(pathways.protein_mask),
        torch.as_tensor(pathways.site_pathway_index),
        torch.as_tensor(pathways.site_pathway_mask),
        torch.as_tensor(pathways.site_pathway_weight),
        torch.as_tensor(pathways.parent_protein_index),
        torch.as_tensor(pathways.parent_protein_mask),
        torch.as_tensor(pathways.site_kinase_index),
        torch.as_tensor(pathways.site_kinase_mask),
    ).to(device)


def checkpoint_payload(
    model: ProteinAnchoredPathwayGraphModel,
    fold: int,
    rna_genes: Sequence[str],
    protein_genes: Sequence[str],
    targets: Sequence[str],
    pathways: PathwayTensors,
    rna_mean: np.ndarray,
    rna_std: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    reference_pathway_state: torch.Tensor,
    reference_graph_features: np.ndarray,
    reference_sample_ids: Sequence[str],
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "model_metadata": model.checkpoint_metadata(),
        "fold": fold,
        "rna_genes": list(rna_genes),
        "protein_genes": list(protein_genes),
        "phosphosite_targets": list(targets),
        "pathway_names": pathways.names,
        "pathway_genes": pathways.genes,
        "kinase_names": pathways.kinase_names,
        "rna_mean": rna_mean,
        "rna_std": rna_std,
        "train_index": outer_train,
        "test_index": outer_test,
        "reference_pathway_state": reference_pathway_state.float().cpu(),
        "reference_graph_features": reference_graph_features.astype(np.float32),
        "reference_sample_ids": list(reference_sample_ids),
    }


def run_fold(
    fold: int,
    split: tuple[np.ndarray, np.ndarray],
    inputs: dict[str, object],
    paths: InputPaths,
    rna_rank: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    fold_seed = args.seed + fold * 1009
    set_seed(fold_seed)
    outer_train, outer_test = split
    fold_dir = args.output_dir / f"fold_{fold}"
    for name in ["models", "predictions", "tables", "logs"]:
        (fold_dir / name).mkdir(parents=True, exist_ok=True)

    rna_frame: pd.DataFrame = inputs["rna"]  # type: ignore[assignment]
    protein_frame: pd.DataFrame = inputs["protein"]  # type: ignore[assignment]
    phosphosite_frame: pd.DataFrame = inputs["phosphosite"]  # type: ignore[assignment]
    metadata: pd.DataFrame = inputs["metadata"]  # type: ignore[assignment]
    targets: list[str] = inputs["targets"]  # type: ignore[assignment]
    parent_genes: list[str] = inputs["parent_genes"]  # type: ignore[assignment]
    protein_genes: list[str] = inputs["protein_genes"]  # type: ignore[assignment]
    rna_genes = list(map(str, rna_frame.columns))
    rna = rna_frame.to_numpy(dtype=np.float32)
    protein = protein_frame.to_numpy(dtype=np.float32)
    phosphosite = phosphosite_frame.to_numpy(dtype=np.float32)
    groups = metadata[args.group_column].astype(str).to_numpy()

    rna_mean, rna_std = fit_zscore(rna, outer_train)
    rna_z = apply_zscore(rna, rna_mean, rna_std)
    variance = np.nanvar(rna[outer_train], axis=0)
    variance_order = [rna_genes[index] for index in np.argsort(-variance)]
    selected = select_pathways(
        paths.hallmark_gmt,
        paths.c2_gmt,
        rna_genes,
        parent_genes,
        variance_order,
        max_pathways=args.max_pathways,
        max_genes_per_pathway=args.max_rna_members,
    )
    site_kinases = load_site_kinases(paths.kinase_edges, targets)
    pathway_tensors = build_pathway_tensors(
        selected,
        rna_genes,
        protein_genes,
        targets,
        parent_genes,
        site_kinases,
        max_rna_members=args.max_rna_members,
        max_protein_members=args.max_protein_members,
        max_site_pathways=args.max_site_pathways,
        max_site_kinases=args.max_site_kinases,
    )
    write_pathway_manifest(fold_dir / "tables/pathway_manifest.tsv", pathway_tensors)
    model = build_model(args, len(rna_genes), len(protein_genes), len(targets), pathway_tensors, device)

    protein_training = train_protein_anchor(
        model, rna_z, protein, outer_train, groups, args, device
    )
    protein_hat = predict_protein_all(model, rna_z, device, args.cache_batch_size)
    intercept, scale, calibration_n = fit_parent_calibration(
        protein_hat,
        phosphosite,
        pathway_tensors.parent_protein_index,
        pathway_tensors.parent_protein_mask,
        outer_train,
    )
    model.set_parent_calibration(torch.as_tensor(intercept), torch.as_tensor(scale))
    baseline = parent_baseline(
        protein_hat,
        intercept,
        scale,
        pathway_tensors.parent_protein_index,
        pathway_tensors.parent_protein_mask,
    )
    observed_mask = np.isfinite(phosphosite)
    centered_target = masked_row_median_center(phosphosite - baseline, observed_mask)
    graph_features = pathway_summary_features(rna_rank, protein_hat, pathway_tensors)
    train_graph_index, train_graph_similarity = build_pathway_knn(
        graph_features[outer_train], graph_features[outer_train], args.sample_knn, exclude_identity=True
    )

    train_residual_operator(
        model,
        rna_rank,
        protein_hat,
        phosphosite,
        baseline,
        centered_target,
        outer_train,
        train_graph_index,
        train_graph_similarity,
        args,
        device,
        fold_dir / "logs/training_history.tsv",
    )

    test_graph_index, test_graph_similarity = build_pathway_knn(
        graph_features[outer_train], graph_features[outer_test], args.sample_knn, exclude_identity=False
    )
    prediction = predict_residual(
        model,
        rna_rank,
        protein_hat,
        outer_train,
        outer_test,
        test_graph_index,
        test_graph_similarity,
        args,
        device,
    )
    prediction_frame = pd.DataFrame(prediction, index=rna_frame.index[outer_test], columns=targets)
    prediction_frame.to_parquet(fold_dir / "predictions/strict_inductive_oof.parquet")
    baseline_frame = pd.DataFrame(baseline[outer_test], index=rna_frame.index[outer_test], columns=targets)
    baseline_frame.to_parquet(fold_dir / "predictions/parent_protein_baseline_oof.parquet")

    model_metrics = per_site_spearman(
        phosphosite[outer_test], prediction, targets, np.arange(len(outer_test))
    )
    baseline_metrics = per_site_spearman(
        phosphosite[outer_test], baseline[outer_test], targets, np.arange(len(outer_test))
    )
    model_metrics.to_csv(fold_dir / "tables/per_site_spearman.tsv", sep="\t", index=False)
    baseline_metrics.to_csv(fold_dir / "tables/parent_baseline_per_site_spearman.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "target": targets,
            "parent_gene": parent_genes,
            "parent_calibration_n": calibration_n,
            "parent_intercept": intercept,
            "parent_scale": scale,
            "site_shrinkage": torch.sigmoid(model.site_decoder.site_shrinkage_logit).detach().cpu().numpy(),
        }
    ).to_csv(fold_dir / "tables/site_parameters.tsv", sep="\t", index=False)

    reference_pathway_state = encode_pathway_cache(
        model, rna_rank, protein_hat, outer_train, device, args.cache_batch_size
    )
    torch.save(
        checkpoint_payload(
            model,
            fold,
            rna_genes,
            protein_genes,
            targets,
            pathway_tensors,
            rna_mean,
            rna_std,
            outer_train,
            outer_test,
            reference_pathway_state,
            graph_features[outer_train],
            rna_frame.index[outer_train].astype(str).tolist(),
        ),
        fold_dir / "models/protein_anchored_pathway_graph.pt",
    )
    summary = {
        "fold": fold,
        "n_train": int(len(outer_train)),
        "n_test": int(len(outer_test)),
        "n_pathways": len(pathway_tensors.names),
        "n_kinases": len(pathway_tensors.kinase_names),
        "protein_training": protein_training,
        "median_spearman": float(model_metrics["spearman"].median(skipna=True)),
        "parent_baseline_median_spearman": float(baseline_metrics["spearman"].median(skipna=True)),
        "strict_inductive_sample_graph": True,
        "test_to_test_edges": False,
        "observed_phosphosite_used_to_build_sample_graph": False,
    }
    (fold_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.protein_epochs = min(args.protein_epochs, 2)
        args.protein_patience = min(args.protein_patience, 1)
        args.residual_epochs = min(args.residual_epochs, 2)
        args.max_pathways = min(args.max_pathways, 8)
        args.max_rna_members = min(args.max_rna_members, 24)
        args.max_protein_members = min(args.max_protein_members, 24)
        args.site_chunk_size = min(args.site_chunk_size, 256)
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker_tag = "".join(character if character.isalnum() or character in "-_" else "_" for character in args.worker_tag)
    worker_tag = worker_tag or "worker"
    paths = override_paths(discover_input_paths(args.project_root), args)
    write_input_manifest(args.output_dir / f"input_paths_{worker_tag}.json", paths)
    inputs = read_inputs(paths, args.max_samples)
    metadata: pd.DataFrame = inputs["metadata"]  # type: ignore[assignment]
    if args.group_column not in metadata.columns:
        raise ValueError(f"Sample manifest lacks group column: {args.group_column}")
    groups = metadata[args.group_column].astype(str).to_numpy()
    splits = split_indices(groups, args.n_splits, args.seed)
    rna_frame: pd.DataFrame = inputs["rna"]  # type: ignore[assignment]
    rna_rank = sample_rank_encode(rna_frame.to_numpy(dtype=np.float32))
    requested = [int(value) for value in args.folds.split(",") if value.strip()]
    if any(fold < 0 or fold >= len(splits) for fold in requested):
        raise ValueError(f"Requested folds must be between 0 and {len(splits) - 1}")

    run_start = time.time()
    summaries = []
    for fold in requested:
        summaries.append(run_fold(fold, splits[fold], inputs, paths, rna_rank, args, device))
    pd.DataFrame(summaries).to_csv(args.output_dir / f"fold_summary_{worker_tag}.tsv", sep="\t", index=False)
    run_summary = {
        "architecture": "protein_anchored_pathway_specific_sample_graph_residual",
        "folds": requested,
        "runtime_seconds": time.time() - run_start,
        "device": str(device),
        "main_model_pointer_modified": False,
    }
    (args.output_dir / f"run_summary_{worker_tag}.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / f"worker_{worker_tag}.done").write_text("done\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
