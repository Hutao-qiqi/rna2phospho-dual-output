"""CPTAC-only development screen for the TCGA-CPTAC hybrid protein model.

This entry uses a case-blocked development split entirely inside the locked
916 selection-training samples. The 229 selection-validation labels and 286
outer-test labels are sealed before target preprocessing and evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

try:
    from . import train_protein as base
    from .candidate_groupwise_rna_normalization import (
        build_strict_inner_partitions,
        seal_outer_protein_labels,
    )
    from .candidate_multiscale_protein_query import build_protein_local_gene_prior
    from .candidate_tcga_cptac_hybrid_protein import (
        CPTAC_PLATFORM,
        HybridProteinConfig,
        TCGACPTACHybridProteinTranslator,
    )
    from .hybrid_protein_training_contract import (
        TrainFittedProteinScale,
        make_development_split,
        masked_per_protein_pearson_loss,
        masked_per_protein_mse,
    )
    from .metrics import (
        batchwise_flattened_cosine,
        masked_mse,
        per_protein_spearman,
        summarize_spearman,
    )
    from .protein_graph_prior import ProteinGraphArtifact, sha256_file
except ImportError:
    import train_protein as base
    from candidate_groupwise_rna_normalization import (
        build_strict_inner_partitions,
        seal_outer_protein_labels,
    )
    from candidate_multiscale_protein_query import build_protein_local_gene_prior
    from candidate_tcga_cptac_hybrid_protein import (
        CPTAC_PLATFORM,
        HybridProteinConfig,
        TCGACPTACHybridProteinTranslator,
    )
    from hybrid_protein_training_contract import (
        TrainFittedProteinScale,
        make_development_split,
        masked_per_protein_pearson_loss,
        masked_per_protein_mse,
    )
    from metrics import (
        batchwise_flattened_cosine,
        masked_mse,
        per_protein_spearman,
        summarize_spearman,
    )
    from protein_graph_prior import ProteinGraphArtifact, sha256_file


MODEL_FAMILY = "tcga_cptac_hybrid_protein_development"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protein-graph-artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--development-folds", type=int, default=5)
    parser.add_argument("--development-fold-index", type=int, default=0)
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--strata-column", default="cancer_label")
    parser.add_argument("--group-column", default="case_submitter_id")
    parser.add_argument("--inner-group-column", default="case_submitter_id")
    parser.add_argument("--study-column", default="pdc_study_id")
    parser.add_argument("--rna-file", default="rna_log2_tpm_paired.parquet")
    parser.add_argument("--protein-raw-file", default="total_protein_gene_logratio_all.parquet")
    parser.add_argument("--protein-vocab-file", default="total_protein_gene_study_zscore_min20pct.parquet")
    parser.add_argument("--manifest-file", default="sample_manifest.tsv")
    parser.add_argument("--lower-quantile", type=float, default=0.01)
    parser.add_argument("--upper-quantile", type=float, default=0.99)
    parser.add_argument("--map-hidden", type=int, default=512)
    parser.add_argument("--map-states", type=int, default=256)
    parser.add_argument("--global-states", type=int, default=128)
    parser.add_argument("--module-states", type=int, default=64)
    parser.add_argument("--module-depth", type=int, default=2)
    parser.add_argument("--protein-chunk", type=int, default=512)
    parser.add_argument("--max-local-genes", type=int, default=64)
    parser.add_argument("--branch-dropout", type=float, default=0.05)
    parser.add_argument("--gate-penalty", type=float, default=1e-3)
    parser.add_argument("--pearson-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument("--pearson-min-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--translator-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-genes", type=int)
    parser.add_argument("--max-proteins", type=int)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cancer_vocabulary(labels: np.ndarray, fit_indices: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
    known = tuple(sorted(set(labels[fit_indices].astype(str).tolist())))
    mapping = {label: index + 1 for index, label in enumerate(known)}
    encoded = np.asarray([mapping.get(str(label), 0) for label in labels], dtype=np.int64)
    return ("<unknown>", *known), encoded


@dataclass(frozen=True)
class DevelopmentArrays:
    rna: np.ndarray
    target: np.ndarray
    mask: np.ndarray
    cancer_index: np.ndarray


class DevelopmentDataset(Dataset):
    def __init__(self, arrays: DevelopmentArrays, indices: np.ndarray) -> None:
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        row = int(self.indices[item])
        return {
            "rna": torch.from_numpy(self.arrays.rna[row]),
            "target": torch.from_numpy(self.arrays.target[row]),
            "mask": torch.from_numpy(self.arrays.mask[row]),
            "cancer_index": torch.tensor(self.arrays.cancer_index[row], dtype=torch.long),
        }


def make_loader(
    arrays: DevelopmentArrays,
    indices: np.ndarray,
    args: argparse.Namespace,
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        DevelopmentDataset(arrays, indices),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )


def autocast_context(args: argparse.Namespace):
    if args.precision == "fp32":
        return torch.autocast("cuda", enabled=False)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def optimizer_and_scheduler(
    model: TCGACPTACHybridProteinTranslator,
    args: argparse.Namespace,
    updates: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    encoder_parameters = list(model.rna_encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    other_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": other_parameters, "lr": args.translator_lr},
        ],
        weight_decay=args.weight_decay,
    )
    warmup = max(1, int(updates * args.warmup_fraction))

    def factor(step: int) -> float:
        if step < warmup:
            return max((step + 1) / warmup, 1e-3)
        progress = (step - warmup) / max(updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    return optimizer, scheduler


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def predict(
    model: TCGACPTACHybridProteinTranslator,
    arrays: DevelopmentArrays,
    indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    for batch in make_loader(arrays, indices, args, shuffle=False):
        rna = batch["rna"].to(device, non_blocking=True)
        cancer = batch["cancer_index"].to(device, non_blocking=True)
        platform = torch.full_like(cancer, CPTAC_PLATFORM)
        with autocast_context(args):
            output = model(rna, platform_index=platform, cancer_index=cancer)["protein"]
        parts.append(output.float().cpu().numpy())
    return np.vstack(parts)


def validation_tables(
    prediction: np.ndarray,
    arrays: DevelopmentArrays,
    indices: np.ndarray,
    protein_names: list[str],
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    observed = np.where(arrays.mask[indices], arrays.target[indices], np.nan)
    mask = arrays.mask[indices]
    per_protein = per_protein_spearman(
        observed, prediction, protein_names, mask=mask, min_samples=10
    )
    observed_nan = np.where(mask, observed, np.nan)
    prediction_nan = np.where(mask, prediction, np.nan)
    observed_sd = np.nanstd(observed_nan, axis=0, ddof=1)
    prediction_sd = np.nanstd(prediction_nan, axis=0, ddof=1)
    ratio = np.divide(
        prediction_sd,
        observed_sd,
        out=np.full_like(prediction_sd, np.nan, dtype=np.float64),
        where=np.isfinite(observed_sd) & (observed_sd > 0),
    )
    per_protein["observed_sd"] = observed_sd
    per_protein["predicted_sd"] = prediction_sd
    per_protein["predicted_to_observed_sd_ratio"] = ratio
    cosine = batchwise_flattened_cosine(
        observed, prediction, mask=mask, batch_size=batch_size
    )
    spearman = pd.to_numeric(per_protein["spearman"], errors="coerce").to_numpy(float)
    finite = np.isfinite(spearman)
    finite_ratio = ratio[np.isfinite(ratio)]
    summary = summarize_spearman(per_protein)
    summary.update(
        {
            "fraction_spearman_gt_0_3": float(np.mean(spearman[finite] > 0.3)),
            "fraction_spearman_gt_0_5": float(np.mean(spearman[finite] > 0.5)),
            "median_predicted_to_observed_sd_ratio": float(np.median(finite_ratio)),
            "validation_mse": float(masked_mse(observed, prediction, mask=mask)),
            "validation_mean_batch_cosine": float(cosine["cosine_similarity"].mean()),
            "n_validation_batches": int(len(cosine)),
        }
    )
    return per_protein, cosine, summary


def main() -> int:
    args = parse_args()
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = args.output_dir / "tables"
    model_dir = args.output_dir / "models"
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_status.txt").write_text("running\n", encoding="utf-8")
    (args.output_dir / "config.json").write_text(
        json.dumps(base.jsonable_args(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    inputs = base.load_inputs(args)
    manifest = inputs["manifest"]
    partitions = build_strict_inner_partitions(
        inputs["sample_ids"],
        manifest[args.strata_column].astype(str).to_numpy(),
        manifest[args.group_column].astype(str).to_numpy(),
        seed=args.seed,
        n_folds=args.n_folds,
        inner_folds=args.inner_folds,
        fold_index=args.fold_index,
    )
    sizes = (
        partitions.selection_train.size,
        partitions.selection_validation.size,
        partitions.outer_test.size,
    )
    if sizes != (916, 229, 286):
        raise RuntimeError(f"locked split differs from 916/229/286: {sizes}")
    development = make_development_split(
        partitions.selection_train,
        inputs["sample_ids"],
        manifest[args.strata_column].astype(str).to_numpy(),
        manifest[args.inner_group_column].astype(str).to_numpy(),
        n_splits=args.development_folds,
        seed=args.seed + 3011,
        fold_index=args.development_fold_index,
    )

    sealed_raw = seal_outer_protein_labels(inputs["protein_raw"], partitions.outer_test)
    sealed_raw = seal_outer_protein_labels(sealed_raw, partitions.selection_validation)
    raw_mask = np.isfinite(sealed_raw)
    rna_scaler = base.FeatureStandardizer.fit(
        inputs["rna"], inputs["gene_names"], indices=development.train_indices
    )
    rna = rna_scaler.transform(inputs["rna"], inputs["gene_names"]).astype(np.float32)
    target_scaler = TrainFittedProteinScale.fit(
        sealed_raw,
        development.train_indices,
        feature_names=inputs["protein_names"],
        mask=raw_mask,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
    )
    target = np.zeros_like(sealed_raw, dtype=np.float32)
    target_mask = np.zeros_like(raw_mask, dtype=bool)
    train_changed = target_scaler.transform(
        sealed_raw[development.train_indices],
        mask=raw_mask[development.train_indices],
        feature_names=inputs["protein_names"],
        clip=True,
    )
    validation_changed = target_scaler.transform(
        sealed_raw[development.validation_indices],
        mask=raw_mask[development.validation_indices],
        feature_names=inputs["protein_names"],
        clip=False,
    )
    target[development.train_indices] = np.nan_to_num(train_changed, nan=0.0)
    target[development.validation_indices] = np.nan_to_num(validation_changed, nan=0.0)
    target_mask[development.train_indices] = np.isfinite(train_changed)
    target_mask[development.validation_indices] = np.isfinite(validation_changed)
    cancer_labels = manifest[args.strata_column].astype(str).to_numpy()
    cancers, cancer_index = cancer_vocabulary(cancer_labels, development.train_indices)
    arrays = DevelopmentArrays(rna, target, target_mask, cancer_index)

    graph = ProteinGraphArtifact.load(args.protein_graph_artifact)
    if graph.protein_names != tuple(map(str, inputs["protein_names"])):
        raise ValueError("protein graph order differs from output vocabulary")
    parent = base.build_parent_gene_index(inputs)
    local_prior = build_protein_local_gene_prior(
        inputs["gene_names"],
        inputs["protein_names"],
        parent,
        graph,
        max_local_genes=args.max_local_genes,
    )
    config = HybridProteinConfig(
        n_genes=len(inputs["gene_names"]),
        n_proteins=len(inputs["protein_names"]),
        n_cancers=len(cancers),
        map_hidden=args.map_hidden,
        map_states=args.map_states,
        n_global_states=args.global_states,
        n_module_states=args.module_states,
        module_depth=args.module_depth,
        protein_chunk=args.protein_chunk,
        max_local_genes=args.max_local_genes,
        branch_dropout=args.branch_dropout,
    )
    model = TCGACPTACHybridProteinTranslator(config, local_prior).to(device)
    model.fix_projection_matrices_()
    train_loader = make_loader(arrays, development.train_indices, args, shuffle=True)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = max(updates_per_epoch * args.max_epochs, 1)
    optimizer, scheduler = optimizer_and_scheduler(model, args, total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    best_epoch = 0
    best_spearman = float("-inf")
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_value = 0.0
        epoch_pearson = 0.0
        epoch_gate = 0.0
        for step, batch in enumerate(train_loader):
            rna_batch = batch["rna"].to(device, non_blocking=True)
            target_batch = batch["target"].to(device, non_blocking=True)
            mask_batch = batch["mask"].to(device, non_blocking=True)
            cancer_batch = batch["cancer_index"].to(device, non_blocking=True)
            platform_batch = torch.full_like(cancer_batch, CPTAC_PLATFORM)
            window_start = (step // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(
                args.gradient_accumulation, len(train_loader) - window_start
            )
            with autocast_context(args):
                output = model(
                    rna_batch,
                    platform_index=platform_batch,
                    cancer_index=cancer_batch,
                    return_hidden=True,
                )
                value_loss = masked_per_protein_mse(
                    output["protein"], target_batch, mask_batch
                )
                pearson_loss = masked_per_protein_pearson_loss(
                    output["protein"],
                    target_batch,
                    mask_batch,
                    minimum_observations=args.pearson_min_samples,
                )
                gate_loss = output["map_gate_penalty"]
                loss = (
                    args.pearson_weight * pearson_loss
                    + args.mse_weight * value_loss
                    + args.gate_penalty * gate_loss
                )
            scaler.scale(loss / window_size).backward()
            epoch_loss += float(loss.detach().float().cpu())
            epoch_value += float(value_loss.detach().float().cpu())
            epoch_pearson += float(pearson_loss.detach().float().cpu())
            epoch_gate += float(gate_loss.detach().float().cpu())
            synchronize = (
                (step + 1) % args.gradient_accumulation == 0
                or step + 1 == len(train_loader)
            )
            if synchronize:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        prediction = predict(
            model, arrays, development.validation_indices, args, device
        )
        per_protein, cosine, summary = validation_tables(
            prediction,
            arrays,
            development.validation_indices,
            inputs["protein_names"],
            args.batch_size,
        )
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_loader),
            "train_per_protein_mse": epoch_value / len(train_loader),
            "train_per_protein_pearson_loss": epoch_pearson / len(train_loader),
            "train_map_gate_penalty": epoch_gate / len(train_loader),
            "development_median_spearman": summary["median_spearman"],
            "development_mse": summary["validation_mse"],
            "development_batch_cosine": summary["validation_mean_batch_cosine"],
            "development_sd_ratio": summary["median_predicted_to_observed_sd_ratio"],
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "translator_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        pd.DataFrame(history).to_csv(
            table_dir / "training_history_running.tsv", sep="\t", index=False
        )
        monitor = float(summary["median_spearman"])
        if monitor > best_spearman + 1e-6:
            best_spearman = monitor
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
            atomic_torch_save(
                {
                    "schema_version": 1,
                    "model_family": MODEL_FAMILY,
                    "epoch": epoch,
                    "development_median_spearman": monitor,
                    "model_config": config.to_dict(),
                    "model_state": best_state,
                },
                model_dir / "best_checkpoint_running.pt",
            )
        else:
            stale += 1
        atomic_torch_save(
            {
                "schema_version": 1,
                "model_family": MODEL_FAMILY,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_development_median_spearman": best_spearman,
                "model_config": config.to_dict(),
                "model_state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            },
            model_dir / "latest_checkpoint_running.pt",
        )
        if epoch >= args.min_epochs and stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("training produced no selectable checkpoint")
    model.load_state_dict(best_state)
    prediction = predict(model, arrays, development.validation_indices, args, device)
    per_protein, cosine, summary = validation_tables(
        prediction,
        arrays,
        development.validation_indices,
        inputs["protein_names"],
        args.batch_size,
    )
    per_protein.to_csv(table_dir / "development_per_protein_metrics.tsv", sep="\t", index=False)
    cosine.to_csv(table_dir / "development_batch_cosine.tsv", sep="\t", index=False)
    pd.DataFrame(history).to_csv(table_dir / "training_history.tsv", sep="\t", index=False)
    np.savez_compressed(
        table_dir / "development_split_indices.npz",
        development_train=development.train_indices,
        development_validation=development.validation_indices,
        locked_selection_validation=partitions.selection_validation,
        locked_outer_test=partitions.outer_test,
    )
    np.savez_compressed(table_dir / "protein_target_scale.npz", **target_scaler.state_dict())
    np.savez_compressed(
        table_dir / "rna_global_standardizer.npz",
        mean=rna_scaler.mean,
        scale=rna_scaler.scale,
        gene_names=np.asarray(inputs["gene_names"], dtype=str),
    )
    np.savez_compressed(
        table_dir / "local_prior.npz",
        gene_index=local_prior.gene_index,
        gene_mask=local_prior.gene_mask,
        relation_index=local_prior.relation_index,
        edge_strength=local_prior.edge_strength,
        relation_names=np.asarray(local_prior.relation_names, dtype=str),
    )
    checkpoint = {
        "schema_version": 1,
        "model_family": MODEL_FAMILY,
        "model_class": "TCGACPTACHybridProteinTranslator",
        "model_config": config.to_dict(),
        "model_state": best_state,
        "selected_epoch": best_epoch,
        "gene_names": inputs["gene_names"],
        "protein_names": inputs["protein_names"],
        "cancer_vocabulary": cancers,
        "rna_standardizer": {
            "mean": rna_scaler.mean,
            "scale": rna_scaler.scale,
        },
        "target_scale": target_scaler.state_dict(),
        "architecture_contract": model.architecture_contract(),
        "split_contract": {
            "model_development_inside_locked_916": True,
            "locked_229_labels_evaluated": False,
            "locked_286_labels_evaluated": False,
        },
        "protein_graph_sha256": sha256_file(args.protein_graph_artifact),
    }
    torch.save(checkpoint, model_dir / f"{MODEL_FAMILY}_fold0.pt")
    summary.update(
        {
            "model_family": MODEL_FAMILY,
            "selected_epoch": best_epoch,
            "n_development_train": int(development.train_indices.size),
            "n_development_validation": int(development.validation_indices.size),
            "n_locked_selection_validation": int(partitions.selection_validation.size),
            "n_locked_outer_test": int(partitions.outer_test.size),
            "locked_selection_validation_labels_evaluated": False,
            "outer_test_evaluated": False,
            "parameter_count": model.parameter_count(),
            "target_scale": "train_per_protein_quantile_01_99",
            "training_target_clipping": True,
            "development_validation_clipping": False,
            "training_loss": "per_protein_pearson_plus_scaled_mse_plus_map_gate_penalty",
            "pearson_weight": args.pearson_weight,
            "mse_weight": args.mse_weight,
            "pearson_min_samples": args.pearson_min_samples,
            "checkpoint_selection": "development_median_per_protein_spearman",
            "runtime_seconds": time.time() - started,
        }
    )
    encoded = json.dumps(summary, ensure_ascii=False, indent=2)
    (args.output_dir / "final_summary.json").write_text(encoded, encoding="utf-8")
    (args.output_dir / "done.txt").write_text(encoded, encoding="utf-8")
    (args.output_dir / "run_status.txt").write_text("completed\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
