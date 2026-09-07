"""Joint TCGA-TCPA and CPTAC development training for the hybrid model.

Model selection remains inside the locked CPTAC 916-sample selection-training
partition. TCPA supervision is restricted to the audited total-protein mask.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from . import train_protein as base
    from . import train_hybrid_protein_development_screen as dev
    from .candidate_groupwise_rna_normalization import (
        build_strict_inner_partitions,
        seal_outer_protein_labels,
    )
    from .candidate_multiscale_protein_query import build_protein_local_gene_prior
    from .candidate_tcga_cptac_hybrid_protein import (
        CPTAC_PLATFORM,
        TCPA_PLATFORM,
        HybridProteinConfig,
        TCGACPTACHybridProteinTranslator,
    )
    from .hybrid_protein_training_contract import (
        TrainFittedProteinScale,
        make_development_split,
        masked_per_protein_pearson_loss,
        masked_per_protein_mse,
    )
    from .protein_graph_prior import ProteinGraphArtifact, sha256_file
except ImportError:
    import train_protein as base
    import train_hybrid_protein_development_screen as dev
    from candidate_groupwise_rna_normalization import (
        build_strict_inner_partitions,
        seal_outer_protein_labels,
    )
    from candidate_multiscale_protein_query import build_protein_local_gene_prior
    from candidate_tcga_cptac_hybrid_protein import (
        CPTAC_PLATFORM,
        TCPA_PLATFORM,
        HybridProteinConfig,
        TCGACPTACHybridProteinTranslator,
    )
    from hybrid_protein_training_contract import (
        TrainFittedProteinScale,
        make_development_split,
        masked_per_protein_pearson_loss,
        masked_per_protein_mse,
    )
    from protein_graph_prior import ProteinGraphArtifact, sha256_file


MODEL_FAMILY = "tcga_tcpa_cptac_hybrid_joint_development"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tcpa-prepared-dir", type=Path, required=True)
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
    parser.add_argument("--tcpa-loss-weight", type=float, default=0.1)
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
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


class PlatformDataset(Dataset):
    def __init__(
        self,
        rna: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        cancer_index: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.rna = rna
        self.target = target
        self.mask = mask
        self.cancer_index = cancer_index
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        row = int(self.indices[item])
        return {
            "rna": torch.from_numpy(self.rna[row]),
            "target": torch.from_numpy(self.target[row]),
            "mask": torch.from_numpy(self.mask[row]),
            "cancer_index": torch.tensor(self.cancer_index[row], dtype=torch.long),
        }


def combined_cancer_vocabulary(
    cptac_labels: np.ndarray,
    cptac_fit: np.ndarray,
    tcpa_labels: np.ndarray,
    tcpa_fit: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    known = tuple(
        sorted(
            set(cptac_labels[cptac_fit].astype(str)).union(
                tcpa_labels[tcpa_fit].astype(str)
            )
        )
    )
    mapping = {label: index + 1 for index, label in enumerate(known)}
    encode = lambda values: np.asarray(
        [mapping.get(str(value), 0) for value in values], dtype=np.int64
    )
    return ("<unknown>", *known), encode(cptac_labels), encode(tcpa_labels)


def balanced_loader(
    dataset: PlatformDataset,
    labels: np.ndarray,
    args: argparse.Namespace,
    *,
    samples_per_epoch: int,
) -> DataLoader:
    selected_labels = labels[dataset.indices].astype(str)
    names, counts = np.unique(selected_labels, return_counts=True)
    inverse = {name: 1.0 / count for name, count in zip(names, counts)}
    weights = torch.as_tensor(
        [inverse[value] for value in selected_labels], dtype=torch.double
    )
    sampler = WeightedRandomSampler(
        weights,
        num_samples=samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        # Pearson is estimated across samples for each protein.  A short tail
        # batch can contain fewer than pearson_min_samples observations and
        # makes the objective undefined, so joint training uses full batches.
        drop_last=True,
    )


def load_tcpa_arrays(
    prepared: Path,
    gene_names: Sequence[str],
    protein_names: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    arrays = prepared / "arrays"
    tables = prepared / "tables"
    rna = np.load(arrays / "tcpa_supervised_pretrain_rna.npy", mmap_mode="r")
    raw = np.load(arrays / "tcpa_supervised_pretrain_rppa.npy", mmap_mode="r")
    raw_mask = np.load(arrays / "tcpa_supervised_pretrain_rppa_mask.npy", mmap_mode="r")
    samples = pd.read_csv(tables / "tcpa_supervised_pretrain_sample_manifest.tsv", sep="\t")
    antibodies = pd.read_csv(
        tables / "tcpa_supervised_pretrain_antibody_manifest.tsv", sep="\t"
    )
    genes = pd.read_csv(tables / "tcpa_supervised_pretrain_gene_manifest.tsv", sep="\t")
    if tuple(genes["gene"].astype(str)) != tuple(map(str, gene_names)):
        raise ValueError("TCPA and CPTAC RNA gene order differs")
    if "cancer_label" not in samples.columns:
        raise ValueError("prepared TCPA sample manifest lacks cancer_label")
    protein_position = {value: index for index, value in enumerate(map(str, protein_names))}
    mapped_names = antibodies["model_protein"].astype(str).tolist()
    if len(mapped_names) != len(set(mapped_names)):
        raise ValueError("TCPA model-protein mapping is not one-to-one")
    mapped_indices = np.asarray([protein_position[value] for value in mapped_names], dtype=np.int64)
    train = np.flatnonzero(samples["split"].astype(str).to_numpy() == "train")
    validation = np.flatnonzero(samples["split"].astype(str).to_numpy() == "validation")
    rna_scaler = base.FeatureStandardizer.fit(rna, gene_names, indices=train)
    changed_rna = rna_scaler.transform(rna, gene_names).astype(np.float32)
    target_scaler = TrainFittedProteinScale.fit(
        raw,
        train,
        feature_names=mapped_names,
        mask=raw_mask,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
    )
    target = np.zeros(raw.shape, dtype=np.float32)
    target_mask = np.zeros(raw.shape, dtype=bool)
    for indices, clip in ((train, True), (validation, False)):
        changed = target_scaler.transform(
            raw[indices],
            mask=raw_mask[indices],
            feature_names=mapped_names,
            clip=clip,
        )
        target[indices] = np.nan_to_num(changed, nan=0.0)
        target_mask[indices] = np.isfinite(changed)
    return {
        "rna": changed_rna,
        "target": target,
        "mask": target_mask,
        "cancer_labels": samples["cancer_label"].astype(str).to_numpy(),
        "train": train,
        "validation": validation,
        "mapped_names": mapped_names,
        "mapped_indices": mapped_indices,
        "rna_valid_mask": genes["present_in_tcga"].astype(bool).to_numpy(),
        "rna_scaler": rna_scaler,
        "target_scaler": target_scaler,
    }


def main() -> int:
    args = parse_args()
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    dev.seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = args.output_dir / "tables"
    models = args.output_dir / "models"
    tables.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
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
    sealed = seal_outer_protein_labels(inputs["protein_raw"], partitions.outer_test)
    sealed = seal_outer_protein_labels(sealed, partitions.selection_validation)
    cptac_raw_mask = np.isfinite(sealed)
    cptac_rna_scaler = base.FeatureStandardizer.fit(
        inputs["rna"], inputs["gene_names"], indices=development.train_indices
    )
    cptac_rna = cptac_rna_scaler.transform(
        inputs["rna"], inputs["gene_names"]
    ).astype(np.float32)
    cptac_target_scaler = TrainFittedProteinScale.fit(
        sealed,
        development.train_indices,
        feature_names=inputs["protein_names"],
        mask=cptac_raw_mask,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
    )
    cptac_target = np.zeros_like(sealed, dtype=np.float32)
    cptac_mask = np.zeros_like(cptac_raw_mask, dtype=bool)
    for indices, clip in (
        (development.train_indices, True),
        (development.validation_indices, False),
    ):
        changed = cptac_target_scaler.transform(
            sealed[indices],
            mask=cptac_raw_mask[indices],
            feature_names=inputs["protein_names"],
            clip=clip,
        )
        cptac_target[indices] = np.nan_to_num(changed, nan=0.0)
        cptac_mask[indices] = np.isfinite(changed)

    tcpa = load_tcpa_arrays(
        args.tcpa_prepared_dir,
        inputs["gene_names"],
        inputs["protein_names"],
        args,
    )
    cptac_cancers = manifest[args.strata_column].astype(str).to_numpy()
    cancers, cptac_cancer_index, tcpa_cancer_index = combined_cancer_vocabulary(
        cptac_cancers,
        development.train_indices,
        tcpa["cancer_labels"],
        tcpa["train"],
    )
    cptac_dataset = PlatformDataset(
        cptac_rna,
        cptac_target,
        cptac_mask,
        cptac_cancer_index,
        development.train_indices,
    )
    tcpa_dataset = PlatformDataset(
        tcpa["rna"],
        tcpa["target"],
        tcpa["mask"],
        tcpa_cancer_index,
        tcpa["train"],
    )
    samples_per_epoch = int(development.train_indices.size)
    cptac_loader = balanced_loader(
        cptac_dataset, cptac_cancers, args, samples_per_epoch=samples_per_epoch
    )
    tcpa_loader = balanced_loader(
        tcpa_dataset,
        tcpa["cancer_labels"],
        args,
        samples_per_epoch=samples_per_epoch,
    )

    graph = ProteinGraphArtifact.load(args.protein_graph_artifact)
    if graph.protein_names != tuple(map(str, inputs["protein_names"])):
        raise ValueError("protein graph order differs from output vocabulary")
    local_prior = build_protein_local_gene_prior(
        inputs["gene_names"],
        inputs["protein_names"],
        base.build_parent_gene_index(inputs),
        graph,
        max_local_genes=args.max_local_genes,
    )
    calibration_mask = np.zeros(len(inputs["protein_names"]), dtype=bool)
    calibration_mask[tcpa["mapped_indices"]] = True
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
    model = TCGACPTACHybridProteinTranslator(
        config, local_prior, tcpa_calibrated_protein_mask=calibration_mask
    ).to(device)
    model.fix_projection_matrices_()
    if args.smoke_only:
        cptac_batch = next(iter(cptac_loader))
        tcpa_batch = next(iter(tcpa_loader))
        c_rna = cptac_batch["rna"].to(device)
        c_target = cptac_batch["target"].to(device)
        c_mask = cptac_batch["mask"].to(device)
        c_cancer = cptac_batch["cancer_index"].to(device)
        t_rna = tcpa_batch["rna"].to(device)
        t_target = tcpa_batch["target"].to(device)
        t_mask = tcpa_batch["mask"].to(device)
        t_cancer = tcpa_batch["cancer_index"].to(device)
        tcpa_indices_device = torch.as_tensor(tcpa["mapped_indices"], device=device)
        tcpa_rna_mask = torch.as_tensor(
            tcpa["rna_valid_mask"], device=device
        ).unsqueeze(0)
        with dev.autocast_context(args):
            c_output = model(
                c_rna,
                platform_index=torch.full_like(c_cancer, CPTAC_PLATFORM),
                cancer_index=c_cancer,
                return_hidden=True,
            )
            t_output = model(
                t_rna,
                platform_index=torch.full_like(t_cancer, TCPA_PLATFORM),
                cancer_index=t_cancer,
                rna_valid_mask=tcpa_rna_mask.expand(t_rna.shape[0], -1),
                protein_indices=tcpa_indices_device,
                return_hidden=True,
            )
            c_loss = masked_per_protein_mse(c_output["protein"], c_target, c_mask)
            t_loss = masked_per_protein_mse(t_output["protein"], t_target, t_mask)
            c_pearson = masked_per_protein_pearson_loss(
                c_output["protein"],
                c_target,
                c_mask,
                minimum_observations=args.pearson_min_samples,
            )
            t_pearson = masked_per_protein_pearson_loss(
                t_output["protein"],
                t_target,
                t_mask,
                minimum_observations=args.pearson_min_samples,
            )
            loss = (
                args.pearson_weight
                * (c_pearson + args.tcpa_loss_weight * t_pearson)
                + args.mse_weight * (c_loss + args.tcpa_loss_weight * t_loss)
            )
        loss.backward()
        smoke = {
            "status": "passed",
            "cptac_output_shape": list(c_output["protein"].shape),
            "tcpa_output_shape": list(t_output["protein"].shape),
            "n_tcpa_calibrated_proteins": int(calibration_mask.sum()),
            "cptac_loss": float(c_loss.detach().float().cpu()),
            "tcpa_loss": float(t_loss.detach().float().cpu()),
            "rna_encoder_gradient_present": any(
                parameter.grad is not None
                for parameter in model.rna_encoder.parameters()
            ),
            "locked_selection_validation_labels_evaluated": False,
            "outer_test_evaluated": False,
        }
        encoded = json.dumps(smoke, ensure_ascii=False, indent=2)
        (args.output_dir / "smoke_summary.json").write_text(encoded, encoding="utf-8")
        print(encoded)
        return 0
    updates = math.ceil(len(cptac_loader) / args.gradient_accumulation) * args.max_epochs
    optimizer, scheduler = dev.optimizer_and_scheduler(model, args, max(updates, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    tcpa_indices_device = torch.as_tensor(tcpa["mapped_indices"], device=device)
    tcpa_rna_mask = torch.as_tensor(tcpa["rna_valid_mask"], device=device).unsqueeze(0)
    cptac_validation_arrays = dev.DevelopmentArrays(
        cptac_rna, cptac_target, cptac_mask, cptac_cancer_index
    )
    best_spearman = float("-inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {
            "loss": 0.0,
            "cptac_mse": 0.0,
            "tcpa_mse": 0.0,
            "cptac_pearson": 0.0,
            "tcpa_pearson": 0.0,
            "gate": 0.0,
        }
        for step, (cptac_batch, tcpa_batch) in enumerate(zip(cptac_loader, tcpa_loader)):
            c_rna = cptac_batch["rna"].to(device, non_blocking=True)
            c_target = cptac_batch["target"].to(device, non_blocking=True)
            c_mask = cptac_batch["mask"].to(device, non_blocking=True)
            c_cancer = cptac_batch["cancer_index"].to(device, non_blocking=True)
            t_rna = tcpa_batch["rna"].to(device, non_blocking=True)
            t_target = tcpa_batch["target"].to(device, non_blocking=True)
            t_mask = tcpa_batch["mask"].to(device, non_blocking=True)
            t_cancer = tcpa_batch["cancer_index"].to(device, non_blocking=True)
            with dev.autocast_context(args):
                c_output = model(
                    c_rna,
                    platform_index=torch.full_like(c_cancer, CPTAC_PLATFORM),
                    cancer_index=c_cancer,
                    return_hidden=True,
                )
                t_output = model(
                    t_rna,
                    platform_index=torch.full_like(t_cancer, TCPA_PLATFORM),
                    cancer_index=t_cancer,
                    rna_valid_mask=tcpa_rna_mask.expand(t_rna.shape[0], -1),
                    protein_indices=tcpa_indices_device,
                    return_hidden=True,
                )
                cptac_loss = masked_per_protein_mse(
                    c_output["protein"], c_target, c_mask
                )
                tcpa_loss = masked_per_protein_mse(
                    t_output["protein"], t_target, t_mask
                )
                cptac_pearson_loss = masked_per_protein_pearson_loss(
                    c_output["protein"],
                    c_target,
                    c_mask,
                    minimum_observations=args.pearson_min_samples,
                )
                tcpa_pearson_loss = masked_per_protein_pearson_loss(
                    t_output["protein"],
                    t_target,
                    t_mask,
                    minimum_observations=args.pearson_min_samples,
                )
                gate_loss = 0.5 * (
                    c_output["map_gate_penalty"] + t_output["map_gate_penalty"]
                )
                loss = (
                    args.pearson_weight
                    * (
                        cptac_pearson_loss
                        + args.tcpa_loss_weight * tcpa_pearson_loss
                    )
                    + args.mse_weight
                    * (cptac_loss + args.tcpa_loss_weight * tcpa_loss)
                    + args.gate_penalty * gate_loss
                )
            window_start = (step // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(args.gradient_accumulation, len(cptac_loader) - window_start)
            scaler.scale(loss / window_size).backward()
            totals["loss"] += float(loss.detach().float().cpu())
            totals["cptac_mse"] += float(cptac_loss.detach().float().cpu())
            totals["tcpa_mse"] += float(tcpa_loss.detach().float().cpu())
            totals["cptac_pearson"] += float(
                cptac_pearson_loss.detach().float().cpu()
            )
            totals["tcpa_pearson"] += float(
                tcpa_pearson_loss.detach().float().cpu()
            )
            totals["gate"] += float(gate_loss.detach().float().cpu())
            synchronize = (
                (step + 1) % args.gradient_accumulation == 0
                or step + 1 == len(cptac_loader)
            )
            if synchronize:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        prediction = dev.predict(
            model,
            cptac_validation_arrays,
            development.validation_indices,
            args,
            device,
        )
        per_protein, cosine, summary = dev.validation_tables(
            prediction,
            cptac_validation_arrays,
            development.validation_indices,
            inputs["protein_names"],
            args.batch_size,
        )
        record = {
            "epoch": epoch,
            "train_loss": totals["loss"] / len(cptac_loader),
            "train_cptac_mse": totals["cptac_mse"] / len(cptac_loader),
            "train_tcpa_mse": totals["tcpa_mse"] / len(cptac_loader),
            "train_cptac_pearson_loss": totals["cptac_pearson"] / len(cptac_loader),
            "train_tcpa_pearson_loss": totals["tcpa_pearson"] / len(cptac_loader),
            "train_map_gate_penalty": totals["gate"] / len(cptac_loader),
            "development_median_spearman": summary["median_spearman"],
            "development_mse": summary["validation_mse"],
            "development_batch_cosine": summary["validation_mean_batch_cosine"],
            "development_sd_ratio": summary["median_predicted_to_observed_sd_ratio"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        pd.DataFrame(history).to_csv(
            tables / "training_history_running.tsv", sep="\t", index=False
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
            dev.atomic_torch_save(
                {
                    "schema_version": 1,
                    "model_family": MODEL_FAMILY,
                    "epoch": epoch,
                    "development_median_spearman": monitor,
                    "model_config": config.to_dict(),
                    "model_state": best_state,
                },
                models / "best_checkpoint_running.pt",
            )
        else:
            stale += 1
        dev.atomic_torch_save(
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
            models / "latest_checkpoint_running.pt",
        )
        if epoch >= args.min_epochs and stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("training produced no selectable checkpoint")
    model.load_state_dict(best_state)
    prediction = dev.predict(
        model,
        cptac_validation_arrays,
        development.validation_indices,
        args,
        device,
    )
    per_protein, cosine, summary = dev.validation_tables(
        prediction,
        cptac_validation_arrays,
        development.validation_indices,
        inputs["protein_names"],
        args.batch_size,
    )
    per_protein.to_csv(tables / "development_per_protein_metrics.tsv", sep="\t", index=False)
    cosine.to_csv(tables / "development_batch_cosine.tsv", sep="\t", index=False)
    pd.DataFrame(history).to_csv(tables / "training_history.tsv", sep="\t", index=False)
    np.savez_compressed(
        tables / "development_split_indices.npz",
        development_train=development.train_indices,
        development_validation=development.validation_indices,
        locked_selection_validation=partitions.selection_validation,
        locked_outer_test=partitions.outer_test,
        tcpa_train=tcpa["train"],
        tcpa_validation=tcpa["validation"],
    )
    torch.save(
        {
            "schema_version": 1,
            "model_family": MODEL_FAMILY,
            "model_config": config.to_dict(),
            "model_state": best_state,
            "selected_epoch": best_epoch,
            "gene_names": inputs["gene_names"],
            "protein_names": inputs["protein_names"],
            "cancer_vocabulary": cancers,
            "tcpa_model_proteins": tcpa["mapped_names"],
            "tcpa_model_protein_indices": tcpa["mapped_indices"],
            "cptac_rna_standardizer": {
                "mean": cptac_rna_scaler.mean,
                "scale": cptac_rna_scaler.scale,
            },
            "tcpa_rna_standardizer": {
                "mean": tcpa["rna_scaler"].mean,
                "scale": tcpa["rna_scaler"].scale,
            },
            "cptac_target_scale": cptac_target_scaler.state_dict(),
            "tcpa_target_scale": tcpa["target_scaler"].state_dict(),
            "architecture_contract": model.architecture_contract(),
            "protein_graph_sha256": sha256_file(args.protein_graph_artifact),
        },
        models / f"{MODEL_FAMILY}_fold0.pt",
    )
    summary.update(
        {
            "model_family": MODEL_FAMILY,
            "selected_epoch": best_epoch,
            "n_development_train": int(development.train_indices.size),
            "n_development_validation": int(development.validation_indices.size),
            "n_tcpa_training_samples": int(tcpa["train"].size),
            "n_tcpa_training_proteins": int(tcpa["mapped_indices"].size),
            "tcpa_loss_weight": args.tcpa_loss_weight,
            "pearson_weight": args.pearson_weight,
            "mse_weight": args.mse_weight,
            "pearson_min_samples": args.pearson_min_samples,
            "platform_balanced_batches": True,
            "cancer_balanced_sampling": True,
            "locked_selection_validation_labels_evaluated": False,
            "outer_test_evaluated": False,
            "parameter_count": model.parameter_count(),
            "runtime_seconds": time.time() - started,
        }
    )
    encoded = json.dumps(summary, ensure_ascii=False, indent=2)
    (args.output_dir / "final_summary.json").write_text(encoded, encoding="utf-8")
    (args.output_dir / "done.txt").write_text(encoded, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
