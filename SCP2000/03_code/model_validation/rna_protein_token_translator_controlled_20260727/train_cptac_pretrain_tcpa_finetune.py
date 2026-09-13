"""Full CPTAC pretraining followed by TCGA-TCPA supervised fine-tuning.

The fixed 916/229/286 CPTAC contract is preserved.  Stage one trains on all
916 selection-training samples and selects on the 229 selection-validation
samples.  Stage two traverses the complete eligible TCGA-TCPA training split,
selects on its own validation split, and audits the frozen CPTAC 229 partition
before and after fine-tuning.  The 286 outer-test labels are never evaluated.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.utils.extmath import randomized_svd

try:
    from . import train_protein as base
    from . import train_hybrid_protein_development_screen as dev
    from . import train_hybrid_tcga_cptac_joint_development as joint
    from .candidate_groupwise_rna_normalization import (
        build_strict_inner_partitions,
        seal_outer_protein_labels,
    )
    from .candidate_multiscale_protein_query import (
        LocalGenePrior,
        build_protein_local_gene_prior,
    )
    from .candidate_tcga_cptac_hybrid_protein import (
        CPTAC_PLATFORM,
        TCPA_PLATFORM,
        HybridProteinConfig,
        TCGACPTACHybridProteinTranslator,
    )
    from .hybrid_protein_training_contract import (
        TrainFittedProteinScale,
        masked_per_protein_mse,
        masked_per_protein_pearson_loss,
        masked_per_protein_variance_loss,
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
    import train_hybrid_protein_development_screen as dev
    import train_hybrid_tcga_cptac_joint_development as joint
    from candidate_groupwise_rna_normalization import (
        build_strict_inner_partitions,
        seal_outer_protein_labels,
    )
    from candidate_multiscale_protein_query import (
        LocalGenePrior,
        build_protein_local_gene_prior,
    )
    from candidate_tcga_cptac_hybrid_protein import (
        CPTAC_PLATFORM,
        TCPA_PLATFORM,
        HybridProteinConfig,
        TCGACPTACHybridProteinTranslator,
    )
    from hybrid_protein_training_contract import (
        TrainFittedProteinScale,
        masked_per_protein_mse,
        masked_per_protein_pearson_loss,
        masked_per_protein_variance_loss,
    )
    from metrics import (
        batchwise_flattened_cosine,
        masked_mse,
        per_protein_spearman,
        summarize_spearman,
    )
    from protein_graph_prior import ProteinGraphArtifact, sha256_file


MODEL_FAMILY = "cptac_full_pretrain_tcpa_finetune"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tcpa-prepared-dir", type=Path, required=True)
    parser.add_argument("--protein-graph-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
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

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim-head", type=int, default=64)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--map-hidden", type=int, default=1024)
    parser.add_argument("--map-states", type=int, default=512)
    parser.add_argument("--global-states", type=int, default=128)
    parser.add_argument("--module-states", type=int, default=128)
    parser.add_argument("--module-depth", type=int, default=2)
    parser.add_argument("--protein-chunk", type=int, default=256)
    parser.add_argument(
        "--max-local-genes",
        type=int,
        default=1,
        help="Local RNA width; values above one add fixed STRING-neighbour RNAs.",
    )
    parser.add_argument(
        "--direct-local-linear-genes",
        type=int,
        default=0,
        help=(
            "Width of a separate sparse STRING-RNA linear value branch; zero "
            "disables it without widening local attention."
        ),
    )
    parser.add_argument("--initial-local-linear-mix", type=float, default=0.5)
    parser.add_argument("--initialize-direct-local-ridge", action="store_true")
    parser.add_argument("--direct-local-ridge-alpha", type=float, default=10.0)
    parser.add_argument(
        "--direct-local-supervised-selection",
        action="store_true",
        help=(
            "Select each protein's direct local RNA genes using only the 916 "
            "selection-training samples instead of fixed STRING neighbours."
        ),
    )
    parser.add_argument("--direct-global-components", type=int, default=0)
    parser.add_argument("--initial-global-linear-mix", type=float, default=0.5)
    parser.add_argument("--initialize-direct-global-ridge", action="store_true")
    parser.add_argument("--direct-global-ridge-alpha", type=float, default=100.0)
    parser.add_argument("--direct-supervised-linear-genes", type=int, default=0)
    parser.add_argument("--initial-supervised-linear-mix", type=float, default=0.25)
    parser.add_argument("--initialize-direct-supervised-ridge", action="store_true")
    parser.add_argument("--direct-supervised-ridge-alpha", type=float, default=100.0)
    parser.add_argument(
        "--direct-initialization-checkpoint",
        type=Path,
        help=(
            "Reuse frozen train-fitted direct branch values and supervised "
            "gene indices from a compatible 916-sample checkpoint."
        ),
    )
    parser.add_argument(
        "--freeze-direct-linear-branches",
        action="store_true",
        help=(
            "Keep train-fitted direct local, global, supervised weights and "
            "their convex mixing coefficients fixed during neural training."
        ),
    )
    parser.add_argument(
        "--freeze-direct-linear-values-only",
        action="store_true",
        help=(
            "Keep train-fitted direct branch weights and biases fixed while "
            "allowing a joint value-fusion gate to train."
        ),
    )
    parser.add_argument("--joint-direct-value-fusion", action="store_true")
    parser.add_argument("--initial-joint-neural-weight", type=float, default=0.0)
    parser.add_argument("--branch-dropout", type=float, default=0.05)
    parser.add_argument("--protein-subsample", type=int, default=2048)

    parser.add_argument("--pearson-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.2)
    parser.add_argument("--variance-weight", type=float, default=0.5)
    parser.add_argument(
        "--within-study-pearson-weight",
        type=float,
        default=0.0,
        help=(
            "Additional Pearson loss between the neural CPTAC value and the "
            "target after subtracting its train-fitted study-protein mean."
        ),
    )
    parser.add_argument("--pearson-min-samples", type=int, default=8)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--finetune-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--pretrain-min-epochs", type=int, default=12)
    parser.add_argument("--pretrain-patience", type=int, default=7)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--finetune-min-epochs", type=int, default=4)
    parser.add_argument("--finetune-patience", type=int, default=3)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--translator-lr", type=float, default=2e-4)
    parser.add_argument("--finetune-encoder-lr", type=float, default=2e-6)
    parser.add_argument("--finetune-translator-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--cptac-study-baseline",
        action="store_true",
        help=(
            "Add a train-fitted study-by-protein mean to the CPTAC neural output; "
            "the neural network then learns within-study variation."
        ),
    )
    parser.add_argument(
        "--cptac-study-center-rna",
        action="store_true",
        help=(
            "Subtract train-fitted study-by-gene RNA means before scaling; "
            "requires the matching CPTAC study-protein baseline."
        ),
    )
    parser.add_argument(
        "--stop-after-pretrain",
        action="store_true",
        help="Finish after CPTAC pretraining and its complete 229-sample audit.",
    )
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-stage", choices=("cptac", "tcpa"), default="cptac")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-genes", type=int)
    parser.add_argument("--max-proteins", type=int)
    return parser.parse_args()


def cognate_only_prior(parent_gene_index: Sequence[int], n_genes: int) -> LocalGenePrior:
    parent = np.asarray(parent_gene_index, dtype=np.int64)
    if parent.ndim != 1 or (parent < -1).any() or (parent >= n_genes).any():
        raise ValueError("invalid parent-gene index")
    valid = parent >= 0
    return LocalGenePrior(
        gene_index=np.where(valid, parent, -1).reshape(-1, 1),
        gene_mask=valid.reshape(-1, 1),
        relation_index=np.where(valid, 0, -1).reshape(-1, 1),
        edge_strength=np.where(valid, 1.0, 0.0).astype(np.float32).reshape(-1, 1),
        relation_names=("cognate_rna",),
        metadata={
            "construction": "cognate RNA only",
            "uses_string_neighbors": False,
            "uses_validation_samples": False,
        },
    )


def cptac_local_prior(
    gene_names: Sequence[str],
    protein_names: Sequence[str],
    parent_gene_index: Sequence[int],
    graph: ProteinGraphArtifact,
    max_local_genes: int,
) -> LocalGenePrior:
    """Select the controlled cognate-only or cognate-plus-STRING local reader."""

    if max_local_genes < 1:
        raise ValueError("max_local_genes must be positive")
    if max_local_genes == 1:
        return cognate_only_prior(parent_gene_index, len(gene_names))
    return build_protein_local_gene_prior(
        gene_names,
        protein_names,
        parent_gene_index,
        graph,
        max_local_genes=max_local_genes,
    )


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


class StudyBaselineDevelopmentDataset(dev.DevelopmentDataset):
    """Development dataset carrying one fixed train-fitted baseline per sample."""

    def __init__(
        self,
        arrays: dev.DevelopmentArrays,
        indices: np.ndarray,
        study_baseline: np.ndarray,
    ) -> None:
        super().__init__(arrays, indices)
        baseline = np.asarray(study_baseline, dtype=np.float32)
        if baseline.shape != arrays.target.shape:
            raise ValueError("study baseline must match the complete CPTAC target matrix")
        self.study_baseline = baseline

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        result = super().__getitem__(item)
        row = int(self.indices[item])
        result["study_baseline"] = torch.from_numpy(self.study_baseline[row])
        return result


def fit_train_study_protein_baseline(
    target: np.ndarray,
    mask: np.ndarray,
    fit_indices: np.ndarray,
    study_labels: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit study-protein means using only the supplied training indices."""

    target = np.asarray(target, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    fit = np.asarray(fit_indices, dtype=np.int64)
    studies = np.asarray(study_labels, dtype=str)
    if target.shape != mask.shape or target.shape[0] != studies.size:
        raise ValueError("study baseline inputs have incompatible shapes")
    if fit.ndim != 1 or fit.size == 0:
        raise ValueError("study baseline requires non-empty one-dimensional fit indices")
    fit_target = target[fit]
    fit_mask = mask[fit]
    counts = fit_mask.sum(axis=0)
    global_mean = np.divide(
        np.where(fit_mask, fit_target, 0.0).sum(axis=0, dtype=np.float64),
        counts,
        out=np.zeros(target.shape[1], dtype=np.float64),
        where=counts > 0,
    ).astype(np.float32)
    vocabulary = tuple(sorted(set(studies[fit].tolist())))
    mapping = {study: position + 1 for position, study in enumerate(vocabulary)}
    means = np.repeat(global_mean[None, :], len(vocabulary) + 1, axis=0)
    study_counts = np.zeros_like(means, dtype=np.int64)
    study_counts[0] = counts
    for study, position in mapping.items():
        rows = fit[studies[fit] == study]
        observed = mask[rows]
        current_counts = observed.sum(axis=0)
        current_mean = np.divide(
            np.where(observed, target[rows], 0.0).sum(axis=0, dtype=np.float64),
            current_counts,
            out=global_mean.astype(np.float64, copy=True),
            where=current_counts > 0,
        )
        means[position] = current_mean.astype(np.float32)
        study_counts[position] = current_counts
    encoded = np.asarray([mapping.get(study, 0) for study in studies], dtype=np.int64)
    sample_baseline = means[encoded]
    state = {
        "study_vocabulary": ("<unknown>", *vocabulary),
        "study_protein_mean": means,
        "study_protein_count": study_counts,
        "unknown_study_fallback": "selection_train_global_protein_mean",
        "fit_sample_count": int(fit.size),
    }
    return sample_baseline.astype(np.float32, copy=False), state


def fit_transform_train_study_centered_rna(
    rna: np.ndarray,
    fit_indices: np.ndarray,
    study_labels: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply train-fitted study centering and a shared train-fitted gene scale."""

    values = np.asarray(rna, dtype=np.float32)
    fit = np.asarray(fit_indices, dtype=np.int64)
    studies = np.asarray(study_labels, dtype=str)
    if values.ndim != 2 or values.shape[0] != studies.size:
        raise ValueError("study-centred RNA inputs have incompatible shapes")
    if fit.ndim != 1 or fit.size == 0:
        raise ValueError("study-centred RNA requires non-empty fit indices")
    if not np.isfinite(values).all():
        raise ValueError("study-centred RNA requires finite expression values")
    global_mean = values[fit].mean(axis=0, dtype=np.float64).astype(np.float32)
    vocabulary = tuple(sorted(set(studies[fit].tolist())))
    mapping = {study: position + 1 for position, study in enumerate(vocabulary)}
    means = np.repeat(global_mean[None, :], len(vocabulary) + 1, axis=0)
    counts = np.zeros(len(vocabulary) + 1, dtype=np.int64)
    counts[0] = fit.size
    for study, position in mapping.items():
        rows = fit[studies[fit] == study]
        means[position] = values[rows].mean(axis=0, dtype=np.float64).astype(np.float32)
        counts[position] = rows.size
    encoded = np.asarray([mapping.get(study, 0) for study in studies], dtype=np.int64)
    centered = values - means[encoded]
    scale = centered[fit].std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    transformed = centered / scale
    state = {
        "mode": "selection_train_study_gene_center_then_shared_scale",
        "study_vocabulary": ("<unknown>", *vocabulary),
        "study_gene_mean": means,
        "study_sample_count": counts,
        "gene_scale": scale,
        "unknown_study_fallback": "selection_train_global_gene_mean",
        "fit_sample_count": int(fit.size),
    }
    return transformed.astype(np.float32, copy=False), state


def fit_direct_local_ridge_initialization(
    rna: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    study_baseline: np.ndarray,
    fit_indices: np.ndarray,
    prior: LocalGenePrior,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one masked train-only local ridge value per protein for initialization."""

    if alpha < 0:
        raise ValueError("direct local ridge alpha must be non-negative")
    values = np.asarray(rna, dtype=np.float64)
    targets = np.asarray(target, dtype=np.float64)
    observed_mask = np.asarray(mask, dtype=bool)
    baseline = np.asarray(study_baseline, dtype=np.float64)
    fit = np.asarray(fit_indices, dtype=np.int64)
    if targets.shape != observed_mask.shape or targets.shape != baseline.shape:
        raise ValueError("direct local ridge target inputs have incompatible shapes")
    if values.shape[0] != targets.shape[0] or prior.n_targets != targets.shape[1]:
        raise ValueError("direct local ridge RNA and prior dimensions are incompatible")
    weights = np.zeros(prior.gene_index.shape, dtype=np.float32)
    biases = np.zeros(prior.n_targets, dtype=np.float32)
    fitted = np.zeros(prior.n_targets, dtype=bool)
    observation_counts = np.zeros(prior.n_targets, dtype=np.int64)
    for protein in range(prior.n_targets):
        positions = np.flatnonzero(prior.gene_mask[protein])
        genes = prior.gene_index[protein, positions]
        use = observed_mask[fit, protein]
        observation_counts[protein] = int(use.sum())
        if genes.size == 0 or use.sum() < 8:
            continue
        x = values[fit[use]][:, genes]
        y = targets[fit[use], protein] - baseline[fit[use], protein]
        x_mean = x.mean(axis=0)
        y_mean = float(y.mean())
        centered_x = x - x_mean
        centered_y = y - y_mean
        gram = centered_x.T @ centered_x
        gram.flat[:: gram.shape[0] + 1] += alpha
        coefficient = np.linalg.solve(gram, centered_x.T @ centered_y)
        weights[protein, positions] = coefficient.astype(np.float32)
        biases[protein] = np.float32(y_mean - x_mean @ coefficient)
        fitted[protein] = True
    state = {
        "fit_sample_count": int(fit.size),
        "alpha": float(alpha),
        "fitted_proteins": int(fitted.sum()),
        "minimum_observations": 8,
        "uses_validation_samples": False,
        "uses_outer_test_samples": False,
        "observation_counts": observation_counts,
    }
    return weights, biases, state


def fit_supervised_local_gene_prior(
    rna: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    study_baseline: np.ndarray,
    fit_indices: np.ndarray,
    gene_names: Sequence[str],
    protein_names: Sequence[str],
    *,
    selected_genes: int,
    protein_block_size: int = 128,
) -> LocalGenePrior:
    """Select protein-specific RNA genes from selection-training covariance only."""

    values = np.asarray(rna, dtype=np.float32)
    targets = np.asarray(target, dtype=np.float32)
    observed_mask = np.asarray(mask, dtype=bool)
    baseline = np.asarray(study_baseline, dtype=np.float32)
    fit = np.asarray(fit_indices, dtype=np.int64)
    if selected_genes < 1 or protein_block_size < 1:
        raise ValueError("supervised local prior controls must be positive")
    if targets.shape != observed_mask.shape or targets.shape != baseline.shape:
        raise ValueError("supervised local prior target inputs have incompatible shapes")
    if values.shape[0] != targets.shape[0]:
        raise ValueError("supervised local prior RNA and target rows differ")
    if values.shape[1] != len(gene_names) or targets.shape[1] != len(protein_names):
        raise ValueError("supervised local prior vocabulary dimensions differ")
    if fit.ndim != 1 or fit.size < 8:
        raise ValueError("supervised local prior requires at least eight fit samples")
    width = min(int(selected_genes), values.shape[1])
    train_rna = np.ascontiguousarray(values[fit], dtype=np.float32)
    residual = np.where(observed_mask, targets - baseline, 0.0).astype(np.float32)
    x_scale = np.sqrt(np.square(train_rna, dtype=np.float64).sum(axis=0))
    x_scale = np.maximum(x_scale, 1e-8)
    gene_index = np.empty((targets.shape[1], width), dtype=np.int64)
    edge_strength = np.zeros((targets.shape[1], width), dtype=np.float32)
    for start in range(0, targets.shape[1], protein_block_size):
        stop = min(start + protein_block_size, targets.shape[1])
        y_block = np.ascontiguousarray(residual[fit, start:stop], dtype=np.float32)
        covariance = train_rna.T @ y_block
        y_scale = np.sqrt(np.square(y_block, dtype=np.float64).sum(axis=0))
        score = np.abs(covariance) / np.maximum(
            x_scale[:, None] * y_scale[None, :], 1e-8
        )
        top = np.argpartition(score, -width, axis=0)[-width:]
        for offset, protein in enumerate(range(start, stop)):
            genes = top[:, offset]
            order = np.argsort(-score[genes, offset], kind="stable")
            genes = genes[order]
            gene_index[protein] = genes
            edge_strength[protein] = score[genes, offset].astype(np.float32)
    return LocalGenePrior(
        gene_index=gene_index,
        gene_mask=np.ones_like(gene_index, dtype=bool),
        relation_index=np.zeros_like(gene_index, dtype=np.int64),
        edge_strength=edge_strength,
        relation_names=("train_supervised_rna",),
        metadata={
            "construction": "selection-training protein-specific RNA covariance",
            "selected_genes": width,
            "fit_sample_count": int(fit.size),
            "uses_string_neighbors": False,
            "uses_validation_samples": False,
            "uses_outer_test_samples": False,
        },
    )


def freeze_direct_linear_branches(
    model: TCGACPTACHybridProteinTranslator,
) -> tuple[str, ...]:
    """Freeze all train-fitted direct value parameters and convex mixes."""

    names = (
        "direct_local_weight",
        "direct_local_bias",
        "direct_local_mix_logit",
        "direct_global_weight",
        "direct_global_bias",
        "direct_global_mix_logit",
        "direct_supervised_weight",
        "direct_supervised_bias",
        "direct_supervised_mix_logit",
    )
    frozen: list[str] = []
    for name in names:
        parameter = getattr(model, name, None)
        if parameter is not None:
            parameter.requires_grad_(False)
            frozen.append(name)
    return tuple(frozen)


def freeze_direct_linear_values(
    model: TCGACPTACHybridProteinTranslator,
) -> tuple[str, ...]:
    """Freeze train-fitted branch values and leave fusion parameters trainable."""

    names = (
        "direct_local_weight",
        "direct_local_bias",
        "direct_global_weight",
        "direct_global_bias",
        "direct_supervised_weight",
        "direct_supervised_bias",
    )
    frozen: list[str] = []
    for name in names:
        parameter = getattr(model, name, None)
        if parameter is not None:
            parameter.requires_grad_(False)
            frozen.append(name)
    return tuple(frozen)


def fit_direct_global_pca_ridge_initialization(
    rna: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    study_baseline: np.ndarray,
    fit_indices: np.ndarray,
    *,
    components: int,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a train-only fixed RNA-PCA projection and protein ridge readouts."""

    if components < 1 or alpha < 0:
        raise ValueError("direct global PCA ridge controls are invalid")
    values = np.asarray(rna, dtype=np.float64)
    targets = np.asarray(target, dtype=np.float64)
    observed_mask = np.asarray(mask, dtype=bool)
    baseline = np.asarray(study_baseline, dtype=np.float64)
    fit = np.asarray(fit_indices, dtype=np.int64)
    if targets.shape != observed_mask.shape or targets.shape != baseline.shape:
        raise ValueError("direct global ridge target inputs have incompatible shapes")
    if values.shape[0] != targets.shape[0] or components >= fit.size:
        raise ValueError("direct global PCA dimensions are incompatible")
    _, singular_values, axes = randomized_svd(
        values[fit],
        n_components=components,
        n_iter=5,
        random_state=seed,
    )
    scores = values[fit] @ axes.T
    score_scale = scores.std(axis=0)
    score_scale = np.where(score_scale > 1e-8, score_scale, 1.0)
    projection = (axes.T / score_scale).astype(np.float32)
    standardized_scores = values[fit] @ projection
    residual = targets - baseline
    weights = np.zeros((targets.shape[1], components), dtype=np.float32)
    biases = np.zeros(targets.shape[1], dtype=np.float32)
    fitted = np.zeros(targets.shape[1], dtype=bool)
    observation_counts = np.zeros(targets.shape[1], dtype=np.int64)
    for protein in range(targets.shape[1]):
        use = observed_mask[fit, protein]
        observation_counts[protein] = int(use.sum())
        if use.sum() < 8:
            continue
        x = standardized_scores[use]
        y = residual[fit[use], protein]
        x_mean = x.mean(axis=0)
        y_mean = float(y.mean())
        centered_x = x - x_mean
        gram = centered_x.T @ centered_x
        gram.flat[:: gram.shape[0] + 1] += alpha
        coefficient = np.linalg.solve(gram, centered_x.T @ (y - y_mean))
        weights[protein] = coefficient.astype(np.float32)
        biases[protein] = np.float32(y_mean - x_mean @ coefficient)
        fitted[protein] = True
    state = {
        "fit_sample_count": int(fit.size),
        "components": int(components),
        "alpha": float(alpha),
        "fitted_proteins": int(fitted.sum()),
        "minimum_observations": 8,
        "explained_singular_value_fraction": float(
            np.square(singular_values).sum()
            / np.square(np.linalg.norm(values[fit], ord="fro"))
        ),
        "uses_validation_samples": False,
        "uses_outer_test_samples": False,
        "observation_counts": observation_counts,
    }
    return projection, weights, biases, state


def epoch_protein_schedule(
    n_proteins: int,
    n_steps: int,
    width: int,
    *,
    seed: int,
) -> list[np.ndarray]:
    if width >= n_proteins:
        complete = np.arange(n_proteins, dtype=np.int64)
        return [complete] * n_steps
    rng = np.random.default_rng(seed)
    needed = n_steps * width
    parts: list[np.ndarray] = []
    while sum(part.size for part in parts) < needed:
        parts.append(rng.permutation(n_proteins).astype(np.int64))
    stream = np.concatenate(parts)[:needed]
    return [stream[start : start + width] for start in range(0, needed, width)]


def combined_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    args: argparse.Namespace,
) -> tuple[Tensor, dict[str, Tensor]]:
    pearson = masked_per_protein_pearson_loss(
        prediction,
        target,
        mask,
        minimum_observations=args.pearson_min_samples,
    )
    mse = masked_per_protein_mse(prediction, target, mask)
    variance = masked_per_protein_variance_loss(
        prediction,
        target,
        mask,
        minimum_observations=args.pearson_min_samples,
    )
    loss = (
        args.pearson_weight * pearson
        + args.mse_weight * mse
        + args.variance_weight * variance
    )
    return loss, {"pearson": pearson, "mse": mse, "variance": variance}


def combined_cptac_loss(
    neural_prediction: Tensor,
    study_baseline: Tensor,
    target: Tensor,
    mask: Tensor,
    args: argparse.Namespace,
) -> tuple[Tensor, dict[str, Tensor], Tensor]:
    """Combine final-value losses with optional study-centered correlation."""

    prediction = neural_prediction + study_baseline
    loss, pieces = combined_loss(prediction, target, mask, args)
    within_study_pearson = prediction.new_zeros(())
    if args.within_study_pearson_weight > 0:
        within_study_pearson = masked_per_protein_pearson_loss(
            neural_prediction,
            target - study_baseline,
            mask,
            minimum_observations=args.pearson_min_samples,
        )
        loss = loss + args.within_study_pearson_weight * within_study_pearson
    pieces["within_study_pearson"] = within_study_pearson
    return loss, pieces, prediction


def optimizer_for_stage(
    model: TCGACPTACHybridProteinTranslator,
    args: argparse.Namespace,
    updates: int,
    *,
    encoder_lr: float,
    translator_lr: float,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    stage_args = copy.copy(args)
    stage_args.encoder_lr = encoder_lr
    stage_args.translator_lr = translator_lr
    return dev.optimizer_and_scheduler(model, stage_args, updates)


def evaluate_cptac(
    model: TCGACPTACHybridProteinTranslator,
    arrays: dev.DevelopmentArrays,
    indices: np.ndarray,
    protein_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
    study_baseline: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eval_args = copy.copy(args)
    eval_args.batch_size = args.eval_batch_size
    prediction = dev.predict(model, arrays, indices, eval_args, device)
    if study_baseline is not None:
        baseline = np.asarray(study_baseline, dtype=np.float32)
        if baseline.shape != arrays.target.shape:
            raise ValueError("evaluation study baseline has an incompatible shape")
        prediction = prediction + baseline[indices]
    return dev.validation_tables(prediction, arrays, indices, protein_names, 8)


def high_coverage_median_spearman(
    per_protein: pd.DataFrame, high_coverage_indices: np.ndarray
) -> float:
    values = per_protein.iloc[np.asarray(high_coverage_indices, dtype=np.int64)][
        "spearman"
    ].to_numpy(dtype=float)
    finite = np.isfinite(values)
    return float(np.median(values[finite])) if finite.any() else float("nan")


@torch.no_grad()
def evaluate_tcpa(
    model: TCGACPTACHybridProteinTranslator,
    tcpa: dict[str, Any],
    cancer_index: np.ndarray,
    indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model.eval()
    dataset = joint.PlatformDataset(
        tcpa["rna"], tcpa["target"], tcpa["mask"], cancer_index, indices
    )
    loader = make_loader(dataset, args.eval_batch_size, args.num_workers, shuffle=False)
    selected = torch.as_tensor(tcpa["mapped_indices"], device=device)
    valid_rna = torch.as_tensor(tcpa["rna_valid_mask"], device=device).unsqueeze(0)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for batch in loader:
        rna = batch["rna"].to(device, non_blocking=True)
        cancer = batch["cancer_index"].to(device, non_blocking=True)
        with dev.autocast_context(args):
            output = model(
                rna,
                platform_index=torch.full_like(cancer, TCPA_PLATFORM),
                cancer_index=cancer,
                rna_valid_mask=valid_rna.expand(rna.shape[0], -1),
                protein_indices=selected,
            )["protein"]
        predictions.append(output.float().cpu().numpy())
        targets.append(batch["target"].numpy())
        masks.append(batch["mask"].numpy())
    prediction = np.vstack(predictions)
    target = np.vstack(targets)
    mask = np.vstack(masks).astype(bool)
    observed = np.where(mask, target, np.nan)
    table = per_protein_spearman(
        observed,
        prediction,
        tcpa["mapped_names"],
        mask=mask,
        min_samples=10,
    )
    observed_sd = np.nanstd(observed, axis=0, ddof=1)
    prediction_sd = np.nanstd(np.where(mask, prediction, np.nan), axis=0, ddof=1)
    ratio = np.divide(
        prediction_sd,
        observed_sd,
        out=np.full_like(prediction_sd, np.nan, dtype=np.float64),
        where=np.isfinite(observed_sd) & (observed_sd > 0),
    )
    table["observed_sd"] = observed_sd
    table["predicted_sd"] = prediction_sd
    table["predicted_to_observed_sd_ratio"] = ratio
    cosine = batchwise_flattened_cosine(observed, prediction, mask=mask, batch_size=8)
    summary = summarize_spearman(table)
    finite_ratio = ratio[np.isfinite(ratio)]
    summary.update(
        {
            "validation_mse": masked_mse(observed, prediction, mask=mask),
            "validation_mean_batch_cosine": float(cosine["cosine_similarity"].mean()),
            "median_predicted_to_observed_sd_ratio": float(np.median(finite_ratio)),
        }
    )
    return table, summary


def cpu_state(model: torch.nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main() -> int:
    args = parse_args()
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.pretrain_batch_size < args.pearson_min_samples:
        raise ValueError("pretrain batch is too small for Pearson loss")
    if args.finetune_batch_size < args.pearson_min_samples:
        raise ValueError("fine-tune batch is too small for Pearson loss")
    if args.within_study_pearson_weight < 0:
        raise ValueError("within-study Pearson weight must be non-negative")
    if args.within_study_pearson_weight > 0 and not args.cptac_study_baseline:
        raise ValueError("within-study Pearson loss requires CPTAC study baseline")
    if args.cptac_study_center_rna and not args.cptac_study_baseline:
        raise ValueError("study-centred CPTAC RNA requires CPTAC study baseline")
    if args.direct_local_linear_genes < 0:
        raise ValueError("direct local linear width must be non-negative")
    if args.direct_local_linear_genes > 0 and not (
        0.0 < args.initial_local_linear_mix < 1.0
    ):
        raise ValueError("initial local linear mix must lie in (0, 1)")
    if args.direct_local_ridge_alpha < 0:
        raise ValueError("direct local ridge alpha must be non-negative")
    if args.initialize_direct_local_ridge:
        if args.direct_local_linear_genes < 1:
            raise ValueError("ridge initialization requires direct local linear branch")
        if not args.cptac_study_baseline or not args.cptac_study_center_rna:
            raise ValueError(
                "ridge initialization requires paired CPTAC study centering"
            )
    if args.direct_local_supervised_selection:
        if args.direct_local_linear_genes < 1:
            raise ValueError("supervised local selection requires its linear branch")
        if not args.initialize_direct_local_ridge:
            raise ValueError("supervised local selection requires ridge initialization")
        if not args.cptac_study_baseline or not args.cptac_study_center_rna:
            raise ValueError(
                "supervised local selection requires paired CPTAC study centering"
            )
    if args.direct_global_components < 0:
        raise ValueError("direct global component count must be non-negative")
    if args.direct_global_components > 0 and not (
        0.0 < args.initial_global_linear_mix < 1.0
    ):
        raise ValueError("initial global linear mix must lie in (0, 1)")
    if args.direct_global_ridge_alpha < 0:
        raise ValueError("direct global ridge alpha must be non-negative")
    if args.initialize_direct_global_ridge:
        if args.direct_global_components < 1:
            raise ValueError("global ridge initialization requires its branch")
        if not args.cptac_study_baseline or not args.cptac_study_center_rna:
            raise ValueError(
                "global ridge initialization requires paired CPTAC study centering"
            )
    if args.direct_supervised_linear_genes < 0:
        raise ValueError("direct supervised linear width must be non-negative")
    if args.direct_supervised_linear_genes > 0 and not (
        0.0 < args.initial_supervised_linear_mix < 1.0
    ):
        raise ValueError("initial supervised linear mix must lie in (0, 1)")
    if (
        args.direct_supervised_linear_genes > 0
        and not args.initialize_direct_supervised_ridge
    ):
        raise ValueError(
            "direct supervised linear branch requires train-fitted ridge initialization"
        )
    if args.direct_supervised_ridge_alpha < 0:
        raise ValueError("direct supervised ridge alpha must be non-negative")
    if args.initialize_direct_supervised_ridge:
        if args.direct_supervised_linear_genes < 1:
            raise ValueError(
                "supervised ridge initialization requires its linear branch"
            )
        if not args.cptac_study_baseline or not args.cptac_study_center_rna:
            raise ValueError(
                "supervised ridge initialization requires paired CPTAC study centering"
            )
    if args.freeze_direct_linear_branches and not (
        args.initialize_direct_local_ridge
        or args.initialize_direct_global_ridge
        or args.initialize_direct_supervised_ridge
    ):
        raise ValueError(
            "freezing direct linear branches requires at least one fitted branch"
        )
    if args.freeze_direct_linear_values_only and args.freeze_direct_linear_branches:
        raise ValueError("direct branch freeze modes are mutually exclusive")
    if args.freeze_direct_linear_values_only and not (
        args.initialize_direct_local_ridge
        or args.initialize_direct_global_ridge
        or args.initialize_direct_supervised_ridge
    ):
        raise ValueError(
            "freezing direct linear values requires at least one fitted branch"
        )
    if args.joint_direct_value_fusion:
        if not 0.0 < args.initial_joint_neural_weight < 1.0:
            raise ValueError("initial joint neural weight must lie in (0, 1)")
        if not (
            args.direct_local_linear_genes > 0
            or args.direct_global_components > 0
            or args.direct_supervised_linear_genes > 0
        ):
            raise ValueError("joint value fusion requires a direct branch")
    elif args.initial_joint_neural_weight != 0.0:
        raise ValueError(
            "initial joint neural weight requires joint direct value fusion"
        )
    device = torch.device("cuda:0")
    dev.seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = args.output_dir / "tables"
    models = args.output_dir / "models"
    logs = args.output_dir / "logs"
    for folder in (tables, models, logs):
        folder.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_status.txt").write_text("running\n", encoding="utf-8")
    (args.output_dir / "config.json").write_text(
        json.dumps(base.jsonable_args(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    inputs = base.load_inputs(args)
    direct_initialization_checkpoint: dict[str, Any] | None = None
    direct_initialization_state: dict[str, Tensor] | None = None
    if args.direct_initialization_checkpoint is not None:
        if not args.direct_initialization_checkpoint.is_file():
            raise FileNotFoundError(args.direct_initialization_checkpoint)
        direct_initialization_checkpoint = torch.load(
            args.direct_initialization_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(direct_initialization_checkpoint, dict):
            raise ValueError("direct initialization checkpoint must be a mapping")
        direct_initialization_state = direct_initialization_checkpoint.get(
            "model_state"
        )
        if not isinstance(direct_initialization_state, dict):
            raise ValueError("direct initialization checkpoint has no model state")
        split_contract = direct_initialization_checkpoint.get("split_contract", {})
        if split_contract.get("cptac_train") != 916 or split_contract.get(
            "cptac_selection_validation"
        ) != 229 or split_contract.get("cptac_outer_test") != 286:
            raise ValueError("direct initialization checkpoint split differs")
        if split_contract.get("outer_test_evaluated") is not False:
            raise ValueError("direct initialization checkpoint used outer labels")
        if tuple(map(str, direct_initialization_checkpoint.get("gene_names", ()))) != tuple(
            map(str, inputs["gene_names"])
        ):
            raise ValueError("direct initialization checkpoint gene order differs")
        if tuple(
            map(str, direct_initialization_checkpoint.get("protein_names", ()))
        ) != tuple(map(str, inputs["protein_names"])):
            raise ValueError("direct initialization checkpoint protein order differs")
        if bool(
            direct_initialization_checkpoint.get(
                "cptac_study_baseline_enabled", False
            )
        ) != bool(args.cptac_study_baseline):
            raise ValueError("direct initialization checkpoint baseline mode differs")
        frozen_source = set(
            map(
                str,
                direct_initialization_checkpoint.get(
                    "frozen_direct_parameters", ()
                ),
            )
        )
        required_frozen = {
            name
            for name in (
                "direct_local_weight",
                "direct_local_bias",
                "direct_global_weight",
                "direct_global_bias",
                "direct_supervised_weight",
                "direct_supervised_bias",
            )
            if name in direct_initialization_state
        }
        if not required_frozen.issubset(frozen_source):
            raise ValueError(
                "checkpoint direct values were not frozen during source training"
            )
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

    cptac_raw = seal_outer_protein_labels(inputs["protein_raw"], partitions.outer_test)
    cptac_raw_mask = np.isfinite(cptac_raw)
    if args.cptac_study_center_rna:
        cptac_rna, cptac_rna_scaler_state = fit_transform_train_study_centered_rna(
            inputs["rna"],
            partitions.selection_train,
            manifest[args.study_column].astype(str).to_numpy(),
        )
    else:
        cptac_rna_scaler = base.FeatureStandardizer.fit(
            inputs["rna"], inputs["gene_names"], indices=partitions.selection_train
        )
        cptac_rna = cptac_rna_scaler.transform(
            inputs["rna"], inputs["gene_names"]
        ).astype(np.float32)
        cptac_rna_scaler_state = {
            "mode": "selection_train_global_gene_standardization",
            "mean": cptac_rna_scaler.mean,
            "scale": cptac_rna_scaler.scale,
        }
    cptac_target_scaler = TrainFittedProteinScale.fit(
        cptac_raw,
        partitions.selection_train,
        feature_names=inputs["protein_names"],
        mask=cptac_raw_mask,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
    )
    cptac_target = np.zeros_like(cptac_raw, dtype=np.float32)
    cptac_mask = np.zeros_like(cptac_raw_mask, dtype=bool)
    for indices, clip in (
        (partitions.selection_train, True),
        (partitions.selection_validation, False),
    ):
        changed = cptac_target_scaler.transform(
            cptac_raw[indices],
            mask=cptac_raw_mask[indices],
            feature_names=inputs["protein_names"],
            clip=clip,
        )
        cptac_target[indices] = np.nan_to_num(changed, nan=0.0)
        cptac_mask[indices] = np.isfinite(changed)

    tcpa = joint.load_tcpa_arrays(
        args.tcpa_prepared_dir, inputs["gene_names"], inputs["protein_names"], args
    )
    cptac_labels = manifest[args.strata_column].astype(str).to_numpy()
    cancers, cptac_cancer_index, tcpa_cancer_index = joint.combined_cancer_vocabulary(
        cptac_labels,
        partitions.selection_train,
        tcpa["cancer_labels"],
        tcpa["train"],
    )
    cptac_arrays = dev.DevelopmentArrays(
        cptac_rna, cptac_target, cptac_mask, cptac_cancer_index
    )
    cptac_training_coverage = cptac_mask[partitions.selection_train].sum(axis=0)
    cptac_high_coverage_1000 = np.argsort(
        -cptac_training_coverage, kind="stable"
    )[:1000]
    study_baseline_state: dict[str, Any] | None = None
    cptac_study_baseline = np.zeros_like(cptac_target, dtype=np.float32)
    if args.cptac_study_baseline:
        cptac_study_baseline, study_baseline_state = fit_train_study_protein_baseline(
            cptac_target,
            cptac_mask,
            partitions.selection_train,
            manifest[args.study_column].astype(str).to_numpy(),
        )

    graph = ProteinGraphArtifact.load(args.protein_graph_artifact)
    if graph.protein_names != tuple(map(str, inputs["protein_names"])):
        raise ValueError("protein graph order differs from output vocabulary")
    local_prior = cptac_local_prior(
        inputs["gene_names"],
        inputs["protein_names"],
        base.build_parent_gene_index(inputs),
        graph,
        args.max_local_genes,
    )
    direct_local_prior = None
    if args.direct_local_linear_genes > 0:
        if args.direct_local_supervised_selection:
            direct_local_prior = fit_supervised_local_gene_prior(
                cptac_rna,
                cptac_target,
                cptac_mask,
                cptac_study_baseline,
                partitions.selection_train,
                inputs["gene_names"],
                inputs["protein_names"],
                selected_genes=args.direct_local_linear_genes,
            )
        else:
            direct_local_prior = cptac_local_prior(
                inputs["gene_names"],
                inputs["protein_names"],
                base.build_parent_gene_index(inputs),
                graph,
                args.direct_local_linear_genes,
            )
    direct_supervised_prior = None
    if args.direct_supervised_linear_genes > 0:
        if direct_initialization_state is None:
            direct_supervised_prior = fit_supervised_local_gene_prior(
                cptac_rna,
                cptac_target,
                cptac_mask,
                cptac_study_baseline,
                partitions.selection_train,
                inputs["gene_names"],
                inputs["protein_names"],
                selected_genes=args.direct_supervised_linear_genes,
            )
        else:
            gene_index = np.asarray(
                direct_initialization_state["direct_supervised_gene_index"],
                dtype=np.int64,
            )
            gene_mask = np.asarray(
                direct_initialization_state["direct_supervised_gene_mask"],
                dtype=bool,
            )
            expected_shape = (
                len(inputs["protein_names"]),
                args.direct_supervised_linear_genes,
            )
            if gene_index.shape != expected_shape or gene_mask.shape != expected_shape:
                raise ValueError(
                    "checkpoint supervised direct prior dimensions differ"
                )
            direct_supervised_prior = LocalGenePrior(
                gene_index=gene_index,
                gene_mask=gene_mask,
                relation_index=np.where(gene_mask, 0, -1).astype(np.int64),
                edge_strength=gene_mask.astype(np.float32),
                relation_names=("checkpoint_train_supervised_rna",),
                metadata={
                    "construction": "compatible train-fitted checkpoint",
                    "source_checkpoint": str(args.direct_initialization_checkpoint),
                    "uses_validation_samples": False,
                    "uses_outer_test_samples": False,
                },
            )
    calibration_mask = np.zeros(len(inputs["protein_names"]), dtype=bool)
    calibration_mask[tcpa["mapped_indices"]] = True
    config = HybridProteinConfig(
        n_genes=len(inputs["gene_names"]),
        n_proteins=len(inputs["protein_names"]),
        n_cancers=len(cancers),
        d_model=args.d_model,
        n_heads=args.heads,
        dim_head=args.dim_head,
        ff_mult=args.ff_mult,
        map_hidden=args.map_hidden,
        map_states=args.map_states,
        n_global_states=args.global_states,
        n_module_states=args.module_states,
        module_depth=args.module_depth,
        protein_chunk=args.protein_chunk,
        max_local_genes=args.max_local_genes,
        branch_dropout=args.branch_dropout,
        initial_map_gate=0.30,
        initial_local_gate=0.25,
        initial_global_gate=0.20,
        use_direct_cognate=True,
        initial_cognate_gate=0.25,
        use_direct_local_linear=direct_local_prior is not None,
        max_direct_local_genes=max(1, args.direct_local_linear_genes),
        initial_local_linear_mix=(
            args.initial_local_linear_mix if direct_local_prior is not None else 0.0
        ),
        use_direct_global_linear=args.direct_global_components > 0,
        direct_global_components=max(1, args.direct_global_components),
        initial_global_linear_mix=(
            args.initial_global_linear_mix
            if args.direct_global_components > 0
            else 0.0
        ),
        use_direct_supervised_linear=direct_supervised_prior is not None,
        max_direct_supervised_genes=max(1, args.direct_supervised_linear_genes),
        initial_supervised_linear_mix=(
            args.initial_supervised_linear_mix
            if direct_supervised_prior is not None
            else 0.0
        ),
        joint_direct_value_fusion=args.joint_direct_value_fusion,
        initial_joint_neural_weight=args.initial_joint_neural_weight,
    )
    model = TCGACPTACHybridProteinTranslator(
        config,
        local_prior,
        direct_local_prior=direct_local_prior,
        direct_supervised_prior=direct_supervised_prior,
        tcpa_calibrated_protein_mask=calibration_mask,
    ).to(device)
    if args.cptac_study_baseline:
        with torch.no_grad():
            model.shared_output.weight.zero_()
            model.shared_output.bias.zero_()
    direct_local_ridge_state: dict[str, Any] | None = None
    if args.initialize_direct_local_ridge and direct_initialization_state is None:
        assert direct_local_prior is not None
        initial_weight, initial_bias, direct_local_ridge_state = (
            fit_direct_local_ridge_initialization(
                cptac_rna,
                cptac_target,
                cptac_mask,
                cptac_study_baseline,
                partitions.selection_train,
                direct_local_prior,
                alpha=args.direct_local_ridge_alpha,
            )
        )
        with torch.no_grad():
            assert model.direct_local_weight is not None
            assert model.direct_local_bias is not None
            model.direct_local_weight.copy_(
                torch.as_tensor(initial_weight, device=device)
            )
            model.direct_local_bias.copy_(torch.as_tensor(initial_bias, device=device))
    direct_global_ridge_state: dict[str, Any] | None = None
    if args.initialize_direct_global_ridge and direct_initialization_state is None:
        (
            initial_projection,
            initial_global_weight,
            initial_global_bias,
            direct_global_ridge_state,
        ) = fit_direct_global_pca_ridge_initialization(
            cptac_rna,
            cptac_target,
            cptac_mask,
            cptac_study_baseline,
            partitions.selection_train,
            components=args.direct_global_components,
            alpha=args.direct_global_ridge_alpha,
            seed=args.seed,
        )
        with torch.no_grad():
            assert model.direct_global_weight is not None
            assert model.direct_global_bias is not None
            model.direct_global_projection.copy_(
                torch.as_tensor(initial_projection, device=device)
            )
            model.direct_global_weight.copy_(
                torch.as_tensor(initial_global_weight, device=device)
            )
            model.direct_global_bias.copy_(
                torch.as_tensor(initial_global_bias, device=device)
            )
    direct_supervised_ridge_state: dict[str, Any] | None = None
    if args.initialize_direct_supervised_ridge and direct_initialization_state is None:
        assert direct_supervised_prior is not None
        (
            initial_supervised_weight,
            initial_supervised_bias,
            direct_supervised_ridge_state,
        ) = fit_direct_local_ridge_initialization(
            cptac_rna,
            cptac_target,
            cptac_mask,
            cptac_study_baseline,
            partitions.selection_train,
            direct_supervised_prior,
            alpha=args.direct_supervised_ridge_alpha,
        )
        with torch.no_grad():
            assert model.direct_supervised_weight is not None
            assert model.direct_supervised_bias is not None
            model.direct_supervised_weight.copy_(
                torch.as_tensor(initial_supervised_weight, device=device)
            )
            model.direct_supervised_bias.copy_(
                torch.as_tensor(initial_supervised_bias, device=device)
            )
    if direct_initialization_state is not None:
        copied_names: list[str] = []
        branch_names = (
            "direct_local_weight",
            "direct_local_bias",
            "direct_global_projection",
            "direct_global_weight",
            "direct_global_bias",
            "direct_supervised_weight",
            "direct_supervised_bias",
        )
        current_state = model.state_dict()
        with torch.no_grad():
            for name in branch_names:
                if name not in current_state:
                    continue
                if name not in direct_initialization_state:
                    raise ValueError(f"checkpoint lacks direct parameter {name}")
                source = direct_initialization_state[name]
                if tuple(source.shape) != tuple(current_state[name].shape):
                    raise ValueError(f"checkpoint direct parameter shape differs: {name}")
                current_state[name].copy_(source.to(current_state[name].device))
                copied_names.append(name)
        if direct_local_prior is not None:
            for name in ("direct_local_gene_index", "direct_local_gene_mask"):
                if name not in direct_initialization_state or not torch.equal(
                    current_state[name].cpu(), direct_initialization_state[name].cpu()
                ):
                    raise ValueError(f"checkpoint direct local prior differs: {name}")
        checkpoint_source = {
            "path": str(args.direct_initialization_checkpoint),
            "selected_epoch": direct_initialization_checkpoint.get("selected_epoch"),
            "copied_state_names": copied_names,
            "uses_validation_labels_for_values": False,
            "uses_outer_test_labels": False,
        }
        direct_local_ridge_state = checkpoint_source
        direct_global_ridge_state = checkpoint_source
        direct_supervised_ridge_state = checkpoint_source
    frozen_direct_parameters: tuple[str, ...] = ()
    if args.freeze_direct_linear_branches:
        frozen_direct_parameters = freeze_direct_linear_branches(model)
    elif args.freeze_direct_linear_values_only:
        frozen_direct_parameters = freeze_direct_linear_values(model)
    model.fix_projection_matrices_()

    cptac_train_dataset = StudyBaselineDevelopmentDataset(
        cptac_arrays, partitions.selection_train, cptac_study_baseline
    )
    cptac_loader = make_loader(
        cptac_train_dataset,
        args.pretrain_batch_size,
        args.num_workers,
        shuffle=True,
    )
    tcpa_train_dataset = joint.PlatformDataset(
        tcpa["rna"],
        tcpa["target"],
        tcpa["mask"],
        tcpa_cancer_index,
        tcpa["train"],
    )
    tcpa_loader = make_loader(
        tcpa_train_dataset,
        args.finetune_batch_size,
        args.num_workers,
        shuffle=True,
    )

    if args.smoke_only:
        torch.cuda.reset_peak_memory_stats(device)
        smoke_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        is_tcpa_smoke = args.smoke_stage == "tcpa"
        batch = next(iter(tcpa_loader if is_tcpa_smoke else cptac_loader))
        selected = (
            torch.as_tensor(tcpa["mapped_indices"], device=device)
            if is_tcpa_smoke
            else torch.arange(min(args.protein_subsample, config.n_proteins), device=device)
        )
        rna = batch["rna"].to(device)
        cancer = batch["cancer_index"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        if not is_tcpa_smoke:
            target = target.index_select(1, selected)
            mask = mask.index_select(1, selected)
        with dev.autocast_context(args):
            neural_prediction = model(
                rna,
                platform_index=torch.full_like(
                    cancer, TCPA_PLATFORM if is_tcpa_smoke else CPTAC_PLATFORM
                ),
                cancer_index=cancer,
                rna_valid_mask=(
                    torch.as_tensor(tcpa["rna_valid_mask"], device=device)
                    .unsqueeze(0)
                    .expand(rna.shape[0], -1)
                    if is_tcpa_smoke
                    else None
                ),
                protein_indices=selected,
            )["protein"]
            prediction = neural_prediction
            within_study_pearson = prediction.new_zeros(())
            if not is_tcpa_smoke:
                smoke_baseline = batch["study_baseline"].to(device).index_select(1, selected)
                loss, pieces, prediction = combined_cptac_loss(
                    neural_prediction,
                    smoke_baseline,
                    target,
                    mask,
                    args,
                )
                within_study_pearson = pieces["within_study_pearson"]
            else:
                loss, pieces = combined_loss(prediction, target, mask, args)
        loss.backward()
        smoke_optimizer.step()
        smoke = {
            "status": "passed",
            "smoke_stage": args.smoke_stage,
            "optimizer_step": True,
            "output_shape": list(prediction.shape),
            "loss": float(loss.detach().cpu()),
            "pearson_loss": float(pieces["pearson"].detach().cpu()),
            "variance_loss": float(pieces["variance"].detach().cpu()),
            "within_study_pearson_loss": float(
                within_study_pearson.detach().cpu()
            ),
            "parameter_count": model.parameter_count(),
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
            "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 2**20),
            "outer_test_evaluated": False,
        }
        encoded = json.dumps(smoke, ensure_ascii=False, indent=2)
        (args.output_dir / "smoke_summary.json").write_text(encoded, encoding="utf-8")
        print(encoded, flush=True)
        return 0

    pretrain_updates = len(cptac_loader) * args.pretrain_epochs
    optimizer, scheduler = optimizer_for_stage(
        model,
        args,
        pretrain_updates,
        encoder_lr=args.encoder_lr,
        translator_lr=args.translator_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    pretrain_history: list[dict[str, Any]] = []
    best_pretrain_state: dict[str, Tensor] | None = None
    best_pretrain_epoch = 0
    best_pretrain_spearman = float("-inf")
    stale = 0
    if (
        args.initialize_direct_local_ridge
        or args.initialize_direct_global_ridge
        or args.initialize_direct_supervised_ridge
    ):
        initialization_table, _, initialization_summary = evaluate_cptac(
            model,
            cptac_arrays,
            partitions.selection_validation,
            inputs["protein_names"],
            args,
            device,
            cptac_study_baseline,
        )
        initialization_record = {
            "stage": "cptac_pretrain_initialization",
            "epoch": 0,
            "train_loss": float("nan"),
            "train_pearson": float("nan"),
            "train_mse": float("nan"),
            "train_variance": float("nan"),
            "train_within_study_pearson": float("nan"),
            "cptac_validation_median_spearman": initialization_summary[
                "median_spearman"
            ],
            "cptac_validation_high_coverage_1000_median_spearman": (
                high_coverage_median_spearman(
                    initialization_table, cptac_high_coverage_1000
                )
            ),
            "cptac_validation_mse": initialization_summary["validation_mse"],
            "cptac_validation_batch_cosine": initialization_summary[
                "validation_mean_batch_cosine"
            ],
            "cptac_validation_sd_ratio": initialization_summary[
                "median_predicted_to_observed_sd_ratio"
            ],
        }
        pretrain_history.append(initialization_record)
        print(json.dumps(initialization_record), flush=True)
        best_pretrain_spearman = float(initialization_summary["median_spearman"])
        best_pretrain_state = cpu_state(model)
        dev.atomic_torch_save(
            {
                "schema_version": 1,
                "model_family": MODEL_FAMILY,
                "stage": "cptac_pretrain_initialization",
                "epoch": 0,
                "selection_validation_median_spearman": best_pretrain_spearman,
                "model_config": config.to_dict(),
                "model_state": best_pretrain_state,
            },
            models / "stage1_best_running.pt",
        )
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "pearson": 0.0,
            "mse": 0.0,
            "variance": 0.0,
            "within_study_pearson": 0.0,
        }
        schedule = epoch_protein_schedule(
            config.n_proteins,
            len(cptac_loader),
            args.protein_subsample,
            seed=args.seed + epoch * 1009,
        )
        for step, batch in enumerate(cptac_loader):
            selected = torch.as_tensor(schedule[step], device=device)
            rna = batch["rna"].to(device, non_blocking=True)
            cancer = batch["cancer_index"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True).index_select(1, selected)
            mask = batch["mask"].to(device, non_blocking=True).index_select(1, selected)
            baseline = batch["study_baseline"].to(
                device, non_blocking=True
            ).index_select(1, selected)
            optimizer.zero_grad(set_to_none=True)
            with dev.autocast_context(args):
                neural_prediction = model(
                    rna,
                    platform_index=torch.full_like(cancer, CPTAC_PLATFORM),
                    cancer_index=cancer,
                    protein_indices=selected,
                )["protein"]
                loss, pieces, prediction = combined_cptac_loss(
                    neural_prediction, baseline, target, mask, args
                )
                within_study_pearson = pieces["within_study_pearson"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            totals["loss"] += float(loss.detach().float().cpu())
            for name in ("pearson", "mse", "variance"):
                totals[name] += float(pieces[name].detach().float().cpu())
            totals["within_study_pearson"] += float(
                within_study_pearson.detach().float().cpu()
            )
        per_protein, cosine, summary = evaluate_cptac(
            model,
            cptac_arrays,
            partitions.selection_validation,
            inputs["protein_names"],
            args,
            device,
            cptac_study_baseline,
        )
        record = {
            "stage": "cptac_pretrain",
            "epoch": epoch,
            **{f"train_{key}": value / len(cptac_loader) for key, value in totals.items()},
            "cptac_validation_median_spearman": summary["median_spearman"],
            "cptac_validation_high_coverage_1000_median_spearman": (
                high_coverage_median_spearman(
                    per_protein, cptac_high_coverage_1000
                )
            ),
            "cptac_validation_mse": summary["validation_mse"],
            "cptac_validation_batch_cosine": summary["validation_mean_batch_cosine"],
            "cptac_validation_sd_ratio": summary["median_predicted_to_observed_sd_ratio"],
        }
        pretrain_history.append(record)
        print(json.dumps(record), flush=True)
        pd.DataFrame(pretrain_history).to_csv(
            tables / "stage1_pretrain_history_running.tsv", sep="\t", index=False
        )
        monitor = float(summary["median_spearman"])
        if monitor > best_pretrain_spearman + 1e-6:
            best_pretrain_spearman = monitor
            best_pretrain_epoch = epoch
            best_pretrain_state = cpu_state(model)
            stale = 0
            dev.atomic_torch_save(
                {
                    "schema_version": 1,
                    "model_family": MODEL_FAMILY,
                    "stage": "cptac_pretrain",
                    "epoch": epoch,
                    "selection_validation_median_spearman": monitor,
                    "model_config": config.to_dict(),
                    "model_state": best_pretrain_state,
                },
                models / "stage1_best_running.pt",
            )
        else:
            stale += 1
        if epoch >= args.pretrain_min_epochs and stale >= args.pretrain_patience:
            break

    if best_pretrain_state is None:
        raise RuntimeError("CPTAC pretraining produced no checkpoint")
    model.load_state_dict(best_pretrain_state)
    stage1_table, stage1_cosine, stage1_summary = evaluate_cptac(
        model,
        cptac_arrays,
        partitions.selection_validation,
        inputs["protein_names"],
        args,
        device,
        cptac_study_baseline,
    )
    stage1_summary["high_coverage_1000_median_spearman"] = (
        high_coverage_median_spearman(stage1_table, cptac_high_coverage_1000)
    )
    stage1_table.to_csv(tables / "stage1_cptac_per_protein.tsv", sep="\t", index=False)
    stage1_cosine.to_csv(tables / "stage1_cptac_batch_cosine.tsv", sep="\t", index=False)
    pd.DataFrame(pretrain_history).to_csv(
        tables / "stage1_pretrain_history.tsv", sep="\t", index=False
    )
    stage1_checkpoint = {
        "schema_version": 1,
        "model_family": MODEL_FAMILY,
        "stage": "cptac_pretrain",
        "model_config": config.to_dict(),
        "model_state": best_pretrain_state,
        "selected_epoch": best_pretrain_epoch,
        "gene_names": inputs["gene_names"],
        "protein_names": inputs["protein_names"],
        "cancer_vocabulary": cancers,
        "cptac_rna_standardizer": cptac_rna_scaler_state,
        "cptac_target_scale": cptac_target_scaler.state_dict(),
        "cptac_study_baseline_enabled": bool(args.cptac_study_baseline),
        "cptac_study_baseline": study_baseline_state,
        "direct_local_ridge_initialization": direct_local_ridge_state,
        "direct_global_ridge_initialization": direct_global_ridge_state,
        "direct_supervised_ridge_initialization": direct_supervised_ridge_state,
        "frozen_direct_parameters": frozen_direct_parameters,
        "split_contract": {
            "cptac_train": 916,
            "cptac_selection_validation": 229,
            "cptac_outer_test": 286,
            "outer_test_evaluated": False,
        },
    }
    torch.save(stage1_checkpoint, models / "stage1_cptac_pretrained.pt")

    if args.stop_after_pretrain:
        np.savez_compressed(
            tables / "split_indices.npz",
            cptac_train=partitions.selection_train,
            cptac_selection_validation=partitions.selection_validation,
            cptac_outer_test=partitions.outer_test,
            tcpa_train=tcpa["train"],
            tcpa_validation=tcpa["validation"],
        )
        np.savez_compressed(
            tables / "cognate_only_prior.npz",
            gene_index=local_prior.gene_index,
            gene_mask=local_prior.gene_mask,
            relation_index=local_prior.relation_index,
            edge_strength=local_prior.edge_strength,
        )
        summary = {
            "model_family": MODEL_FAMILY,
            "parameter_count": model.parameter_count(),
            "cptac_pretrain_selected_epoch": best_pretrain_epoch,
            "cptac_pretrain_validation": stage1_summary,
            "cptac_study_baseline_enabled": bool(args.cptac_study_baseline),
            "n_cptac_train": 916,
            "n_cptac_selection_validation": 229,
            "n_cptac_outer_test": 286,
            "cptac_outer_test_evaluated": False,
            "stopped_after_pretrain_by_configuration": True,
            "protein_graph_sha256": sha256_file(args.protein_graph_artifact),
            "runtime_seconds": time.time() - started,
        }
        encoded = json.dumps(summary, ensure_ascii=False, indent=2)
        (args.output_dir / "final_summary.json").write_text(
            encoded, encoding="utf-8"
        )
        (args.output_dir / "done.txt").write_text(encoded, encoding="utf-8")
        (args.output_dir / "run_status.txt").write_text(
            "completed_pretrain_only\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0

    finetune_updates = len(tcpa_loader) * args.finetune_epochs
    optimizer, scheduler = optimizer_for_stage(
        model,
        args,
        finetune_updates,
        encoder_lr=args.finetune_encoder_lr,
        translator_lr=args.finetune_translator_lr,
    )
    tcpa_indices = torch.as_tensor(tcpa["mapped_indices"], device=device)
    tcpa_rna_mask = torch.as_tensor(tcpa["rna_valid_mask"], device=device).unsqueeze(0)
    finetune_history: list[dict[str, Any]] = []
    best_finetune_state: dict[str, Tensor] | None = None
    best_finetune_epoch = 0
    best_tcpa_spearman = float("-inf")
    stale = 0
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        totals = {"loss": 0.0, "pearson": 0.0, "mse": 0.0, "variance": 0.0}
        used_batches = 0
        for batch in tcpa_loader:
            if batch["rna"].shape[0] < args.pearson_min_samples:
                continue
            rna = batch["rna"].to(device, non_blocking=True)
            cancer = batch["cancer_index"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with dev.autocast_context(args):
                prediction = model(
                    rna,
                    platform_index=torch.full_like(cancer, TCPA_PLATFORM),
                    cancer_index=cancer,
                    rna_valid_mask=tcpa_rna_mask.expand(rna.shape[0], -1),
                    protein_indices=tcpa_indices,
                )["protein"]
                loss, pieces = combined_loss(prediction, target, mask, args)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            used_batches += 1
            totals["loss"] += float(loss.detach().float().cpu())
            for name in ("pearson", "mse", "variance"):
                totals[name] += float(pieces[name].detach().float().cpu())
        tcpa_table, tcpa_summary = evaluate_tcpa(
            model, tcpa, tcpa_cancer_index, tcpa["validation"], args, device
        )
        cptac_table, _, cptac_summary = evaluate_cptac(
            model,
            cptac_arrays,
            partitions.selection_validation,
            inputs["protein_names"],
            args,
            device,
            cptac_study_baseline,
        )
        record = {
            "stage": "tcpa_finetune",
            "epoch": epoch,
            **{f"train_{key}": value / used_batches for key, value in totals.items()},
            "tcpa_validation_median_spearman": tcpa_summary["median_spearman"],
            "tcpa_validation_mse": tcpa_summary["validation_mse"],
            "tcpa_validation_sd_ratio": tcpa_summary["median_predicted_to_observed_sd_ratio"],
            "cptac_validation_median_spearman_audit": cptac_summary["median_spearman"],
            "cptac_validation_high_coverage_1000_median_spearman_audit": (
                high_coverage_median_spearman(
                    cptac_table, cptac_high_coverage_1000
                )
            ),
            "cptac_validation_mse_audit": cptac_summary["validation_mse"],
            "cptac_validation_sd_ratio_audit": cptac_summary[
                "median_predicted_to_observed_sd_ratio"
            ],
        }
        finetune_history.append(record)
        print(json.dumps(record), flush=True)
        pd.DataFrame(finetune_history).to_csv(
            tables / "stage2_finetune_history_running.tsv", sep="\t", index=False
        )
        monitor = float(tcpa_summary["median_spearman"])
        if monitor > best_tcpa_spearman + 1e-6:
            best_tcpa_spearman = monitor
            best_finetune_epoch = epoch
            best_finetune_state = cpu_state(model)
            stale = 0
            dev.atomic_torch_save(
                {
                    "schema_version": 1,
                    "model_family": MODEL_FAMILY,
                    "stage": "tcpa_finetune",
                    "epoch": epoch,
                    "tcpa_validation_median_spearman": monitor,
                    "model_config": config.to_dict(),
                    "model_state": best_finetune_state,
                },
                models / "stage2_best_running.pt",
            )
        else:
            stale += 1
        if epoch >= args.finetune_min_epochs and stale >= args.finetune_patience:
            break

    if best_finetune_state is None:
        raise RuntimeError("TCGA-TCPA fine-tuning produced no checkpoint")
    model.load_state_dict(best_finetune_state)
    stage2_tcpa_table, stage2_tcpa_summary = evaluate_tcpa(
        model, tcpa, tcpa_cancer_index, tcpa["validation"], args, device
    )
    stage2_cptac_table, stage2_cptac_cosine, stage2_cptac_summary = evaluate_cptac(
        model,
        cptac_arrays,
        partitions.selection_validation,
        inputs["protein_names"],
        args,
        device,
        cptac_study_baseline,
    )
    stage2_cptac_summary["high_coverage_1000_median_spearman"] = (
        high_coverage_median_spearman(
            stage2_cptac_table, cptac_high_coverage_1000
        )
    )
    stage2_tcpa_table.to_csv(tables / "stage2_tcpa_per_protein.tsv", sep="\t", index=False)
    stage2_cptac_table.to_csv(tables / "stage2_cptac_per_protein.tsv", sep="\t", index=False)
    stage2_cptac_cosine.to_csv(
        tables / "stage2_cptac_batch_cosine.tsv", sep="\t", index=False
    )
    pd.DataFrame(finetune_history).to_csv(
        tables / "stage2_finetune_history.tsv", sep="\t", index=False
    )
    np.savez_compressed(
        tables / "split_indices.npz",
        cptac_train=partitions.selection_train,
        cptac_selection_validation=partitions.selection_validation,
        cptac_outer_test=partitions.outer_test,
        tcpa_train=tcpa["train"],
        tcpa_validation=tcpa["validation"],
    )
    np.savez_compressed(
        tables / "cognate_only_prior.npz",
        gene_index=local_prior.gene_index,
        gene_mask=local_prior.gene_mask,
        relation_index=local_prior.relation_index,
        edge_strength=local_prior.edge_strength,
    )
    if direct_local_prior is not None:
        np.savez_compressed(
            tables / "direct_local_linear_prior.npz",
            gene_index=direct_local_prior.gene_index,
            gene_mask=direct_local_prior.gene_mask,
            relation_index=direct_local_prior.relation_index,
            edge_strength=direct_local_prior.edge_strength,
        )
    if direct_supervised_prior is not None:
        np.savez_compressed(
            tables / "direct_supervised_linear_prior.npz",
            gene_index=direct_supervised_prior.gene_index,
            gene_mask=direct_supervised_prior.gene_mask,
            relation_index=direct_supervised_prior.relation_index,
            edge_strength=direct_supervised_prior.edge_strength,
        )
    stage2_checkpoint = {
        **stage1_checkpoint,
        "stage": "tcpa_finetune",
        "model_state": best_finetune_state,
        "selected_epoch": best_finetune_epoch,
        "tcpa_model_proteins": tcpa["mapped_names"],
        "tcpa_model_protein_indices": tcpa["mapped_indices"],
        "tcpa_rna_standardizer": {
            "mean": tcpa["rna_scaler"].mean,
            "scale": tcpa["rna_scaler"].scale,
        },
        "tcpa_target_scale": tcpa["target_scaler"].state_dict(),
    }
    torch.save(stage2_checkpoint, models / "stage2_tcpa_finetuned.pt")
    summary = {
        "model_family": MODEL_FAMILY,
        "parameter_count": model.parameter_count(),
        "cptac_pretrain_selected_epoch": best_pretrain_epoch,
        "cptac_pretrain_validation": stage1_summary,
        "tcpa_finetune_selected_epoch": best_finetune_epoch,
        "tcpa_finetune_validation": stage2_tcpa_summary,
        "cptac_after_tcpa_finetune_audit": stage2_cptac_summary,
        "cptac_median_spearman_change_after_finetune": float(
            stage2_cptac_summary["median_spearman"] - stage1_summary["median_spearman"]
        ),
        "n_cptac_train": 916,
        "n_cptac_selection_validation": 229,
        "n_cptac_outer_test": 286,
        "n_tcpa_train": int(tcpa["train"].size),
        "n_tcpa_validation": int(tcpa["validation"].size),
        "n_tcpa_proteins": int(len(tcpa["mapped_names"])),
        "cptac_outer_test_evaluated": False,
        "protein_graph_sha256": sha256_file(args.protein_graph_artifact),
        "runtime_seconds": time.time() - started,
    }
    encoded = json.dumps(summary, ensure_ascii=False, indent=2)
    (args.output_dir / "final_summary.json").write_text(encoded, encoding="utf-8")
    (args.output_dir / "done.txt").write_text(encoded, encoding="utf-8")
    (args.output_dir / "run_status.txt").write_text("completed\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
