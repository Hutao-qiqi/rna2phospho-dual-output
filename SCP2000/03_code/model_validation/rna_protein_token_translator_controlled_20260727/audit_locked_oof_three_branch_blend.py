"""Audit a train-only OOF blend of STRING, global PCA and supervised RNA ridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_protein as base
import train_hybrid_protein_development_screen as dev
from audit_locked_oof_string_pca_blend import (
    SHRINKAGE_GRID,
    choose_global_weight,
    convert_scale,
    evaluate,
    fit_global_prediction,
    individual_weights,
    input_args,
    local_prediction,
    protein_spearman,
    transform_subset,
)
from candidate_groupwise_rna_normalization import (
    build_strict_inner_partitions,
    seal_outer_protein_labels,
)
from candidate_multiscale_protein_query import build_protein_local_gene_prior
from candidate_rank_gaussian_rna_normalization import TrainQuantileGaussianizer
from hybrid_protein_training_contract import TrainFittedProteinScale
from protein_graph_prior import ProteinGraphArtifact
from splits import make_strict_folds
from train_cptac_pretrain_tcpa_finetune import (
    fit_direct_local_ridge_initialization,
    fit_supervised_local_gene_prior,
    fit_train_study_protein_baseline,
    fit_transform_train_study_centered_rna,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--protein-graph-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--components", type=int, default=512)
    parser.add_argument("--global-ridge-alpha", type=float, default=100.0)
    parser.add_argument("--local-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--local-genes", type=int, default=64)
    parser.add_argument("--supervised-genes", type=int, default=128)
    parser.add_argument("--supervised-ridge-alpha", type=float, default=100.0)
    parser.add_argument("--protein-block-size", type=int, default=128)
    parser.add_argument("--meta-folds", type=int, default=5)
    parser.add_argument(
        "--supervised-rna-transform",
        choices=("study_centered", "study_centered_rank_gaussian"),
        default="study_centered",
    )
    return parser.parse_args()


def learn_shrunk_weights(
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, list[dict[str, float]], float, list[dict[str, float]], np.ndarray, np.ndarray]:
    global_weight, global_screen = choose_global_weight(left, right, target, mask)
    raw_weights, counts = individual_weights(
        left, right, target, mask, global_weight
    )
    shrink_screen: list[dict[str, float]] = []
    selected_shrinkage = float(SHRINKAGE_GRID[0])
    selected_median = float("-inf")
    for shrinkage in SHRINKAGE_GRID:
        reliability = counts / (counts + shrinkage)
        weights = global_weight + reliability * (raw_weights - global_weight)
        prediction = (1.0 - weights[None, :]) * left + weights[None, :] * right
        median = float(np.nanmedian(protein_spearman(prediction, target, mask)))
        shrink_screen.append(
            {"shrinkage": float(shrinkage), "median_oof_spearman": median}
        )
        if median > selected_median:
            selected_median = median
            selected_shrinkage = float(shrinkage)
    reliability = counts / (counts + selected_shrinkage)
    weights = global_weight + reliability * (raw_weights - global_weight)
    return (
        global_weight,
        global_screen,
        selected_shrinkage,
        shrink_screen,
        weights,
        counts,
    )


def main() -> int:
    args = parse_args()
    if min(
        args.components,
        args.local_genes,
        args.supervised_genes,
        args.protein_block_size,
    ) < 1 or args.meta_folds < 2:
        raise ValueError("branch dimensions and meta-fold count must be positive")
    if min(
        args.global_ridge_alpha,
        args.local_ridge_alpha,
        args.supervised_ridge_alpha,
    ) < 0:
        raise ValueError("ridge parameters must be non-negative")

    inputs = base.load_inputs(input_args(args.data_dir))
    protein_names = tuple(map(str, inputs["protein_names"]))
    gene_names = tuple(map(str, inputs["gene_names"]))
    manifest = inputs["manifest"]
    studies = manifest["pdc_study_id"].astype(str).to_numpy()
    cancer = manifest["cancer_label"].astype(str).to_numpy()
    cases = manifest["case_submitter_id"].astype(str).to_numpy()
    partitions = build_strict_inner_partitions(
        inputs["sample_ids"], cancer, cases, seed=args.seed,
        n_folds=5, inner_folds=5, fold_index=0,
    )
    train = partitions.selection_train
    validation = partitions.selection_validation
    sizes = (train.size, validation.size, partitions.outer_test.size)
    if sizes != (916, 229, 286):
        raise RuntimeError(f"locked split differs from 916/229/286: {sizes}")

    raw = seal_outer_protein_labels(inputs["protein_raw"], partitions.outer_test)
    raw_mask = np.isfinite(raw)
    full_scaler = TrainFittedProteinScale.fit(
        raw, train, feature_names=protein_names, mask=raw_mask,
        lower_quantile=0.01, upper_quantile=0.99,
    )
    target = np.zeros_like(raw, dtype=np.float32)
    mask = np.zeros_like(raw_mask, dtype=bool)
    for indices, clip in ((train, True), (validation, False)):
        target[indices], mask[indices] = transform_subset(
            full_scaler, raw, indices, protein_names, clip=clip
        )

    graph = ProteinGraphArtifact.load(args.protein_graph_artifact)
    parent = base.build_parent_gene_index(inputs)
    string_prior = build_protein_local_gene_prior(
        gene_names, protein_names, parent, graph,
        max_local_genes=args.local_genes,
    )

    shape = (train.size, len(protein_names))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_local = np.full(shape, np.nan, dtype=np.float32)
    oof_global = np.full(shape, np.nan, dtype=np.float32)
    oof_supervised = np.full(shape, np.nan, dtype=np.float32)
    train_position = {int(sample): position for position, sample in enumerate(train)}
    folds = make_strict_folds(
        np.asarray(inputs["sample_ids"])[train], cancer[train],
        n_splits=args.meta_folds, random_state=args.seed + 2017,
        blocking_groups=cases[train],
    )
    fold_records: list[dict[str, object]] = []
    for fold_index, fold in enumerate(folds):
        fit = train[fold.train_idx]
        holdout = train[fold.test_idx]
        fold_scaler = TrainFittedProteinScale.fit(
            raw, fit, feature_names=protein_names, mask=raw_mask,
            lower_quantile=0.01, upper_quantile=0.99,
        )
        fold_target = np.zeros_like(raw, dtype=np.float32)
        fold_mask = np.zeros_like(raw_mask, dtype=bool)
        for indices, clip in ((fit, True), (holdout, False)):
            fold_target[indices], fold_mask[indices] = transform_subset(
                fold_scaler, raw, indices, protein_names, clip=clip
            )
        fold_baseline, _ = fit_train_study_protein_baseline(
            fold_target, fold_mask, fit, studies
        )
        fold_rna, _ = fit_transform_train_study_centered_rna(
            inputs["rna"], fit, studies
        )
        fold_supervised_rna = fold_rna
        if args.supervised_rna_transform == "study_centered_rank_gaussian":
            fold_gaussianizer = TrainQuantileGaussianizer.fit(
                fold_rna,
                fit,
                feature_names=gene_names,
            )
            fold_supervised_rna = fold_gaussianizer.transform(
                fold_rna,
                feature_names=gene_names,
            )
            fold_gaussianizer.save(
                args.output_dir / "rank_gaussian_mappings" / f"fold_{fold_index}.npz",
                fit_indices=fit,
            )
        local_weights, local_biases, local_state = (
            fit_direct_local_ridge_initialization(
                fold_rna, fold_target, fold_mask, fold_baseline, fit,
                string_prior, alpha=args.local_ridge_alpha,
            )
        )
        local_fold = local_prediction(
            fold_rna, fold_baseline, local_weights, local_biases,
            string_prior, holdout,
        )
        global_fold, explained = fit_global_prediction(
            fold_rna, fold_target, fold_mask, fold_baseline, fit, holdout,
            components=args.components, alpha=args.global_ridge_alpha,
            seed=args.seed + fold_index,
        )
        supervised_prior = fit_supervised_local_gene_prior(
            fold_supervised_rna, fold_target, fold_mask, fold_baseline, fit,
            gene_names, protein_names, selected_genes=args.supervised_genes,
            protein_block_size=args.protein_block_size,
        )
        supervised_weights, supervised_biases, supervised_state = (
            fit_direct_local_ridge_initialization(
                fold_supervised_rna, fold_target, fold_mask, fold_baseline, fit,
                supervised_prior, alpha=args.supervised_ridge_alpha,
            )
        )
        supervised_fold = local_prediction(
            fold_supervised_rna, fold_baseline, supervised_weights, supervised_biases,
            supervised_prior, holdout,
        )
        positions = np.asarray([train_position[int(i)] for i in holdout])
        oof_local[positions] = convert_scale(local_fold, fold_scaler, full_scaler)
        oof_global[positions] = convert_scale(global_fold, fold_scaler, full_scaler)
        oof_supervised[positions] = convert_scale(
            supervised_fold, fold_scaler, full_scaler
        )
        fold_records.append(
            {
                "fold": fold_index,
                "fit_samples": int(fit.size),
                "holdout_samples": int(holdout.size),
                "local_fitted_proteins": int(local_state["fitted_proteins"]),
                "supervised_fitted_proteins": int(
                    supervised_state["fitted_proteins"]
                ),
                "explained_singular_value_fraction": explained,
            }
        )
        print(json.dumps(fold_records[-1]), flush=True)
    if not all(
        np.isfinite(value).all()
        for value in (oof_local, oof_global, oof_supervised)
    ):
        raise RuntimeError("out-of-fold prediction matrices are incomplete")

    (
        base_global_weight,
        base_global_screen,
        base_shrinkage,
        base_shrink_screen,
        base_protein_weights,
        observation_counts,
    ) = learn_shrunk_weights(
        oof_local, oof_global, target[train], mask[train]
    )
    oof_base = (
        (1.0 - base_protein_weights[None, :]) * oof_local
        + base_protein_weights[None, :] * oof_global
    ).astype(np.float32)
    (
        supervised_global_weight,
        supervised_global_screen,
        supervised_shrinkage,
        supervised_shrink_screen,
        supervised_protein_weights,
        _,
    ) = learn_shrunk_weights(
        oof_base, oof_supervised, target[train], mask[train]
    )

    full_baseline, _ = fit_train_study_protein_baseline(
        target, mask, train, studies
    )
    full_rna, _ = fit_transform_train_study_centered_rna(
        inputs["rna"], train, studies
    )
    full_supervised_rna = full_rna
    if args.supervised_rna_transform == "study_centered_rank_gaussian":
        full_gaussianizer = TrainQuantileGaussianizer.fit(
            full_rna,
            train,
            feature_names=gene_names,
        )
        full_supervised_rna = full_gaussianizer.transform(
            full_rna,
            feature_names=gene_names,
        )
        full_gaussianizer.save(
            args.output_dir / "rank_gaussian_mappings" / "full_train.npz",
            fit_indices=train,
        )
    local_weights, local_biases, full_local_state = (
        fit_direct_local_ridge_initialization(
            full_rna, target, mask, full_baseline, train, string_prior,
            alpha=args.local_ridge_alpha,
        )
    )
    validation_local = local_prediction(
        full_rna, full_baseline, local_weights, local_biases,
        string_prior, validation,
    )
    validation_global, full_explained = fit_global_prediction(
        full_rna, target, mask, full_baseline, train, validation,
        components=args.components, alpha=args.global_ridge_alpha,
        seed=args.seed,
    )
    full_supervised_prior = fit_supervised_local_gene_prior(
        full_supervised_rna, target, mask, full_baseline, train,
        gene_names, protein_names, selected_genes=args.supervised_genes,
        protein_block_size=args.protein_block_size,
    )
    supervised_weights, supervised_biases, full_supervised_state = (
        fit_direct_local_ridge_initialization(
            full_supervised_rna, target, mask, full_baseline, train,
            full_supervised_prior, alpha=args.supervised_ridge_alpha,
        )
    )
    validation_supervised = local_prediction(
        full_supervised_rna, full_baseline, supervised_weights, supervised_biases,
        full_supervised_prior, validation,
    )
    validation_base = (
        (1.0 - base_protein_weights[None, :]) * validation_local
        + base_protein_weights[None, :] * validation_global
    ).astype(np.float32)
    predictions = {
        "oof_two_branch_base": validation_base,
        "oof_three_branch_global_weight": (
            (1.0 - supervised_global_weight) * validation_base
            + supervised_global_weight * validation_supervised
        ).astype(np.float32),
        "oof_three_branch_protein_weight_shrunk": (
            (1.0 - supervised_protein_weights[None, :]) * validation_base
            + supervised_protein_weights[None, :] * validation_supervised
        ).astype(np.float32),
    }

    arrays = dev.DevelopmentArrays(
        rna=full_rna, target=target, mask=mask,
        cancer_index=np.zeros(target.shape[0], dtype=np.int64),
    )
    coverage = mask[train].sum(axis=0)
    high = np.argsort(-coverage, kind="stable")[:1000]
    validation_records: list[dict[str, object]] = []
    tables: dict[str, pd.DataFrame] = {}
    for name, prediction in predictions.items():
        record, table = evaluate(
            name, prediction, arrays, validation, protein_names, high
        )
        validation_records.append(record)
        tables[name] = table
    selected = max(validation_records, key=lambda row: row["median_spearman_all"])
    selected_name = str(selected["candidate"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(base_global_screen).to_csv(
        args.output_dir / "base_global_weight_screen.tsv", sep="\t", index=False
    )
    pd.DataFrame(base_shrink_screen).to_csv(
        args.output_dir / "base_shrinkage_screen.tsv", sep="\t", index=False
    )
    pd.DataFrame(supervised_global_screen).to_csv(
        args.output_dir / "supervised_global_weight_screen.tsv",
        sep="\t", index=False,
    )
    pd.DataFrame(supervised_shrink_screen).to_csv(
        args.output_dir / "supervised_shrinkage_screen.tsv",
        sep="\t", index=False,
    )
    pd.DataFrame(
        {
            "protein": protein_names,
            "train_observations": observation_counts,
            "base_global_weight": base_protein_weights,
            "supervised_weight": supervised_protein_weights,
        }
    ).to_csv(args.output_dir / "protein_blend_weights.tsv", sep="\t", index=False)
    for name, table in tables.items():
        table.to_csv(args.output_dir / f"{name}_per_protein.tsv", sep="\t", index=False)
    np.savez_compressed(
        args.output_dir / "oof_predictions.npz",
        local=oof_local,
        global_value=oof_global,
        supervised=oof_supervised,
        two_branch_base=oof_base,
        train_indices=train,
        protein_names=np.asarray(protein_names, dtype=str),
    )
    np.savez_compressed(
        args.output_dir / "selected_validation_predictions.npz",
        prediction=predictions[selected_name],
        validation_indices=validation,
        protein_names=np.asarray(protein_names, dtype=str),
    )
    np.savez_compressed(
        args.output_dir / "full_supervised_gene_prior.npz",
        gene_index=full_supervised_prior.gene_index,
        gene_mask=full_supervised_prior.gene_mask,
        edge_strength=full_supervised_prior.edge_strength,
    )
    result = {
        "seed": args.seed,
        "split_sizes": list(map(int, sizes)),
        "meta_folds": args.meta_folds,
        "components": args.components,
        "global_ridge_alpha": args.global_ridge_alpha,
        "local_ridge_alpha": args.local_ridge_alpha,
        "local_genes": args.local_genes,
        "supervised_genes": args.supervised_genes,
        "supervised_ridge_alpha": args.supervised_ridge_alpha,
        "supervised_rna_transform": args.supervised_rna_transform,
        "folds": fold_records,
        "full_explained_singular_value_fraction": full_explained,
        "full_local_fitted_proteins": int(full_local_state["fitted_proteins"]),
        "full_supervised_fitted_proteins": int(
            full_supervised_state["fitted_proteins"]
        ),
        "base_oof_global_weight": base_global_weight,
        "base_oof_selected_shrinkage": base_shrinkage,
        "supervised_oof_global_weight": supervised_global_weight,
        "supervised_oof_selected_shrinkage": supervised_shrinkage,
        "locked_229_validation": validation_records,
        "selected_by_locked_229_median_spearman": selected,
        "selection_train_only_branch_and_weight_fitting": True,
        "outer_test_evaluated": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
