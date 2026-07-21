"""Strict reference-library inference for the axial phosphosite model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .axial_dynamic_data import (
        apply_feature_zscore,
        build_query_reference_knn,
        masked_row_median_center,
        parent_baseline,
        sample_rank_encode,
    )
    from .axial_dynamic_hypergraph import (
        AxialHypergraphConfig,
        ProteinAnchoredAxialDynamicHypergraph,
    )
except ImportError:
    from axial_dynamic_data import (
        apply_feature_zscore,
        build_query_reference_knn,
        masked_row_median_center,
        parent_baseline,
        sample_rank_encode,
    )
    from axial_dynamic_hypergraph import (
        AxialHypergraphConfig,
        ProteinAnchoredAxialDynamicHypergraph,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--protein-prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def _build_model(checkpoint: dict[str, object], device: torch.device):
    config = AxialHypergraphConfig(**checkpoint["model_metadata"]["config"])
    prior = checkpoint["prior_tensors"]
    model = ProteinAnchoredAxialDynamicHypergraph(
        config,
        rna_pathway_index=torch.as_tensor(prior["rna_pathway_index"]),
        rna_pathway_mask=torch.as_tensor(prior["rna_pathway_mask"]),
        protein_pathway_index=torch.as_tensor(prior["protein_pathway_index"]),
        protein_pathway_mask=torch.as_tensor(prior["protein_pathway_mask"]),
        pathway_relation_mask=torch.as_tensor(prior["pathway_relation_mask"]),
        site_pathway_index=torch.as_tensor(prior["site_pathway_index"]),
        site_pathway_mask=torch.as_tensor(prior["site_pathway_mask"]),
        site_pathway_weight=torch.as_tensor(prior["site_pathway_weight"]),
        parent_protein_index=torch.as_tensor(prior["parent_protein_index"]),
        parent_protein_mask=torch.as_tensor(prior["parent_protein_mask"]),
        site_kinase_index=torch.as_tensor(prior["site_kinase_index"]),
        site_kinase_mask=torch.as_tensor(prior["site_kinase_mask"]),
        site_coverage=torch.as_tensor(prior["site_coverage"]),
        site_anchor_quality=torch.as_tensor(checkpoint["site_anchor_quality"]),
        site_anchor_coverage=torch.as_tensor(checkpoint["site_anchor_coverage"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not checkpoint["model_metadata"].get("total_protein_is_frozen_input", False):
        raise ValueError("checkpoint does not declare a frozen total-protein input")
    model = _build_model(checkpoint, device)

    rna = pd.read_parquet(args.rna)
    protein = pd.read_parquet(args.protein_prediction)
    rna.index = rna.index.astype(str)
    protein.index = protein.index.astype(str)
    common = [sample for sample in rna.index if sample in protein.index]
    if set(rna.index) != set(protein.index) or not common:
        raise ValueError("RNA and total-protein prediction sample sets must be identical")
    rna = rna.reindex(common)
    protein = protein.reindex(common)
    study_contract = checkpoint.get("study_site_standardization")
    if not isinstance(study_contract, dict):
        raise ValueError("checkpoint lacks the required study-site standardization contract")
    required_rna = list(map(str, checkpoint["rna_genes"]))
    required_protein = list(map(str, checkpoint["protein_genes"]))
    missing_rna = [gene for gene in required_rna if gene not in rna.columns]
    missing_protein = [gene for gene in required_protein if gene not in protein.columns]
    if missing_rna or missing_protein:
        raise ValueError(
            f"external matrices lack checkpoint features; RNA={missing_rna[:5]}, protein={missing_protein[:5]}"
        )
    rna_rank = sample_rank_encode(rna[required_rna].to_numpy(np.float32))
    protein_raw = protein[required_protein].to_numpy(np.float32)
    if not np.isfinite(protein_raw).all():
        raise ValueError("external total-protein predictions contain missing values")
    protein_z = apply_feature_zscore(
        protein_raw,
        np.asarray(checkpoint["protein_mean"], dtype=np.float32),
        np.asarray(checkpoint["protein_scale"], dtype=np.float32),
    )
    prior = checkpoint["prior_tensors"]
    baseline_raw = parent_baseline(
        protein_raw,
        np.asarray(checkpoint["parent_intercept"], dtype=np.float32),
        np.asarray(checkpoint["parent_slope"], dtype=np.float32),
        np.asarray(prior["parent_protein_index"], dtype=np.int64),
        np.asarray(prior["parent_protein_mask"], dtype=bool),
    )
    baseline_mask = np.broadcast_to(
        np.asarray(prior["parent_protein_mask"], dtype=bool)[None, :], baseline_raw.shape
    )
    baseline_centered, baseline_offset = masked_row_median_center(baseline_raw, baseline_mask)
    candidate_columns = np.asarray(checkpoint["candidate_rna_columns"], dtype=np.int64)
    neighbour, similarity = build_query_reference_knn(
        rna_rank[:, candidate_columns],
        np.asarray(checkpoint["reference_candidate_features"], dtype=np.float32),
        int(checkpoint["sample_knn"]),
        query_ids=common,
        reference_ids=list(map(str, checkpoint["reference_sample_ids"])),
    )
    reference_cache = tuple(
        torch.as_tensor(value, device=device) for value in checkpoint["reference_cache"]
    )
    reference_residual = torch.as_tensor(
        checkpoint["reference_residual"], device=device, dtype=torch.float32
    ).detach()
    reference_residual_mask = torch.as_tensor(
        checkpoint["reference_residual_mask"], device=device, dtype=torch.bool
    ).detach()
    rna_tensor = torch.as_tensor(rna_rank, device=device)
    protein_tensor = torch.as_tensor(protein_z, device=device)
    baseline_tensor = torch.as_tensor(baseline_centered, device=device)
    neighbour_tensor = torch.as_tensor(neighbour, device=device)
    similarity_tensor = torch.as_tensor(similarity, device=device)
    predictions = []
    corrections = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(common), args.batch_size):
            end = min(len(common), start + args.batch_size)
            result = model(
                rna_tensor[start:end],
                protein_tensor[start:end],
                baseline_tensor[start:end],
                reference_cache,
                neighbour_tensor[start:end],
                similarity_tensor[start:end],
                reference_residual,
                reference_residual_mask,
            )
            predictions.append(result["prediction"].float().cpu().numpy())
            corrections.append(result["correction"].float().cpu().numpy())
    prediction = np.concatenate(predictions, axis=0)
    correction = np.concatenate(corrections, axis=0)
    targets = list(map(str, checkpoint["phosphosite_targets"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction, index=common, columns=targets).to_parquet(
        args.output_dir / "centered_phosphosite_prediction.parquet"
    )
    pd.DataFrame(baseline_centered, index=common, columns=targets).to_parquet(
        args.output_dir / "centered_parent_protein_baseline.parquet"
    )
    pd.DataFrame(correction, index=common, columns=targets).to_parquet(
        args.output_dir / "centered_pathway_residual.parquet"
    )
    pd.DataFrame(
        {"sample_id": common, "parent_baseline_raw_median": baseline_offset}
    ).to_csv(args.output_dir / "sample_offsets.tsv", sep="\t", index=False)
    manifest = {
        "architecture": checkpoint["model_metadata"]["architecture"],
        "n_samples": len(common),
        "n_sites": len(targets),
        "sample_graph_mode": "training_reference_only",
        "query_to_query_edges": False,
        "phosphosite_labels_used": False,
        "study_column": study_contract["study_column"],
        "study_site_standardization_fit_role": "selection_train_only",
        "reference_phosphosite_residual_source": "training_reference_checkpoint_only",
        "final_projection": "fixed-vocabulary sample-row zero median",
        "output_scale": "training-fold within-study site standardized, then sample-row median centered",
        "unknown_study_policy": "prediction requires no outcome-derived study scaling",
        "total_protein_source": str(args.protein_prediction),
    }
    (args.output_dir / "prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
