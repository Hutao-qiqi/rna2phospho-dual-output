from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__:
    from .data import PathwayTensors, build_pathway_knn, pathway_summary_features, sample_rank_encode
    from .model import ModelConfig, ProteinAnchoredPathwayGraphModel
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import PathwayTensors, build_pathway_knn, pathway_summary_features, sample_rank_encode  # type: ignore
    from model import ModelConfig, ProteinAnchoredPathwayGraphModel  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict phosphosite abundance with a trained protein-anchored pathway-graph model."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-graph-mode", choices=["cohort", "reference"], default="cohort")
    parser.add_argument("--sample-knn", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint is not None:
        return [args.checkpoint]
    paths = sorted(args.checkpoint_dir.glob("fold_*/models/protein_anchored_pathway_graph.pt"))
    if not paths:
        paths = sorted(args.checkpoint_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found under {args.checkpoint_dir}")
    return paths


def reconstruct_model(payload: dict[str, object], device: torch.device) -> tuple[ProteinAnchoredPathwayGraphModel, PathwayTensors]:
    state: dict[str, torch.Tensor] = payload["model_state_dict"]  # type: ignore[assignment]
    metadata: dict[str, object] = payload["model_metadata"]  # type: ignore[assignment]
    config = ModelConfig(**metadata["config"])  # type: ignore[arg-type]
    tensors = PathwayTensors(
        names=list(payload["pathway_names"]),
        genes=[list(values) for values in payload["pathway_genes"]],
        rna_index=state["pathway_encoder.rna_pathway_index"].cpu().numpy(),
        rna_mask=state["pathway_encoder.rna_pathway_mask"].cpu().numpy(),
        protein_index=state["pathway_encoder.protein_pathway_index"].cpu().numpy(),
        protein_mask=state["pathway_encoder.protein_pathway_mask"].cpu().numpy(),
        site_pathway_index=state["site_decoder.site_pathway_index"].cpu().numpy(),
        site_pathway_mask=state["site_decoder.site_pathway_mask"].cpu().numpy(),
        site_pathway_weight=state["site_decoder.site_pathway_weight"].cpu().numpy(),
        parent_protein_index=state["parent_protein_index"].cpu().numpy(),
        parent_protein_mask=state["parent_protein_mask"].cpu().numpy(),
        site_kinase_index=state["site_decoder.site_kinase_index"].cpu().numpy(),
        site_kinase_mask=state["site_decoder.site_kinase_mask"].cpu().numpy(),
        kinase_names=list(payload["kinase_names"]),
    )
    model = ProteinAnchoredPathwayGraphModel(
        config,
        torch.as_tensor(tensors.rna_index),
        torch.as_tensor(tensors.rna_mask),
        torch.as_tensor(tensors.protein_index),
        torch.as_tensor(tensors.protein_mask),
        torch.as_tensor(tensors.site_pathway_index),
        torch.as_tensor(tensors.site_pathway_mask),
        torch.as_tensor(tensors.site_pathway_weight),
        torch.as_tensor(tensors.parent_protein_index),
        torch.as_tensor(tensors.parent_protein_mask),
        torch.as_tensor(tensors.site_kinase_index),
        torch.as_tensor(tensors.site_kinase_mask),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, tensors


def align_rna(frame: pd.DataFrame, payload: dict[str, object]) -> tuple[np.ndarray, np.ndarray, int]:
    genes = list(payload["rna_genes"])
    mean = np.asarray(payload["rna_mean"], dtype=np.float32)
    std = np.asarray(payload["rna_std"], dtype=np.float32)
    missing = [gene for gene in genes if gene not in frame.columns]
    aligned = frame.reindex(columns=genes).apply(pd.to_numeric, errors="coerce")
    aligned = aligned.fillna(pd.Series(mean, index=genes))
    raw = aligned.to_numpy(dtype=np.float32)
    rna_z = np.nan_to_num((raw - mean[None, :]) / std[None, :], nan=0.0).clip(-6.0, 6.0).astype(np.float32)
    return rna_z, sample_rank_encode(raw), len(missing)


def encode_all(
    model: ProteinAnchoredPathwayGraphModel,
    rna_rank: np.ndarray,
    protein_hat: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    with torch.no_grad():
        for start in range(0, len(rna_rank), batch_size):
            end = min(start + batch_size, len(rna_rank))
            chunks.append(
                model.encode_pathways(
                    torch.as_tensor(rna_rank[start:end], device=device),
                    torch.as_tensor(protein_hat[start:end], device=device),
                )
            )
    return torch.cat(chunks, dim=0)


def predict_one(
    checkpoint: Path,
    rna: pd.DataFrame,
    mode: str,
    sample_knn: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, pathway_tensors = reconstruct_model(payload, device)
    rna_z, rna_rank, missing_genes = align_rna(rna, payload)
    with torch.no_grad():
        protein_hat = []
        for start in range(0, len(rna), batch_size):
            end = min(start + batch_size, len(rna))
            protein_hat.append(
                model.predict_protein(torch.as_tensor(rna_z[start:end], device=device)).float().cpu().numpy()
            )
    protein_hat_np = np.concatenate(protein_hat, axis=0).astype(np.float32)
    query_features = pathway_summary_features(rna_rank, protein_hat_np, pathway_tensors)
    query_state = encode_all(model, rna_rank, protein_hat_np, device, batch_size)

    effective_mode = mode
    if mode == "cohort" and len(rna) > 1:
        source_state = query_state
        source_features = query_features
        graph_index, graph_similarity = build_pathway_knn(
            source_features, query_features, sample_knn, exclude_identity=True
        )
    else:
        effective_mode = "reference"
        source_state = payload["reference_pathway_state"].to(device)
        source_features = np.asarray(payload["reference_graph_features"], dtype=np.float32)
        graph_index, graph_similarity = build_pathway_knn(
            source_features, query_features, sample_knn, exclude_identity=False
        )

    predictions = []
    baselines = []
    corrections = []
    with torch.no_grad():
        for start in range(0, len(rna), batch_size):
            end = min(start + batch_size, len(rna))
            output = model.residual_forward(
                torch.as_tensor(rna_rank[start:end], device=device),
                torch.as_tensor(protein_hat_np[start:end], device=device),
                source_state,
                torch.as_tensor(graph_index[:, start:end], device=device),
                torch.as_tensor(graph_similarity[:, start:end], device=device),
            )
            predictions.append(output["prediction"].float().cpu().numpy())
            baselines.append(output["baseline"].float().cpu().numpy())
            corrections.append(output["correction"].float().cpu().numpy())
    return {
        "prediction": np.concatenate(predictions, axis=0),
        "baseline": np.concatenate(baselines, axis=0),
        "correction": np.concatenate(corrections, axis=0),
        "targets": list(payload["phosphosite_targets"]),
        "missing_rna_genes": missing_genes,
        "sample_graph_mode": effective_mode,
        "checkpoint": str(checkpoint),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    rna = pd.read_parquet(args.rna)
    if rna.index.has_duplicates:
        raise ValueError("RNA sample identifiers must be unique")
    paths = checkpoint_paths(args)
    outputs = [
        predict_one(path, rna, args.sample_graph_mode, args.sample_knn, args.batch_size, device)
        for path in paths
    ]
    targets = outputs[0]["targets"]
    if any(output["targets"] != targets for output in outputs[1:]):
        raise ValueError("Checkpoint phosphosite orders differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction = np.mean([output["prediction"] for output in outputs], axis=0)
    baseline = np.mean([output["baseline"] for output in outputs], axis=0)
    correction = np.mean([output["correction"] for output in outputs], axis=0)
    pd.DataFrame(prediction, index=rna.index, columns=targets).to_parquet(
        args.output_dir / "phosphosite_prediction.parquet"
    )
    pd.DataFrame(baseline, index=rna.index, columns=targets).to_parquet(
        args.output_dir / "parent_protein_baseline.parquet"
    )
    pd.DataFrame(correction, index=rna.index, columns=targets).to_parquet(
        args.output_dir / "pathway_graph_correction.parquet"
    )
    summary = {
        "n_samples": len(rna),
        "n_phosphosites": len(targets),
        "n_checkpoints": len(paths),
        "requested_sample_graph_mode": args.sample_graph_mode,
        "effective_sample_graph_modes": [output["sample_graph_mode"] for output in outputs],
        "missing_rna_genes_per_checkpoint": [output["missing_rna_genes"] for output in outputs],
        "observed_phosphosite_used": False,
        "checkpoints": [str(path) for path in paths],
    }
    (args.output_dir / "prediction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
