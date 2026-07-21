"""Synthetic forward/backward and memory check for the axial model."""

from __future__ import annotations

import argparse
import json

import torch

from axial_dynamic_hypergraph import AxialHypergraphConfig, ProteinAnchoredAxialDynamicHypergraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--references", type=int, default=64)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--rna", type=int, default=256)
    parser.add_argument("--proteins", type=int, default=128)
    parser.add_argument("--sites", type=int, default=512)
    parser.add_argument("--pathways", type=int, default=16)
    parser.add_argument("--kinases", type=int, default=32)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    torch.manual_seed(20260720)
    hidden = 64
    members = min(16, args.rna, args.proteins)
    site_paths = 4
    site_kinases = 4
    config = AxialHypergraphConfig(
        n_rna=args.rna,
        n_proteins=args.proteins,
        n_sites=args.sites,
        n_pathways=args.pathways,
        n_kinases=args.kinases,
        hidden=hidden,
        heads=4,
        axial_layers=2,
        pathway_adapter_rank=8,
        dropout=0.0,
        site_chunk_size=256,
    )
    rna_index = torch.randint(0, args.rna, (args.pathways, members))
    protein_index = torch.randint(0, args.proteins, (args.pathways, members))
    model = ProteinAnchoredAxialDynamicHypergraph(
        config,
        rna_pathway_index=rna_index,
        rna_pathway_mask=torch.ones_like(rna_index, dtype=torch.bool),
        protein_pathway_index=protein_index,
        protein_pathway_mask=torch.ones_like(protein_index, dtype=torch.bool),
        pathway_relation_mask=torch.ones(args.pathways, args.pathways, dtype=torch.bool),
        site_pathway_index=torch.randint(0, args.pathways, (args.sites, site_paths)),
        site_pathway_mask=torch.ones(args.sites, site_paths, dtype=torch.bool),
        site_pathway_weight=torch.full((args.sites, site_paths), 1.0 / site_paths),
        parent_protein_index=torch.randint(0, args.proteins, (args.sites,)),
        parent_protein_mask=torch.ones(args.sites, dtype=torch.bool),
        site_kinase_index=torch.randint(0, args.kinases + 1, (args.sites, site_kinases)),
        site_kinase_mask=torch.ones(args.sites, site_kinases, dtype=torch.bool),
        site_coverage=torch.rand(args.sites),
        site_anchor_quality=torch.rand(args.sites),
        site_anchor_coverage=torch.rand(args.sites),
    ).to(device)
    reference_rna = torch.randn(args.references, args.rna, device=device)
    reference_protein = torch.randn(args.references, args.proteins, device=device)
    neighbour = torch.randint(0, args.references, (args.references, args.knn), device=device)
    similarity = torch.rand(args.references, args.knn, device=device)
    model.eval()
    cache = model.build_reference_cache(
        reference_rna, reference_protein, neighbour, similarity, chunk_size=args.queries
    )
    model.train()
    query_rna = torch.randn(args.queries, args.rna, device=device, requires_grad=True)
    query_protein = torch.randn(
        args.queries, args.proteins, device=device, requires_grad=True
    )
    baseline = torch.randn(args.queries, args.sites, device=device)
    query_neighbour = torch.randint(0, args.references, (args.queries, args.knn), device=device)
    query_similarity = torch.rand(args.queries, args.knn, device=device)
    reference_residual = torch.randn(args.references, args.sites, device=device)
    reference_residual_mask = torch.rand(args.references, args.sites, device=device) > 0.2
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=args.amp and device.type == "cuda",
    ):
        output = model(
            query_rna,
            query_protein,
            baseline,
            cache,
            query_neighbour,
            query_similarity,
            reference_residual,
            reference_residual_mask,
        )
    output["prediction"].square().mean().backward()
    if query_protein.grad is not None:
        raise RuntimeError("phosphosite loss propagated into total-protein predictions")
    result = {
        "device": str(device),
        "prediction_shape": list(output["prediction"].shape),
        "finite": bool(torch.isfinite(output["prediction"]).all()),
        "protein_input_gradient": False,
        "bfloat16_autocast": bool(args.amp and device.type == "cuda"),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
