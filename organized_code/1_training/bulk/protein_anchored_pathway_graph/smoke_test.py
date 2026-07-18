from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import ModelConfig, ProteinAnchoredPathwayGraphModel  # noqa: E402


def main() -> int:
    torch.manual_seed(7)
    batch = 3
    source_samples = 6
    n_rna = 17
    n_proteins = 11
    n_sites = 23
    n_pathways = 5
    hidden = 12
    rna_pathway_index = torch.randint(0, n_rna, (n_pathways, 6))
    protein_pathway_index = torch.randint(0, n_proteins, (n_pathways, 4))
    site_pathway_index = torch.randint(0, n_pathways, (n_sites, 3))
    parent_index = torch.randint(0, n_proteins, (n_sites,))
    kinase_index = torch.randint(0, 4, (n_sites, 2))
    config = ModelConfig(
        n_rna=n_rna,
        n_proteins=n_proteins,
        n_sites=n_sites,
        n_pathways=n_pathways,
        n_kinases=3,
        protein_hidden=24,
        pathway_hidden=hidden,
        pathway_adapter_rank=3,
        site_chunk_size=7,
    )
    model = ProteinAnchoredPathwayGraphModel(
        config,
        rna_pathway_index,
        torch.ones_like(rna_pathway_index, dtype=torch.bool),
        protein_pathway_index,
        torch.ones_like(protein_pathway_index, dtype=torch.bool),
        site_pathway_index,
        torch.ones_like(site_pathway_index, dtype=torch.bool),
        torch.full_like(site_pathway_index, 1.0 / site_pathway_index.shape[1], dtype=torch.float32),
        parent_index,
        torch.ones(n_sites, dtype=torch.bool),
        kinase_index,
        kinase_index.ne(0),
    )
    model.set_parent_calibration(torch.randn(n_sites), torch.randn(n_sites))
    rna_z = torch.randn(batch, n_rna)
    rna_rank = torch.randn(batch, n_rna)
    protein_hat = model.predict_protein(rna_z)
    source_state = torch.randn(source_samples, n_pathways, hidden)
    neighbour_index = torch.randint(0, source_samples, (n_pathways, batch, 3))
    neighbour_similarity = torch.rand(n_pathways, batch, 3) * 2.0 - 1.0
    output = model.residual_forward(
        rna_rank,
        protein_hat,
        source_state,
        neighbour_index,
        neighbour_similarity,
        return_attention=True,
    )
    assert output["prediction"].shape == (batch, n_sites)
    assert output["sample_attention"].shape == (batch, n_pathways, 3)
    assert output["site_pathway_attention"].shape == (batch, n_sites, 3)
    centered = output["centered_residual"]
    assert torch.allclose(centered.median(dim=1).values, torch.zeros(batch), atol=1.0e-5)

    output["prediction"].sum().backward()
    protein_grads = [parameter.grad for parameter in model.protein_predictor.parameters()]
    assert all(gradient is None for gradient in protein_grads)
    print("smoke_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
