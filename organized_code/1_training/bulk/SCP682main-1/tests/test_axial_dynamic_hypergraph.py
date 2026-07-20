from __future__ import annotations

import sys
from pathlib import Path

import torch


CODE = Path(__file__).parents[1] / "code"
sys.path.insert(0, str(CODE))

from axial_dynamic_hypergraph import (  # noqa: E402
    AxialHypergraphConfig,
    ProteinAnchoredAxialDynamicHypergraph,
    masked_site_equal_mse,
    masked_site_equal_pearson_loss,
)


def make_model(site_chunk_size: int = 3):
    config = AxialHypergraphConfig(
        n_rna=6,
        n_proteins=4,
        n_sites=7,
        n_pathways=3,
        n_kinases=2,
        hidden=8,
        heads=2,
        axial_layers=2,
        pathway_adapter_rank=2,
        dropout=0.0,
        site_chunk_size=site_chunk_size,
        initial_site_shrinkage=0.2,
    )
    return ProteinAnchoredAxialDynamicHypergraph(
        config,
        rna_pathway_index=torch.tensor([[0, 1], [2, 3], [4, 5]]),
        rna_pathway_mask=torch.ones(3, 2, dtype=torch.bool),
        protein_pathway_index=torch.tensor([[0, 1], [1, 2], [2, 3]]),
        protein_pathway_mask=torch.ones(3, 2, dtype=torch.bool),
        pathway_relation_mask=torch.ones(3, 3, dtype=torch.bool),
        site_pathway_index=torch.tensor(
            [[0, 0], [1, 0], [2, 0], [1, 2], [2, 1], [0, 0], [1, 0]]
        ),
        site_pathway_mask=torch.tensor(
            [[1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 0], [1, 0]],
            dtype=torch.bool,
        ),
        site_pathway_weight=torch.tensor(
            [[1, 0], [1, 0], [1, 0], [0.5, 0.5], [0.5, 0.5], [1, 0], [1, 0]],
            dtype=torch.float32,
        ),
        parent_protein_index=torch.tensor([0, 1, 2, 3, 0, 1, 2]),
        parent_protein_mask=torch.ones(7, dtype=torch.bool),
        site_kinase_index=torch.tensor(
            [[1, 0], [2, 0], [1, 2], [0, 0], [2, 0], [1, 0], [0, 0]]
        ),
        site_kinase_mask=torch.tensor(
            [[1, 0], [1, 0], [1, 1], [0, 0], [1, 0], [1, 0], [0, 0]],
            dtype=torch.bool,
        ),
        site_coverage=torch.linspace(0.2, 1.0, 7),
    )


def reference_inputs():
    torch.manual_seed(4)
    rna = torch.randn(5, 6)
    protein = torch.randn(5, 4)
    neighbour = torch.tensor(
        [[1, 2], [0, 2], [1, 3], [2, 4], [3, 2]], dtype=torch.long
    )
    similarity = torch.tensor(
        [[0.8, 0.4], [0.7, 0.2], [0.6, 0.3], [0.9, 0.1], [0.5, 0.4]]
    )
    return rna, protein, neighbour, similarity


def test_forward_backward_and_frozen_total_protein_input():
    model = make_model()
    reference_rna, reference_protein, reference_neighbour, reference_similarity = reference_inputs()
    model.eval()
    cache = model.build_reference_cache(
        reference_rna, reference_protein, reference_neighbour, reference_similarity, chunk_size=2
    )
    query_rna = torch.randn(2, 6, requires_grad=True)
    query_protein = torch.randn(2, 4, requires_grad=True)
    baseline = torch.randn(2, 7)
    output = model(
        query_rna,
        query_protein,
        baseline,
        cache,
        torch.tensor([[0, 1], [2, 3]]),
        torch.tensor([[0.8, 0.4], [0.7, 0.3]]),
        return_attention=True,
    )
    assert output["prediction"].shape == (2, 7)
    assert output["correction"].shape == (2, 7)
    assert len(output["sample_attention"]) == 2
    assert output["sample_attention"][0].shape == (2, 3, 2, 2)
    assert output["hyperedge_type_gate"].shape == (2, 7, 3)
    assert torch.allclose(
        output["prediction"], baseline + output["correction"], atol=1.0e-6
    )
    assert torch.allclose(
        torch.quantile(output["centered_residual"], 0.5, dim=1),
        torch.zeros(2),
        atol=1.0e-6,
    )
    output["prediction"].square().mean().backward()
    assert query_rna.grad is not None and torch.isfinite(query_rna.grad).all()
    assert query_protein.grad is None


def test_external_query_is_invariant_to_other_query_samples():
    model = make_model().eval()
    reference_rna, reference_protein, reference_neighbour, reference_similarity = reference_inputs()
    cache = model.build_reference_cache(
        reference_rna, reference_protein, reference_neighbour, reference_similarity, chunk_size=3
    )
    rna = torch.randn(1, 6)
    protein = torch.randn(1, 4)
    baseline = torch.randn(1, 7)
    neighbour = torch.tensor([[0, 2]])
    similarity = torch.tensor([[0.8, 0.4]])
    with torch.no_grad():
        single = model(rna, protein, baseline, cache, neighbour, similarity)["prediction"]
        paired = model(
            torch.cat([rna, torch.randn_like(rna)], dim=0),
            torch.cat([protein, torch.randn_like(protein)], dim=0),
            torch.cat([baseline, torch.randn_like(baseline)], dim=0),
            cache,
            torch.cat([neighbour, torch.tensor([[1, 3]])], dim=0),
            torch.cat([similarity, torch.tensor([[0.7, 0.2]])], dim=0),
        )["prediction"]
    assert torch.allclose(single[0], paired[0], atol=1.0e-6)


def test_site_chunking_does_not_change_definition():
    small = make_model(site_chunk_size=2).eval()
    full = make_model(site_chunk_size=7).eval()
    full.load_state_dict(small.state_dict())
    reference_rna, reference_protein, reference_neighbour, reference_similarity = reference_inputs()
    cache_small = small.build_reference_cache(
        reference_rna, reference_protein, reference_neighbour, reference_similarity
    )
    cache_full = full.build_reference_cache(
        reference_rna, reference_protein, reference_neighbour, reference_similarity
    )
    query_rna = torch.randn(2, 6)
    query_protein = torch.randn(2, 4)
    baseline = torch.randn(2, 7)
    neighbour = torch.tensor([[0, 1], [2, 3]])
    similarity = torch.tensor([[0.8, 0.4], [0.7, 0.3]])
    with torch.no_grad():
        left = small(query_rna, query_protein, baseline, cache_small, neighbour, similarity)
        right = full(query_rna, query_protein, baseline, cache_full, neighbour, similarity)
    assert torch.allclose(left["prediction"], right["prediction"], atol=1.0e-6)


def test_equal_site_losses_are_finite_and_differentiable():
    prediction = torch.randn(12, 5, requires_grad=True)
    target = torch.randn(12, 5)
    mask = torch.ones(12, 5, dtype=torch.bool)
    loss = masked_site_equal_mse(prediction, target, mask)
    loss = loss + masked_site_equal_pearson_loss(prediction, target, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
