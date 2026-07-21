from __future__ import annotations

import sys
from pathlib import Path

import torch


CODE = Path(__file__).parents[1] / "code"
sys.path.insert(0, str(CODE))

from axial_dynamic_hypergraph import (  # noqa: E402
    AxialHypergraphConfig,
    ProteinAnchoredAxialDynamicHypergraph,
    compose_training_objective,
    masked_sample_intercept_site_equal_mse,
    masked_site_equal_mse,
    masked_site_equal_pearson_loss,
    masked_site_equal_variance_loss,
)
from train_axial_dynamic_hypergraph import _slice_query_neighbours  # noqa: E402


def test_query_neighbour_slicing_uses_the_sample_axis():
    shared = torch.arange(10 * 4).reshape(10, 4)
    expanded = shared.unsqueeze(0).expand(3, -1, -1)
    assert torch.equal(_slice_query_neighbours(shared, 2, 6), shared[2:6])
    assert torch.equal(_slice_query_neighbours(expanded, 2, 6), expanded[:, 2:6])


def test_constant_site_pearson_loss_has_finite_gradients():
    prediction = torch.zeros(12, 4, requires_grad=True)
    target = torch.arange(48, dtype=torch.float32).reshape(12, 4)
    mask = torch.ones_like(target, dtype=torch.bool)
    loss = masked_site_equal_pearson_loss(prediction, target, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()


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
        site_anchor_quality=torch.linspace(0.1, 0.7, 7),
        site_anchor_coverage=torch.linspace(0.3, 0.9, 7),
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


def reference_label_memory():
    residual = torch.arange(35, dtype=torch.float32).reshape(5, 7) / 10.0
    mask = torch.ones_like(residual, dtype=torch.bool)
    mask[1, 2] = False
    residual[1, 2] = 9999.0
    return residual, mask


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
    reference_residual, reference_residual_mask = reference_label_memory()
    output = model(
        query_rna,
        query_protein,
        baseline,
        cache,
        torch.tensor([[0, 1], [2, 3]]),
        torch.tensor([[0.8, 0.4], [0.7, 0.3]]),
        reference_residual,
        reference_residual_mask,
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
    assert torch.allclose(
        torch.quantile(output["prediction"], 0.5, dim=1),
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
    reference_residual, reference_residual_mask = reference_label_memory()
    with torch.no_grad():
        single = model(
            rna, protein, baseline, cache, neighbour, similarity,
            reference_residual, reference_residual_mask,
        )["prediction"]
        paired = model(
            torch.cat([rna, torch.randn_like(rna)], dim=0),
            torch.cat([protein, torch.randn_like(protein)], dim=0),
            torch.cat([baseline, torch.randn_like(baseline)], dim=0),
            cache,
            torch.cat([neighbour, torch.tensor([[1, 3]])], dim=0),
            torch.cat([similarity, torch.tensor([[0.7, 0.2]])], dim=0),
            reference_residual,
            reference_residual_mask,
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
    reference_residual, reference_residual_mask = reference_label_memory()
    with torch.no_grad():
        left = small(
            query_rna, query_protein, baseline, cache_small, neighbour, similarity,
            reference_residual, reference_residual_mask,
        )
        right = full(
            query_rna, query_protein, baseline, cache_full, neighbour, similarity,
            reference_residual, reference_residual_mask,
        )
    assert torch.allclose(left["prediction"], right["prediction"], atol=1.0e-6)


def test_equal_site_losses_are_finite_and_differentiable():
    prediction = torch.randn(12, 5, requires_grad=True)
    target = torch.randn(12, 5)
    mask = torch.ones(12, 5, dtype=torch.bool)
    loss = masked_site_equal_mse(prediction, target, mask)
    loss = loss + masked_site_equal_pearson_loss(prediction, target, mask)
    loss = loss + masked_site_equal_variance_loss(prediction, target, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_value_loss_ignores_one_offset_per_sample():
    target = torch.randn(12, 5)
    prediction = target + torch.linspace(-3.0, 3.0, 12).unsqueeze(1)
    mask = torch.ones_like(target, dtype=torch.bool)
    loss = masked_sample_intercept_site_equal_mse(prediction, target, mask)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1.0e-6)


def test_reference_missing_mask_blocks_hidden_residual_value():
    model = make_model().eval()
    reference_rna, reference_protein, reference_neighbour, reference_similarity = reference_inputs()
    cache = model.build_reference_cache(
        reference_rna, reference_protein, reference_neighbour, reference_similarity
    )
    query_rna = torch.randn(1, 6)
    query_protein = torch.randn(1, 4)
    baseline = torch.randn(1, 7)
    neighbour = torch.tensor([[1, 0]])
    similarity = torch.tensor([[0.9, 0.1]])
    residual, residual_mask = reference_label_memory()
    changed = residual.clone()
    changed[1, 2] = -9999.0
    with torch.no_grad():
        left = model(
            query_rna, query_protein, baseline, cache, neighbour, similarity,
            residual, residual_mask,
        )["prediction"]
        right = model(
            query_rna, query_protein, baseline, cache, neighbour, similarity,
            changed, residual_mask,
        )["prediction"]
    assert torch.allclose(left, right, atol=1.0e-6)


def test_reference_labels_do_not_change_query_graph_attention():
    model = make_model().eval()
    reference_rna, reference_protein, reference_neighbour, reference_similarity = reference_inputs()
    cache = model.build_reference_cache(
        reference_rna, reference_protein, reference_neighbour, reference_similarity
    )
    query_rna = torch.randn(1, 6)
    query_protein = torch.randn(1, 4)
    baseline = torch.randn(1, 7)
    neighbour = torch.tensor([[0, 2]])
    similarity = torch.tensor([[0.8, 0.4]])
    residual, residual_mask = reference_label_memory()
    with torch.no_grad():
        left = model(
            query_rna, query_protein, baseline, cache, neighbour, similarity,
            residual, residual_mask, return_attention=True,
        )
        right = model(
            query_rna, query_protein, baseline, cache, neighbour, similarity,
            residual + 1000.0, residual_mask, return_attention=True,
        )
    for left_attention, right_attention in zip(
        left["sample_attention"], right["sample_attention"]
    ):
        assert torch.allclose(left_attention, right_attention, atol=1.0e-7)


def test_training_objective_has_no_duplicate_residual_mse():
    values = [torch.tensor(float(index), requires_grad=True) for index in range(1, 5)]
    total, components = compose_training_objective(
        *values,
        pearson_weight=0.2,
        variance_weight=0.1,
        shrinkage_weight=0.01,
    )
    assert set(components) == {
        "site_equal_mse", "site_equal_pearson", "site_equal_variance", "site_shrinkage"
    }
    assert "residual_mse" not in components
    total.backward()
