import importlib.util
import sys
from pathlib import Path

import torch
import pytest


pytest.importorskip("torch_geometric")


CODE = Path(__file__).parents[1] / "code"
sys.path.insert(0, str(CODE))
SPEC = importlib.util.spec_from_file_location(
    "scp682main1_train", CODE / "train_scp682main1_phosphosite_centered.py"
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_pathway_state_enters_full_residual_decoder():
    pathway_gene_index = torch.tensor([[0, 1], [2, 3]])
    pathway_gene_mask = torch.ones_like(pathway_gene_index, dtype=torch.bool)
    site_pathway_index = torch.tensor([[0], [0], [1], [1], [0], [1]])
    site_pathway_mask = torch.ones_like(site_pathway_index, dtype=torch.bool)
    model = MOD.SCP682GeneralGraphResidual(
        n_sites=6,
        n_samples=4,
        hidden=8,
        latent=4,
        inter_dim=8,
        embd_dim=4,
        num_layers=1,
        sample_context_dim=4,
        pathway_gene_index=pathway_gene_index,
        pathway_gene_mask=pathway_gene_mask,
        site_pathway_index=site_pathway_index,
        site_pathway_mask=site_pathway_mask,
        pathway_hidden=4,
        pathway_adapter_rank=2,
        site_pathway_chunk=3,
    )
    rna = torch.randn(4, 4, requires_grad=True)
    neighbour = torch.tensor([[1, 2], [0, 2], [0, 1], [1, 2]])
    similarity = torch.rand(4, 2)
    pathway_state, _ = model.encode_pathways(rna, rna, neighbour, similarity)
    row_embed = torch.randn(6, 4, requires_grad=True)
    col_embed = torch.randn(4, 4, requires_grad=True)
    baseline = torch.randn(4, 6)
    mask = torch.ones(4, 6, dtype=torch.bool)
    site_prior = torch.rand(6)
    pred, delta, *_ = model.decode(
        row_embed,
        col_embed,
        baseline,
        mask,
        site_prior,
        pathway_state,
        sample_context=rna,
    )
    assert pred.shape == delta.shape == baseline.shape
    assert torch.allclose(torch.quantile(delta, 0.5, dim=1), torch.zeros(4), atol=1e-6)
    pred.square().mean().backward()
    assert torch.isfinite(rna.grad).all()
    assert model.pathway_graph.pathway_query.grad is not None
