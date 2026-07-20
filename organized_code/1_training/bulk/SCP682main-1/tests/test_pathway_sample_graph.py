import importlib.util
from pathlib import Path

import numpy as np
import torch


MODULE = Path(__file__).parents[1] / "code" / "pathway_sample_graph.py"
SPEC = importlib.util.spec_from_file_location("pathway_sample_graph", MODULE)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_query_knn_uses_reference_indices_only():
    reference = np.eye(5, dtype=np.float32)
    query = np.asarray([[0.9, 0.1, 0, 0, 0], [0, 0, 0, 0.2, 0.8]], dtype=np.float32)
    index, similarity = MOD.build_query_reference_knn(query, reference, k=2)
    assert index.shape == similarity.shape == (2, 2)
    assert index.min() >= 0 and index.max() < reference.shape[0]
    assert index[0, 0] == 0
    assert index[1, 0] == 4


def test_pathway_weights_and_site_reader_are_differentiable():
    gene_index = torch.tensor([[0, 1], [2, 3]])
    gene_mask = torch.ones_like(gene_index, dtype=torch.bool)
    graph = MOD.PathwaySpecificSampleGraph(4, gene_index, gene_mask, hidden=8, adapter_rank=2, dropout=0.0)
    rna = torch.tensor([[2.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0]], requires_grad=True)
    neighbour = torch.tensor([[1, 2], [0, 2], [0, 1]])
    similarity = torch.tensor([[0.8, 0.3], [0.7, 0.2], [0.5, 0.4]])
    pathway_state, attention = graph(rna, rna, neighbour, similarity)
    reader = MOD.SparseSitePathwayReader(
        site_hidden=6,
        pathway_hidden=8,
        output_hidden=5,
        site_pathway_index=torch.tensor([[0, 0], [1, 0], [0, 1]]),
        site_pathway_mask=torch.tensor([[1, 0], [1, 0], [1, 1]], dtype=torch.bool),
    )
    site_state = torch.randn(3, 6, requires_grad=True)
    output = reader(pathway_state, site_state, chunk_size=2)
    assert pathway_state.shape == (3, 2, 8)
    assert attention.shape == (3, 2, 2)
    assert output.shape == (3, 3, 5)
    output.square().mean().backward()
    assert torch.isfinite(rna.grad).all()
    assert torch.isfinite(site_state.grad).all()


def test_self_excluded_candidate_graph_has_no_diagonal():
    x = np.eye(4, dtype=np.float32)
    index, _ = MOD.build_query_reference_knn(x, x, k=2, exclude_matching_rows=True)
    assert all(i not in index[i].tolist() for i in range(4))


def test_site_keeps_pathway_identity_when_parent_is_not_rna_encoder_gene(tmp_path):
    genes = [f"G{i}" for i in range(8)]
    pathway_genes = genes + ["PARENT"]
    hallmark = tmp_path / "hallmark.gmt"
    canonical = tmp_path / "canonical.gmt"
    hallmark.write_text("HALLMARK_SIGNAL\tna\t" + "\t".join(pathway_genes) + "\n", encoding="utf-8")
    canonical.write_text("CANONICAL_SIGNAL\tna\t" + "\t".join(pathway_genes) + "\n", encoding="utf-8")
    prior = MOD.build_pathway_prior(
        genes,
        ["PARENT|S10"],
        hallmark,
        canonical,
        max_pathways=3,
        min_genes=8,
    )
    assert prior["site_pathway_mask"][0, 0]
    assert prior["site_pathway_index"][0, 0] != 0
