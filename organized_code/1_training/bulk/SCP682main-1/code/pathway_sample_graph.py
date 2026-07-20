"""RNA-defined candidate graph with pathway-specific sample aggregation."""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


SIGNALLING_TERMS = (
    "AKT", "APOPTOSIS", "B_CELL", "CALCIUM", "CELL_CYCLE", "DNA_DAMAGE",
    "EGFR", "ERK", "FGFR", "HIPPO", "IGF", "INSULIN", "INTERFERON",
    "JAK", "MAPK", "MTOR", "NF_KAPPA", "NOTCH", "PI3K", "RAS",
    "RECEPTOR", "SIGNAL", "STAT", "STRESS", "T_CELL", "TGFB", "TNF",
    "VEGF", "WNT",
)


def read_gmt(path: str | Path) -> OrderedDict[str, list[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing pathway GMT: {path}")
    out: OrderedDict[str, list[str]] = OrderedDict()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            out[fields[0]] = list(dict.fromkeys(x.strip().upper() for x in fields[2:] if x.strip()))
    if not out:
        raise ValueError(f"No pathway gene sets found in {path}")
    return out


def _parent_gene(target: str) -> str:
    return str(target).split("|", 1)[0].split("_", 1)[0].upper()


def build_pathway_prior(
    rna_genes: Sequence[str],
    targets: Sequence[str],
    hallmark_gmt: str | Path,
    canonical_gmt: str | Path,
    max_pathways: int = 72,
    min_genes: int = 8,
    max_genes_per_pathway: int = 128,
    max_site_pathways: int = 8,
) -> dict[str, object]:
    """Create RNA membership and sparse site-to-pathway tensors.

    Pathways are selected using training RNA genes and phosphosite parent genes.
    GLOBAL_CONTEXT guarantees that every site has at least one valid route.
    """
    gene_to_index = {str(g).upper(): i for i, g in enumerate(rna_genes)}
    available = set(gene_to_index)
    parents = {_parent_gene(t) for t in targets}
    hallmark = read_gmt(hallmark_gmt)
    canonical = read_gmt(canonical_gmt)

    selected: OrderedDict[str, list[str]] = OrderedDict()
    selected_full: OrderedDict[str, list[str]] = OrderedDict()
    selected["GLOBAL_CONTEXT"] = [str(g).upper() for g in rna_genes[:max_genes_per_pathway]]
    selected_full["GLOBAL_CONTEXT"] = list(selected["GLOBAL_CONTEXT"])

    def overlap(genes: Sequence[str]) -> list[str]:
        return [g for g in genes if g in available][:max_genes_per_pathway]

    for name, genes in hallmark.items():
        kept = overlap(genes)
        if len(kept) >= min_genes:
            selected[name] = kept
            selected_full[name] = list(genes)

    candidates = []
    for name, genes in canonical.items():
        kept = overlap(genes)
        parent_overlap = len(parents.intersection(genes))
        if len(kept) >= min_genes and (parent_overlap > 0 or any(x in name.upper() for x in SIGNALLING_TERMS)):
            candidates.append((parent_overlap, len(kept), name, kept, list(genes)))
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
    for _, _, name, genes, full_genes in candidates:
        if len(selected) >= max_pathways:
            break
        selected.setdefault(name, genes)
        selected_full.setdefault(name, full_genes)

    if len(selected) < 2:
        raise ValueError("Pathway selection produced only GLOBAL_CONTEXT")

    names = list(selected)
    max_members = max(len(x) for x in selected.values())
    gene_index = np.zeros((len(names), max_members), dtype=np.int64)
    gene_mask = np.zeros((len(names), max_members), dtype=bool)
    gene_to_pathways: dict[str, list[int]] = {}
    for p, genes in enumerate(selected.values()):
        mapped = [gene_to_index[g] for g in genes]
        gene_index[p, :len(mapped)] = mapped
        gene_mask[p, :len(mapped)] = True
        for gene in selected_full[names[p]]:
            gene_to_pathways.setdefault(gene, []).append(p)

    site_pathway_index = np.zeros((len(targets), max_site_pathways), dtype=np.int64)
    site_pathway_mask = np.zeros((len(targets), max_site_pathways), dtype=bool)
    for s, target in enumerate(targets):
        related = gene_to_pathways.get(_parent_gene(target), [])
        related = [p for p in related if p != 0][:max_site_pathways]
        if not related:
            related = [0]
        site_pathway_index[s, :len(related)] = related
        site_pathway_mask[s, :len(related)] = True

    return {
        "pathway_names": names,
        "pathway_genes": list(selected.values()),
        "pathway_full_genes": list(selected_full.values()),
        "rna_gene_index": gene_index,
        "rna_gene_mask": gene_mask,
        "site_pathway_index": site_pathway_index,
        "site_pathway_mask": site_pathway_mask,
    }


def build_query_reference_knn(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
    exclude_matching_rows: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a reusable RNA-only candidate graph from query nodes to references."""
    q = np.nan_to_num(np.asarray(query, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    r = np.nan_to_num(np.asarray(reference, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-6)
    r = r / np.linalg.norm(r, axis=1, keepdims=True).clip(min=1e-6)
    similarity = q @ r.T
    if exclude_matching_rows:
        if q.shape[0] != r.shape[0]:
            raise ValueError("Matching-row exclusion requires equal query/reference lengths")
        np.fill_diagonal(similarity, -np.inf)
    kk = min(max(1, int(k)), r.shape[0] - int(exclude_matching_rows))
    if kk < 1:
        raise ValueError("Reference library must contain at least two samples for self-excluded kNN")
    index = np.argpartition(-similarity, kth=kk - 1, axis=1)[:, :kk]
    value = np.take_along_axis(similarity, index, axis=1)
    order = np.argsort(-value, axis=1)
    index = np.take_along_axis(index, order, axis=1)
    value = np.take_along_axis(value, order, axis=1)
    return index.astype(np.int64), value.astype(np.float32)


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = torch.softmax(logits.masked_fill(~mask, -1.0e4), dim=dim) * mask.to(logits.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1.0e-8)


class PathwaySpecificSampleGraph(nn.Module):
    """Pool RNA by pathway, then reweight a shared candidate neighbourhood."""

    def __init__(
        self,
        n_rna: int,
        rna_gene_index: torch.Tensor,
        rna_gene_mask: torch.Tensor,
        hidden: int = 32,
        adapter_rank: int = 4,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.n_pathways = int(rna_gene_index.shape[0])
        self.register_buffer("rna_gene_index", rna_gene_index.long())
        self.register_buffer("rna_gene_mask", rna_gene_mask.bool())
        self.gene_embedding = nn.Embedding(n_rna, hidden)
        self.pathway_query = nn.Parameter(torch.empty(self.n_pathways, hidden))
        self.value_direction = nn.Parameter(torch.empty(hidden))
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)
        self.adapter_v = nn.Parameter(torch.empty(self.n_pathways, adapter_rank, hidden))
        self.adapter_u = nn.Parameter(torch.zeros(self.n_pathways, hidden, adapter_rank))
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.gene_embedding.weight, std=0.02)
        nn.init.normal_(self.pathway_query, std=0.02)
        nn.init.normal_(self.value_direction, std=0.02)
        nn.init.normal_(self.adapter_v, std=0.01)

    def encode_rna(self, rna: torch.Tensor) -> torch.Tensor:
        selected = rna[:, self.rna_gene_index]
        token = self.gene_embedding(self.rna_gene_index).unsqueeze(0)
        token = token + selected.unsqueeze(-1) * self.value_direction.view(1, 1, 1, -1)
        logits = (token * self.pathway_query.view(1, self.n_pathways, 1, self.hidden)).sum(-1)
        logits = logits / math.sqrt(self.hidden)
        mask = self.rna_gene_mask.view(1, *self.rna_gene_mask.shape)
        weight = _masked_softmax(logits, mask, dim=2)
        return (weight.unsqueeze(-1) * token).sum(dim=2)

    def forward(
        self,
        query_rna: torch.Tensor,
        reference_rna: torch.Tensor,
        neighbour_index: torch.Tensor,
        neighbour_similarity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_state = self.encode_rna(query_rna)
        reference_state = self.encode_rna(reference_rna)
        neighbours = reference_state.index_select(0, neighbour_index.reshape(-1))
        neighbours = neighbours.reshape(neighbour_index.shape[0], neighbour_index.shape[1], self.n_pathways, self.hidden)
        q = self.query(query_state).unsqueeze(1)
        key = self.key(neighbours)
        logits = (q * key).sum(-1) / math.sqrt(self.hidden)
        prior = ((neighbour_similarity + 1.0) * 0.5).clamp_min(1.0e-4).log().unsqueeze(-1)
        attention = torch.softmax(logits + prior, dim=1)
        message = (attention.unsqueeze(-1) * self.value(neighbours)).sum(dim=1)
        low_rank = torch.einsum("bph,prh->bpr", message, self.adapter_v)
        adapted = torch.einsum("bpr,phr->bph", low_rank, self.adapter_u)
        state = self.norm(query_state + self.dropout(self.output(message) + adapted))
        return state, attention.permute(0, 2, 1).contiguous()


class SparseSitePathwayReader(nn.Module):
    """Let each phosphosite read only pathways linked to its parent gene."""

    def __init__(
        self,
        site_hidden: int,
        pathway_hidden: int,
        output_hidden: int,
        site_pathway_index: torch.Tensor,
        site_pathway_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("site_pathway_index", site_pathway_index.long())
        self.register_buffer("site_pathway_mask", site_pathway_mask.bool())
        self.site_query = nn.Linear(site_hidden, pathway_hidden, bias=False)
        self.pathway_key = nn.Linear(pathway_hidden, pathway_hidden, bias=False)
        self.output = nn.Sequential(nn.Linear(pathway_hidden, output_hidden), nn.GELU(), nn.LayerNorm(output_hidden))

    def forward(self, pathway_state: torch.Tensor, site_state: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
        parts = []
        for start in range(0, site_state.shape[0], chunk_size):
            end = min(site_state.shape[0], start + chunk_size)
            index = self.site_pathway_index[start:end]
            selected = pathway_state[:, index]
            query = self.site_query(site_state[start:end]).view(1, end - start, 1, -1)
            logits = (query * self.pathway_key(selected)).sum(-1) / math.sqrt(selected.shape[-1])
            mask = self.site_pathway_mask[start:end].view(1, end - start, -1)
            attention = _masked_softmax(logits, mask, dim=2)
            context = (attention.unsqueeze(-1) * selected).sum(dim=2)
            parts.append(self.output(context))
        return torch.cat(parts, dim=1)
