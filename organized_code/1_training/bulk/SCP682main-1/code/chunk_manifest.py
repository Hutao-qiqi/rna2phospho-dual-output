"""Deterministic target chunking and sparse kinase-edge selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SiteChunk:
    chunk_id: int
    site_index: np.ndarray
    kinase_site_edge_index: np.ndarray
    kinase_edge_count_before_cap: int


def parent_grouped_site_chunks(
    parent_index: np.ndarray,
    parent_mask: np.ndarray,
    *,
    chunk_size: int = 1000,
) -> list[np.ndarray]:
    """Pack complete parent groups without splitting a parent across chunks."""
    parent_index = np.asarray(parent_index, dtype=np.int64)
    parent_mask = np.asarray(parent_mask, dtype=bool)
    if parent_index.ndim != 1 or parent_mask.shape != parent_index.shape:
        raise ValueError("parent mappings must be one-dimensional and aligned")
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")

    groups: dict[tuple[str, int], list[int]] = {}
    for site, (parent, mapped) in enumerate(zip(parent_index, parent_mask)):
        key = ("parent", int(parent)) if mapped else ("unmapped", int(site))
        groups.setdefault(key, []).append(site)
    ordered_groups = sorted(groups.values(), key=lambda values: (values[0], len(values)))
    if any(len(values) > chunk_size for values in ordered_groups):
        raise ValueError("a parent group is larger than the configured site chunk")

    chunks: list[np.ndarray] = []
    current: list[int] = []
    for group in ordered_groups:
        if current and len(current) + len(group) > chunk_size:
            chunks.append(np.asarray(current, dtype=np.int64))
            current = []
        current.extend(group)
    if current:
        chunks.append(np.asarray(current, dtype=np.int64))

    flattened = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
    expected = np.arange(len(parent_index), dtype=np.int64)
    if not np.array_equal(np.sort(flattened), expected):
        raise RuntimeError("site chunks do not form an exact target partition")
    return chunks


def local_kinase_edges(
    global_edge_index: np.ndarray,
    site_index: np.ndarray,
    *,
    maximum_kinases_per_site: int = 128,
    edge_priority: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Select a bounded kinase candidate set and remap destinations locally.

    Learned HGT attention supplies the edge weights.  When an audited priority
    is unavailable, selection is deterministic by kinase index and therefore
    does not use phosphosite labels.
    """
    edge = np.asarray(global_edge_index, dtype=np.int64)
    sites = np.asarray(site_index, dtype=np.int64)
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError("kinase-site edge index must have shape [2, edge]")
    if maximum_kinases_per_site < 1:
        raise ValueError("maximum kinases per site must be positive")
    if edge_priority is not None:
        priority = np.asarray(edge_priority, dtype=np.float64).reshape(-1)
        if priority.shape[0] != edge.shape[1]:
            raise ValueError("edge priority length differs from edge count")
    else:
        priority = np.zeros(edge.shape[1], dtype=np.float64)

    destination_lookup = {int(site): local for local, site in enumerate(sites)}
    candidates: dict[int, list[int]] = {local: [] for local in range(len(sites))}
    for edge_id, destination in enumerate(edge[1]):
        local = destination_lookup.get(int(destination))
        if local is not None:
            candidates[local].append(edge_id)
    before = sum(len(values) for values in candidates.values())
    selected: list[int] = []
    for values in candidates.values():
        if len(values) > maximum_kinases_per_site:
            raise ValueError(
                "kinase capacity would truncate prior edges; increase maximum_kinases_per_site"
            )
        values.sort(key=lambda idx: (-priority[idx], int(edge[0, idx]), idx))
        selected.extend(values)
    selected.sort(key=lambda idx: (destination_lookup[int(edge[1, idx])], int(edge[0, idx])))
    if not selected:
        return np.empty((2, 0), dtype=np.int64), before
    chosen = edge[:, np.asarray(selected, dtype=np.int64)].copy()
    chosen[1] = np.asarray([destination_lookup[int(site)] for site in chosen[1]], dtype=np.int64)
    return np.ascontiguousarray(chosen), before


def build_site_chunks(
    parent_index: np.ndarray,
    parent_mask: np.ndarray,
    kinase_site_edge_index: np.ndarray,
    *,
    chunk_size: int = 1000,
    maximum_kinases_per_site: int = 128,
    edge_priority: np.ndarray | None = None,
) -> list[SiteChunk]:
    chunks = parent_grouped_site_chunks(
        parent_index, parent_mask, chunk_size=chunk_size
    )
    output: list[SiteChunk] = []
    for chunk_id, sites in enumerate(chunks):
        local_edge, before = local_kinase_edges(
            kinase_site_edge_index,
            sites,
            maximum_kinases_per_site=maximum_kinases_per_site,
            edge_priority=edge_priority,
        )
        output.append(SiteChunk(chunk_id, sites, local_edge, before))
    return output
