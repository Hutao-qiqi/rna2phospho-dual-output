"""Fixed STRING protein graphs for RNA-to-protein candidates."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


GRAPH_SCHEMA_VERSION = 1


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_undirected_edges(edge_index: np.ndarray) -> np.ndarray:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, n_edges)")
    if edges.shape[1] == 0:
        return np.empty((2, 0), dtype=np.int64)
    low = np.minimum(edges[0], edges[1])
    high = np.maximum(edges[0], edges[1])
    if np.any(low == high):
        raise ValueError("self-loops are not allowed in the canonical graph")
    packed = np.stack((low, high), axis=1)
    packed = np.unique(packed, axis=0)
    order = np.lexsort((packed[:, 1], packed[:, 0]))
    return packed[order].T.astype(np.int64, copy=False)


def undirected_degree(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    edges = canonicalize_undirected_edges(edge_index)
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive")
    if edges.size and (edges.min() < 0 or edges.max() >= n_nodes):
        raise IndexError("edge_index contains a node outside n_nodes")
    degree = np.zeros(n_nodes, dtype=np.int64)
    if edges.shape[1]:
        np.add.at(degree, edges[0], 1)
        np.add.at(degree, edges[1], 1)
    return degree


def degree_preserving_rewire(
    edge_index: np.ndarray,
    *,
    n_nodes: int,
    seed: int,
    swaps_per_edge: float = 2.0,
    max_attempts_per_swap: int = 30,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Randomize a simple undirected graph with degree-preserving edge swaps."""

    if swaps_per_edge < 0 or max_attempts_per_swap < 1:
        raise ValueError("rewiring controls are outside their valid range")
    canonical = canonicalize_undirected_edges(edge_index)
    n_edges = canonical.shape[1]
    if n_edges < 2 or swaps_per_edge == 0:
        return canonical.copy(), {
            "requested_swaps": 0,
            "successful_swaps": 0,
            "attempts": 0,
            "edge_retention_fraction": 1.0,
        }

    original_degree = undirected_degree(canonical, n_nodes)
    edge_list = [tuple(map(int, pair)) for pair in canonical.T]
    edge_set = set(edge_list)
    rng = np.random.default_rng(seed)
    requested = int(np.ceil(swaps_per_edge * n_edges))
    max_attempts = max(requested * max_attempts_per_swap, 1000)
    successful = 0
    attempts = 0

    while successful < requested and attempts < max_attempts:
        attempts += 1
        first, second = rng.integers(0, n_edges, size=2)
        if first == second:
            continue
        a, b = edge_list[int(first)]
        c, d = edge_list[int(second)]
        if len({a, b, c, d}) != 4:
            continue
        if rng.random() < 0.5:
            proposed = ((a, c), (b, d))
        else:
            proposed = ((a, d), (b, c))
        proposed = tuple((min(x, y), max(x, y)) for x, y in proposed)
        if proposed[0] == proposed[1]:
            continue

        old_first = edge_list[int(first)]
        old_second = edge_list[int(second)]
        edge_set.remove(old_first)
        edge_set.remove(old_second)
        valid = proposed[0] not in edge_set and proposed[1] not in edge_set
        if not valid:
            edge_set.add(old_first)
            edge_set.add(old_second)
            continue
        edge_list[int(first)] = proposed[0]
        edge_list[int(second)] = proposed[1]
        edge_set.add(proposed[0])
        edge_set.add(proposed[1])
        successful += 1

    rewired = canonicalize_undirected_edges(np.asarray(edge_list, dtype=np.int64).T)
    if rewired.shape[1] != n_edges:
        raise RuntimeError("rewiring changed the number of edges")
    if not np.array_equal(undirected_degree(rewired, n_nodes), original_degree):
        raise RuntimeError("rewiring changed the node degree sequence")
    retained = len(set(map(tuple, canonical.T)) & set(map(tuple, rewired.T)))
    return rewired, {
        "requested_swaps": requested,
        "successful_swaps": successful,
        "attempts": attempts,
        "edge_retention_fraction": float(retained / n_edges),
    }


@dataclass(frozen=True)
class ProteinGraphArtifact:
    protein_names: tuple[str, ...]
    edge_index: np.ndarray
    edge_score: np.ndarray
    graph_mode: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.protein_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("protein_names must be non-empty and unique")
        edges = canonicalize_undirected_edges(self.edge_index)
        scores = np.asarray(self.edge_score, dtype=np.float32)
        if scores.shape != (edges.shape[1],):
            raise ValueError("edge_score must contain one value per edge")
        if scores.size and not np.isfinite(scores).all():
            raise ValueError("edge_score contains non-finite values")
        if edges.size and (edges.min() < 0 or edges.max() >= len(names)):
            raise IndexError("edge_index is outside the protein axis")
        object.__setattr__(self, "protein_names", names)
        object.__setattr__(self, "edge_index", edges)
        object.__setattr__(self, "edge_score", scores)

    @property
    def n_nodes(self) -> int:
        return len(self.protein_names)

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def degree(self) -> np.ndarray:
        return undirected_degree(self.edge_index, self.n_nodes)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            schema_version=np.asarray([GRAPH_SCHEMA_VERSION], dtype=np.int64),
            protein_names=np.asarray(self.protein_names, dtype=str),
            edge_index=self.edge_index.astype(np.int64),
            edge_score=self.edge_score.astype(np.float32),
            graph_mode=np.asarray([self.graph_mode], dtype=str),
            metadata_json=np.asarray(
                [json.dumps(self.metadata, ensure_ascii=True, sort_keys=True)],
                dtype=str,
            ),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ProteinGraphArtifact":
        with np.load(Path(path), allow_pickle=False) as payload:
            version = int(np.asarray(payload["schema_version"]).reshape(-1)[0])
            if version != GRAPH_SCHEMA_VERSION:
                raise ValueError(f"unsupported graph schema version: {version}")
            return cls(
                protein_names=tuple(np.asarray(payload["protein_names"]).astype(str)),
                edge_index=np.asarray(payload["edge_index"], dtype=np.int64),
                edge_score=np.asarray(payload["edge_score"], dtype=np.float32),
                graph_mode=str(np.asarray(payload["graph_mode"]).reshape(-1)[0]),
                metadata=json.loads(
                    str(np.asarray(payload["metadata_json"]).reshape(-1)[0])
                ),
            )


def load_protein_names_from_matrix(path: str | Path) -> tuple[str, ...]:
    matrix_path = Path(path)
    suffix = matrix_path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(matrix_path)
        physical_names = list(parquet.schema_arrow.names)
        pandas_metadata = parquet.schema_arrow.metadata or {}
        index_names: set[str] = set()
        if b"pandas" in pandas_metadata:
            metadata = json.loads(pandas_metadata[b"pandas"].decode("utf-8"))
            index_names = {
                value
                for value in metadata.get("index_columns", [])
                if isinstance(value, str)
            }
        names = tuple(name for name in physical_names if name not in index_names)
    elif suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(matrix_path, sep="\t", nrows=1)
        names = tuple(str(value) for value in frame.columns)
    elif suffix == ".csv":
        frame = pd.read_csv(matrix_path, nrows=1)
        names = tuple(str(value) for value in frame.columns)
    else:
        raise ValueError(f"unsupported protein matrix format: {matrix_path.suffix}")
    if not names or len(set(names)) != len(names):
        raise ValueError("protein matrix columns must be non-empty and unique")
    return names


def read_string_mapping(
    info_path: str | Path,
    protein_names: Sequence[str],
) -> tuple[dict[str, int], pd.DataFrame, dict[str, int]]:
    output_names = tuple(str(value) for value in protein_names)
    output_index = {name: index for index, name in enumerate(output_names)}
    string_to_output: dict[str, int] = {}
    string_ids_by_output: list[list[str]] = [[] for _ in output_names]
    preferred_name_counts: dict[str, int] = {}
    total_rows = 0

    with gzip.open(Path(info_path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"#string_protein_id", "preferred_name"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("STRING info file lacks required columns")
        for row in reader:
            total_rows += 1
            string_id = row["#string_protein_id"].strip()
            preferred = row["preferred_name"].strip()
            preferred_name_counts[preferred] = preferred_name_counts.get(preferred, 0) + 1
            index = output_index.get(preferred)
            if index is None:
                continue
            string_to_output[string_id] = index
            string_ids_by_output[index].append(string_id)

    rows = []
    for index, name in enumerate(output_names):
        ids = sorted(string_ids_by_output[index])
        rows.append(
            {
                "protein_index": index,
                "protein_symbol": name,
                "mapping_status": "exact_preferred_name" if ids else "unmapped",
                "string_protein_id_count": len(ids),
                "string_protein_ids": ";".join(ids),
                "preferred_name_multiplicity_in_string": preferred_name_counts.get(name, 0),
            }
        )
    audit = {
        "string_info_rows": total_rows,
        "output_proteins": len(output_names),
        "mapped_output_proteins": sum(bool(ids) for ids in string_ids_by_output),
        "mapped_string_protein_ids": len(string_to_output),
    }
    return string_to_output, pd.DataFrame(rows), audit


def build_string_graph(
    links_path: str | Path,
    string_to_output: dict[str, int],
    *,
    min_combined_score: int = 700,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if min_combined_score < 0 or min_combined_score > 1000:
        raise ValueError("min_combined_score must lie in [0, 1000]")
    edge_scores: dict[tuple[int, int], int] = {}
    rows_scanned = 0
    rows_above_threshold = 0
    rows_with_both_nodes_mapped = 0

    with gzip.open(Path(links_path), "rt", encoding="utf-8") as handle:
        header = handle.readline().strip().split()
        if header != ["protein1", "protein2", "combined_score"]:
            raise ValueError("unexpected STRING links header")
        for line in handle:
            rows_scanned += 1
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"malformed STRING links row at line {rows_scanned + 1}")
            score = int(fields[2])
            if score < min_combined_score:
                continue
            rows_above_threshold += 1
            first = string_to_output.get(fields[0])
            second = string_to_output.get(fields[1])
            if first is None or second is None:
                continue
            rows_with_both_nodes_mapped += 1
            if first == second:
                continue
            edge = (min(first, second), max(first, second))
            previous = edge_scores.get(edge)
            if previous is None or score > previous:
                edge_scores[edge] = score

    ordered = sorted(edge_scores)
    if ordered:
        edge_index = np.asarray(ordered, dtype=np.int64).T
        scores = np.asarray([edge_scores[edge] for edge in ordered], dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        scores = np.empty(0, dtype=np.float32)
    audit = {
        "string_link_rows_scanned": rows_scanned,
        "rows_above_score_threshold": rows_above_threshold,
        "rows_with_both_nodes_mapped": rows_with_both_nodes_mapped,
        "collapsed_gene_level_edges": len(ordered),
    }
    return edge_index, scores, audit


def connected_components(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    parent = np.arange(n_nodes, dtype=np.int64)
    size = np.ones(n_nodes, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for first, second in canonicalize_undirected_edges(edge_index).T:
        root_first = find(int(first))
        root_second = find(int(second))
        if root_first == root_second:
            continue
        if size[root_first] < size[root_second]:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        size[root_first] += size[root_second]
    roots = np.asarray([find(index) for index in range(n_nodes)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def summarize_graph(artifact: ProteinGraphArtifact) -> dict[str, Any]:
    degree = artifact.degree
    components = connected_components(artifact.edge_index, artifact.n_nodes)
    component_sizes = np.bincount(components, minlength=int(components.max()) + 1)
    quantiles = np.quantile(degree, [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    return {
        "graph_mode": artifact.graph_mode,
        "n_nodes": artifact.n_nodes,
        "n_edges": artifact.n_edges,
        "n_nonisolated_nodes": int(np.sum(degree > 0)),
        "nonisolated_fraction": float(np.mean(degree > 0)),
        "n_components_including_isolates": int(component_sizes.size),
        "largest_component_nodes": int(component_sizes.max()),
        "degree_min": int(quantiles[0]),
        "degree_q25": float(quantiles[1]),
        "degree_median": float(quantiles[2]),
        "degree_q75": float(quantiles[3]),
        "degree_q90": float(quantiles[4]),
        "degree_q99": float(quantiles[5]),
        "degree_max": int(quantiles[6]),
        "combined_score_min": (
            float(artifact.edge_score.min()) if artifact.n_edges else None
        ),
        "combined_score_median": (
            float(np.median(artifact.edge_score)) if artifact.n_edges else None
        ),
        "combined_score_max": (
            float(artifact.edge_score.max()) if artifact.n_edges else None
        ),
    }


def write_edge_table(
    path: str | Path,
    artifact: ProteinGraphArtifact,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if destination.suffix == ".gz" else open
    with opener(destination, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["protein_index_1", "protein_symbol_1", "protein_index_2", "protein_symbol_2", "combined_score"]
        )
        for position, (first, second) in enumerate(artifact.edge_index.T):
            writer.writerow(
                [
                    int(first),
                    artifact.protein_names[int(first)],
                    int(second),
                    artifact.protein_names[int(second)],
                    float(artifact.edge_score[position]),
                ]
            )
    return destination


__all__ = [
    "GRAPH_SCHEMA_VERSION",
    "ProteinGraphArtifact",
    "build_string_graph",
    "canonicalize_undirected_edges",
    "connected_components",
    "degree_preserving_rewire",
    "load_protein_names_from_matrix",
    "read_string_mapping",
    "sha256_file",
    "summarize_graph",
    "undirected_degree",
    "write_edge_table",
]
