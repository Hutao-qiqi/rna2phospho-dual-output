"""Load and validate fixed binary CoPhee graph priors for the experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path: str | Path) -> str:
    """Hash the exact vocabulary file bytes used by the data bundle."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class CoPheePriorBundle:
    site_site_edge_index: np.ndarray
    kinase_site_edge_index: np.ndarray
    target_vocabulary: list[str]
    kinase_vocabulary: list[str]
    audit: dict[str, object]
    report: dict[str, object]

    def checkpoint_tensors(self) -> dict[str, np.ndarray]:
        return {
            "site_site_edge_index": self.site_site_edge_index,
            "kinase_site_edge_index": self.kinase_site_edge_index,
        }


def _read_indexed_vocabulary(
    path: Path, index_column: str, value_column: str
) -> list[str]:
    table = pd.read_csv(path, sep="\t", dtype={value_column: str})
    missing = {index_column, value_column}.difference(table.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns: {sorted(missing)}")
    indices = pd.to_numeric(table[index_column], errors="raise").to_numpy(np.int64)
    expected = np.arange(len(table), dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise ValueError(f"{path.name} indices must be consecutive from zero")
    values = table[value_column].astype(str).tolist()
    if len(values) != len(set(values)):
        raise ValueError(f"{path.name} contains duplicate vocabulary values")
    return values


def _read_edge_index(
    path: Path,
    source_key: str,
    destination_key: str,
    source_size: int,
    destination_size: int,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        required = {source_key, destination_key}
        if not required.issubset(archive.files):
            raise ValueError(f"{path.name} lacks arrays: {sorted(required.difference(archive.files))}")
        source = np.asarray(archive[source_key], dtype=np.int64).reshape(-1)
        destination = np.asarray(archive[destination_key], dtype=np.int64).reshape(-1)
    if source.shape != destination.shape:
        raise ValueError(f"{path.name} edge arrays have different lengths")
    if source.size:
        if source.min() < 0 or source.max() >= source_size:
            raise ValueError(f"{path.name} source index is outside its vocabulary")
        if destination.min() < 0 or destination.max() >= destination_size:
            raise ValueError(f"{path.name} destination index is outside its vocabulary")
        pairs = np.stack([source, destination], axis=1)
        if len(np.unique(pairs, axis=0)) != len(pairs):
            raise ValueError(f"{path.name} contains duplicate binary edges")
    return np.stack([source, destination], axis=0)


def _check_audit_hash(
    audit: dict[str, object], names: tuple[str, ...], observed: str, label: str
) -> None:
    expected = next((str(audit[name]) for name in names if name in audit), None)
    if expected is not None and expected != observed:
        raise ValueError(f"{label} hash differs from audit_summary.json")


def load_cophee_prior_bundle(
    directory: str | Path,
    expected_targets: list[str] | None = None,
    *,
    include_site_site: bool = True,
    include_kinase_site: bool = True,
) -> CoPheePriorBundle:
    root = Path(directory)
    required = {
        "site_site": root / "site_site_edges.npz",
        "kinase_site": root / "kinase_site_edges.npz",
        "targets": root / "target_vocab.tsv",
        "kinases": root / "kinase_vocab.tsv",
        "audit": root / "audit_summary.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("CoPhee prior bundle is incomplete: " + ", ".join(missing))

    targets = _read_indexed_vocabulary(required["targets"], "site_index", "gene_site")
    kinases = _read_indexed_vocabulary(required["kinases"], "kinase_index", "kinase")
    if expected_targets is not None and list(map(str, expected_targets)) != targets:
        raise ValueError("CoPhee target vocabulary order differs from phosphosite manifest")

    site_site = _read_edge_index(
        required["site_site"], "src_site_index", "dst_site_index", len(targets), len(targets)
    )
    if site_site.shape[1]:
        pairs = set(map(tuple, site_site.T.tolist()))
        missing_reverse = next(((left, right) for left, right in pairs if (right, left) not in pairs), None)
        if missing_reverse is not None:
            raise ValueError(f"site_site_edges.npz lacks reverse edge for {missing_reverse}")
    kinase_site = _read_edge_index(
        required["kinase_site"], "kinase_index", "site_index", len(kinases), len(targets)
    )
    audit = json.loads(required["audit"].read_text(encoding="utf-8"))
    target_hash = file_sha256(required["targets"])
    kinase_hash = file_sha256(required["kinases"])
    _check_audit_hash(audit, ("target_vocab_sha256", "target_vocabulary_sha256"), target_hash, "target vocabulary")
    _check_audit_hash(audit, ("kinase_vocab_sha256", "kinase_vocabulary_sha256"), kinase_hash, "kinase vocabulary")

    if not include_site_site:
        site_site = np.empty((2, 0), dtype=np.int64)
    if not include_kinase_site:
        kinase_site = np.empty((2, 0), dtype=np.int64)
    directly_mapped = int(np.unique(kinase_site[1]).size) if kinase_site.shape[1] else 0
    cophee_connected = int(np.unique(site_site).size) if site_site.shape[1] else 0
    report: dict[str, object] = {
        "prior_type": "fixed_binary_sparse_graph",
        "n_targets": len(targets),
        "n_kinases": len(kinases),
        "n_site_site_directed_edges": int(site_site.shape[1]),
        "n_kinase_site_edges": int(kinase_site.shape[1]),
        "direct_kinase_site_coverage_count": directly_mapped,
        "direct_kinase_site_coverage_fraction": directly_mapped / max(len(targets), 1),
        "cophee_site_graph_coverage_count": cophee_connected,
        "cophee_site_graph_coverage_fraction": cophee_connected / max(len(targets), 1),
        "target_vocab_sha256": target_hash,
        "kinase_vocab_sha256": kinase_hash,
        "site_site_graph_enabled": include_site_site,
        "kinase_site_graph_enabled": include_kinase_site,
        "edge_confidence_used": False,
        "sample_dynamic_kinase_weights_used": False,
    }
    return CoPheePriorBundle(site_site, kinase_site, targets, kinases, audit, report)
