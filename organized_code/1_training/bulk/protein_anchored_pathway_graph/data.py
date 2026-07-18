from __future__ import annotations

import json
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr


SIGNALLING_TERMS = (
    "AKT",
    "APOPTOSIS",
    "B_CELL",
    "CALCIUM",
    "CELL_CYCLE",
    "DNA_DAMAGE",
    "EGFR",
    "ERK",
    "FGFR",
    "GROWTH_FACTOR",
    "HIPPO",
    "IGF",
    "INSULIN",
    "INTERFERON",
    "JAK",
    "MAPK",
    "MTOR",
    "NF_KAPPA",
    "NOTCH",
    "PI3K",
    "RAS",
    "RECEPTOR",
    "SIGNAL",
    "STAT",
    "STRESS",
    "T_CELL",
    "TGFB",
    "TNF",
    "VEGF",
    "WNT",
)


@dataclass(frozen=True)
class InputPaths:
    rna: Path
    total_protein: Path
    phosphosite: Path
    sample_manifest: Path
    phosphosite_manifest: Path
    total_protein_manifest: Path | None
    hallmark_gmt: Path
    c2_gmt: Path
    kinase_edges: Path | None


@dataclass(frozen=True)
class PathwayTensors:
    names: list[str]
    genes: list[list[str]]
    rna_index: np.ndarray
    rna_mask: np.ndarray
    protein_index: np.ndarray
    protein_mask: np.ndarray
    site_pathway_index: np.ndarray
    site_pathway_mask: np.ndarray
    site_pathway_weight: np.ndarray
    parent_protein_index: np.ndarray
    parent_protein_mask: np.ndarray
    site_kinase_index: np.ndarray
    site_kinase_mask: np.ndarray
    kinase_names: list[str]


def _first_existing(paths: Sequence[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    checked = "\n".join(str(path) for path in paths)
    raise FileNotFoundError(f"Could not locate {label}. Checked:\n{checked}")


def discover_input_paths(project_root: Path) -> InputPaths:
    portable = project_root / "SCP682_PORTABLE"
    main = project_root / "SCP682_MAIN"
    v4_data = portable / "v4_engine/data/pancancer_multi_task_locked_v2"
    search_roots = [project_root, *project_root.parents]
    multiomics_roots = [
        root / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2" for root in search_roots
    ]
    prior_roots = [root / "01_data/pathway_prior" for root in search_roots]
    kinase_candidates = [
        *[
            prior / "intermediate/kstar_20260516/kstar_default_network_edges_long.tsv"
            for prior in prior_roots
        ],
        main / "priors/kstar_default_network_edges_long.tsv",
        *[
            prior / "intermediate/kstar_20260519/scp682_ppko_v5_kstar_kinase_site_edges.tsv"
            for prior in prior_roots
        ],
    ]
    kinase_path = next((path for path in kinase_candidates if path.exists()), None)
    return InputPaths(
        rna=_first_existing(
            [
                v4_data / "rna_log2_tpm_paired.parquet",
                main / "training_set/rna_log2_tpm_paired.parquet",
                *[root / "rna_log2_tpm_paired.parquet" for root in multiomics_roots],
            ],
            "paired RNA matrix",
        ),
        total_protein=_first_existing(
            [
                v4_data / "total_protein_gene_study_zscore_min20pct.parquet",
                *[
                    root / "total_protein_gene_study_zscore_min20pct.parquet"
                    for root in multiomics_roots
                ],
            ],
            "paired total-protein matrix",
        ),
        phosphosite=_first_existing(
            [main / "training_set/observed_phosphosite.parquet", portable / "training_set/observed_phosphosite.parquet"],
            "observed phosphosite matrix",
        ),
        sample_manifest=_first_existing(
            [
                v4_data / "sample_manifest.tsv",
                main / "training_set/sample_manifest.tsv",
                *[root / "sample_manifest.tsv" for root in multiomics_roots],
            ],
            "sample manifest",
        ),
        phosphosite_manifest=_first_existing(
            [main / "training_set/phosphosite_target_manifest.tsv", portable / "training_set/phosphosite_target_manifest.tsv"],
            "phosphosite target manifest",
        ),
        total_protein_manifest=next(
            (
                path
                for path in [
                    main / "training_set/total_protein_target_manifest.tsv",
                    portable / "training_set/total_protein_target_manifest.tsv",
                ]
                if path.exists()
            ),
            None,
        ),
        hallmark_gmt=_first_existing(
            [root / "resources/msigdb/h.all.v2025.1.Hs.symbols.gmt" for root in search_roots],
            "MSigDB Hallmark GMT",
        ),
        c2_gmt=_first_existing(
            [root / "resources/msigdb/c2.cp.v2025.1.Hs.symbols.gmt" for root in search_roots],
            "MSigDB canonical-pathway GMT",
        ),
        kinase_edges=kinase_path,
    )


def read_gmt(path: Path) -> OrderedDict[str, list[str]]:
    gene_sets: OrderedDict[str, list[str]] = OrderedDict()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            genes = list(dict.fromkeys(gene.strip().upper() for gene in fields[2:] if gene.strip()))
            gene_sets[fields[0]] = genes
    return gene_sets


def select_pathways(
    hallmark_gmt: Path,
    c2_gmt: Path,
    available_genes: Sequence[str],
    parent_genes: Sequence[str],
    variance_order: Sequence[str],
    max_pathways: int = 72,
    min_genes: int = 8,
    max_genes_per_pathway: int = 128,
) -> OrderedDict[str, list[str]]:
    available = set(map(str, available_genes))
    parents = set(map(str, parent_genes))
    priority = {gene: index for index, gene in enumerate(map(str, variance_order))}
    hallmark = read_gmt(hallmark_gmt)
    c2 = read_gmt(c2_gmt)

    def trim(genes: Iterable[str]) -> list[str]:
        overlap = [gene for gene in genes if gene in available]
        overlap.sort(key=lambda gene: (priority.get(gene, len(priority)), gene))
        return overlap[:max_genes_per_pathway]

    chosen: OrderedDict[str, list[str]] = OrderedDict()
    global_genes = [gene for gene in variance_order if gene in available][:max_genes_per_pathway]
    chosen["GLOBAL_CONTEXT"] = global_genes
    for name, genes in hallmark.items():
        kept = trim(genes)
        if len(kept) >= min_genes:
            chosen[name] = kept

    candidates: list[tuple[int, int, str, list[str]]] = []
    for name, genes in c2.items():
        if not any(term in name.upper() for term in SIGNALLING_TERMS):
            continue
        kept = trim(genes)
        if len(kept) < min_genes:
            continue
        parent_overlap = len(set(kept) & parents)
        candidates.append((parent_overlap, len(kept), name, kept))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))

    existing_sets = [set(genes) for genes in chosen.values()]
    for _, _, name, genes in candidates:
        if len(chosen) >= max_pathways:
            break
        gene_set = set(genes)
        redundant = any(
            len(gene_set & prior) / max(len(gene_set | prior), 1) >= 0.85 for prior in existing_sets
        )
        if redundant:
            continue
        chosen[name] = genes
        existing_sets.append(gene_set)
    if len(chosen) < 2:
        raise ValueError("Pathway selection produced fewer than two usable pathways")
    return chosen


def _padded_indices(
    members: Sequence[Sequence[str]],
    name_to_index: Mapping[str, int],
    max_members: int,
) -> tuple[np.ndarray, np.ndarray]:
    index = np.zeros((len(members), max_members), dtype=np.int64)
    mask = np.zeros((len(members), max_members), dtype=bool)
    for row, names in enumerate(members):
        mapped = [name_to_index[name] for name in names if name in name_to_index][:max_members]
        if mapped:
            index[row, : len(mapped)] = mapped
            mask[row, : len(mapped)] = True
    return index, mask


def load_site_kinases(path: Path | None, targets: Sequence[str]) -> dict[str, set[str]]:
    target_set = set(map(str, targets))
    mapping: dict[str, set[str]] = defaultdict(set)
    if path is None or not path.exists():
        return mapping
    separator = "\t" if path.suffix.lower() != ".csv" else ","
    columns = pd.read_csv(path, sep=separator, nrows=0).columns.tolist()
    kinase_column = next((name for name in ["kinase", "kinases", "regulator"] if name in columns), None)
    if kinase_column is None:
        return mapping
    direct_site_columns = [
        name for name in ["gene_site", "gene_site_id", "target_id", "model_site_id"] if name in columns
    ]
    has_gene_site_parts = {"substrate_gene", "site"}.issubset(columns)
    if not direct_site_columns and not has_gene_site_parts:
        return mapping
    use_columns = [kinase_column, *direct_site_columns]
    if has_gene_site_parts:
        use_columns.extend(["substrate_gene", "site"])
    use_columns = list(dict.fromkeys(use_columns))
    for frame in pd.read_csv(path, sep=separator, usecols=use_columns, chunksize=500_000):
        kinase_values = frame[kinase_column].fillna("").astype(str).str.upper()
        candidate_columns: list[pd.Series] = [frame[name].fillna("").astype(str) for name in direct_site_columns]
        if has_gene_site_parts:
            candidate_columns.append(
                frame["substrate_gene"].fillna("").astype(str).str.upper()
                + "|"
                + frame["site"].fillna("").astype(str).str.upper()
            )
        for candidate in candidate_columns:
            keep = candidate.isin(target_set) & kinase_values.ne("")
            for site, kinase in zip(candidate[keep], kinase_values[keep]):
                mapping[str(site)].add(str(kinase))
    return mapping


def build_pathway_tensors(
    pathways: Mapping[str, Sequence[str]],
    rna_genes: Sequence[str],
    protein_genes: Sequence[str],
    targets: Sequence[str],
    parent_genes: Sequence[str],
    site_kinases: Mapping[str, set[str]],
    max_rna_members: int = 128,
    max_protein_members: int = 128,
    max_site_pathways: int = 8,
    max_site_kinases: int = 4,
) -> PathwayTensors:
    pathway_names = list(pathways)
    pathway_genes = [list(map(str, pathways[name])) for name in pathway_names]
    pathway_sets = [set(genes) for genes in pathway_genes]
    rna_index, rna_mask = _padded_indices(
        pathway_genes, {gene: i for i, gene in enumerate(map(str, rna_genes))}, max_rna_members
    )
    protein_index, protein_mask = _padded_indices(
        pathway_genes, {gene: i for i, gene in enumerate(map(str, protein_genes))}, max_protein_members
    )

    protein_lookup = {gene: i for i, gene in enumerate(map(str, protein_genes))}
    parent_index = np.zeros(len(targets), dtype=np.int64)
    parent_mask = np.zeros(len(targets), dtype=bool)
    for site_index, parent in enumerate(map(str, parent_genes)):
        if parent in protein_lookup:
            parent_index[site_index] = protein_lookup[parent]
            parent_mask[site_index] = True

    kinase_names = sorted({kinase for values in site_kinases.values() for kinase in values})
    kinase_lookup = {kinase: i + 1 for i, kinase in enumerate(kinase_names)}
    site_kinase_index = np.zeros((len(targets), max_site_kinases), dtype=np.int64)
    site_kinase_mask = np.zeros((len(targets), max_site_kinases), dtype=bool)
    site_pathway_index = np.zeros((len(targets), max_site_pathways), dtype=np.int64)
    site_pathway_mask = np.zeros((len(targets), max_site_pathways), dtype=bool)
    site_pathway_weight = np.zeros((len(targets), max_site_pathways), dtype=np.float32)

    for site_index, (target, parent) in enumerate(zip(map(str, targets), map(str, parent_genes))):
        kinases = sorted(site_kinases.get(target, set()))[:max_site_kinases]
        for offset, kinase in enumerate(kinases):
            site_kinase_index[site_index, offset] = kinase_lookup[kinase]
            site_kinase_mask[site_index, offset] = True

        scores: dict[int, float] = defaultdict(float)
        for path_index, genes in enumerate(pathway_sets):
            if parent in genes:
                scores[path_index] += 2.0
            scores[path_index] += sum(1.0 for kinase in kinases if kinase in genes)
        selected = sorted(scores.items(), key=lambda item: (-item[1], pathway_names[item[0]]))
        selected = [item for item in selected if item[1] > 0][:max_site_pathways]
        if not selected:
            selected = [(0, 1.0)]
        total_weight = sum(weight for _, weight in selected)
        for offset, (path_index, weight) in enumerate(selected):
            site_pathway_index[site_index, offset] = path_index
            site_pathway_mask[site_index, offset] = True
            site_pathway_weight[site_index, offset] = weight / total_weight

    return PathwayTensors(
        names=pathway_names,
        genes=pathway_genes,
        rna_index=rna_index,
        rna_mask=rna_mask,
        protein_index=protein_index,
        protein_mask=protein_mask,
        site_pathway_index=site_pathway_index,
        site_pathway_mask=site_pathway_mask,
        site_pathway_weight=site_pathway_weight,
        parent_protein_index=parent_index,
        parent_protein_mask=parent_mask,
        site_kinase_index=site_kinase_index,
        site_kinase_mask=site_kinase_mask,
        kinase_names=kinase_names,
    )


def fit_zscore(values: np.ndarray, train_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = values[train_index]
    mean = np.nanmean(train, axis=0).astype(np.float32)
    std = np.nanstd(train, axis=0).astype(np.float32)
    mean[~np.isfinite(mean)] = 0.0
    std[(~np.isfinite(std)) | (std < 1.0e-6)] = 1.0
    return mean, std


def apply_zscore(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    output = (values - mean[None, :]) / std[None, :]
    return np.nan_to_num(output, nan=0.0, posinf=6.0, neginf=-6.0).clip(-6.0, 6.0).astype(np.float32)


def sample_rank_encode(values: np.ndarray, chunk_size: int = 128) -> np.ndarray:
    output = np.empty(values.shape, dtype=np.float32)
    denominator = max(values.shape[1] - 1, 1)
    for start in range(0, values.shape[0], chunk_size):
        end = min(start + chunk_size, values.shape[0])
        ranked = rankdata(values[start:end], axis=1, method="average")
        output[start:end] = ((ranked - 1.0) / denominator * 2.0 - 1.0).astype(np.float32)
    return output


def fit_parent_calibration(
    protein_hat: np.ndarray,
    phosphosite: np.ndarray,
    parent_index: np.ndarray,
    parent_mask: np.ndarray,
    train_index: np.ndarray,
    min_samples: int = 20,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_sites = phosphosite.shape[1]
    intercept = np.zeros(n_sites, dtype=np.float32)
    scale = np.zeros(n_sites, dtype=np.float32)
    sample_count = np.zeros(n_sites, dtype=np.int32)
    for start in range(0, n_sites, chunk_size):
        end = min(start + chunk_size, n_sites)
        y = phosphosite[np.ix_(train_index, np.arange(start, end))].astype(np.float64)
        mapped = parent_mask[start:end]
        x = protein_hat[np.ix_(train_index, parent_index[start:end])].astype(np.float64)
        valid = np.isfinite(y) & np.isfinite(x) & mapped[None, :]
        count = valid.sum(axis=0)
        sample_count[start:end] = count
        safe_count = np.maximum(count, 1)
        x_mean = np.where(valid, x, 0.0).sum(axis=0) / safe_count
        y_mean = np.where(valid, y, 0.0).sum(axis=0) / safe_count
        x_center = np.where(valid, x - x_mean[None, :], 0.0)
        y_center = np.where(valid, y - y_mean[None, :], 0.0)
        variance = (x_center * x_center).sum(axis=0)
        covariance = (x_center * y_center).sum(axis=0)
        beta = np.divide(covariance, variance, out=np.zeros_like(covariance), where=variance > 1.0e-8)
        beta = np.clip(beta, -3.0, 3.0)
        valid_fit = (count >= min_samples) & mapped
        beta[~valid_fit] = 0.0
        site_intercept = y_mean - beta * x_mean
        fallback = np.nanmedian(y, axis=0)
        fallback[~np.isfinite(fallback)] = 0.0
        site_intercept[~valid_fit] = fallback[~valid_fit]
        intercept[start:end] = site_intercept.astype(np.float32)
        scale[start:end] = beta.astype(np.float32)
    return intercept, scale, sample_count


def parent_baseline(
    protein_hat: np.ndarray,
    intercept: np.ndarray,
    scale: np.ndarray,
    parent_index: np.ndarray,
    parent_mask: np.ndarray,
) -> np.ndarray:
    parent = protein_hat[:, parent_index] * parent_mask[None, :]
    return (intercept[None, :] + scale[None, :] * parent).astype(np.float32)


def masked_row_median_center(values: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    masked = np.where(observed_mask, values, np.nan)
    median = np.nanmedian(masked, axis=1, keepdims=True)
    median[~np.isfinite(median)] = 0.0
    output = values - median
    output[~observed_mask] = 0.0
    return output.astype(np.float32)


def pathway_summary_features(
    rna_rank: np.ndarray,
    protein_hat: np.ndarray,
    tensors: PathwayTensors,
) -> np.ndarray:
    n_samples = rna_rank.shape[0]
    n_pathways = len(tensors.names)
    output = np.zeros((n_samples, n_pathways, 6), dtype=np.float32)
    for path_index in range(n_pathways):
        rna_ids = tensors.rna_index[path_index, tensors.rna_mask[path_index]]
        protein_ids = tensors.protein_index[path_index, tensors.protein_mask[path_index]]
        if len(rna_ids):
            block = rna_rank[:, rna_ids]
            output[:, path_index, 0] = block.mean(axis=1)
            output[:, path_index, 1] = block.std(axis=1)
            output[:, path_index, 2] = np.median(block, axis=1)
        if len(protein_ids):
            block = protein_hat[:, protein_ids]
            output[:, path_index, 3] = block.mean(axis=1)
            output[:, path_index, 4] = block.std(axis=1)
            output[:, path_index, 5] = np.median(block, axis=1)
    return output


def build_pathway_knn(
    source_features: np.ndarray,
    query_features: np.ndarray,
    k: int,
    exclude_identity: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    n_source, n_pathways, _ = source_features.shape
    n_query = query_features.shape[0]
    max_k = n_source - 1 if exclude_identity and n_source == n_query else n_source
    use_k = max(1, min(int(k), max_k))
    index = np.empty((n_pathways, n_query, use_k), dtype=np.int64)
    similarity = np.empty((n_pathways, n_query, use_k), dtype=np.float32)
    for path_index in range(n_pathways):
        source = source_features[:, path_index].astype(np.float64)
        query = query_features[:, path_index].astype(np.float64)
        mean = source.mean(axis=0, keepdims=True)
        std = source.std(axis=0, keepdims=True)
        std[std < 1.0e-6] = 1.0
        source = (source - mean) / std
        query = (query - mean) / std
        source /= np.linalg.norm(source, axis=1, keepdims=True).clip(min=1.0e-8)
        query /= np.linalg.norm(query, axis=1, keepdims=True).clip(min=1.0e-8)
        score = query @ source.T
        if exclude_identity and n_source == n_query:
            np.fill_diagonal(score, -np.inf)
        candidate = np.argpartition(-score, kth=use_k - 1, axis=1)[:, :use_k]
        candidate_score = np.take_along_axis(score, candidate, axis=1)
        order = np.argsort(-candidate_score, axis=1)
        candidate = np.take_along_axis(candidate, order, axis=1)
        candidate_score = np.take_along_axis(candidate_score, order, axis=1)
        index[path_index] = candidate
        similarity[path_index] = candidate_score.astype(np.float32)
    return index, similarity


def masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    loss = torch.nn.functional.huber_loss(prediction, target, reduction="none", delta=delta)
    weight = mask.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def per_site_spearman(
    observed: np.ndarray,
    predicted: np.ndarray,
    targets: Sequence[str],
    sample_index: np.ndarray,
    min_samples: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for site_index, target in enumerate(targets):
        y = observed[sample_index, site_index]
        p = predicted[sample_index, site_index]
        valid = np.isfinite(y) & np.isfinite(p)
        if valid.sum() < min_samples or np.nanstd(p[valid]) < 1.0e-12:
            rho, p_value = np.nan, np.nan
        else:
            rho, p_value = spearmanr(y[valid], p[valid])
        rows.append(
            {
                "target": target,
                "n_samples_used": int(valid.sum()),
                "spearman": float(rho) if np.isfinite(rho) else np.nan,
                "rho_p_value": float(p_value) if np.isfinite(p_value) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_pathway_manifest(path: Path, tensors: PathwayTensors) -> None:
    rows = []
    for index, (name, genes) in enumerate(zip(tensors.names, tensors.genes)):
        rows.append(
            {
                "pathway_index": index,
                "pathway": name,
                "n_genes": len(genes),
                "genes": ";".join(genes),
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def write_input_manifest(path: Path, inputs: InputPaths) -> None:
    payload = {
        key: str(value) if value is not None else None
        for key, value in inputs.__dict__.items()
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
