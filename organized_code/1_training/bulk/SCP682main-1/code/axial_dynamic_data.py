"""Leakage-controlled data utilities for the axial phosphosite model."""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


TRAIN_ROLE = "selection_train"
VALIDATION_ROLE = "selection_validation"
SEALED_ROLE = "sealed_test"
TRAIN_PROTEIN_ROLE = "cross_fitted"
VALIDATION_PROTEIN_ROLE = "selection_train_only"


@dataclass(frozen=True)
class SplitContract:
    sample_ids: np.ndarray
    roles: np.ndarray
    train_ids: np.ndarray
    validation_ids: np.ndarray
    sealed_ids: np.ndarray

    @property
    def development_ids(self) -> np.ndarray:
        return np.concatenate([self.train_ids, self.validation_ids])


@dataclass(frozen=True)
class BiologicalPrior:
    pathway_names: list[str]
    pathway_genes: list[list[str]]
    pathway_full_genes: list[list[str]]
    rna_pathway_index: np.ndarray
    rna_pathway_mask: np.ndarray
    protein_pathway_index: np.ndarray
    protein_pathway_mask: np.ndarray
    pathway_relation_mask: np.ndarray
    site_pathway_index: np.ndarray
    site_pathway_mask: np.ndarray
    site_pathway_weight: np.ndarray
    parent_protein_index: np.ndarray
    parent_protein_mask: np.ndarray
    site_kinase_index: np.ndarray
    site_kinase_mask: np.ndarray
    kinase_names: list[str]
    site_coverage: np.ndarray


@dataclass(frozen=True)
class PathwaySelection:
    members: OrderedDict[str, list[str]]
    full_genes: OrderedDict[str, list[str]]


@dataclass(frozen=True)
class StudySiteStandardization:
    """Training-fold parameters for study-specific phosphosite scaling."""

    study_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    count: np.ndarray
    available: np.ndarray


def read_sample_studies(
    path: str | Path,
    sample_ids: Sequence[str],
    *,
    study_column: str,
    sample_id_column: str = "sample_id",
) -> np.ndarray:
    """Read the required study identifier without touching outcome matrices."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing sample metadata: {path}")
    separator = "," if path.suffix.lower() == ".csv" else "\t"
    table = pd.read_csv(path, sep=separator, low_memory=False)
    required = {str(sample_id_column), str(study_column)}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"sample metadata lacks required columns: {sorted(missing)}")
    table = table.loc[:, [str(sample_id_column), str(study_column)]].copy()
    table[str(sample_id_column)] = table[str(sample_id_column)].astype(str)
    if table[str(sample_id_column)].duplicated().any():
        raise ValueError("sample metadata contains duplicate sample identifiers")
    if table[str(study_column)].isna().any():
        raise ValueError(f"sample metadata column {study_column!r} contains missing values")
    table[str(study_column)] = table[str(study_column)].astype(str).str.strip()
    if (table[str(study_column)] == "").any():
        raise ValueError(f"sample metadata column {study_column!r} contains empty values")
    lookup = table.set_index(str(sample_id_column))[str(study_column)]
    wanted = [str(value) for value in sample_ids]
    absent = [sample for sample in wanted if sample not in lookup.index]
    if absent:
        raise ValueError(
            f"sample metadata lacks {len(absent)} required samples; examples={absent[:5]}"
        )
    return lookup.loc[wanted].to_numpy(str)


def validate_case_split_disjointness(
    path: str | Path,
    split: SplitContract,
    *,
    sample_id_column: str,
    case_id_column: str = "case_submitter_id",
) -> None:
    """Reject patient/case reuse across train, validation and sealed roles."""
    path = Path(path)
    separator = "," if path.suffix.lower() == ".csv" else "\t"
    table = pd.read_csv(path, sep=separator, low_memory=False)
    required = {str(sample_id_column), str(case_id_column)}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"sample metadata lacks case-split columns: {sorted(missing)}")
    table = table.loc[:, [str(sample_id_column), str(case_id_column)]].copy()
    table[str(sample_id_column)] = table[str(sample_id_column)].astype(str)
    if table[str(sample_id_column)].duplicated().any():
        raise ValueError("sample metadata contains duplicate sample identifiers")
    if table[str(case_id_column)].isna().any():
        raise ValueError(f"sample metadata column {case_id_column!r} contains missing values")
    role_lookup = dict(zip(split.sample_ids.tolist(), split.roles.tolist()))
    table = table[table[str(sample_id_column)].isin(role_lookup)].copy()
    if len(table) != len(split.sample_ids):
        raise ValueError("sample metadata must cover every locked split sample for case auditing")
    table["__role"] = table[str(sample_id_column)].map(role_lookup)
    crossed = table.groupby(str(case_id_column))["__role"].nunique()
    bad_cases = crossed[crossed > 1].index.astype(str).tolist()
    if bad_cases:
        raise ValueError(
            "case identifiers overlap locked split roles; examples=" f"{bad_cases[:5]}"
        )


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def validate_split_manifest(
    path: str | Path,
    *,
    expected_sizes: tuple[int, int, int] | None = (916, 229, 286),
) -> SplitContract:
    table = pd.read_csv(path, sep="\t")
    required = {"sample_id", "role"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"split manifest lacks columns: {sorted(missing)}")
    table = table.loc[:, ["sample_id", "role"]].copy()
    table["sample_id"] = table["sample_id"].astype(str)
    table["role"] = table["role"].astype(str)
    if table["sample_id"].duplicated().any():
        raise ValueError("split manifest contains duplicate sample identifiers")
    allowed = {TRAIN_ROLE, VALIDATION_ROLE, SEALED_ROLE}
    unknown = sorted(set(table["role"]) - allowed)
    if unknown:
        raise ValueError(f"split manifest contains unknown roles: {unknown}")
    train = table.loc[table["role"] == TRAIN_ROLE, "sample_id"].to_numpy(str)
    validation = table.loc[table["role"] == VALIDATION_ROLE, "sample_id"].to_numpy(str)
    sealed = table.loc[table["role"] == SEALED_ROLE, "sample_id"].to_numpy(str)
    sizes = (len(train), len(validation), len(sealed))
    if expected_sizes is not None and sizes != tuple(expected_sizes):
        raise ValueError(f"locked split sizes differ: observed={sizes}, expected={expected_sizes}")
    if sizes[0] < 1 or sizes[1] < 1 or sizes[2] < 0:
        raise ValueError("training and validation roles must contain samples")
    return SplitContract(
        sample_ids=table["sample_id"].to_numpy(str),
        roles=table["role"].to_numpy(str),
        train_ids=train,
        validation_ids=validation,
        sealed_ids=sealed,
    )


def validate_protein_prediction_provenance(
    path: str | Path,
    split: SplitContract,
) -> pd.DataFrame:
    """Require one auditable prediction source for every development sample."""
    table = pd.read_csv(path, sep="\t")
    required = {
        "sample_id",
        "prediction_role",
        "phosphosite_labels_used",
        "source_archive",
        "source_row_index",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"protein provenance lacks columns: {sorted(missing)}")
    table = table.copy()
    table["sample_id"] = table["sample_id"].astype(str)
    table["prediction_role"] = table["prediction_role"].astype(str)
    if table["sample_id"].duplicated().any():
        raise ValueError("protein provenance contains duplicate sample identifiers")
    if table["phosphosite_labels_used"].astype(str).str.lower().isin({"1", "true", "yes"}).any():
        raise ValueError("protein predictions declare use of phosphosite labels")
    enhanced = {"source_model_id", "source_fold", "sample_in_source_training"}
    present = enhanced & set(table.columns)
    if present and present != enhanced:
        raise ValueError("protein provenance contains an incomplete enhanced cross-fit contract")
    if present == enhanced:
        source_training_flag = (
            table["sample_in_source_training"].astype(str).str.lower().str.strip()
        )
        if not source_training_flag.isin({"0", "false", "no"}).all():
            raise ValueError("protein provenance shows a sample was used to train its own predictor")
        for column in ["source_model_id", "source_fold"]:
            if table[column].isna().any() or (table[column].astype(str).str.strip() == "").any():
                raise ValueError(f"protein provenance lacks per-sample {column} evidence")
        evidence_level = "per_sample_fold_and_model"
    else:
        if table["source_archive"].isna().any() or (
            table["source_archive"].astype(str).str.strip() == ""
        ).any():
            raise ValueError("legacy protein provenance lacks source archive evidence")
        source_rows = pd.to_numeric(table["source_row_index"], errors="coerce")
        if source_rows.isna().any() or (source_rows < 0).any():
            raise ValueError("legacy protein provenance has invalid source row evidence")
        evidence_level = "audited_archive_and_role_only"
    expected_ids = set(split.development_ids.tolist())
    observed_ids = set(table["sample_id"].tolist())
    if observed_ids != expected_ids:
        missing_ids = sorted(expected_ids - observed_ids)[:5]
        extra_ids = sorted(observed_ids - expected_ids)[:5]
        raise ValueError(
            f"protein provenance sample set differs; missing={missing_ids}, extra={extra_ids}"
        )
    role = table.set_index("sample_id")["prediction_role"]
    if not (role.loc[split.train_ids] == TRAIN_PROTEIN_ROLE).all():
        raise ValueError("all 916 training predictions must be marked cross_fitted")
    if not (role.loc[split.validation_ids] == VALIDATION_PROTEIN_ROLE).all():
        raise ValueError("all 229 validation predictions must be marked selection_train_only")
    if set(split.sealed_ids) & observed_ids:
        raise ValueError("sealed samples are forbidden in the protein prediction input")
    table.attrs["crossfit_evidence_level"] = evidence_level
    return table


def read_parquet_rows(
    path: str | Path,
    sample_ids: Sequence[str],
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read only declared development rows through a parquet index filter."""
    wanted = [str(value) for value in sample_ids]
    selected_columns = None if columns is None else list(map(str, columns))
    try:
        frame = pd.read_parquet(
            path,
            columns=selected_columns,
            filters=[("__index_level_0__", "in", wanted)],
            engine="pyarrow",
        )
    except Exception as exc:
        raise RuntimeError(
            "filtered parquet reading failed; refusing to load a full matrix containing sealed labels"
        ) from exc
    frame.index = frame.index.astype(str)
    if frame.index.duplicated().any():
        raise ValueError(f"matrix contains duplicate sample identifiers: {path}")
    missing = [sample for sample in wanted if sample not in frame.index]
    if missing:
        raise ValueError(f"matrix lacks {len(missing)} requested samples; examples={missing[:5]}")
    return frame.reindex(wanted)


def read_prediction_matrix(path: str | Path, split: SplitContract) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    expected = list(split.development_ids)
    if set(frame.index) != set(expected):
        raise ValueError("protein prediction matrix must contain exactly the 916+229 development samples")
    if frame.index.duplicated().any() or frame.columns.duplicated().any():
        raise ValueError("protein prediction matrix has duplicate rows or columns")
    values = frame.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=np.float32)).all():
        raise ValueError("protein prediction matrix contains missing or non-finite values")
    return values.reindex(expected).astype(np.float32)


def read_gmt(path: str | Path) -> OrderedDict[str, list[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing pathway GMT: {path}")
    result: OrderedDict[str, list[str]] = OrderedDict()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                result[fields[0]] = list(
                    dict.fromkeys(gene.strip().upper() for gene in fields[2:] if gene.strip())
                )
    if not result:
        raise ValueError(f"no pathways found in {path}")
    return result


def select_pathways(
    hallmark_gmt: str | Path,
    canonical_gmt: str | Path,
    rna_genes: Sequence[str],
    parent_genes: Sequence[str],
    train_variance: np.ndarray,
    *,
    max_pathways: int = 128,
    max_members: int = 128,
    minimum_members: int = 8,
) -> PathwaySelection:
    if max_pathways < 1:
        raise ValueError("max_pathways must be at least one")
    available = {str(gene).upper() for gene in rna_genes}
    parents = {str(gene).upper() for gene in parent_genes}
    order = np.argsort(-np.nan_to_num(train_variance, nan=-np.inf))
    variance_genes = [str(rna_genes[index]).upper() for index in order]
    priority = {gene: index for index, gene in enumerate(variance_genes)}

    def trim(genes: Iterable[str]) -> list[str]:
        overlap = [gene for gene in genes if gene in available]
        overlap.sort(key=lambda gene: (priority.get(gene, len(priority)), gene))
        return overlap[:max_members]

    selected: OrderedDict[str, list[str]] = OrderedDict()
    selected_full: OrderedDict[str, list[str]] = OrderedDict()
    selected["GLOBAL_CONTEXT"] = variance_genes[:max_members]
    selected_full["GLOBAL_CONTEXT"] = list(selected["GLOBAL_CONTEXT"])
    full_sources = [read_gmt(hallmark_gmt), read_gmt(canonical_gmt)]
    candidates: list[tuple[int, int, str, list[str], list[str]]] = []
    for source_index, source in enumerate(full_sources):
        for name, full_genes in source.items():
            kept = trim(full_genes)
            if len(kept) < minimum_members:
                continue
            parent_overlap = len(parents.intersection(full_genes))
            candidates.append((source_index, -parent_overlap, name, kept, list(full_genes)))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    existing: list[set[str]] = [set(selected["GLOBAL_CONTEXT"])]
    for _, _, name, genes, full_genes in candidates:
        if len(selected) >= max_pathways:
            break
        if name in selected:
            continue
        gene_set = set(genes)
        if any(len(gene_set & prior) / max(len(gene_set | prior), 1) >= 0.90 for prior in existing):
            continue
        selected[name] = genes
        selected_full[name] = full_genes
        existing.append(gene_set)
    required_pathways = 1 if max_pathways == 1 else 2
    if len(selected) < required_pathways:
        raise ValueError(
            f"pathway selection produced fewer than {required_pathways} pathways"
        )
    return PathwaySelection(members=selected, full_genes=selected_full)


def load_site_kinases(
    paths: Sequence[str | Path], targets: Sequence[str]
) -> dict[str, set[str]]:
    target_lookup = {str(target).upper(): str(target) for target in targets}
    target_set = set(target_lookup)
    mapping: dict[str, set[str]] = defaultdict(set)
    for item in paths:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"missing kinase prior: {path}")
        separator = "," if path.suffix.lower() == ".csv" else "\t"
        table = pd.read_csv(path, sep=separator, low_memory=False)
        kinase_column = next(
            (
                column
                for column in ["kinase", "kinases", "regulator", "kinase_gene"]
                if column in table.columns
            ),
            None,
        )
        target_column = next(
            (column for column in ["target", "gene_site", "gene_site_id"] if column in table.columns),
            None,
        )
        if target_column is None and "target_id" in table.columns:
            direct_overlap = table["target_id"].astype(str).str.upper().isin(target_set).any()
            if direct_overlap:
                target_column = "target_id"
        if target_column is None and "phosphosite" in table.columns:
            direct_overlap = table["phosphosite"].astype(str).str.upper().isin(target_set).any()
            if direct_overlap:
                target_column = "phosphosite"
        if target_column is None:
            gene_site_pairs = [
                ("substrate_gene", "substrate_site"),
                ("parent_gene", "site"),
                ("gene", "site_canonical"),
            ]
            pair = next(
                (
                    (gene, site)
                    for gene, site in gene_site_pairs
                    if {gene, site} <= set(table.columns)
                ),
                None,
            )
            if pair is not None:
                gene_column, site_column = pair
                table = table.copy()
                table["__target"] = (
                    table[gene_column].astype(str).str.upper().str.strip()
                    + "|"
                    + table[site_column].astype(str).str.upper().str.strip()
                )
                target_column = "__target"
        if target_column is None and "target_index" in table.columns:
            target_by_index = {index: target for index, target in enumerate(targets)}
            table = table.copy()
            table["__target"] = pd.to_numeric(table["target_index"], errors="coerce").map(target_by_index)
            target_column = "__target"
        if kinase_column is None or target_column is None:
            raise ValueError(f"kinase prior has unsupported columns: {path}")
        for kinase, target in table[[kinase_column, target_column]].dropna().itertuples(index=False):
            target_key = str(target).upper().strip()
            if target_key in target_set:
                canonical_target = target_lookup[target_key]
                for kinase_name in str(kinase).replace(";", ",").split(","):
                    kinase_name = kinase_name.upper().strip()
                    if kinase_name:
                        mapping[canonical_target].add(kinase_name)
    return mapping


def _padded_indices(
    members: Sequence[Sequence[str]],
    vocabulary: Mapping[str, int],
    max_members: int,
) -> tuple[np.ndarray, np.ndarray]:
    index = np.zeros((len(members), max_members), dtype=np.int64)
    mask = np.zeros((len(members), max_members), dtype=bool)
    for row, names in enumerate(members):
        mapped = [vocabulary[name] for name in names if name in vocabulary][:max_members]
        if mapped:
            index[row, : len(mapped)] = mapped
            mask[row, : len(mapped)] = True
    return index, mask


def build_biological_prior(
    pathways: Mapping[str, Sequence[str]],
    pathway_full_genes: Mapping[str, Sequence[str]],
    rna_genes: Sequence[str],
    protein_genes: Sequence[str],
    targets: Sequence[str],
    parent_genes: Sequence[str],
    site_kinases: Mapping[str, set[str]],
    site_observation_count: np.ndarray,
    n_training_samples: int,
    *,
    max_rna_members: int = 128,
    max_protein_members: int = 128,
    max_site_pathways: int = 8,
    max_site_kinases: int = 8,
) -> BiologicalPrior:
    names = list(pathways)
    pathway_genes = [[str(gene).upper() for gene in pathways[name]] for name in names]
    full_genes = [[str(gene).upper() for gene in pathway_full_genes[name]] for name in names]
    rna_vocabulary = {str(gene).upper(): index for index, gene in enumerate(rna_genes)}
    protein_vocabulary = {str(gene).upper(): index for index, gene in enumerate(protein_genes)}
    rna_index, rna_mask = _padded_indices(pathway_genes, rna_vocabulary, max_rna_members)
    protein_index, protein_mask = _padded_indices(
        full_genes, protein_vocabulary, max_protein_members
    )

    gene_to_pathways: dict[str, list[int]] = defaultdict(list)
    for pathway_index, genes in enumerate(full_genes):
        for gene in genes:
            gene_to_pathways[gene].append(pathway_index)
    site_pathway_index = np.zeros((len(targets), max_site_pathways), dtype=np.int64)
    site_pathway_mask = np.zeros((len(targets), max_site_pathways), dtype=bool)
    site_pathway_weight = np.zeros((len(targets), max_site_pathways), dtype=np.float32)
    for site, parent in enumerate(parent_genes):
        related = [index for index in gene_to_pathways.get(str(parent).upper(), []) if index != 0]
        related = related[:max_site_pathways] or [0]
        site_pathway_index[site, : len(related)] = related
        site_pathway_mask[site, : len(related)] = True
        site_pathway_weight[site, : len(related)] = 1.0 / len(related)

    relation = np.eye(len(names), dtype=bool)
    relation[0, :] = True
    relation[:, 0] = True
    sets = [set(genes) for genes in full_genes]
    for left in range(1, len(names)):
        for right in range(left + 1, len(names)):
            overlap = len(sets[left].intersection(sets[right]))
            threshold = max(2, int(math.ceil(min(len(sets[left]), len(sets[right])) * 0.05)))
            if overlap >= threshold:
                relation[left, right] = relation[right, left] = True

    parent_index = np.asarray(
        [protein_vocabulary.get(str(parent).upper(), 0) for parent in parent_genes],
        dtype=np.int64,
    )
    parent_mask = np.asarray(
        [str(parent).upper() in protein_vocabulary for parent in parent_genes], dtype=bool
    )

    kinase_names = sorted({kinase for values in site_kinases.values() for kinase in values})
    kinase_vocabulary = {name: index + 1 for index, name in enumerate(kinase_names)}
    kinase_index = np.zeros((len(targets), max_site_kinases), dtype=np.int64)
    kinase_mask = np.zeros((len(targets), max_site_kinases), dtype=bool)
    for site, target in enumerate(targets):
        mapped = [kinase_vocabulary[name] for name in sorted(site_kinases.get(str(target), set()))]
        mapped = mapped[:max_site_kinases]
        if mapped:
            kinase_index[site, : len(mapped)] = mapped
            kinase_mask[site, : len(mapped)] = True

    coverage = np.asarray(site_observation_count, dtype=np.float32) / max(n_training_samples, 1)
    return BiologicalPrior(
        pathway_names=names,
        pathway_genes=pathway_genes,
        pathway_full_genes=full_genes,
        rna_pathway_index=rna_index,
        rna_pathway_mask=rna_mask,
        protein_pathway_index=protein_index,
        protein_pathway_mask=protein_mask,
        pathway_relation_mask=relation,
        site_pathway_index=site_pathway_index,
        site_pathway_mask=site_pathway_mask,
        site_pathway_weight=site_pathway_weight,
        parent_protein_index=parent_index,
        parent_protein_mask=parent_mask,
        site_kinase_index=kinase_index,
        site_kinase_mask=kinase_mask,
        kinase_names=kinase_names,
        site_coverage=coverage,
    )


def sample_rank_encode(values: np.ndarray, chunk_size: int = 64) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    output = np.empty_like(array)
    for start in range(0, array.shape[0], chunk_size):
        end = min(array.shape[0], start + chunk_size)
        block = array[start:end]
        order = np.argsort(np.argsort(np.nan_to_num(block, nan=-np.inf), axis=1), axis=1)
        output[start:end] = order / max(array.shape[1] - 1, 1) * 2.0 - 1.0
    return output.astype(np.float32)


def fit_feature_zscore(values: np.ndarray, train_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(values, dtype=np.float32)[np.asarray(train_index, dtype=np.int64)]
    mean = np.nanmean(train, axis=0)
    scale = np.nanstd(train, axis=0)
    mean = np.nan_to_num(mean, nan=0.0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-6), scale, 1.0).astype(np.float32)
    return mean, scale


def apply_feature_zscore(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    output = (np.asarray(values, dtype=np.float32) - mean) / scale
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).clip(-8, 8).astype(np.float32)


def fit_study_site_standardization(
    phosphosite: np.ndarray,
    studies: Sequence[str],
    train_index: np.ndarray,
    *,
    minimum_observations: int = 2,
) -> StudySiteStandardization:
    """Fit every study-site location and scale using training rows only.

    A study-site pair with insufficient training observations is marked
    unavailable. No pooled or validation-derived fallback is permitted.
    """
    values = np.asarray(phosphosite, dtype=np.float32)
    studies = np.asarray(studies, dtype=str)
    train_index = np.asarray(train_index, dtype=np.int64)
    if values.ndim != 2 or len(studies) != values.shape[0]:
        raise ValueError("phosphosite matrix and study identifiers have incompatible shapes")
    if minimum_observations < 1:
        raise ValueError("minimum_observations must be positive")
    study_names = sorted(set(studies[train_index].tolist()))
    if not study_names:
        raise ValueError("training fold contains no study identifiers")
    n_sites = values.shape[1]
    mean = np.zeros((len(study_names), n_sites), dtype=np.float32)
    scale = np.ones((len(study_names), n_sites), dtype=np.float32)
    count = np.zeros((len(study_names), n_sites), dtype=np.int64)
    available = np.zeros((len(study_names), n_sites), dtype=bool)
    for row, study in enumerate(study_names):
        selected = train_index[studies[train_index] == study]
        block = values[selected]
        count[row] = np.isfinite(block).sum(axis=0)
        finite = np.isfinite(block)
        block_sum = np.where(finite, block, 0.0).sum(axis=0, dtype=np.float64)
        block_mean = np.divide(
            block_sum,
            count[row],
            out=np.full(n_sites, np.nan, dtype=np.float64),
            where=count[row] > 0,
        )
        squared = np.where(finite, (block - block_mean[None, :]) ** 2, 0.0).sum(
            axis=0, dtype=np.float64
        )
        block_scale = np.sqrt(
            np.divide(
                squared,
                count[row],
                out=np.full(n_sites, np.nan, dtype=np.float64),
                where=count[row] > 0,
            )
        )
        supported = (count[row] >= int(minimum_observations)) & np.isfinite(block_mean)
        mean[row, supported] = block_mean[supported].astype(np.float32)
        valid_scale = supported & np.isfinite(block_scale) & (block_scale > 1.0e-6)
        if bool(valid_scale.any()):
            scale_floor = max(
                float(np.quantile(block_scale[valid_scale], 0.10)) * 0.10,
                1.0e-3,
            )
            scale[row, valid_scale] = np.maximum(
                block_scale[valid_scale], scale_floor
            ).astype(np.float32)
        available[row] = supported
    return StudySiteStandardization(
        study_names=study_names,
        mean=mean,
        scale=scale,
        count=count,
        available=available,
    )


def apply_study_site_standardization(
    phosphosite: np.ndarray,
    studies: Sequence[str],
    parameters: StudySiteStandardization,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply locked training parameters and return values plus support mask."""
    values = np.asarray(phosphosite, dtype=np.float32)
    studies = np.asarray(studies, dtype=str)
    if values.ndim != 2 or len(studies) != values.shape[0]:
        raise ValueError("phosphosite matrix and study identifiers have incompatible shapes")
    if values.shape[1] != parameters.mean.shape[1]:
        raise ValueError("phosphosite site count differs from the standardization contract")
    lookup = {study: index for index, study in enumerate(parameters.study_names)}
    unknown = sorted(set(studies.tolist()) - set(lookup))
    if unknown:
        raise ValueError(
            "study identifiers lack training-fold standardization parameters: "
            f"{unknown[:5]}"
        )
    output = np.full(values.shape, np.nan, dtype=np.float32)
    support = np.zeros(values.shape, dtype=bool)
    for study, parameter_index in lookup.items():
        rows = np.flatnonzero(studies == study)
        if not len(rows):
            continue
        observed = np.isfinite(values[rows])
        allowed = observed & parameters.available[parameter_index][None, :]
        transformed = (
            values[rows] - parameters.mean[parameter_index][None, :]
        ) / parameters.scale[parameter_index][None, :]
        output[rows] = np.where(allowed, transformed, np.nan)
        support[rows] = allowed
    return output, support


def parent_anchor_statistics(
    protein_prediction: np.ndarray,
    standardized_phosphosite: np.ndarray,
    parent_index: np.ndarray,
    parent_mask: np.ndarray,
    train_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return training-only absolute correlation, paired coverage and count."""
    n_sites = standardized_phosphosite.shape[1]
    quality = np.zeros(n_sites, dtype=np.float32)
    coverage = np.zeros(n_sites, dtype=np.float32)
    count = np.zeros(n_sites, dtype=np.int64)
    train_index = np.asarray(train_index, dtype=np.int64)
    for site in range(n_sites):
        if not bool(parent_mask[site]):
            continue
        x = protein_prediction[train_index, int(parent_index[site])]
        y = standardized_phosphosite[train_index, site]
        valid = np.isfinite(x) & np.isfinite(y)
        count[site] = int(valid.sum())
        coverage[site] = count[site] / max(len(train_index), 1)
        if count[site] >= 3:
            x_valid = x[valid].astype(np.float64)
            y_valid = y[valid].astype(np.float64)
            if np.std(x_valid) > 1.0e-8 and np.std(y_valid) > 1.0e-8:
                quality[site] = float(abs(np.corrcoef(x_valid, y_valid)[0, 1]))
    return quality, coverage, count


def fit_parent_calibration(
    protein_prediction: np.ndarray,
    phosphosite: np.ndarray,
    parent_index: np.ndarray,
    parent_mask: np.ndarray,
    train_index: np.ndarray,
    *,
    ridge: float = 1.0,
    minimum_observations: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_sites = phosphosite.shape[1]
    intercept = np.zeros(n_sites, dtype=np.float32)
    slope = np.zeros(n_sites, dtype=np.float32)
    count = np.zeros(n_sites, dtype=np.int64)
    train_index = np.asarray(train_index, dtype=np.int64)
    for site in range(n_sites):
        y = phosphosite[train_index, site]
        if parent_mask[site]:
            x = protein_prediction[train_index, parent_index[site]]
            valid = np.isfinite(x) & np.isfinite(y)
        else:
            x = np.zeros_like(y)
            valid = np.isfinite(y)
        count[site] = int(valid.sum())
        if not valid.any():
            continue
        y_valid = y[valid].astype(np.float64)
        y_mean = float(y_valid.mean())
        if parent_mask[site] and count[site] >= minimum_observations:
            x_valid = x[valid].astype(np.float64)
            x_mean = float(x_valid.mean())
            x_centered = x_valid - x_mean
            beta = float(np.dot(x_centered, y_valid - y_mean) / (np.dot(x_centered, x_centered) + ridge))
            slope[site] = beta
            intercept[site] = y_mean - beta * x_mean
        else:
            intercept[site] = y_mean
    return intercept, slope, count


def parent_baseline(
    protein_prediction: np.ndarray,
    intercept: np.ndarray,
    slope: np.ndarray,
    parent_index: np.ndarray,
    parent_mask: np.ndarray,
) -> np.ndarray:
    parent = protein_prediction[:, np.maximum(parent_index, 0)]
    parent = parent * parent_mask[None, :]
    return (intercept[None, :] + slope[None, :] * parent).astype(np.float32)


def masked_row_median_center(
    values: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("values and mask shapes differ")
    offsets = np.nanmedian(np.where(mask, values, np.nan), axis=1)
    offsets = np.nan_to_num(offsets, nan=0.0).astype(np.float32)
    centered = values - offsets[:, None]
    return np.where(mask, centered, 0.0).astype(np.float32), offsets


def build_query_reference_knn(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
    *,
    query_ids: Sequence[str] | None = None,
    reference_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.nan_to_num(np.asarray(query, dtype=np.float32), nan=0.0)
    reference = np.nan_to_num(np.asarray(reference, dtype=np.float32), nan=0.0)
    query = query / np.linalg.norm(query, axis=1, keepdims=True).clip(min=1.0e-6)
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True).clip(min=1.0e-6)
    similarity = query @ reference.T
    excluded_per_row = np.zeros(similarity.shape[0], dtype=np.int64)
    if query_ids is not None and reference_ids is not None:
        reference_lookup = {str(sample): index for index, sample in enumerate(reference_ids)}
        for row, sample in enumerate(query_ids):
            match = reference_lookup.get(str(sample))
            if match is not None:
                similarity[row, match] = -np.inf
                excluded_per_row[row] = 1
    available = similarity.shape[1] - excluded_per_row
    if int(available.min()) < 1:
        raise ValueError("no reference neighbour remains after self exclusion")
    kk = min(max(int(k), 1), int(available.min()))
    index = np.argpartition(-similarity, kth=kk - 1, axis=1)[:, :kk]
    values = np.take_along_axis(similarity, index, axis=1)
    order = np.argsort(-values, axis=1)
    index = np.take_along_axis(index, order, axis=1)
    values = np.take_along_axis(values, order, axis=1)
    if not np.isfinite(values).all():
        raise RuntimeError("self-excluded neighbour selection returned a forbidden reference")
    return index.astype(np.int64), values.astype(np.float32)


def per_site_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    targets: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for site, name in enumerate(targets):
        valid = mask[:, site] & np.isfinite(target[:, site]) & np.isfinite(prediction[:, site])
        if valid.sum() >= 3:
            observed = pd.Series(target[valid, site])
            predicted = pd.Series(prediction[valid, site])
            spearman = observed.corr(predicted, method="spearman")
            pearson = observed.corr(predicted, method="pearson")
            mse = float(np.mean((observed.to_numpy() - predicted.to_numpy()) ** 2))
            sd_ratio = float(predicted.std(ddof=0) / max(observed.std(ddof=0), 1.0e-8))
        else:
            spearman = pearson = mse = sd_ratio = np.nan
        rows.append(
            {
                "target": str(name),
                "n": int(valid.sum()),
                "spearman": spearman,
                "pearson": pearson,
                "mse": mse,
                "prediction_to_target_sd_ratio": sd_ratio,
            }
        )
    return pd.DataFrame(rows)


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def prior_metadata(prior: BiologicalPrior) -> dict[str, object]:
    payload = asdict(prior)
    return {
        "n_pathways": len(prior.pathway_names),
        "n_kinases": len(prior.kinase_names),
        "pathway_names": prior.pathway_names,
        "kinase_names": prior.kinase_names,
        "mapped_parent_sites": int(prior.parent_protein_mask.sum()),
        "mapped_kinase_sites": int(prior.site_kinase_mask.any(axis=1).sum()),
        "mapped_specific_pathway_sites": int(
            ((prior.site_pathway_index != 0) & prior.site_pathway_mask).any(axis=1).sum()
        ),
        "tensor_shapes": {
            key: list(value.shape)
            for key, value in payload.items()
            if isinstance(value, np.ndarray)
        },
    }
