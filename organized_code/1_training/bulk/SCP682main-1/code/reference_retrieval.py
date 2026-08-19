"""Leakage-safe training-label retrieval with module-level fallback summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REFERENCE_COUNT_BINS = ((0, 3), (4, 7), (8, 15), (16, 32))


@dataclass(frozen=True)
class ReferenceBatch:
    features: np.ndarray  # [query, site, reference-token, 6]
    mask: np.ndarray  # final position is valid only when module evidence exists
    site_reference_count: np.ndarray
    site_reference_indices: np.ndarray
    last_selected_similarity: np.ndarray
    similarity_at_rank_32: np.ndarray
    effective_sample_size: np.ndarray


def cosine_candidates(
    query_embedding: np.ndarray,
    training_embedding: np.ndarray,
    *,
    candidate_count: int = 256,
    query_training_rows: np.ndarray | None = None,
    query_case_ids: np.ndarray | None = None,
    training_case_ids: np.ndarray | None = None,
    query_cancer_ids: np.ndarray | None = None,
    training_cancer_ids: np.ndarray | None = None,
    prefer_same_cancer: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return training-only candidates while excluding self and same-case rows."""
    if candidate_count not in {128, 256, 916}:
        raise ValueError("candidate_count must be one of 128, 256, or 916")
    q = np.asarray(query_embedding, dtype=np.float32)
    r = np.asarray(training_embedding, dtype=np.float32)
    q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(min=1.0e-8)
    r = r / np.linalg.norm(r, axis=1, keepdims=True).clip(min=1.0e-8)
    similarity = q @ r.T
    if query_training_rows is not None:
        rows = np.asarray(query_training_rows, dtype=np.int64)
        valid = rows >= 0
        similarity[np.flatnonzero(valid), rows[valid]] = -np.inf
    if query_case_ids is not None or training_case_ids is not None:
        if query_case_ids is None or training_case_ids is None:
            raise ValueError("both query and training case identifiers are required")
        qcase = np.asarray(query_case_ids).astype(str)
        rcase = np.asarray(training_case_ids).astype(str)
        similarity[qcase[:, None] == rcase[None, :]] = -np.inf
    available = np.isfinite(similarity).sum(axis=1)
    if candidate_count < 916 and np.any(available < candidate_count):
        raise ValueError("too few leakage-free training references")
    width = candidate_count
    index = np.full((q.shape[0], width), -1, dtype=np.int64)
    score = np.full((q.shape[0], width), -np.inf, dtype=np.float32)
    for row in range(q.shape[0]):
        valid = np.flatnonzero(np.isfinite(similarity[row]))
        ordered = valid[np.argsort(-similarity[row, valid], kind="stable")]
        if prefer_same_cancer:
            if query_cancer_ids is None or training_cancer_ids is None:
                raise ValueError("cancer-aware retrieval requires query and training cancers")
            same = np.asarray(training_cancer_ids).astype(str)[ordered] == str(
                np.asarray(query_cancer_ids).astype(str)[row]
            )
            ordered = np.concatenate([ordered[same], ordered[~same]])
        selected = ordered[: min(width, ordered.size)]
        index[row, : selected.size] = selected
        score[row, : selected.size] = similarity[row, selected]
    return index, score


def build_module_summaries(
    residual: np.ndarray,
    observed: np.ndarray,
    parent_index: np.ndarray,
    kinase_site_edge_index: np.ndarray,
    parent_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed-only leave-one-site-out module means for training references.

    Parent and kinase modules are combined with their original multiplicity, but
    the queried site's own residual is removed from every contributing module.
    This prevents the low-reference fallback from recovering the direct label it
    is required to suppress.
    """
    residual = np.asarray(residual, dtype=np.float32)
    observed = np.asarray(observed, dtype=bool)
    parent = np.asarray(parent_index, dtype=np.int64)
    mapped_parent = (
        np.ones(parent.size, dtype=bool)
        if parent_mask is None else np.asarray(parent_mask, dtype=bool)
    )
    if residual.shape != observed.shape or residual.shape[1] != parent.size:
        raise ValueError("module summary inputs differ from the site vocabulary")
    if mapped_parent.shape != parent.shape:
        raise ValueError("parent mask differs from the site vocabulary")
    n_samples, n_sites = residual.shape
    total = np.zeros_like(residual, dtype=np.float32)
    count = np.zeros_like(residual, dtype=np.int32)
    for group in np.unique(parent[mapped_parent]):
        sites = np.flatnonzero(mapped_parent & (parent == group))
        module_mask = observed[:, sites]
        module_sum = (residual[:, sites] * module_mask).sum(1)
        module_count = module_mask.sum(1, dtype=np.int32)
        total[:, sites] += module_sum[:, None]
        count[:, sites] += module_count[:, None]
    kinase_members: dict[int, np.ndarray] = {}
    edge = np.asarray(kinase_site_edge_index, dtype=np.int64)
    if edge.size:
        for kinase in np.unique(edge[0]):
            kinase_members[int(kinase)] = np.unique(edge[1, edge[0] == kinase])
    site_kinase_count = np.zeros(n_sites, dtype=np.int32)
    for kinase, members in kinase_members.items():
        module_mask = observed[:, members]
        module_sum = (residual[:, members] * module_mask).sum(1)
        module_count = module_mask.sum(1, dtype=np.int32)
        total[:, members] += module_sum[:, None]
        count[:, members] += module_count[:, None]
        site_kinase_count[members] += 1

    # The target occurs once in each module to which it belongs.  Subtract all
    # occurrences in one vectorized operation after module-wise aggregation.
    self_multiplicity = mapped_parent.astype(np.int32) + site_kinase_count
    total -= residual * observed * self_multiplicity[None, :]
    count -= observed.astype(np.int32) * self_multiplicity[None, :]
    if np.any(count < 0):
        raise RuntimeError("leave-one-site-out module support became negative")
    summary = np.divide(
        total, np.maximum(count, 1), out=np.zeros_like(total), where=count > 0
    ).astype(np.float32)
    support = np.minimum(count, np.iinfo(np.int16).max).astype(np.int16)
    return summary, support


def select_references_for_chunk(
    candidate_index: np.ndarray,
    candidate_similarity: np.ndarray,
    reference_residual: np.ndarray,
    reference_ridge: np.ndarray,
    reference_observed: np.ndarray,
    module_summary: np.ndarray,
    module_support: np.ndarray,
    site_index: np.ndarray,
    *,
    maximum_site_references: int = 32,
) -> ReferenceBatch:
    """Select observed per-site labels and append a module fallback token."""
    candidate_index = np.asarray(candidate_index, dtype=np.int64)
    candidate_similarity = np.asarray(candidate_similarity, dtype=np.float32)
    sites = np.asarray(site_index, dtype=np.int64)
    batch, candidates = candidate_index.shape
    width = maximum_site_references + 1
    features = np.zeros((batch, sites.size, width, 6), dtype=np.float32)
    mask = np.zeros((batch, sites.size, width), dtype=bool)
    selected = np.full((batch, sites.size, maximum_site_references), -1, dtype=np.int64)
    candidate_valid = candidate_index >= 0
    safe_candidate = np.maximum(candidate_index, 0)
    # Candidate order already follows RNA similarity.  A stable partition moves
    # observed references to the front without a Python loop over sites.
    observed_candidate = reference_observed[safe_candidate[:, :, None], sites[None, None, :]]
    observed_candidate &= candidate_valid[:, :, None]
    observed_candidate = observed_candidate.transpose(0, 2, 1)  # [query, site, candidate]
    order = np.argsort(~observed_candidate, axis=2, kind="stable")[:, :, :maximum_site_references]
    selected_valid = np.take_along_axis(observed_candidate, order, axis=2)
    candidate_rows = np.broadcast_to(safe_candidate[:, None, :], observed_candidate.shape)
    selected_rows = np.take_along_axis(candidate_rows, order, axis=2)
    selected[:] = np.where(selected_valid, selected_rows, -1)
    count = selected_valid.sum(axis=2).astype(np.int16)
    safe_rows = np.maximum(selected_rows, 0)
    site_grid = sites[None, :, None]
    score_grid = np.broadcast_to(candidate_similarity[:, None, :], observed_candidate.shape)
    selected_scores = np.take_along_axis(score_grid, order, axis=2)
    features[..., :-1, 0] = np.where(
        selected_valid, reference_residual[safe_rows, site_grid], 0
    )
    features[..., :-1, 1] = np.where(
        selected_valid,
        reference_residual[safe_rows, site_grid] - reference_ridge[safe_rows, site_grid],
        0,
    )
    features[..., :-1, 2] = np.where(selected_valid, selected_scores, 0)
    features[..., :-1, 3] = np.where(
        selected_valid, module_summary[safe_rows, site_grid], 0
    )
    features[..., :-1, 4] = np.where(
        selected_valid, np.log1p(module_support[safe_rows, site_grid]), 0
    )
    features[..., :-1, 5] = selected_valid
    mask[..., :-1] = selected_valid

    last_similarity = np.full((batch, sites.size), np.nan, dtype=np.float32)
    rank32_similarity = np.full_like(last_similarity, np.nan)
    selected_count = selected_valid.sum(axis=2)
    has_any = selected_count > 0
    last_position = np.maximum(selected_count - 1, 0)[..., None]
    gathered_last = np.take_along_axis(selected_scores, last_position, axis=2).squeeze(2)
    last_similarity[has_any] = gathered_last[has_any]
    has_32 = selected_count >= maximum_site_references
    rank32_similarity[has_32] = selected_scores[..., maximum_site_references - 1][has_32]
    # Similarity weights are positive and normalized only across observed labels.
    maximum_score = np.where(has_any, selected_scores[..., 0], 0.0)
    shifted = selected_scores - maximum_score[..., None]
    weights = np.where(selected_valid, np.exp(np.clip(shifted / 0.1, -50, 0)), 0).astype(np.float32)
    weight_square_sum = np.square(weights).sum(axis=2)
    effective_n = np.divide(
        np.square(weights.sum(axis=2)), weight_square_sum,
        out=np.zeros((batch, sites.size), dtype=np.float32),
        where=weight_square_sum > 0,
    )

    module_value = module_summary[safe_candidate[:, :, None], sites[None, None, :]].transpose(0, 2, 1)
    module_n = module_support[safe_candidate[:, :, None], sites[None, None, :]].transpose(0, 2, 1)
    module_valid = (module_n > 0) & candidate_valid[:, None, :]
    module_weights = np.maximum(candidate_similarity[:, None, :] + 1.0, 0.0) * module_valid
    denominator = module_weights.sum(axis=2)
    finite_score = np.where(candidate_valid, candidate_similarity, np.nan)
    features[..., -1, 2] = np.nanmean(finite_score, axis=1)[:, None]
    features[..., -1, 3] = np.divide(
        (module_weights * module_value).sum(axis=2), denominator,
        out=np.zeros_like(denominator, dtype=np.float32), where=denominator > 0,
    )
    features[..., -1, 4] = np.log1p((module_n * module_valid).sum(axis=2))
    # Queries without any observed module member must not receive a learned
    # constant from a projected all-zero token.
    mask[..., -1] = denominator > 0
    return ReferenceBatch(
        features, mask, count, selected, last_similarity, rank32_similarity, effective_n
    )


def reference_count_group(count: np.ndarray) -> np.ndarray:
    count = np.asarray(count)
    output = np.full(count.shape, "outside", dtype=object)
    for low, high in REFERENCE_COUNT_BINS:
        output[(count >= low) & (count <= high)] = f"{low}-{high}"
    return output
