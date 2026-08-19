"""Training-only per-site ridge floor for the chunked phosphosite model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


@dataclass
class SiteRidgeFloor:
    prediction: np.ndarray
    intercept: np.ndarray
    pca_coefficient: np.ndarray
    parent_rna_coefficient: np.ndarray
    parent_protein_coefficient: np.ndarray
    parent_reliability_coefficient: np.ndarray
    kinase_rna_coefficient: np.ndarray
    kinase_protein_coefficient: np.ndarray
    kinase_reliability_coefficient: np.ndarray
    rna_impute_mean: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    pca_explained_variance_ratio: np.ndarray
    fitted: np.ndarray

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            prediction=self.prediction,
            intercept=self.intercept,
            pca_coefficient=self.pca_coefficient,
            parent_rna_coefficient=self.parent_rna_coefficient,
            parent_protein_coefficient=self.parent_protein_coefficient,
            parent_reliability_coefficient=self.parent_reliability_coefficient,
            kinase_rna_coefficient=self.kinase_rna_coefficient,
            kinase_protein_coefficient=self.kinase_protein_coefficient,
            kinase_reliability_coefficient=self.kinase_reliability_coefficient,
            rna_impute_mean=self.rna_impute_mean,
            pca_mean=self.pca_mean,
            pca_components=self.pca_components,
            pca_explained_variance_ratio=self.pca_explained_variance_ratio,
            fitted=self.fitted,
        )


def _training_impute(values: np.ndarray, train_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values[train_index], axis=0).astype(np.float32)
    mean[~np.isfinite(mean)] = 0.0
    filled = np.where(np.isfinite(values), values, mean[None, :]).astype(np.float32)
    return filled, mean


def _site_kinase_lists(edge_index: np.ndarray, n_sites: int) -> list[np.ndarray]:
    mapping: list[list[int]] = [[] for _ in range(n_sites)]
    edge = np.asarray(edge_index, dtype=np.int64)
    for kinase, site in zip(edge[0].tolist(), edge[1].tolist()):
        mapping[site].append(kinase)
    return [np.asarray(sorted(set(values)), dtype=np.int64) for values in mapping]


def fit_site_ridge_floor(
    *,
    rna_rank: np.ndarray,
    protein_value: np.ndarray,
    protein_reliability: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    train_index: np.ndarray,
    parent_rna_index: np.ndarray,
    parent_rna_mask: np.ndarray,
    parent_protein_index: np.ndarray,
    parent_protein_mask: np.ndarray,
    kinase_rna_index: np.ndarray,
    kinase_rna_mask: np.ndarray,
    kinase_protein_index: np.ndarray,
    kinase_protein_mask: np.ndarray,
    kinase_site_edge_index: np.ndarray,
    n_components: int = 64,
    alpha: float = 10.0,
    minimum_observations: int = 32,
    n_jobs: int = 16,
    random_state: int = 20260722,
) -> SiteRidgeFloor:
    """Fit independent ridge models without reading validation targets."""
    rna, rna_impute_mean = _training_impute(
        np.asarray(rna_rank, dtype=np.float32), np.asarray(train_index, dtype=np.int64)
    )
    protein, _ = _training_impute(
        np.asarray(protein_value, dtype=np.float32), np.asarray(train_index, dtype=np.int64)
    )
    reliability = np.asarray(protein_reliability, dtype=np.float32)
    pca = PCA(
        n_components=int(n_components),
        svd_solver="randomized",
        random_state=int(random_state),
    )
    pca.fit(rna[train_index])
    pcs = pca.transform(rna).astype(np.float32)
    n_samples, n_sites = target.shape
    n_kinases = len(kinase_rna_index)
    site_kinases = _site_kinase_lists(kinase_site_edge_index, n_sites)

    def fit_one(site: int):
        pieces = [pcs]
        layout: list[tuple[str, np.ndarray | None]] = [("pca", None)]
        if parent_rna_mask[site]:
            pieces.append(rna[:, parent_rna_index[site] : parent_rna_index[site] + 1])
            layout.append(("parent_rna", None))
        if parent_protein_mask[site]:
            parent = int(parent_protein_index[site])
            pieces.extend(
                [
                    protein[:, parent : parent + 1],
                    reliability[:, parent : parent + 1],
                ]
            )
            layout.extend([("parent_protein", None), ("parent_reliability", None)])
        kinases = site_kinases[site]
        rna_kinases = kinases[kinase_rna_mask[kinases]]
        protein_kinases = kinases[kinase_protein_mask[kinases]]
        if rna_kinases.size:
            pieces.append(rna[:, kinase_rna_index[rna_kinases]])
            layout.append(("kinase_rna", rna_kinases))
        if protein_kinases.size:
            protein_columns = kinase_protein_index[protein_kinases]
            pieces.extend(
                [
                    protein[:, protein_columns],
                    reliability[:, protein_columns],
                ]
            )
            layout.extend(
                [
                    ("kinase_protein", protein_kinases),
                    ("kinase_reliability", protein_kinases),
                ]
            )
        design = np.concatenate(pieces, axis=1).astype(np.float64)
        valid = observed[train_index, site] & np.isfinite(target[train_index, site])
        if int(valid.sum()) < int(minimum_observations):
            return site, None
        fit_x = design[train_index][valid]
        fit_y = target[train_index, site][valid].astype(np.float64)
        mean = fit_x.mean(axis=0)
        scale = fit_x.std(axis=0)
        scale[~np.isfinite(scale) | (scale < 1.0e-8)] = 1.0
        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit((fit_x - mean) / scale, fit_y)
        coefficient = np.asarray(model.coef_, dtype=np.float64) / scale
        intercept = float(model.intercept_ - np.dot(mean, coefficient))
        prediction = (design @ coefficient + intercept).astype(np.float32)
        return site, (intercept, coefficient.astype(np.float32), layout, prediction)

    results = Parallel(n_jobs=int(n_jobs), prefer="threads")(
        delayed(fit_one)(site) for site in range(n_sites)
    )
    prediction = np.zeros((n_samples, n_sites), dtype=np.float32)
    intercept = np.zeros(n_sites, dtype=np.float32)
    pca_coefficient = np.zeros((n_sites, n_components), dtype=np.float32)
    parent_rna_coefficient = np.zeros(n_sites, dtype=np.float32)
    parent_protein_coefficient = np.zeros(n_sites, dtype=np.float32)
    parent_reliability_coefficient = np.zeros(n_sites, dtype=np.float32)
    kinase_rna_coefficient = np.zeros((n_sites, n_kinases), dtype=np.float32)
    kinase_protein_coefficient = np.zeros((n_sites, n_kinases), dtype=np.float32)
    kinase_reliability_coefficient = np.zeros((n_sites, n_kinases), dtype=np.float32)
    fitted = np.zeros(n_sites, dtype=bool)
    for site, result in results:
        if result is None:
            continue
        fitted[site] = True
        intercept[site], coefficient, layout, prediction[:, site] = result
        cursor = 0
        for name, indices in layout:
            width = n_components if name == "pca" else (1 if indices is None else len(indices))
            values = coefficient[cursor : cursor + width]
            cursor += width
            if name == "pca":
                pca_coefficient[site] = values
            elif name == "parent_rna":
                parent_rna_coefficient[site] = values[0]
            elif name == "parent_protein":
                parent_protein_coefficient[site] = values[0]
            elif name == "parent_reliability":
                parent_reliability_coefficient[site] = values[0]
            elif name == "kinase_rna":
                kinase_rna_coefficient[site, indices] = values
            elif name == "kinase_protein":
                kinase_protein_coefficient[site, indices] = values
            elif name == "kinase_reliability":
                kinase_reliability_coefficient[site, indices] = values
    return SiteRidgeFloor(
        prediction=prediction,
        intercept=intercept,
        pca_coefficient=pca_coefficient,
        parent_rna_coefficient=parent_rna_coefficient,
        parent_protein_coefficient=parent_protein_coefficient,
        parent_reliability_coefficient=parent_reliability_coefficient,
        kinase_rna_coefficient=kinase_rna_coefficient,
        kinase_protein_coefficient=kinase_protein_coefficient,
        kinase_reliability_coefficient=kinase_reliability_coefficient,
        rna_impute_mean=rna_impute_mean,
        pca_mean=np.asarray(pca.mean_, dtype=np.float32),
        pca_components=np.asarray(pca.components_, dtype=np.float32),
        pca_explained_variance_ratio=np.asarray(
            pca.explained_variance_ratio_, dtype=np.float32
        ),
        fitted=fitted,
    )
