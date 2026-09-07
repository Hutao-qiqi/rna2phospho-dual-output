#!/usr/bin/env python3
"""SCP682 Main inference using the locked M2.2 latent-coordinate model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_matrix(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", index_col=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict 18,592 phosphosites with SCP682 M2.2")
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--predicted-protein", type=Path, required=True)
    parser.add_argument("--runtime-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-metadata", type=Path)
    parser.add_argument("--study-column", default="study")
    args = parser.parse_args()

    package = np.load(args.runtime_bundle, allow_pickle=False)
    if str(package["model_id"]) != "SCP682-M2.2-LR-rank128":
        raise ValueError("runtime bundle is not the locked SCP682 M2.2 model")
    rna = read_matrix(args.rna)
    protein = read_matrix(args.predicted_protein)
    if not rna.index.equals(protein.index):
        protein = protein.reindex(rna.index)
    rna_vocab = package["rna_vocabulary"].astype(str)
    protein_vocab = package["protein_vocabulary"].astype(str)
    missing_rna = np.setdiff1d(rna_vocab, rna.columns.astype(str))
    missing_protein = np.setdiff1d(protein_vocab, protein.columns.astype(str))
    if len(missing_rna) or len(missing_protein):
        raise ValueError(f"missing required inputs: RNA={len(missing_rna)}, predicted_protein={len(missing_protein)}")
    rna_array = rna.reindex(columns=rna_vocab).to_numpy(np.float32)
    protein_array = protein.reindex(columns=protein_vocab).to_numpy(np.float32)

    rz = np.nan_to_num((rna_array - package["rna_mean"]) / package["rna_scale"], nan=0.0).astype(np.float32)
    pz = np.nan_to_num((protein_array - package["protein_mean"]) / package["protein_scale"], nan=0.0).astype(np.float32)
    feature = np.concatenate([rz @ package["rna_components"].T, pz @ package["protein_components"].T], axis=1)
    feature = ((feature - package["feature_mean"]) / package["feature_scale"]).astype(np.float32)
    coordinate = feature @ package["ridge_coef"].T + package["ridge_intercept"]

    rescue_rna = np.nan_to_num(
        (rna_array - package["rescue_rna_mean"]) / package["rescue_rna_scale"], nan=0.0
    ).astype(np.float32)
    rescue_protein = np.nan_to_num(
        (protein_array - package["rescue_protein_mean"]) / package["rescue_protein_scale"], nan=0.0
    ).astype(np.float32)
    for row, latent_id in enumerate(package["rescue_latent"].astype(int)):
        x = np.concatenate(
            [feature, rescue_rna[:, package["rescue_rna_index"][row]], rescue_protein[:, package["rescue_protein_index"][row]]],
            axis=1,
        )
        standardized = x @ package["rescue_coef"][row] + package["rescue_intercept"][row]
        coordinate[:, latent_id] = standardized * package["coordinate_scale"][latent_id] + package["coordinate_mean"][latent_id]

    residual = coordinate @ package["site_basis"] + package["site_mean"][None, :]
    parent = package["parent_index"].astype(int)
    if args.study_metadata:
        metadata = pd.read_csv(args.study_metadata, sep="\t", index_col=0).reindex(rna.index)
        labels = metadata[args.study_column].astype(str).to_numpy()
        protein_center = np.empty_like(protein_array)
        for study in np.unique(labels):
            rows = labels == study
            local = np.nanmean(protein_array[rows], axis=0)
            protein_center[rows] = np.where(np.isfinite(local), local, package["global_protein_mean"])
    else:
        protein_center = np.broadcast_to(package["global_protein_mean"], protein_array.shape)
    parent_component = np.nan_to_num(protein_array[:, parent] - protein_center[:, parent], nan=0.0) * package["parent_beta"][None, :]
    prediction = package["global_site_mean"][None, :] + parent_component + residual

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sites = package["phosphosite_vocabulary"].astype(str)
    if args.output.suffix.lower() in {".parquet", ".pq"}:
        pd.DataFrame(prediction, index=rna.index, columns=sites).to_parquet(args.output)
    else:
        sample_ids = np.asarray(rna.index.astype(str).tolist(), dtype="U")
        np.savez_compressed(args.output, sample_id=sample_ids, phosphosite=sites, prediction=prediction.astype(np.float32))
    report = {"status": "complete", "model_id": str(package["model_id"]), "patients": len(rna), "phosphosites": len(sites), "output": str(args.output)}
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
