from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-cells", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.contract_dir / "gse300551_cell_manifest.tsv", sep="\t").head(
        args.n_cells
    )
    source_h5ad = (
        args.project_root
        / r"01_data\single_cell\intermediate\foundation_model_h5ad_inputs_v1\gse300551_iccite_plex_kinase_2025.h5ad"
    )
    input_dir = (
        args.project_root
        / r"01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_gse300551_signal_seq_multidomain_v1"
    )
    rna = ad.read_h5ad(source_h5ad)
    lookup = {str(cell): index for index, cell in enumerate(rna.obs_names.astype(str))}
    rows = [lookup[str(cell)] for cell in manifest["cell_id"].astype(str)]
    smoke_h5ad = args.output_dir / "smoke_rna.h5ad"
    rna[rows].copy().write_h5ad(smoke_h5ad)
    embedding = np.load(input_dir / "embeddings.npy", mmap_mode="r")
    smoke_embedding = args.output_dir / "smoke_embeddings.npy"
    np.save(
        smoke_embedding,
        np.asarray(embedding[manifest["global_cell_index"].to_numpy(np.int64)]),
    )

    prediction_dir = args.output_dir / "prediction"
    subprocess.run(
        [
            sys.executable,
            str(args.release_dir / "code" / "predict_scp682_sc15_locked.py"),
            "--release-dir",
            str(args.release_dir),
            "--rna-h5ad",
            str(smoke_h5ad),
            "--scfoundation-embeddings",
            str(smoke_embedding),
            "--output-dir",
            str(prediction_dir),
            "--device",
            args.device,
        ],
        check=True,
    )
    prediction = np.load(prediction_dir / "scp682_sc15_prediction.npy")
    if prediction.shape != (len(manifest), 20):
        raise RuntimeError(f"Unexpected prediction shape: {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise RuntimeError("Prediction contains non-finite values")
    (args.output_dir / "SMOKE_SUCCESS").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
