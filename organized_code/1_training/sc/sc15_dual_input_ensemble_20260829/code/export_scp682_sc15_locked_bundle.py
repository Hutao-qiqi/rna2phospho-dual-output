from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_log1p(matrix):
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
    matrix = sparse.diags(scale) @ matrix
    matrix.data = np.log1p(matrix.data)
    return matrix.tocsr()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--copy-models", action="store_true")
    args = parser.parse_args()

    release = args.release_root
    release.mkdir(parents=True, exist_ok=True)
    (release / "models").mkdir(exist_ok=True)
    (release / "preprocessing").mkdir(exist_ok=True)

    checkpoints = pd.read_csv(release / "CHECKPOINTS.tsv", sep="\t")
    panel = pd.read_csv(args.contract_dir / "full20_panel.tsv", sep="\t")
    manifest = pd.read_csv(args.contract_dir / "gse300551_cell_manifest.tsv", sep="\t")
    contract = pd.read_csv(args.contract_dir / "target_split_contract.tsv", sep="\t")
    panel.to_csv(release / "readout_schema.tsv", sep="\t", index=False)

    input_dir = (
        args.project_root
        / r"01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_gse300551_signal_seq_multidomain_v1"
    )
    rna_path = (
        args.project_root
        / r"01_data\single_cell\intermediate\foundation_model_h5ad_inputs_v1\gse300551_iccite_plex_kinase_2025.h5ad"
    )
    rna = ad.read_h5ad(rna_path)
    lookup = {str(cell): i for i, cell in enumerate(rna.obs_names.astype(str))}
    positions = np.asarray(
        [lookup[str(cell)] for cell in manifest["cell_id"].astype(str)], dtype=np.int64
    )
    matrix = normalize_log1p(rna.X[positions])
    gene_names = rna.var_names.astype(str).to_numpy()

    global_indices = manifest["global_cell_index"].to_numpy(np.int64)
    target_indices = panel["target_index"].to_numpy(np.int64)
    raw_targets = np.asarray(
        np.load(input_dir / "targets.npy", mmap_mode="r")[
            np.ix_(global_indices, target_indices)
        ],
        dtype=np.float32,
    )
    embedding_dim = int(np.load(input_dir / "embeddings.npy", mmap_mode="r").shape[1])

    model_rows = []
    hvg_columns = {"rank": np.arange(1, 4001, dtype=np.int64)}
    hvg_manifest = []
    ensemble_splits = []

    for row in checkpoints.itertuples(index=False):
        split_seed = int(row.split_seed)
        split = contract[contract["seed"].eq(split_seed)]
        optimization = split.loc[
            split["split"].eq("optimization"), "dataset_cell_index"
        ].to_numpy(np.int64)

        member_dirs = [Path(row.member1), Path(row.member2), Path(row.member3)]
        source_dirs = [args.development_root / member for member in member_dirs]
        hvg_sets = [np.load(source / "hvg_indices.npy") for source in source_dirs]
        if not all(np.array_equal(hvg_sets[0], values) for values in hvg_sets[1:]):
            raise RuntimeError(f"HVG mismatch within split {split_seed}")
        hvg_indices = hvg_sets[0].astype(np.int64)
        if len(hvg_indices) != 4000:
            raise RuntimeError(f"Expected 4000 HVGs for split {split_seed}")

        train_matrix = matrix[optimization][:, hvg_indices]
        gene_mean = np.asarray(train_matrix.mean(axis=0)).ravel().astype(np.float32)
        gene_second = np.asarray(train_matrix.power(2).mean(axis=0)).ravel().astype(np.float32)
        gene_std = np.sqrt(np.maximum(gene_second - gene_mean**2, 1e-6)).astype(
            np.float32
        )
        target_mean = raw_targets[optimization].mean(axis=0).astype(np.float32)
        target_std = raw_targets[optimization].std(axis=0).astype(np.float32)
        target_std[target_std < 1e-6] = 1.0

        preprocessing_path = release / "preprocessing" / f"split_{split_seed}.npz"
        np.savez_compressed(
            preprocessing_path,
            gene_names=np.asarray(gene_names[hvg_indices], dtype=str),
            gene_indices=hvg_indices,
            gene_mean=gene_mean,
            gene_std=gene_std,
            target_ids=np.asarray(panel["target_id"].astype(str).to_numpy(), dtype=str),
            target_mean=target_mean,
            target_std=target_std,
            embedding_dim=np.asarray(embedding_dim, dtype=np.int64),
        )
        hvg_columns[f"split_{split_seed}"] = gene_names[hvg_indices]
        hvg_manifest.append(
            {
                "split_seed": split_seed,
                "n_genes": len(hvg_indices),
                "preprocessing_file": preprocessing_path.relative_to(release).as_posix(),
                "preprocessing_sha256": sha256(preprocessing_path),
            }
        )

        members = []
        for member_number, source_dir in enumerate(source_dirs, start=1):
            source_checkpoint = source_dir / "best_direct_checkpoint.pt"
            if not source_checkpoint.exists():
                raise FileNotFoundError(source_checkpoint)
            destination = (
                release
                / "models"
                / f"split_{split_seed}"
                / f"member{member_number}.pt"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if args.copy_models:
                shutil.copy2(source_checkpoint, destination)
            model_path = destination if destination.exists() else source_checkpoint
            members.append(destination.relative_to(release).as_posix())
            model_rows.append(
                {
                    "split_seed": split_seed,
                    "member": member_number,
                    "training_directory": member_dirs[member_number - 1].as_posix(),
                    "release_file": destination.relative_to(release).as_posix(),
                    "bytes": source_checkpoint.stat().st_size,
                    "sha256": sha256(source_checkpoint),
                }
            )

        ensemble_splits.append(
            {
                "split_seed": split_seed,
                "aggregation": str(row.selected_ensemble),
                "members": members,
                "preprocessing": preprocessing_path.relative_to(release).as_posix(),
            }
        )

    pd.DataFrame(model_rows).to_csv(
        release / "model_hashes.tsv", sep="\t", index=False
    )
    pd.DataFrame(hvg_columns).to_csv(
        release / "hvg4000_ordered.txt", sep="\t", index=False
    )
    pd.DataFrame(hvg_manifest).to_csv(
        release / "hvg4000_manifest.tsv", sep="\t", index=False
    )

    normalization = {
        "rna": {
            "library_size": 10000,
            "transform": "log1p",
            "hvg_selection": "top variance on each optimization fold",
            "standardization": "fold-specific optimization mean and standard deviation",
            "missing_external_gene": "standardized mean imputation (0 after z-score)",
        },
        "scfoundation": {
            "embedding_dim": embedding_dim,
            "frozen": True,
            "cell_order_requirement": "embedding rows must match input h5ad obs_names",
        },
        "output": {
            "training_scale": "fold-specific target z-score",
            "external_output": "inverse transformed to GSE300551 training scale",
        },
    }
    (release / "input_normalization.json").write_text(
        json.dumps(normalization, indent=2), encoding="utf-8"
    )

    ensemble_config = {
        "model": "SCP682-SC15 Dual-Input Ensemble",
        "variant": "A2",
        "n_hvg": 4000,
        "n_readouts": len(panel),
        "n_folds": len(ensemble_splits),
        "members_per_fold": 3,
        "folds": ensemble_splits,
        "cross_fold_aggregation": "arithmetic mean after inverse target scaling",
    }
    (release / "ensemble_config.json").write_text(
        json.dumps(ensemble_config, indent=2), encoding="utf-8"
    )

    lock_path = release / "FINAL_MODEL_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["portable_release"] = {
        "inference_entry": "code/predict_scp682_sc15_locked.py",
        "model_hashes": "model_hashes.tsv",
        "hvg_order": "hvg4000_ordered.txt",
        "normalization": "input_normalization.json",
        "readout_schema": "readout_schema.tsv",
        "ensemble_config": "ensemble_config.json",
        "external_protocol": "external_validation_protocol.md",
    }
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps({"models": len(model_rows), "splits": len(ensemble_splits)}))


if __name__ == "__main__":
    main()
