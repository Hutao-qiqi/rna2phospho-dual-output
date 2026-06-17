import argparse
import json
import os
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phospho_model_common import ensure_dir, read_lines, save_json
from precompute_scgpt_embeddings import load_rna as load_legacy_rna
from precompute_scfoundation_embeddings import (
    build_gene_projection,
    encode_batch_with_fallback,
    load_scfoundation_gene_list,
    load_scfoundation_model,
)


GSE_DATASET = "gse300551_iccite_plex_kinase_2025"
SIGNAL_DATASETS = {
    "signal_seq_gse256403_hela_2024",
    "signal_seq_gse256404_pdo_caf_2024",
}


def clean_gene_symbols(genes: np.ndarray) -> np.ndarray:
    cleaned = []
    for gene in genes.astype(str):
        for prefix in ("GRCh38_", "GRCh37_", "hg38_", "hg19_", "mm10_", "mm9_"):
            if gene.startswith(prefix):
                gene = gene[len(prefix):]
                break
        cleaned.append(gene)
    return np.asarray(cleaned)


def read_10x_h5(path: Path):
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        shape = tuple(int(x) for x in group["shape"][:])
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=shape,
            dtype=np.float32,
        )
        barcodes = np.asarray([
            x.decode() if isinstance(x, bytes) else str(x)
            for x in group["barcodes"][:]
        ])
        feature_group = group["features"]
        if "name" in feature_group:
            genes = np.asarray([
                x.decode() if isinstance(x, bytes) else str(x)
                for x in feature_group["name"][:]
            ])
        else:
            genes = np.asarray([
                x.decode() if isinstance(x, bytes) else str(x)
                for x in feature_group["id"][:]
            ])
    return genes, barcodes, matrix


def load_gse300551_rna(root: Path):
    paired = root / "01_data" / "single_cell" / "intermediate" / "paired_matrices" / GSE_DATASET
    cells = np.asarray(read_lines(paired / "barcodes.tsv"))
    meta = pd.read_csv(paired / "cell_metadata.tsv", sep="\t")
    manifest = pd.read_csv(paired / "rna_h5_manifest.tsv", sep="\t")
    parts = []
    part_cells = []
    genes_ref = None
    for _, row in manifest.iterrows():
        plate = str(row["plate"])
        genes, barcodes, mat = read_10x_h5(Path(row["rna_h5"]))
        if genes_ref is None:
            genes_ref = genes
        elif not np.array_equal(genes_ref, genes):
            raise RuntimeError(f"GSE300551 gene order differs in plate {plate}")
        keep_meta = meta[meta["plate"].astype(str) == plate]
        wanted_raw = keep_meta["raw_barcode"].astype(str).tolist()
        barcode_to_i = {bc: i for i, bc in enumerate(barcodes.astype(str))}
        missing = [bc for bc in wanted_raw if bc not in barcode_to_i]
        if missing:
            raise RuntimeError(f"{plate} missing RNA barcodes: {missing[:5]}")
        pos = np.asarray([barcode_to_i[bc] for bc in wanted_raw], dtype=np.int64)
        parts.append(mat[:, pos].T.tocsr())
        part_cells.extend(keep_meta["cell_id"].astype(str).tolist())
    x = sparse.vstack(parts, format="csr")
    pos = {cell: i for i, cell in enumerate(part_cells)}
    order = np.asarray([pos[cell] for cell in cells.astype(str)], dtype=np.int64)
    return cells, genes_ref, x[order].tocsr()


def load_signal_rna(root: Path, dataset_id: str):
    paired = root / "01_data" / "single_cell" / "intermediate" / "paired_matrices" / "signal_seq_gse256405_processed_h5ad_v1"
    meta = pd.read_csv(paired / "cell_metadata.tsv", sep="\t")
    meta = meta[meta["dataset_id"].astype(str) == dataset_id].reset_index(drop=True)
    if meta.empty:
        raise RuntimeError(f"no SIGNAL-seq cells for {dataset_id}")
    rna_path = Path(meta["rna_h5ad"].astype(str).iloc[0])
    adata = ad.read_h5ad(rna_path)
    genes = np.asarray(adata.var_names.astype(str))
    obs = np.asarray(adata.obs_names.astype(str))
    obs_to_i = {name: i for i, name in enumerate(obs)}
    if set(meta["pair_key"].astype(str)).issubset(obs_to_i):
        keys = meta["pair_key"].astype(str).tolist()
    elif set(meta["raw_barcode"].astype(str)).issubset(obs_to_i):
        keys = meta["raw_barcode"].astype(str).tolist()
    else:
        missing_pair = [x for x in meta["pair_key"].astype(str).head(20).tolist() if x not in obs_to_i]
        missing_raw = [x for x in meta["raw_barcode"].astype(str).head(20).tolist() if x not in obs_to_i]
        raise RuntimeError(f"cannot match SIGNAL-seq obs names; pair missing={missing_pair[:3]}, raw missing={missing_raw[:3]}")
    pos = np.asarray([obs_to_i[x] for x in keys], dtype=np.int64)
    x = adata.X[pos]
    if sparse.issparse(x):
        x = x.tocsr().astype(np.float32)
    else:
        x = sparse.csr_matrix(np.asarray(x, dtype=np.float32))
    cells = meta["cell_id"].astype(str).to_numpy()
    return cells, genes, x


def load_multidomain_rna(root: Path, dataset_id: str):
    if dataset_id == GSE_DATASET:
        return load_gse300551_rna(root)
    if dataset_id in SIGNAL_DATASETS:
        return load_signal_rna(root, dataset_id)
    return load_legacy_rna(root, dataset_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    parser.add_argument("--code-dir", default=r"D:\data\lsy\repos\scFoundation_model")
    parser.add_argument("--weight-path", default=r"D:\data\lsy\models\scfoundation\models.ckpt")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-gene-tokens", type=int, default=12000)
    parser.add_argument("--highres", type=float, default=4.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    root = Path(args.root)
    out_root = ensure_dir(root / args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    target_genes = load_scfoundation_gene_list(Path(args.code_dir))
    model, config = load_scfoundation_model(Path(args.code_dir), Path(args.weight_path), device)

    for dataset_id in args.datasets:
        cells, genes, x = load_multidomain_rna(root, dataset_id)
        if dataset_id == GSE_DATASET:
            genes = clean_gene_symbols(np.asarray(genes))
        genes = np.asarray([g.upper() if dataset_id == "vivo_seq_th17_2025" else g for g in genes])
        idx = np.arange(len(cells), dtype=np.int64)[args.shard_index::args.num_shards]
        cells_out = np.asarray(cells).astype(str)[idx]
        out = ensure_dir(out_root / dataset_id)
        emb_path = out / "embeddings.npy"
        meta_path = out / "embedding_metadata.json"
        if args.skip_existing and emb_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("complete", False):
                print(f"{dataset_id} shard {args.shard_index}/{args.num_shards} exists; skip", flush=True)
                continue

        proj, n_source_used, n_target_hit = build_gene_projection(genes, target_genes)
        x = x.tocsr() if sparse.issparse(x) else sparse.csr_matrix(np.asarray(x, dtype=np.float32))
        emb_store = None
        for start in range(0, len(idx), args.batch_size):
            stop = min(start + args.batch_size, len(idx))
            batch_rows = idx[start:stop]
            x_aligned = (x[batch_rows] @ proj).tocsr().astype(np.float32)
            emb = encode_batch_with_fallback(model, config, x_aligned, device, args.highres, args.max_gene_tokens)
            if emb_store is None:
                emb_store = np.lib.format.open_memmap(
                    emb_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(idx), int(emb.shape[1])),
                )
            emb_store[start:stop, :] = emb
            emb_store.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"{dataset_id} shard {args.shard_index}/{args.num_shards} encoded {stop}/{len(idx)}", flush=True)

        if emb_store is None:
            np.save(emb_path, np.empty((0, 0), dtype=np.float32))
            emb_dim = 0
        else:
            emb_dim = int(emb_store.shape[1])
            del emb_store
        (out / "barcodes.tsv").write_text("\n".join(cells_out.tolist()) + "\n", encoding="utf-8")
        (out / "genes_used.tsv").write_text("\n".join(target_genes) + "\n", encoding="utf-8")
        save_json(
            meta_path,
            {
                "dataset_id": dataset_id,
                "n_cells": int(len(cells_out)),
                "n_input_cells_total": int(len(cells)),
                "embedding_dim": int(emb_dim),
                "n_input_genes": int(len(genes)),
                "n_source_genes_used": int(n_source_used),
                "n_scfoundation_genes_hit": int(n_target_hit),
                "code_dir": str(args.code_dir),
                "weight_path": str(args.weight_path),
                "max_gene_tokens": int(args.max_gene_tokens),
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
                "complete": True,
            },
        )


if __name__ == "__main__":
    main()
