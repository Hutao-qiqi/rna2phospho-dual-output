import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phospho_model_common import ensure_dir
from precompute_scfoundation_embeddings_multidomain import clean_gene_symbols, load_multidomain_rna


def normalize_gene_symbols(genes: np.ndarray, dataset_id: str) -> np.ndarray:
    genes = clean_gene_symbols(np.asarray(genes, dtype=str))
    if dataset_id == "vivo_seq_th17_2025":
        genes = np.asarray([g.upper() for g in genes], dtype=str)
    genes = np.asarray([g.split(".")[0] if g.startswith(("ENSG", "ENSMUSG")) else g for g in genes], dtype=str)
    genes = np.asarray([g.strip() if str(g).strip() else f"unknown_gene_{i}" for i, g in enumerate(genes)], dtype=str)
    return genes


def collapse_duplicate_genes(x, genes: np.ndarray):
    genes = np.asarray(genes, dtype=str)
    keep = np.asarray([g not in {"", "nan", "None"} for g in genes])
    if not np.all(keep):
        x = x[:, keep]
        genes = genes[keep]
    unique, codes = np.unique(genes, return_inverse=True)
    if len(unique) == len(genes):
        return x.tocsr() if sparse.issparse(x) else sparse.csr_matrix(np.asarray(x, dtype=np.float32)), unique, pd.DataFrame({
            "gene_symbol": unique,
            "n_source_features": np.ones(len(unique), dtype=int),
        })
    mapper = sparse.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (np.arange(len(codes)), codes)),
        shape=(len(codes), len(unique)),
    )
    x = x.tocsr() if sparse.issparse(x) else sparse.csr_matrix(np.asarray(x, dtype=np.float32))
    collapsed = (x @ mapper).tocsr().astype(np.float32)
    counts = np.bincount(codes, minlength=len(unique))
    return collapsed, unique, pd.DataFrame({
        "gene_symbol": unique,
        "n_source_features": counts.astype(int),
    })


def sanitize_frame_for_h5ad(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_categorical_dtype(series):
            out[col] = series.astype(str).replace({"nan": "", "None": "", "<NA>": ""})
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            out[col] = series.where(series.notna(), "").astype(str)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    parser.add_argument(
        "--model-input-dir",
        default=r"01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_gse300551_signal_seq_multidomain_v1",
    )
    parser.add_argument(
        "--output-dir",
        default=r"01_data\single_cell\intermediate\foundation_model_h5ad_inputs_v1",
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--compression", default="gzip")
    args = parser.parse_args()

    root = Path(args.root)
    input_dir = root / args.model_input_dir
    out_dir = ensure_dir(root / args.output_dir)
    meta = pd.read_csv(input_dir / "cell_metadata.tsv", sep="\t", low_memory=False)
    dataset_order = args.datasets or meta["dataset_id"].astype(str).drop_duplicates().tolist()
    rows = []

    for dataset_id in dataset_order:
        ds_meta = meta.loc[meta["dataset_id"].astype(str).eq(dataset_id)].copy().reset_index(drop=True)
        if ds_meta.empty:
            continue
        print(f"load {dataset_id}: n_cells={len(ds_meta)}", flush=True)
        cells, genes, x = load_multidomain_rna(root, dataset_id)
        cells = np.asarray(cells, dtype=str)
        genes = normalize_gene_symbols(np.asarray(genes, dtype=str), dataset_id)
        x = x.tocsr() if sparse.issparse(x) else sparse.csr_matrix(np.asarray(x, dtype=np.float32))
        cell_to_i = {cell: i for i, cell in enumerate(cells)}
        wanted = ds_meta["cell_id"].astype(str).tolist()
        missing = [cell for cell in wanted if cell not in cell_to_i]
        if missing:
            raise RuntimeError(f"{dataset_id} missing cells in RNA matrix: {missing[:5]}")
        pos = np.asarray([cell_to_i[cell] for cell in wanted], dtype=np.int64)
        x = x[pos].tocsr().astype(np.float32)
        x, unique_genes, var = collapse_duplicate_genes(x, genes)
        var = sanitize_frame_for_h5ad(var)
        var.index = var["gene_symbol"].astype(str)
        var["ensembl_id"] = ""
        var["feature_id"] = var["gene_symbol"].astype(str)

        obs = sanitize_frame_for_h5ad(ds_meta)
        obs.index = obs["cell_id"].astype(str)
        adata = ad.AnnData(X=x, obs=obs, var=var)
        adata.uns["dataset_id"] = dataset_id
        adata.uns["source_model_input_dir"] = str(input_dir)
        out_path = out_dir / f"{dataset_id}.h5ad"
        adata.write_h5ad(out_path, compression=args.compression)
        rows.append({
            "dataset_id": dataset_id,
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_nonzero": int(adata.X.nnz if sparse.issparse(adata.X) else np.count_nonzero(adata.X)),
            "h5ad": str(out_path),
        })
        pd.DataFrame(rows).to_csv(out_dir / "foundation_h5ad_manifest.tsv", sep="\t", index=False)
        print(f"wrote {out_path} cells={adata.n_obs} genes={adata.n_vars}", flush=True)

    manifest = {
        "model_input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "datasets": rows,
    }
    (out_dir / "foundation_h5ad_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
