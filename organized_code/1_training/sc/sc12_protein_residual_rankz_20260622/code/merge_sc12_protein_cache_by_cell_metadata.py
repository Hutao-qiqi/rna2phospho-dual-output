# 合并各队列 scTranslator 蛋白预测缓存，并按 SCP682-SC12 模型输入细胞顺序对齐。

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MATRIX_NAME = "protein_predicted.npy"
PROTEIN_TABLE = "protein_features.tsv"
CELL_TABLE = "cells.tsv"


def read_table(path):
    return pd.read_csv(path, sep="\t", low_memory=False)


def cell_key(table):
    for col in ("cell_id", "cell", "barcode", "obs_name", "sample_barcode", "cell_key"):
        if col in table.columns:
            return col
    raise ValueError(f"no cell key column found in {list(table.columns)}")


def load_cache(cache_dir):
    cache_dir = Path(cache_dir)
    matrix_path = cache_dir / MATRIX_NAME
    protein_path = cache_dir / PROTEIN_TABLE
    cell_path = cache_dir / CELL_TABLE
    missing = [str(p) for p in (matrix_path, protein_path, cell_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("missing cache files: " + "; ".join(missing))
    matrix = np.load(matrix_path, mmap_mode="r")
    proteins = read_table(protein_path)
    cells = read_table(cell_path)
    if matrix.shape[0] != len(cells):
        raise ValueError(f"{cache_dir}: matrix rows {matrix.shape[0]} != cells {len(cells)}")
    if matrix.shape[1] != len(proteins):
        raise ValueError(f"{cache_dir}: matrix cols {matrix.shape[1]} != proteins {len(proteins)}")
    return matrix, proteins, cells


def protein_ids(protein_table):
    for col in ("protein_id", "protein_symbol", "protein", "gene", "canonical_gene_symbol"):
        if col in protein_table.columns:
            return protein_table[col].astype(str).str.upper().tolist()
    return protein_table.iloc[:, 0].astype(str).str.upper().tolist()


def build_dataset_cache_map(input_root, dataset_ids):
    mapping = {}
    for dataset_id in dataset_ids:
        d = Path(input_root) / str(dataset_id)
        if d.exists():
            mapping[str(dataset_id)] = d
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-input-dir", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--allow-extra-cache-datasets", action="store_true")
    args = ap.parse_args()

    model_input = Path(args.model_input_dir)
    cache_root = Path(args.cache_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_table(model_input / "cell_metadata.tsv")
    if "dataset_id" not in meta.columns:
        raise ValueError("model input cell_metadata.tsv lacks dataset_id")
    meta_key = cell_key(meta)
    dataset_ids = meta["dataset_id"].astype(str).drop_duplicates().tolist()
    cache_map = build_dataset_cache_map(cache_root, dataset_ids)
    missing_datasets = [x for x in dataset_ids if x not in cache_map]
    if missing_datasets:
        raise FileNotFoundError("missing dataset cache directories: " + ", ".join(missing_datasets))

    reference_proteins = None
    protein_table = None
    merged = None
    audit_rows = []

    for dataset_id in dataset_ids:
        matrix, proteins, cells = load_cache(cache_map[dataset_id])
        ids = protein_ids(proteins)
        if reference_proteins is None:
            reference_proteins = ids
            protein_table = proteins.copy()
            merged = np.lib.format.open_memmap(
                out_dir / MATRIX_NAME,
                mode="w+",
                dtype=np.float32,
                shape=(len(meta), len(reference_proteins)),
            )
            merged[:, :] = np.nan
            merged.flush()
        elif ids != reference_proteins:
            raise ValueError(f"{dataset_id}: protein order differs from reference")

        ckey = cell_key(cells)
        cache_index = {}
        duplicated = set()
        for i, value in enumerate(cells[ckey].astype(str)):
            if value in cache_index:
                duplicated.add(value)
            cache_index[value] = i
        if duplicated:
            raise ValueError(f"{dataset_id}: duplicated cache cell ids, examples: {list(duplicated)[:5]}")

        rows = np.flatnonzero(meta["dataset_id"].astype(str).to_numpy() == str(dataset_id))
        wanted = meta.iloc[rows][meta_key].astype(str).tolist()
        missing_cells = [x for x in wanted if x not in cache_index]
        if missing_cells:
            raise ValueError(f"{dataset_id}: missing {len(missing_cells)} model cells in cache, first={missing_cells[:5]}")
        order = np.asarray([cache_index[x] for x in wanted], dtype=np.int64)
        merged[rows, :] = np.asarray(matrix[order, :], dtype=np.float32)
        merged.flush()

        audit_rows.append(
            {
                "dataset_id": dataset_id,
                "model_cells": int(len(rows)),
                "cache_cells": int(matrix.shape[0]),
                "matched_cells": int(len(order)),
                "n_proteins": int(matrix.shape[1]),
                "cache_dir": str(cache_map[dataset_id]),
            }
        )

    if np.isnan(np.asarray(merged)).any():
        raise ValueError("merged matrix still contains NaN after all datasets")

    protein_table.to_csv(out_dir / PROTEIN_TABLE, sep="\t", index=False)
    meta.to_csv(out_dir / "cell_metadata.tsv", sep="\t", index=False)
    meta[["cell_id", "dataset_id"]].to_csv(out_dir / CELL_TABLE, sep="\t", index=False)
    pd.DataFrame(audit_rows).to_csv(out_dir / "merge_audit.tsv", sep="\t", index=False)
    manifest = {
        "schema_version": "scp682_sc12_merged_protein_cache_v1",
        "model_input_dir": str(model_input),
        "cache_root": str(cache_root),
        "n_cells": int(len(meta)),
        "n_proteins": int(len(reference_proteins)),
        "matrix_file": MATRIX_NAME,
        "protein_table": PROTEIN_TABLE,
        "cell_table": CELL_TABLE,
        "datasets": audit_rows,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
