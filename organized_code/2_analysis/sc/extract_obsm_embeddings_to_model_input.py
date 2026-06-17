import argparse
import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    parser.add_argument("--template-input-dir", required=True)
    parser.add_argument("--h5ad-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--obsm-key", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    template = Path(args.template_input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(template / "cell_metadata.tsv", sep="\t", low_memory=False)
    wanted = set(args.datasets) if args.datasets else None
    if wanted is not None:
        keep = meta["dataset_id"].astype(str).isin(wanted).to_numpy()
        source_row_indices = np.flatnonzero(keep)
        meta = meta.iloc[source_row_indices].copy().reset_index(drop=True)
    else:
        source_row_indices = None
    manifest = pd.read_csv(args.h5ad_manifest, sep="\t")
    manifest = manifest.loc[manifest["status"].astype(str).isin(["completed", "skipped_existing"])].copy()
    by_dataset = {str(row["dataset_id"]): Path(str(row["output_h5ad"])) for row in manifest.to_dict("records")}
    dataset_ids = meta["dataset_id"].astype(str).drop_duplicates().tolist()
    missing = [dataset_id for dataset_id in dataset_ids if dataset_id not in by_dataset]
    if missing:
        raise RuntimeError(f"missing embedded h5ad for datasets: {missing}")

    stores = {}
    emb_dim = None
    for dataset_id in dataset_ids:
        h5ad = by_dataset[dataset_id]
        if not h5ad.exists():
            raise FileNotFoundError(h5ad)
        adata = ad.read_h5ad(h5ad)
        if args.obsm_key not in adata.obsm:
            raise RuntimeError(f"{h5ad} missing obsm[{args.obsm_key}]")
        emb = np.asarray(adata.obsm[args.obsm_key], dtype=np.float32)
        if emb_dim is None:
            emb_dim = int(emb.shape[1])
        elif int(emb.shape[1]) != emb_dim:
            raise RuntimeError(f"embedding dim mismatch: {dataset_id} {emb.shape[1]} vs {emb_dim}")
        stores[dataset_id] = {
            "obs_names": np.asarray(adata.obs_names.astype(str)),
            "embeddings": emb,
            "path": str(h5ad),
        }
        print(f"loaded {dataset_id}: cells={emb.shape[0]} dim={emb.shape[1]}", flush=True)

    dataset_values = meta["dataset_id"].astype(str).to_numpy()
    store = np.lib.format.open_memmap(out / "embeddings.npy", mode="w+", dtype=np.float32, shape=(len(meta), emb_dim))
    source_rows = []
    for dataset_id in dataset_ids:
        idx = np.flatnonzero(dataset_values == dataset_id)
        ds_meta = meta.iloc[idx]
        source = stores[dataset_id]
        obs_to_i = {cell: i for i, cell in enumerate(source["obs_names"])}
        cells = ds_meta["cell_id"].astype(str).tolist()
        missing_cells = [cell for cell in cells if cell not in obs_to_i]
        if missing_cells:
            raise RuntimeError(f"{dataset_id} missing cells: {missing_cells[:5]}")
        pos = np.asarray([obs_to_i[cell] for cell in cells], dtype=np.int64)
        store[idx, :] = source["embeddings"][pos]
        store.flush()
        source_rows.append({
            "dataset_id": dataset_id,
            "n_cells": int(len(idx)),
            "source_h5ad": source["path"],
            "obsm_key": args.obsm_key,
            "embedding_dim": int(emb_dim),
        })
        print(f"assembled {dataset_id}: {len(idx)}", flush=True)
    del store

    meta.to_csv(out / "cell_metadata.tsv", sep="\t", index=False)
    copy_file(template / "phospho_target_table.tsv", out / "phospho_target_table.tsv")
    if source_row_indices is None:
        copy_file(template / "targets.npy", out / "targets.npy")
        copy_file(template / "target_mask.npy", out / "target_mask.npy")
    else:
        targets = np.load(template / "targets.npy", mmap_mode="r")
        target_mask = np.load(template / "target_mask.npy", mmap_mode="r")
        np.save(out / "targets.npy", np.asarray(targets[source_row_indices], dtype=np.float32))
        np.save(out / "target_mask.npy", np.asarray(target_mask[source_row_indices], dtype=target_mask.dtype))
    pd.DataFrame(source_rows).to_csv(out / "embedding_sources.tsv", sep="\t", index=False)
    (out / "foundation_embedding_manifest.json").write_text(
        json.dumps({
            "method_name": args.method_name,
            "template_input_dir": str(template),
            "output_dir": str(out),
            "obsm_key": args.obsm_key,
            "embedding_dim": int(emb_dim),
            "n_cells": int(len(meta)),
            "sources": source_rows,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"done output={out}", flush=True)


if __name__ == "__main__":
    main()
