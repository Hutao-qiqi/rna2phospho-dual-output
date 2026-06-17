import argparse
import json
from pathlib import Path

import numpy as np

from phospho_model_common import ensure_dir, read_lines, save_json


def read_barcodes(path: Path) -> list[str]:
    return [line.strip() for line in path.open("r", encoding="utf-8") if line.strip()]


def expected_barcodes(root: Path, dataset_id: str) -> list[str]:
    if dataset_id in {"signal_seq_gse256403_hela_2024", "signal_seq_gse256404_pdo_caf_2024"}:
        import pandas as pd

        paired = root / "01_data" / "single_cell" / "intermediate" / "paired_matrices" / "signal_seq_gse256405_processed_h5ad_v1"
        meta = pd.read_csv(paired / "cell_metadata.tsv", sep="\t", usecols=["cell_id", "dataset_id"])
        return meta.loc[meta["dataset_id"].astype(str) == dataset_id, "cell_id"].astype(str).tolist()
    paired = root / "01_data" / "single_cell" / "intermediate" / "paired_matrices" / dataset_id
    return read_lines(paired / "barcodes.tsv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    shard_dirs = [root / p for p in args.shard_dirs]
    out_root = ensure_dir(root / args.output_dir)
    manifest = {"datasets": [], "shards": [str(p) for p in shard_dirs]}

    for dataset_id in args.datasets:
        cells = expected_barcodes(root, dataset_id)
        cell_to_i = {cell: i for i, cell in enumerate(cells)}
        seen = np.zeros(len(cells), dtype=bool)
        store = None
        emb_dim = None
        out = ensure_dir(out_root / dataset_id)
        loaded = 0
        for shard_dir in shard_dirs:
            ds_dir = shard_dir / dataset_id
            meta_path = ds_dir / "embedding_metadata.json"
            emb_path = ds_dir / "embeddings.npy"
            barcodes_path = ds_dir / "barcodes.tsv"
            if not (meta_path.exists() and emb_path.exists() and barcodes_path.exists()):
                raise FileNotFoundError(f"incomplete shard for {dataset_id}: {ds_dir}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not meta.get("complete", False):
                raise RuntimeError(f"shard not complete for {dataset_id}: {ds_dir}")
            shard_cells = read_barcodes(barcodes_path)
            emb = np.load(emb_path, mmap_mode="r")
            if emb.shape[0] != len(shard_cells):
                raise RuntimeError(f"barcode/embedding mismatch for {ds_dir}")
            if store is None:
                emb_dim = int(emb.shape[1])
                store = np.lib.format.open_memmap(out / "embeddings.npy", mode="w+", dtype=np.float32, shape=(len(cells), emb_dim))
            if emb.shape[1] != emb_dim:
                raise RuntimeError(f"embedding dimension mismatch for {ds_dir}")
            pos = np.asarray([cell_to_i[cell] for cell in shard_cells], dtype=np.int64)
            if seen[pos].any():
                raise RuntimeError(f"duplicate cells in shards for {dataset_id}")
            store[pos, :] = emb.astype(np.float32, copy=False)
            store.flush()
            seen[pos] = True
            loaded += len(shard_cells)
            genes_path = ds_dir / "genes_used.tsv"
            if genes_path.exists() and not (out / "genes_used.tsv").exists():
                (out / "genes_used.tsv").write_text(genes_path.read_text(encoding="utf-8"), encoding="utf-8")
        missing = int((~seen).sum())
        if missing:
            raise RuntimeError(f"{dataset_id} missing {missing} cells after merging")
        del store
        (out / "barcodes.tsv").write_text("\n".join(cells) + "\n", encoding="utf-8")
        save_json(out / "embedding_metadata.json", {
            "dataset_id": dataset_id,
            "n_cells": int(len(cells)),
            "embedding_dim": int(emb_dim),
            "merged_from": [str(p / dataset_id) for p in shard_dirs],
            "complete": True,
        })
        manifest["datasets"].append({"dataset_id": dataset_id, "n_cells": int(len(cells)), "loaded_cells": int(loaded), "embedding_dim": int(emb_dim)})
        print(f"merged {dataset_id}: {len(cells)} cells dim={emb_dim}", flush=True)

    save_json(out_root / "merge_manifest.json", manifest)


if __name__ == "__main__":
    main()
