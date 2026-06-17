import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phospho_model_common import ensure_dir, save_json
from precompute_scgpt_embeddings import load_rna


def read_barcodes(path: Path) -> list[str]:
    return [line.strip() for line in path.open("r", encoding="utf-8") if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    ap.add_argument("--shard-dirs", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=[
        "iccite_seq_tcell_2025",
        "qurie_seq_bjab_2021",
        "phospho_seq_blair_2025_phospho_multi",
        "vivo_seq_th17_2025",
    ])
    args = ap.parse_args()

    root = Path(args.root)
    shard_dirs = [root / p for p in args.shard_dirs]
    out_root = ensure_dir(root / args.output_dir)

    manifest = {"datasets": [], "shards": [str(p) for p in shard_dirs]}
    for dataset_id in args.datasets:
        target_cells = load_rna(root, dataset_id)[0].astype(str).tolist()
        target_index = {cell: i for i, cell in enumerate(target_cells)}
        seen = np.zeros(len(target_cells), dtype=bool)
        store = None
        emb_dim = None
        genes_written = False
        loaded_cells = 0

        out = ensure_dir(out_root / dataset_id)
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
            cells = read_barcodes(barcodes_path)
            emb = np.load(emb_path, mmap_mode="r")
            if emb.shape[0] != len(cells):
                raise RuntimeError(f"barcode/embedding mismatch for {ds_dir}: {len(cells)} vs {emb.shape[0]}")
            if store is None:
                emb_dim = int(emb.shape[1])
                store = np.lib.format.open_memmap(out / "embeddings.npy", mode="w+", dtype=np.float32, shape=(len(target_cells), emb_dim))
            elif emb.shape[1] != emb_dim:
                raise RuntimeError(f"embedding dimension mismatch for {ds_dir}")
            pos = np.asarray([target_index[c] for c in cells], dtype=np.int64)
            if seen[pos].any():
                dup = np.asarray(cells, dtype=object)[seen[pos]]
                raise RuntimeError(f"duplicate cells in shards for {dataset_id}: {dup[:5].tolist()}")
            store[pos, :] = emb.astype(np.float32, copy=False)
            store.flush()
            seen[pos] = True
            loaded_cells += len(cells)
            if not genes_written:
                genes_path = ds_dir / "genes_used.tsv"
                if genes_path.exists():
                    (out / "genes_used.tsv").write_text(genes_path.read_text(encoding="utf-8"), encoding="utf-8")
                    genes_written = True

        missing = int((~seen).sum())
        if missing:
            raise RuntimeError(f"{dataset_id} missing {missing} cells after merging shards")
        del store
        (out / "barcodes.tsv").write_text("\n".join(target_cells) + "\n", encoding="utf-8")
        save_json(out / "embedding_metadata.json", {
            "dataset_id": dataset_id,
            "n_cells": int(len(target_cells)),
            "embedding_dim": int(emb_dim),
            "merged_from": [str(p / dataset_id) for p in shard_dirs],
            "complete": True,
        })
        manifest["datasets"].append({
            "dataset_id": dataset_id,
            "n_cells": int(len(target_cells)),
            "loaded_cells": int(loaded_cells),
            "embedding_dim": int(emb_dim),
        })
        print(f"merged {dataset_id}: {len(target_cells)} cells dim={emb_dim}", flush=True)

    save_json(out_root / "merge_manifest.json", manifest)


if __name__ == "__main__":
    main()
