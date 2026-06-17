import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def read_barcodes(path: Path) -> list[str]:
    return [line.strip() for line in path.open("r", encoding="utf-8") if line.strip()]


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_embedding_sources(root: Path, source_dirs: list[str], dataset_ids: list[str]):
    sources = {}
    for source_dir in source_dirs:
        base = root / source_dir
        for dataset_id in dataset_ids:
            ds = base / dataset_id
            emb_path = ds / "embeddings.npy"
            barcodes_path = ds / "barcodes.tsv"
            if not (emb_path.exists() and barcodes_path.exists()):
                continue
            sources[dataset_id] = {
                "dir": ds,
                "barcodes": read_barcodes(barcodes_path),
                "embeddings": np.load(emb_path, mmap_mode="r"),
            }
    missing = [dataset_id for dataset_id in dataset_ids if dataset_id not in sources]
    if missing:
        raise FileNotFoundError(f"missing embedding sources: {missing}")
    return sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    parser.add_argument("--template-input-dir", required=True)
    parser.add_argument("--source-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    template = Path(args.template_input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(template / "cell_metadata.tsv", sep="\t", low_memory=False)
    dataset_ids = meta["dataset_id"].astype(str).drop_duplicates().tolist()
    sources = load_embedding_sources(root, args.source_dirs, dataset_ids)
    emb_dim = int(next(iter(sources.values()))["embeddings"].shape[1])
    store = np.lib.format.open_memmap(out / "embeddings.npy", mode="w+", dtype=np.float32, shape=(len(meta), emb_dim))
    source_summary = []
    dataset_values = meta["dataset_id"].astype(str).to_numpy()

    for dataset_id in dataset_ids:
        ds_idx = np.flatnonzero(dataset_values == dataset_id)
        ds_meta = meta.iloc[ds_idx]
        src = sources[dataset_id]
        emb = src["embeddings"]
        if emb.shape[1] != emb_dim:
            raise RuntimeError(f"embedding dimension mismatch for {dataset_id}: {emb.shape[1]} vs {emb_dim}")
        barcode_to_i = {cell: i for i, cell in enumerate(src["barcodes"])}
        cells = ds_meta["cell_id"].astype(str).tolist()
        missing = [cell for cell in cells if cell not in barcode_to_i]
        if missing:
            raise RuntimeError(f"{dataset_id} missing embedding cells: {missing[:5]}")
        pos = np.asarray([barcode_to_i[cell] for cell in cells], dtype=np.int64)
        store[ds_idx, :] = emb[pos].astype(np.float32, copy=False)
        store.flush()
        source_summary.append({
            "dataset_id": dataset_id,
            "n_cells": int(len(ds_idx)),
            "source_dir": str(src["dir"]),
            "embedding_dim": emb_dim,
        })
        print(f"assembled {dataset_id}: {len(ds_idx)} cells", flush=True)
    del store

    for name in ["cell_metadata.tsv", "phospho_target_table.tsv", "targets.npy", "target_mask.npy"]:
        copy_file(template / name, out / name)
    for name in ["model_input_manifest.json", "signal_seq_target_mapping_used.tsv"]:
        src = template / name
        if src.exists():
            copy_file(src, out / name)

    manifest = {
        "method_name": args.method_name,
        "template_input_dir": str(template),
        "output_dir": str(out),
        "n_cells": int(len(meta)),
        "embedding_dim": emb_dim,
        "source_dirs": args.source_dirs,
        "sources": source_summary,
    }
    (out / "foundation_embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"done output={out}", flush=True)


if __name__ == "__main__":
    main()
