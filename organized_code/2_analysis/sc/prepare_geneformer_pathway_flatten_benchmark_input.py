import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-input-dir", required=True)
    parser.add_argument("--feature-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-name", default="geneformer_pathway_features.npy")
    parser.add_argument("--dtype", default="float32", choices=["float16", "float32"])
    args = parser.parse_args()

    template = Path(args.template_input_dir)
    feature_run = Path(args.feature_run_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    feature_path = feature_run / "intermediate" / args.feature_name
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    meta = pd.read_csv(template / "cell_metadata.tsv", sep="\t", low_memory=False)
    features = np.load(feature_path, mmap_mode="r")
    if features.shape[0] != len(meta):
        raise RuntimeError(f"cell count mismatch: features={features.shape[0]} metadata={len(meta)}")
    if features.ndim != 3:
        raise RuntimeError(f"expected 3D pathway features, got shape={features.shape}")

    out_dtype = np.float32 if args.dtype == "float32" else np.float16
    n_cells = int(features.shape[0])
    emb_dim = int(features.shape[1] * features.shape[2])
    store = np.lib.format.open_memmap(output / "embeddings.npy", mode="w+", dtype=out_dtype, shape=(n_cells, emb_dim))
    chunk = 4096
    for start in range(0, n_cells, chunk):
        stop = min(start + chunk, n_cells)
        store[start:stop] = np.asarray(features[start:stop], dtype=out_dtype).reshape(stop - start, emb_dim)
        store.flush()
        print(f"flattened {stop}/{n_cells}", flush=True)
    del store

    for name in ["cell_metadata.tsv", "phospho_target_table.tsv", "targets.npy", "target_mask.npy"]:
        copy_file(template / name, output / name)
    for name in ["model_input_manifest.json", "signal_seq_target_mapping_used.tsv"]:
        src = template / name
        if src.exists():
            copy_file(src, output / name)

    manifest = {
        "template_input_dir": str(template),
        "feature_run_dir": str(feature_run),
        "feature_path": str(feature_path),
        "output_dir": str(output),
        "n_cells": n_cells,
        "pathway_tokens": int(features.shape[1]),
        "feature_dim": int(features.shape[2]),
        "embedding_dim": emb_dim,
        "dtype": args.dtype,
    }
    (output / "foundation_embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"done output={output}", flush=True)


if __name__ == "__main__":
    main()
