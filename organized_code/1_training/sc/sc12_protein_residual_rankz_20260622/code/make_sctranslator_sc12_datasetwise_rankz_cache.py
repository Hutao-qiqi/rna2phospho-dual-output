from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-cache-dir", required=True)
    ap.add_argument("--output-cache-dir", required=True)
    ap.add_argument("--clip", type=float, default=6.0)
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    in_dir = Path(args.input_cache_dir)
    out_dir = Path(args.output_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(in_dir / "cells.tsv", sep="\t")
    proteins = pd.read_csv(in_dir / "protein_features.tsv", sep="\t")
    src = np.load(in_dir / "protein_predicted.npy", mmap_mode="r")
    n_cells, n_proteins = src.shape

    dst_path = out_dir / "protein_predicted.npy"
    dst = np.lib.format.open_memmap(dst_path, mode="w+", dtype=np.float32, shape=(n_cells, n_proteins))

    rows = []
    for dataset_id, idx in cells.groupby("dataset_id", sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        x = np.asarray(src[idx_arr, :], dtype=np.float32)
        mean = np.nanmean(x, axis=0, dtype=np.float64).astype(np.float32)
        sd = np.nanstd(x, axis=0, dtype=np.float64).astype(np.float32)
        bad = (~np.isfinite(sd)) | (sd < args.eps)
        sd[bad] = 1.0
        z = (x - mean.reshape(1, -1)) / sd.reshape(1, -1)
        z[:, bad] = 0.0
        if args.clip and args.clip > 0:
            z = np.clip(z, -args.clip, args.clip)
        dst[idx_arr, :] = z.astype(np.float32)
        dst.flush()
        rows.append(
            {
                "dataset_id": dataset_id,
                "n_cells": int(len(idx_arr)),
                "n_proteins": int(n_proteins),
                "n_zero_sd_proteins": int(bad.sum()),
                "raw_min": float(np.nanmin(x)),
                "raw_max": float(np.nanmax(x)),
                "raw_sd_median": float(np.nanmedian(np.nanstd(x, axis=0))),
                "z_min": float(np.nanmin(z)),
                "z_max": float(np.nanmax(z)),
                "z_mean_median": float(np.nanmedian(np.nanmean(z, axis=0))),
                "z_sd_median": float(np.nanmedian(np.nanstd(z, axis=0))),
            }
        )
        del x, z
        print(f"{dataset_id}: done", flush=True)

    cells.to_csv(out_dir / "cells.tsv", sep="\t", index=False)
    cells.to_csv(out_dir / "cell_metadata.tsv", sep="\t", index=False)
    proteins.to_csv(out_dir / "protein_features.tsv", sep="\t", index=False)
    if (in_dir / "manifest.json").exists():
        manifest = json.loads((in_dir / "manifest.json").read_text(encoding="utf-8"))
    else:
        manifest = {}
    manifest.update(
        {
            "schema_version": "scp682_sc12_datasetwise_rankz_cache_v1",
            "source_cache_dir": str(in_dir),
            "matrix_file": "protein_predicted.npy",
            "protein_table": "protein_features.tsv",
            "cell_table": "cells.tsv",
            "n_cells": int(n_cells),
            "n_proteins": int(n_proteins),
            "transform": "datasetwise_protein_zscore",
            "clip": float(args.clip),
            "eps": float(args.eps),
            "note": "Each protein column is z-scored within each dataset_id; values are clipped to reduce cross-cohort scale drift.",
        }
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "datasetwise_rankz_summary.tsv", sep="\t", index=False)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
