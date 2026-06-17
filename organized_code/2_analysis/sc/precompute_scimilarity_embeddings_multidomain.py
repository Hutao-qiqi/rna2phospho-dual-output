import argparse
import gc
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def has_embedding(path: Path, obsm_key: str) -> bool:
    if not path.exists():
        return False
    try:
        obj = ad.read_h5ad(path, backed="r")
        ok = obsm_key in obj.obsm
        obj.file.close()
        return bool(ok)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    parser.add_argument("--scimilarity-repo", default=r"D:\data\lsy\repos\scimilarity")
    parser.add_argument("--model-dir", default=r"D:\data\lsy\models\scimilarity\model_v1.1")
    parser.add_argument(
        "--h5ad-dir",
        default=r"01_data\single_cell\intermediate\foundation_model_h5ad_inputs_v1",
    )
    parser.add_argument(
        "--output-dir",
        default=r"01_data\single_cell\intermediate\foundation_model_embeddings\scimilarity_v1_1_multidomain_v1",
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    repo_src = Path(args.scimilarity_repo) / "src"
    sys.path.insert(0, str(repo_src.resolve()))

    from scimilarity.cell_embedding import CellEmbedding
    from scimilarity.utils import align_dataset, lognorm_counts

    root = Path(args.root)
    h5ad_dir = root / args.h5ad_dir
    out_root = root / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(h5ad_dir / "foundation_h5ad_manifest.tsv", sep="\t")
    wanted = set(args.datasets) if args.datasets else None
    rows = []

    encoder = CellEmbedding(model_path=str(Path(args.model_dir)), use_gpu=bool(args.use_gpu))
    print(
        f"scimilarity model genes={encoder.n_genes} dim={encoder.latent_dim} gpu={args.use_gpu}",
        flush=True,
    )

    for row in manifest.to_dict("records"):
        dataset_id = str(row["dataset_id"])
        if wanted is not None and dataset_id not in wanted:
            continue
        h5ad_path = Path(str(row["h5ad"]))
        out_dir = out_root / dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_h5ad = out_dir / f"{h5ad_path.stem}_scimilarity_adata.h5ad"
        if args.skip_existing and has_embedding(out_h5ad, "X_scimilarity"):
            rows.append({
                "dataset_id": dataset_id,
                "status": "skipped_existing",
                "output_h5ad": str(out_h5ad),
            })
            pd.DataFrame(rows).to_csv(out_root / "scimilarity_embedding_manifest.tsv", sep="\t", index=False)
            continue

        print(f"scimilarity start {dataset_id} cells={row['n_cells']} genes={row['n_genes']}", flush=True)
        adata = ad.read_h5ad(h5ad_path)
        adata.var_names = adata.var["gene_symbol"].astype(str).to_numpy()
        adata.var_names_make_unique()
        adata.layers["counts"] = adata.X.copy()
        aligned = align_dataset(adata, encoder.gene_order, keep_obsm=False)
        aligned.layers["counts"] = aligned.X.copy()
        aligned = lognorm_counts(aligned)
        emb = encoder.get_embeddings(aligned.X, buffer_size=args.buffer_size)
        emb = np.asarray(emb, dtype=np.float32)
        out_obj = adata[:, []].copy()
        out_obj.obsm["X_scimilarity"] = emb
        out_obj.uns["scimilarity_model_dir"] = str(Path(args.model_dir))
        out_obj.write_h5ad(out_h5ad)
        rows.append({
            "dataset_id": dataset_id,
            "status": "completed",
            "n_cells": int(emb.shape[0]),
            "embedding_dim": int(emb.shape[1]),
            "output_h5ad": str(out_h5ad),
        })
        pd.DataFrame(rows).to_csv(out_root / "scimilarity_embedding_manifest.tsv", sep="\t", index=False)
        print(f"scimilarity done {dataset_id}: cells={emb.shape[0]} dim={emb.shape[1]}", flush=True)
        del adata, aligned, out_obj, emb
        gc.collect()

    (out_root / "scimilarity_embedding_manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
