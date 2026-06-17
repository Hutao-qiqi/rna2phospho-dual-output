import argparse
import gc
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import pandas as pd
import torch
from accelerate import Accelerator


def to_uce_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def dataset_species(dataset_id: str) -> str:
    return "mouse" if dataset_id == "vivo_seq_th17_2025" else "human"


def has_uce_embedding(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        adata = ad.read_h5ad(path, backed="r")
        ok = "X_uce" in adata.obsm
        adata.file.close()
        return bool(ok)
    except Exception:
        return False


def run_one(dataset_id: str, h5ad_path: Path, out_dir: Path, args) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    final_h5ad = out_dir / f"{h5ad_path.stem}_uce_adata.h5ad"
    if args.skip_existing and has_uce_embedding(final_h5ad):
        return {
            "dataset_id": dataset_id,
            "status": "skipped_existing",
            "species": dataset_species(dataset_id),
            "output_h5ad": str(final_h5ad),
        }

    ns = SimpleNamespace(
        adata_path=to_uce_path(h5ad_path),
        dir=to_uce_path(out_dir) + "/",
        species=dataset_species(dataset_id),
        filter=False,
        skip=True,
        model_loc=args.model_loc,
        batch_size=args.batch_size,
        pad_length=args.pad_length,
        pad_token_idx=0,
        chrom_token_left_idx=1,
        chrom_token_right_idx=2,
        cls_token_idx=3,
        CHROM_TOKEN_OFFSET=143574,
        sample_size=args.sample_size,
        CXG=True,
        nlayers=args.nlayers,
        output_dim=args.output_dim,
        d_hid=args.d_hid,
        token_dim=args.token_dim,
        multi_gpu=False,
        spec_chrom_csv_path="./model_files/species_chrom.csv",
        token_file="./model_files/all_tokens.torch",
        protein_embeddings_dir="./model_files/protein_embeddings/",
        offset_pkl_path="./model_files/species_offsets.pkl",
    )
    accelerator = Accelerator(project_dir=ns.dir)
    processor = args.processor_cls(ns, accelerator)
    processor.preprocess_anndata()
    processor.generate_idxs()
    processor.run_evaluation()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return {
        "dataset_id": dataset_id,
        "status": "completed",
        "species": ns.species,
        "output_h5ad": str(final_h5ad),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    parser.add_argument("--uce-repo", default=r"D:\data\lsy\repos\UCE")
    parser.add_argument(
        "--h5ad-dir",
        default=r"01_data\single_cell\intermediate\foundation_model_h5ad_inputs_v1",
    )
    parser.add_argument(
        "--output-dir",
        default=r"01_data\single_cell\intermediate\foundation_model_embeddings\uce_4layer_multidomain_v1",
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--model-loc", default=None)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--output-dim", type=int, default=1280)
    parser.add_argument("--d-hid", type=int, default=5120)
    parser.add_argument("--token-dim", type=int, default=5120)
    parser.add_argument("--pad-length", type=int, default=1536)
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    h5ad_dir = root / args.h5ad_dir
    out_root = root / args.output_dir
    manifest = pd.read_csv(h5ad_dir / "foundation_h5ad_manifest.tsv", sep="\t")
    wanted = set(args.datasets) if args.datasets else None
    rows = []

    os.chdir(args.uce_repo)
    sys.path.insert(0, str(Path(args.uce_repo).resolve()))
    from evaluate import AnndataProcessor

    args.processor_cls = AnndataProcessor
    for row in manifest.to_dict("records"):
        dataset_id = str(row["dataset_id"])
        if wanted is not None and dataset_id not in wanted:
            continue
        h5ad_path = Path(str(row["h5ad"]))
        print(f"UCE start {dataset_id} cells={row['n_cells']} genes={row['n_genes']}", flush=True)
        result = run_one(dataset_id, h5ad_path, out_root / dataset_id, args)
        rows.append(result)
        pd.DataFrame(rows).to_csv(out_root / "uce_embedding_manifest.tsv", sep="\t", index=False)
        print(f"UCE {dataset_id}: {result['status']}", flush=True)

    (out_root / "uce_embedding_manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
