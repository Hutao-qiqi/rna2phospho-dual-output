from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch


def read_protein_list(path: str | Path) -> list[str]:
    return [x.strip().upper() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def load_gene_id_map(repo_dir: str | Path):
    id_dir = Path(repo_dir) / "code" / "model" / "ID_dic"
    with (id_dir / "hgs_to_EntrezID.pkl").open("rb") as fh:
        hgs_to_entrez = pickle.load(fh)
    with (id_dir / "EntrezID_to_myID.pkl").open("rb") as fh:
        entrez_to_myid = pickle.load(fh)
    return hgs_to_entrez, entrez_to_myid


def gene_to_myid(gene: str, hgs_to_entrez: dict, entrez_to_myid: dict) -> int:
    entrez = hgs_to_entrez.get(str(gene).upper())
    if entrez is None:
        return 0
    vals = list(entrez) if isinstance(entrez, (list, tuple, set)) else [entrez]
    for val in vals:
        for key in [val, str(val), str(val).split(".")[0]]:
            if key in entrez_to_myid:
                return int(entrez_to_myid[key])
    return 0


def add_myid(adata, repo_dir: str | Path):
    hgs_to_entrez, entrez_to_myid = load_gene_id_map(repo_dir)
    adata.var["my_Id"] = [gene_to_myid(g, hgs_to_entrez, entrez_to_myid) for g in adata.var_names]
    return adata


def normalize_rows(x: np.ndarray, low: float = 1e-8, high: float = 1.0) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = np.nanmin(x, axis=1, keepdims=True)
    mx = np.nanmax(x, axis=1, keepdims=True)
    denom = np.where((mx - mn) > 0, mx - mn, 1.0)
    return low + (x - mn) / denom * (high - low)


def make_rna_tensors(rna_chunk, enc_len: int, device: str):
    x = rna_chunk.X
    if sp.issparse(x):
        x = x.toarray()
    x = normalize_rows(np.asarray(x, dtype=np.float32))
    genes = rna_chunk.var["my_Id"].astype(int).to_numpy()
    n_cells, n_genes = x.shape
    if n_genes >= enc_len:
        x_pad = x[:, :enc_len]
        gene_pad = genes[:enc_len]
        mask = np.ones((n_cells, enc_len), dtype=bool)
    else:
        x_pad = np.zeros((n_cells, enc_len), dtype=np.float32)
        x_pad[:, :n_genes] = x
        gene_pad = np.zeros(enc_len, dtype=np.int64)
        gene_pad[:n_genes] = genes
        mask = np.zeros((n_cells, enc_len), dtype=bool)
        mask[:, :n_genes] = True
    gene_mat = np.tile(gene_pad.reshape(1, -1), (n_cells, 1))
    return (
        torch.tensor(x_pad, dtype=torch.float32, device=device),
        torch.tensor(gene_mat, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.bool, device=device),
    )


def make_shards(proteins: list[str], repo_dir: str | Path, shard_size: int):
    hgs_to_entrez, entrez_to_myid = load_gene_id_map(repo_dir)
    shards = []
    for start in range(0, len(proteins), shard_size):
        end = min(start + shard_size, len(proteins))
        shard = proteins[start:end]
        gene_ids = np.zeros(shard_size, dtype=np.int64)
        mask = np.zeros(shard_size, dtype=bool)
        for i, protein in enumerate(shard):
            gene_ids[i] = gene_to_myid(protein, hgs_to_entrez, entrez_to_myid)
            mask[i] = gene_ids[i] != 0
        if not mask[: len(shard)].all():
            bad = [p for p, ok in zip(shard, mask[: len(shard)]) if not ok]
            raise ValueError(f"unmapped proteins in shard {start}:{end}: {bad[:10]}")
        shards.append({"start": start, "end": end, "proteins": shard, "gene_ids": gene_ids, "mask": mask})
    return shards


def write_cell_table(rna, out_path: Path):
    cells = pd.DataFrame({"cell_id": rna.obs_names.astype(str).tolist()})
    for col in ["dataset_id", "sample_id", "condition", "cell_type", "celltype"]:
        if col in rna.obs.columns:
            cells[col] = rna.obs[col].astype(str).tolist()
    cells.to_csv(out_path, sep="\t", index=False)


def write_protein_table(proteins: list[str], out_path: Path):
    pd.DataFrame(
        {
            "protein_id": proteins,
            "raw_feature_id": proteins,
            "canonical_gene_symbol": proteins,
            "source_model": "scTranslator",
            "model_feature_id": proteins,
            "match_status": "hgnc_protein_coding_query",
        }
    ).to_csv(out_path, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default=r"D:\data\lsy\models\scTranslator")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rna-h5ad", required=True)
    parser.add_argument("--protein-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--enc-max-seq-len", type=int, default=20000)
    parser.add_argument("--protein-shard-size", type=int, default=1000)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.repo_dir) / "code" / "model"
    sys.path.insert(0, str(model_dir))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_dir = out_dir / "_chunk_done"
    done_dir.mkdir(exist_ok=True)

    proteins = read_protein_list(args.protein_list)
    shards = make_shards(proteins, args.repo_dir, args.protein_shard_size)
    rna = sc.read_h5ad(args.rna_h5ad)
    if args.max_cells and args.max_cells > 0:
        rna = rna[: args.max_cells].copy()
    if rna.obs_names.has_duplicates:
        rna.obs_names_make_unique()
    rna = add_myid(rna, args.repo_dir)
    n_cells = int(rna.n_obs)
    n_proteins = len(proteins)

    pred_path = out_dir / "protein_predicted.npy"
    pred_mode = "r+" if args.resume and pred_path.exists() else "w+"
    pred = np.lib.format.open_memmap(pred_path, mode=pred_mode, dtype=np.float32, shape=(n_cells, n_proteins))

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    try:
        model = torch.load(args.checkpoint, map_location=torch.device(device), weights_only=False)
    except TypeError:
        model = torch.load(args.checkpoint, map_location=torch.device(device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for start in range(0, n_cells, args.chunk_size):
            end = min(start + args.chunk_size, n_cells)
            done = done_dir / f"{start}_{end}.done"
            if args.resume and done.exists():
                print(f"skip chunk {start}:{end}", flush=True)
                continue
            rna_chunk = rna[start:end].copy()
            x_value, rna_gene_id, rna_mask = make_rna_tensors(rna_chunk, args.enc_max_seq_len, device)
            for b0 in range(0, end - start, args.batch_size):
                b1 = min(b0 + args.batch_size, end - start)
                enc = model.enc(
                    x_value[b0:b1],
                    rna_gene_id[b0:b1],
                    return_encodings=True,
                    mask=rna_mask[b0:b1],
                )
                seq_out = model.translator(enc.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                for shard in shards:
                    gene_ids = torch.tensor(shard["gene_ids"], dtype=torch.long, device=device).unsqueeze(0).repeat(b1 - b0, 1)
                    mask = torch.tensor(shard["mask"], dtype=torch.bool, device=device).unsqueeze(0).repeat(b1 - b0, 1)
                    y_hat = model.dec(seq_out, gene_ids, mask=mask)
                    y_hat = torch.squeeze(y_hat)
                    if y_hat.ndim == 1:
                        y_hat = y_hat.unsqueeze(0)
                    n_shard = shard["end"] - shard["start"]
                    pred[start + b0 : start + b1, shard["start"] : shard["end"]] = (
                        y_hat[:, :n_shard].detach().cpu().numpy().astype(np.float32)
                    )
                pred.flush()
                del enc, seq_out
                if device == "cuda":
                    torch.cuda.empty_cache()
            done.write_text("done\n", encoding="utf-8")
            print(f"chunk {start}:{end} done", flush=True)
            del rna_chunk, x_value, rna_gene_id, rna_mask
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    write_cell_table(rna, out_dir / "cells.tsv")
    write_protein_table(proteins, out_dir / "protein_features.tsv")
    manifest = {
        "repo_dir": str(args.repo_dir),
        "checkpoint": str(args.checkpoint),
        "rna_h5ad": str(args.rna_h5ad),
        "protein_list": str(args.protein_list),
        "n_cells_input": n_cells,
        "n_cells_output": n_cells,
        "n_proteins": n_proteins,
        "n_shards": len(shards),
        "protein_shard_size": args.protein_shard_size,
        "matrix_file": "protein_predicted.npy",
        "protein_table": "protein_features.tsv",
        "cell_table": "cells.tsv",
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "enc_max_seq_len": args.enc_max_seq_len,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
