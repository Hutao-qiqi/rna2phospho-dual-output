import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phospho_model_common import dataset_dir, ensure_dir, load_mtx, read_dense_tsv, read_lines, save_json
from precompute_scgpt_embeddings import load_rna


def load_scfoundation_model(code_dir: Path, weight_path: Path, device: torch.device):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    old_cwd = Path.cwd()
    try:
        os.chdir(code_dir)
        from load import load_model_frommmf

        model, config = load_model_frommmf(str(weight_path), key="cell")
    finally:
        os.chdir(old_cwd)
    model.eval()
    model.to(device)
    return model, config


def load_scfoundation_gene_list(code_dir: Path) -> list[str]:
    path = code_dir / "OS_scRNA_gene_index.19264.tsv"
    table = pd.read_csv(path, sep="\t")
    if "gene_name" not in table.columns:
        raise ValueError(f"gene_name column missing in {path}")
    return table["gene_name"].astype(str).tolist()


def build_gene_projection(input_genes: np.ndarray, target_genes: list[str]):
    source = {}
    for i, gene in enumerate(input_genes.astype(str)):
        key = gene.upper()
        source.setdefault(key, []).append(i)

    rows = []
    cols = []
    for j, gene in enumerate(target_genes):
        for i in source.get(str(gene).upper(), []):
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    proj = sparse.csr_matrix((data, (rows, cols)), shape=(len(input_genes), len(target_genes)), dtype=np.float32)
    return proj, len(set(rows)), len(set(cols))


def align_to_scfoundation_genes(x, genes: np.ndarray, target_genes: list[str]):
    proj, n_source_used, n_target_hit = build_gene_projection(genes, target_genes)
    if sparse.issparse(x):
        aligned = x.tocsr() @ proj
    else:
        aligned = sparse.csr_matrix(np.asarray(x, dtype=np.float32)) @ proj
    return aligned.tocsr().astype(np.float32), n_source_used, n_target_hit


def normalize_singlecell_counts(x_batch: np.ndarray, highres_value: float) -> np.ndarray:
    x_batch = np.maximum(x_batch.astype(np.float32, copy=False), 0.0)
    total = x_batch.sum(axis=1, keepdims=True)
    safe_total = np.maximum(total, 1.0)
    norm = np.log1p(x_batch / safe_total * 10000.0).astype(np.float32)
    total_log10 = np.log10(safe_total).astype(np.float32)
    highres = np.full((x_batch.shape[0], 1), highres_value, dtype=np.float32)
    return np.concatenate([norm, highres, total_log10], axis=1)


def build_value_labels(pretrain_gene_x: torch.Tensor, max_gene_tokens: int) -> torch.Tensor:
    gene_values = pretrain_gene_x[:, :-2]
    gene_labels = gene_values > 0
    if max_gene_tokens and max_gene_tokens > 0 and max_gene_tokens < gene_values.shape[1]:
        k = min(max_gene_tokens, gene_values.shape[1])
        top_values, top_index = torch.topk(gene_values, k=k, dim=1)
        top_labels = torch.zeros_like(gene_labels)
        top_labels.scatter_(1, top_index, top_values > 0)
        gene_labels = gene_labels & top_labels
    special_labels = pretrain_gene_x[:, -2:] > 0
    return torch.cat([gene_labels, special_labels], dim=1)


@torch.inference_mode()
def encode_batch(model, config: dict, x_batch, device: torch.device, highres_value: float, max_gene_tokens: int) -> np.ndarray:
    from load import gatherData

    if sparse.issparse(x_batch):
        x_batch = x_batch.toarray()
    prepared = normalize_singlecell_counts(np.asarray(x_batch, dtype=np.float32), highres_value)
    pretrain_gene_x = torch.from_numpy(prepared).to(device)
    data_gene_ids = torch.arange(pretrain_gene_x.shape[1], device=device).repeat(pretrain_gene_x.shape[0], 1)
    value_labels = build_value_labels(pretrain_gene_x, max_gene_tokens)

    x, x_padding = gatherData(pretrain_gene_x, value_labels, config["pad_token_id"])
    position_gene_ids, _ = gatherData(data_gene_ids, value_labels, config["pad_token_id"])

    x = model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
    x = x + model.pos_emb(position_gene_ids)
    geneemb = model.encoder(x, x_padding)

    geneemb1 = geneemb[:, -1, :]
    geneemb2 = geneemb[:, -2, :]
    gene_tokens = geneemb[:, :-2, :]
    geneemb3, _ = torch.max(gene_tokens, dim=1)
    geneemb4 = torch.mean(gene_tokens, dim=1)
    merged = torch.cat([geneemb1, geneemb2, geneemb3, geneemb4], dim=1)
    return merged.detach().cpu().numpy().astype(np.float32)


def encode_batch_with_fallback(model, config: dict, x_batch, device: torch.device, highres_value: float, max_gene_tokens: int) -> np.ndarray:
    try:
        return encode_batch(model, config, x_batch, device, highres_value, max_gene_tokens)
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        n = x_batch.shape[0]
        if n <= 1:
            raise
        mid = n // 2
        print(f"CUDA OOM at batch={n}; retry {mid}+{n - mid}", flush=True)
        left = encode_batch_with_fallback(model, config, x_batch[:mid], device, highres_value, max_gene_tokens)
        right = encode_batch_with_fallback(model, config, x_batch[mid:], device, highres_value, max_gene_tokens)
        return np.vstack([left, right]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    ap.add_argument("--code-dir", default=r"D:\data\lsy\repos\scFoundation_model")
    ap.add_argument("--weight-path", default=r"D:\data\lsy\models\scfoundation\models.ckpt")
    ap.add_argument("--datasets", nargs="+", default=[
        "iccite_seq_tcell_2025",
        "qurie_seq_bjab_2021",
        "phospho_seq_blair_2025_phospho_multi",
        "vivo_seq_th17_2025",
    ])
    ap.add_argument("--output-dir", default=r"01_data\single_cell\intermediate\scfoundation_embeddings\frozen_cell_all_v1")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--highres", type=float, default=4.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-gene-tokens", type=int, default=2048)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    root = Path(args.root)
    code_dir = Path(args.code_dir)
    weight_path = Path(args.weight_path)
    out_root = ensure_dir(root / args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    rng = np.random.default_rng(args.seed)

    target_genes = load_scfoundation_gene_list(code_dir)
    model, config = load_scfoundation_model(code_dir, weight_path, device)

    for dataset_id in args.datasets:
        cells, genes, x = load_rna(root, dataset_id)
        genes = np.asarray([g.upper() if dataset_id == "vivo_seq_th17_2025" else g for g in genes])
        if args.max_cells and len(cells) > args.max_cells:
            idx = np.sort(rng.choice(len(cells), size=args.max_cells, replace=False))
        else:
            idx = np.arange(len(cells))
        idx = idx[args.shard_index::args.num_shards]
        cells_out = np.asarray(cells)[idx]
        out = ensure_dir(out_root / dataset_id)
        meta_path = out / "embedding_metadata.json"
        emb_path = out / "embeddings.npy"
        if args.skip_existing and meta_path.exists() and emb_path.exists():
            print(f"{dataset_id} shard {args.shard_index}/{args.num_shards} exists; skip", flush=True)
            continue
        proj, n_source_used, n_target_hit = build_gene_projection(genes, target_genes)
        x = x.tocsr() if sparse.issparse(x) else sparse.csr_matrix(np.asarray(x, dtype=np.float32))

        emb_store = None
        emb_dim = None
        for start in range(0, len(cells_out), args.batch_size):
            stop = min(start + args.batch_size, len(cells_out))
            batch_rows = idx[start:stop]
            x_aligned_batch = (x[batch_rows] @ proj).tocsr().astype(np.float32)
            emb_batch = encode_batch_with_fallback(model, config, x_aligned_batch, device, args.highres, args.max_gene_tokens)
            if emb_store is None:
                emb_dim = int(emb_batch.shape[1])
                emb_store = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(len(cells_out), emb_dim))
            emb_store[start:stop, :] = emb_batch
            emb_store.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"{dataset_id} encoded {stop}/{len(cells_out)}", flush=True)
        if emb_store is None:
            emb_dim = 0
            np.save(emb_path, np.empty((0, 0), dtype=np.float32))
        else:
            del emb_store

        (out / "barcodes.tsv").write_text("\n".join(cells_out.astype(str)) + "\n", encoding="utf-8")
        (out / "genes_used.tsv").write_text("\n".join(target_genes) + "\n", encoding="utf-8")
        save_json(
            meta_path,
            {
                "dataset_id": dataset_id,
                "n_cells": int(len(cells_out)),
                "n_input_genes": int(len(genes)),
                "n_source_genes_used": int(n_source_used),
                "n_scfoundation_genes_hit": int(n_target_hit),
                "n_scfoundation_genes_total": int(len(target_genes)),
                "embedding_dim": int(emb_dim),
                "model": "scFoundation_cell",
                "weight_path": str(weight_path),
                "code_dir": str(code_dir),
                "highres": float(args.highres),
                "max_gene_tokens": int(args.max_gene_tokens),
                "max_cells": int(args.max_cells),
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
                "complete": True,
            },
        )


if __name__ == "__main__":
    main()
