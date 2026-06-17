import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phospho_model_common import ensure_dir, save_json
from precompute_scgpt_embeddings import load_rna
from run_scgpt_frozen_anchor_probe import build_model, bin_expression
from train_rps6_gene_attention_rank import rps6_local_gene_table


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    print(f"[{now()}] {message}", flush=True)


def load_model_input_meta(path):
    meta = pd.read_csv(path / "cell_metadata.tsv", sep="\t", low_memory=False)
    meta = meta.copy()
    meta["source_row_index"] = np.arange(len(meta), dtype=np.int64)
    return meta


def unique_in_order(values):
    seen = set()
    out = []
    for value in values:
        value = str(value)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def make_forced_gene_list(vocab, ribosomal_cap):
    vocab_map = getattr(vocab, "token_to_idx", vocab)
    manifest = rps6_local_gene_table(vocab_map, ribosomal_cap=ribosomal_cap)
    genes = manifest["gene"].astype(str).tolist()
    return genes, manifest


def build_full_gene_order(model_genes, vocab, forced_genes):
    raw_vocab_genes = unique_in_order(g for g in model_genes if g in vocab)
    forced = [g for g in forced_genes if g in vocab]
    forced_set = set(forced)
    full_genes = forced + [g for g in raw_vocab_genes if g not in forced_set]
    gene_to_raw_col = {}
    for i, gene in enumerate(model_genes):
        if gene in vocab and gene not in gene_to_raw_col:
            gene_to_raw_col[gene] = i
    source_cols = np.asarray([gene_to_raw_col.get(g, -1) for g in full_genes], dtype=np.int64)
    present_positions = np.flatnonzero(source_cols >= 0)
    present_source_cols = source_cols[present_positions]
    gene_ids = np.asarray([vocab[g] for g in full_genes], dtype=np.int64)
    return full_genes, gene_ids, present_positions, present_source_cols


def materialize_chunk(x, row_start, row_stop, n_genes, present_positions, present_source_cols):
    out = np.zeros((row_stop - row_start, n_genes), dtype=np.float32)
    if len(present_source_cols) == 0:
        return out
    sub = x[row_start:row_stop, present_source_cols]
    if sparse.issparse(sub):
        sub = sub.toarray()
    out[:, present_positions] = np.asarray(sub, dtype=np.float32)
    return out


def tokenize_and_pad_forced(values, gene_ids, forced_count, max_len, vocab, pad_token, pad_value):
    pad_id = vocab[pad_token]
    cls_id = vocab["<cls>"]
    n = values.shape[0]
    genes_out = np.full((n, max_len), pad_id, dtype=np.int64)
    values_out = np.full((n, max_len), pad_value, dtype=np.float32)
    max_gene_tokens = max_len - 1
    if forced_count > max_gene_tokens:
        raise RuntimeError(f"forced genes={forced_count} exceeds token capacity={max_gene_tokens}")
    extra_capacity = max_gene_tokens - forced_count
    forced_idx = np.arange(forced_count, dtype=np.int64)
    for i in range(n):
        genes_out[i, 0] = cls_id
        values_out[i, 0] = 0.0
        if extra_capacity > 0:
            extra_nz = np.flatnonzero(values[i, forced_count:] > 0) + forced_count
            if len(extra_nz) > extra_capacity:
                order = np.argsort(values[i, extra_nz], kind="mergesort")[-extra_capacity:]
                extra_nz = extra_nz[order]
            selected = np.concatenate([forced_idx, extra_nz])
        else:
            selected = forced_idx
        length = len(selected) + 1
        genes_out[i, 1:length] = gene_ids[selected]
        values_out[i, 1:length] = values[i, selected]
    return {
        "genes": torch.from_numpy(genes_out).long(),
        "values": torch.from_numpy(values_out).float(),
    }


def encode_chunk(
    model,
    vocab,
    model_args,
    gene_ids,
    forced_count,
    x_full,
    seq_len,
    device,
    batch_size,
):
    x_full = np.log1p(np.maximum(np.asarray(x_full, dtype=np.float32), 0))
    values = bin_expression(x_full, model_args.get("n_bins", 51))
    tokenized = tokenize_and_pad_forced(
        values,
        gene_ids,
        forced_count=forced_count,
        max_len=seq_len,
        vocab=vocab,
        pad_token=model_args.get("pad_token", "<pad>"),
        pad_value=model_args.get("pad_value", -2),
    )
    src = tokenized["genes"].to(device)
    val = tokenized["values"].to(device)
    pad_id = vocab[model_args.get("pad_token", "<pad>")]
    src_key_padding_mask = src.eq(pad_id)
    with torch.no_grad():
        encoded = model.encode_batch(
            src,
            val,
            src_key_padding_mask,
            batch_size=batch_size,
            output_to_cpu=True,
            return_np=True,
        )
    if encoded.ndim != 3:
        raise RuntimeError(f"Expected token-level scGPT output, got shape {encoded.shape}")
    cell_emb = encoded[:, 0, :].astype(np.float32, copy=False)
    gene_emb = encoded[:, 1:, :]
    gene_mask = (~src_key_padding_mask[:, 1:].cpu().numpy()).astype(bool, copy=False)
    gene_ids_out = tokenized["genes"][:, 1:].numpy().astype(np.int32, copy=False)
    return cell_emb, gene_emb, gene_mask, gene_ids_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    ap.add_argument("--model-dir", default=r"D:\data\lsy\models\scgpt-human")
    ap.add_argument("--model-input-dir", default=r"01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_masked_multisite_v1")
    ap.add_argument("--output-dir", default=r"01_data\single_cell\intermediate\scgpt_token_embeddings\true_scgpt_rps6forced_v1")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=96)
    ap.add_argument("--max-cells-per-dataset", type=int, default=0)
    ap.add_argument("--max-seq-len", type=int, default=0)
    ap.add_argument("--token-dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ribosomal-cap", type=int, default=64)
    args = ap.parse_args()

    root = Path(args.root)
    input_dir = root / args.model_input_dir
    out_dir = ensure_dir(root / args.output_dir)
    model_input_meta = load_model_input_meta(input_dir)

    if args.datasets:
        dataset_order = args.datasets
    else:
        dataset_order = (
            model_input_meta.groupby("dataset_id")["source_row_index"]
            .min()
            .sort_values()
            .index.tolist()
        )

    selected_meta = []
    for dataset_id in dataset_order:
        part = model_input_meta.loc[model_input_meta["dataset_id"].eq(dataset_id)].copy()
        if args.max_cells_per_dataset and len(part) > args.max_cells_per_dataset:
            part = part.iloc[: args.max_cells_per_dataset].copy()
        selected_meta.append(part)
    out_meta = pd.concat(selected_meta, axis=0, ignore_index=True)
    out_meta.to_csv(out_dir / "cell_metadata.tsv", sep="\t", index=False)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, vocab, model_args = build_model(Path(args.model_dir), device)
    forced_genes, forced_manifest = make_forced_gene_list(vocab, args.ribosomal_cap)
    forced_manifest.to_csv(out_dir / "forced_rps6_local_gene_manifest.tsv", sep="\t", index=False)
    model_seq_len = int(model_args.get("max_seq_len", 1200))
    seq_len = int(args.max_seq_len) if args.max_seq_len else model_seq_len
    seq_len = min(seq_len, model_seq_len)
    token_len = seq_len - 1
    if len(forced_genes) > token_len:
        raise RuntimeError(f"forced genes={len(forced_genes)} exceeds token_len={token_len}")
    d_model = int(model_args["embsize"])
    token_dtype = np.float16 if args.token_dtype == "float16" else np.float32

    n_cells = len(out_meta)
    log(
        "creating forced token memmaps: "
        f"cells={n_cells} seq_len={seq_len} forced_genes={len(forced_genes)} "
        f"d_model={d_model} token_dtype={args.token_dtype}"
    )
    cell_mem = np.lib.format.open_memmap(out_dir / "cell_emb.npy", mode="w+", dtype=np.float32, shape=(n_cells, d_model))
    gene_mem = np.lib.format.open_memmap(out_dir / "gene_token_emb.npy", mode="w+", dtype=token_dtype, shape=(n_cells, token_len, d_model))
    mask_mem = np.lib.format.open_memmap(out_dir / "gene_token_mask.npy", mode="w+", dtype=np.bool_, shape=(n_cells, token_len))
    id_mem = np.lib.format.open_memmap(out_dir / "gene_token_ids.npy", mode="w+", dtype=np.int32, shape=(n_cells, token_len))

    cursor = 0
    dataset_summaries = []
    for dataset_id in dataset_order:
        part = out_meta.loc[out_meta["dataset_id"].eq(dataset_id)].copy()
        if part.empty:
            continue
        raw_cells, raw_genes, x = load_rna(root, dataset_id)
        raw_cells = np.asarray(raw_cells).astype(str)
        expected_cells = part["cell_id"].astype(str).to_numpy()
        local_rows = np.arange(len(part), dtype=np.int64)
        if not np.array_equal(raw_cells[local_rows], expected_cells):
            raise RuntimeError(f"cell order mismatch for {dataset_id}")

        model_genes = np.asarray([g.upper() if dataset_id == "vivo_seq_th17_2025" else g for g in raw_genes])
        full_genes, gene_ids, present_positions, present_source_cols = build_full_gene_order(
            model_genes,
            vocab,
            forced_genes,
        )
        if len(full_genes) == 0:
            raise RuntimeError(f"No scGPT vocab overlap for {dataset_id}")

        start_global = cursor
        stop_global = cursor + len(part)
        raw_gene_set = set(model_genes)
        forced_present_in_raw = int(sum(1 for g in forced_genes if g in raw_gene_set))
        log(
            f"{dataset_id}: cells={len(part)} full_genes={len(full_genes)} "
            f"forced_in_raw={forced_present_in_raw}/{len(forced_genes)} global={start_global}:{stop_global}"
        )
        for start in range(0, len(part), args.chunk_size):
            stop = min(start + args.chunk_size, len(part))
            rows = local_rows[start:stop]
            if not np.all(np.diff(rows) == 1):
                raise RuntimeError("selected rows are not contiguous; this script expects ordered dataset chunks")
            x_full = materialize_chunk(
                x,
                int(rows[0]),
                int(rows[-1]) + 1,
                len(full_genes),
                present_positions,
                present_source_cols,
            )
            cell_emb, gene_emb, gene_mask, gene_ids_out = encode_chunk(
                model,
                vocab,
                model_args,
                gene_ids,
                len(forced_genes),
                x_full,
                seq_len,
                device,
                args.batch_size,
            )
            g0 = cursor + start
            g1 = cursor + stop
            cell_mem[g0:g1] = cell_emb
            gene_mem[g0:g1] = gene_emb.astype(token_dtype, copy=False)
            mask_mem[g0:g1] = gene_mask
            id_mem[g0:g1] = gene_ids_out
            if stop == len(part) or stop % max(args.chunk_size * 10, 1) == 0:
                log(f"{dataset_id} encoded {stop}/{len(part)}")
        dataset_summaries.append(
            {
                "dataset_id": dataset_id,
                "n_cells": int(len(part)),
                "n_genes_used": int(len(full_genes)),
                "n_forced_genes": int(len(forced_genes)),
                "n_forced_genes_present_in_raw": int(forced_present_in_raw),
                "start": int(start_global),
                "stop": int(stop_global),
            }
        )
        cursor = stop_global

    cell_mem.flush()
    gene_mem.flush()
    mask_mem.flush()
    id_mem.flush()
    save_json(
        out_dir / "embedding_metadata.json",
        {
            "model_dir": str(args.model_dir),
            "model_input_dir": str(input_dir),
            "n_cells": int(n_cells),
            "seq_len": int(seq_len),
            "token_len": int(token_len),
            "d_model": int(d_model),
            "token_dtype": args.token_dtype,
            "forced_gene_mode": "rps6_local_genes_always_tokenized",
            "n_forced_genes": int(len(forced_genes)),
            "datasets": dataset_summaries,
        },
    )
    log(f"done: {out_dir}")


if __name__ == "__main__":
    main()
