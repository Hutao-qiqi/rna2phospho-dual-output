import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phospho_model_common import ensure_dir, save_json
from precompute_scfoundation_embeddings_multidomain import clean_gene_symbols, load_multidomain_rna
from run_scgpt_frozen_anchor_probe import build_model, bin_expression, tokenize_and_pad


def encode_batch(model, vocab, model_args, genes, x_batch, device, batch_size):
    if sparse.issparse(x_batch):
        x_batch = x_batch.toarray()
    x_batch = np.log1p(np.maximum(np.asarray(x_batch, dtype=np.float32), 0))
    values = bin_expression(x_batch, model_args.get("n_bins", 51))
    gene_ids = np.asarray([vocab[g] for g in genes], dtype=np.int64)
    tokenized = tokenize_and_pad(
        values,
        gene_ids,
        max_len=min(model_args.get("max_seq_len", 1200), len(gene_ids) + 1),
        vocab=vocab,
        pad_token=model_args.get("pad_token", "<pad>"),
        pad_value=model_args.get("pad_value", -2),
    )
    src = tokenized["genes"].to(device)
    val = tokenized["values"].to(device)
    mask = src.eq(vocab[model_args.get("pad_token", "<pad>")])
    with torch.no_grad():
        emb = model.encode_batch(src, val, mask, batch_size=batch_size, output_to_cpu=True, return_np=True)
    if emb.ndim == 3:
        emb = emb[:, 0, :]
    return np.asarray(emb, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    parser.add_argument("--model-dir", default=r"D:\data\lsy\models\scgpt-human")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "iccite_seq_tcell_2025",
            "qurie_seq_bjab_2021",
            "gse300551_iccite_plex_kinase_2025",
            "signal_seq_gse256403_hela_2024",
            "signal_seq_gse256404_pdo_caf_2024",
            "phospho_seq_blair_2025_phospho_multi",
            "vivo_seq_th17_2025",
        ],
    )
    parser.add_argument("--output-dir", default=r"01_data\single_cell\intermediate\scgpt_embeddings\frozen_multidomain_v1")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = ensure_dir(root / args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, vocab, model_args = build_model(Path(args.model_dir), device)
    rng = np.random.default_rng(args.seed)

    for dataset_id in args.datasets:
        out = ensure_dir(out_root / dataset_id)
        emb_path = out / "embeddings.npy"
        meta_path = out / "embedding_metadata.json"
        if args.skip_existing and emb_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("complete", False):
                print(f"{dataset_id} exists; skip", flush=True)
                continue

        cells, genes, x = load_multidomain_rna(root, dataset_id)
        genes = clean_gene_symbols(np.asarray(genes, dtype=str))
        if dataset_id == "vivo_seq_th17_2025":
            genes = np.asarray([g.upper() for g in genes], dtype=str)
        keep = np.asarray([g in vocab for g in genes])
        kept_genes = genes[keep]
        if int(keep.sum()) == 0:
            raise RuntimeError(f"No scGPT vocab overlap for {dataset_id}")
        x = x[:, keep] if sparse.issparse(x) else np.asarray(x)[:, keep]

        if args.max_cells and len(cells) > args.max_cells:
            idx = np.sort(rng.choice(len(cells), size=args.max_cells, replace=False))
        else:
            idx = np.arange(len(cells))
        cells_out = np.asarray(cells, dtype=str)[idx]
        x = x[idx]

        emb_parts = []
        for start in range(0, len(cells_out), args.chunk_size):
            stop = min(start + args.chunk_size, len(cells_out))
            emb_parts.append(encode_batch(model, vocab, model_args, kept_genes, x[start:stop], device, args.batch_size))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"{dataset_id} encoded {stop}/{len(cells_out)}", flush=True)
        emb = np.vstack(emb_parts).astype(np.float32)

        np.save(emb_path, emb)
        (out / "barcodes.tsv").write_text("\n".join(cells_out.tolist()) + "\n", encoding="utf-8")
        (out / "genes_used.tsv").write_text("\n".join(kept_genes.astype(str).tolist()) + "\n", encoding="utf-8")
        save_json(
            meta_path,
            {
                "dataset_id": dataset_id,
                "n_cells": int(len(cells_out)),
                "n_input_cells_total": int(len(cells)),
                "n_input_genes": int(len(genes)),
                "n_genes_used": int(len(kept_genes)),
                "embedding_dim": int(emb.shape[1]),
                "model_dir": str(args.model_dir),
                "max_cells": int(args.max_cells),
                "complete": True,
            },
        )


if __name__ == "__main__":
    main()
