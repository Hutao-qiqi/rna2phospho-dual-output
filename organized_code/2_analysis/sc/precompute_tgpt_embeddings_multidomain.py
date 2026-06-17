import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_DATASETS = [
    "iccite_seq_tcell_2025",
    "phospho_seq_blair_2025_phospho_multi",
    "qurie_seq_bjab_2021",
    "vivo_seq_th17_2025",
    "gse300551_iccite_plex_kinase_2025",
    "signal_seq_gse256403_hela_2024",
    "signal_seq_gse256404_pdo_caf_2024",
]


def top_gene_sentences(x, gene_names, max_len):
    sentences = []
    if sp.issparse(x):
        x = x.tocsr()
        indptr = x.indptr
        indices = x.indices
        data = x.data
        for i in range(x.shape[0]):
            lo, hi = indptr[i], indptr[i + 1]
            row_idx = indices[lo:hi]
            row_val = data[lo:hi]
            if len(row_idx) == 0:
                sentences.append("")
                continue
            keep = min(max_len, len(row_idx))
            order = np.argpartition(-row_val, keep - 1)[:keep]
            order = order[np.argsort(-row_val[order])]
            sentences.append(" ".join(gene_names[row_idx[order]].tolist()))
    else:
        arr = np.asarray(x)
        for row in arr:
            nz = np.flatnonzero(row > 0)
            if len(nz) == 0:
                sentences.append("")
                continue
            vals = row[nz]
            keep = min(max_len, len(nz))
            order = np.argpartition(-vals, keep - 1)[:keep]
            order = order[np.argsort(-vals[order])]
            sentences.append(" ".join(gene_names[nz[order]].tolist()))
    return sentences


@torch.inference_mode()
def embed_dataset(adata_path, out_path, model, tokenizer, device, batch_size, max_len):
    adata = ad.read_h5ad(adata_path)
    genes = np.asarray(adata.var_names.astype(str))
    n_cells = adata.n_obs
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first_param = next(model.parameters())
    emb_dim = int(first_param.shape[-1])
    emb = np.lib.format.open_memmap(
        str(out_path.with_suffix(".embeddings.npy")),
        mode="w+",
        dtype=np.float32,
        shape=(n_cells, emb_dim),
    )

    for start in tqdm(range(0, n_cells, batch_size), desc=adata_path.stem):
        end = min(start + batch_size, n_cells)
        sentences = top_gene_sentences(adata.X[start:end], genes, max_len)
        encoded = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        output = model(**encoded)
        hidden = output.last_hidden_state
        mask = encoded.get("attention_mask")
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask_f = mask.unsqueeze(-1).to(hidden.dtype)
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            pooled = (hidden * mask_f).sum(dim=1) / denom
        emb[start:end] = pooled.detach().cpu().numpy().astype(np.float32)
        emb.flush()

    emb_arr = np.asarray(emb).copy()
    del emb

    out_adata = ad.AnnData(
        X=sp.csr_matrix((n_cells, 0), dtype=np.float32),
        obs=adata.obs.copy(),
    )
    out_adata.obsm["X_tgpt"] = emb_arr
    out_adata.write_h5ad(out_path, compression="gzip")
    out_path.with_suffix(".embeddings.npy").unlink(missing_ok=True)
    return {
        "n_cells": int(n_cells),
        "embedding_dim": int(out_adata.obsm["X_tgpt"].shape[1]),
        "output_h5ad": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name", default="lixiangchun/transcriptome-gpt-1024-8-16-64")
    parser.add_argument("--cache-dir", default=r"D:\data\lsy\models\tgpt_hf_cache")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    input_dir = Path(args.input_dir) if args.input_dir else root / "01_data" / "single_cell" / "intermediate" / "foundation_model_h5ad_inputs_v1"
    output_dir = Path(args.output_dir) if args.output_dir else root / "01_data" / "single_cell" / "intermediate" / "foundation_model_embeddings" / "tgpt_top64_multidomain_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    model = AutoModel.from_pretrained(args.model_name, cache_dir=args.cache_dir, trust_remote_code=True)
    model.to(device)
    model.eval()

    datasets = args.datasets if args.datasets else DEFAULT_DATASETS
    rows = []
    for dataset_id in datasets:
        h5ad = input_dir / f"{dataset_id}.h5ad"
        out_h5ad = output_dir / dataset_id / f"{dataset_id}_tgpt_adata.h5ad"
        if args.skip_existing and out_h5ad.exists():
            status = "skipped_existing"
            adata = ad.read_h5ad(out_h5ad)
            info = {
                "n_cells": int(adata.n_obs),
                "embedding_dim": int(adata.obsm["X_tgpt"].shape[1]),
                "output_h5ad": str(out_h5ad),
            }
        else:
            status = "completed"
            info = embed_dataset(h5ad, out_h5ad, model, tokenizer, device, args.batch_size, args.max_len)
        rows.append({
            "dataset_id": dataset_id,
            "status": status,
            "model_name": args.model_name,
            "max_len": int(args.max_len),
            "output_h5ad": info["output_h5ad"],
            "n_cells": info["n_cells"],
            "embedding_dim": info["embedding_dim"],
        })
        pd.DataFrame(rows).to_csv(output_dir / "tgpt_embedding_manifest.tsv", sep="\t", index=False)

    (output_dir / "tgpt_embedding_manifest.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
