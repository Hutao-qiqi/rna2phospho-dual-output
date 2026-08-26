from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch


def load_gene_id_map(repo_dir: Path):
    id_dir = repo_dir / "code" / "model" / "ID_dic"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rna-h5ad", required=True)
    parser.add_argument("--protein-list", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--n-cells", type=int, default=64)
    parser.add_argument("--n-proteins", type=int, default=1000)
    parser.add_argument("--enc-max-seq-len", type=int, default=20000)
    parser.add_argument("--dec-max-seq-len", type=int, default=1000)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    sys.path.insert(0, str(repo_dir / "code" / "model"))
    from utils import fix_SCDataset  # noqa: PLC0415

    hgs_to_entrez, entrez_to_myid = load_gene_id_map(repo_dir)
    proteins_all = [x.strip().upper() for x in Path(args.protein_list).read_text().splitlines() if x.strip()]
    proteins = proteins_all[: args.n_proteins]
    protein_myid = [gene_to_myid(x, hgs_to_entrez, entrez_to_myid) for x in proteins]
    if any(x == 0 for x in protein_myid):
        bad = [p for p, mid in zip(proteins, protein_myid) if mid == 0][:10]
        raise RuntimeError(f"unmapped protein query: {bad}")

    rna = sc.read_h5ad(args.rna_h5ad)[: args.n_cells].copy()
    if hasattr(rna.X, "toarray"):
        rna.X = rna.X.toarray().astype(np.float32)
    rna.var["my_Id"] = [gene_to_myid(g, hgs_to_entrez, entrez_to_myid) for g in rna.var_names]
    pro_x = np.tile(np.linspace(0.001, 1.0, len(proteins), dtype=np.float32), (rna.n_obs, 1))
    pro = ad.AnnData(X=pro_x, obs=rna.obs.copy(), var=pd.DataFrame(index=proteins))
    pro.var["my_Id"] = protein_myid

    dataset = fix_SCDataset(rna, pro, args.enc_max_seq_len, args.dec_max_seq_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.n_cells, drop_last=False)
    x, y = next(iter(loader))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = torch.load(args.checkpoint, map_location=torch.device(device), weights_only=False)
    except TypeError:
        model = torch.load(args.checkpoint, map_location=torch.device(device))
    model = model.to(device)
    model.eval()

    rna_gene_id = torch.tensor(x[:, 1].tolist()).long().to(device)
    protein_gene_id = torch.tensor(y[:, 1].tolist()).long().to(device)
    rna_mask = torch.tensor(x[:, 2].tolist()).bool().to(device)
    protein_mask = torch.tensor(y[:, 2].tolist()).bool().to(device)
    x_value = torch.tensor(x[:, 0].tolist(), dtype=torch.float32).to(device)

    with torch.no_grad():
        _, y_official = model(x_value, rna_gene_id, protein_gene_id, enc_mask=rna_mask, dec_mask=protein_mask)
        enc = model.enc(x_value, rna_gene_id, return_encodings=True, mask=rna_mask)
        seq_out = model.translator(enc.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        y_manual = model.dec(seq_out, protein_gene_id, mask=protein_mask)

    y_official = torch.squeeze(y_official).detach().cpu().numpy()
    y_manual = torch.squeeze(y_manual).detach().cpu().numpy()
    cached = np.load(Path(args.cache_dir) / "protein_predicted.npy", mmap_mode="r")[: args.n_cells, : args.n_proteins]
    diff = y_official - y_manual
    cache_diff = y_official - cached

    def stats(name: str, arr: np.ndarray) -> dict:
        return {
            f"{name}_min": float(np.nanmin(arr)),
            f"{name}_p01": float(np.nanpercentile(arr, 1)),
            f"{name}_median": float(np.nanmedian(arr)),
            f"{name}_p99": float(np.nanpercentile(arr, 99)),
            f"{name}_max": float(np.nanmax(arr)),
            f"{name}_mean": float(np.nanmean(arr)),
            f"{name}_sd": float(np.nanstd(arr)),
        }

    out = {
        "rna_h5ad": args.rna_h5ad,
        "cache_dir": args.cache_dir,
        "n_cells": int(args.n_cells),
        "n_proteins": int(args.n_proteins),
        "official_manual_max_abs_diff": float(np.nanmax(np.abs(diff))),
        "official_cache_max_abs_diff": float(np.nanmax(np.abs(cache_diff))),
        "official_cache_mean_abs_diff": float(np.nanmean(np.abs(cache_diff))),
    }
    out.update(stats("official", y_official))
    out.update(stats("manual", y_manual))
    out.update(stats("cached", np.asarray(cached)))
    out.update(stats("official_minus_cached", cache_diff))
    Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
