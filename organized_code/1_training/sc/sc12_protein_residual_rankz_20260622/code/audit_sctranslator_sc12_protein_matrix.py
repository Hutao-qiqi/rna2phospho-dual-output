import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def qstats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    qs = np.quantile(x, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "min": float(qs[0]),
        "q01": float(qs[1]),
        "q05": float(qs[2]),
        "q25": float(qs[3]),
        "median": float(qs[4]),
        "q75": float(qs[5]),
        "q95": float(qs[6]),
        "q99": float(qs[7]),
        "max": float(qs[8]),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x)),
    }


def spearman_np(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    ap.add_argument("--protein-cache-dir", required=True)
    ap.add_argument("--h5ad-dir", default=r"01_data\single_cell\intermediate\foundation_model_h5ad_inputs_v1")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sample-cells-per-dataset", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=682)
    args = ap.parse_args()

    root = Path(args.root)
    cache_dir = root / args.protein_cache_dir
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    protein = np.load(cache_dir / "protein_predicted.npy", mmap_mode="r")
    meta = pd.read_csv(cache_dir / "cell_metadata.tsv", sep="\t", low_memory=False)
    ptab = pd.read_csv(cache_dir / "protein_features.tsv", sep="\t")
    if len(meta) != protein.shape[0]:
        raise ValueError(f"cell metadata rows {len(meta)} != protein rows {protein.shape[0]}")

    rng = np.random.default_rng(args.seed)
    rows = []
    global_sample_idx = []
    for ds, sub in meta.groupby("dataset_id", sort=False):
        idx = sub.index.to_numpy()
        if len(idx) > args.sample_cells_per_dataset:
            idx = rng.choice(idx, size=args.sample_cells_per_dataset, replace=False)
        global_sample_idx.extend(idx.tolist())
        mat = np.asarray(protein[idx, :], dtype=np.float32)
        rows.append({
            "dataset_id": ds,
            "n_cells_total": int((meta["dataset_id"] == ds).sum()),
            "n_cells_sampled": int(len(idx)),
            "n_proteins": int(protein.shape[1]),
            "nan_fraction": float(np.isnan(mat).mean()),
            "finite_fraction": float(np.isfinite(mat).mean()),
            **{f"value_{k}": v for k, v in qstats(mat.ravel()).items()},
            **{f"cell_mean_{k}": v for k, v in qstats(np.nanmean(mat, axis=1)).items()},
            **{f"cell_sd_{k}": v for k, v in qstats(np.nanstd(mat, axis=1)).items()},
            **{f"protein_mean_{k}": v for k, v in qstats(np.nanmean(mat, axis=0)).items()},
            **{f"protein_sd_{k}": v for k, v in qstats(np.nanstd(mat, axis=0)).items()},
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "sctranslator_sc12_protein_matrix_distribution_by_dataset.tsv", sep="\t", index=False)

    global_idx = np.asarray(sorted(set(global_sample_idx)), dtype=np.int64)
    sample = np.asarray(protein[global_idx, :], dtype=np.float32)
    protein_qc = pd.DataFrame({
        "protein_index": np.arange(protein.shape[1]),
        "protein_id": ptab["protein_id"].astype(str).to_numpy() if "protein_id" in ptab.columns else ptab.iloc[:, 0].astype(str).to_numpy(),
        "mean": np.nanmean(sample, axis=0),
        "sd": np.nanstd(sample, axis=0),
        "min": np.nanmin(sample, axis=0),
        "max": np.nanmax(sample, axis=0),
        "nan_fraction": np.isnan(sample).mean(axis=0),
    })
    protein_qc["dynamic_range"] = protein_qc["max"] - protein_qc["min"]
    protein_qc.to_csv(out_dir / "sctranslator_sc12_protein_qc_sampled.tsv", sep="\t", index=False)

    # Check same-gene RNA correlation for matched h5ad inputs when scanpy/anndata is available.
    rna_rows = []
    try:
        import anndata as ad
        import scipy.sparse as sp
        h5ad_dir = root / args.h5ad_dir
        protein_name_to_idx = {str(v).upper(): i for i, v in enumerate(protein_qc["protein_id"].tolist())}
        for ds, sub in meta.groupby("dataset_id", sort=False):
            h5ad_path = h5ad_dir / f"{ds}.h5ad"
            if not h5ad_path.exists():
                continue
            adata = ad.read_h5ad(h5ad_path, backed="r")
            genes = [str(g).upper() for g in adata.var_names]
            gene_to_idx = {g: i for i, g in enumerate(genes)}
            shared = sorted(set(gene_to_idx).intersection(protein_name_to_idx))
            if len(shared) > 300:
                shared = shared[:300]
            idx_cells = sub.index.to_numpy()
            local_n = min(args.sample_cells_per_dataset, len(idx_cells))
            chosen_global = rng.choice(idx_cells, size=local_n, replace=False) if len(idx_cells) > local_n else idx_cells
            # Merged cache is ordered by global meta. Per-dataset h5ad is ordered within dataset.
            local_pos_lookup = {int(g): i for i, g in enumerate(sub.index.to_numpy())}
            local_pos = [local_pos_lookup[int(g)] for g in chosen_global]
            for gene in shared:
                gi = gene_to_idx[gene]
                pi = protein_name_to_idx[gene]
                x = adata.X[local_pos, gi]
                if sp.issparse(x):
                    x = x.toarray()
                x = np.asarray(x).reshape(-1).astype(np.float32)
                y = np.asarray(protein[chosen_global, pi], dtype=np.float32)
                rna_rows.append({
                    "dataset_id": ds,
                    "gene": gene,
                    "n": int(local_n),
                    "rna_vs_predicted_protein_spearman": spearman_np(x, y),
                    "rna_mean": float(np.nanmean(x)),
                    "rna_sd": float(np.nanstd(x)),
                    "protein_mean": float(np.nanmean(y)),
                    "protein_sd": float(np.nanstd(y)),
                })
            adata.file.close()
    except Exception as exc:
        rna_rows.append({"dataset_id": "ERROR", "gene": "ERROR", "n": 0, "rna_vs_predicted_protein_spearman": np.nan, "error": str(exc)})
    rna_df = pd.DataFrame(rna_rows)
    rna_df.to_csv(out_dir / "sctranslator_sc12_same_gene_rna_correlation_sampled.tsv", sep="\t", index=False)

    payload = {
        "protein_shape": [int(protein.shape[0]), int(protein.shape[1])],
        "n_datasets": int(meta["dataset_id"].nunique()),
        "tables": {
            "distribution": str(out_dir / "sctranslator_sc12_protein_matrix_distribution_by_dataset.tsv"),
            "protein_qc": str(out_dir / "sctranslator_sc12_protein_qc_sampled.tsv"),
            "same_gene_rna_correlation": str(out_dir / "sctranslator_sc12_same_gene_rna_correlation_sampled.tsv"),
        },
    }
    with (out_dir / "sctranslator_sc12_protein_matrix_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))
    print("\nDistribution")
    print(summary[["dataset_id", "n_cells_total", "value_median", "value_sd", "protein_sd_median", "protein_sd_q01", "protein_sd_q99", "nan_fraction"]].to_string(index=False))
    if not rna_df.empty and "error" not in rna_df.columns:
        print("\nSame gene RNA correlation summary")
        print(rna_df.groupby("dataset_id")["rna_vs_predicted_protein_spearman"].median().reset_index().to_string(index=False))
    else:
        print("\nSame gene RNA correlation failed or unavailable")
        print(rna_df.to_string(index=False))


if __name__ == "__main__":
    main()
