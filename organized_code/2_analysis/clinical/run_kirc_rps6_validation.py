import argparse
import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from scipy import sparse
from scipy.stats import kruskal, pearsonr, spearmanr


PATHWAY_GENESETS = {
    "mTOR_S6_score": [
        "MTOR",
        "RPTOR",
        "RHEB",
        "AKT1",
        "AKT2",
        "AKT3",
        "RPS6KB1",
        "RPS6KB2",
        "EIF4EBP1",
        "RPS6",
        "PIK3CA",
        "PIK3CB",
        "PDPK1",
        "TSC1",
        "TSC2",
    ],
    "cell_cycle_score": [
        "MKI67",
        "TOP2A",
        "PCNA",
        "MCM2",
        "MCM3",
        "MCM4",
        "MCM5",
        "MCM6",
        "CCNA2",
        "CCNB1",
        "CDK1",
        "CDC20",
    ],
    "hypoxia_score": [
        "VEGFA",
        "CA9",
        "EGLN3",
        "BNIP3",
        "LDHA",
        "SLC2A1",
        "PDK1",
        "NDRG1",
    ],
    "angiogenesis_score": [
        "PECAM1",
        "KDR",
        "FLT1",
        "VWF",
        "ENG",
        "ANGPT2",
        "ESAM",
        "EMCN",
    ],
    "MYC_targets_score": [
        "MYC",
        "NPM1",
        "NCL",
        "MCM2",
        "MCM4",
        "MCM5",
        "PCNA",
        "LDHA",
        "CAD",
        "ODC1",
    ],
    "EMT_score": [
        "VIM",
        "FN1",
        "SNAI1",
        "SNAI2",
        "ZEB1",
        "ZEB2",
        "COL1A1",
        "COL1A2",
        "ITGA5",
        "CDH2",
    ],
    "interferon_stress_score": [
        "ISG15",
        "IFIT1",
        "IFIT2",
        "IFIT3",
        "MX1",
        "OAS1",
        "STAT1",
        "IRF7",
        "DDIT3",
        "HSPA1A",
        "HSP90AA1",
    ],
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def maybe_decompress_h5ad(gz_path: Path, h5ad_path: Path, force: bool = False) -> Path:
    if h5ad_path.exists() and not force:
        return h5ad_path
    ensure_dir(h5ad_path.parent)
    tmp_path = h5ad_path.with_suffix(h5ad_path.suffix + ".tmp")
    log(f"decompress {gz_path} -> {h5ad_path}")
    with gzip.open(gz_path, "rb") as src, tmp_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
    tmp_path.replace(h5ad_path)
    return h5ad_path


def clean_symbols(values) -> np.ndarray:
    out = []
    for value in pd.Series(values).astype(str).tolist():
        gene = value.strip()
        for prefix in ("GRCh38_", "GRCh37_", "hg38_", "hg19_", "mm10_", "mm9_"):
            if gene.startswith(prefix):
                gene = gene[len(prefix) :]
                break
        out.append(gene)
    return np.asarray(out, dtype=object)


def choose_gene_symbols(var: pd.DataFrame, var_names) -> tuple[np.ndarray, str]:
    candidates = [("var_names", clean_symbols(var_names))]
    for col in var.columns:
        low = str(col).lower()
        if low in {
            "gene",
            "genes",
            "gene_name",
            "gene_names",
            "gene_symbol",
            "gene_symbols",
            "symbol",
            "symbols",
            "feature_name",
            "feature_names",
            "name",
        }:
            candidates.append((str(col), clean_symbols(var[col].values)))
    scored = []
    for name, genes in candidates:
        upper = pd.Series(genes).astype(str).str.upper()
        has_rps6 = int((upper == "RPS6").any())
        non_empty = int((upper.str.len() > 0).sum())
        hgnc_like = int(upper.str.match(r"^[A-Z0-9.-]+$").sum())
        scored.append((has_rps6, hgnc_like, non_empty, name, genes))
    scored.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    best = scored[0]
    return best[4], best[3]


def choose_expression_source(adata: ad.AnnData):
    layer_priority = [
        "counts",
        "count",
        "raw_counts",
        "raw count",
        "umi",
        "UMI",
        "spliced",
    ]
    for layer in layer_priority:
        if layer in adata.layers:
            return adata.layers[layer], adata.var.copy(), adata.var_names.copy(), f"layer:{layer}"
    if adata.raw is not None:
        return adata.raw.X, adata.raw.var.copy(), adata.raw.var_names.copy(), "raw"
    return adata.X, adata.var.copy(), adata.var_names.copy(), "X"


def to_csr_float(x):
    if sparse.issparse(x):
        return x.tocsr().astype(np.float32)
    return sparse.csr_matrix(np.asarray(x, dtype=np.float32))


def sample_matrix_values(x, max_rows: int = 1024) -> np.ndarray:
    n = x.shape[0]
    rows = np.linspace(0, n - 1, min(n, max_rows), dtype=int)
    sub = x[rows]
    if sparse.issparse(sub):
        vals = sub.data
    else:
        vals = np.asarray(sub).ravel()
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if len(vals) > 10000:
        vals = vals[:10000]
    return vals.astype(np.float64, copy=False)


def infer_counts_like(x) -> bool:
    vals = sample_matrix_values(x)
    if len(vals) == 0:
        return True
    integer_like = np.mean(np.isclose(vals, np.round(vals), atol=1e-4))
    return bool(np.nanmax(vals) > 50 or integer_like > 0.98)


def pick_obs_column(obs: pd.DataFrame, exact_names, contains_names=(), max_unique=200):
    normalized = {str(c).lower().replace(" ", "_").replace("-", "_"): c for c in obs.columns}
    for name in exact_names:
        key = name.lower().replace(" ", "_").replace("-", "_")
        if key in normalized:
            return normalized[key]
    for c in obs.columns:
        low = str(c).lower()
        if any(token in low for token in contains_names):
            nunique = obs[c].astype(str).nunique(dropna=True)
            if 1 < nunique <= max_unique:
                return c
    return None


def build_metadata(adata: ad.AnnData, dataset_id: str) -> tuple[pd.DataFrame, dict]:
    obs = adata.obs.copy()
    meta = pd.DataFrame(index=obs.index)
    meta["cell_id"] = pd.Series(adata.obs_names.astype(str), index=obs.index).values
    meta["dataset_id"] = dataset_id

    patient_col = pick_obs_column(
        obs,
        ["patient_id", "patient", "donor_id", "donor", "case_id", "case", "sample_id", "sample", "orig.ident"],
        ["patient", "donor", "sample"],
    )
    cell_type_col = pick_obs_column(
        obs,
        ["cell_type", "celltype", "cell_type_major", "major_cell_type", "major_celltype", "annotation", "cell_annotation", "final_annotation"],
        ["celltype", "cell_type", "annotation"],
    )
    tissue_col = pick_obs_column(
        obs,
        ["tissue", "tissue_type", "condition", "sample_type", "source", "tumor_normal", "tumour_normal", "region"],
        ["tissue", "condition", "tumor", "tumour", "normal", "region"],
    )
    malignant_col = pick_obs_column(
        obs,
        ["malignant_status", "malignant", "malignancy", "cnv_status", "copykat", "infercnv", "tumor_status", "tumour_status"],
        ["malignan", "copykat", "infercnv", "cnv"],
    )

    meta["patient_id"] = obs[patient_col].astype(str).values if patient_col else "unknown"
    meta["cell_type"] = obs[cell_type_col].astype(str).values if cell_type_col else "unknown"
    meta["tissue"] = obs[tissue_col].astype(str).values if tissue_col else "unknown"

    if malignant_col:
        raw = obs[malignant_col].astype(str).str.lower()
        status = np.full(len(raw), "unknown", dtype=object)
        status[raw.str.contains("malignant|tumou?r|cancer|aneuploid|tumor", regex=True, na=False).values] = "malignant"
        status[raw.str.contains("non|normal|diploid|benign|healthy", regex=True, na=False).values] = "non_malignant"
        unresolved = status == "unknown"
        status[unresolved] = obs[malignant_col].astype(str).values[unresolved]
        meta["malignant_status"] = status
        malignant_source = str(malignant_col)
    else:
        ct = meta["cell_type"].astype(str).str.lower()
        tissue = meta["tissue"].astype(str).str.lower()
        epithelial = ct.str.contains("epithelial|tumou?r|cancer|malignant", regex=True, na=False)
        tumor_tissue = tissue.str.contains("tumou?r|ccrcc|kirc|cancer", regex=True, na=False)
        normal_tissue = tissue.str.contains("normal|healthy|adjacent", regex=True, na=False)
        status = np.full(len(meta), "unknown", dtype=object)
        status[(epithelial & tumor_tissue).values] = "malignant_inferred"
        status[(normal_tissue | (~epithelial & tumor_tissue)).values] = "non_malignant_inferred"
        meta["malignant_status"] = status
        malignant_source = "inferred_from_cell_type_and_tissue"

    optional = {}
    for key, names in {
        "grade": ["grade", "histologic_grade", "histological_grade"],
        "stage": ["stage", "tumor_stage", "clinical_stage"],
        "response": ["response", "therapy_response", "treatment_response"],
        "survival": ["survival", "os", "overall_survival"],
        "region": ["region", "tumor_region", "tumour_region"],
    }.items():
        col = pick_obs_column(obs, names, names, max_unique=300)
        if col:
            meta[key] = obs[col].astype(str).values
            optional[key] = str(col)

    source = {
        "patient_id": str(patient_col) if patient_col else "",
        "cell_type": str(cell_type_col) if cell_type_col else "",
        "tissue": str(tissue_col) if tissue_col else "",
        "malignant_status": malignant_source,
        "optional": optional,
    }
    return meta, source


def get_umap(adata: ad.AnnData, expr, genes: np.ndarray, counts_like: bool, out_dir: Path) -> tuple[np.ndarray, str]:
    for key in ("X_umap", "umap", "UMAP"):
        if key in adata.obsm:
            arr = np.asarray(adata.obsm[key])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2].astype(np.float32), f"obsm:{key}"
    obs = adata.obs
    lower_cols = {str(c).lower(): c for c in obs.columns}
    for x_name, y_name in (("umap_1", "umap_2"), ("umap1", "umap2"), ("x_umap", "y_umap")):
        if x_name in lower_cols and y_name in lower_cols:
            arr = obs[[lower_cols[x_name], lower_cols[y_name]]].astype(float).to_numpy()
            return arr.astype(np.float32), f"obs:{lower_cols[x_name]},{lower_cols[y_name]}"

    log("UMAP not found in h5ad; compute with scanpy")
    x = expr.copy()
    tmp = ad.AnnData(X=x)
    tmp.var_names = pd.Index(genes.astype(str))
    if counts_like:
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=min(3000, tmp.n_vars), flavor="seurat")
    if "highly_variable" in tmp.var:
        tmp = tmp[:, tmp.var["highly_variable"].values].copy()
    sc.pp.scale(tmp, max_value=10)
    sc.tl.pca(tmp, n_comps=40, svd_solver="arpack")
    sc.pp.neighbors(tmp, n_neighbors=15, n_pcs=40)
    sc.tl.umap(tmp, random_state=13)
    np.save(out_dir / "computed_umap.npy", tmp.obsm["X_umap"].astype(np.float32))
    return tmp.obsm["X_umap"].astype(np.float32), "computed_scanpy"


def extract_panel_scores(expr, genes: np.ndarray, counts_like: bool) -> tuple[pd.DataFrame, dict]:
    genes_upper = pd.Series(genes).astype(str).str.upper().to_numpy()
    wanted = sorted(set(["RPS6"] + [g for geneset in PATHWAY_GENESETS.values() for g in geneset]))
    gene_to_indices = {gene: np.flatnonzero(genes_upper == gene) for gene in wanted}
    present = {gene: idx for gene, idx in gene_to_indices.items() if len(idx) > 0}
    libsize = None
    if counts_like:
        libsize = np.asarray(expr.sum(axis=1)).ravel().astype(np.float64)
        libsize = np.maximum(libsize, 1.0)

    values = {}
    for gene, idx in present.items():
        col = expr[:, idx]
        if sparse.issparse(col):
            arr = np.asarray(col.sum(axis=1)).ravel().astype(np.float64)
        else:
            arr = np.asarray(col).sum(axis=1).astype(np.float64)
        if counts_like:
            arr = np.log1p(arr / libsize * 10000.0)
        values[gene] = arr.astype(np.float32)

    out = pd.DataFrame(index=np.arange(expr.shape[0]))
    out["RPS6_mRNA"] = values.get("RPS6", np.full(expr.shape[0], np.nan, dtype=np.float32))
    summary = {}
    for score_name, gene_set in PATHWAY_GENESETS.items():
        matched = [gene for gene in gene_set if gene in values]
        summary[score_name] = {"requested": gene_set, "matched": matched}
        if not matched:
            out[score_name] = np.nan
            continue
        mat = np.vstack([values[g] for g in matched]).T.astype(np.float64)
        mean = np.nanmean(mat, axis=0, keepdims=True)
        sd = np.nanstd(mat, axis=0, keepdims=True)
        sd[sd <= 0] = 1.0
        z = (mat - mean) / sd
        out[score_name] = np.nanmean(z, axis=1).astype(np.float32)
    return out, summary


def safe_corr(x, y, method: str) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3 or np.nanstd(x[ok]) <= 0 or np.nanstd(y[ok]) <= 0:
        return np.nan, np.nan, n
    if method == "spearman":
        res = spearmanr(x[ok], y[ok])
    else:
        res = pearsonr(x[ok], y[ok])
    return float(res.statistic), float(res.pvalue), n


def make_correlations(cell_table: pd.DataFrame) -> pd.DataFrame:
    y = cell_table["predicted_RPS6_pS235_S236"].to_numpy()
    variables = [
        "RPS6_mRNA",
        "mTOR_S6_score",
        "cell_cycle_score",
        "hypoxia_score",
        "angiogenesis_score",
        "MYC_targets_score",
        "EMT_score",
        "interferon_stress_score",
    ]
    rows = []
    for var in variables:
        sr, sp, n = safe_corr(cell_table[var], y, "spearman")
        pr, pp, _ = safe_corr(cell_table[var], y, "pearson")
        rows.append(
            {
                "variable": var,
                "n": n,
                "spearman_r": sr,
                "spearman_p": sp,
                "pearson_r": pr,
                "pearson_p": pp,
                "target": "predicted_RPS6_pS235_S236",
            }
        )
    return pd.DataFrame(rows)


def make_group_stats(cell_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_col = "predicted_RPS6_pS235_S236"
    for group_var in ["cell_type", "malignant_status", "tissue", "patient_id"]:
        if group_var not in cell_table.columns:
            continue
        groups = []
        for group, sub in cell_table.groupby(group_var, dropna=False):
            vals = sub[y_col].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            groups.append(vals)
            rows.append(
                {
                    "group_variable": group_var,
                    "group": str(group),
                    "n": int(len(vals)),
                    "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals)),
                    "median": float(np.median(vals)),
                    "q25": float(np.quantile(vals, 0.25)),
                    "q75": float(np.quantile(vals, 0.75)),
                    "test": "",
                    "p_global": np.nan,
                }
            )
        valid = [g for g in groups if len(g) >= 3]
        p_global = np.nan
        test = ""
        if len(valid) >= 2:
            try:
                p_global = float(kruskal(*valid).pvalue)
                test = "Kruskal-Wallis"
            except Exception:
                p_global = np.nan
        for row in rows:
            if row["group_variable"] == group_var:
                row["test"] = test
                row["p_global"] = p_global
    return pd.DataFrame(rows)


def compute_embeddings(
    expr,
    genes: np.ndarray,
    embedding_dir: Path,
    modeling_code_dir: Path,
    scfoundation_code_dir: Path,
    scfoundation_weight_path: Path,
    batch_size: int,
    max_gene_tokens: int,
    highres: float,
    device_name: str,
    skip_existing: bool,
    log_every: int,
):
    ensure_dir(embedding_dir)
    emb_path = embedding_dir / "embeddings.npy"
    meta_path = embedding_dir / "embedding_metadata.json"
    if skip_existing and emb_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("complete"):
            log(f"reuse embeddings {emb_path}")
            return np.load(emb_path, mmap_mode="r")

    if str(modeling_code_dir) not in sys.path:
        sys.path.insert(0, str(modeling_code_dir))
    from precompute_scfoundation_embeddings import (
        build_gene_projection,
        encode_batch_with_fallback,
        load_scfoundation_gene_list,
        load_scfoundation_model,
    )

    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")
    target_genes = load_scfoundation_gene_list(scfoundation_code_dir)
    model, config = load_scfoundation_model(scfoundation_code_dir, scfoundation_weight_path, device)
    proj, n_source_used, n_target_hit = build_gene_projection(genes, target_genes)
    x = to_csr_float(expr)
    n_cells = x.shape[0]
    emb_store = None
    emb_dim = 0
    for start in range(0, n_cells, batch_size):
        stop = min(start + batch_size, n_cells)
        x_aligned = (x[start:stop] @ proj).tocsr().astype(np.float32)
        emb = encode_batch_with_fallback(model, config, x_aligned, device, highres, max_gene_tokens)
        if emb_store is None:
            emb_dim = int(emb.shape[1])
            emb_store = np.lib.format.open_memmap(
                emb_path, mode="w+", dtype=np.float32, shape=(n_cells, emb_dim)
            )
        emb_store[start:stop, :] = emb
        emb_store.flush()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if stop == n_cells or stop % max(1, log_every) == 0:
            log(f"scFoundation encoded {stop}/{n_cells}")
    if emb_store is None:
        np.save(emb_path, np.empty((0, 0), dtype=np.float32))
    else:
        del emb_store
    (embedding_dir / "genes_used.tsv").write_text("\n".join(target_genes) + "\n", encoding="utf-8")
    meta = {
        "n_cells": int(n_cells),
        "embedding_dim": int(emb_dim),
        "n_input_genes": int(len(genes)),
        "n_source_genes_used": int(n_source_used),
        "n_scfoundation_genes_hit": int(n_target_hit),
        "n_scfoundation_genes_total": int(len(target_genes)),
        "max_gene_tokens": int(max_gene_tokens),
        "highres": float(highres),
        "batch_size": int(batch_size),
        "device": str(device),
        "complete": True,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return np.load(emb_path, mmap_mode="r")


def run_scp682_prediction(
    embeddings,
    output_dir: Path,
    model_path: Path,
    formal_result_dir: Path,
    modeling_code_dir: Path,
    batch_size: int,
    device_name: str,
):
    if str(modeling_code_dir) not in sys.path:
        sys.path.insert(0, str(modeling_code_dir))
    from train_scp682_sc11_expanded_scnet_site_gnn import ScFoundationPathwayPredictor

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**ckpt["args"])
    target_rows = ckpt["target_rows"]
    target_ids = [str(row["target_id"]) for row in target_rows]
    if "RPS6_pSitePending" not in target_ids:
        raise RuntimeError("RPS6_pSitePending missing from SCP682-SC checkpoint")
    rps6_pos = target_ids.index("RPS6_pSitePending")
    pathway_names = ckpt["pathway_names"]
    edge_index = ckpt.get("scnet_site_graph_edge_index")
    edge_weight = ckpt.get("scnet_site_graph_edge_weight")
    n_graph_nodes = None
    if edge_index is not None:
        n_graph_nodes = int(np.asarray(edge_index).max()) + 1
    model = ScFoundationPathwayPredictor(
        n_pathways=len(pathway_names),
        n_targets=len(target_rows),
        d_input=int(embeddings.shape[1]),
        target_pathway_prior=ckpt["target_pathway_prior"],
        hidden=int(args.hidden),
        n_layers=int(args.pathway_layers),
        n_heads=int(args.attention_heads),
        dropout=float(args.dropout),
        bulk_pathway_embedding=ckpt.get("scp68222_full_pathway_embedding"),
        bulk_site_embedding=ckpt.get("scp68222_full_site_embedding"),
        bulk_site_mask=ckpt.get("scp68222_full_site_mask"),
        full_transfer_scale=float(args.full_transfer_scale),
        site_graph_edge_index=edge_index,
        site_graph_edge_weight=edge_weight,
        n_graph_nodes=n_graph_nodes,
        site_graph_scale=float(args.site_graph_scale),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")
    model.to(device)
    model.eval()
    present = torch.ones((batch_size, len(pathway_names)), dtype=torch.float32, device=device)
    pred_z = np.empty(embeddings.shape[0], dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, embeddings.shape[0], batch_size):
            stop = min(start + batch_size, embeddings.shape[0])
            x = torch.as_tensor(np.asarray(embeddings[start:stop]), dtype=torch.float32, device=device)
            p = present[: stop - start]
            pred, _ = model(x, p)
            pred_z[start:stop] = pred[:, rps6_pos].detach().cpu().numpy().astype(np.float32)
            log(f"SCP682-SC predicted {stop}/{embeddings.shape[0]}")

    transform = pd.read_csv(formal_result_dir / "tables" / "scp682_sc11_target_transform.tsv", sep="\t")
    row = transform.loc[transform["target_id"].astype(str).eq("RPS6_pSitePending")].iloc[0]
    if str(row["mode"]) == "zscore":
        pred_raw = pred_z * float(row["sd"]) + float(row["mean"])
    else:
        pred_raw = pred_z.copy()
    pred_table = pd.DataFrame(
        {
            "predicted_RPS6_pS235_S236": pred_raw.astype(np.float32),
            "predicted_RPS6_pS235_S236_z": pred_z.astype(np.float32),
        }
    )
    metadata = {
        "checkpoint": str(model_path),
        "target_id_in_model": "RPS6_pSitePending",
        "reported_site": "RPS6 pS235/S236",
        "target_transform_mode": str(row["mode"]),
        "target_transform_mean": float(row["mean"]),
        "target_transform_sd": float(row["sd"]),
        "n_pathways": int(len(pathway_names)),
        "n_targets": int(len(target_rows)),
    }
    (output_dir / "prediction_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return pred_table, metadata


def save_figure(fig, out_base: Path):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figures(cell_table: pd.DataFrame, corr: pd.DataFrame, fig_dir: Path):
    ensure_dir(fig_dir)
    sns.set_theme(style="whitegrid", context="paper")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sca = ax.scatter(
        cell_table["UMAP1"],
        cell_table["UMAP2"],
        c=cell_table["predicted_RPS6_pS235_S236"],
        s=3,
        cmap="viridis",
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Predicted RPS6 pS235/S236")
    cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("predicted phospho-state")
    save_figure(fig, fig_dir / "kirc_rps6_umap")

    plot = cell_table.copy()
    counts = plot["cell_type"].value_counts()
    keep = counts[counts >= 20].index[:20]
    plot = plot[plot["cell_type"].isin(keep)].copy()
    order = (
        plot.groupby("cell_type")["predicted_RPS6_pS235_S236"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(max(7, 0.42 * max(len(order), 1)), 4.5))
    sns.violinplot(
        data=plot,
        x="cell_type",
        y="predicted_RPS6_pS235_S236",
        order=order,
        inner="quartile",
        cut=0,
        linewidth=0.4,
        ax=ax,
        color="#6aa6b8",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Predicted RPS6 pS235/S236")
    ax.tick_params(axis="x", rotation=60)
    save_figure(fig, fig_dir / "kirc_rps6_celltype_violin")

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ok = np.isfinite(cell_table["RPS6_mRNA"]) & np.isfinite(cell_table["predicted_RPS6_pS235_S236"])
    hb = ax.hexbin(
        cell_table.loc[ok, "RPS6_mRNA"],
        cell_table.loc[ok, "predicted_RPS6_pS235_S236"],
        gridsize=55,
        mincnt=1,
        cmap="mako",
    )
    r = corr.loc[corr["variable"].eq("RPS6_mRNA"), "spearman_r"]
    r_text = f"Spearman r={float(r.iloc[0]):.3f}" if len(r) and np.isfinite(float(r.iloc[0])) else "Spearman r=NA"
    ax.text(0.03, 0.97, r_text, transform=ax.transAxes, va="top", ha="left")
    ax.set_xlabel("RPS6 mRNA")
    ax.set_ylabel("Predicted RPS6 pS235/S236")
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04).set_label("cells")
    save_figure(fig, fig_dir / "kirc_rps6_vs_rps6_mrna")

    heat = corr.set_index("variable")[["spearman_r"]].T
    fig, ax = plt.subplots(figsize=(8.2, 2.0))
    sns.heatmap(
        heat,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Spearman r"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Correlation with predicted RPS6 pS235/S236")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, fig_dir / "kirc_rps6_pathway_heatmap")


def write_summary(
    report_path: Path,
    dataset_id: str,
    cell_table: pd.DataFrame,
    metadata_sources: dict,
    expr_source: str,
    gene_symbol_source: str,
    umap_source: str,
    prediction_metadata: dict,
    corr: pd.DataFrame,
):
    n_cells = len(cell_table)
    n_patients = cell_table["patient_id"].astype(str).replace("unknown", np.nan).dropna().nunique()
    malignant_values = cell_table["malignant_status"].astype(str)
    has_malignant = bool(malignant_values.str.contains("malignant", case=False, na=False).any())
    mtor_r = corr.loc[corr["variable"].eq("mTOR_S6_score"), "spearman_r"]
    rps6_r = corr.loc[corr["variable"].eq("RPS6_mRNA"), "spearman_r"]
    mtor_text = "NA" if mtor_r.empty or not np.isfinite(float(mtor_r.iloc[0])) else f"{float(mtor_r.iloc[0]):.3f}"
    rps6_text = "NA" if rps6_r.empty or not np.isfinite(float(rps6_r.iloc[0])) else f"{float(rps6_r.iloc[0]):.3f}"

    group_lines = []
    if has_malignant:
        group_summary = (
            cell_table.groupby("malignant_status")["predicted_RPS6_pS235_S236"]
            .agg(["count", "median", "mean"])
            .reset_index()
        )
        for _, row in group_summary.iterrows():
            group_lines.append(
                f"- {row['malignant_status']}: n={int(row['count'])}, median={float(row['median']):.3f}, mean={float(row['mean']):.3f}"
            )
    else:
        group_lines.append("- 未找到明确恶性细胞注释；若有 tumor/normal 与 epithelial 信息，脚本已生成推断标签。")

    text = f"""# KIRC single-cell RPS6 validation

## 数据

- 数据集: {dataset_id}
- 细胞数: {n_cells}
- 患者数: {n_patients if n_patients else 'unknown'}
- RNA 矩阵来源: {expr_source}
- 基因名来源: {gene_symbol_source}
- UMAP 来源: {umap_source}
- 细胞类型字段: {metadata_sources.get('cell_type', '') or '未识别'}
- 恶性状态字段: {metadata_sources.get('malignant_status', '') or '未识别'}

## RPS6 pS235/S236 状态

- 模型输出为 predicted phospho-state，不是直接测得的磷酸化。
- SCP682-SC 内部 target_id: RPS6_pSitePending；本文图表报告为 RPS6 pS235/S236。
- 反变换均值: {prediction_metadata.get('target_transform_mean', np.nan):.6g}
- 反变换标准差: {prediction_metadata.get('target_transform_sd', np.nan):.6g}

## 分组结果

{chr(10).join(group_lines)}

## 通路相关

- predicted RPS6 pS235/S236 与 mTOR/S6 RNA score 的 Spearman r: {mtor_text}
- predicted RPS6 pS235/S236 与 RPS6 mRNA 的 Spearman r: {rps6_text}

## 判断

- 若 mTOR/S6 相关强于 RPS6 mRNA，结果支持 predicted RPS6 pS235/S236 代表单细胞磷酸化状态，而不是 RPS6 表达量的简单替代。
- 与 bulk TCGA-KIRC 结果的方向一致性需要结合恶性细胞、mTOR/S6 高状态和可用临床分组共同判断；对应数值见 tables 目录。
"""
    report_path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    parser.add_argument("--h5ad-gz", required=True)
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--dataset-id", default="GSE242299_ccRCC")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--formal-result-dir", required=True)
    parser.add_argument("--modeling-code-dir", required=True)
    parser.add_argument("--scfoundation-code-dir", default=r"D:\data\lsy\repos\scFoundation_model")
    parser.add_argument("--scfoundation-weight-path", default=r"D:\data\lsy\models\scfoundation\models.ckpt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=1)
    parser.add_argument("--prediction-batch-size", type=int, default=1024)
    parser.add_argument("--max-gene-tokens", type=int, default=12000)
    parser.add_argument("--highres", type=float, default=4.0)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--embedding-log-every", type=int, default=100)
    parser.add_argument("--force-decompress", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = ensure_dir(root / args.output_dir)
    table_dir = ensure_dir(out_dir / "tables")
    fig_dir = ensure_dir(out_dir / "figures")
    report_dir = ensure_dir(out_dir / "reports")
    intermediate_dir = ensure_dir(out_dir / "intermediate")

    h5ad_path = maybe_decompress_h5ad(Path(args.h5ad_gz), Path(args.h5ad), args.force_decompress)
    log(f"read h5ad {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    if args.max_cells and adata.n_obs > args.max_cells:
        rng = np.random.default_rng(13)
        keep = np.sort(rng.choice(adata.n_obs, size=args.max_cells, replace=False))
        adata = adata[keep].copy()
        log(f"subsample cells for test: {adata.n_obs}")

    expr_raw, var, var_names, expr_source = choose_expression_source(adata)
    genes, gene_symbol_source = choose_gene_symbols(var, var_names)
    expr = to_csr_float(expr_raw)
    counts_like = infer_counts_like(expr)
    log(f"expression source={expr_source} genes={len(genes)} counts_like={counts_like}")

    meta, metadata_sources = build_metadata(adata, args.dataset_id)
    umap, umap_source = get_umap(adata, expr, genes, counts_like, intermediate_dir)
    scores, gene_score_summary = extract_panel_scores(expr, genes, counts_like)

    emb_dir = ensure_dir(intermediate_dir / "scfoundation_embeddings")
    (emb_dir / "barcodes.tsv").write_text("\n".join(meta["cell_id"].astype(str)) + "\n", encoding="utf-8")
    embeddings = compute_embeddings(
        expr=expr,
        genes=genes,
        embedding_dir=emb_dir,
        modeling_code_dir=Path(args.modeling_code_dir),
        scfoundation_code_dir=Path(args.scfoundation_code_dir),
        scfoundation_weight_path=Path(args.scfoundation_weight_path),
        batch_size=int(args.embedding_batch_size),
        max_gene_tokens=int(args.max_gene_tokens),
        highres=float(args.highres),
        device_name=args.device,
        skip_existing=bool(args.skip_existing),
        log_every=int(args.embedding_log_every),
    )

    pred, prediction_metadata = run_scp682_prediction(
        embeddings=embeddings,
        output_dir=intermediate_dir,
        model_path=Path(args.model_path),
        formal_result_dir=Path(args.formal_result_dir),
        modeling_code_dir=Path(args.modeling_code_dir),
        batch_size=int(args.prediction_batch_size),
        device_name=args.device,
    )

    cell_table = pd.concat([meta.reset_index(drop=True), pred.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    cell_table["UMAP1"] = umap[:, 0]
    cell_table["UMAP2"] = umap[:, 1]
    required_order = [
        "cell_id",
        "dataset_id",
        "patient_id",
        "cell_type",
        "malignant_status",
        "predicted_RPS6_pS235_S236",
        "RPS6_mRNA",
        "mTOR_S6_score",
        "cell_cycle_score",
        "hypoxia_score",
        "angiogenesis_score",
    ]
    remaining = [c for c in cell_table.columns if c not in required_order]
    cell_table = cell_table[required_order + remaining]
    cell_table.to_csv(table_dir / "kirc_cell_rps6_prediction.tsv", sep="\t", index=False)

    group_stats = make_group_stats(cell_table)
    group_stats.to_csv(table_dir / "kirc_rps6_by_cell_type.tsv", sep="\t", index=False)
    corr = make_correlations(cell_table)
    corr.to_csv(table_dir / "kirc_rps6_pathway_correlation.tsv", sep="\t", index=False)

    (table_dir / "kirc_rps6_gene_score_manifest.json").write_text(
        json.dumps(gene_score_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "dataset_id": args.dataset_id,
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "expr_source": expr_source,
                "gene_symbol_source": gene_symbol_source,
                "counts_like": bool(counts_like),
                "metadata_sources": metadata_sources,
                "umap_source": umap_source,
                "prediction": prediction_metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    make_figures(cell_table, corr, fig_dir)
    write_summary(
        report_path=report_dir / "summary.md",
        dataset_id=args.dataset_id,
        cell_table=cell_table,
        metadata_sources=metadata_sources,
        expr_source=expr_source,
        gene_symbol_source=gene_symbol_source,
        umap_source=umap_source,
        prediction_metadata=prediction_metadata,
        corr=corr,
    )
    log(f"done output={out_dir}")


if __name__ == "__main__":
    main()
