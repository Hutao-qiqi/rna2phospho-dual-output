# 模型: SCP682-SC
# 作用: 训练 scFoundation + pathway attention + expanded ScNET site graph 的单细胞磷酸化预测模型，支持正式内部五折。
# 输入: ./data_root 下的统一 scFoundation embedding、磷酸化 targets、mask、pathway manifest 和 bulk site graph 先验。
# 输出: 模型权重、逐位点重建结果、外部验证结果、QuRIE delta 拟合结果和注意力表。
# 依赖: Python, numpy, pandas, torch, scikit-learn, scipy。
# 原始路径: D:\data\lsy\vm_lsy_parent\lsy\03_code\single_cell\modeling\train_scp682_sc11_expanded_scnet_site_gnn.py
# 原始版本: 2026-05-27 支持 cv_folds/cv_fold 的 SC11 正式五折版本。

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn
import torch.nn.functional as F


MODEL_NAME = "SCP682-SC11-expanded-ScNET-site-GNN"
QURIE = "qurie_seq_bjab_2021"
FOCUS_TARGETS = (
    "PLCG2_Y759",
    "SYK_pSitePending",
    "p-BLNK",
    "RPS6_pSitePending",
    "MAPK14_pSitePending",
    "p-p65",
    "MAPK1_MAPK3_pSitePending",
    "AKT_pSitePending",
)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_list(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_bool(values):
    return pd.Series(values).astype(str).str.lower().isin(["true", "1", "yes", "y"])


def safe_spearman(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() < 3 or np.std(y[ok]) == 0 or np.std(pred[ok]) == 0:
        return float("nan")
    return float(spearmanr(y[ok], pred[ok]).statistic)


def safe_pearson(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() < 3 or np.std(y[ok]) == 0 or np.std(pred[ok]) == 0:
        return float("nan")
    return float(pearsonr(y[ok], pred[ok]).statistic)


def choose_targets(target_table, target_ids):
    tt = target_table.copy()
    tt["target_index"] = tt["target_index"].astype(int)
    if target_ids == ["include_in_loss"]:
        sub = tt[tt["include_in_loss"].astype(str).str.lower().eq("true")].copy()
    elif target_ids == ["focus"]:
        keys = set(FOCUS_TARGETS)
        sub = tt[tt["target_id"].astype(str).isin(keys)].copy()
    else:
        keys = set(target_ids)
        m = np.zeros(len(tt), dtype=bool)
        for col in ("target_id", "feature_id", "canonical_label"):
            if col in tt.columns:
                m |= tt[col].astype(str).isin(keys).to_numpy()
        sub = tt[m].copy()
    keep = []
    seen = set()
    for _, row in sub.sort_values(["target_index", "dataset_id", "feature_id"]).iterrows():
        idx = int(row["target_index"])
        if idx not in seen:
            keep.append(idx)
            seen.add(idx)
    labels = tt[tt["target_index"].isin(keep)].drop_duplicates("target_index").sort_values("target_index")
    rows = []
    for _, row in labels.iterrows():
        rows.append(
            {
                "target_index": int(row["target_index"]),
                "target_id": str(row["target_id"]),
                "feature_id": str(row["feature_id"]),
                "protein_symbol": str(row.get("protein_symbol", "")),
                "residue": str(row.get("residue", "")),
                "canonical_label": str(row.get("canonical_label", "")),
                "evaluation_tier": str(row.get("evaluation_tier", "")),
            }
        )
    return np.asarray(keep, dtype=np.int64), rows


def split_symbols(text):
    vals = []
    for part in str(text).replace("/", ";").replace(",", ";").split(";"):
        token = part.strip()
        if token and token.lower() != "nan":
            vals.append(token.upper())
    return vals


def build_target_pathway_prior(target_rows, pathway_names, manifest):
    genes_by_pathway = {}
    for pathway, sub in manifest.groupby("pathway"):
        genes_by_pathway[str(pathway)] = set(sub["gene"].astype(str).str.upper().tolist())
    alias = {
        "P-P65": ["RELA"],
        "P-RB": ["RB1"],
        "P-BLNK": ["BLNK"],
        "P-CD79A": ["CD79A"],
        "P-CD79A": ["CD79A"],
        "P-CD79A/B": ["CD79A", "CD79B"],
        "P-HISTONH2A.X": ["H2AFX"],
        "P-HISTONE_H2A.X": ["H2AFX"],
        "P-HISTONH3": ["H3C1", "H3F3A", "H3F3B"],
        "P-PKC-B1": ["PRKCB"],
        "P-SRC": ["SRC", "LYN", "FYN", "LCK"],
        "P-TOR": ["MTOR"],
        "P-C-JUN": ["JUN"],
        "P-JNK": ["MAPK8", "MAPK9"],
        "P-IKKA/B": ["CHUK", "IKBKB"],
    }
    prior = np.zeros((len(target_rows), len(pathway_names)), dtype=np.float32)
    rows = []
    for i, row in enumerate(target_rows):
        symbols = split_symbols(row.get("protein_symbol", ""))
        tid = str(row.get("target_id", "")).upper()
        canonical = str(row.get("canonical_label", "")).upper()
        for key, val in alias.items():
            if key in tid or key in canonical:
                symbols.extend(val)
        for p, pathway in enumerate(pathway_names):
            genes = genes_by_pathway.get(pathway, set())
            if any(sym in genes for sym in symbols):
                prior[i, p] = 1.0
        if prior[i].sum() == 0:
            if any(key in tid for key in ("RPS6", "EIF4EBP1", "AKT", "MTOR", "PDPK1", "NDRG1")):
                for p, pathway in enumerate(pathway_names):
                    if pathway == "AKT_mTOR_S6_axis":
                        prior[i, p] = 1.0
            if any(key in tid for key in ("MAPK", "MAP2K", "ERK", "JUN", "FOS", "P38")):
                for p, pathway in enumerate(pathway_names):
                    if pathway in ("ERK_axis", "stress_ifn"):
                        prior[i, p] = 1.0
            if any(key in tid for key in ("STAT", "RELA", "NFKB", "IKK")):
                for p, pathway in enumerate(pathway_names):
                    if pathway in ("NFkB_axis", "stress_ifn"):
                        prior[i, p] = 1.0
            if any(key in tid for key in ("RB", "CDK", "H2AFX", "HISTON", "HISTONE", "MCM", "AURK", "CCN")):
                for p, pathway in enumerate(pathway_names):
                    if pathway == "cell_cycle":
                        prior[i, p] = 1.0
        total = prior[i].sum()
        if total > 0:
            prior[i] /= total
        for p, pathway in enumerate(pathway_names):
            if prior[i, p] > 0:
                rows.append(
                    {
                        "target_id": row["target_id"],
                        "target_index": int(row["target_index"]),
                        "protein_symbol": row.get("protein_symbol", ""),
                        "pathway": pathway,
                        "weight": float(prior[i, p]),
                    }
                )
    return prior, pd.DataFrame(rows)


def parse_gene_preview(text):
    genes = []
    for token in str(text).replace(";", ",").split(","):
        gene = token.strip().upper()
        if gene and gene not in {"NAN", "NONE"}:
            genes.append(gene)
    return genes


def target_symbols_for_transfer(row):
    symbols = split_symbols(row.get("protein_symbol", ""))
    tid = str(row.get("target_id", "")).upper()
    canonical = str(row.get("canonical_label", "")).upper()
    alias = {
        "P-P65": ["RELA"],
        "P-RB": ["RB1"],
        "P-BLNK": ["BLNK"],
        "P-CD79A": ["CD79A"],
        "P-HISTONH2A.X": ["H2AFX"],
        "P-HISTONE_H2A.X": ["H2AFX"],
        "P-HISTONH3": ["H3C1", "H3F3A", "H3F3B"],
        "P-PKC-B1": ["PRKCB"],
        "P-SRC": ["SRC", "LYN", "FYN", "LCK"],
        "P-TOR": ["MTOR"],
        "P-C-JUN": ["JUN"],
        "P-JNK": ["MAPK8", "MAPK9"],
        "P-IKKA/B": ["CHUK", "IKBKB"],
        "P-AMPK": ["PRKAA1", "PRKAA2"],
    }
    for key, val in alias.items():
        if key in tid or key in canonical:
            symbols.extend(val)
    clean = []
    for sym in symbols:
        sym = str(sym).upper().strip()
        if not sym or sym in {"NAN", "UNKNOWN", "PENDING", "PENDING_ANTIBODY_CLONE"}:
            continue
        if sym.startswith("P-"):
            continue
        clean.append(sym)
    return sorted(set(clean))


def build_scp68222_transfer_prior(target_rows, pathway_names, manifest, transfer_dir, alpha):
    transfer_dir = Path(transfer_dir)
    token_path = transfer_dir / "tables" / "scp682_22_pathway_token_summary.tsv"
    if not token_path.exists() or alpha <= 0:
        prior = np.zeros((len(target_rows), len(pathway_names)), dtype=np.float32)
        return prior, pd.DataFrame(), pd.DataFrame()

    bulk_tokens = pd.read_csv(token_path, sep="\t")
    bulk_gene_sets = []
    for _, row in bulk_tokens.iterrows():
        genes = set(parse_gene_preview(row.get("gene_preview", "")))
        bulk_gene_sets.append(genes)

    sc_gene_sets = []
    for pathway in pathway_names:
        genes = set(
            manifest.loc[manifest["pathway"].astype(str).eq(str(pathway)), "gene"]
            .astype(str)
            .str.upper()
            .tolist()
        )
        sc_gene_sets.append(genes)

    align = np.zeros((len(pathway_names), len(bulk_gene_sets)), dtype=np.float32)
    align_rows = []
    for i, sc_genes in enumerate(sc_gene_sets):
        for j, bulk_genes in enumerate(bulk_gene_sets):
            inter = len(sc_genes & bulk_genes)
            denom = max(1, min(len(sc_genes), len(bulk_genes)))
            score = inter / denom
            align[i, j] = score
            if score > 0:
                align_rows.append(
                    {
                        "sc_pathway": pathway_names[i],
                        "scp68222_pathway": str(bulk_tokens.iloc[j].get("pathway", "")),
                        "overlap": int(inter),
                        "score": float(score),
                    }
                )
    col_sum = align.sum(axis=0, keepdims=True)
    align_norm = np.divide(align, np.maximum(col_sum, 1e-8), out=np.zeros_like(align), where=col_sum > 0)

    transfer = np.zeros((len(target_rows), len(pathway_names)), dtype=np.float32)
    rows = []
    for t, row in enumerate(target_rows):
        symbols = target_symbols_for_transfer(row)
        if not symbols:
            continue
        bulk_score = np.zeros(len(bulk_gene_sets), dtype=np.float32)
        sym_set = set(symbols)
        for j, genes in enumerate(bulk_gene_sets):
            if genes & sym_set:
                bulk_score[j] = 1.0
        if bulk_score.sum() == 0:
            continue
        sc_score = align_norm @ bulk_score
        if sc_score.sum() <= 0:
            continue
        sc_score = sc_score / sc_score.sum()
        transfer[t] = sc_score.astype(np.float32)
        for p, val in enumerate(sc_score):
            if val > 0:
                rows.append(
                    {
                        "target_id": row["target_id"],
                        "target_index": int(row["target_index"]),
                        "protein_symbol": row.get("protein_symbol", ""),
                        "pathway": pathway_names[p],
                        "scp68222_transfer_weight": float(val),
                        "matched_symbols": ";".join(symbols),
                    }
                )
    return transfer, pd.DataFrame(rows), pd.DataFrame(align_rows)


def target_residues_for_transfer(row):
    vals = []
    for text in (row.get("target_id", ""), row.get("feature_id", ""), row.get("canonical_label", ""), row.get("residue", "")):
        for token in str(text).upper().replace("/", "_").replace("|", "_").replace("-", "_").split("_"):
            token = token.strip()
            if len(token) >= 2 and token[0] in {"S", "T", "Y"} and token[1:].isdigit():
                vals.append(token)
    return sorted(set(vals))


def load_scp68222_full_transfer(target_rows, pathway_names, manifest, transfer_dir):
    transfer_dir = Path(transfer_dir)
    array_path = transfer_dir / "tables" / "scp682_22_full_model_arrays.npz"
    model_paths = sorted((transfer_dir / "models").glob("scp682_22_cancer_group_pathway_residual*_fold*.pt"))
    token_path = transfer_dir / "tables" / "scp682_22_pathway_token_summary.tsv"
    spearman_path = transfer_dir / "tables" / "scp682_22_per_site_spearman.tsv"
    site_weight_path = transfer_dir / "tables" / "scp682_22_site_prior_weights.tsv"
    if array_path.exists():
        arr = np.load(array_path, allow_pickle=True)
        pathway_emb = arr["pathway_emb"].astype(np.float32)
        site_query = arr["site_query"].astype(np.float32)
        site_bias = arr["site_bias"].astype(np.float32)
        bulk_targets = [str(x) for x in arr["bulk_targets"].tolist()]
        bulk_pathway_names = [str(x) for x in arr["bulk_pathway_names"].tolist()]
        bulk_gene_sets = [set(str(x).upper() for x in genes.split(";") if x) for genes in arr["bulk_pathway_genes"].tolist()]
        n_model_files = int(arr["n_model_files"][0]) if "n_model_files" in arr.files else 0
    else:
        if not model_paths:
            return None
        pathway_embs = []
        site_queries = []
        site_biases = []
        bulk_targets = None
        bulk_pathways = None
        for path in model_paths:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            if "pathway_emb.weight" not in state or "site_query.weight" not in state:
                continue
            pathway_embs.append(state["pathway_emb.weight"].detach().cpu().numpy().astype(np.float32))
            site_queries.append(state["site_query.weight"].detach().cpu().numpy().astype(np.float32))
            if "site_bias" in state:
                site_biases.append(state["site_bias"].detach().cpu().numpy().astype(np.float32))
            if bulk_targets is None:
                bulk_targets = [str(x) for x in ckpt.get("targets", [])]
            if bulk_pathways is None:
                bulk_pathways = ckpt.get("pathways", [])
        if not pathway_embs or not site_queries or not bulk_targets or not bulk_pathways:
            return None
        pathway_emb = np.mean(np.stack(pathway_embs, axis=0), axis=0).astype(np.float32)
        site_query = np.mean(np.stack(site_queries, axis=0), axis=0).astype(np.float32)
        site_bias = np.mean(np.stack(site_biases, axis=0), axis=0).astype(np.float32) if site_biases else np.zeros(len(bulk_targets), dtype=np.float32)
        bulk_gene_sets = []
        bulk_pathway_names = []
        for item in bulk_pathways:
            bulk_pathway_names.append(str(item.get("name", "")))
            bulk_gene_sets.append(set(str(g).upper() for g in item.get("gene_names", []) if str(g).strip()))
        n_model_files = len(model_paths)
    if len(bulk_gene_sets) != pathway_emb.shape[0]:
        return None

    sc_gene_sets = []
    for pathway in pathway_names:
        genes = set(
            manifest.loc[manifest["pathway"].astype(str).eq(str(pathway)), "gene"]
            .astype(str)
            .str.upper()
            .tolist()
        )
        sc_gene_sets.append(genes)

    align = np.zeros((len(pathway_names), len(bulk_gene_sets)), dtype=np.float32)
    align_rows = []
    token_scores = np.ones(len(bulk_gene_sets), dtype=np.float32)
    if token_path.exists():
        token_table = pd.read_csv(token_path, sep="\t")
        score_map = {str(r["pathway"]): float(r.get("score", 1.0)) for _, r in token_table.iterrows() if pd.notna(r.get("score", 1.0))}
        for j, name in enumerate(bulk_pathway_names):
            token_scores[j] = max(score_map.get(name, 1.0), 1.0)
        token_scores = np.log1p(token_scores)
        token_scores = token_scores / max(float(np.nanmean(token_scores)), 1e-6)
    for i, sc_genes in enumerate(sc_gene_sets):
        for j, bulk_genes in enumerate(bulk_gene_sets):
            inter = len(sc_genes & bulk_genes)
            if inter <= 0:
                continue
            denom = math.sqrt(max(len(sc_genes), 1) * max(len(bulk_genes), 1))
            score = float(inter / max(denom, 1e-6)) * float(token_scores[j])
            align[i, j] = score
            align_rows.append(
                {
                    "sc_pathway": pathway_names[i],
                    "scp68222_pathway": bulk_pathway_names[j],
                    "overlap": int(inter),
                    "score": float(score),
                    "bulk_n_genes": int(len(bulk_genes)),
                    "sc_n_genes": int(len(sc_genes)),
                }
            )
    row_sum = align.sum(axis=1, keepdims=True)
    align_norm = np.divide(align, np.maximum(row_sum, 1e-8), out=np.zeros_like(align), where=row_sum > 0)
    sc_pathway_embedding = (align_norm @ pathway_emb).astype(np.float32)

    spearman = {}
    if spearman_path.exists():
        sp = pd.read_csv(spearman_path, sep="\t")
        sp = sp[sp["model"].astype(str).eq("scp682_22_cancer_group_pathway_residual")].copy()
        for _, row in sp.iterrows():
            try:
                spearman[str(row["target"])] = max(float(row["spearman"]), 0.0)
            except Exception:
                pass
    site_weight = {}
    if site_weight_path.exists():
        sw = pd.read_csv(site_weight_path, sep="\t")
        for _, row in sw.iterrows():
            try:
                site_weight[str(row["target"])] = float(row.get("site_weight", 1.0))
            except Exception:
                pass

    by_gene = {}
    by_gene_residue = {}
    for idx, target in enumerate(bulk_targets):
        parts = str(target).upper().split("|", 1)
        gene = parts[0]
        residue = parts[1] if len(parts) > 1 else ""
        by_gene.setdefault(gene, []).append(idx)
        if residue:
            by_gene_residue.setdefault((gene, residue), []).append(idx)

    sc_site_embedding = np.zeros((len(target_rows), site_query.shape[1]), dtype=np.float32)
    sc_site_mask = np.zeros(len(target_rows), dtype=np.float32)
    site_rows = []
    for t, row in enumerate(target_rows):
        symbols = target_symbols_for_transfer(row)
        residues = target_residues_for_transfer(row)
        candidates = []
        match_type = "none"
        for gene in symbols:
            for residue in residues:
                candidates.extend(by_gene_residue.get((gene, residue), []))
        if candidates:
            match_type = "exact_gene_residue"
        else:
            for gene in symbols:
                candidates.extend(by_gene.get(gene, []))
            if candidates:
                match_type = "same_parent_gene"
        candidates = sorted(set(candidates))
        if not candidates:
            continue
        weights = np.asarray(
            [max(spearman.get(bulk_targets[i], 0.05), 0.05) * max(site_weight.get(bulk_targets[i], 1.0), 0.1) for i in candidates],
            dtype=np.float32,
        )
        weights = weights / max(float(weights.sum()), 1e-8)
        sc_site_embedding[t] = np.sum(site_query[candidates] * weights[:, None], axis=0)
        sc_site_mask[t] = 1.0
        top = candidates[int(np.argmax(weights))]
        site_rows.append(
            {
                "target_id": row["target_id"],
                "target_index": int(row["target_index"]),
                "protein_symbol": row.get("protein_symbol", ""),
                "residue": ";".join(residues),
                "match_type": match_type,
                "n_bulk_sites": int(len(candidates)),
                "top_bulk_target": bulk_targets[top],
                "top_weight": float(np.max(weights)),
                "mean_bulk_spearman": float(np.mean([spearman.get(bulk_targets[i], np.nan) for i in candidates])),
                "mean_bulk_site_bias": float(np.mean(site_bias[candidates])),
            }
        )

    return {
        "pathway_embedding": sc_pathway_embedding.astype(np.float32),
        "site_embedding": sc_site_embedding.astype(np.float32),
        "site_mask": sc_site_mask.astype(np.float32),
        "pathway_alignment": pd.DataFrame(align_rows),
        "site_matches": pd.DataFrame(site_rows),
        "n_model_files": n_model_files,
        "bulk_dim": int(site_query.shape[1]),
    }


def parse_bulk_target_gene_residue(target):
    text = str(target).upper()
    if "|" in text:
        gene, residue = text.split("|", 1)
    else:
        gene, residue = text, ""
    return gene.strip(), residue.strip()


def load_scp682_main_transfer(target_rows, pathway_names, manifest, transfer_dir):
    transfer_dir = Path(transfer_dir)
    array_candidates = [
        transfer_dir / "tables" / "scp682_main_sc_transfer_arrays.npz",
        transfer_dir / "tables" / "scp682_main_transfer_arrays.npz",
    ]
    array_path = next((p for p in array_candidates if p.exists()), None)
    if array_path is None:
        return None

    arr = np.load(array_path, allow_pickle=True)
    site_query = arr["site_embedding"].astype(np.float32)
    bulk_targets = [str(x) for x in arr["bulk_targets"].tolist()]
    if "site_spearman" in arr.files:
        site_spearman = arr["site_spearman"].astype(np.float32)
    else:
        site_spearman = np.ones(len(bulk_targets), dtype=np.float32)
    site_spearman = np.nan_to_num(site_spearman, nan=0.05, posinf=0.05, neginf=0.0).astype(np.float32)
    if "site_prior" in arr.files:
        site_prior = arr["site_prior"].astype(np.float32)
    else:
        site_prior = np.ones(len(bulk_targets), dtype=np.float32)
    site_prior = np.nan_to_num(site_prior, nan=0.1, posinf=1.0, neginf=0.1).astype(np.float32)
    if site_query.ndim != 2 or site_query.shape[0] != len(bulk_targets):
        return None

    by_gene = {}
    by_gene_residue = {}
    for idx, target in enumerate(bulk_targets):
        gene, residue = parse_bulk_target_gene_residue(target)
        if gene:
            by_gene.setdefault(gene, []).append(idx)
        if gene and residue:
            by_gene_residue.setdefault((gene, residue), []).append(idx)

    sc_site_embedding = np.zeros((len(target_rows), site_query.shape[1]), dtype=np.float32)
    sc_site_mask = np.zeros(len(target_rows), dtype=np.float32)
    site_rows = []
    for t, row in enumerate(target_rows):
        symbols = target_symbols_for_transfer(row)
        residues = target_residues_for_transfer(row)
        candidates = []
        match_type = "none"
        for gene in symbols:
            for residue in residues:
                candidates.extend(by_gene_residue.get((gene, residue), []))
        if candidates:
            match_type = "exact_gene_residue"
        else:
            for gene in symbols:
                candidates.extend(by_gene.get(gene, []))
            if candidates:
                match_type = "same_parent_gene"
        candidates = sorted(set(candidates))
        if not candidates:
            continue
        weights = np.asarray(
            [
                max(float(site_spearman[i]), 0.05) * max(float(site_prior[i]), 0.1)
                for i in candidates
            ],
            dtype=np.float32,
        )
        weights = weights / max(float(weights.sum()), 1e-8)
        sc_site_embedding[t] = np.sum(site_query[candidates] * weights[:, None], axis=0)
        sc_site_mask[t] = 1.0
        top = candidates[int(np.argmax(weights))]
        site_rows.append(
            {
                "target_id": row["target_id"],
                "target_index": int(row["target_index"]),
                "protein_symbol": row.get("protein_symbol", ""),
                "residue": ";".join(residues),
                "match_type": match_type,
                "n_bulk_sites": int(len(candidates)),
                "top_bulk_target": bulk_targets[top],
                "top_weight": float(np.max(weights)),
                "mean_bulk_spearman": float(np.mean([site_spearman[i] for i in candidates])),
                "mean_bulk_site_prior": float(np.mean([site_prior[i] for i in candidates])),
            }
        )

    sc_pathway_embedding = np.zeros((len(pathway_names), site_query.shape[1]), dtype=np.float32)
    align_rows = []
    for p, pathway in enumerate(pathway_names):
        genes = set(
            manifest.loc[manifest["pathway"].astype(str).eq(str(pathway)), "gene"]
            .astype(str)
            .str.upper()
            .tolist()
        )
        candidates = []
        for gene in genes:
            candidates.extend(by_gene.get(gene, []))
        candidates = sorted(set(candidates))
        if not candidates:
            continue
        weights = np.asarray(
            [
                max(float(site_spearman[i]), 0.05) * max(float(site_prior[i]), 0.1)
                for i in candidates
            ],
            dtype=np.float32,
        )
        weights = weights / max(float(weights.sum()), 1e-8)
        sc_pathway_embedding[p] = np.sum(site_query[candidates] * weights[:, None], axis=0)
        for idx in candidates[:100]:
            align_rows.append(
                {
                    "sc_pathway": pathway,
                    "scp682_main_site": bulk_targets[idx],
                    "matched_gene": parse_bulk_target_gene_residue(bulk_targets[idx])[0],
                    "bulk_spearman": float(site_spearman[idx]),
                    "bulk_site_prior": float(site_prior[idx]),
                }
            )

    transfer = np.zeros((len(target_rows), len(pathway_names)), dtype=np.float32)
    transfer_rows = []
    path_norm = np.linalg.norm(sc_pathway_embedding, axis=1)
    for t, row in enumerate(target_rows):
        if sc_site_mask[t] <= 0:
            continue
        site_norm = np.linalg.norm(sc_site_embedding[t])
        if site_norm <= 0:
            continue
        score = np.zeros(len(pathway_names), dtype=np.float32)
        for p in range(len(pathway_names)):
            if path_norm[p] <= 0:
                continue
            score[p] = float(np.dot(sc_site_embedding[t], sc_pathway_embedding[p]) / max(site_norm * path_norm[p], 1e-8))
        score = np.maximum(score, 0.0)
        if score.sum() <= 0:
            symbols = set(target_symbols_for_transfer(row))
            for p, pathway in enumerate(pathway_names):
                genes = set(
                    manifest.loc[manifest["pathway"].astype(str).eq(str(pathway)), "gene"]
                    .astype(str)
                    .str.upper()
                    .tolist()
                )
                if symbols & genes:
                    score[p] = 1.0
        if score.sum() <= 0:
            continue
        score = score / score.sum()
        transfer[t] = score
        for p, val in enumerate(score):
            if val > 0:
                transfer_rows.append(
                    {
                        "target_id": row["target_id"],
                        "target_index": int(row["target_index"]),
                        "protein_symbol": row.get("protein_symbol", ""),
                        "pathway": pathway_names[p],
                        "scp682_main_transfer_weight": float(val),
                    }
                )

    return {
        "pathway_embedding": sc_pathway_embedding.astype(np.float32),
        "site_embedding": sc_site_embedding.astype(np.float32),
        "site_mask": sc_site_mask.astype(np.float32),
        "pathway_alignment": pd.DataFrame(align_rows),
        "site_matches": pd.DataFrame(site_rows),
        "transfer_prior": transfer.astype(np.float32),
        "transfer_rows": pd.DataFrame(transfer_rows),
        "n_model_files": int(arr["n_model_files"][0]) if "n_model_files" in arr.files else 1,
        "bulk_dim": int(site_query.shape[1]),
        "source_model": str(arr["source_model"][0]) if "source_model" in arr.files else "SCP682_PORTABLE",
    }


def mix_pathway_priors(local_prior, transfer_prior, alpha):
    if alpha <= 0:
        return local_prior.astype(np.float32)
    prior = local_prior.astype(np.float32) + float(alpha) * transfer_prior.astype(np.float32)
    row_sum = prior.sum(axis=1, keepdims=True)
    prior = np.divide(prior, np.maximum(row_sum, 1e-8), out=np.zeros_like(prior), where=row_sum > 0)
    return prior.astype(np.float32)


def normalize_site_graph(weight):
    weight = np.asarray(weight, dtype=np.float32)
    np.fill_diagonal(weight, np.maximum(np.diag(weight), 1.0))
    degree = weight.sum(axis=1, keepdims=True)
    return np.divide(weight, np.maximum(degree, 1e-8), out=np.zeros_like(weight), where=degree > 0).astype(np.float32)


def add_site_graph_edge(weight, edge_rows, source_counts, i, j, value, source, target_rows):
    if i == j or not np.isfinite(value) or value <= 0:
        return
    value = float(value)
    weight[i, j] += value
    weight[j, i] += value
    source_counts[source] = source_counts.get(source, 0) + 2
    edge_rows.append(
        {
            "source": source,
            "target_id_1": target_rows[i]["target_id"],
            "target_id_2": target_rows[j]["target_id"],
            "weight": value,
        }
    )


def build_scnet_site_graph_prior(target_rows, prior_root, topk_per_target=12, min_weight=0.0, candidate_limit=96):
    prior_root = Path(prior_root)
    n_targets = len(target_rows)
    weight = np.zeros((n_targets, n_targets), dtype=np.float32)
    edge_rows = []
    source_counts = {}

    map_path = prior_root / "copheemap_20260519_scp682_ppko_v6" / "tables" / "model_site_cophee_map.tsv"
    if not map_path.exists():
        binary = np.eye(n_targets, dtype=np.float32)
        return {
            "adjacency": normalize_site_graph(binary.copy()),
            "binary": binary,
            "edges": pd.DataFrame(edge_rows),
            "candidates": pd.DataFrame(),
            "summary": {"available": False, "reason": f"missing {map_path}", "n_edges": 0},
        }

    site_map = pd.read_csv(map_path, sep="\t", low_memory=False)
    site_map["target_index"] = site_map["target_index"].astype(int)
    site_map["molecule_upper"] = site_map["molecule"].fillna("").astype(str).str.upper()
    site_map["site_upper"] = site_map["cophee_site"].fillna("").astype(str).str.upper()
    site_map["match_score"] = pd.to_numeric(site_map.get("match_score", 1.0), errors="coerce").fillna(0.0)

    candidates = {}
    candidate_rows = []
    for t, row in enumerate(target_rows):
        symbols = set(target_symbols_for_transfer(row))
        residues = set(target_residues_for_transfer(row))
        sub = site_map[site_map["molecule_upper"].isin(symbols)].copy() if symbols else site_map.iloc[0:0].copy()
        match_type = "parent_gene"
        if residues and not sub.empty:
            exact = sub[sub["site_upper"].isin(residues)].copy()
            if not exact.empty:
                sub = exact
                match_type = "gene_residue"
        if not sub.empty:
            sub = sub.sort_values("match_score", ascending=False).drop_duplicates("target_index").head(int(candidate_limit))
            idx = sub["target_index"].astype(int).tolist()
        else:
            idx = []
            match_type = "none"
        candidates[t] = set(idx)
        for bulk_idx in idx:
            candidate_rows.append(
                {
                    "target_order": t,
                    "target_id": row["target_id"],
                    "protein_symbol": row.get("protein_symbol", ""),
                    "match_type": match_type,
                    "bulk_target_index": int(bulk_idx),
                }
            )

    bulk_to_sc = {}
    for t, vals in candidates.items():
        for idx in vals:
            bulk_to_sc.setdefault(int(idx), []).append(t)

    cophee_path = prior_root / "copheemap_20260519_scp682_ppko_v6" / "tables" / "copheemap_model_site_site_edges.tsv"
    if cophee_path.exists() and bulk_to_sc:
        cophee = pd.read_csv(cophee_path, sep="\t", usecols=["target_index_1", "target_index_2", "edge_weight"], low_memory=False)
        for r in cophee.itertuples(index=False):
            left = bulk_to_sc.get(int(r.target_index_1), [])
            right = bulk_to_sc.get(int(r.target_index_2), [])
            if not left or not right:
                continue
            val = math.log1p(float(r.edge_weight))
            for i in left:
                for j in right:
                    add_site_graph_edge(weight, edge_rows, source_counts, i, j, val, "CoPheeMap", target_rows)

    kinase_sources = [
        (
            "CoPheeKSA",
            prior_root / "copheemap_20260519_scp682_ppko_v6" / "tables" / "copheeksa_model_kinase_site_edges.tsv",
            "score",
        ),
        (
            "KSTAR",
            prior_root / "kstar_20260519" / "scp682_ppko_v5_kstar_kinase_site_edges.tsv",
            "edge_frequency",
        ),
    ]
    for source, path, score_col in kinase_sources:
        if not path.exists() or not bulk_to_sc:
            continue
        kin = pd.read_csv(path, sep="\t", low_memory=False)
        if "target_index" not in kin.columns or "kinase" not in kin.columns:
            continue
        kin["target_index"] = pd.to_numeric(kin["target_index"], errors="coerce")
        kin = kin.dropna(subset=["target_index", "kinase"])
        if score_col not in kin.columns:
            kin[score_col] = 1.0
        kin[score_col] = pd.to_numeric(kin[score_col], errors="coerce").fillna(0.0)
        for _, sub in kin.groupby("kinase"):
            sc_score = {}
            for r in sub.itertuples(index=False):
                sc_targets = bulk_to_sc.get(int(getattr(r, "target_index")), [])
                if not sc_targets:
                    continue
                val = float(getattr(r, score_col))
                for t in sc_targets:
                    sc_score[t] = max(sc_score.get(t, 0.0), val)
            ids = sorted(sc_score)
            if len(ids) < 2:
                continue
            for a_pos in range(len(ids)):
                for b_pos in range(a_pos + 1, len(ids)):
                    i, j = ids[a_pos], ids[b_pos]
                    val = math.sqrt(max(sc_score[i], 0.0) * max(sc_score[j], 0.0))
                    add_site_graph_edge(weight, edge_rows, source_counts, i, j, val, source, target_rows)

    if topk_per_target and topk_per_target > 0:
        trimmed = np.zeros_like(weight)
        for i in range(n_targets):
            row = weight[i].copy()
            row[i] = 0.0
            keep = np.argsort(-row)[: int(topk_per_target)]
            keep = [j for j in keep if row[j] > float(min_weight)]
            trimmed[i, keep] = row[keep]
        weight = np.maximum(trimmed, trimmed.T)

    binary = (weight > 0).astype(np.float32)
    np.fill_diagonal(binary, 1.0)
    adjacency = normalize_site_graph(np.log1p(weight) + np.eye(n_targets, dtype=np.float32))
    summary = {
        "available": True,
        "n_targets": int(n_targets),
        "n_edges": int(binary.sum() - n_targets),
        "source_counts": source_counts,
        "candidate_matches": int(len(candidate_rows)),
        "targets_with_candidate": int(sum(len(v) > 0 for v in candidates.values())),
        "topk_per_target": int(topk_per_target),
        "min_weight": float(min_weight),
    }
    edges = pd.DataFrame(edge_rows)
    if not edges.empty:
        edges = edges.groupby(["source", "target_id_1", "target_id_2"], as_index=False)["weight"].sum()
    return {
        "adjacency": adjacency.astype(np.float32),
        "binary": binary.astype(np.float32),
        "edges": edges,
        "candidates": pd.DataFrame(candidate_rows),
        "summary": summary,
    }


def add_expanded_edge(edge_weight, edge_rows, source_counts, node_a, node_b, value, source, node_table):
    if node_a == node_b or not np.isfinite(value) or value <= 0:
        return
    value = float(value)
    a, b = int(node_a), int(node_b)
    key = (a, b, source)
    edge_weight[key] = max(edge_weight.get(key, 0.0), value)
    key_rev = (b, a, source)
    edge_weight[key_rev] = max(edge_weight.get(key_rev, 0.0), value)
    source_counts[source] = source_counts.get(source, 0) + 2
    edge_rows.append(
        {
            "source": source,
            "node_1": a,
            "node_2": b,
            "node_1_label": node_table[a]["label"],
            "node_2_label": node_table[b]["label"],
            "weight": value,
        }
    )


def build_expanded_scnet_site_graph_prior(target_rows, prior_root, candidate_limit=96, max_aux_nodes=12000):
    prior_root = Path(prior_root)
    n_targets = len(target_rows)
    node_table = [
        {
            "node_index": i,
            "node_type": "sc_target",
            "label": row["target_id"],
            "target_order": i,
            "bulk_target_index": "",
            "protein_symbol": row.get("protein_symbol", ""),
        }
        for i, row in enumerate(target_rows)
    ]
    edge_weight = {}
    edge_rows = []
    source_counts = {}

    map_path = prior_root / "copheemap_20260519_scp682_ppko_v6" / "tables" / "model_site_cophee_map.tsv"
    if not map_path.exists():
        idx = np.arange(n_targets, dtype=np.int64)
        return {
            "edge_index": np.vstack([idx, idx]).astype(np.int64),
            "edge_weight": np.ones(n_targets, dtype=np.float32),
            "node_table": pd.DataFrame(node_table),
            "edges": pd.DataFrame(edge_rows),
            "candidates": pd.DataFrame(),
            "summary": {"available": False, "reason": f"missing {map_path}", "n_graph_nodes": n_targets, "n_edges": n_targets},
        }

    site_map = pd.read_csv(map_path, sep="\t", low_memory=False)
    site_map["target_index"] = site_map["target_index"].astype(int)
    site_map["molecule_upper"] = site_map["molecule"].fillna("").astype(str).str.upper()
    site_map["site_upper"] = site_map["cophee_site"].fillna("").astype(str).str.upper()
    site_map["match_score"] = pd.to_numeric(site_map.get("match_score", 1.0), errors="coerce").fillna(0.0)
    bulk_label = {}
    bulk_gene = {}
    bulk_site = {}
    for r in site_map.itertuples(index=False):
        idx = int(getattr(r, "target_index"))
        bulk_label[idx] = str(getattr(r, "target_id"))
        bulk_gene[idx] = str(getattr(r, "molecule")).upper()
        bulk_site[idx] = str(getattr(r, "cophee_site")).upper()

    bulk_to_node = {}

    def get_bulk_node(bulk_idx):
        bulk_idx = int(bulk_idx)
        if bulk_idx in bulk_to_node:
            return bulk_to_node[bulk_idx]
        node_i = len(node_table)
        bulk_to_node[bulk_idx] = node_i
        node_table.append(
            {
                "node_index": node_i,
                "node_type": "bulk_aux_site",
                "label": bulk_label.get(bulk_idx, f"bulk_site_{bulk_idx}"),
                "target_order": "",
                "bulk_target_index": bulk_idx,
                "protein_symbol": bulk_gene.get(bulk_idx, ""),
            }
        )
        return node_i

    seed_bulk = set()
    candidate_rows = []
    for t, row in enumerate(target_rows):
        symbols = set(target_symbols_for_transfer(row))
        residues = set(target_residues_for_transfer(row))
        sub = site_map[site_map["molecule_upper"].isin(symbols)].copy() if symbols else site_map.iloc[0:0].copy()
        match_type = "parent_gene"
        if residues and not sub.empty:
            exact = sub[sub["site_upper"].isin(residues)].copy()
            if not exact.empty:
                sub = exact
                match_type = "gene_residue"
        if not sub.empty:
            sub = sub.sort_values("match_score", ascending=False).drop_duplicates("target_index").head(int(candidate_limit))
        else:
            sub = site_map.iloc[0:0].copy()
            match_type = "none"
        for r in sub.itertuples(index=False):
            bulk_idx = int(getattr(r, "target_index"))
            seed_bulk.add(bulk_idx)
            aux_node = get_bulk_node(bulk_idx)
            score = max(float(getattr(r, "match_score")), 0.05)
            add_expanded_edge(edge_weight, edge_rows, source_counts, t, aux_node, score, "target_to_bulk_candidate", node_table)
            candidate_rows.append(
                {
                    "target_order": t,
                    "target_id": row["target_id"],
                    "protein_symbol": row.get("protein_symbol", ""),
                    "match_type": match_type,
                    "bulk_target_index": bulk_idx,
                    "bulk_target_id": bulk_label.get(bulk_idx, f"bulk_site_{bulk_idx}"),
                    "match_score": score,
                }
            )

    cophee_path = prior_root / "copheemap_20260519_scp682_ppko_v6" / "tables" / "copheemap_model_site_site_edges.tsv"
    if cophee_path.exists() and seed_bulk:
        cophee = pd.read_csv(cophee_path, sep="\t", usecols=["target_index_1", "target_index_2", "edge_weight"], low_memory=False)
        for r in cophee.itertuples(index=False):
            a = int(r.target_index_1)
            b = int(r.target_index_2)
            if a not in seed_bulk and b not in seed_bulk:
                continue
            if len(node_table) >= max_aux_nodes + n_targets and (a not in bulk_to_node or b not in bulk_to_node):
                continue
            na = get_bulk_node(a)
            nb = get_bulk_node(b)
            add_expanded_edge(edge_weight, edge_rows, source_counts, na, nb, math.log1p(float(r.edge_weight)), "CoPheeMap_onehop", node_table)

    kinase_sources = [
        (
            "CoPheeKSA_onehop",
            prior_root / "copheemap_20260519_scp682_ppko_v6" / "tables" / "copheeksa_model_kinase_site_edges.tsv",
            "score",
        ),
        (
            "KSTAR_onehop",
            prior_root / "kstar_20260519" / "scp682_ppko_v5_kstar_kinase_site_edges.tsv",
            "edge_frequency",
        ),
    ]
    for source, path, score_col in kinase_sources:
        if not path.exists() or not seed_bulk:
            continue
        kin = pd.read_csv(path, sep="\t", low_memory=False)
        if "target_index" not in kin.columns or "kinase" not in kin.columns:
            continue
        kin["target_index"] = pd.to_numeric(kin["target_index"], errors="coerce")
        kin = kin.dropna(subset=["target_index", "kinase"])
        if score_col not in kin.columns:
            kin[score_col] = 1.0
        kin[score_col] = pd.to_numeric(kin[score_col], errors="coerce").fillna(0.0)
        for _, sub in kin.groupby("kinase"):
            vals = [(int(r.target_index), float(getattr(r, score_col))) for r in sub.itertuples(index=False)]
            seeds = [(idx, val) for idx, val in vals if idx in seed_bulk]
            if not seeds:
                continue
            for seed_idx, seed_score in seeds:
                seed_node = get_bulk_node(seed_idx)
                for idx, val in vals:
                    if idx == seed_idx:
                        continue
                    if len(node_table) >= max_aux_nodes + n_targets and idx not in bulk_to_node:
                        continue
                    node = get_bulk_node(idx)
                    w = math.sqrt(max(seed_score, 0.0) * max(val, 0.0))
                    add_expanded_edge(edge_weight, edge_rows, source_counts, seed_node, node, w, source, node_table)

    for i in range(len(node_table)):
        key = (i, i, "self")
        edge_weight[key] = 1.0

    merged = {}
    for (a, b, source), val in edge_weight.items():
        key = (a, b)
        merged[key] = max(merged.get(key, 0.0), val)
    edge_index = np.asarray(sorted(merged), dtype=np.int64).T
    raw_weight = np.asarray([merged[tuple(x)] for x in edge_index.T], dtype=np.float32)
    src = edge_index[0]
    degree = np.bincount(src, weights=raw_weight, minlength=len(node_table)).astype(np.float32)
    norm_weight = raw_weight / np.maximum(degree[src], 1e-8)
    edge_table = pd.DataFrame(edge_rows)
    if not edge_table.empty:
        edge_table = edge_table.groupby(["source", "node_1", "node_2", "node_1_label", "node_2_label"], as_index=False)["weight"].max()
    summary = {
        "available": True,
        "n_sc_targets": int(n_targets),
        "n_graph_nodes": int(len(node_table)),
        "n_aux_nodes": int(len(node_table) - n_targets),
        "n_edges": int(edge_index.shape[1]),
        "n_nonself_edges": int(edge_index.shape[1] - len(node_table)),
        "seed_bulk_sites": int(len(seed_bulk)),
        "candidate_matches": int(len(candidate_rows)),
        "source_counts": source_counts,
        "max_aux_nodes": int(max_aux_nodes),
    }
    return {
        "edge_index": edge_index.astype(np.int64),
        "edge_weight": norm_weight.astype(np.float32),
        "node_table": pd.DataFrame(node_table),
        "edges": edge_table,
        "candidates": pd.DataFrame(candidate_rows),
        "summary": summary,
    }


def build_train_idx(meta, mask, args):
    in_train = meta["dataset_id"].astype(str).isin(parse_list(args.train_datasets)).to_numpy()
    keep = in_train & mask.any(axis=1)
    control_only = set(parse_list(getattr(args, "train_control_only_datasets", "")))
    if control_only:
        ds = meta["dataset_id"].astype(str)
        control_ds = ds.isin(control_only).to_numpy()
        if "ibrutinib" in meta.columns:
            drug = parse_bool(meta["ibrutinib"]).to_numpy()
            keep &= ~(control_ds & drug)
    return np.flatnonzero(keep)


def transform_targets(raw, mask, train_idx, mode, target_rows):
    out = np.zeros(raw.shape, dtype=np.float32)
    stats = []
    for j, row in enumerate(target_rows):
        idx = np.asarray(train_idx, dtype=np.int64)[mask[train_idx, j] & np.isfinite(raw[train_idx, j])]
        vals = raw[idx, j].astype(np.float64)
        stat = {"target_order": j, **row, "mode": mode, "n_train_observed": int(len(vals))}
        obs = np.flatnonzero(mask[:, j] & np.isfinite(raw[:, j]))
        if len(vals) == 0:
            if len(obs):
                obs_vals = raw[obs, j].astype(np.float32)
                out[obs, j] = obs_vals
                stat.update(
                    {
                        "mode": "raw_no_train_reference",
                        "mean": float(np.mean(obs_vals)),
                        "sd": float(np.std(obs_vals)),
                        "min": float(np.min(obs_vals)),
                        "max": float(np.max(obs_vals)),
                    }
                )
            stats.append(stat)
            continue
        if mode == "zscore":
            mean = float(np.mean(vals))
            sd = float(np.std(vals))
            if sd <= 0:
                sd = 1.0
            out[obs, j] = ((raw[obs, j].astype(np.float32) - mean) / sd).astype(np.float32)
            stat.update({"mean": mean, "sd": sd, "min": float(np.min(vals)), "max": float(np.max(vals))})
        elif mode == "percentile":
            sorted_vals = np.sort(vals.astype(np.float32))
            pct = (np.searchsorted(sorted_vals, raw[obs, j], side="right").astype(np.float32) - 0.5) / max(len(sorted_vals), 1)
            out[obs, j] = np.clip(pct, 0.0, 1.0)
            stat.update({"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "min": float(sorted_vals[0]), "max": float(sorted_vals[-1])})
        else:
            out[obs, j] = raw[obs, j].astype(np.float32)
            stat.update({"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))})
        stats.append(stat)
    return out, pd.DataFrame(stats)


class SiteGraphRefiner(nn.Module):
    def __init__(self, hidden, n_layers=2, dropout=0.10):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, site_embedding, edge_index, edge_weight):
        z = site_embedding
        src = edge_index[0]
        dst = edge_index[1]
        w = edge_weight.to(z.dtype).unsqueeze(-1)
        for layer in self.layers:
            neigh = torch.zeros_like(z)
            neigh.index_add_(0, dst, z.index_select(0, src) * w)
            z = z + layer(neigh)
        return self.norm(z)


class ScFoundationPathwayPredictor(nn.Module):
    def __init__(
        self,
        n_pathways,
        n_targets,
        d_input,
        target_pathway_prior,
        hidden=384,
        n_layers=2,
        n_heads=4,
        dropout=0.15,
        bulk_pathway_embedding=None,
        bulk_site_embedding=None,
        bulk_site_mask=None,
        full_transfer_scale=0.0,
        site_graph_edge_index=None,
        site_graph_edge_weight=None,
        n_graph_nodes=None,
        site_graph_scale=0.25,
        disable_pathway_attention=False,
    ):
        super().__init__()
        self.n_pathways = int(n_pathways)
        self.hidden = int(hidden)
        self.disable_pathway_attention = bool(disable_pathway_attention)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(d_input),
            nn.Linear(d_input, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pathway_embedding = nn.Parameter(torch.randn(n_pathways, hidden) * 0.02)
        self.cell_to_pathway = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, n_pathways * hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pathway_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.site_embedding = nn.Parameter(torch.randn(n_targets, hidden) * 0.02)
        prior = torch.as_tensor(target_pathway_prior, dtype=torch.float32)
        if prior.shape != (n_targets, n_pathways):
            raise ValueError(f"target_pathway_prior shape {tuple(prior.shape)} does not match {(n_targets, n_pathways)}")
        self.register_buffer("target_pathway_prior", prior)
        self.site_pathway_embedding = nn.Parameter(torch.randn(n_pathways, hidden) * 0.02)
        self.site_context = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.cross_attention = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        if n_graph_nodes is None:
            n_graph_nodes = n_targets
        if site_graph_edge_index is None:
            idx = np.arange(n_graph_nodes, dtype=np.int64)
            site_graph_edge_index = np.vstack([idx, idx])
            site_graph_edge_weight = np.ones(n_graph_nodes, dtype=np.float32)
        if site_graph_edge_weight is None:
            site_graph_edge_weight = np.ones(np.asarray(site_graph_edge_index).shape[1], dtype=np.float32)
        self.n_graph_nodes = int(n_graph_nodes)
        self.n_aux_nodes = max(0, self.n_graph_nodes - n_targets)
        self.register_buffer("site_graph_edge_index", torch.as_tensor(site_graph_edge_index, dtype=torch.long))
        self.register_buffer("site_graph_edge_weight", torch.as_tensor(site_graph_edge_weight, dtype=torch.float32))
        graph_offdiag = float(np.asarray(site_graph_edge_index).shape[1] - self.n_graph_nodes)
        self.register_buffer("site_graph_enabled", torch.tensor(1.0 if graph_offdiag > 0 else 0.0))
        self.aux_site_embedding = nn.Parameter(torch.randn(self.n_aux_nodes, hidden) * 0.02)
        self.site_graph_refiner = SiteGraphRefiner(hidden, n_layers=2, dropout=dropout)
        self.site_graph_scale = nn.Parameter(torch.tensor(float(site_graph_scale), dtype=torch.float32))
        self.site_graph_output_scale = nn.Parameter(torch.tensor(float(site_graph_scale), dtype=torch.float32))
        self.site_graph_context = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.site_graph_gate = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.full_transfer_enabled = bool(
            bulk_pathway_embedding is not None
            and bulk_site_embedding is not None
            and float(full_transfer_scale) > 0
        )
        if self.full_transfer_enabled:
            bulk_pathway = torch.as_tensor(bulk_pathway_embedding, dtype=torch.float32)
            bulk_site = torch.as_tensor(bulk_site_embedding, dtype=torch.float32)
            if bulk_pathway.ndim != 2 or bulk_pathway.shape[0] != n_pathways:
                raise ValueError(f"bulk_pathway_embedding shape {tuple(bulk_pathway.shape)} does not match n_pathways={n_pathways}")
            if bulk_site.ndim != 2 or bulk_site.shape[0] != n_targets:
                raise ValueError(f"bulk_site_embedding shape {tuple(bulk_site.shape)} does not match n_targets={n_targets}")
            site_mask = torch.as_tensor(bulk_site_mask, dtype=torch.float32).view(-1, 1) if bulk_site_mask is not None else torch.ones(n_targets, 1)
            self.register_buffer("bulk_pathway_embedding", bulk_pathway)
            self.register_buffer("bulk_site_embedding", bulk_site)
            self.register_buffer("bulk_site_mask", site_mask)
            bulk_dim = int(bulk_pathway.shape[1])
            self.bulk_pathway_proj = nn.Sequential(nn.LayerNorm(bulk_dim), nn.Linear(bulk_dim, hidden), nn.GELU())
            self.bulk_site_proj = nn.Sequential(nn.LayerNorm(bulk_dim), nn.Linear(bulk_dim, hidden), nn.GELU())
            self.bulk_pathway_scale = nn.Parameter(torch.tensor(float(full_transfer_scale), dtype=torch.float32))
            self.bulk_site_scale = nn.Parameter(torch.tensor(float(full_transfer_scale), dtype=torch.float32))
        else:
            self.register_buffer("bulk_pathway_embedding", torch.zeros(n_pathways, 1))
            self.register_buffer("bulk_site_embedding", torch.zeros(n_targets, 1))
            self.register_buffer("bulk_site_mask", torch.zeros(n_targets, 1))
            self.bulk_pathway_proj = None
            self.bulk_site_proj = None
            self.bulk_pathway_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.bulk_site_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.site_norm = nn.LayerNorm(hidden)
        self.site_ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def site_graph_loss(self):
        if float(self.site_graph_enabled.detach().cpu()) <= 0:
            return self.site_embedding.sum() * 0
        pathway_site_prior = torch.matmul(self.target_pathway_prior, self.site_pathway_embedding)
        site_embedding = self.site_embedding + pathway_site_prior
        if self.n_aux_nodes > 0:
            graph_input = torch.cat([site_embedding, self.aux_site_embedding], dim=0)
        else:
            graph_input = site_embedding
        graph_site = self.site_graph_refiner(graph_input, self.site_graph_edge_index, self.site_graph_edge_weight)
        z = F.normalize(graph_site, p=2, dim=-1)
        edge = self.site_graph_edge_index
        nonself = edge[0] != edge[1]
        pos_edge = edge[:, nonself]
        if pos_edge.shape[1] == 0:
            return z.sum() * 0
        max_pos = min(pos_edge.shape[1], 200000)
        if pos_edge.shape[1] > max_pos:
            keep = torch.randperm(pos_edge.shape[1], device=pos_edge.device)[:max_pos]
            pos_edge = pos_edge[:, keep]
        pos_score = (z.index_select(0, pos_edge[0]) * z.index_select(0, pos_edge[1])).sum(dim=-1) / 0.2
        n_neg = pos_edge.shape[1]
        neg_src = torch.randint(0, z.shape[0], (n_neg,), device=z.device)
        neg_dst = torch.randint(0, z.shape[0], (n_neg,), device=z.device)
        neg_score = (z.index_select(0, neg_src) * z.index_select(0, neg_dst)).sum(dim=-1) / 0.2
        pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return pos_loss + 0.25 * neg_loss

    def encode(self, x, present):
        cell = self.input_proj(x)
        token_base = self.pathway_embedding
        if self.full_transfer_enabled:
            bulk_path = self.bulk_pathway_proj(self.bulk_pathway_embedding)
            token_base = token_base + torch.tanh(self.bulk_pathway_scale) * bulk_path
        h = token_base.unsqueeze(0) + self.cell_to_pathway(cell).view(x.shape[0], self.n_pathways, self.hidden)
        effective_present = present.clone()
        all_missing = effective_present.sum(dim=1) <= 0
        if bool(all_missing.any()):
            effective_present[all_missing] = 1.0
        key_padding_mask = effective_present <= 0
        h = self.pathway_encoder(h, src_key_padding_mask=key_padding_mask)
        token_mask = (effective_present > 0).to(h.dtype)
        return h * token_mask.unsqueeze(-1)

    def phospho_from_state(self, h, present, return_attention=False):
        b = h.shape[0]
        pathway_site_prior = torch.matmul(self.target_pathway_prior, self.site_pathway_embedding)
        site_embedding = self.site_embedding + pathway_site_prior
        if self.full_transfer_enabled:
            bulk_site = self.bulk_site_proj(self.bulk_site_embedding) * self.bulk_site_mask
            site_embedding = site_embedding + torch.tanh(self.bulk_site_scale) * bulk_site
        if self.n_aux_nodes > 0:
            graph_input = torch.cat([site_embedding, self.aux_site_embedding], dim=0)
        else:
            graph_input = site_embedding
        graph_all = self.site_graph_refiner(graph_input, self.site_graph_edge_index, self.site_graph_edge_weight)
        graph_site = graph_all[: site_embedding.shape[0]]
        site_embedding = site_embedding + self.site_graph_enabled * torch.tanh(self.site_graph_scale) * graph_site
        query = site_embedding.unsqueeze(0).expand(b, -1, -1)
        query = query + self.site_context(site_embedding).unsqueeze(0)
        effective_present = present.clone()
        all_missing = effective_present.sum(dim=1) <= 0
        if bool(all_missing.any()):
            effective_present[all_missing] = 1.0
        key_padding_mask = effective_present <= 0
        if self.disable_pathway_attention:
            pooled = (h * (effective_present > 0).to(h.dtype).unsqueeze(-1)).sum(dim=1)
            denom = (effective_present > 0).to(h.dtype).sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            pooled = pooled / denom
            site = query + pooled.unsqueeze(1)
            if return_attention:
                weights = torch.zeros(b, query.shape[1], h.shape[1], dtype=h.dtype, device=h.device)
        else:
            site, weights = self.cross_attention(
                query,
                h,
                h,
                key_padding_mask=key_padding_mask,
                need_weights=return_attention,
                average_attn_weights=True,
            )
        site = self.site_norm(site + self.site_ffn(site))
        graph_context = self.site_graph_context(graph_site).unsqueeze(0).expand(b, -1, -1)
        graph_gate = torch.sigmoid(self.site_graph_gate(torch.cat([site, graph_context], dim=-1)))
        site = site + self.site_graph_enabled * torch.tanh(self.site_graph_output_scale) * graph_gate * graph_context
        pred = self.head(site).squeeze(-1)
        if return_attention:
            return pred, weights
        return pred

    def teacher_attention(self):
        if not self.full_transfer_enabled:
            return None
        bulk_path = self.bulk_pathway_proj(self.bulk_pathway_embedding)
        bulk_site = self.bulk_site_proj(self.bulk_site_embedding) * self.bulk_site_mask
        score = torch.matmul(bulk_site, bulk_path.T) / math.sqrt(bulk_path.shape[-1])
        return torch.softmax(score, dim=-1)

    def forward(self, x, present, return_attention=False):
        h = self.encode(x, present)
        if return_attention:
            pred, weights = self.phospho_from_state(h, present, True)
            return pred, h, weights
        return self.phospho_from_state(h, present), h


def target_loss_weights(mask, train_idx, max_weight):
    obs = mask[train_idx].sum(axis=0).astype(np.float64)
    med = np.median(obs[obs > 0]) if np.any(obs > 0) else 1.0
    weights = np.sqrt(med / np.maximum(obs, 1.0))
    weights = np.clip(weights, 1.0 / max_weight, max_weight)
    return weights.astype(np.float32)


def masked_loss(pred, y, mask, target_weights, beta, loss_type):
    if int(mask.sum()) == 0:
        return pred.sum() * 0
    err = pred - y
    if loss_type == "huber":
        abs_err = err.abs()
        loss = torch.where(abs_err < beta, 0.5 * err.square() / beta, abs_err - 0.5 * beta)
    else:
        loss = err.square()
    weights = target_weights.view(1, -1).expand_as(loss)
    loss = loss[mask] * weights[mask]
    denom = weights[mask].sum().clamp_min(1.0)
    return loss.sum() / denom


def masked_token_mean(x, present):
    w = present.clamp_min(0.0).to(x.dtype)
    denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * w.unsqueeze(-1)).sum(dim=1) / denom


def embedding_prior_loss(x, present, h, args):
    if args.prior_weight <= 0 or x.shape[0] < 4:
        return h.sum() * 0
    with torch.no_grad():
        if x.ndim == 2:
            raw_state = torch.nn.functional.normalize(x, dim=1)
        else:
            raw_state = torch.nn.functional.normalize(masked_token_mean(x, present), dim=1)
        raw_sim = raw_state @ raw_state.T
        n = raw_sim.shape[0]
        raw_sim.fill_diagonal_(-1e4)
        k = min(int(args.prior_neighbors), n - 1)
        if k <= 0:
            return h.sum() * 0
        target_sim, nn_idx = torch.topk(raw_sim, k=k, dim=1)
        valid = target_sim > float(args.prior_min_similarity)
        weights = torch.softmax(target_sim / max(float(args.prior_temperature), 1e-4), dim=1)
        weights = weights * valid.to(weights.dtype)
    latent_state = torch.nn.functional.normalize(masked_token_mean(h, present), dim=1)
    latent_sim = latent_state @ latent_state.T
    pred_sim = latent_sim.gather(1, nn_idx)
    denom = weights.sum().clamp_min(1.0)
    return (((pred_sim - target_sim).square()) * weights).sum() / denom


def transfer_attention_loss(attn, transfer_prior):
    if transfer_prior is None or float(transfer_prior.sum().detach().cpu()) <= 0:
        return attn.sum() * 0
    active = transfer_prior.sum(dim=1) > 0
    if int(active.sum()) == 0:
        return attn.sum() * 0
    target = transfer_prior[active]
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    pred = attn[:, active, :].clamp_min(1e-8)
    return torch.mean((pred - target.unsqueeze(0)).square())


def teacher_distillation_loss(attn, model):
    teacher = model.teacher_attention()
    if teacher is None:
        return attn.sum() * 0
    active = model.bulk_site_mask.squeeze(-1) > 0
    if int(active.sum()) == 0:
        return attn.sum() * 0
    pred = attn[:, active, :].clamp_min(1e-8)
    target = teacher[active].clamp_min(1e-8)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.sum(target.unsqueeze(0) * (torch.log(target.unsqueeze(0)) - torch.log(pred)), dim=-1).mean()


def load_warm_start(model, root, model_path):
    if not model_path:
        return {"used": False, "reason": "empty_path", "matched": 0, "skipped": 0}
    path = Path(model_path)
    if not path.is_absolute():
        path = Path(root) / path
    if not path.exists():
        return {"used": False, "reason": f"missing:{path}", "matched": 0, "skipped": 0}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    own = model.state_dict()
    matched = {}
    skipped = 0
    for key, value in state.items():
        if key.startswith("target_pathway_prior") or key.startswith("bulk_"):
            skipped += 1
            continue
        if key in own and tuple(own[key].shape) == tuple(value.shape):
            matched[key] = value
        else:
            skipped += 1
    model.load_state_dict(matched, strict=False)
    return {"used": True, "path": str(path), "matched": len(matched), "skipped": skipped}


def build_prior_idx(meta, present, train_idx, args):
    if args.prior_weight <= 0:
        return np.asarray([], dtype=np.int64)
    if args.prior_datasets:
        keep = meta["dataset_id"].astype(str).isin(parse_list(args.prior_datasets)).to_numpy()
    elif args.prior_include_holdout_rna:
        ds = set(parse_list(args.train_datasets)) | set(parse_list(args.holdout_datasets))
        keep = meta["dataset_id"].astype(str).isin(ds).to_numpy()
    else:
        keep = np.zeros(len(meta), dtype=bool)
        keep[np.asarray(train_idx, dtype=np.int64)] = True
    keep &= np.asarray(present).sum(axis=1) > 0
    return np.flatnonzero(keep)


def balanced_epoch_indices(train_idx, meta, args, rng):
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if not args.balance_datasets:
        order = train_idx.copy()
        rng.shuffle(order)
        return order
    groups = []
    ds_values = meta.iloc[train_idx]["dataset_id"].astype(str).to_numpy()
    for ds in parse_list(args.train_datasets):
        arr = train_idx[ds_values == ds]
        if len(arr):
            groups.append(arr)
    if len(groups) <= 1:
        order = train_idx.copy()
        rng.shuffle(order)
        return order
    target_n = max(len(g) for g in groups)
    if args.balance_dataset_size > 0:
        target_n = min(target_n, args.balance_dataset_size)
    pieces = []
    for g in groups:
        replace = len(g) < target_n
        pieces.append(rng.choice(g, size=target_n, replace=replace))
    order = np.concatenate(pieces)
    rng.shuffle(order)
    return order


def sample_idx(idx, n, rng):
    idx = np.asarray(idx, dtype=np.int64)
    if n <= 0 or len(idx) <= n:
        return idx
    return np.sort(rng.choice(idx, size=n, replace=False))


def make_context(name, time_min, ctrl, drug, y, mask):
    real_delta = np.full(y.shape[1], np.nan, dtype=np.float32)
    n_ctrl = np.zeros(y.shape[1], dtype=np.int64)
    n_drug = np.zeros(y.shape[1], dtype=np.int64)
    for j in range(y.shape[1]):
        mc = mask[ctrl, j] & np.isfinite(y[ctrl, j])
        md = mask[drug, j] & np.isfinite(y[drug, j])
        n_ctrl[j] = int(mc.sum())
        n_drug[j] = int(md.sum())
        if mc.sum() and md.sum():
            real_delta[j] = float(np.mean(y[drug][md, j]) - np.mean(y[ctrl][mc, j]))
    return {
        "context": name,
        "time_min": float(time_min),
        "control_idx": np.asarray(ctrl, dtype=np.int64),
        "drug_idx": np.asarray(drug, dtype=np.int64),
        "real_delta": real_delta,
        "n_control_target": n_ctrl,
        "n_drug_target": n_drug,
        "n_control": int(len(ctrl)),
        "n_drug": int(len(drug)),
    }


def build_drug_contexts(meta, y, mask, args):
    time_min = pd.to_numeric(meta["time_min"], errors="coerce")
    bcr = parse_bool(meta["bcr_stim"])
    drug = parse_bool(meta["ibrutinib"])
    dataset = meta["dataset_id"].astype(str).eq(QURIE)
    rows = []
    for t in parse_list(args.context_times):
        tv = float(t)
        ctrl = np.flatnonzero(dataset.to_numpy() & time_min.eq(tv).to_numpy() & bcr.to_numpy() & (~drug.to_numpy()))
        dr = np.flatnonzero(dataset.to_numpy() & time_min.eq(tv).to_numpy() & bcr.to_numpy() & drug.to_numpy())
        if len(ctrl) and len(dr):
            rows.append(make_context(f"time{int(tv)}", tv, ctrl, dr, y, mask))
    if len(rows) >= 2:
        ctrl = np.concatenate([r["control_idx"] for r in rows])
        dr = np.concatenate([r["drug_idx"] for r in rows])
        rows.append(make_context("time6_180_pooled", 93.0, ctrl, dr, y, mask))
    return rows


@torch.no_grad()
def predict(model, features, present, idx, args, device, return_attention=False):
    preds = []
    attns = []
    model.eval()
    for start in range(0, len(idx), args.eval_batch_size):
        batch = idx[start : start + args.eval_batch_size]
        xb = torch.as_tensor(np.asarray(features[batch]), dtype=torch.float32, device=device)
        pb = torch.as_tensor(np.asarray(present[batch]), dtype=torch.float32, device=device)
        if return_attention:
            pred, _, attn = model(xb, pb, True)
            attns.append(attn.detach().cpu().numpy().astype(np.float32))
        else:
            pred, _ = model(xb, pb)
        preds.append(pred.detach().cpu().numpy().astype(np.float32))
    pred_all = np.vstack(preds) if preds else np.zeros((0, model.site_embedding.shape[0]), dtype=np.float32)
    if return_attention:
        attn_all = np.vstack(attns) if attns else np.zeros((0, model.site_embedding.shape[0], model.target_pathway_prior.shape[1]), dtype=np.float32)
        return pred_all, attn_all
    return pred_all


def score_prediction(y, mask, pred, meta, idx, target_rows, evaluation):
    rows = []
    idx = np.asarray(idx, dtype=np.int64)
    datasets = ["all"] + sorted(meta.iloc[idx]["dataset_id"].astype(str).unique().tolist())
    for ds in datasets:
        sub = np.ones(len(idx), dtype=bool) if ds == "all" else meta.iloc[idx]["dataset_id"].astype(str).eq(ds).to_numpy()
        if not sub.sum():
            continue
        sub_idx = idx[sub]
        for j, row in enumerate(target_rows):
            m = mask[sub_idx, j] & np.isfinite(y[sub_idx, j])
            if int(m.sum()) < 3:
                continue
            rows.append(
                {
                    "model": MODEL_NAME,
                    "evaluation": evaluation,
                    "test_dataset": ds,
                    "target_id": row["target_id"],
                    "target_index": int(row["target_index"]),
                    "n": int(m.sum()),
                    "spearman": safe_spearman(y[sub_idx, j][m], pred[sub, j][m]),
                    "pearson": safe_pearson(y[sub_idx, j][m], pred[sub, j][m]),
                }
            )
    return rows


@torch.no_grad()
def estimate_recon_loss(model, features, present, y, mask, idx, target_weights, args, device):
    model.eval()
    vals = []
    counts = []
    for start in range(0, len(idx), args.eval_batch_size):
        batch = idx[start : start + args.eval_batch_size]
        xb = torch.as_tensor(np.asarray(features[batch]), dtype=torch.float32, device=device)
        pb = torch.as_tensor(np.asarray(present[batch]), dtype=torch.float32, device=device)
        yb = torch.as_tensor(y[batch], dtype=torch.float32, device=device)
        mb = torch.as_tensor(mask[batch], dtype=torch.bool, device=device)
        pred, _ = model(xb, pb)
        loss = masked_loss(pred, yb, mb, target_weights, args.huber_beta, args.loss_type)
        vals.append(float(loss.detach().cpu()) * len(batch))
        counts.append(len(batch))
    return float(np.sum(vals) / max(np.sum(counts), 1))


def split_train_validation(train_idx, meta, args, rng):
    train_idx = np.asarray(train_idx, dtype=np.int64)
    cv_folds = int(getattr(args, "cv_folds", 0) or 0)
    cv_fold = int(getattr(args, "cv_fold", 0) or 0)
    if cv_folds > 1:
        if cv_fold < 1 or cv_fold > cv_folds:
            raise ValueError(f"cv_fold must be 1..{cv_folds}, got {cv_fold}")
        ds_values = meta.iloc[train_idx]["dataset_id"].astype(str).to_numpy()
        val_parts = []
        for ds in sorted(pd.unique(ds_values)):
            arr = train_idx[ds_values == ds]
            perm = rng.permutation(arr)
            splits = np.array_split(perm, cv_folds)
            val_parts.append(splits[cv_fold - 1])
        val_idx = np.sort(np.unique(np.concatenate(val_parts))) if val_parts else np.asarray([], dtype=np.int64)
        val_set = set(int(x) for x in val_idx)
        fit_idx = np.asarray([int(x) for x in train_idx if int(x) not in val_set], dtype=np.int64)
        return fit_idx, val_idx
    if args.val_fraction <= 0 or args.smoke:
        return train_idx, np.asarray([], dtype=np.int64)
    ds_values = meta.iloc[train_idx]["dataset_id"].astype(str).to_numpy()
    val_parts = []
    for ds in sorted(pd.unique(ds_values)):
        arr = train_idx[ds_values == ds]
        n = int(round(len(arr) * args.val_fraction))
        if n > 0:
            val_parts.append(rng.choice(arr, size=n, replace=False))
    if not val_parts:
        return train_idx, np.asarray([], dtype=np.int64)
    val_idx = np.sort(np.unique(np.concatenate(val_parts)))
    val_set = set(int(x) for x in val_idx)
    fit_idx = np.asarray([int(x) for x in train_idx if int(x) not in val_set], dtype=np.int64)
    if args.val_cells > 0 and len(val_idx) > args.val_cells:
        val_idx = sample_idx(val_idx, args.val_cells, rng)
    return fit_idx, val_idx


def train_model(features, present, y, mask, meta, contexts, pathway_names, target_rows, target_pathway_prior, transfer_prior, full_transfer, site_graph, args, out_dir):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    model = ScFoundationPathwayPredictor(
        len(pathway_names),
        y.shape[1],
        features.shape[1],
        target_pathway_prior,
        hidden=args.hidden,
        n_layers=args.pathway_layers,
        n_heads=args.attention_heads,
        dropout=args.dropout,
        bulk_pathway_embedding=full_transfer.get("pathway_embedding") if full_transfer else None,
        bulk_site_embedding=full_transfer.get("site_embedding") if full_transfer else None,
        bulk_site_mask=full_transfer.get("site_mask") if full_transfer else None,
        full_transfer_scale=args.full_transfer_scale,
        site_graph_edge_index=site_graph.get("edge_index") if site_graph else None,
        site_graph_edge_weight=site_graph.get("edge_weight") if site_graph else None,
        n_graph_nodes=int(site_graph.get("summary", {}).get("n_graph_nodes", y.shape[1])) if site_graph else y.shape[1],
        site_graph_scale=args.site_graph_scale,
        disable_pathway_attention=args.disable_pathway_attention,
    ).to(device)
    warm_start_info = load_warm_start(model, args.root, args.warm_start_model)
    if warm_start_info.get("used"):
        log(f"warm_start matched={warm_start_info['matched']} skipped={warm_start_info['skipped']} path={warm_start_info['path']}")
    elif args.warm_start_model:
        log(f"warm_start skipped reason={warm_start_info.get('reason')}")
    transfer_prior_t = torch.as_tensor(transfer_prior, dtype=torch.float32, device=device) if transfer_prior is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_idx = build_train_idx(meta, mask, args)
    if args.smoke:
        train_idx = sample_idx(train_idx, args.smoke_cells, rng)
    elif args.max_train_cells:
        train_idx = sample_idx(train_idx, args.max_train_cells, rng)
    train_idx = train_idx[mask[train_idx].any(axis=1)]
    fit_idx, val_idx = split_train_validation(train_idx, meta, args, rng)
    prior_idx = build_prior_idx(meta, present, train_idx, args)
    tw = torch.as_tensor(target_loss_weights(mask, fit_idx, args.max_target_weight), dtype=torch.float32, device=device)
    target_to_order = {r["target_id"]: i for i, r in enumerate(target_rows)}
    focus_idx = [target_to_order[t] for t in FOCUS_TARGETS if t in target_to_order]
    log(
        "fit cells={} val cells={} pathways={} targets={} contexts={} balanced={}".format(
            len(fit_idx), len(val_idx), len(pathway_names), len(target_rows), len(contexts), bool(args.balance_datasets)
        )
    )
    log(
        "prior cells={} prior_weight={} neighbors={} include_holdout_rna={}".format(
            len(prior_idx), float(args.prior_weight), int(args.prior_neighbors), bool(args.prior_include_holdout_rna)
        )
    )

    best_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = balanced_epoch_indices(fit_idx, meta, args, rng)
        total = 0.0
        recon_total = 0.0
        prior_total = 0.0
        transfer_total = 0.0
        teacher_total = 0.0
        site_graph_value = 0.0
        prior_seen = 0
        n_seen = 0
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            xb = torch.as_tensor(np.asarray(features[batch]), dtype=torch.float32, device=device)
            pb = torch.as_tensor(np.asarray(present[batch]), dtype=torch.float32, device=device)
            yb = torch.as_tensor(y[batch], dtype=torch.float32, device=device)
            mb = torch.as_tensor(mask[batch], dtype=torch.bool, device=device)
            if args.transfer_attention_weight > 0 or args.teacher_distill_weight > 0:
                pred, h, attn = model(xb, pb, True)
                transfer_loss = transfer_attention_loss(attn, transfer_prior_t)
                teacher_loss = teacher_distillation_loss(attn, model)
            else:
                pred, h = model(xb, pb)
                transfer_loss = pred.sum() * 0
                teacher_loss = pred.sum() * 0
            recon_loss = masked_loss(pred, yb, mb, tw, args.huber_beta, args.loss_type)
            prior_loss = embedding_prior_loss(xb, pb, h, args)
            loss = (
                args.recon_weight * recon_loss
                + args.prior_weight * prior_loss
                + args.transfer_attention_weight * transfer_loss
                + args.teacher_distill_weight * teacher_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total += float(loss.detach().cpu()) * len(batch)
            recon_total += float(recon_loss.detach().cpu()) * len(batch)
            prior_total += float(prior_loss.detach().cpu()) * len(batch)
            transfer_total += float(transfer_loss.detach().cpu()) * len(batch)
            teacher_total += float(teacher_loss.detach().cpu()) * len(batch)
            prior_seen += len(batch)
            n_seen += len(batch)
            batch_no = start // args.batch_size + 1
            if args.batch_log_interval > 0 and batch_no % args.batch_log_interval == 0:
                log(f"epoch={epoch} batch={batch_no} seen={n_seen} recon={float(recon_loss.detach().cpu()):.6f}")

        if args.site_graph_weight > 0:
            site_graph_loss = model.site_graph_loss()
            loss = args.site_graph_weight * site_graph_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            site_graph_value = float(site_graph_loss.detach().cpu())

        if args.prior_weight > 0 and args.prior_steps_per_epoch > 0 and len(prior_idx):
            for _ in range(args.prior_steps_per_epoch):
                replace = len(prior_idx) < args.prior_batch_size
                batch = rng.choice(prior_idx, size=min(args.prior_batch_size, len(prior_idx)), replace=replace)
                xb = torch.as_tensor(np.asarray(features[batch]), dtype=torch.float32, device=device)
                pb = torch.as_tensor(np.asarray(present[batch]), dtype=torch.float32, device=device)
                _, h = model(xb, pb)
                prior_loss = embedding_prior_loss(xb, pb, h, args)
                loss = args.prior_weight * prior_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                prior_total += float(prior_loss.detach().cpu()) * len(batch)
                prior_seen += len(batch)

        delta_value = 0.0
        delta_n = 0
        if args.delta_weight > 0 and contexts and epoch > args.warmup_epochs:
            for context in contexts:
                ctrl = sample_idx(context["control_idx"], args.context_cells, rng)
                dr = sample_idx(context["drug_idx"], args.context_cells, rng)
                if len(ctrl) < 5 or len(dr) < 5:
                    continue
                xb_ctrl = torch.as_tensor(np.asarray(features[ctrl]), dtype=torch.float32, device=device)
                pb_ctrl = torch.as_tensor(np.asarray(present[ctrl]), dtype=torch.float32, device=device)
                xb_drug = torch.as_tensor(np.asarray(features[dr]), dtype=torch.float32, device=device)
                pb_drug = torch.as_tensor(np.asarray(present[dr]), dtype=torch.float32, device=device)
                pred_ctrl, _ = model(xb_ctrl, pb_ctrl)
                pred_drug, _ = model(xb_drug, pb_drug)
                pred_delta = pred_drug.mean(dim=0) - pred_ctrl.mean(dim=0)
                real_delta = torch.as_tensor(context["real_delta"], dtype=torch.float32, device=device)
                usable = torch.isfinite(real_delta)
                if focus_idx and args.delta_focus_only:
                    fm = torch.zeros_like(usable)
                    fm[focus_idx] = True
                    usable = usable & fm
                if int(usable.sum()) == 0:
                    continue
                if args.loss_type == "huber":
                    loss_delta = torch.nn.functional.smooth_l1_loss(pred_delta[usable], real_delta[usable], beta=args.huber_beta)
                else:
                    loss_delta = torch.nn.functional.mse_loss(pred_delta[usable], real_delta[usable])
                if int(usable.sum()) >= 2 and args.delta_cosine_weight > 0:
                    cos = 1.0 - torch.nn.functional.cosine_similarity(
                        pred_delta[usable].unsqueeze(0),
                        real_delta[usable].unsqueeze(0),
                        dim=1,
                    ).mean()
                else:
                    cos = pred_delta.sum() * 0
                loss = args.delta_weight * loss_delta + args.delta_cosine_weight * cos
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                delta_value += float(loss.detach().cpu())
                delta_n += 1

        recon_value = recon_total / max(n_seen, 1)
        prior_value = prior_total / max(prior_seen, 1)
        transfer_value = transfer_total / max(n_seen, 1)
        teacher_value = teacher_total / max(n_seen, 1)
        epoch_loss = total / max(n_seen, 1) + delta_value / max(delta_n, 1)
        val_value = estimate_recon_loss(model, features, present, y, mask, val_idx, tw, args, device) if len(val_idx) else float("nan")
        monitor_loss = val_value if np.isfinite(val_value) else epoch_loss
        log(f"epoch={epoch} loss={epoch_loss:.6f} recon={recon_value:.6f} prior={prior_value:.6f} transfer={transfer_value:.6f} teacher={teacher_value:.6f} site_graph={site_graph_value:.6f} delta={delta_value / max(delta_n, 1):.6f} val_recon={val_value:.6f}")
        can_stop = epoch > args.warmup_epochs
        if monitor_loss < best_loss - 1e-5:
            best_loss = monitor_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1 if can_stop else 0
            if can_stop and stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    transfer_source = (
        "scp682_main"
        if (args.scp682_main_transfer_dir or args.scp682_main_pathway_token_transfer_dir)
        else ("scp68222" if args.scp68222_transfer_dir else "none")
    )
    teacher_name = (
        "scp682_main_graph_site_to_pathway_attention_distribution"
        if transfer_source == "scp682_main"
        else "scp682_22_frozen_site_query_to_pathway_attention_distribution"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "n_pathways": int(len(pathway_names)),
                "n_targets": int(y.shape[1]),
                "d_input": int(features.shape[1]),
                "hidden": int(args.hidden),
                "pathway_layers": int(args.pathway_layers),
                "attention_heads": int(args.attention_heads),
                "dropout": float(args.dropout),
                "target_transform": args.target_transform,
                "architecture": "scfoundation_cell_embedding_to_pathway_transformer_expanded_scnet_site_graph",
                "transfer": transfer_source,
                "site_graph": "CoPheeMap_CoPheeKSA_KSTAR_projected_to_single_cell_targets",
            },
            "args": vars(args),
            "pathway_names": pathway_names,
            "target_rows": target_rows,
            "best_loss": float(best_loss),
            "drug_delta_supervision_used": bool(args.delta_weight > 0),
            "warm_start": warm_start_info,
            "full_transfer_summary": {
                "enabled": bool(full_transfer),
                "n_model_files": int(full_transfer.get("n_model_files", 0)) if full_transfer else 0,
                "bulk_dim": int(full_transfer.get("bulk_dim", 0)) if full_transfer else 0,
                "n_site_matches": int(np.sum(full_transfer.get("site_mask", np.zeros(0)))) if full_transfer else 0,
                "full_transfer_scale": float(args.full_transfer_scale),
                "teacher_distill_weight": float(args.teacher_distill_weight),
                "teacher_distillation": teacher_name,
            },
            "target_pathway_prior": target_pathway_prior.astype(np.float32),
            "transfer_prior": transfer_prior.astype(np.float32) if transfer_prior is not None else None,
            "full_pathway_embedding": full_transfer.get("pathway_embedding").astype(np.float32) if full_transfer else None,
            "full_site_embedding": full_transfer.get("site_embedding").astype(np.float32) if full_transfer else None,
            "full_site_mask": full_transfer.get("site_mask").astype(np.float32) if full_transfer else None,
            "scp682_main_transfer_prior": transfer_prior.astype(np.float32) if transfer_source == "scp682_main" and transfer_prior is not None else None,
            "scp682_main_full_pathway_embedding": full_transfer.get("pathway_embedding").astype(np.float32) if transfer_source == "scp682_main" and full_transfer else None,
            "scp682_main_full_site_embedding": full_transfer.get("site_embedding").astype(np.float32) if transfer_source == "scp682_main" and full_transfer else None,
            "scp682_main_full_site_mask": full_transfer.get("site_mask").astype(np.float32) if transfer_source == "scp682_main" and full_transfer else None,
            "scp68222_transfer_prior": transfer_prior.astype(np.float32) if transfer_source == "scp68222" and transfer_prior is not None else None,
            "scp68222_full_pathway_embedding": full_transfer.get("pathway_embedding").astype(np.float32) if transfer_source == "scp68222" and full_transfer else None,
            "scp68222_full_site_embedding": full_transfer.get("site_embedding").astype(np.float32) if transfer_source == "scp68222" and full_transfer else None,
            "scp68222_full_site_mask": full_transfer.get("site_mask").astype(np.float32) if transfer_source == "scp68222" and full_transfer else None,
            "scnet_site_graph_edge_index": site_graph.get("edge_index").astype(np.int64) if site_graph else None,
            "scnet_site_graph_edge_weight": site_graph.get("edge_weight").astype(np.float32) if site_graph else None,
            "scnet_site_graph_summary": site_graph.get("summary") if site_graph else {},
        },
        out_dir / "models" / "scp682_sc11_final.pt",
    )
    return model, {"train_idx": train_idx, "fit_idx": fit_idx, "val_idx": val_idx}


def evaluate_drug_delta(model, features, present, contexts, target_rows, args, out_dir, device):
    rows = []
    summary_rows = []
    for context in contexts:
        pred_ctrl = predict(model, features, present, context["control_idx"], args, device)
        pred_drug = predict(model, features, present, context["drug_idx"], args, device)
        pred_delta = pred_drug.mean(axis=0) - pred_ctrl.mean(axis=0)
        real_delta = context["real_delta"]
        ok = np.isfinite(real_delta) & np.isfinite(pred_delta)
        for j, row in enumerate(target_rows):
            if ok[j]:
                rows.append(
                    {
                        "context": context["context"],
                        "target_id": row["target_id"],
                        "target_index": int(row["target_index"]),
                        "real_delta": float(real_delta[j]),
                        "pred_delta": float(pred_delta[j]),
                        "direction_match": bool(np.sign(real_delta[j]) == np.sign(pred_delta[j])) if real_delta[j] != 0 and pred_delta[j] != 0 else False,
                        "abs_ratio": float(abs(pred_delta[j]) / (abs(real_delta[j]) + 1e-12)),
                        "n_control": context["n_control"],
                        "n_drug": context["n_drug"],
                        "n_control_target": int(context["n_control_target"][j]),
                        "n_drug_target": int(context["n_drug_target"][j]),
                    }
                )
        for label, targets in [("focus8", FOCUS_TARGETS), ("all", [r["target_id"] for r in target_rows])]:
            idx = [i for i, r in enumerate(target_rows) if r["target_id"] in set(targets)]
            use = np.asarray([i for i in idx if ok[i]], dtype=np.int64)
            if len(use) < 2:
                continue
            a = real_delta[use].astype(np.float64)
            b = pred_delta[use].astype(np.float64)
            den = np.linalg.norm(a) * np.linalg.norm(b)
            cos = float(np.dot(a, b) / den) if den > 0 else float("nan")
            norm_ratio = float(np.linalg.norm(b) / (np.linalg.norm(a) + 1e-12))
            summary_rows.append(
                {
                    "context": context["context"],
                    "signature": label,
                    "n_targets": int(len(use)),
                    "cosine": cos,
                    "spearman": safe_spearman(a, b),
                    "sign_acc": float(np.mean(np.sign(a) == np.sign(b))),
                    "norm_ratio": norm_ratio,
                    "n_control": context["n_control"],
                    "n_drug": context["n_drug"],
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "tables" / "scp682_sc11_observed_ibrutinib_delta_targets.tsv", sep="\t", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "tables" / "scp682_sc11_observed_ibrutinib_delta_summary.tsv", sep="\t", index=False)


def export_attention(model, features, present, meta, train_idx, hold_idx, pathway_names, target_rows, args, out_dir, device):
    rng = np.random.default_rng(args.seed + 17)
    samples = []
    if len(train_idx):
        samples.append(("train_sample", sample_idx(train_idx, args.attention_cells, rng)))
    if len(hold_idx):
        samples.append(("external_sample", sample_idx(hold_idx, args.attention_cells, rng)))
    rows = []
    for label, idx in samples:
        _, attn = predict(model, features, present, idx, args, device, True)
        if attn.size == 0:
            continue
        ds = meta.iloc[idx]["dataset_id"].astype(str).to_numpy()
        groups = [("all", np.ones(len(idx), dtype=bool))]
        for name in sorted(pd.unique(ds)):
            groups.append((name, ds == name))
        for group_name, gm in groups:
            if not gm.sum():
                continue
            mean_attn = attn[gm].mean(axis=0)
            for t, row in enumerate(target_rows):
                for p, pathway in enumerate(pathway_names):
                    rows.append(
                        {
                            "model": MODEL_NAME,
                            "sample": label,
                            "dataset": group_name,
                            "target_id": row["target_id"],
                            "target_index": int(row["target_index"]),
                            "pathway": pathway,
                            "mean_attention": float(mean_attn[t, p]),
                        }
                    )
    pd.DataFrame(rows).to_csv(out_dir / "tables" / "scp682_sc11_site_pathway_attention.tsv", sep="\t", index=False)


def evaluate(model, features, present, y, mask, meta, train_info, contexts, pathway_names, target_rows, args, out_dir):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    rows = []
    rng = np.random.default_rng(args.seed)
    if isinstance(train_info, dict):
        train_idx = np.asarray(train_info.get("train_idx", []), dtype=np.int64)
        fit_idx = np.asarray(train_info.get("fit_idx", train_idx), dtype=np.int64)
        val_idx = np.asarray(train_info.get("val_idx", []), dtype=np.int64)
    else:
        train_idx = np.asarray(train_info, dtype=np.int64)
        fit_idx = train_idx
        val_idx = np.asarray([], dtype=np.int64)
    if len(val_idx):
        pred_val = predict(model, features, present, val_idx, args, device)
        rows.extend(score_prediction(y, mask, pred_val, meta, val_idx, target_rows, "internal_cv_reconstruction"))
    eval_train = sample_idx(fit_idx, args.max_eval_train_cells if not args.smoke else 4096, rng)
    pred_train = predict(model, features, present, eval_train, args, device)
    rows.extend(score_prediction(y, mask, pred_train, meta, eval_train, target_rows, "train_reconstruction"))
    hold_idx = np.flatnonzero(meta["dataset_id"].astype(str).isin(parse_list(args.holdout_datasets)).to_numpy() & mask.any(axis=1))
    if getattr(args, "exclude_train_from_holdout", False) and len(hold_idx):
        train_set = set(int(x) for x in np.asarray(train_idx, dtype=np.int64))
        hold_idx = np.asarray([int(x) for x in hold_idx if int(x) not in train_set], dtype=np.int64)
    if len(hold_idx):
        pred_hold = predict(model, features, present, hold_idx, args, device)
        rows.extend(score_prediction(y, mask, pred_hold, meta, hold_idx, target_rows, "external_reconstruction"))
    pd.DataFrame(rows).to_csv(out_dir / "tables" / "scp682_sc11_reconstruction_performance.tsv", sep="\t", index=False)
    if contexts:
        evaluate_drug_delta(model, features, present, contexts, target_rows, args, out_dir, device)
    if args.export_attention:
        export_attention(model, features, present, meta, train_idx, hold_idx, pathway_names, target_rows, args, out_dir, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"./data_root")
    ap.add_argument("--pathway-manifest", default=r"02_results\single_cell\20260519_scp682_sc3_multidomain_features_v1\intermediate\pathway_gene_manifest.tsv")
    ap.add_argument("--model-input-dir", default=r"01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_gse300551_signal_seq_multidomain_v1")
    ap.add_argument("--output-dir", default=r"./results/SCP682_SC")
    ap.add_argument("--train-datasets", default="iccite_seq_tcell_2025,qurie_seq_bjab_2021")
    ap.add_argument("--train-control-only-datasets", default="__none__")
    ap.add_argument("--holdout-datasets", default="gse300551_iccite_plex_kinase_2025,phospho_seq_blair_2025_phospho_multi,vivo_seq_th17_2025,signal_seq_gse256403_hela_2024,signal_seq_gse256404_pdo_caf_2024")
    ap.add_argument("--exclude-train-from-holdout", action="store_true", default=True)
    ap.add_argument("--include-drug-delta-eval", action="store_true", default=False)
    ap.add_argument("--target-ids", default="include_in_loss")
    ap.add_argument("--target-transform", choices=("zscore", "percentile", "raw"), default="zscore")
    ap.add_argument("--context-times", default="6,180")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--batch-log-interval", type=int, default=0)
    ap.add_argument("--eval-batch-size", type=int, default=4096)
    ap.add_argument("--context-cells", type=int, default=0)
    ap.add_argument("--max-eval-train-cells", type=int, default=20000)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--pathway-layers", type=int, default=2)
    ap.add_argument("--attention-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--recon-weight", type=float, default=1.0)
    ap.add_argument("--prior-weight", type=float, default=0.03)
    ap.add_argument("--prior-neighbors", type=int, default=12)
    ap.add_argument("--prior-temperature", type=float, default=0.08)
    ap.add_argument("--prior-min-similarity", type=float, default=0.15)
    ap.add_argument("--prior-steps-per-epoch", type=int, default=24)
    ap.add_argument("--prior-batch-size", type=int, default=1024)
    ap.add_argument("--prior-datasets", default="")
    ap.add_argument("--prior-include-holdout-rna", action="store_true", default=False)
    ap.add_argument("--delta-weight", type=float, default=0.0)
    ap.add_argument("--delta-cosine-weight", type=float, default=0.0)
    ap.add_argument("--delta-focus-only", action="store_true")
    ap.add_argument("--scp682-main-transfer-dir", default="")
    ap.add_argument("--scp682-main-pathway-token-transfer-dir", default="")
    ap.add_argument("--scp68222-transfer-dir", default="")
    ap.add_argument("--transfer-alpha", type=float, default=0.0)
    ap.add_argument("--transfer-attention-weight", type=float, default=0.0)
    ap.add_argument("--full-transfer-scale", type=float, default=0.0)
    ap.add_argument("--teacher-distill-weight", type=float, default=0.0)
    ap.add_argument("--site-graph-prior-root", default=r"01_data\pathway_prior\intermediate")
    ap.add_argument("--site-graph-weight", type=float, default=0.03)
    ap.add_argument("--site-graph-scale", type=float, default=0.25)
    ap.add_argument("--site-graph-topk", type=int, default=12)
    ap.add_argument("--site-graph-min-weight", type=float, default=0.0)
    ap.add_argument("--site-graph-candidate-limit", type=int, default=96)
    ap.add_argument("--site-graph-max-aux-nodes", type=int, default=12000)
    ap.add_argument("--disable-pathway-attention", action="store_true", default=False)
    ap.add_argument("--warm-start-model", default="")
    ap.add_argument("--huber-beta", type=float, default=0.5)
    ap.add_argument("--loss-type", choices=("mse", "huber"), default="mse")
    ap.add_argument("--warmup-epochs", type=int, default=15)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--val-cells", type=int, default=20000)
    ap.add_argument("--cv-folds", type=int, default=0)
    ap.add_argument("--cv-fold", type=int, default=0)
    ap.add_argument("--max-target-weight", type=float, default=4.0)
    ap.add_argument("--balance-datasets", action="store_true", default=True)
    ap.add_argument("--no-balance-datasets", dest="balance_datasets", action="store_false")
    ap.add_argument("--balance-dataset-size", type=int, default=60000)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--max-train-cells", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260517)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--export-attention", action="store_true", default=True)
    ap.add_argument("--attention-cells", type=int, default=4096)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-cells", type=int, default=4096)
    args = ap.parse_args()

    root = Path(args.root)
    input_dir = root / args.model_input_dir
    pathway_manifest = root / args.pathway_manifest
    out_dir = ensure_dir(root / args.output_dir)
    for name in ("tables", "models", "logs", "reports"):
        ensure_dir(out_dir / name)

    features = np.load(input_dir / "embeddings.npy", mmap_mode="r")
    meta = pd.read_csv(input_dir / "cell_metadata.tsv", sep="\t", low_memory=False)
    manifest = pd.read_csv(pathway_manifest, sep="\t")
    pathway_names = list(dict.fromkeys(manifest["pathway"].astype(str).tolist()))
    present = np.ones((len(meta), len(pathway_names)), dtype=np.float32)

    target_table = pd.read_csv(input_dir / "phospho_target_table.tsv", sep="\t")
    target_indices, target_rows = choose_targets(target_table, parse_list(args.target_ids))
    y_all = np.load(input_dir / "targets.npy", mmap_mode="r")
    mask_all = np.load(input_dir / "target_mask.npy", mmap_mode="r")
    y_raw = np.asarray(y_all[:, target_indices], dtype=np.float32)
    obs_mask = np.asarray(mask_all[:, target_indices], dtype=bool) & np.isfinite(y_raw)
    train_idx_for_transform = build_train_idx(meta, obs_mask, args)
    transform_reference_idx = train_idx_for_transform
    if int(getattr(args, "cv_folds", 0) or 0) > 1:
        cv_rng = np.random.default_rng(args.seed)
        transform_reference_idx, _ = split_train_validation(train_idx_for_transform, meta, args, cv_rng)
    y, transform_stats = transform_targets(y_raw, obs_mask, transform_reference_idx, args.target_transform, target_rows)
    contexts = []
    if args.delta_weight > 0 or args.include_drug_delta_eval:
        contexts = build_drug_contexts(meta, y, obs_mask, args)
    train_meta = meta.iloc[train_idx_for_transform].copy()
    if "ibrutinib" in train_meta.columns:
        train_meta["_ibrutinib_bool"] = parse_bool(train_meta["ibrutinib"]).to_numpy()
    else:
        train_meta["_ibrutinib_bool"] = False
    train_manifest = (
        train_meta.groupby(["dataset_id", "_ibrutinib_bool"], dropna=False)
        .size()
        .reset_index(name="n_cells")
        .rename(columns={"_ibrutinib_bool": "ibrutinib"})
    )
    train_manifest.to_csv(out_dir / "tables" / "scp682_sc11_train_manifest.tsv", sep="\t", index=False)

    local_target_pathway_prior, target_pathway_rows = build_target_pathway_prior(target_rows, pathway_names, manifest)
    transfer_prior = np.zeros_like(local_target_pathway_prior, dtype=np.float32)
    transfer_rows = pd.DataFrame()
    transfer_align = pd.DataFrame()
    full_transfer = None
    transfer_source = "none"
    if args.scp682_main_transfer_dir:
        transfer_root = root / args.scp682_main_transfer_dir
        main_transfer = load_scp682_main_transfer(target_rows, pathway_names, manifest, transfer_root)
        if main_transfer:
            transfer_source = "scp682_main"
            transfer_prior = main_transfer.get("transfer_prior", transfer_prior)
            transfer_rows = main_transfer.get("transfer_rows", pd.DataFrame())
            transfer_align = main_transfer.get("pathway_alignment", pd.DataFrame())
            if args.full_transfer_scale > 0:
                full_transfer = main_transfer
            log(
                "scp682_main_transfer loaded "
                f"models={main_transfer.get('n_model_files', 0)} "
                f"bulk_dim={main_transfer.get('bulk_dim', 0)} "
                f"site_matches={int(np.sum(main_transfer.get('site_mask', np.zeros(0))))}/{len(target_rows)} "
                f"source={main_transfer.get('source_model', 'SCP682_MAIN')}"
            )
        else:
            log("scp682_main_transfer unavailable; falling back to local pathway prior")
    elif args.scp682_main_pathway_token_transfer_dir or args.scp68222_transfer_dir:
        transfer_source = "scp682_main" if args.scp682_main_pathway_token_transfer_dir else "scp68222"
        transfer_root = root / (args.scp682_main_pathway_token_transfer_dir or args.scp68222_transfer_dir)
        transfer_prior, transfer_rows, transfer_align = build_scp68222_transfer_prior(
            target_rows,
            pathway_names,
            manifest,
            transfer_root,
            args.transfer_alpha,
        )
        if args.full_transfer_scale > 0:
            full_transfer = load_scp68222_full_transfer(target_rows, pathway_names, manifest, transfer_root)
            if full_transfer:
                log(
                    "scp682_main_pathway_token_transfer loaded "
                    f"models={full_transfer.get('n_model_files', 0)} "
                    f"bulk_dim={full_transfer.get('bulk_dim', 0)} "
                    f"site_matches={int(np.sum(full_transfer.get('site_mask', np.zeros(0))))}/{len(target_rows)}"
                )
            else:
                log("scp682_main_pathway_token_transfer unavailable; falling back to attention prior only")
    target_pathway_prior = mix_pathway_priors(local_target_pathway_prior, transfer_prior, args.transfer_alpha)
    site_graph = build_expanded_scnet_site_graph_prior(
        target_rows,
        root / args.site_graph_prior_root,
        candidate_limit=args.site_graph_candidate_limit,
        max_aux_nodes=args.site_graph_max_aux_nodes,
    )
    log(
        "expanded_site_graph targets={} graph_nodes={} aux_nodes={} edges={} seed_bulk_sites={} candidate_matches={}".format(
            len(target_rows),
            int(site_graph.get("summary", {}).get("n_graph_nodes", 0)),
            int(site_graph.get("summary", {}).get("n_aux_nodes", 0)),
            int(site_graph.get("summary", {}).get("n_edges", 0)),
            int(site_graph.get("summary", {}).get("seed_bulk_sites", 0)),
            int(site_graph.get("summary", {}).get("candidate_matches", 0)),
        )
    )
    transform_stats.to_csv(out_dir / "tables" / "scp682_sc11_target_transform.tsv", sep="\t", index=False)
    pd.DataFrame(target_rows).to_csv(out_dir / "tables" / "scp682_sc11_target_table.tsv", sep="\t", index=False)
    target_pathway_rows.to_csv(out_dir / "tables" / "scp682_sc11_target_pathway_prior.tsv", sep="\t", index=False)
    transfer_prefix = "scp682_main" if transfer_source in {"scp682_main", "scp68222"} else "transfer"
    if transfer_prefix == "scp682_main" and not transfer_rows.empty:
        transfer_rows = transfer_rows.rename(columns={"scp68222_transfer_weight": "scp682_main_transfer_weight"})
    if transfer_prefix == "scp682_main" and not transfer_align.empty:
        transfer_align = transfer_align.rename(columns={"scp68222_pathway": "scp682_main_pathway"})
    transfer_rows.to_csv(out_dir / "tables" / f"scp682_sc11_{transfer_prefix}_transfer_prior.tsv", sep="\t", index=False)
    transfer_align.to_csv(out_dir / "tables" / f"scp682_sc11_{transfer_prefix}_pathway_alignment.tsv", sep="\t", index=False)
    if full_transfer:
        full_pathway_alignment = full_transfer["pathway_alignment"]
        if transfer_prefix == "scp682_main" and not full_pathway_alignment.empty:
            full_pathway_alignment = full_pathway_alignment.rename(columns={"scp68222_pathway": "scp682_main_pathway"})
        full_pathway_alignment.to_csv(out_dir / "tables" / f"scp682_sc11_{transfer_prefix}_full_pathway_alignment.tsv", sep="\t", index=False)
        full_transfer["site_matches"].to_csv(out_dir / "tables" / f"scp682_sc11_{transfer_prefix}_full_site_matches.tsv", sep="\t", index=False)
        pd.DataFrame(full_transfer["pathway_embedding"]).to_csv(out_dir / "tables" / f"scp682_sc11_{transfer_prefix}_full_pathway_embedding.tsv", sep="\t", index=False)
        pd.DataFrame(full_transfer["site_embedding"]).to_csv(out_dir / "tables" / f"scp682_sc11_{transfer_prefix}_full_site_embedding.tsv", sep="\t", index=False)
    site_graph.get("edges", pd.DataFrame()).to_csv(out_dir / "tables" / "scp682_sc11_scnet_site_graph_edges.tsv", sep="\t", index=False)
    site_graph.get("candidates", pd.DataFrame()).to_csv(out_dir / "tables" / "scp682_sc11_scnet_site_graph_candidates.tsv", sep="\t", index=False)
    site_graph.get("node_table", pd.DataFrame()).to_csv(out_dir / "tables" / "scp682_sc11_scnet_site_graph_nodes.tsv", sep="\t", index=False)
    if "edge_index" in site_graph:
        pd.DataFrame(
            {
                "source_node": site_graph["edge_index"][0],
                "target_node": site_graph["edge_index"][1],
                "edge_weight": site_graph["edge_weight"],
            }
        ).to_csv(out_dir / "tables" / "scp682_sc11_scnet_site_graph_edge_index.tsv", sep="\t", index=False)
    with (out_dir / "reports" / "scp682_sc11_scnet_site_graph_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(site_graph.get("summary", {}), fh, indent=2, ensure_ascii=False)
    pd.DataFrame(
        [
            {
                "context": c["context"],
                "time_min": c["time_min"],
                "n_control": c["n_control"],
                "n_drug": c["n_drug"],
            }
            for c in contexts
        ]
    ).to_csv(out_dir / "tables" / "scp682_sc11_drug_contexts.tsv", sep="\t", index=False)
    log(f"features={features.shape} targets={len(target_rows)} contexts={len(contexts)} output={out_dir}")

    model, train_info = train_model(features, present, y, obs_mask, meta, contexts, pathway_names, target_rows, target_pathway_prior, transfer_prior, full_transfer, site_graph, args, out_dir)
    evaluate(model, features, present, y, obs_mask, meta, train_info, contexts, pathway_names, target_rows, args, out_dir)
    with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": MODEL_NAME,
                "args": vars(args),
                "base_model": "scFoundation",
                "scfoundation_calling_rule": {
                    "gene_order": "OS_scRNA_gene_index.19264.tsv",
                    "embedding_type": "cell",
                    "pool_type": "all",
                    "input_gene_cap": "cap12000",
                    "precomputed_embedding_dim": int(features.shape[1])
                },
                "pathway_manifest": str(pathway_manifest),
                "model_input_dir": str(input_dir),
                "pathway_names": pathway_names,
                "target_rows": target_rows,
                "n_train_cells": int(len(train_info.get("train_idx", []))),
                "n_fit_cells": int(len(train_info.get("fit_idx", []))),
                "n_internal_cv_cells": int(len(train_info.get("val_idx", []))),
                "drug_delta_supervision_used": bool(args.delta_weight > 0),
                "transfer_source": transfer_source,
                "scp682_main_transfer_used": bool((args.scp682_main_transfer_dir or args.scp682_main_pathway_token_transfer_dir) and args.transfer_alpha > 0),
                "scp682_main_full_transfer_used": bool(full_transfer and transfer_source == "scp682_main"),
                "scp682_main_full_site_matches": int(np.sum(full_transfer.get("site_mask", np.zeros(0)))) if full_transfer and transfer_source == "scp682_main" else 0,
                "scp68222_transfer_used": bool(args.scp68222_transfer_dir and args.transfer_alpha > 0 and transfer_source == "scp68222"),
                "scp68222_full_transfer_used": bool(full_transfer and transfer_source == "scp68222"),
                "scp68222_full_site_matches": int(np.sum(full_transfer.get("site_mask", np.zeros(0)))) if full_transfer and transfer_source == "scp68222" else 0,
                "scnet_site_graph": site_graph.get("summary", {}),
            },
            fh,
            indent=2,
        )
    (out_dir / "logs" / "done.txt").write_text(f"done {now()}\n", encoding="utf-8")
    log(f"done: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        out = Path(r"./data_root") / r"./results/SCP682_SC"
        argv = list(sys.argv)
        root = Path(r"./data_root")
        if "--root" in argv:
            pos = argv.index("--root")
            if pos + 1 < len(argv):
                root = Path(argv[pos + 1])
        if "--output-dir" in argv:
            pos = argv.index("--output-dir")
            if pos + 1 < len(argv):
                out = root / argv[pos + 1]
        ensure_dir(out / "logs")
        (out / "logs" / "fatal.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise






