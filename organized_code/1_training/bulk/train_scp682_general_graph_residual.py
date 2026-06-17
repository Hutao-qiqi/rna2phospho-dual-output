"""
模型: SCP682
作用: 从原始实验脚本提取的最小可复现代码片段，文件名 train_scp682_general_graph_residual.py
输入: ./data_root 下的训练数据、图先验和冻结基线预测
输出: ./paper_materials_SCP682 或结果目录中的模型、表格、报告
依赖: Python、pandas、numpy、torch、torch_geometric
原始路径: remote_scripts/train_scp682_general_graph_residual.py
原始版本: 20260523 结果目录对应脚本
"""

"""Train the SCP682 graph-constrained phosphosite residual model.

The model is described as one integrated predictor:

    phosphosite_hat = general_baseline_hat
                      + graph_residual_theta(general_baseline_hat, RNA,
                                             phosphosite_graph, sample_graph)

The general baseline is the frozen SCP682 baseline head. The graph model treats
it as one baseline function and does not decompose it into historical candidate
branches during training. The trainable part is the exact ScNET
graph-constrained residual operator.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import sequential, SAGEConv, GCNConv, TransformerConv, InnerProductDecoder
from torch_geometric.utils import negative_sampling, softmax


EPS = 1e-15


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_numeric(df):
    df = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def row_normalize(mat):
    mat = mat.astype(np.float32, copy=True)
    denom = np.maximum(1.0, np.abs(mat).sum(axis=1, keepdims=True))
    return mat / denom


def masked_huber(pred, true, mask, site_weight):
    loss = F.smooth_l1_loss(pred, true, reduction="none")
    w = mask.float() * site_weight.unsqueeze(0)
    return (loss * w).sum() / w.sum().clamp_min(1.0)


def masked_l2(value, mask, site_weight):
    w = mask.float() * site_weight.unsqueeze(0)
    return (value.square() * w).sum() / w.sum().clamp_min(1.0)


def cosine_loss(pred, true, mask, eps=1e-8):
    m = mask.float()
    pred = pred * m
    true = true * m
    num = (pred * true).sum(dim=1)
    den = torch.sqrt(pred.square().sum(dim=1) + eps) * torch.sqrt(true.square().sum(dim=1) + eps)
    return 1.0 - (num / den.clamp_min(eps)).mean()


def cosine_np(a, b, mask, eps=1e-8):
    ok = mask.astype(bool) & np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return np.nan
    den = np.linalg.norm(a[ok]) * np.linalg.norm(b[ok])
    return float(np.dot(a[ok], b[ok]) / max(den, eps))


def spearman_cols(y, pred, mask, targets):
    rows = []
    for j, t in enumerate(targets):
        ok = mask[:, j] & np.isfinite(y[:, j]) & np.isfinite(pred[:, j])
        if ok.sum() < 3:
            sp = np.nan
        else:
            sp = pd.Series(y[ok, j]).corr(pd.Series(pred[ok, j]), method="spearman")
        rows.append({"target": t, "n": int(ok.sum()), "spearman": sp})
    return pd.DataFrame(rows)


def summarize(per, model):
    v = pd.to_numeric(per["spearman"], errors="coerce").dropna()
    return {
        "model": model,
        "n_targets": int(v.shape[0]),
        "median_spearman": float(v.median()),
        "mean_spearman": float(v.mean()),
        "ge_0_3": int((v >= 0.3).sum()),
        "ge_0_5": int((v >= 0.5).sum()),
    }


def build_sample_knn_edge_index(baseline, valid, k=10):
    x = np.where(valid, baseline, 0.0).astype(np.float32)
    x = x - x.mean(axis=1, keepdims=True)
    norm = np.sqrt((x * x).sum(axis=1, keepdims=True)).clip(min=1e-6)
    z = x / norm
    sim = z @ z.T
    np.fill_diagonal(sim, 1.0)
    edges = []
    for i in range(sim.shape[0]):
        keep = np.argsort(-sim[i])[: max(1, min(k + 1, sim.shape[0]))]
        for j in keep:
            edges.append((i, int(j)))
            edges.append((int(j), i))
    return np.asarray(sorted(set(edges)), dtype=np.int64).T


def l2_sample_block(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(axis=1, keepdims=True)
    norm = np.sqrt((x * x).sum(axis=1, keepdims=True)).clip(min=1e-6)
    return x / norm


def build_sample_knn_edge_index_from_features(features, k=10):
    z = l2_sample_block(features)
    sim = z @ z.T
    np.fill_diagonal(sim, 1.0)
    edges = []
    for i in range(sim.shape[0]):
        keep = np.argsort(-sim[i])[: max(1, min(k + 1, sim.shape[0]))]
        for j in keep:
            edges.append((i, int(j)))
            edges.append((int(j), i))
    return np.asarray(sorted(set(edges)), dtype=np.int64).T


def align_by_sample_key(df, target_index):
    direct = df.reindex(target_index)
    if direct.notna().any(axis=None):
        return direct
    tmp = df.copy()
    tmp["_sample_key"] = [str(x).split("::", 1)[1] if "::" in str(x) else str(x) for x in tmp.index]
    tmp = tmp.drop_duplicates("_sample_key").set_index("_sample_key")
    keys = [str(x).split("::", 1)[1] if "::" in str(x) else str(x) for x in target_index]
    out = tmp.reindex(keys)
    out.index = target_index
    return out


def make_rna_context(train_rna, train_samples, external_rna=None, external_samples=None, n_genes=2048):
    train_rna = clean_numeric(train_rna)
    train_aligned = align_by_sample_key(train_rna, train_samples)
    if external_rna is not None and external_samples is not None:
        external_rna = clean_numeric(external_rna)
        external_aligned = align_by_sample_key(external_rna, external_samples)
        common = [g for g in train_aligned.columns if g in external_aligned.columns]
    else:
        external_aligned = None
        common = list(train_aligned.columns)
    train_block = train_aligned[common].apply(pd.to_numeric, errors="coerce")
    var = train_block.var(axis=0, skipna=True).sort_values(ascending=False)
    genes = list(var.index[: min(n_genes, len(var))])
    mean = train_block[genes].mean(axis=0, skipna=True)
    std = train_block[genes].std(axis=0, skipna=True).replace(0, np.nan).fillna(1.0)
    train_ctx = ((train_block[genes] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-6, 6)
    if external_aligned is not None:
        ext_block = external_aligned[genes].apply(pd.to_numeric, errors="coerce")
        ext_ctx = ((ext_block - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-6, 6)
        ctx = pd.concat([train_ctx, ext_ctx], axis=0)
    else:
        ctx = train_ctx
    return ctx.astype(np.float32), genes


def load_group_labels(sample_manifest_path, samples, group_column):
    meta = pd.read_csv(sample_manifest_path, sep="\t")
    sample_col = "sample_id" if "sample_id" in meta.columns else "index"
    if sample_col not in meta.columns:
        raise ValueError(f"sample manifest lacks sample_id/index column: {sample_manifest_path}")
    if group_column not in meta.columns:
        raise ValueError(f"sample manifest lacks requested group column: {group_column}")
    mapping = dict(zip(meta[sample_col].astype(str), meta[group_column].astype(str)))
    labels = []
    for s in samples:
        key = str(s)
        labels.append(mapping.get(key, "UNKNOWN"))
    return np.asarray(labels, dtype=object)


def make_group_indices(labels):
    groups = {}
    for i, g in enumerate(labels):
        groups.setdefault(str(g), []).append(i)
    return {g: np.asarray(v, dtype=np.int64) for g, v in sorted(groups.items()) if len(v) >= 3}


def anchor_col_embed_torch(col_embed, sample_feature, query_idx, anchor_idx, k=25, temperature=0.08):
    q_feat = sample_feature.index_select(0, query_idx)
    a_feat = sample_feature.index_select(0, anchor_idx)
    sim = torch.matmul(q_feat, a_feat.T) / max(float(temperature), 1e-6)
    kk = min(int(k), sim.shape[1])
    val, loc = torch.topk(sim, k=kk, dim=1)
    weight = torch.softmax(val, dim=1)
    anchor_embed = col_embed.index_select(0, anchor_idx)
    picked = anchor_embed.index_select(0, loc.reshape(-1)).reshape(loc.shape[0], loc.shape[1], -1)
    return (picked * weight.unsqueeze(-1)).sum(dim=1)


def add_edges_from_pairs(edge_set, site_to_i, a_values, b_values, max_edges=None):
    n = 0
    for a, b in zip(a_values, b_values):
        ia = site_to_i.get(str(a))
        ib = site_to_i.get(str(b))
        if ia is None or ib is None or ia == ib:
            continue
        edge_set.add((ia, ib))
        edge_set.add((ib, ia))
        n += 2
        if max_edges and n >= max_edges:
            break
    return n


def build_site_graph(targets, prior_root, max_kinase_neighbors=16):
    site_to_i = {t: i for i, t in enumerate(targets)}
    edge_set = {(i, i) for i in range(len(targets))}
    source_counts = {}

    cophee_map_path = prior_root / "processed/copheemap_v1/copheemap_site_id_to_model_gene_site.tsv"
    cophee_id_to_gene_site = {}
    if cophee_map_path.exists():
        m = pd.read_csv(cophee_map_path, sep="\t")
        cophee_id_to_gene_site = dict(zip(m["cophee_site_id"].astype(str), m["gene_site_id"].astype(str)))

    raw_copheemap = prior_root / "raw/copheemap_20260519_files/Table_S2_CoPheeMap.tsv.zip"
    if not raw_copheemap.exists():
        raw_copheemap = prior_root / "raw/copheemap/CoPheeMap/Supplementary_table/Table_S2_CoPheeMap.tsv.zip"
    if raw_copheemap.exists() and cophee_id_to_gene_site:
        df = pd.read_csv(raw_copheemap, sep="\t")
        a = df["site1"].astype(str).map(cophee_id_to_gene_site)
        b = df["site2"].astype(str).map(cophee_id_to_gene_site)
        pair = pd.DataFrame({"a": a, "b": b}).dropna()
        source_counts["original_copheemap_site_site_edges_added"] = add_edges_from_pairs(edge_set, site_to_i, pair["a"], pair["b"])

    raw_copheeksa = prior_root / "raw/copheemap_20260519_files/K_S_CoPhee_llr55.csv"
    if not raw_copheeksa.exists():
        raw_copheeksa = prior_root / "raw/copheemap/CoPheeMap/CoPheeKSA/positive_KSA.csv"
    if raw_copheeksa.exists() and cophee_id_to_gene_site:
        df = pd.read_csv(raw_copheeksa)
        kinase_col = "kinase" if "kinase" in df.columns else "kinases"
        df["gene_site"] = df["sites"].astype(str).map(cophee_id_to_gene_site)
        added = 0
        for _, sub in df[[kinase_col, "gene_site"]].dropna().drop_duplicates().groupby(kinase_col):
            ids = [site_to_i[s] for s in sub["gene_site"].astype(str) if s in site_to_i]
            ids = ids[:max_kinase_neighbors]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    edge_set.add((ids[i], ids[j]))
                    edge_set.add((ids[j], ids[i]))
                    added += 2
        source_counts["original_copheeksa_cokinase_edges_added"] = added

    kstar_long = prior_root / "intermediate/kstar_20260516/kstar_default_network_edges_long.tsv"
    if kstar_long.exists():
        usecols = ["kinase", "substrate_gene", "site"]
        df = pd.read_csv(kstar_long, sep="\t", usecols=usecols)
        df["gene_site"] = df["substrate_gene"].fillna("").astype(str) + "|" + df["site"].fillna("").astype(str)
        added = 0
        for _, sub in df[["kinase", "gene_site"]].dropna().drop_duplicates().groupby("kinase"):
            ids = [site_to_i[s] for s in sub["gene_site"].astype(str) if s in site_to_i]
            ids = ids[:max_kinase_neighbors]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    edge_set.add((ids[i], ids[j]))
                    edge_set.add((ids[j], ids[i]))
                    added += 2
        source_counts["original_kstar_cokinase_edges_added"] = added

    cophee = prior_root / "intermediate/copheemap_20260519_scp682_ppko_v6/tables/copheemap_model_site_site_edges.tsv"
    if cophee.exists():
        df = pd.read_csv(cophee, sep="\t")
        a_col = "target_id_a" if "target_id_a" in df.columns else ("site_a" if "site_a" in df.columns else df.columns[0])
        b_col = "target_id_b" if "target_id_b" in df.columns else ("site_b" if "site_b" in df.columns else df.columns[1])
        source_counts["copheemap_site_site_edges_added"] = add_edges_from_pairs(edge_set, site_to_i, df[a_col], df[b_col])

    kstar = prior_root / "intermediate/kstar_20260519/scp682_ppko_v5_kstar_kinase_site_edges.tsv"
    if kstar.exists():
        df = pd.read_csv(kstar, sep="\t")
        site_col = "target_id" if "target_id" in df.columns else ("model_site_id" if "model_site_id" in df.columns else None)
        kinase_col = "kinase" if "kinase" in df.columns else None
        if site_col and kinase_col:
            added = 0
            for _, sub in df[[kinase_col, site_col]].dropna().drop_duplicates().groupby(kinase_col):
                ids = [site_to_i[s] for s in sub[site_col].astype(str) if s in site_to_i]
                if len(ids) < 2:
                    continue
                ids = ids[:max_kinase_neighbors]
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        edge_set.add((ids[i], ids[j]))
                        edge_set.add((ids[j], ids[i]))
                        added += 2
            source_counts["kstar_cokinase_edges_added"] = added

    copheeksa = prior_root / "intermediate/copheemap_20260519_scp682_ppko_v6/tables/copheeksa_model_kinase_site_edges.tsv"
    if copheeksa.exists():
        df = pd.read_csv(copheeksa, sep="\t")
        site_col = "target_id" if "target_id" in df.columns else ("model_site_id" if "model_site_id" in df.columns else None)
        kinase_col = "kinase" if "kinase" in df.columns else None
        if site_col and kinase_col:
            added = 0
            for _, sub in df[[kinase_col, site_col]].dropna().drop_duplicates().groupby(kinase_col):
                ids = [site_to_i[s] for s in sub[site_col].astype(str) if s in site_to_i]
                ids = ids[:max_kinase_neighbors]
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        edge_set.add((ids[i], ids[j]))
                        edge_set.add((ids[j], ids[i]))
                        added += 2
            source_counts["copheeksa_cokinase_edges_added"] = added

    edge_index = np.asarray(sorted(edge_set), dtype=np.int64).T
    degree = np.bincount(edge_index[0], minlength=len(targets)).astype(np.float32)
    prior_strength = np.clip(np.log1p(degree) / np.log1p(max(float(degree.max()), 1.0)), 0, 1).astype(np.float32)
    return edge_index, prior_strength, source_counts


class FeatureDecoder(nn.Module):
    def __init__(self, feature_dim, embd_dim, inter_dim, drop_p=0.0):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embd_dim, inter_dim),
            nn.Dropout(drop_p),
            nn.ReLU(),
            nn.Linear(inter_dim, inter_dim),
            nn.Dropout(drop_p),
            nn.ReLU(),
            nn.Linear(inter_dim, feature_dim),
            nn.Dropout(drop_p),
        )

    def forward(self, z):
        return self.decoder(z)


class MutualEncoder(nn.Module):
    def __init__(self, col_dim, row_dim, num_layers=3, drop_p=0.25):
        super().__init__()
        self.rows_layers = nn.ModuleList([
            sequential.Sequential("x,edge_index", [
                (SAGEConv(row_dim, row_dim), "x, edge_index -> x1"),
                (nn.Dropout(drop_p, inplace=False), "x1 -> x2"),
                nn.LeakyReLU(inplace=True),
            ]) for _ in range(num_layers)
        ])
        self.cols_layers = nn.ModuleList([
            sequential.Sequential("x,edge_index", [
                (SAGEConv(col_dim, col_dim), "x, edge_index -> x1"),
                nn.LeakyReLU(inplace=True),
                (nn.Dropout(drop_p, inplace=False), "x1 -> x2"),
            ]) for _ in range(num_layers)
        ])

    def forward(self, x, col_edge_index, row_edge_index):
        embedded = x.clone()
        for i in range(len(self.rows_layers)):
            embedded = self.cols_layers[i](embedded.T, col_edge_index).T
            embedded = self.rows_layers[i](embedded, row_edge_index)
        return embedded


class TransformerConvReducrLayer(TransformerConv):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0, add_self_loops=True, scale_param=2, **kwargs):
        super().__init__(in_channels, out_channels, heads, dropout, add_self_loops, **kwargs)
        self.treshold_alpha = None
        self.scale_param = scale_param

    def message(self, query_i, key_j, value_j, edge_attr, index, ptr, size_i):
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j += edge_attr
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        if self.scale_param is not None:
            alpha = alpha - alpha.mean()
            alpha = alpha / ((1 / self.scale_param) * alpha.std().clamp_min(1e-6))
            alpha = torch.sigmoid(alpha)
        else:
            alpha = softmax(alpha, index, ptr, size_i)
        self.treshold_alpha = alpha
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = value_j
        if edge_attr is not None:
            out += edge_attr
        out *= alpha.view(-1, self.heads, 1)
        return out


class DimEncoder(nn.Module):
    def __init__(self, feature_dim, inter_dim, embd_dim, reducer=False, drop_p=0.2, scale_param=3):
        super().__init__()
        self.reducer = reducer
        self.encoder = sequential.Sequential("x, edge_index", [
            (GCNConv(feature_dim, inter_dim), "x, edge_index -> x1"),
            nn.LeakyReLU(inplace=True),
            (nn.Dropout(drop_p, inplace=False), "x1 -> x2"),
        ])
        if reducer:
            self.atten_layer = TransformerConvReducrLayer(inter_dim, embd_dim, dropout=drop_p, add_self_loops=False, heads=1, scale_param=scale_param)
        else:
            self.atten_layer = TransformerConv(inter_dim, embd_dim, dropout=drop_p)
        self.atten_map = None
        self.atten_weights = None

    def reduce_network(self, threshold=0.2, min_connect=5):
        graph = self.atten_weights.cpu().detach().numpy()
        threshold_bound = np.percentile(graph, 10)
        threshold = min(threshold, threshold_bound)
        df = pd.DataFrame({
            "v1": self.atten_map[0].cpu().detach().numpy(),
            "v2": self.atten_map[1].cpu().detach().numpy(),
            "atten": graph.squeeze(),
        })
        saved_edges = df.groupby("v1")["atten"].nlargest(min_connect).index.values
        saved_edges = [v2 for _, v2 in saved_edges]
        df.iloc[saved_edges, 2] = threshold + EPS
        idx = list(df.loc[df.atten >= threshold].index)
        atten_map = self.atten_map[:, idx]
        self.atten_map = None
        self.atten_weights = None
        return atten_map, df

    def forward(self, x, edge_index, infrance=False):
        embedded = self.encoder(x.clone(), edge_index)
        embedded, atten_map = self.atten_layer(embedded, edge_index, return_attention_weights=True)
        if self.reducer and not infrance:
            if self.atten_map is None:
                self.atten_map = atten_map[0].detach()
                self.atten_weights = atten_map[1].detach()
            else:
                self.atten_map = torch.concat([self.atten_map.T, atten_map[0].detach().T]).T
                self.atten_weights = torch.concat([self.atten_weights, atten_map[1].detach()])
        return embedded


class ExactScNETGraph(nn.Module):
    def __init__(self, col_dim, row_dim, inter_dim=192, embd_dim=64, num_layers=2, drop_p=0.2):
        super().__init__()
        self.encoder = MutualEncoder(col_dim, row_dim, num_layers=num_layers, drop_p=drop_p)
        self.rows_encoder = DimEncoder(row_dim, inter_dim, embd_dim, drop_p=drop_p, scale_param=None, reducer=False)
        self.cols_encoder = DimEncoder(col_dim, inter_dim, embd_dim, drop_p=drop_p, reducer=True)
        self.feature_decoder = FeatureDecoder(col_dim, embd_dim, inter_dim, drop_p=0)
        self.ipd = InnerProductDecoder()

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None, sig=False):
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        if not sig:
            max_edges = 250000
            if pos_edge_index.shape[1] > max_edges:
                keep = torch.randperm(pos_edge_index.shape[1], device=pos_edge_index.device)[:max_edges]
                pos_edge_index = pos_edge_index[:, keep]
            if neg_edge_index.shape[1] > max_edges:
                keep = torch.randperm(neg_edge_index.shape[1], device=neg_edge_index.device)[:max_edges]
                neg_edge_index = neg_edge_index[:, keep]
            z_norm = F.normalize(z, p=2, dim=1)
            pos_score = (z_norm[pos_edge_index[0]] * z_norm[pos_edge_index[1]]).sum(dim=1)
            neg_score = (z_norm[neg_edge_index[0]] * z_norm[neg_edge_index[1]]).sum(dim=1)
            pos = torch.sigmoid(pos_score)
            neg = torch.sigmoid(neg_score)
            return -torch.log(pos + EPS).mean() - torch.log(1 - neg + EPS).mean()
        return -torch.log(self.ipd(z, pos_edge_index, sigmoid=sig) + EPS).mean() - torch.log(1 - self.ipd(z, neg_edge_index, sigmoid=sig) + EPS).mean()

    def forward(self, x, col_edge_index, row_edge_index, collect_attention=True):
        embedded = self.encoder(x, col_edge_index, row_edge_index)
        row_embed = self.rows_encoder(embedded, row_edge_index)
        col_embed = self.cols_encoder(embedded.T, col_edge_index, infrance=not collect_attention)
        out_features = self.feature_decoder(col_embed)
        return row_embed, col_embed, out_features


class SCP682GeneralGraphResidual(nn.Module):
    def __init__(
        self,
        n_sites,
        n_samples,
        hidden=160,
        latent=64,
        inter_dim=192,
        embd_dim=64,
        num_layers=2,
        sample_context_dim=0,
    ):
        super().__init__()
        self.n_sites = n_sites
        self.sample_context_dim = int(sample_context_dim)
        self.graph_core = ExactScNETGraph(n_sites, n_samples, inter_dim=inter_dim, embd_dim=embd_dim, num_layers=num_layers)
        self.site_prior_proj = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.baseline_proj = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.site_proj = nn.Sequential(nn.Linear(embd_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.sample_proj = nn.Sequential(nn.Linear(embd_dim, latent), nn.GELU(), nn.LayerNorm(latent))
        self.sample_context_proj = (
            nn.Sequential(nn.Linear(self.sample_context_dim, latent), nn.GELU(), nn.LayerNorm(latent))
            if self.sample_context_dim > 0 else None
        )
        self.prior_attention = nn.Sequential(nn.LayerNorm(hidden + hidden + latent + 1), nn.Linear(hidden + hidden + latent + 1, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.graph_decoder = nn.Sequential(nn.LayerNorm(hidden + hidden + latent + 1), nn.Linear(hidden + hidden + latent + 1, hidden), nn.GELU(), nn.Dropout(0.08), nn.Linear(hidden, 1))
        self.residual = nn.Sequential(nn.LayerNorm(hidden + hidden + latent), nn.Linear(hidden + hidden + latent, hidden), nn.GELU(), nn.Dropout(0.10), nn.Linear(hidden, 1))
        self.graph_scale = nn.Parameter(torch.tensor(-0.2))
        self.residual_scale = nn.Parameter(torch.tensor(-1.4))

    def decode(self, row_embed, col_embed, baseline, mask, site_prior, sample_idx=None, sample_context=None):
        if sample_idx is not None:
            col_embed = col_embed.index_select(0, sample_idx)
        base_h = self.baseline_proj(torch.stack([baseline, mask.float()], dim=-1))
        site_h = self.site_proj(row_embed).unsqueeze(0).expand(baseline.shape[0], -1, -1)
        sample_z0 = self.sample_proj(col_embed)
        if self.sample_context_proj is not None and sample_context is not None:
            if sample_idx is not None:
                sample_context = sample_context.index_select(0, sample_idx)
            sample_z0 = sample_z0 + self.sample_context_proj(sample_context)
        sample_z = sample_z0.unsqueeze(1).expand(-1, baseline.shape[1], -1)
        prior = site_prior.unsqueeze(0).expand(baseline.shape[0], -1)
        prior_h = self.site_prior_proj(prior.unsqueeze(-1))
        graph_input = torch.cat([base_h + prior_h, site_h, sample_z, prior.unsqueeze(-1)], dim=-1)
        attention = torch.sigmoid(self.prior_attention(graph_input)).squeeze(-1)
        graph_delta = attention * self.graph_decoder(graph_input).squeeze(-1) * torch.sigmoid(self.graph_scale)
        residual_delta = self.residual(torch.cat([base_h, site_h, sample_z], dim=-1)).squeeze(-1) * torch.sigmoid(self.residual_scale)
        delta = graph_delta + residual_delta
        pred = baseline + delta
        return pred, delta, graph_delta, residual_delta, attention

    def forward(self, feature_x, col_edge_index, row_edge_index, baseline, mask, site_prior, collect_attention=True, sample_idx=None, sample_context=None):
        row_embed, col_embed, out_features = self.graph_core(feature_x, col_edge_index, row_edge_index, collect_attention=collect_attention)
        pred, delta, graph_delta, residual_delta, attention = self.decode(
            row_embed, col_embed, baseline, mask, site_prior, sample_idx=sample_idx, sample_context=sample_context
        )
        return pred, delta, graph_delta, residual_delta, attention, row_embed, col_embed, out_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-dir", required=True)
    ap.add_argument("--prior-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=260)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--reduce-interval", type=int, default=30)
    ap.add_argument("--min-connect", type=int, default=5)
    ap.add_argument(
        "--general-baseline-path",
        default="./data_root/SCP682-main/inputs/general_baseline_internal_cptac_pdc_phosphosite.parquet",
    )
    ap.add_argument(
        "--rna-path",
        default="./data_root/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/rna_log2_tpm_paired.parquet",
    )
    ap.add_argument(
        "--sample-manifest-path",
        default="./data_root/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/sample_manifest.tsv",
    )
    ap.add_argument("--group-column", default="cancer_label")
    ap.add_argument("--rna-context-genes", type=int, default=2048)
    ap.add_argument("--knn-baseline-weight", type=float, default=0.70)
    ap.add_argument("--knn-rna-weight", type=float, default=1.00)
    ap.add_argument("--anchor-k", type=int, default=25)
    ap.add_argument("--anchor-temperature", type=float, default=0.08)
    ap.add_argument("--pseudo-weight", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=20260522)
    ap.add_argument("--ppi-weight", type=float, default=0.08)
    ap.add_argument("--baseline-weight", type=float, default=0.08)
    ap.add_argument("--attention-l1", type=float, default=0.004)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=160)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--inter-dim", type=int, default=192)
    ap.add_argument("--embd-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    for sub in ("models", "tables", "logs", "predictions", "reports"):
        ensure_dir(out / sub)

    pkg = Path(args.package_dir)
    train = pkg / "training_set"
    y_df = clean_numeric(pd.read_parquet(train / "observed_phosphosite.parquet"))
    general_baseline_df = clean_numeric(pd.read_parquet(args.general_baseline_path))
    rna_df = clean_numeric(pd.read_parquet(args.rna_path))
    samples = y_df.index.intersection(general_baseline_df.index)
    targets = [c for c in y_df.columns if c in general_baseline_df.columns]
    y_df = y_df.loc[samples, targets]
    baseline_df = general_baseline_df.loc[samples, targets]
    sample_context_df, rna_context_genes = make_rna_context(rna_df, samples, n_genes=args.rna_context_genes)
    group_labels = load_group_labels(args.sample_manifest_path, samples, args.group_column)
    group_indices = make_group_indices(group_labels)

    y = y_df.to_numpy(np.float32)
    baseline = baseline_df.to_numpy(np.float32)
    mask = np.isfinite(y) & np.isfinite(baseline)
    residual_true = y - baseline
    residual_true[~mask] = 0.0
    baseline = np.nan_to_num(baseline, nan=0.0).astype(np.float32)

    edge_index_np, site_prior_np, source_counts = build_site_graph(targets, Path(args.prior_root))
    sample_feature_np = np.concatenate([
        args.knn_baseline_weight * l2_sample_block(np.where(mask, baseline, 0.0)),
        args.knn_rna_weight * l2_sample_block(sample_context_df.to_numpy(np.float32)),
    ], axis=1).astype(np.float32)
    col_edge_np = build_sample_knn_edge_index_from_features(sample_feature_np, k=args.knn)
    feature_x_np = np.where(mask.T, baseline.T, 0.0).astype(np.float32)
    mu = feature_x_np.mean(axis=1, keepdims=True)
    sd = feature_x_np.std(axis=1, keepdims=True) + 1e-5
    feature_x_np = ((feature_x_np - mu) / sd).astype(np.float32)
    hv_count = min(3000, feature_x_np.shape[0])
    hv_idx_np = np.argsort(-feature_x_np.var(axis=1))[:hv_count].astype(np.int64)

    site_weight = np.ones(len(targets), dtype=np.float32)
    site_weight[site_prior_np > 0.1] = 1.15
    site_weight[site_prior_np > 0.5] = 1.30

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    tensors = {
        "feature_x": torch.as_tensor(feature_x_np, dtype=torch.float32, device=device),
        "baseline": torch.as_tensor(baseline, dtype=torch.float32, device=device),
        "residual_true": torch.as_tensor(residual_true, dtype=torch.float32, device=device),
        "y": torch.as_tensor(np.nan_to_num(y, nan=0.0), dtype=torch.float32, device=device),
        "mask": torch.as_tensor(mask, dtype=torch.bool, device=device),
        "site_prior": torch.as_tensor(site_prior_np, dtype=torch.float32, device=device),
        "site_weight": torch.as_tensor(site_weight, dtype=torch.float32, device=device),
        "sample_context": torch.as_tensor(sample_context_df.to_numpy(np.float32), dtype=torch.float32, device=device),
        "sample_feature": torch.as_tensor(sample_feature_np, dtype=torch.float32, device=device),
        "row_edge_index": torch.as_tensor(edge_index_np, dtype=torch.long, device=device),
        "col_edge_index": torch.as_tensor(col_edge_np, dtype=torch.long, device=device),
        "hv_idx": torch.as_tensor(hv_idx_np, dtype=torch.long, device=device),
    }
    model = SCP682GeneralGraphResidual(
        n_sites=len(targets),
        n_samples=len(samples),
        hidden=args.hidden,
        latent=args.latent,
        inter_dim=args.inter_dim,
        embd_dim=args.embd_dim,
        num_layers=args.num_layers,
        sample_context_dim=sample_context_df.shape[1],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, foreach=False)
    best = {"median_spearman": -1e9, "epoch": 0}
    logs = []
    start = time.time()

    meta = {
        "n_samples": len(samples),
        "n_sites": len(targets),
        "n_site_edges": int(edge_index_np.shape[1]),
        "n_sample_edges": int(col_edge_np.shape[1]),
        "source_counts": source_counts,
        "device": str(device),
        "baseline": "SCP682_general_baseline_predictor",
        "general_baseline_path": str(args.general_baseline_path),
        "formula": "phosphosite_hat = general_baseline_hat + graph_residual_theta(general_baseline_hat, RNA, phosphosite_graph, sample_graph)",
        "training_objective": "study/cancer pseudo-external anchoring inside CPTAC: held-out internal group uses RNA-conditioned attention over other groups",
        "group_column": str(args.group_column),
        "n_groups": int(len(group_indices)),
        "group_sizes": {str(k): int(len(v)) for k, v in group_indices.items()},
        "rna_path": str(args.rna_path),
        "rna_context_genes": int(len(rna_context_genes)),
        "knn_baseline_weight": float(args.knn_baseline_weight),
        "knn_rna_weight": float(args.knn_rna_weight),
        "anchor_k": int(args.anchor_k),
        "anchor_temperature": float(args.anchor_temperature),
        "pseudo_weight": float(args.pseudo_weight),
    }
    (out / "reports/input_graph_summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    group_names = list(group_indices.keys())
    all_sample_np = np.arange(len(samples), dtype=np.int64)

    def decode_pseudo_external(row_embed, col_embed, groups_to_run=None):
        model.eval()
        pred_np = np.zeros_like(baseline, dtype=np.float32)
        run_groups = group_names if groups_to_run is None else list(groups_to_run)
        with torch.no_grad():
            for g in run_groups:
                q_np = group_indices[g]
                if q_np.shape[0] < 3 or q_np.shape[0] >= len(samples):
                    continue
                a_np = np.setdiff1d(all_sample_np, q_np, assume_unique=False)
                q_idx = torch.as_tensor(q_np, dtype=torch.long, device=device)
                a_idx = torch.as_tensor(a_np, dtype=torch.long, device=device)
                anchored_col = anchor_col_embed_torch(
                    col_embed,
                    tensors["sample_feature"],
                    q_idx,
                    a_idx,
                    k=args.anchor_k,
                    temperature=args.anchor_temperature,
                )
                for local_start in range(0, q_np.shape[0], args.batch_size):
                    local_end = min(q_np.shape[0], local_start + args.batch_size)
                    src = q_idx[local_start:local_end]
                    pred_b, _, _, _, _ = model.decode(
                        row_embed,
                        anchored_col[local_start:local_end],
                        tensors["baseline"].index_select(0, src),
                        tensors["mask"].index_select(0, src),
                        tensors["site_prior"],
                        sample_context=tensors["sample_context"].index_select(0, src),
                    )
                    pred_np[q_np[local_start:local_end], :] = pred_b.detach().cpu().numpy()
        return pred_np

    for epoch in range(1, args.epochs + 1):
        model.train()
        row_embed, col_embed, out_features = model.graph_core(
            tensors["feature_x"], tensors["col_edge_index"], tensors["row_edge_index"], collect_attention=True
        )
        n_samples = tensors["baseline"].shape[0]
        n_batches = int(math.ceil(n_samples / args.batch_size))
        batch_losses = []
        pred_losses = []
        delta_losses = []
        attention_values = []
        graph_abs_values = []
        residual_abs_values = []
        opt.zero_grad()
        for start_i in range(0, n_samples, args.batch_size):
            end_i = min(n_samples, start_i + args.batch_size)
            idx = torch.arange(start_i, end_i, dtype=torch.long, device=device)
            pred_b, delta_b, graph_b, residual_b, attention_b = model.decode(
                row_embed, col_embed,
                tensors["baseline"].index_select(0, idx),
                tensors["mask"].index_select(0, idx),
                tensors["site_prior"],
                sample_context=tensors["sample_context"],
                sample_idx=idx,
            )
            y_b = tensors["y"].index_select(0, idx)
            r_b = tensors["residual_true"].index_select(0, idx)
            m_b = tensors["mask"].index_select(0, idx)
            pred_b_loss = masked_huber(pred_b, y_b, m_b, tensors["site_weight"]) + 0.35 * cosine_loss(pred_b, y_b, m_b)
            delta_b_loss = masked_huber(delta_b, r_b, m_b, tensors["site_weight"]) + 0.20 * cosine_loss(delta_b, r_b, m_b)
            batch_obj = pred_b_loss + 0.35 * delta_b_loss + 0.02 * masked_l2(delta_b, m_b, tensors["site_weight"])
            (batch_obj / n_batches).backward(retain_graph=True)
            batch_losses.append(batch_obj.detach())
            pred_losses.append(pred_b_loss.detach())
            delta_losses.append(delta_b_loss.detach())
            attention_values.append(attention_b.detach().mean())
            graph_abs_values.append(graph_b.detach().abs().mean())
            residual_abs_values.append(residual_b.detach().abs().mean())
        pseudo_losses = []
        pseudo_group = group_names[(epoch - 1) % len(group_names)] if group_names else None
        if pseudo_group is not None:
            q_np = group_indices[pseudo_group]
            if q_np.shape[0] >= 3 and q_np.shape[0] < n_samples:
                a_np = np.setdiff1d(all_sample_np, q_np, assume_unique=False)
                q_idx_all = torch.as_tensor(q_np, dtype=torch.long, device=device)
                a_idx = torch.as_tensor(a_np, dtype=torch.long, device=device)
                anchored_col = anchor_col_embed_torch(
                    col_embed,
                    tensors["sample_feature"],
                    q_idx_all,
                    a_idx,
                    k=args.anchor_k,
                    temperature=args.anchor_temperature,
                )
                n_pseudo_batches = int(math.ceil(q_np.shape[0] / args.batch_size))
                for local_start in range(0, q_np.shape[0], args.batch_size):
                    local_end = min(q_np.shape[0], local_start + args.batch_size)
                    src = q_idx_all[local_start:local_end]
                    pred_b, delta_b, _, _, _ = model.decode(
                        row_embed,
                        anchored_col[local_start:local_end],
                        tensors["baseline"].index_select(0, src),
                        tensors["mask"].index_select(0, src),
                        tensors["site_prior"],
                        sample_context=tensors["sample_context"].index_select(0, src),
                    )
                    y_b = tensors["y"].index_select(0, src)
                    r_b = tensors["residual_true"].index_select(0, src)
                    m_b = tensors["mask"].index_select(0, src)
                    pseudo_obj = (
                        masked_huber(pred_b, y_b, m_b, tensors["site_weight"])
                        + 0.35 * cosine_loss(pred_b, y_b, m_b)
                        + 0.25 * masked_huber(delta_b, r_b, m_b, tensors["site_weight"])
                    )
                    (args.pseudo_weight * pseudo_obj / n_pseudo_batches).backward(retain_graph=True)
                    pseudo_losses.append(pseudo_obj.detach())
        pred_loss = torch.stack(pred_losses).mean()
        delta_loss = torch.stack(delta_losses).mean()
        pseudo_loss = torch.stack(pseudo_losses).mean() if pseudo_losses else torch.tensor(0.0, device=device)
        row_loss = model.graph_core.recon_loss(row_embed, tensors["row_edge_index"], sig=True)
        reg_loss = model.graph_core.recon_loss(out_features.T, tensors["row_edge_index"], sig=False)
        out_norm = (out_features - out_features.mean(axis=0)) / (out_features.std(axis=0) + EPS)
        col_loss = F.mse_loss(tensors["feature_x"][tensors["hv_idx"]].T, out_norm[:, tensors["hv_idx"]])
        attention_penalty = torch.stack(attention_values).mean()
        loss = (
            torch.stack(batch_losses).mean()
            + args.pseudo_weight * pseudo_loss
            + args.ppi_weight * row_loss
            + args.baseline_weight * (col_loss + reg_loss)
            + args.attention_l1 * attention_penalty
        )
        graph_reg_loss = args.ppi_weight * row_loss + args.baseline_weight * (col_loss + reg_loss) + args.attention_l1 * attention_penalty
        graph_reg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if model.graph_core.cols_encoder.atten_map is not None:
            new_col_edge, knn_df = model.graph_core.cols_encoder.reduce_network(min_connect=args.min_connect)
            if epoch % args.reduce_interval == 0:
                tensors["col_edge_index"] = new_col_edge
                knn_df.to_csv(out / "tables" / f"sample_attention_epoch_{epoch}.tsv", sep="\t", index=False)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                row_embed, col_embed, out_features = model.graph_core(
                    tensors["feature_x"], tensors["col_edge_index"], tensors["row_edge_index"], collect_attention=False
                )
                pred_chunks = []
                graph_abs_eval = []
                residual_abs_eval = []
                att_eval = []
                for start_i in range(0, tensors["baseline"].shape[0], args.batch_size):
                    end_i = min(tensors["baseline"].shape[0], start_i + args.batch_size)
                    idx = torch.arange(start_i, end_i, dtype=torch.long, device=device)
                    pred_b, delta_b, graph_b, residual_b, attention_b = model.decode(
                        row_embed, col_embed,
                        tensors["baseline"].index_select(0, idx),
                        tensors["mask"].index_select(0, idx),
                        tensors["site_prior"],
                        sample_context=tensors["sample_context"],
                        sample_idx=idx,
                    )
                    pred_chunks.append(pred_b.detach().cpu())
                    graph_abs_eval.append(graph_b.detach().abs().mean().cpu())
                    residual_abs_eval.append(residual_b.detach().abs().mean().cpu())
                    att_eval.append(attention_b.detach().mean().cpu())
            pred_np = torch.cat(pred_chunks, dim=0).numpy()
            per = spearman_cols(y, pred_np, mask, targets)
            summary = summarize(per, "scp682_general_graph_residual_trainmode")
            pseudo_pred_np = decode_pseudo_external(row_embed, col_embed)
            pseudo_per = spearman_cols(y, pseudo_pred_np, mask, targets)
            pseudo_summary = summarize(pseudo_per, "scp682_general_graph_residual_pseudo_external")
            base_per = spearman_cols(y, baseline, mask, targets)
            base_summary = summarize(base_per, "scp682_general_baseline")
            cos = np.nanmean([cosine_np(pred_np[i], y[i], mask[i]) for i in range(y.shape[0])])
            pseudo_cos = np.nanmean([cosine_np(pseudo_pred_np[i], y[i], mask[i]) for i in range(y.shape[0])])
            row = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "median_spearman": summary["median_spearman"],
                "mean_spearman": summary["mean_spearman"],
                "pseudo_external_median_spearman": pseudo_summary["median_spearman"],
                "pseudo_external_mean_spearman": pseudo_summary["mean_spearman"],
                "baseline_median_spearman": base_summary["median_spearman"],
                "sample_cosine": float(cos),
                "pseudo_external_sample_cosine": float(pseudo_cos),
                "pred_loss": float(pred_loss.detach().cpu()),
                "delta_loss": float(delta_loss.detach().cpu()),
                "pseudo_loss": float(pseudo_loss.detach().cpu()),
                "pseudo_group": str(pseudo_group),
                "row_loss": float(row_loss.detach().cpu()),
                "col_loss": float(col_loss.detach().cpu()),
                "reg_loss": float(reg_loss.detach().cpu()),
                "attention_mean": float(torch.stack(att_eval).mean()),
                "graph_abs": float(torch.stack(graph_abs_eval).mean()),
                "residual_abs": float(torch.stack(residual_abs_eval).mean()),
                "elapsed_sec": round(time.time() - start, 2),
            }
            logs.append(row)
            print(json.dumps(row), flush=True)
            pd.DataFrame(logs).to_csv(out / "logs/training_history.tsv", sep="\t", index=False)
            if row["pseudo_external_median_spearman"] > best.get("pseudo_external_median_spearman", -1e9):
                best = row
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "targets": targets,
                    "samples": list(samples),
                    "site_prior": site_prior_np,
                    "site_edge_index": edge_index_np,
                    "sample_edge_index": tensors["col_edge_index"].detach().cpu(),
                    "group_column": args.group_column,
                    "group_labels": group_labels.tolist(),
                    "meta": meta,
                    "best": best,
                }, out / "models/scp682_general_graph_residual_best.pt")
                per.to_csv(out / "tables/per_site_spearman_best.tsv", sep="\t", index=False)
                pseudo_per.to_csv(out / "tables/per_site_pseudo_external_spearman_best.tsv", sep="\t", index=False)
                pd.DataFrame([base_summary, summary, pseudo_summary]).to_csv(out / "tables/model_summary_best.tsv", sep="\t", index=False)
                pd.DataFrame(pred_np, index=samples, columns=targets).to_parquet(out / "predictions/scp682_general_graph_residual_trainmode_phosphosite_best.parquet")
                pd.DataFrame(pseudo_pred_np, index=samples, columns=targets).to_parquet(out / "predictions/scp682_general_graph_residual_pseudo_external_phosphosite_best.parquet")

    (out / "done.txt").write_text("done\n", encoding="utf-8")
    (out / "reports/final_summary.json").write_text(json.dumps({"best": best, "meta": meta}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"best": best, "meta": meta}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
