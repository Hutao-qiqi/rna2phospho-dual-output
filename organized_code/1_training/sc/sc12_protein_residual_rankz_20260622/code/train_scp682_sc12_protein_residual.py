# 模型: SCP682-SC12 / SCP682-SC-PR
# 作用: 在 SC11 的通路注意力与位点图基础上加入 RNA->总蛋白预测底座，训练蛋白丰度校正后的磷酸化残差头。
# 输入: SC11 统一模型输入 + scTranslator/scProTrans 蛋白预测缓存。
# 缓存格式: protein_pred.npy 或 predicted_protein_matrix.npy；protein_table.tsv；可选 cell_metadata.tsv。

import argparse
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train_scp682_sc11_expanded_scnet_site_gnn as sc11


MODEL_NAME = "SCP682-SC12-protein-residual"
PROTEIN_MATRIX_NAMES = ("protein_predicted.npy", "protein_pred.npy", "predicted_protein_matrix.npy", "proteome_pred.npy")
PROTEIN_TABLE_NAMES = ("protein_features.tsv", "protein_table.tsv", "proteins.tsv", "protein_order.tsv")
CELL_KEY_CANDIDATES = ("cell_id", "cell", "barcode", "obs_name", "sample_barcode", "cell_key")
ORIGINAL_EMBEDDING_PRIOR_LOSS = sc11.embedding_prior_loss


def normalize_symbol(value):
    text = str(value).strip().upper()
    text = text.replace("PHOSPHO-", "").replace("PHOSPHO_", "")
    text = text.replace("ANTI-", "").replace("ANTI_", "")
    if text.startswith("P-") and len(text) > 2:
        text = text[2:]
    text = text.strip("-_ ")
    if not text or text in {"NAN", "NONE", "NULL", "NA"}:
        return ""
    if text.isdigit() or len(text) <= 1:
        return ""
    synonym = {
        "C-JUN": "JUN",
        "ERK1": "MAPK3",
        "ERK2": "MAPK1",
        "H2AFX": "H2AX",
        "H3F3A": "H3-3A",
        "H3F3B": "H3-3B",
    }
    text = synonym.get(text, text)
    if text in {
        "P65",
        "IKKA",
        "IKKB",
        "AMPK-A1",
        "AMPK-A2",
        "AMPK-B1",
        "HISTONH2A.X",
        "HISTONH3",
        "JNK",
        "TOR",
        "PKC-B1",
        "STAT5",
    }:
        return ""
    if text.startswith(("INTRA", "HM", "BHM")) or "IGG" in text:
        return ""
    if len(text) > 15:
        return ""
    if "." in text:
        return ""
    if len(text) >= 2 and text[0] in {"S", "T", "Y"} and text[1:].isdigit():
        return ""
    return text


def protein_symbols_for_target(row):
    symbols = []
    symbols.extend(sc11.target_symbols_for_transfer(row))
    for col in ("protein_symbol", "gene", "gene_symbol"):
        text = str(row.get(col, ""))
        text = text.replace("/", ";").replace(",", ";").replace("|", ";")
        for token in text.split(";"):
            token = normalize_symbol(token)
            if token and token.isalnum() and not token.startswith("P"):
                symbols.append(token)
    alias = {
        "P-P65": ["RELA"],
        "P-RB": ["RB1"],
        "P-BLNK": ["BLNK"],
        "P-HISTONH2A.X": ["H2AX"],
        "P-HISTONE_H2A.X": ["H2AX"],
        "P-HISTONH3": ["H3C1", "H3-3A", "H3-3B"],
        "P-SRC": ["SRC", "LYN", "FYN", "LCK"],
        "P-JNK": ["MAPK8", "MAPK9"],
        "P-IKKA/B": ["CHUK", "IKBKB"],
        "P-AMPK-A1/2": ["PRKAA1", "PRKAA2"],
        "P-AMPK-B1": ["PRKAB1"],
        "P-CDK1": ["CDK1"],
        "P-CDK4": ["CDK4"],
        "P-IRAK4": ["IRAK4"],
        "P-JAK1": ["JAK1"],
        "P-STAT5": ["STAT5A", "STAT5B"],
        "P-STAT6": ["STAT6"],
        "P-C-JUN": ["JUN"],
        "P44/42": ["MAPK1", "MAPK3"],
        "PHOSPHO_P44_42": ["MAPK1", "MAPK3"],
        "HM0025": ["MAPK1", "MAPK3"],
    }
    joined = " ".join(str(row.get(c, "")).upper() for c in ("target_id", "canonical_label", "feature_id"))
    for key, vals in alias.items():
        if key in joined:
            symbols.extend(vals)
    if not symbols:
        for col in ("target_id", "canonical_label"):
            text = str(row.get(col, ""))
            text = text.replace("/", ";").replace(",", ";").replace("|", ";").replace("_", ";")
            for token in text.split(";"):
                token = normalize_symbol(token)
                if token and token.isalnum() and not token.startswith("P"):
                    symbols.append(token)
    clean = []
    for sym in symbols:
        sym = normalize_symbol(sym)
        if sym and sym not in {"PENDING", "PENDING_ANTIBODY_CLONE", "UNKNOWN"}:
            clean.append(sym)
    return sorted(set(clean))


def mapping_level_for_target(row, n_matched):
    joined = " ".join(str(row.get(c, "")).upper() for c in ("target_id", "canonical_label", "feature_id", "residue"))
    if n_matched <= 0:
        return "unmapped_manual_review"
    if "PSITEPENDING" in joined or "PENDING_ANTIBODY_CLONE" in joined:
        return "parent_only_site_pending"
    if any(token in joined for token in ("_S", "_T", "_Y", "|S", "|T", "|Y")):
        return "parent_only_site_named"
    return "parent_only"


def find_existing(base, names):
    for name in names:
        path = base / name
        if path.exists():
            return path
    return None


def protein_name_column(table):
    for col in ("protein_symbol", "gene", "gene_symbol", "protein", "feature", "name", "target"):
        if col in table.columns:
            return col
    return table.columns[0]


def align_cache_cells(cache_meta, input_meta):
    if cache_meta is None:
        return None
    for col in CELL_KEY_CANDIDATES:
        if col in cache_meta.columns and col in input_meta.columns:
            lookup = {str(v): i for i, v in enumerate(cache_meta[col].astype(str).tolist())}
            order = []
            missing = 0
            for value in input_meta[col].astype(str).tolist():
                idx = lookup.get(value)
                if idx is None:
                    missing += 1
                    order.append(-1)
                else:
                    order.append(idx)
            if missing == 0:
                return np.asarray(order, dtype=np.int64)
    return None


def subset_inputs_to_cache_cells_if_smoke(args, protein_cache_dir, meta, features, present, y, obs_mask):
    if not getattr(args, "smoke", False):
        return meta, features, present, y, obs_mask, None
    protein_cache_dir = Path(protein_cache_dir)
    cache_meta_path = protein_cache_dir / "cell_metadata.tsv"
    if not cache_meta_path.exists():
        cache_meta_path = protein_cache_dir / "cells.tsv"
    if not cache_meta_path.exists():
        return meta, features, present, y, obs_mask, None
    cache_meta = pd.read_csv(cache_meta_path, sep="\t", low_memory=False)
    if len(cache_meta) >= len(meta):
        return meta, features, present, y, obs_mask, None
    for col in CELL_KEY_CANDIDATES:
        if col in cache_meta.columns and col in meta.columns:
            wanted = {str(v) for v in cache_meta[col].astype(str).tolist()}
            keep = np.flatnonzero(meta[col].astype(str).isin(wanted).to_numpy())
            if len(keep) == len(cache_meta):
                return (
                    meta.iloc[keep].reset_index(drop=True),
                    np.asarray(features[keep], dtype=np.float32),
                    np.asarray(present[keep], dtype=np.float32),
                    np.asarray(y[keep], dtype=np.float32),
                    np.asarray(obs_mask[keep], dtype=bool),
                    keep,
                )
    return meta, features, present, y, obs_mask, None


def load_protein_cache(cache_dir, input_meta, target_rows, out_dir, require_all_targets=True):
    cache_dir = Path(cache_dir)
    matrix_path = find_existing(cache_dir, PROTEIN_MATRIX_NAMES)
    table_path = find_existing(cache_dir, PROTEIN_TABLE_NAMES)
    if matrix_path is None or table_path is None:
        raise FileNotFoundError(
            "protein cache needs one matrix file "
            f"{PROTEIN_MATRIX_NAMES} and one table file {PROTEIN_TABLE_NAMES}; got {cache_dir}"
        )
    protein_matrix = np.load(matrix_path, mmap_mode="r")
    protein_table = pd.read_csv(table_path, sep="\t")
    name_col = protein_name_column(protein_table)
    protein_names = [normalize_symbol(x) for x in protein_table[name_col].tolist()]
    protein_to_indices = {}
    for i, name in enumerate(protein_names):
        if name:
            protein_to_indices.setdefault(name, []).append(i)

    cache_meta_path = cache_dir / "cell_metadata.tsv"
    if not cache_meta_path.exists():
        cache_meta_path = cache_dir / "cells.tsv"
    cache_meta = pd.read_csv(cache_meta_path, sep="\t", low_memory=False) if cache_meta_path.exists() else None
    if protein_matrix.shape[0] != len(input_meta):
        order = align_cache_cells(cache_meta, input_meta)
        if order is None:
            raise ValueError(
                f"protein matrix rows={protein_matrix.shape[0]} but model cells={len(input_meta)}; "
                "provide cache cell_metadata.tsv with a shared cell key"
            )
        protein_matrix = protein_matrix[order]

    target_protein = np.zeros((len(input_meta), len(target_rows)), dtype=np.float32)
    audit_rows = []
    for j, row in enumerate(target_rows):
        symbols = protein_symbols_for_target(row)
        matched = []
        for sym in symbols:
            matched.extend(protein_to_indices.get(sym, []))
        matched = sorted(set(matched))
        if matched:
            vals = np.asarray(protein_matrix[:, matched], dtype=np.float32)
            target_protein[:, j] = np.nanmean(vals, axis=1).astype(np.float32)
        audit_rows.append(
            {
                "target_order": j,
                "target_id": row.get("target_id", ""),
                "target_index": int(row.get("target_index", j)),
                "source_feature_id": row.get("feature_id", ""),
                "protein_symbol": row.get("protein_symbol", ""),
                "canonical_label": row.get("canonical_label", ""),
                "residue": row.get("residue", ""),
                "candidate_parent_proteins": ";".join(symbols),
                "primary_parent_gene": symbols[0] if symbols else "",
                "parent_candidate_count": int(len(symbols)),
                "matched_cache_indices": ";".join(str(x) for x in matched),
                "matched_cache_names": ";".join(protein_names[x] for x in matched),
                "n_matched_parent_proteins": int(len(matched)),
                "mapping_level": mapping_level_for_target(row, len(matched)),
                "manual_review_required": bool(len(matched) == 0),
                "match_rule": "protein_symbol_or_alias_to_predicted_total_protein",
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out_dir / "tables" / "scp682_sc12_parent_protein_mapping.tsv", sep="\t", index=False)
    missing = audit[audit["n_matched_parent_proteins"].le(0)].copy()
    if require_all_targets and len(missing):
        missing_path = out_dir / "tables" / "scp682_sc12_parent_protein_missing.tsv"
        missing.to_csv(missing_path, sep="\t", index=False)
        raise ValueError(f"{len(missing)} targets have no matched parent protein in cache; see {missing_path}")
    with (out_dir / "reports" / "scp682_sc12_protein_cache.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "cache_dir": str(cache_dir),
                "matrix_path": str(matrix_path),
                "table_path": str(table_path),
                "matrix_shape": [int(protein_matrix.shape[0]), int(protein_matrix.shape[1])],
                "n_targets": int(len(target_rows)),
                "n_targets_with_parent_protein": int((audit["n_matched_parent_proteins"] > 0).sum()),
            },
            fh,
            indent=2,
        )
    return target_protein, audit


def standardize_parent_protein_features(target_protein, train_idx, mask):
    x = np.asarray(target_protein, dtype=np.float32).copy()
    rows = []
    for j in range(x.shape[1]):
        idx = np.asarray(train_idx, dtype=np.int64)[mask[train_idx, j] & np.isfinite(x[train_idx, j])]
        vals = x[idx, j].astype(np.float64)
        if len(vals):
            mean = float(np.mean(vals))
            sd = float(np.std(vals))
            if sd <= 0:
                sd = 1.0
        else:
            obs = np.flatnonzero(np.isfinite(x[:, j]))
            vals = x[obs, j].astype(np.float64) if len(obs) else np.asarray([0.0])
            mean = float(np.mean(vals))
            sd = float(np.std(vals)) or 1.0
        ok = np.isfinite(x[:, j])
        x[~ok, j] = mean
        x[:, j] = ((x[:, j] - mean) / sd).astype(np.float32)
        rows.append({"target_order": j, "protein_mean": mean, "protein_sd": sd, "n_reference_cells": int(len(vals))})
    return x, pd.DataFrame(rows)


def fit_frozen_parent_protein_baseline(y, mask, target_protein, train_idx, target_rows):
    train_idx = np.asarray(train_idx, dtype=np.int64)
    beta = np.zeros(y.shape[1], dtype=np.float32)
    bias = np.zeros(y.shape[1], dtype=np.float32)
    baseline = np.zeros_like(y, dtype=np.float32)
    rows = []
    for j in range(y.shape[1]):
        usable = train_idx[mask[train_idx, j] & np.isfinite(y[train_idx, j]) & np.isfinite(target_protein[train_idx, j])]
        xj = target_protein[usable, j].astype(np.float64)
        yj = y[usable, j].astype(np.float64)
        fit_status = "fit"
        if len(usable) >= 3 and float(np.var(xj)) > 1e-8:
            x_mean = float(np.mean(xj))
            y_mean = float(np.mean(yj))
            slope = float(np.sum((xj - x_mean) * (yj - y_mean)) / max(np.sum((xj - x_mean) ** 2), 1e-12))
            intercept = float(y_mean - slope * x_mean)
        elif len(usable):
            slope = 0.0
            intercept = float(np.mean(yj))
            fit_status = "mean_only"
        else:
            slope = 0.0
            intercept = 0.0
            fit_status = "no_training_observation"
        beta[j] = slope
        bias[j] = intercept
        baseline[:, j] = (target_protein[:, j].astype(np.float32) * beta[j] + bias[j]).astype(np.float32)
        rows.append(
            {
                "target_order": j,
                "target_id": target_rows[j].get("target_id", ""),
                "protein_symbol": target_rows[j].get("protein_symbol", ""),
                "protein_beta": float(beta[j]),
                "protein_bias": float(bias[j]),
                "n_fit_cells": int(len(usable)),
                "fit_status": fit_status,
            }
        )
    residual = (y.astype(np.float32) - baseline).astype(np.float32)
    return baseline, residual, beta, bias, pd.DataFrame(rows)


def sc12_embedding_prior_loss(x, present, h, args):
    rna_dim = int(getattr(args, "sc12_rna_dim_for_prior", 0) or 0)
    if rna_dim > 0 and x.ndim == 2 and x.shape[1] > rna_dim:
        x = x[:, :rna_dim]
    return ORIGINAL_EMBEDDING_PRIOR_LOSS(x, present, h, args)


class ProteinResidualPathwayPredictor(sc11.ScFoundationPathwayPredictor):
    default_disable_pathway_attention = False
    default_train_residual_target = False
    default_protein_beta = None
    default_protein_bias = None

    def __init__(self, n_pathways, n_targets, d_input, target_pathway_prior, **kwargs):
        if d_input <= n_targets:
            raise ValueError("SC12 expects features = RNA embedding concatenated with one parent-protein feature per target")
        disable_pathway_attention = bool(kwargs.pop("disable_pathway_attention", self.default_disable_pathway_attention))
        train_residual_target = bool(kwargs.pop("train_residual_target", self.default_train_residual_target))
        dropout = float(kwargs.get("dropout", 0.15))
        self.rna_dim = int(d_input - n_targets)
        self.n_protein_targets = int(n_targets)
        super().__init__(n_pathways, n_targets, self.rna_dim, target_pathway_prior, **kwargs)
        self.disable_pathway_attention = disable_pathway_attention
        self.train_residual_target = train_residual_target
        self.output_residual_target = train_residual_target
        beta = torch.zeros(n_targets, dtype=torch.float32)
        bias = torch.zeros(n_targets, dtype=torch.float32)
        if self.default_protein_beta is not None:
            beta = torch.as_tensor(self.default_protein_beta, dtype=torch.float32).clone()
        if self.default_protein_bias is not None:
            bias = torch.as_tensor(self.default_protein_bias, dtype=torch.float32).clone()
        self.protein_beta = nn.Parameter(beta, requires_grad=False)
        self.protein_bias = nn.Parameter(bias, requires_grad=False)
        self.protein_context_proj = nn.Sequential(
            nn.LayerNorm(1),
            nn.Linear(1, self.hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.protein_context_gate = nn.Sequential(
            nn.LayerNorm(self.hidden * 2),
            nn.Linear(self.hidden * 2, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        self.protein_context_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def encode(self, x, present):
        return super().encode(x[:, : self.rna_dim], present)

    def parent_protein_features(self, x):
        return x[:, self.rna_dim : self.rna_dim + self.n_protein_targets]

    def protein_component(self, x):
        protein_x = self.parent_protein_features(x)
        return protein_x * self.protein_beta.view(1, -1) + self.protein_bias.view(1, -1)

    def apply_protein_context(self, site, protein_x):
        if protein_x is None:
            return site
        context = self.protein_context_proj(protein_x.unsqueeze(-1).to(site.dtype))
        gate = torch.sigmoid(self.protein_context_gate(torch.cat([site, context], dim=-1)))
        return site + torch.tanh(self.protein_context_scale) * gate * context

    def phospho_from_state(self, h, present, return_attention=False, protein_x=None):
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
        if self.disable_pathway_attention:
            token_mask = (effective_present > 0).to(h.dtype)
            pooled = (h * token_mask.unsqueeze(-1)).sum(dim=1)
            denom = token_mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            pooled = pooled / denom
            site = query + pooled.unsqueeze(1)
            weights = torch.zeros(b, site_embedding.shape[0], h.shape[1], dtype=h.dtype, device=h.device)
        else:
            key_padding_mask = effective_present <= 0
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
        site = self.apply_protein_context(site, protein_x)
        pred = self.head(site).squeeze(-1)
        if return_attention:
            return pred, weights
        return pred

    def forward(self, x, present, return_attention=False):
        h = self.encode(x, present)
        protein_x = self.parent_protein_features(x)
        if return_attention:
            residual, weights = self.phospho_from_state(h, present, True, protein_x=protein_x)
            pred = residual if self.output_residual_target else residual + self.protein_component(x)
            return pred, h, weights
        residual = self.phospho_from_state(h, present, protein_x=protein_x)
        pred = residual if self.output_residual_target else residual + self.protein_component(x)
        return pred, h


@torch.no_grad()
def predict_components(model, features, present, idx, args, device):
    full = []
    protein = []
    residual = []
    model.eval()
    for start in range(0, len(idx), args.eval_batch_size):
        batch = idx[start : start + args.eval_batch_size]
        xb = torch.as_tensor(np.asarray(features[batch]), dtype=torch.float32, device=device)
        pb = torch.as_tensor(np.asarray(present[batch]), dtype=torch.float32, device=device)
        pred, _ = model(xb, pb)
        protein_part = model.protein_component(xb)
        full.append(pred.detach().cpu().numpy().astype(np.float32))
        protein.append(protein_part.detach().cpu().numpy().astype(np.float32))
        residual.append((pred - protein_part).detach().cpu().numpy().astype(np.float32))
    empty = np.zeros((0, model.site_embedding.shape[0]), dtype=np.float32)
    return (
        np.vstack(full) if full else empty,
        np.vstack(protein) if protein else empty,
        np.vstack(residual) if residual else empty,
    )


def evaluate_component_tables(model, features, present, y, mask, meta, train_info, target_rows, args, out_dir):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    rng = np.random.default_rng(args.seed + 101)
    train_idx = np.asarray(train_info.get("train_idx", []), dtype=np.int64)
    fit_idx = np.asarray(train_info.get("fit_idx", train_idx), dtype=np.int64)
    val_idx = np.asarray(train_info.get("val_idx", []), dtype=np.int64)
    eval_sets = []
    if len(val_idx):
        eval_sets.append(("internal_cv_reconstruction", val_idx))
    if len(fit_idx):
        eval_sets.append(("train_reconstruction", sc11.sample_idx(fit_idx, args.max_eval_train_cells if not args.smoke else 4096, rng)))
    hold_idx = np.flatnonzero(meta["dataset_id"].astype(str).isin(sc11.parse_list(args.holdout_datasets)).to_numpy() & mask.any(axis=1))
    if getattr(args, "exclude_train_from_holdout", False) and len(hold_idx):
        train_set = set(int(x) for x in np.asarray(train_idx, dtype=np.int64))
        hold_idx = np.asarray([int(x) for x in hold_idx if int(x) not in train_set], dtype=np.int64)
    if len(hold_idx):
        eval_sets.append(("external_reconstruction", hold_idx))

    rows = []
    for evaluation, idx in eval_sets:
        full_pred, protein_pred, residual_pred = predict_components(model, features, present, idx, args, device)
        for component, pred in (
            ("protein_only", protein_pred),
            ("phospho_residual_only", residual_pred),
            ("protein_plus_residual", full_pred),
        ):
            part = sc11.score_prediction(y, mask, pred, meta, idx, target_rows, evaluation)
            for rec in part:
                rec["component"] = component
            rows.extend(part)
    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "tables" / "scp682_sc12_component_performance.tsv", sep="\t", index=False)
    if comp.empty:
        return
    summary = (
        comp.groupby(["evaluation", "test_dataset", "component"], as_index=False)
        .agg(n_targets=("target_id", "nunique"), median_spearman=("spearman", "median"), mean_spearman=("spearman", "mean"))
    )
    wide = summary.pivot_table(
        index=["evaluation", "test_dataset"],
        columns="component",
        values="median_spearman",
        aggfunc="first",
    ).reset_index()
    if "protein_only" in wide.columns and "protein_plus_residual" in wide.columns:
        wide["residual_gain_over_protein_only"] = wide["protein_plus_residual"] - wide["protein_only"]
    if "phospho_residual_only" in wide.columns and "protein_plus_residual" in wide.columns:
        wide["protein_component_gain_over_residual_only"] = wide["protein_plus_residual"] - wide["phospho_residual_only"]
    wide.to_csv(out_dir / "tables" / "scp682_sc12_component_gain_summary.tsv", sep="\t", index=False)


def add_sc12_args(ap):
    ap.add_argument("--root", default=r"./data_root")
    ap.add_argument("--pathway-manifest", default=r"02_results\single_cell\20260519_scp682_sc3_multidomain_features_v1\intermediate\pathway_gene_manifest.tsv")
    ap.add_argument("--model-input-dir", default=r"01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_gse300551_signal_seq_multidomain_v1")
    ap.add_argument("--protein-cache-dir", required=True)
    ap.add_argument("--allow-missing-parent-protein", action="store_true", default=False)
    ap.add_argument("--output-dir", default=r"02_results\single_cell\20260621_scp682_sc12_protein_residual_v1")
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
    ap.add_argument("--disable-pathway-attention", action="store_true", default=False)
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
    ap.add_argument("--site-graph-edge-mode", choices=("all", "copheemap", "copheeksa", "kstar", "no_copheemap", "no_copheeksa", "no_kstar", "rewired_all"), default="all")
    ap.add_argument("--site-graph-rewire-seed", type=int, default=20260602)
    ap.add_argument("--site-graph-rewire-swaps-per-edge", type=int, default=10)
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


def run():
    sc11.MODEL_NAME = MODEL_NAME
    sc11.ScFoundationPathwayPredictor = ProteinResidualPathwayPredictor
    ap = argparse.ArgumentParser()
    add_sc12_args(ap)
    args = ap.parse_args()
    ProteinResidualPathwayPredictor.default_disable_pathway_attention = bool(args.disable_pathway_attention)

    root = Path(args.root)
    input_dir = root / args.model_input_dir
    pathway_manifest = root / args.pathway_manifest
    out_dir = sc11.ensure_dir(root / args.output_dir)
    for name in ("tables", "models", "logs", "reports"):
        sc11.ensure_dir(out_dir / name)

    features = np.load(input_dir / "embeddings.npy", mmap_mode="r")
    meta = pd.read_csv(input_dir / "cell_metadata.tsv", sep="\t", low_memory=False)
    manifest = pd.read_csv(pathway_manifest, sep="\t")
    pathway_names = list(dict.fromkeys(manifest["pathway"].astype(str).tolist()))
    present = np.ones((len(meta), len(pathway_names)), dtype=np.float32)

    target_table = pd.read_csv(input_dir / "phospho_target_table.tsv", sep="\t")
    target_indices, target_rows = sc11.choose_targets(target_table, sc11.parse_list(args.target_ids))
    y_all = np.load(input_dir / "targets.npy", mmap_mode="r")
    mask_all = np.load(input_dir / "target_mask.npy", mmap_mode="r")
    y_raw = np.asarray(y_all[:, target_indices], dtype=np.float32)
    obs_mask = np.asarray(mask_all[:, target_indices], dtype=bool) & np.isfinite(y_raw)
    train_idx_for_transform = sc11.build_train_idx(meta, obs_mask, args)
    transform_reference_idx = train_idx_for_transform
    if int(getattr(args, "cv_folds", 0) or 0) > 1:
        cv_rng = np.random.default_rng(args.seed)
        transform_reference_idx, _ = sc11.split_train_validation(train_idx_for_transform, meta, args, cv_rng)
    y, transform_stats = sc11.transform_targets(y_raw, obs_mask, transform_reference_idx, args.target_transform, target_rows)

    meta, features, present, y, obs_mask, smoke_subset_idx = subset_inputs_to_cache_cells_if_smoke(
        args,
        root / args.protein_cache_dir,
        meta,
        features,
        present,
        y,
        obs_mask,
    )
    if smoke_subset_idx is not None:
        transform_reference_idx = np.arange(len(meta), dtype=np.int64)
        train_idx_for_transform = transform_reference_idx
        pd.DataFrame({"original_cell_index": smoke_subset_idx}).to_csv(
            out_dir / "tables" / "scp682_sc12_smoke_cache_cell_subset.tsv", sep="\t", index=False
        )

    target_protein, _ = load_protein_cache(
        root / args.protein_cache_dir,
        meta,
        target_rows,
        out_dir,
        require_all_targets=not args.allow_missing_parent_protein,
    )
    target_protein, protein_stats = standardize_parent_protein_features(target_protein, transform_reference_idx, obs_mask)
    protein_stats.to_csv(out_dir / "tables" / "scp682_sc12_parent_protein_transform.tsv", sep="\t", index=False)
    protein_baseline, y_residual, protein_beta, protein_bias, protein_baseline_rows = fit_frozen_parent_protein_baseline(
        y,
        obs_mask,
        target_protein,
        transform_reference_idx,
        target_rows,
    )
    protein_baseline_rows.to_csv(
        out_dir / "tables" / "scp682_sc12_frozen_parent_protein_baseline.tsv",
        sep="\t",
        index=False,
    )
    combined_path = out_dir / "intermediate" / "scp682_sc12_rna_embedding_plus_parent_protein.npy"
    sc11.ensure_dir(combined_path.parent)
    if combined_path.exists():
        features_with_protein = np.load(combined_path, mmap_mode="r")
    else:
        features_with_protein = np.lib.format.open_memmap(
            combined_path,
            mode="w+",
            dtype=np.float32,
            shape=(features.shape[0], features.shape[1] + target_protein.shape[1]),
        )
        block = 8192
        for start in range(0, features.shape[0], block):
            end = min(start + block, features.shape[0])
            features_with_protein[start:end, : features.shape[1]] = np.asarray(features[start:end], dtype=np.float32)
            features_with_protein[start:end, features.shape[1] :] = target_protein[start:end].astype(np.float32)
        features_with_protein.flush()
        features_with_protein = np.load(combined_path, mmap_mode="r")

    contexts = []
    if args.delta_weight > 0 or args.include_drug_delta_eval:
        contexts = sc11.build_drug_contexts(meta, y_residual, obs_mask, args)
    train_meta = meta.iloc[train_idx_for_transform].copy()
    if "ibrutinib" in train_meta.columns:
        train_meta["_ibrutinib_bool"] = sc11.parse_bool(train_meta["ibrutinib"]).to_numpy()
    else:
        train_meta["_ibrutinib_bool"] = False
    train_manifest = train_meta.groupby(["dataset_id", "_ibrutinib_bool"], dropna=False).size().reset_index(name="n_cells").rename(columns={"_ibrutinib_bool": "ibrutinib"})
    train_manifest.to_csv(out_dir / "tables" / "scp682_sc12_train_manifest.tsv", sep="\t", index=False)

    local_target_pathway_prior, target_pathway_rows = sc11.build_target_pathway_prior(target_rows, pathway_names, manifest)
    transfer_prior = np.zeros_like(local_target_pathway_prior, dtype=np.float32)
    transfer_rows = pd.DataFrame()
    transfer_align = pd.DataFrame()
    full_transfer = None
    transfer_source = "none"
    if args.scp682_main_transfer_dir:
        main_transfer = sc11.load_scp682_main_transfer(target_rows, pathway_names, manifest, root / args.scp682_main_transfer_dir)
        if main_transfer:
            transfer_source = "scp682_main"
            transfer_prior = main_transfer.get("transfer_prior", transfer_prior)
            transfer_rows = main_transfer.get("transfer_rows", pd.DataFrame())
            transfer_align = main_transfer.get("pathway_alignment", pd.DataFrame())
            if args.full_transfer_scale > 0:
                full_transfer = main_transfer
    elif args.scp682_main_pathway_token_transfer_dir or args.scp68222_transfer_dir:
        transfer_source = "scp682_main" if args.scp682_main_pathway_token_transfer_dir else "scp68222"
        transfer_root = root / (args.scp682_main_pathway_token_transfer_dir or args.scp68222_transfer_dir)
        transfer_prior, transfer_rows, transfer_align = sc11.build_scp68222_transfer_prior(target_rows, pathway_names, manifest, transfer_root, args.transfer_alpha)
        if args.full_transfer_scale > 0:
            full_transfer = sc11.load_scp68222_full_transfer(target_rows, pathway_names, manifest, transfer_root)
    target_pathway_prior = sc11.mix_pathway_priors(local_target_pathway_prior, transfer_prior, args.transfer_alpha)
    site_graph_kwargs = {
        "candidate_limit": args.site_graph_candidate_limit,
        "max_aux_nodes": args.site_graph_max_aux_nodes,
        "edge_mode": args.site_graph_edge_mode,
        "rewire_seed": args.site_graph_rewire_seed,
        "rewire_swaps_per_edge": args.site_graph_rewire_swaps_per_edge,
    }
    supported_graph_args = set(inspect.signature(sc11.build_expanded_scnet_site_graph_prior).parameters)
    site_graph_kwargs = {k: v for k, v in site_graph_kwargs.items() if k in supported_graph_args}
    site_graph = sc11.build_expanded_scnet_site_graph_prior(
        target_rows,
        root / args.site_graph_prior_root,
        **site_graph_kwargs,
    )

    transform_stats.to_csv(out_dir / "tables" / "scp682_sc12_target_transform.tsv", sep="\t", index=False)
    pd.DataFrame(target_rows).to_csv(out_dir / "tables" / "scp682_sc12_target_table.tsv", sep="\t", index=False)
    target_pathway_rows.to_csv(out_dir / "tables" / "scp682_sc12_target_pathway_prior.tsv", sep="\t", index=False)
    transfer_rows.to_csv(out_dir / "tables" / f"scp682_sc12_{transfer_source}_transfer_prior.tsv", sep="\t", index=False)
    transfer_align.to_csv(out_dir / "tables" / f"scp682_sc12_{transfer_source}_pathway_alignment.tsv", sep="\t", index=False)
    site_graph.get("edges", pd.DataFrame()).to_csv(out_dir / "tables" / "scp682_sc12_site_graph_edges.tsv", sep="\t", index=False)
    site_graph.get("candidates", pd.DataFrame()).to_csv(out_dir / "tables" / "scp682_sc12_site_graph_candidates.tsv", sep="\t", index=False)
    site_graph.get("node_table", pd.DataFrame()).to_csv(out_dir / "tables" / "scp682_sc12_site_graph_nodes.tsv", sep="\t", index=False)

    args.sc12_rna_dim_for_prior = int(features.shape[1])
    sc11.embedding_prior_loss = sc12_embedding_prior_loss
    ProteinResidualPathwayPredictor.default_train_residual_target = True
    ProteinResidualPathwayPredictor.default_protein_beta = protein_beta
    ProteinResidualPathwayPredictor.default_protein_bias = protein_bias

    model, train_info = sc11.train_model(
        features_with_protein,
        present,
        y_residual,
        obs_mask,
        meta,
        contexts,
        pathway_names,
        target_rows,
        target_pathway_prior,
        transfer_prior,
        full_transfer,
        site_graph,
        args,
        out_dir,
    )
    model.output_residual_target = False
    sc11.evaluate(model, features_with_protein, present, y, obs_mask, meta, train_info, contexts, pathway_names, target_rows, args, out_dir)
    full_perf = out_dir / "tables" / "scp682_sc11_reconstruction_performance.tsv"
    if full_perf.exists():
        perf = pd.read_csv(full_perf, sep="\t")
        perf["model"] = MODEL_NAME
        perf.to_csv(out_dir / "tables" / "scp682_sc12_reconstruction_performance.tsv", sep="\t", index=False)
    evaluate_component_tables(model, features_with_protein, present, y, obs_mask, meta, train_info, target_rows, args, out_dir)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "n_pathways": int(len(pathway_names)),
                "n_targets": int(y.shape[1]),
                "d_rna_input": int(features.shape[1]),
                "d_input": int(features_with_protein.shape[1]),
                "hidden": int(args.hidden),
                "pathway_layers": int(args.pathway_layers),
                "attention_heads": int(args.attention_heads),
                "disable_pathway_attention": bool(args.disable_pathway_attention),
                "dropout": float(args.dropout),
                "target_transform": args.target_transform,
                "architecture": "scRNA_to_predicted_protein_to_phospho_residual",
                "formula": "phospho_pred = beta_site * predicted_parent_protein + phospho_residual(RNA_state,pathway_attention,site_graph,protein_context)",
                "protein_context": "predicted_parent_protein enters both frozen abundance component and residual head",
            },
            "args": vars(args),
            "pathway_names": pathway_names,
            "protein_beta": model.protein_beta.detach().cpu().numpy().astype(np.float32),
            "protein_bias": model.protein_bias.detach().cpu().numpy().astype(np.float32),
            "train_target": "phospho_residual_after_frozen_parent_protein_baseline",
            "target_rows": target_rows,
            "protein_cache_dir": str(root / args.protein_cache_dir),
        },
        out_dir / "models" / "scp682_sc12_final.pt",
    )
    with (out_dir / "reports" / "scp682_sc12_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": MODEL_NAME,
                "architecture": "scRNA_to_predicted_protein_to_phospho_residual",
                "formula": "phospho_pred = frozen_beta_site * predicted_parent_protein + frozen_bias_site + phospho_residual(RNA_state,pathway_attention,site_graph,protein_context)",
                "residual_target": "transformed_phospho - frozen_parent_protein_baseline",
                "protein_context": "predicted parent protein is projected into site-level residual decoding",
                "n_cells": int(len(meta)),
                "n_targets": int(len(target_rows)),
                "n_parent_protein_features": int(target_protein.shape[1]),
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        raise
