#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path("/data/lsy/Infinite_Stream")
LOCAL_ROOT = Path(r"E:/data/gongke/TCGA-TCPA")
RELEASE = ROOT / "SCP682/frozen_release/SCP682_main_exact_scnet_gnn_20260522"
TRAIN_SCRIPT = RELEASE / "scripts/train_scp682_exact_scnet_gnn_v1.py"
TRAIN_RNA = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2/rna_log2_tpm_paired.parquet"
TCPA_DIR = ROOT / "01_data/tcga_tcpa/processed/tcpa_32_project_rna_rppa_20260501"
BASELINE_FULL = ROOT / "SCP682-main/results/20260526_tcga_tcpa_overlap_scp682/predictions/tcga_baseline_mean_phosphosite.parquet"
OUT_REMOTE = ROOT / "SCP682-main/results/20260528_fig5_v2_exports"

PXD063604_OBS_LOCAL = LOCAL_ROOT / "remote_scripts/tmp_pxd063604_observed_long.tsv"
PXD039363_OBS_LOCAL = LOCAL_ROOT / "remote_scripts/tmp_pxd039363_observed_long.tsv"

REQUIRED_DRUG_COLUMNS = [
    "sample_id",
    "cell_line",
    "drug",
    "target_gene",
    "time_min",
    "gene_site",
    "predicted_value",
    "observed_value",
    "predicted_delta_vs_control",
    "observed_delta_vs_control",
]

CASES = [
    {"gene": "ARHGEF2", "site": "S886", "target": "ARHGEF2|S886", "cancer": "TCGA-KIRC"},
    {"gene": "HSPA1A", "site": "T265", "target": "HSPA1A|T265", "cancer": "TCGA-KIRC"},
    {"gene": "NFKB1", "site": "S893", "target": "NFKB1|S893", "cancer": "TCGA-LUAD"},
    {"gene": "CIRBP", "site": "S146", "target": "CIRBP|S146", "cancer": "TCGA-PAAD"},
    {"gene": "UCK1", "site": "S244", "target": "UCK1|S244", "cancer": "TCGA-PAAD"},
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    return out


def l2_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.sqrt((x * x).sum(axis=1, keepdims=True)).clip(min=1e-6)
    return x / norm


def topk_softmax_from_sim(sim: np.ndarray, k: int, temperature: float) -> np.ndarray:
    k = max(1, min(int(k), sim.shape[1]))
    top = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    vals = sim[rows, top] / max(float(temperature), 1e-6)
    vals = vals - vals.max(axis=1, keepdims=True)
    weights = np.exp(vals).astype(np.float32)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    out = np.zeros_like(sim, dtype=np.float32)
    out[rows, top] = weights
    return out


def read_tcga_rna() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(TCPA_DIR / "matrices/X_tcpa_32.symbols.parquet")
    manifest = pd.read_csv(TCPA_DIR / "tables/tcpa_32_sample_manifest.tsv", sep="\t")
    sample_cols = [c for c in raw.columns if c != "gene_symbol"]
    raw["gene_symbol"] = raw["gene_symbol"].astype(str)
    raw = raw.loc[raw["gene_symbol"].ne("") & raw["gene_symbol"].ne("nan")]
    raw[sample_cols] = raw[sample_cols].apply(pd.to_numeric, errors="coerce")
    mat = raw.groupby("gene_symbol", sort=False)[sample_cols].median(numeric_only=True)
    x = mat.T
    x.index.name = "sample_id"
    return clean_numeric(x), manifest


def align_by_key(df: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
    direct = df.reindex(target_index)
    if direct.notna().any(axis=None):
        return direct
    tmp = df.copy()
    tmp["_sample_key"] = [str(x).split("::", 1)[-1] for x in tmp.index]
    tmp = tmp.drop_duplicates("_sample_key").set_index("_sample_key")
    keys = [str(x).split("::", 1)[-1] for x in target_index]
    out = tmp.reindex(keys)
    out.index = target_index
    return out


def make_rna_context(train_rna: pd.DataFrame, train_samples: pd.Index, external_rna: pd.DataFrame, external_samples: pd.Index, n_genes: int):
    train_rna = clean_numeric(train_rna)
    external_rna = clean_numeric(external_rna)
    train_aligned = align_by_key(train_rna, train_samples)
    external_aligned = align_by_key(external_rna, external_samples)
    common = [g for g in train_aligned.columns if g in external_aligned.columns]
    train_block = train_aligned[common].apply(pd.to_numeric, errors="coerce")
    var = train_block.var(axis=0, skipna=True).sort_values(ascending=False)
    genes = list(var.index[: min(int(n_genes), len(var))])
    mean = train_block[genes].mean(axis=0, skipna=True)
    std = train_block[genes].std(axis=0, skipna=True).replace(0, np.nan).fillna(1.0)
    train_ctx = ((train_block[genes] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-6, 6)
    ext_block = external_aligned[genes].apply(pd.to_numeric, errors="coerce")
    ext_ctx = ((ext_block - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-6, 6)
    return train_ctx.astype(np.float32), ext_ctx.astype(np.float32), genes


def build_model(train_mod, ckpt: dict, device: torch.device):
    args = ckpt.get("args", {})
    model = train_mod.SCP682ExactScNETResidual(
        n_sites=len(ckpt["targets"]),
        n_samples=len(ckpt["samples"]),
        shrinkage=float(args.get("shrinkage", 0.3)),
        hidden=int(args.get("hidden", 160)),
        latent=int(args.get("latent", 64)),
        inter_dim=int(args.get("inter_dim", 192)),
        embd_dim=int(args.get("embd_dim", 64)),
        num_layers=int(args.get("num_layers", 2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def prepare_scp682_state(args, device: torch.device):
    train_mod = import_module(TRAIN_SCRIPT, "scp682_exact_scnet_release_train_fig5")
    ckpt = torch.load(RELEASE / "models/scp682_exact_scnet_gnn_best.pt", map_location="cpu", weights_only=False)
    model = build_model(train_mod, ckpt, device)
    samples = pd.Index([str(x) for x in ckpt["samples"]])
    targets = [str(x) for x in ckpt["targets"]]
    train_dir = RELEASE / "training_set"
    parent = clean_numeric(pd.read_parquet(train_dir / "oof_candidate_parent_only_phosphosite.parquet")).loc[samples, targets]
    ridge = clean_numeric(pd.read_parquet(train_dir / "oof_candidate_ridge_direct_phosphosite.parquet")).loc[samples, targets]
    rna_direct = clean_numeric(pd.read_parquet(train_dir / "oof_candidate_rna_direct_phosphosite.parquet")).loc[samples, targets]
    y = clean_numeric(pd.read_parquet(train_dir / "observed_phosphosite.parquet")).loc[samples, targets]
    train_base = ((parent + ridge + rna_direct) / 3.0).astype(np.float32)
    train_mask = np.isfinite(y.to_numpy(np.float32)) & np.isfinite(train_base.to_numpy(np.float32))
    train_base_np = np.nan_to_num(train_base.to_numpy(np.float32), nan=0.0).astype(np.float32)
    feature_x = np.where(train_mask.T, train_base_np.T, 0.0).astype(np.float32)
    mu = feature_x.mean(axis=1, keepdims=True)
    sd = feature_x.std(axis=1, keepdims=True) + 1e-5
    feature_x = ((feature_x - mu) / sd).astype(np.float32)
    row_edge = torch.as_tensor(ckpt["site_edge_index"], dtype=torch.long, device=device)
    sample_edge = ckpt["sample_edge_index"]
    if isinstance(sample_edge, torch.Tensor):
        sample_edge = sample_edge.detach().cpu().numpy()
    sample_edge = torch.as_tensor(sample_edge, dtype=torch.long, device=device)
    site_prior = torch.as_tensor(ckpt["site_prior"], dtype=torch.float32, device=device)
    with torch.no_grad():
        row_embed, col_embed_train, _ = model.graph_core(
            torch.as_tensor(feature_x, dtype=torch.float32, device=device),
            sample_edge,
            row_edge,
            collect_attention=False,
        )
    return {
        "train_mod": train_mod,
        "ckpt": ckpt,
        "model": model,
        "samples": samples,
        "targets": targets,
        "train_base": train_base_np,
        "train_mask": train_mask,
        "row_embed": row_embed,
        "col_embed_train": col_embed_train,
        "site_prior": site_prior,
    }


def decode_target(model, row_embed, col_embed_ext, baseline_vec, mask_vec, site_prior, target_idx: int) -> np.ndarray:
    device = col_embed_ext.device
    baseline = torch.as_tensor(baseline_vec, dtype=torch.float32, device=device)
    mask = torch.as_tensor(mask_vec, dtype=torch.bool, device=device)
    base_h = model.baseline_proj(torch.stack([baseline, mask.float()], dim=-1))
    site_h = model.site_proj(row_embed[target_idx]).unsqueeze(0).expand(baseline.shape[0], -1)
    sample_z = model.sample_proj(col_embed_ext)
    prior = site_prior[target_idx].expand(baseline.shape[0])
    prior_h = model.site_prior_proj(prior.unsqueeze(-1))
    graph_input = torch.cat([base_h + prior_h, site_h, sample_z, prior.unsqueeze(-1)], dim=-1)
    attention = torch.sigmoid(model.prior_attention(graph_input)).squeeze(-1)
    graph_delta = attention * model.graph_decoder(graph_input).squeeze(-1) * torch.sigmoid(model.graph_scale)
    residual_delta = model.residual(torch.cat([base_h, site_h, sample_z], dim=-1)).squeeze(-1) * torch.sigmoid(model.residual_scale)
    pred = baseline + model.shrinkage * (graph_delta + residual_delta)
    return pred.detach().cpu().numpy().astype(np.float32)


def load_external_blocks(args, state):
    tcga_rna, manifest = read_tcga_rna()
    baseline_ext = clean_numeric(pd.read_parquet(BASELINE_FULL))
    targets = state["targets"]
    ext_targets = [t for t in targets if t in baseline_ext.columns]
    baseline_full = pd.DataFrame(np.nan, index=baseline_ext.index, columns=targets, dtype=np.float32)
    baseline_full.loc[:, ext_targets] = baseline_ext.loc[:, ext_targets].to_numpy(np.float32)
    ext_base_np = np.nan_to_num(baseline_full.to_numpy(np.float32), nan=0.0).astype(np.float32)
    ext_mask = np.isfinite(baseline_full.to_numpy(np.float32))
    train_rna = pd.read_parquet(TRAIN_RNA)
    train_ctx, ext_ctx, genes = make_rna_context(train_rna, state["samples"], tcga_rna, baseline_full.index, args.rna_context_genes)
    train_base_norm = l2_rows(np.where(state["train_mask"], state["train_base"], 0.0))
    ext_base_norm = l2_rows(np.where(ext_mask, ext_base_np, 0.0))
    train_rna_raw = train_ctx.to_numpy(np.float32)
    ext_rna_raw = ext_ctx.to_numpy(np.float32)
    train_rna_norm = l2_rows(train_rna_raw)
    ext_rna_norm = l2_rows(ext_rna_raw)
    return {
        "tcga_rna": tcga_rna,
        "manifest": manifest,
        "baseline_full": baseline_full,
        "ext_base_np": ext_base_np,
        "ext_mask": ext_mask,
        "genes": genes,
        "train_base_norm": train_base_norm,
        "ext_base_norm": ext_base_norm,
        "train_rna_raw": train_rna_raw,
        "ext_rna_raw": ext_rna_raw,
        "train_rna_norm": train_rna_norm,
        "ext_rna_norm": ext_rna_norm,
    }


def compute_anchor_weights(ext_base_norm, train_base_norm, ext_rna_norm, train_rna_norm, args):
    sim = float(args.knn_baseline_weight) ** 2 * (ext_base_norm @ train_base_norm.T)
    sim += float(args.knn_rna_weight) ** 2 * (ext_rna_norm @ train_rna_norm.T)
    return topk_softmax_from_sim(sim.astype(np.float32), args.anchor_k, args.anchor_temperature)


def export_full_tcga_prediction(args, inputs: Path, state, blocks, device: torch.device) -> Path:
    out_path = inputs / "tcga_full_phosphosite_predicted.parquet"
    if out_path.exists() and not args.force:
        return out_path
    targets = state["targets"]
    weights = compute_anchor_weights(
        blocks["ext_base_norm"],
        blocks["train_base_norm"],
        blocks["ext_rna_norm"],
        blocks["train_rna_norm"],
        args,
    )
    col_embed_ext = torch.as_tensor(weights, dtype=torch.float32, device=device) @ state["col_embed_train"]
    model = state["model"]
    row_embed = state["row_embed"]
    site_prior = state["site_prior"]
    chunks = []
    with torch.no_grad():
        for start in range(0, blocks["ext_base_np"].shape[0], args.batch_size):
            end = min(blocks["ext_base_np"].shape[0], start + args.batch_size)
            pred_b, *_ = model.decode(
                row_embed,
                col_embed_ext[start:end],
                torch.as_tensor(blocks["ext_base_np"][start:end], dtype=torch.float32, device=device),
                torch.as_tensor(blocks["ext_mask"][start:end], dtype=torch.bool, device=device),
                site_prior,
            )
            med = pred_b.median(dim=1, keepdim=True).values
            chunks.append((pred_b - med).detach().cpu().numpy().astype(np.float32))
            print(f"full TCGA decode {end}/{blocks['ext_base_np'].shape[0]}", flush=True)
    pred = pd.DataFrame(np.vstack(chunks), index=blocks["baseline_full"].index, columns=targets).astype(np.float32)
    pred.index.name = "sample_id"
    pred.to_parquet(out_path)
    return out_path


def export_attributions(args, inputs: Path, state, blocks, device: torch.device) -> pd.DataFrame:
    target_to_idx = {t: i for i, t in enumerate(state["targets"])}
    sample_to_project = dict(zip(blocks["manifest"]["rna_sample_id"].astype(str), blocks["manifest"]["project_id"].astype(str)))
    train_base_norm = blocks["train_base_norm"]
    train_rna_norm = blocks["train_rna_norm"]
    train_rna_raw = blocks["train_rna_raw"]
    genes = blocks["genes"]
    status = []
    model = state["model"]
    for cancer in sorted(set(c["cancer"] for c in CASES)):
        sample_mask = np.asarray([sample_to_project.get(str(s), "") == cancer for s in blocks["baseline_full"].index], dtype=bool)
        sample_idx = np.where(sample_mask)[0]
        if len(sample_idx) == 0:
            continue
        ext_base_norm = blocks["ext_base_norm"][sample_idx]
        ext_rna_raw = blocks["ext_rna_raw"][sample_idx]
        ext_rna_norm = blocks["ext_rna_norm"][sample_idx]
        base_sim = float(args.knn_baseline_weight) ** 2 * (ext_base_norm @ train_base_norm.T)
        rna_num = ext_rna_raw @ train_rna_norm.T
        rna_norm = np.sqrt((ext_rna_raw * ext_rna_raw).sum(axis=1, keepdims=True)).clip(min=1e-6)
        sim = base_sim + float(args.knn_rna_weight) ** 2 * (rna_num / rna_norm)
        weights = topk_softmax_from_sim(sim.astype(np.float32), args.anchor_k, args.anchor_temperature)
        col_embed_base = torch.as_tensor(weights, dtype=torch.float32, device=device) @ state["col_embed_train"]
        cases = [c for c in CASES if c["cancer"] == cancer]
        base_pred_by_target = {}
        for c in cases:
            target = c["target"]
            if target not in target_to_idx:
                continue
            j = target_to_idx[target]
            base_pred_by_target[target] = decode_target(
                model,
                state["row_embed"],
                col_embed_base,
                blocks["ext_base_np"][sample_idx, j],
                blocks["ext_mask"][sample_idx, j],
                state["site_prior"],
                j,
            )
        attrib = {c["target"]: np.zeros(len(genes), dtype=np.float32) for c in cases if c["target"] in base_pred_by_target}
        for g_idx, gene in enumerate(genes):
            eg = ext_rna_raw[:, g_idx:g_idx + 1]
            tg = train_rna_norm[:, g_idx:g_idx + 1].T
            new_norm = np.sqrt(np.maximum((rna_norm.squeeze(1) ** 2 - eg.squeeze(1) ** 2), 1e-6))[:, None]
            rna_sim_occ = (rna_num - eg @ tg) / new_norm
            sim_occ = base_sim + float(args.knn_rna_weight) ** 2 * rna_sim_occ
            weights_occ = topk_softmax_from_sim(sim_occ.astype(np.float32), args.anchor_k, args.anchor_temperature)
            col_embed_occ = torch.as_tensor(weights_occ, dtype=torch.float32, device=device) @ state["col_embed_train"]
            for c in cases:
                target = c["target"]
                if target not in base_pred_by_target:
                    continue
                j = target_to_idx[target]
                pred_occ = decode_target(
                    model,
                    state["row_embed"],
                    col_embed_occ,
                    blocks["ext_base_np"][sample_idx, j],
                    blocks["ext_mask"][sample_idx, j],
                    state["site_prior"],
                    j,
                )
                attrib[target][g_idx] = float(np.mean(np.abs(pred_occ - base_pred_by_target[target])))
            if (g_idx + 1) % 200 == 0:
                print(f"attribution {cancer}: {g_idx + 1}/{len(genes)} genes", flush=True)
        for c in cases:
            target = c["target"]
            filename = inputs / f"attribution_{c['gene']}_{c['site']}_{c['cancer']}.tsv"
            if target not in target_to_idx:
                pd.DataFrame([{
                    "input_gene": "TARGET_NOT_IN_SCP682",
                    "attribution_value": np.nan,
                    "attribution_rank": np.nan,
                    "attribution_method": "target_not_in_scp682_main",
                }]).to_csv(filename, sep="\t", index=False)
                status.append({**c, "n_samples": int(len(sample_idx)), "n_genes": 0, "status": "target_not_in_scp682"})
                continue
            values = attrib[target]
            order = np.argsort(-values)
            df = pd.DataFrame({
                "input_gene": [genes[i] for i in order],
                "attribution_value": values[order],
                "attribution_rank": np.arange(1, len(order) + 1, dtype=int),
                "attribution_method": "single_gene_occlusion_on_scp682_tcga_rna_context",
            })
            df.to_csv(filename, sep="\t", index=False)
            status.append({**c, "n_samples": int(len(sample_idx)), "n_genes": int(len(genes)), "status": "ok"})
    status_df = pd.DataFrame(status)
    status_df.to_csv(inputs / "attribution_status.tsv", sep="\t", index=False)
    return status_df


def find_possible_rna_files() -> list[str]:
    roots = [
        LOCAL_ROOT / "01_data/single_cell/raw/external_bulk_phospho_validation_v1",
        LOCAL_ROOT / "01_data/single_cell/intermediate/external_bulk_phospho_validation_v1",
        LOCAL_ROOT / "remote_scripts",
    ]
    out = []
    pats = re.compile(r"(rna|RNA|seq|expression|tpm|count|counts|fpkm|rsem|transcript)")
    tags = re.compile(r"(PXD063604|pxd063604|PXD039363|pxd039363|G401)")
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and pats.search(p.name) and tags.search(str(p)):
                out.append(str(p))
    return out


def write_drug_placeholder(inputs: Path) -> pd.DataFrame:
    status_rows = []
    for dataset, filename, obs_path in [
        ("PXD063604", "pxd063604_predicted_vs_observed.tsv", PXD063604_OBS_LOCAL),
        ("PXD039363", "pxd039363_predicted_vs_observed.tsv", PXD039363_OBS_LOCAL),
    ]:
        pd.DataFrame(columns=REQUIRED_DRUG_COLUMNS).to_csv(inputs / filename, sep="\t", index=False)
        status_rows.append({
            "dataset": dataset,
            "output_file": filename,
            "status": "no_paired_rna_matrix_found",
            "reason": "本地和远端已找到处理前后 phosphoproteomics 观测表，但未找到与处理条件配对的 RNA-seq 表；SCP682 主模型不能从 phospho 或全蛋白输入生成 RNA→phospho 预测。",
            "observed_table_available": bool(obs_path.exists()),
            "observed_table_path": str(obs_path),
        })
    possible = find_possible_rna_files()
    pd.DataFrame({"candidate_file": possible}).to_csv(inputs / "drug_perturbation_rna_file_search_hits.tsv", sep="\t", index=False)
    status = pd.DataFrame(status_rows)
    status.to_csv(inputs / "drug_perturbation_input_status.tsv", sep="\t", index=False)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_REMOTE))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rna-context-genes", type=int, default=2048)
    parser.add_argument("--knn-baseline-weight", type=float, default=0.70)
    parser.add_argument("--knn-rna-weight", type=float, default=1.00)
    parser.add_argument("--anchor-k", type=int, default=8)
    parser.add_argument("--anchor-temperature", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    inputs = out / "inputs"
    logs = out / "logs"
    ensure_dir(inputs)
    ensure_dir(logs)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    status = {"device": str(device), "release": str(RELEASE)}

    drug_status = write_drug_placeholder(inputs)
    status["drug_status"] = drug_status.to_dict(orient="records")

    state = prepare_scp682_state(args, device)
    blocks = load_external_blocks(args, state)
    pd.DataFrame({"gene": blocks["genes"]}).to_csv(inputs / "tcga_rna_context_genes.tsv", sep="\t", index=False)

    attr_status = export_attributions(args, inputs, state, blocks, device)
    status["attribution_status"] = attr_status.to_dict(orient="records")

    if not args.skip_full:
        full_path = export_full_tcga_prediction(args, inputs, state, blocks, device)
        status["tcga_full_phosphosite_predicted"] = str(full_path)
        status["tcga_full_phosphosite_predicted_size"] = int(full_path.stat().st_size)

    (logs / "run_summary.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "done.txt").write_text("done\n", encoding="utf-8")


if __name__ == "__main__":
    main()
