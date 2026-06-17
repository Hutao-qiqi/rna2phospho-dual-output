"""
模型: SCP682
作用: 从原始实验脚本提取的最小可复现代码片段，文件名 predict_scp682_general_graph_external.py
输入: ./data_root 下的训练数据、图先验和冻结基线预测
输出: ./paper_materials_SCP682 或结果目录中的模型、表格、报告
依赖: Python、pandas、numpy、torch、torch_geometric
原始路径: remote_scripts/predict_scp682_general_graph_external.py
原始版本: 20260523 结果目录对应脚本
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path("./data_root")
TRAIN_SCRIPT = ROOT / "remote_scripts/train_scp682_general_graph_residual.py"
SCP6828_EXTERNAL = ROOT / "SCP682-8/scripts/run_scp682_8_external_validation.py"
DEFAULT_PACKAGE = ROOT / "SCP682-22/frozen_release/SCP682_22_paper_package_20260520"
DEFAULT_BASELINE_DIR = ROOT / "SCP682-main/inputs/general_baseline_predictions"
DEFAULT_INTERNAL = ROOT / "SCP682-main/results/20260523_general_graph_residual_e160"
DEFAULT_TRAIN_RNA = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2/rna_log2_tpm_paired.parquet"
DEFAULT_OUT = ROOT / "SCP682-main/results/20260523_general_graph_external_fixed_anchor"


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def sample_median_center(df: pd.DataFrame) -> pd.DataFrame:
    values = df.apply(pd.to_numeric, errors="coerce")
    return values.sub(values.median(axis=1, skipna=True), axis=0).astype(np.float32)


def read_external_general_baseline(dataset: str, baseline_dir: Path) -> pd.DataFrame:
    path = baseline_dir / f"general_baseline_{dataset}_phosphosite.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return clean_numeric(pd.read_parquet(path))


def topk_softmax_attention(query: np.ndarray, key: np.ndarray, k: int, temperature: float) -> np.ndarray:
    sim = query @ key.T
    k = max(1, min(k, sim.shape[1]))
    top = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    out = np.zeros_like(sim, dtype=np.float32)
    rows = np.arange(sim.shape[0])[:, None]
    vals = sim[rows, top] / max(float(temperature), 1e-6)
    vals = vals - vals.max(axis=1, keepdims=True)
    weights = np.exp(vals).astype(np.float32)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    out[rows, top] = weights
    return out


def summarize_anchor(weights: np.ndarray, ext_index: pd.Index, train_index: pd.Index, out_file: Path) -> None:
    top = np.argsort(-weights, axis=1)[:, :5]
    rows = []
    for i, sample in enumerate(ext_index.astype(str)):
        for rank, j in enumerate(top[i], 1):
            rows.append({
                "external_sample": sample,
                "rank": rank,
                "train_sample": str(train_index[j]),
                "weight": float(weights[i, j]),
            })
    pd.DataFrame(rows).to_csv(out_file, sep="\t", index=False)


def build_model(train_mod, ckpt: dict, device: torch.device):
    args = ckpt["args"]
    model = train_mod.SCP682GeneralGraphResidual(
        n_sites=len(ckpt["targets"]),
        n_samples=len(ckpt["samples"]),
        hidden=int(args.get("hidden", 64)),
        latent=int(args.get("latent", 32)),
        inter_dim=int(args.get("inter_dim", 96)),
        embd_dim=int(args.get("embd_dim", 32)),
        num_layers=int(args.get("num_layers", 1)),
        sample_context_dim=int(args.get("rna_context_genes", 2048)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def run_one_dataset(args, train_mod, ext8, model, ckpt, device, dataset: str, out: Path):
    train_dir = Path(args.package_dir) / "training_set"
    y_df = clean_numeric(pd.read_parquet(train_dir / "observed_phosphosite.parquet"))
    train_general_baseline = clean_numeric(pd.read_parquet(args.internal_general_baseline_path))
    train_rna = clean_numeric(pd.read_parquet(args.train_rna_path))
    datasets = ext8.load_datasets([dataset])
    spec = datasets[dataset]
    ext_general_baseline = read_external_general_baseline(dataset, Path(args.general_baseline_dir))

    samples = pd.Index([str(x) for x in ckpt["samples"]])
    targets = [str(x) for x in ckpt["targets"]]
    train_base_df = train_general_baseline.loc[samples, targets]
    y_df = y_df.loc[samples, targets]
    train_mask = np.isfinite(y_df.to_numpy(np.float32)) & np.isfinite(train_base_df.to_numpy(np.float32))
    train_base = np.nan_to_num(train_base_df.to_numpy(np.float32), nan=0.0).astype(np.float32)

    ext_targets = [t for t in targets if t in ext_general_baseline.columns]
    ext_base_df = ext_general_baseline.loc[:, ext_targets]
    ext_base_full = pd.DataFrame(np.nan, index=ext_base_df.index, columns=targets, dtype=np.float32)
    ext_base_full.loc[:, ext_targets] = ext_base_df.to_numpy(np.float32)
    ext_mask = np.isfinite(ext_base_full.to_numpy(np.float32))
    ext_base = np.nan_to_num(ext_base_full.to_numpy(np.float32), nan=0.0).astype(np.float32)

    ctx = train_mod.make_rna_context(
        train_rna,
        samples,
        external_rna=spec["rna"],
        external_samples=ext_base_full.index,
        n_genes=int(args.rna_context_genes),
    )[0]
    train_ctx = ctx.iloc[:len(samples)].to_numpy(np.float32)
    ext_ctx = ctx.iloc[len(samples):].to_numpy(np.float32)

    train_feature = np.concatenate([
        args.knn_baseline_weight * train_mod.l2_sample_block(np.where(train_mask, train_base, 0.0)),
        args.knn_rna_weight * train_mod.l2_sample_block(train_ctx),
    ], axis=1).astype(np.float32)
    ext_feature = np.concatenate([
        args.knn_baseline_weight * train_mod.l2_sample_block(np.where(ext_mask, ext_base, 0.0)),
        args.knn_rna_weight * train_mod.l2_sample_block(ext_ctx),
    ], axis=1).astype(np.float32)
    anchor_weights = topk_softmax_attention(ext_feature, train_feature, k=args.anchor_k, temperature=args.anchor_temperature)

    feature_x = np.where(train_mask.T, train_base.T, 0.0).astype(np.float32)
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
        weight_t = torch.as_tensor(anchor_weights, dtype=torch.float32, device=device)
        col_embed_ext = weight_t @ col_embed_train
        pred_parts = []
        for start in range(0, ext_base.shape[0], args.batch_size):
            end = min(ext_base.shape[0], start + args.batch_size)
            pred_b, *_ = model.decode(
                row_embed,
                col_embed_ext[start:end],
                torch.as_tensor(ext_base[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(ext_mask[start:end], dtype=torch.bool, device=device),
                site_prior,
                sample_context=torch.as_tensor(ext_ctx[start:end], dtype=torch.float32, device=device),
            )
            pred_parts.append(pred_b.detach().cpu())
    pred = torch.cat(pred_parts, dim=0).numpy().astype(np.float32)
    pred_df = pd.DataFrame(pred, index=ext_base_full.index, columns=targets)
    pred_df = pred_df.loc[:, ext_targets]

    raw_path = out / "predictions" / f"{dataset}_scp682_general_graph_residual_raw.parquet"
    centered_path = out / "predictions" / f"{dataset}_scp682_general_graph_residual_sample_centered.parquet"
    pred_df.to_parquet(raw_path)
    sample_median_center(pred_df).to_parquet(centered_path)
    summarize_anchor(anchor_weights, ext_base_full.index, samples, out / "tables" / f"{dataset}_anchor_top5.tsv")

    baseline_raw = ext_base_df
    baseline_centered = sample_median_center(ext_base_df)
    pred_centered = pd.read_parquet(centered_path)
    summaries = [
        ext8.evaluate(dataset, "phosphosite", "scp682_general_baseline_raw", baseline_raw, spec["true_phospho"], args.min_site_n, args.min_sample_n, out),
        ext8.evaluate(dataset, "phosphosite", "scp682_general_baseline_sample_centered", baseline_centered, spec["true_phospho"], args.min_site_n, args.min_sample_n, out),
        ext8.evaluate(dataset, "phosphosite", "scp682_general_graph_residual_raw", pred_df, spec["true_phospho"], args.min_site_n, args.min_sample_n, out),
        ext8.evaluate(dataset, "phosphosite", "scp682_general_graph_residual_sample_centered", pred_centered, spec["true_phospho"], args.min_site_n, args.min_sample_n, out),
    ]
    pd.DataFrame(summaries).to_csv(out / "tables" / f"{dataset}_fixed_anchor_summary.tsv", sep="\t", index=False)
    return summaries


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_INTERNAL / "models/scp682_general_graph_residual_best.pt"))
    ap.add_argument("--package-dir", default=str(DEFAULT_PACKAGE))
    ap.add_argument("--general-baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    ap.add_argument("--internal-general-baseline-path", default=str(DEFAULT_BASELINE_DIR / "general_baseline_internal_cptac_pdc_phosphosite.parquet"))
    ap.add_argument("--train-rna-path", default=str(DEFAULT_TRAIN_RNA))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--datasets", nargs="+", default=["fu_icca", "tu_sclc", "chcc_hbv_fpkm", "chcc_hbv_rsem"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--rna-context-genes", type=int, default=2048)
    ap.add_argument("--knn-baseline-weight", type=float, default=0.70)
    ap.add_argument("--knn-rna-weight", type=float, default=1.00)
    ap.add_argument("--anchor-k", type=int, default=25)
    ap.add_argument("--anchor-temperature", type=float, default=0.08)
    ap.add_argument("--min-site-n", type=int, default=8)
    ap.add_argument("--min-sample-n", type=int, default=50)
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    for sub in ("predictions", "tables", "logs", "reports"):
        ensure_dir(out / sub)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    train_mod = import_module("scp682_general_graph_train", TRAIN_SCRIPT)
    ext8 = import_module("scp6828_external", SCP6828_EXTERNAL)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(train_mod, ckpt, device)
    config = vars(args).copy()
    config["device_resolved"] = str(device)
    config["formula"] = "phosphosite_hat = general_baseline_hat + graph_residual_theta(general_baseline_hat, RNA, phosphosite_graph, sample_anchor_attention)"
    (out / "logs/fixed_anchor_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    all_rows = []
    for dataset in args.datasets:
        all_rows.extend(run_one_dataset(args, train_mod, ext8, model, ckpt, device, dataset, out))
        pd.DataFrame(all_rows).to_csv(out / "tables/scp682_general_graph_external_summary.tsv", sep="\t", index=False)
    (out / "done.txt").write_text("done\n", encoding="utf-8")
    print(pd.DataFrame(all_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
