#!/usr/bin/env python
"""export_scp682_site_attention.py

Load the frozen SCP682 general-graph-residual (e160) checkpoint, run a single
graph-core pass, and dump the LEARNED site-axis and sample-axis attention
weights.

This is NOT the input prior (site_weight = 0.7). It is the attention the model
learned on top of the 420,102-edge phosphosite graph (rows_encoder) and the
sample graph (cols_encoder), and is what supports an honest interpretability
panel (Fig. 2d candidate): "the model up-weights functionally coherent
site-site edges beyond the uniform input prior".

USAGE (run on the A800 box, same data + checkpoint as the e160 training run):

    python export_scp682_site_attention.py \
        --package-dir   <SCP682 paper package dir, has training_set/> \
        --prior-root    <prior root used by build_site_graph in training> \
        --checkpoint    <path to e160 best checkpoint .pt> \
        --general-baseline-path <general_baseline_internal_cptac_pdc_phosphosite.parquet> \
        --rna-path      <rna_log2_tpm_paired.parquet> \
        --sample-manifest-path <sample_manifest.tsv> \
        --output-dir    <out dir> \
        --hidden 64 --latent 32 --inter-dim 96 --embd-dim 32 --num-layers 1

!!! IMPORTANT — architecture hyperparameters MUST match the e160 run, or
    load_state_dict will fail with a shape mismatch. The training-script
    DEFAULTS (hidden 160 / latent 64 / inter 192 / embd 64 / layers 2) are NOT
    the released values. Read launch_scp682_general_graph_residual_e160.sh and
    pass the exact --hidden/--latent/--inter-dim/--embd-dim/--num-layers there.
    The defaults below are set to the values you previously verified for the
    released model (64 / 32 / 96 / 32 / 1) — double-check against the launch script.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Reuse the exact data-loading + model code from the training script so that
# feature_x, the site graph, the sample graph and the model are byte-identical
# to the e160 run. On the A800 box, point SCP682_TRAIN_DIR at the directory that
# holds train_scp682_general_graph_residual.py, e.g.
#   export SCP682_TRAIN_DIR=/data/lsy/.../03_code/training
TRAIN_SCRIPT_DIR = Path(os.environ.get(
    "SCP682_TRAIN_DIR",
    "E:/data/gongke/TCGA-TCPA/paper_materials_SCP682/03_code/training",
))
sys.path.insert(0, str(TRAIN_SCRIPT_DIR))

from train_scp682_general_graph_residual import (  # noqa: E402
    SCP682GeneralGraphResidual,
    clean_numeric,
    make_rna_context,
    build_site_graph,
    build_sample_knn_edge_index_from_features,
    l2_sample_block,
    ensure_dir,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-dir", required=True)
    ap.add_argument("--prior-root", required=True)
    ap.add_argument("--checkpoint", required=True, help="e160 best checkpoint .pt")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--general-baseline-path", required=True)
    ap.add_argument("--rna-path", required=True)
    ap.add_argument("--sample-manifest-path", required=True)
    ap.add_argument("--group-column", default="cancer_label")
    ap.add_argument("--rna-context-genes", type=int, default=2048)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--knn-baseline-weight", type=float, default=0.70)
    ap.add_argument("--knn-rna-weight", type=float, default=1.00)
    # released architecture hyperparameters — confirm against the e160 launch script
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--inter-dim", type=int, default=96)
    ap.add_argument("--embd-dim", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--top-frac", type=float, default=1.0,
                    help="fraction of edges to keep (1.0 = all 420,102); set <1 to keep only top-attention edges")
    args = ap.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"
    )
    out = Path(args.output_dir)
    ensure_dir(out / "tables")

    # ------------------------------------------------------------------ data
    # (identical order to train_scp682_general_graph_residual.main)
    pkg = Path(args.package_dir)
    train = pkg / "training_set"
    y_df = clean_numeric(pd.read_parquet(train / "observed_phosphosite.parquet"))
    general_baseline_df = clean_numeric(pd.read_parquet(args.general_baseline_path))
    rna_df = clean_numeric(pd.read_parquet(args.rna_path))
    samples = y_df.index.intersection(general_baseline_df.index)
    targets = [c for c in y_df.columns if c in general_baseline_df.columns]
    y_df = y_df.loc[samples, targets]
    baseline_df = general_baseline_df.loc[samples, targets]
    sample_context_df, _ = make_rna_context(rna_df, samples, n_genes=args.rna_context_genes)

    y = y_df.to_numpy(np.float32)
    baseline = baseline_df.to_numpy(np.float32)
    mask = np.isfinite(y) & np.isfinite(baseline)
    baseline = np.nan_to_num(baseline, nan=0.0).astype(np.float32)

    edge_index_np, site_prior_np, _ = build_site_graph(targets, Path(args.prior_root))  # SITE graph
    sample_feature_np = np.concatenate([
        args.knn_baseline_weight * l2_sample_block(np.where(mask, baseline, 0.0)),
        args.knn_rna_weight * l2_sample_block(sample_context_df.to_numpy(np.float32)),
    ], axis=1).astype(np.float32)
    col_edge_np = build_sample_knn_edge_index_from_features(sample_feature_np, k=args.knn)  # SAMPLE graph

    feature_x_np = np.where(mask.T, baseline.T, 0.0).astype(np.float32)
    mu = feature_x_np.mean(axis=1, keepdims=True)
    sd = feature_x_np.std(axis=1, keepdims=True) + 1e-5
    feature_x_np = ((feature_x_np - mu) / sd).astype(np.float32)

    feature_x = torch.as_tensor(feature_x_np, dtype=torch.float32, device=device)
    row_edge_index = torch.as_tensor(edge_index_np, dtype=torch.long, device=device)  # site
    col_edge_index = torch.as_tensor(col_edge_np, dtype=torch.long, device=device)    # sample

    run_info = {
        "package_dir": str(pkg),
        "prior_root": str(args.prior_root),
        "checkpoint": str(args.checkpoint),
        "general_baseline_path": str(args.general_baseline_path),
        "rna_path": str(args.rna_path),
        "sample_manifest_path": str(args.sample_manifest_path),
        "samples": int(len(samples)),
        "sites": int(len(targets)),
        "site_edges": int(edge_index_np.shape[1]),
        "sample_edges": int(col_edge_np.shape[1]),
        "hidden": int(args.hidden),
        "latent": int(args.latent),
        "inter_dim": int(args.inter_dim),
        "embd_dim": int(args.embd_dim),
        "num_layers": int(args.num_layers),
        "knn": int(args.knn),
        "knn_baseline_weight": float(args.knn_baseline_weight),
        "knn_rna_weight": float(args.knn_rna_weight),
    }
    print(f"samples={len(samples)} sites={len(targets)} "
          f"site_edges={edge_index_np.shape[1]} sample_edges={col_edge_np.shape[1]}")

    # ----------------------------------------------------------------- model
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

    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict):
        state = (
            ckpt.get("model_state_dict")
            or ckpt.get("model")
            or ckpt.get("state_dict")
            or ckpt
        )
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (first few: {missing[:5]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (first few: {unexpected[:5]})")
    if missing or unexpected:
        if len(missing) > 10 or len(unexpected) > 10:
            raise RuntimeError(
                "Checkpoint did not match the model architecture. "
                "Check --hidden/--latent/--inter-dim/--embd-dim/--num-layers "
                "and checkpoint state key."
            )
    model.eval()

    # ----------------------------------------------- forward + collect attention
    #
    # Important: the training DimEncoder stores attention only for reducer=True.
    # In the released graph core, rows_encoder is the site axis and reducer=False,
    # so rows_encoder.atten_weights is None after graph_core.forward(). We collect
    # both axes explicitly from the underlying attention layers instead of relying
    # on cached attributes.
    def collect_axis_attention(axis_encoder, axis_x, axis_edge_index):
        embedded = axis_encoder.encoder(axis_x.clone(), axis_edge_index)
        _, attn = axis_encoder.atten_layer(
            embedded,
            axis_edge_index,
            return_attention_weights=True,
        )
        edge_map, edge_weight = attn
        return edge_map.detach(), edge_weight.detach()

    with torch.no_grad():
        embedded0 = model.graph_core.encoder(feature_x, col_edge_index, row_edge_index)
        site_edge_map, site_edge_weight = collect_axis_attention(
            model.graph_core.rows_encoder,
            embedded0,
            row_edge_index,
        )
        sample_edge_map, sample_edge_weight = collect_axis_attention(
            model.graph_core.cols_encoder,
            embedded0.T,
            col_edge_index,
        )

    def dump_axis(edge_map, edge_weight, axis_name, idx_to_name):
        if edge_map is None or edge_weight is None:
            print(f"[warn] {axis_name}: no attention collected")
            return None
        v1 = edge_map[0].cpu().numpy()
        v2 = edge_map[1].cpu().numpy()
        w_arr = edge_weight.cpu().numpy()
        if w_arr.ndim > 1:
            w_arr = w_arr.mean(axis=1)
        w = w_arr.reshape(-1).astype(np.float32)
        df = pd.DataFrame({
            "node_i": v1,
            "node_j": v2,
            "name_i": [idx_to_name(int(i)) for i in v1],
            "name_j": [idx_to_name(int(j)) for j in v2],
            "attention": w,
        })
        if 0.0 < args.top_frac < 1.0:
            k = max(1, int(len(df) * args.top_frac))
            df = df.nlargest(k, "attention")
        path = out / "tables" / f"scp682_e160_{axis_name}_attention.tsv"
        df.to_csv(path, sep="\t", index=False, float_format="%.6g")
        print(f"wrote {path}  ({len(df)} edges; attention range "
              f"{df['attention'].min():.4g}-{df['attention'].max():.4g})")
        return df

    # site axis: rows_encoder over the phosphosite graph (this is the key one)
    site_attn_df = dump_axis(site_edge_map, site_edge_weight, "site",
                             idx_to_name=lambda i: targets[i])
    # sample axis: cols_encoder over the sample graph (secondary)
    sample_ids = list(samples)
    sample_attn_df = dump_axis(sample_edge_map, sample_edge_weight, "sample",
                               idx_to_name=lambda i: str(sample_ids[i]))

    # also persist the site->target map + input prior for the downstream
    # "learned attention vs uniform input prior" comparison
    pd.DataFrame({
        "site_index": np.arange(len(targets)),
        "target": targets,
        "gene": [t.split("|")[0] for t in targets],
        "input_site_prior": site_prior_np.astype(np.float32),
    }).to_csv(out / "tables" / "scp682_e160_site_index_map.tsv",
              sep="\t", index=False, float_format="%.6g")
    print("wrote site_index_map (target + input prior, for prior-vs-learned comparison)")

    def summarize_attention(df, prefix):
        if df is None or len(df) == 0:
            return {}
        vals = df["attention"].to_numpy(np.float64)
        return {
            f"{prefix}_n_edges": int(len(vals)),
            f"{prefix}_mean": float(np.mean(vals)),
            f"{prefix}_sd": float(np.std(vals)),
            f"{prefix}_p01": float(np.quantile(vals, 0.01)),
            f"{prefix}_p05": float(np.quantile(vals, 0.05)),
            f"{prefix}_p50": float(np.quantile(vals, 0.50)),
            f"{prefix}_p95": float(np.quantile(vals, 0.95)),
            f"{prefix}_p99": float(np.quantile(vals, 0.99)),
            f"{prefix}_max": float(np.max(vals)),
        }

    run_info.update(summarize_attention(site_attn_df, "site_attention"))
    run_info.update(summarize_attention(sample_attn_df, "sample_attention"))
    pd.DataFrame([run_info]).to_csv(
        out / "tables" / "scp682_e160_attention_export_summary.tsv",
        sep="\t",
        index=False,
    )
    print("wrote attention_export_summary")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
