#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    package_dir = Path(__file__).resolve().parents[1]
    train_script = package_dir / "scripts" / "train_scp682_main_v4_exact_scnet_gnn.py"
    checkpoint_path = package_dir / "models" / "scp682_main_v4_exact_scnet_gnn_best.pt"
    output_path = package_dir / "models" / "scp682_graph_runtime_state.pt"
    mod = import_module(train_script, "scp682_exact_scnet_train_module_for_export")
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    targets = [str(x) for x in ck["targets"]]
    samples = [str(x) for x in ck["samples"]]
    y_df = mod.clean_numeric(pd.read_parquet(package_dir / "training_set" / "observed_phosphosite.parquet")).loc[samples, targets]
    v4_df = mod.clean_numeric(pd.read_parquet(package_dir / "training_set" / "v4_phosphosite_baseline.parquet")).loc[samples, targets]
    y = y_df.to_numpy(np.float32)
    baseline = v4_df.to_numpy(np.float32)
    mask = np.isfinite(y) & np.isfinite(baseline)
    baseline_filled = np.nan_to_num(baseline, nan=0.0).astype(np.float32)
    feature_x = np.where(mask.T, baseline_filled.T, 0.0).astype(np.float32)
    mu = feature_x.mean(axis=1, keepdims=True)
    sd = feature_x.std(axis=1, keepdims=True) + 1e-5
    feature_x = ((feature_x - mu) / sd).astype(np.float32)

    args = ck["args"]
    model = mod.SCP682ExactScNETResidual(
        n_sites=len(targets),
        n_samples=len(samples),
        shrinkage=float(args.get("shrinkage", 0.3)),
        hidden=int(args.get("hidden", 64)),
        latent=int(args.get("latent", 32)),
        inter_dim=int(args.get("inter_dim", 96)),
        embd_dim=int(args.get("embd_dim", 32)),
        num_layers=int(args.get("num_layers", 1)),
    )
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        row_embed, col_embed, _ = model.graph_core(
            torch.as_tensor(feature_x, dtype=torch.float32),
            ck["sample_edge_index"].long(),
            torch.as_tensor(ck["site_edge_index"], dtype=torch.long),
            collect_attention=False,
        )
    decoder_keys = [
        "graph_scale",
        "residual_scale",
        "site_prior_proj.",
        "baseline_proj.",
        "site_proj.",
        "sample_proj.",
        "prior_attention.",
        "graph_decoder.",
        "residual.",
    ]
    decoder_state = {
        k: v.detach().cpu()
        for k, v in ck["model_state_dict"].items()
        if any(k == key or k.startswith(key) for key in decoder_keys)
    }
    runtime = {
        "targets": targets,
        "samples": samples,
        "row_embed": row_embed.detach().cpu().to(torch.float16),
        "train_col_embed": col_embed.detach().cpu().to(torch.float16),
        "site_prior": torch.as_tensor(ck["site_prior"], dtype=torch.float32),
        "train_v4_baseline": torch.as_tensor(baseline_filled, dtype=torch.float16),
        "decoder_state_dict": decoder_state,
        "decoder_config": {
            "hidden": int(args.get("hidden", 64)),
            "latent": int(args.get("latent", 32)),
            "embd_dim": int(args.get("embd_dim", 32)),
            "shrinkage": float(args.get("shrinkage", 0.3)),
        },
        "meta": ck.get("meta", {}),
        "best": ck.get("best", {}),
        "runtime_note": "New samples are projected into the frozen sample-embedding space by nearest-neighbour induction over the V4 baseline prediction space.",
    }
    torch.save(runtime, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
