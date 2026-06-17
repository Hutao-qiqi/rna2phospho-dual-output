import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_numeric(df):
    df = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def load_main_module(script_path):
    spec = importlib.util.spec_from_file_location("scp682_main_exact_scnet", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_spearman(main_dir, targets):
    path = Path(main_dir) / "performance" / "per_site_spearman.tsv"
    out = np.ones(len(targets), dtype=np.float32) * 0.05
    if not path.exists():
        return out
    df = pd.read_csv(path, sep="\t")
    value = {}
    for _, row in df.iterrows():
        try:
            value[str(row["target"])] = float(row["spearman"])
        except Exception:
            continue
    for i, target in enumerate(targets):
        out[i] = max(float(value.get(str(target), 0.05)), 0.0)
    out = np.nan_to_num(out, nan=0.05, posinf=0.05, neginf=0.0)
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scp682-main-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    main_dir = Path(args.scp682_main_dir)
    out_dir = ensure_dir(args.output_dir)
    for sub in ("tables", "logs", "reports"):
        ensure_dir(out_dir / sub)

    runtime_path = main_dir / "models" / "scp682_graph_runtime_state.pt"
    script_path = main_dir / "scripts" / "train_scp682_main_v4_exact_scnet_gnn.py"
    checkpoint_path = main_dir / "models" / "scp682_main_v4_exact_scnet_gnn_best.pt"
    observed_path = main_dir / "training_set" / "observed_phosphosite.parquet"
    baseline_path = main_dir / "training_set" / "v4_phosphosite_baseline.parquet"
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    if runtime_path.exists():
        runtime = torch.load(runtime_path, map_location="cpu", weights_only=False)
        use_targets = [str(x) for x in runtime["targets"]]
        use_samples = [str(x) for x in runtime.get("samples", [])]
        site_embedding = runtime["row_embed"].detach().cpu().numpy().astype(np.float32)
        site_prior = np.asarray(runtime.get("site_prior", np.ones(len(use_targets), dtype=np.float32)), dtype=np.float32)
        source_label = "SCP682_PORTABLE_runtime_exact_ScNET_GNN"
    else:
        missing = [str(p) for p in (script_path, checkpoint_path, observed_path, baseline_path) if not p.exists()]
        if missing:
            raise FileNotFoundError("missing required SCP682 portable files: " + "; ".join(missing))

        module = load_main_module(script_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_args = ckpt.get("args", {})
        targets = [str(x) for x in ckpt["targets"]]
        samples = [str(x) for x in ckpt["samples"]]

        y_df = clean_numeric(pd.read_parquet(observed_path))
        baseline_df = clean_numeric(pd.read_parquet(baseline_path))
        use_samples = [s for s in samples if s in y_df.index and s in baseline_df.index]
        use_targets = [t for t in targets if t in y_df.columns and t in baseline_df.columns]
        if len(use_samples) != len(samples) or len(use_targets) != len(targets):
            raise RuntimeError(
                f"training_set mismatch: samples {len(use_samples)}/{len(samples)}, targets {len(use_targets)}/{len(targets)}"
            )

        y = y_df.loc[use_samples, use_targets].to_numpy(np.float32)
        baseline = baseline_df.loc[use_samples, use_targets].to_numpy(np.float32)
        mask = np.isfinite(y) & np.isfinite(baseline)
        baseline = np.nan_to_num(baseline, nan=0.0).astype(np.float32)
        feature_x = np.where(mask.T, baseline.T, 0.0).astype(np.float32)
        mu = feature_x.mean(axis=1, keepdims=True)
        sd = feature_x.std(axis=1, keepdims=True) + 1e-5
        feature_x = ((feature_x - mu) / sd).astype(np.float32)

        model = module.SCP682ExactScNETResidual(
            n_sites=len(use_targets),
            n_samples=len(use_samples),
            shrinkage=float(model_args.get("shrinkage", ckpt.get("meta", {}).get("shrinkage", 0.3))),
            hidden=int(model_args.get("hidden", 160)),
            latent=int(model_args.get("latent", 64)),
            inter_dim=int(model_args.get("inter_dim", 192)),
            embd_dim=int(model_args.get("embd_dim", 64)),
            num_layers=int(model_args.get("num_layers", 2)),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()

        with torch.no_grad():
            feature_t = torch.as_tensor(feature_x, dtype=torch.float32, device=device)
            site_edge = torch.as_tensor(ckpt["site_edge_index"], dtype=torch.long, device=device)
            sample_edge = torch.as_tensor(ckpt["sample_edge_index"], dtype=torch.long, device=device)
            row_embed, col_embed, _ = model.graph_core(feature_t, sample_edge, site_edge, collect_attention=False)
            site_embedding = row_embed.detach().cpu().numpy().astype(np.float32)

        site_prior = np.asarray(ckpt.get("site_prior", np.ones(len(use_targets), dtype=np.float32)), dtype=np.float32)
        source_label = "SCP682_MAIN_full_checkpoint_exact_ScNET_GNN"

    site_spearman = read_spearman(main_dir, use_targets)
    source_model = np.asarray([source_label], dtype=object)
    np.savez_compressed(
        out_dir / "tables" / "scp682_main_sc_transfer_arrays.npz",
        site_embedding=site_embedding,
        bulk_targets=np.asarray(use_targets, dtype=object),
        site_spearman=site_spearman,
        site_prior=site_prior.astype(np.float32),
        n_model_files=np.asarray([1], dtype=np.int64),
        source_model=source_model,
    )

    pd.DataFrame(
        {
            "bulk_target": use_targets,
            "site_spearman": site_spearman,
            "site_prior": site_prior.astype(np.float32),
            "embedding_norm": np.linalg.norm(site_embedding, axis=1),
        }
    ).to_csv(out_dir / "tables" / "scp682_main_site_embedding_manifest.tsv", sep="\t", index=False)
    summary = {
        "source_main_dir": str(main_dir),
        "checkpoint": str(checkpoint_path if checkpoint_path.exists() else runtime_path),
        "n_samples": int(len(use_samples)),
        "n_sites": int(len(use_targets)),
        "embedding_dim": int(site_embedding.shape[1]),
        "device": str(device),
    }
    (out_dir / "reports" / "scp682_main_sc_transfer_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "logs" / "done.txt").write_text("done\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
