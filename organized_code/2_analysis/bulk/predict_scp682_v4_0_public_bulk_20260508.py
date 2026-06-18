#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scp682_graph_runtime import SCP682GraphRuntime
from scp682_v4_engine import SCP682V4Engine as SCP682StateEstimator


def read_matrix(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".csv"}:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep="\t")
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    elif df.index.name is None or str(df.index[0]).isdigit():
        first = df.columns[0]
        if first.lower() in {"sample", "sample_id", "id", "barcode"}:
            df = df.set_index(first)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def read_manifest(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def transform_rna(df: pd.DataFrame, scale: str) -> tuple[pd.DataFrame, str]:
    scale = scale.lower()
    if scale == "log2tpm":
        return df.astype(np.float32), "as_provided_log2tpm"
    if scale in {"tpm", "cpm"}:
        return np.log2(df.clip(lower=0) + 1.0).astype(np.float32), "log2_x_plus_1"
    if scale in {"fpkm", "rpkm"}:
        nonneg = df.clip(lower=0)
        denom = nonneg.sum(axis=1, skipna=True).replace(0, np.nan)
        tpm = nonneg.div(denom, axis=0) * 1_000_000.0
        return np.log2(tpm + 1.0).astype(np.float32), "fpkm_to_tpm_then_log2_tpm_plus_1"
    vals = df.to_numpy(dtype=np.float32)
    finite = vals[np.isfinite(vals)]
    if finite.size and np.nanpercentile(finite, 99) > 80:
        return np.log2(df.clip(lower=0) + 1.0).astype(np.float32), "auto_log2_x_plus_1"
    return df.astype(np.float32), "auto_as_provided_assumed_log2tpm"


def main() -> int:
    ap = argparse.ArgumentParser(description="SCP682 主模型可迁移预测入口")
    ap.add_argument("--rna", required=True, help="RNA 表，行为样本，列为基因，支持 tsv/csv/parquet")
    ap.add_argument("--outdir", required=True, help="输出目录")
    ap.add_argument("--manifest", default=None, help="可选样本信息表，列含 sample_id, cptac_cancer_label, cptac_study_id")
    ap.add_argument("--rna-scale", default="auto", choices=["auto", "log2tpm", "tpm", "cpm", "fpkm", "rpkm"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--state-batch-size", type=int, default=32, help="S_phi state-estimator batch size")
    ap.add_argument("--v4-batch-size", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--graph-batch-size", type=int, default=2)
    ap.add_argument("--knn", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.08)
    ap.add_argument("--write-attention", action="store_true")
    args = ap.parse_args()

    package_dir = Path(__file__).resolve().parent
    outdir = Path(args.outdir).resolve()
    (outdir / "predictions").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)

    rna_raw = read_matrix(Path(args.rna))
    rna, transform = transform_rna(rna_raw, args.rna_scale)
    manifest = read_manifest(Path(args.manifest)) if args.manifest else None

    state_batch_size = args.state_batch_size if args.v4_batch_size is None else args.v4_batch_size
    state_estimator = SCP682StateEstimator(package_dir=package_dir, device=args.device, batch_size=state_batch_size)
    state_out = state_estimator.predict(rna, manifest)
    if "total_protein" in state_out:
        state_out["total_protein"].to_parquet(outdir / "predictions" / "scp682_total_protein.parquet")
    state_out["v4_raw"].to_parquet(outdir / "predictions" / "scp682_state_estimator_raw_before_sample_median_centering.parquet")
    state_out["v4_centered"].to_parquet(outdir / "predictions" / "scp682_state_estimator.parquet")
    state_out["sample_median_offsets"].to_csv(outdir / "tables" / "sample_median_offsets.tsv", sep="\t", index=False)
    state_out["manifest"].to_csv(outdir / "tables" / "prediction_manifest.tsv", sep="\t", index=False)

    graph = SCP682GraphRuntime(package_dir=package_dir, device=args.device, knn=args.knn, temperature=args.temperature, batch_size=args.graph_batch_size)
    graph_out = graph.predict(state_out["v4_centered"])
    graph_out["scp682"].to_parquet(outdir / "predictions" / "scp682_main_phosphosite.parquet")
    graph_out["graph_delta"].to_parquet(outdir / "predictions" / "scp682_exact_scnet_gnn_delta.parquet")
    if args.write_attention:
        graph_out["graph_attention"].to_parquet(outdir / "predictions" / "scp682_exact_scnet_attention.parquet")

    summary = {
        "model": "SCP682",
        "formula": "phosphosite_hat = S_phi + 0.3 * graph_residual_delta",
        "n_samples": int(graph_out["scp682"].shape[0]),
        "n_phosphosite_targets": int(graph_out["scp682"].shape[1]),
        "n_total_protein_targets": int(state_out["total_protein"].shape[1]) if "total_protein" in state_out else 0,
        "rna_input_transform": transform,
        "outputs": {
            "total_protein": "predictions/scp682_total_protein.parquet",
            "main_prediction": "predictions/scp682_main_phosphosite.parquet",
            "state_estimator": "predictions/scp682_state_estimator.parquet",
            "graph_delta": "predictions/scp682_exact_scnet_gnn_delta.parquet",
        },
    }
    (outdir / "logs" / "prediction_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
