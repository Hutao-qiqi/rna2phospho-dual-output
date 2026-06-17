#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn


ROOT = Path(os.environ.get("SCP682_PROJECT_ROOT", "/data/lsy/Infinite_Stream"))
SCP682_FINAL_RELEASE = ROOT / "SCP682-22/frozen_release/SCP682_final_20260518_scp682_22"
TCPA_HEAD = ROOT / "02_results/model_release/20260428_current_phosphosite_model_reproducibility_v1/model_artifacts/active_tcpa_rppa_head"
OUTPUT_MANIFEST_DIR = ROOT / "02_results/model_validation/20260429_rna2phospho_best_deployable_model_v2_with_total/manifests"
TOTAL_TRAIN_SCRIPT = ROOT / "03_code/model_validation/train_cptac_total_proteome_film_vae_z_direct_residual_20260428.py"
INPUT_GUARD_SCRIPT = ROOT / "03_code/model_validation/scp682_input_guard_and_sample_centering.py"
MODEL_ID = "SCP682"
MODEL_FULL_NAME = "Sample-centered Cross-platform Proteome and Phosphoproteome Predictor from Bulk RNA"


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_matrix(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep="\t")
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    elif df.columns.size and not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        first = str(df.columns[0])
        if first.lower() in {"sample", "sampleid", "sample_id", "id", "barcode"}:
            df = df.set_index(df.columns[0])
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def maybe_transform(x: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, str]:
    vals = x.to_numpy(dtype=np.float32)
    finite = vals[np.isfinite(vals)]
    if mode == "none":
        return x, "none"
    if mode == "log2":
        return np.log2(x.clip(lower=0) + 1.0), "log2(x+1)"
    if finite.size and np.nanpercentile(finite, 99) > 80:
        return np.log2(x.clip(lower=0) + 1.0), "auto_log2(x+1)"
    return x, "auto_none"


def write_input_qc(report: dict, out_dir: Path) -> None:
    (out_dir / "input_qc_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def enforce_log2tpm_input(rna: pd.DataFrame, out_dir: Path, transform_mode: str) -> dict:
    guard = import_module("scp682_v4_input_guard", INPUT_GUARD_SCRIPT)
    if transform_mode != "none":
        report = {
            "model_id": MODEL_ID,
            "required_input_unit": "log2(TPM+1)",
            "passed": False,
            "failure_reasons": ["input_transform_must_be_none_for_scp682"],
            "requested_transform": transform_mode,
        }
        write_input_qc(report, out_dir)
        raise SystemExit("SCP682 requires RNA input that is already log2(TPM+1); do not submit raw TPM, FPKM, RSEM, counts, or request automatic log transform.")
    reference_features = guard.load_reference_features(guard.DEFAULT_FEATURE_REFERENCE)
    passed, reasons, input_report = guard.assess_log2tpm(rna, reference_features, min_gene_overlap=0.60)
    warnings = []
    if not passed:
        tolerable = {"finite_fraction_below_0.80"}
        reason_set = set(reasons)
        overlap_ok = input_report.get("input_reference_overlap_fraction", 0.0) >= 0.60
        finite_ok = input_report.get("finite_fraction", 0.0) >= 0.70
        if reason_set.issubset(tolerable) and overlap_ok and finite_ok:
            passed = True
            warnings = list(reasons)
            input_report["passed"] = True
            input_report["failure_reasons"] = []
            input_report["warnings"] = warnings
    report = {
        "model_id": MODEL_ID,
        "model_full_name": MODEL_FULL_NAME,
        "required_input_unit": "log2(TPM+1)",
        "rna_input_action": "check_only_no_sample_median_centering",
        "phosphosite_output_action": "sample_median_centering",
        "input_qc": input_report,
        "warnings": warnings,
    }
    write_input_qc(report, out_dir)
    if not passed:
        raise SystemExit(
            "Input RNA did not pass SCP682 log2TPM guard: "
            + ", ".join(reasons)
            + ". Supply log2(TPM+1) RNA, not raw TPM, FPKM, RSEM, counts, or log2FPKM."
        )
    return report


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df.reset_index()), path)


class TCPACVAE(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, latent_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        enc_layers: list[nn.Module] = []
        in_dim = x_dim + y_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)
        self.mu = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)

        dec_layers: list[nn.Module] = []
        in_dim = x_dim + latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if dropout > 0:
                dec_layers.append(nn.Dropout(dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, y_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def decode(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([x, z], dim=1))


def infer_tcpa_cvae_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int, list[int]]:
    dec0 = state_dict["decoder.0.weight"].shape
    y_dim = int(state_dict["decoder.9.weight"].shape[0])
    latent_dim = int(state_dict["mu.weight"].shape[0])
    x_dim = int(dec0[1] - latent_dim)
    hidden_dims = [
        int(state_dict["encoder.0.weight"].shape[0]),
        int(state_dict["encoder.3.weight"].shape[0]),
        int(state_dict["encoder.6.weight"].shape[0]),
    ]
    if state_dict["encoder.0.weight"].shape[1] != x_dim + y_dim:
        raise ValueError("TCPA CVAE metadata and weight dimensions are inconsistent")
    return x_dim, y_dim, latent_dim, hidden_dims


def sample_median_center(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = pred.median(axis=1, skipna=True)
    centered = pred.sub(med, axis=0)
    med_table = pd.DataFrame({
        "sample_id": pred.index.astype(str),
        "prediction_sample_median": med.to_numpy(dtype=float),
    })
    centered.index.name = pred.index.name or "sample_id"
    return centered.astype(np.float32), med_table


def align_cptac_x(x: pd.DataFrame, ckpt: dict) -> np.ndarray:
    features = ckpt["feature_names"]
    mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    std = np.asarray(ckpt["x_std"], dtype=np.float32)
    std = np.where((std > 1e-6) & np.isfinite(std), std, 1.0)
    mat = x.reindex(columns=features).to_numpy(dtype=np.float32)
    mat = np.where(np.isfinite(mat), mat, mean[None, :])
    return ((mat - mean[None, :]) / std[None, :]).astype(np.float32)


def build_direct_total(xz: np.ndarray, features: list[str], total_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    idx = {g: i for i, g in enumerate(features)}
    direct = np.zeros((xz.shape[0], len(total_names)), dtype=np.float32)
    mask = np.zeros_like(direct)
    for j, gene in enumerate(total_names):
        if gene in idx:
            direct[:, j] = xz[:, idx[gene]]
            mask[:, j] = 1.0
    return direct, mask


def build_phospho_maps(xz: np.ndarray, features: list[str], phospho_names: list[str], total_names: list[str]):
    feature_idx = {g: i for i, g in enumerate(features)}
    total_idx = {g: i for i, g in enumerate(total_names)}
    direct = np.zeros((xz.shape[0], len(phospho_names)), dtype=np.float32)
    direct_mask = np.zeros_like(direct)
    parent_idx = np.zeros(len(phospho_names), dtype=np.int64)
    parent_mask = np.zeros(len(phospho_names), dtype=np.float32)
    for j, target in enumerate(phospho_names):
        gene = str(target).split("|", 1)[0]
        if gene in feature_idx:
            direct[:, j] = xz[:, feature_idx[gene]]
            direct_mask[:, j] = 1.0
        if gene in total_idx:
            parent_idx[j] = total_idx[gene]
            parent_mask[j] = 1.0
    return direct, direct_mask, parent_idx, parent_mask


def predict_cptac(x_ext: pd.DataFrame, out_dir: Path, cancer_label: str, study_id: str, batch_size: int, device: torch.device) -> dict:
    _ = (x_ext, out_dir, cancer_label, study_id, batch_size, device)
    raise RuntimeError(
        "Uploaded CPTAC/PDC inference is not exposed in this web runner for SCP682-22 yet. "
        f"Use the frozen transferable package at {SCP682_FINAL_RELEASE}. "
        "The old nas_12438 joint model is not part of SCP682."
    )

def predict_tcpa(x_ext: pd.DataFrame, out_dir: Path, tcpa_project: str, batch_size: int, device: torch.device) -> dict:
    _ = tcpa_project
    meta = joblib.load(TCPA_HEAD / "cvae_meta.pkl")
    features = [str(x) for x in meta["features"]]
    proteins = [str(x) for x in meta["proteins"]]
    state = torch.load(TCPA_HEAD / "cvae.pt", map_location="cpu")
    x_dim, y_dim, latent_dim, hidden_dims = infer_tcpa_cvae_dims(state)
    if x_dim != len(features) or y_dim != len(proteins):
        raise ValueError("TCPA CVAE metadata and weights do not match")
    model = TCPACVAE(x_dim=x_dim, y_dim=y_dim, latent_dim=latent_dim, hidden_dims=hidden_dims, dropout=0.1).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    x_df = x_ext.reindex(columns=features).fillna(0.0)
    x_scaled = meta["scaler"].transform(x_df.to_numpy(dtype=np.float32))
    x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=8.0, neginf=-8.0)
    x_scaled = np.clip(x_scaled, -8.0, 8.0).astype(np.float32)

    batches = []
    with torch.no_grad():
        for start in range(0, x_scaled.shape[0], batch_size):
            xb = torch.tensor(x_scaled[start : start + batch_size], dtype=torch.float32, device=device)
            z = torch.zeros((xb.shape[0], latent_dim), dtype=torch.float32, device=device)
            batches.append(model.decode(xb, z).detach().cpu().numpy().astype(np.float32))

    antibody_names = [p[1:] if len(p) > 1 and p.startswith("X") and p[1].isdigit() else p for p in proteins]
    pred = pd.DataFrame(np.vstack(batches), index=x_ext.index, columns=antibody_names)
    pred.index.name = "sample_id"
    total_cols = pd.read_csv(OUTPUT_MANIFEST_DIR / "tcpa_total_antibody_output_order.tsv", sep="\t")["antibody"].astype(str).tolist()
    phospho_cols = pd.read_csv(OUTPUT_MANIFEST_DIR / "tcpa_phospho_antibody_output_order.tsv", sep="\t")["antibody"].astype(str).tolist()
    missing_total = [c for c in total_cols if c not in pred.columns]
    missing_phospho = [c for c in phospho_cols if c not in pred.columns]
    if missing_total or missing_phospho:
        raise ValueError(f"TCPA antibody manifest does not match CVAE output: missing_total={missing_total[:5]}, missing_phospho={missing_phospho[:5]}")
    write_parquet(pred[total_cols], out_dir / "predicted_tcpa_total_antibody.parquet")
    write_parquet(pred[phospho_cols], out_dir / "predicted_tcpa_phospho_antibody.parquet")
    return {
        "tcpa_project": tcpa_project,
        "model_component": "active_tcpa_rppa_head/cvae_z0",
        "postprocess": "deterministic_z0_decode",
        "total_antibody_targets": int(len(total_cols)),
        "phospho_antibody_targets": int(len(phospho_cols)),
        "files": {
            "tcpa_total": "predicted_tcpa_total_antibody.parquet",
            "tcpa_phospho": "predicted_tcpa_phospho_antibody.parquet",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-rna", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cptac-cancer-label", required=True)
    parser.add_argument("--cptac-study-id", required=True)
    parser.add_argument("--tcpa-project", required=True)
    parser.add_argument("--transform", choices=["auto", "none", "log2"], default="none")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    rna = read_matrix(Path(args.input_rna))
    input_qc = enforce_log2tpm_input(rna, out_dir, args.transform)
    transform = "none_checked_log2tpm"
    total_mod = import_module("total_module", TOTAL_TRAIN_SCRIPT)
    z = total_mod.encode_vae_z(rna, device)
    x_ext = pd.concat([rna, z], axis=1)

    cptac_summary = predict_cptac(x_ext, out_dir, args.cptac_cancer_label, args.cptac_study_id, args.batch_size, device)
    tcpa_summary = predict_tcpa(x_ext, out_dir, args.tcpa_project, args.batch_size, device)
    summary = {
        "model_id": MODEL_ID,
        "model_full_name": MODEL_FULL_NAME,
        "input": str(Path(args.input_rna)),
        "n_samples": int(rna.shape[0]),
        "n_input_genes": int(rna.shape[1]),
        "input_transform": transform,
        "input_qc": input_qc["input_qc"],
        "device": str(device),
        "cptac_pdc": cptac_summary,
        "tcpa": tcpa_summary,
    }
    (out_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
