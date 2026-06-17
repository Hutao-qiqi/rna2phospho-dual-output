#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import os
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from scipy.stats import false_discovery_control
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from statsmodels.duration.hazard_regression import PHReg


REMOTE_ROOT = Path("D:/data/lsy")
PROJECT_ROOT = Path("E:/data/gongke/TCGA-TCPA")

CBIOPORTAL_BASE = "https://www.cbioportal.org/api"
STUDY_BY_PROJECT = {
    "TCGA-HNSC": "hnsc_tcga_pan_can_atlas_2018",
    "TCGA-KIRC": "kirc_tcga_pan_can_atlas_2018",
    "TCGA-KIRP": "kirp_tcga_pan_can_atlas_2018",
    "TCGA-LUAD": "luad_tcga_pan_can_atlas_2018",
    "TCGA-LUSC": "lusc_tcga_pan_can_atlas_2018",
    "TCGA-PAAD": "paad_tcga_pan_can_atlas_2018",
    "TCGA-STAD": "stad_tcga_pan_can_atlas_2018",
    "TCGA-UCEC": "ucec_tcga_pan_can_atlas_2018",
}

CPTAC_TO_TCGA_PROJECT = {
    "HNSCC": "TCGA-HNSC",
    "CCRCC": "TCGA-KIRC",
    "NON_CCRCC": "TCGA-KIRP",
    "LUAD": "TCGA-LUAD",
    "LUAD_CONFIRM": "TCGA-LUAD",
    "LSCC": "TCGA-LUSC",
    "PDA": "TCGA-PAAD",
    "STAD": "TCGA-STAD",
    "UCEC": "TCGA-UCEC",
    "UCEC_CONFIRM": "TCGA-UCEC",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(REMOTE_ROOT))
    p.add_argument("--out-dir", default="")
    p.add_argument("--strict-candidates", default="")
    p.add_argument("--tcpa-manifest", default="")
    p.add_argument("--bridge-root", default="")
    p.add_argument("--scp68222-train-script", default="")
    p.add_argument("--scp68222-model-dir", default="")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--bank-k", type=int, default=8)
    p.add_argument("--bank-chunk", type=int, default=512)
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--min-tcga-n", type=int, default=80)
    p.add_argument("--min-tcga-events", type=int, default=20)
    p.add_argument("--tcga-p", type=float, default=0.05)
    p.add_argument("--skip-prediction-if-present", action="store_true")
    p.add_argument("--prediction-mode", choices=["scp68222", "v4"], default="scp68222")
    return p.parse_args()


def ensure_dirs(out: Path) -> None:
    for sub in ["predictions", "tables", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def download_file(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


def load_probemap(path: Path) -> dict[str, str]:
    probe = pd.read_csv(path, sep="\t")
    id_col = "id"
    gene_col = "gene" if "gene" in probe.columns else "gene_symbol"
    return dict(zip(probe[id_col].astype(str).str.split(".").str[0], probe[gene_col].astype(str)))


def build_tcga_rna_from_xena(manifest: pd.DataFrame, gene_order: list[str], data_dir: Path, out: Path) -> pd.DataFrame:
    out_path = data_dir / "tcga_supported_xena_log2_tpm_for_scp682.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)

    xena_gz = data_dir / "tcga_RSEM_gene_tpm.gz"
    probemap = data_dir / "gencode.v23.annotation.gene.probemap"
    download_file("https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/tcga_RSEM_gene_tpm.gz", xena_gz)
    download_file("https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/probeMap/gencode.v23.annotation.gene.probemap", probemap)

    id_to_symbol = load_probemap(probemap)
    sample_map: dict[str, str] = {}
    for _, row in manifest.iterrows():
        xena_id = str(row["tcpa_sample_id"])[:15]
        sample_map.setdefault(xena_id, str(row["tcpa_sample_id"]))

    with gzip.open(xena_gz, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    wanted_names = [s for s in header[1:] if s in sample_map]
    if not wanted_names:
        raise RuntimeError("No TCGA samples from the manifest were found in the Xena matrix.")

    usecols = ["sample"] + wanted_names
    raw = pd.read_csv(xena_gz, sep="\t", compression="gzip", usecols=usecols)
    raw["gene_symbol"] = raw["sample"].astype(str).str.split(".").str[0].map(id_to_symbol)
    raw = raw.dropna(subset=["gene_symbol"]).drop(columns=["sample"])
    value_cols = [c for c in raw.columns if c != "gene_symbol"]
    values = raw[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    tpm = np.maximum(np.power(2.0, values) - 0.001, 0.0).astype(np.float32)
    log2_tpm1 = np.log2(tpm + 1.0).astype(np.float32)
    mat = pd.DataFrame(log2_tpm1, columns=value_cols)
    mat.insert(0, "gene_symbol", raw["gene_symbol"].to_numpy())
    mat = mat.groupby("gene_symbol", sort=False).median(numeric_only=True)
    mat = mat.reindex(gene_order)
    rna = mat.T
    rna.index = [sample_map[str(s)] for s in rna.index]
    rna.index.name = "sample_id"
    rna = rna.loc[~rna.index.duplicated()].astype(np.float32)
    rna.to_parquet(out_path)

    audit = {
        "xena_file": str(xena_gz),
        "probemap_file": str(probemap),
        "manifest_samples": int(len(manifest)),
        "matched_xena_samples": int(rna.shape[0]),
        "genes_in_model_order": int(len(gene_order)),
        "nonmissing_gene_columns": int(rna.notna().any(axis=0).sum()),
        "transform": "Toil log2(TPM+0.001) converted to log2(TPM+1)",
    }
    (out / "logs/tcga_xena_rna_build_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return rna


def batch_iter(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def long_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.long, device=device)


def load_base_data(bridge, data_dir: Path, device: torch.device) -> dict:
    rna = pd.read_parquet(data_dir / "rna_log2_tpm_paired.parquet")
    total = pd.read_parquet(data_dir / "total_protein_gene_study_zscore_min20pct.parquet").loc[rna.index]
    phospho_full = pd.read_parquet(data_dir / "phosphosite_gene_site_study_zscore_min20pct_targets.parquet").loc[rna.index]
    residual_cols = set(pd.read_parquet(data_dir / "phosphosite_gene_site_total_residual_targets.parquet").columns)
    phospho_cols = [c for c in phospho_full.columns if c in residual_cols]
    phospho = phospho_full[phospho_cols]
    manifest = pd.read_csv(data_dir / "sample_manifest.tsv", sep="\t").set_index("sample_id").loc[rna.index]
    z = bridge.encode(rna, batch_size=256, scaled=True)
    x_df = pd.concat([rna, z], axis=1)
    cancer_cat = manifest["cancer_label"].astype("category")
    study_cat = manifest["pdc_study_id"].astype("category")
    return {
        "x": x_df.to_numpy(dtype=np.float32),
        "total": total.to_numpy(dtype=np.float32),
        "phospho": phospho.to_numpy(dtype=np.float32),
        "feature_names": list(x_df.columns),
        "total_names": list(total.columns),
        "phospho_names": list(phospho.columns),
        "sample_ids": list(x_df.index),
        "manifest": manifest.reset_index(),
        "cancer_ids": cancer_cat.cat.codes.to_numpy(dtype=np.int64),
        "study_ids": study_cat.cat.codes.to_numpy(dtype=np.int64),
        "cancer_levels": list(cancer_cat.cat.categories),
        "study_levels": list(study_cat.cat.categories),
        "device": device,
    }


def make_context_arrays(index: pd.Index, sample_meta: pd.DataFrame, ckpt: dict) -> tuple[np.ndarray, np.ndarray]:
    meta = sample_meta.set_index("tcpa_sample_id").loc[index]
    cancer_levels = [str(x) for x in ckpt["cancer_levels"]]
    study_levels = [str(x) for x in ckpt["study_levels"]]
    cancer_default = 0
    study_default = 0
    cancer = np.array(
        [cancer_levels.index(x) if x in cancer_levels else cancer_default for x in meta["cptac_cancer_label"].astype(str)],
        dtype=np.int64,
    )
    study = np.array(
        [study_levels.index(x) if x in study_levels else study_default for x in meta["cptac_study_id"].astype(str)],
        dtype=np.int64,
    )
    return cancer, study


def standardize_external(x_ext: pd.DataFrame, ckpt: dict) -> np.ndarray:
    mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    std = np.asarray(ckpt["x_std"], dtype=np.float32)
    std = np.where((std > 1e-6) & np.isfinite(std), std, 1.0)
    mat = x_ext.reindex(columns=ckpt["feature_names"]).to_numpy(dtype=np.float32)
    mat = np.where(np.isfinite(mat), mat, mean[None, :])
    return ((mat - mean[None, :]) / std[None, :]).astype(np.float32)


def standardize_internal_x(data: dict, ckpt: dict) -> np.ndarray:
    mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    std = np.asarray(ckpt["x_std"], dtype=np.float32)
    std = np.where((std > 1e-6) & np.isfinite(std), std, 1.0)
    return ((data["x"] - mean[None, :]) / std[None, :]).astype(np.float32)


def aligned_internal_phospho(data: dict, names: list[str]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(data["phospho_names"])}
    cols = [idx[n] for n in names]
    return data["phospho"][:, cols].astype(np.float32)


def build_external_corr_bank(
    x_train_z: np.ndarray,
    y_train_z: np.ndarray,
    y_train_mask: np.ndarray,
    x_ext_z: np.ndarray,
    k: int,
    chunk: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        z = np.zeros((x_ext_z.shape[0], y_train_z.shape[1]), dtype=np.float32)
        return z, z
    x_train = torch.tensor(x_train_z, dtype=torch.float32, device=device)
    x_ext_t = torch.tensor(x_ext_z, dtype=torch.float32, device=device)
    y_train = torch.tensor(np.nan_to_num(y_train_z, nan=0.0), dtype=torch.float32, device=device)
    m_train = torch.tensor(y_train_mask, dtype=torch.float32, device=device)
    banks = []
    masks = []
    with torch.no_grad():
        for start in range(0, y_train_z.shape[1], chunk):
            end = min(start + chunk, y_train_z.shape[1])
            yt = y_train[:, start:end] * m_train[:, start:end]
            counts = m_train[:, start:end].sum(dim=0).clamp_min(1.0)
            corr = x_train.T @ yt / counts[None, :]
            top = torch.topk(torch.abs(corr), k=min(k, corr.shape[0]), dim=0)
            idx = top.indices
            weights = torch.gather(corr, 0, idx)
            weights = weights / torch.abs(weights).sum(dim=0, keepdim=True).clamp_min(1e-6)
            vals = [(x_ext_t[:, idx[:, t]] * weights[:, t][None, :]).sum(dim=1) for t in range(end - start)]
            banks.append(torch.stack(vals, dim=1).detach().cpu().numpy().astype(np.float32))
            valid = (counts.detach().cpu().numpy() >= 10).astype(np.float32)
            masks.append(np.repeat(valid[None, :], x_ext_z.shape[0], axis=0))
    return np.concatenate(banks, axis=1), np.concatenate(masks, axis=1).astype(np.float32)


def predict_v3_family(
    label: str,
    script: Path,
    model_dir: Path,
    model_prefix: str,
    data: dict,
    x_ext: pd.DataFrame,
    sample_meta: pd.DataFrame,
    device: torch.device,
    args: argparse.Namespace,
    v3_external_for_feedback: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mod = import_module(f"{label}_deploy_module", script)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=20260502).split(data["x"], data["cancer_ids"]))
    phospho_fold = []
    total_fold = []
    for fold, (train_idx, _) in enumerate(folds, start=1):
        ck = torch.load(model_dir / f"{model_prefix}_fold{fold}.pt", map_location="cpu", weights_only=False)
        cfg = mod.ExperimentConfig(**ck["config"])
        kwargs = {}
        if label == "v3_1_1_feedback_prior":
            kwargs["n_feedback"] = len(ck.get("feedback_anchor_sites", [])) or 1
        model = mod.ParentResidualKinaseCVAE(
            n_input=len(ck["feature_names"]),
            n_total=len(ck["total_names"]),
            n_phospho=len(ck["phospho_names"]),
            n_cancer=len(ck["cancer_levels"]),
            n_study=len(ck["study_levels"]),
            parent_total_idx=ck["parent_total_idx"],
            parent_total_mask=ck["parent_total_mask"],
            parent_alpha=ck["parent_alpha_init"],
            parent_beta=ck["parent_beta_init"],
            kinase_site_weight=ck["kinase_site_weight"],
            cfg=cfg,
            **kwargs,
        ).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()

        xz_ext = standardize_external(x_ext, ck)
        xz_train_all = standardize_internal_x(data, ck)
        phospho_train = aligned_internal_phospho(data, ck["phospho_names"])
        total_train = data["total"][:, [data["total_names"].index(n) for n in ck["total_names"]]]
        total_z = ((total_train - ck["total_mean"]) / ck["total_std"]).astype(np.float32)
        phospho_z = ((phospho_train - ck["phospho_mean"]) / ck["phospho_std"]).astype(np.float32)
        total_mask = np.isfinite(total_z).astype(np.float32)
        phospho_mask = np.isfinite(phospho_z).astype(np.float32)
        total_z = np.nan_to_num(total_z, nan=0.0).astype(np.float32)
        phospho_z = np.nan_to_num(phospho_z, nan=0.0).astype(np.float32)
        residual_z, residual_mask = mod.compute_residual_target(
            total_z,
            total_mask,
            phospho_z,
            phospho_mask,
            np.asarray(ck["parent_total_idx"], dtype=np.int64),
            np.asarray(ck["parent_total_mask"], dtype=np.float32),
            np.asarray(ck["parent_alpha_init"], dtype=np.float32),
            np.asarray(ck["parent_beta_init"], dtype=np.float32),
        )
        total_direct, total_direct_mask, _ = mod.make_total_direct(xz_ext, ck["feature_names"], ck["total_names"])
        phospho_direct, phospho_direct_mask, _ = mod.make_phospho_direct(xz_ext, ck["feature_names"], ck["phospho_names"])
        if "external_x" in mod.build_corr_bank.__code__.co_varnames:
            bank, bank_mask = mod.build_corr_bank(
                xz_train_all,
                residual_z,
                residual_mask,
                train_idx,
                int(ck.get("bank_k", args.bank_k)),
                args.bank_chunk,
                external_x=xz_ext,
            )
        else:
            bank, bank_mask = build_external_corr_bank(
                xz_train_all[train_idx],
                residual_z[train_idx],
                residual_mask[train_idx],
                xz_ext,
                int(ck.get("bank_k", args.bank_k)),
                args.bank_chunk,
                device,
            )
        cancer, study = make_context_arrays(x_ext.index, sample_meta, ck)

        feedback_ext = None
        if label == "v3_1_1_feedback_prior":
            anchors = [row["target"] for row in ck.get("feedback_anchor_sites", [])]
            if anchors and v3_external_for_feedback is not None:
                v3_oof = pd.read_parquet(model_dir.parent.parent / "v3_train_oof" / "predictions" / "oof_phosphosite_predictions.parquet")
                fb_train = v3_oof.reindex(data["sample_ids"])[anchors].to_numpy(dtype=np.float32)
                fill = np.nanmedian(fb_train[train_idx], axis=0)
                fill = np.where(np.isfinite(fill), fill, 0.0)
                fb_train = np.where(np.isfinite(fb_train), fb_train, fill[None, :])
                mean = fb_train[train_idx].mean(axis=0)
                std = fb_train[train_idx].std(axis=0)
                std = np.where((std > 1e-6) & np.isfinite(std), std, 1.0)
                fb_ext = v3_external_for_feedback.reindex(index=x_ext.index, columns=anchors).to_numpy(dtype=np.float32)
                fb_ext = np.where(np.isfinite(fb_ext), fb_ext, fill[None, :])
                feedback_ext = ((fb_ext - mean[None, :]) / std[None, :]).astype(np.float32)
            else:
                feedback_ext = np.zeros((len(x_ext), kwargs["n_feedback"]), dtype=np.float32)

        total_batches = []
        phospho_batches = []
        with torch.no_grad():
            for sl in batch_iter(len(x_ext), args.batch_size):
                call = dict(
                    x=tensor(xz_ext[sl], device),
                    cancer=long_tensor(cancer[sl], device),
                    study=long_tensor(study[sl], device),
                    total_direct=tensor(total_direct[sl], device),
                    total_direct_mask=tensor(total_direct_mask[sl], device),
                    phospho_direct=tensor(phospho_direct[sl], device),
                    phospho_direct_mask=tensor(phospho_direct_mask[sl], device),
                    phospho_bank=tensor(bank[sl], device),
                    phospho_bank_mask=tensor(bank_mask[sl], device),
                    use_posterior=False,
                    use_observed_parent=False,
                )
                if feedback_ext is not None:
                    call["feedback"] = tensor(feedback_ext[sl], device)
                out = model(**call)
                total_batches.append((out["total"].detach().cpu().numpy() * ck["total_std"] + ck["total_mean"]).astype(np.float32))
                phospho_batches.append((out["phospho"].detach().cpu().numpy() * ck["phospho_std"] + ck["phospho_mean"]).astype(np.float32))
        total_fold.append(pd.DataFrame(np.vstack(total_batches), index=x_ext.index, columns=ck["total_names"]))
        phospho_fold.append(pd.DataFrame(np.vstack(phospho_batches), index=x_ext.index, columns=ck["phospho_names"]))
        print(f"{label} fold {fold} done", flush=True)

    common_total = sorted(set.intersection(*[set(x.columns) for x in total_fold]))
    common_phospho = sorted(set.intersection(*[set(x.columns) for x in phospho_fold]))
    total_pred = sum(df[common_total] for df in total_fold) / len(total_fold)
    phospho_pred = sum(df[common_phospho] for df in phospho_fold) / len(phospho_fold)
    return phospho_pred.astype(np.float32), total_pred.astype(np.float32)


def export_light_candidates(
    data: dict,
    x_ext: pd.DataFrame,
    v3_total_ext: pd.DataFrame,
    sample_meta: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    y = data["phospho"]
    total = data["total"]
    x = data["x"]
    feature_idx = {g: i for i, g in enumerate(data["feature_names"])}
    ext_feature_idx = {g: i for i, g in enumerate(x_ext.columns)}
    total_idx = {g: i for i, g in enumerate(data["total_names"])}
    ext_total_idx = {g: i for i, g in enumerate(v3_total_ext.columns)}
    site_genes = [str(s).split("|", 1)[0] for s in data["phospho_names"]]
    out = {name: np.full((len(x_ext), len(data["phospho_names"])), np.nan, dtype=np.float32) for name in ["parent_only", "rna_direct", "ridge_direct"]}
    counts = {name: np.zeros((len(x_ext), len(data["phospho_names"])), dtype=np.float32) for name in out}
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=20260502).split(x, data["cancer_ids"]))
    train_cancers = list(pd.Series(data["manifest"]["cancer_label"]).astype("category").cat.categories)
    train_cancer_index = {c: i for i, c in enumerate(train_cancers)}
    cancer_onehot = pd.get_dummies(pd.Series(data["cancer_ids"], dtype="category")).to_numpy(dtype=np.float32)
    meta = sample_meta.set_index("tcpa_sample_id").loc[x_ext.index]
    ext_cancer_onehot = np.zeros((len(x_ext), cancer_onehot.shape[1]), dtype=np.float32)
    for i, cancer in enumerate(meta["cptac_cancer_label"].astype(str)):
        if cancer in train_cancer_index:
            ext_cancer_onehot[i, train_cancer_index[cancer]] = 1.0

    for fold, (train_idx, _) in enumerate(folds, start=1):
        fold_pred = {name: np.full_like(out[name], np.nan) for name in out}
        for j, gene in enumerate(site_genes):
            ok_y = np.isfinite(y[train_idx, j])
            if ok_y.sum() < 10:
                continue
            if gene in total_idx and gene in ext_total_idx:
                tj = total_idx[gene]
                ok = ok_y & np.isfinite(total[train_idx, tj])
                if ok.sum() >= 10:
                    xv = total[train_idx, tj][ok]
                    yv = y[train_idx, j][ok]
                    a = np.cov(xv, yv, bias=True)[0, 1] / max(float(np.var(xv)), 1e-6)
                    b = float(yv.mean() - a * xv.mean())
                    fold_pred["parent_only"][:, j] = (a * v3_total_ext.iloc[:, ext_total_idx[gene]].to_numpy(dtype=np.float32) + b).astype(np.float32)
            if gene in feature_idx and gene in ext_feature_idx:
                gi = feature_idx[gene]
                egi = ext_feature_idx[gene]
                xv = x[train_idx, gi]
                ok = ok_y & np.isfinite(xv)
                if ok.sum() >= 10:
                    xx = xv[ok]
                    yy = y[train_idx, j][ok]
                    a = np.cov(xx, yy, bias=True)[0, 1] / max(float(np.var(xx)), 1e-6)
                    b = float(yy.mean() - a * xx.mean())
                    fold_pred["rna_direct"][:, j] = (a * x_ext.iloc[:, egi].to_numpy(dtype=np.float32) + b).astype(np.float32)
                    features_train = [x[:, gi]]
                    features_ext = [x_ext.iloc[:, egi].to_numpy(dtype=np.float32)]
                    if gene in total_idx and gene in ext_total_idx:
                        features_train.append(total[:, total_idx[gene]])
                        features_ext.append(v3_total_ext.iloc[:, ext_total_idx[gene]].to_numpy(dtype=np.float32))
                    features_train.extend([cancer_onehot[:, k] for k in range(cancer_onehot.shape[1])])
                    features_ext.extend([ext_cancer_onehot[:, k] for k in range(ext_cancer_onehot.shape[1])])
                    X = np.vstack(features_train).T.astype(np.float32)
                    Xe = np.vstack(features_ext).T.astype(np.float32)
                    ok = ok_y & np.all(np.isfinite(X[train_idx]), axis=1)
                    if ok.sum() >= 20:
                        col_mean = np.nanmean(X[train_idx][ok], axis=0).astype(np.float32)
                        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
                        Xe = np.where(np.isfinite(Xe), Xe, col_mean[None, :]).astype(np.float32)
                        model = Ridge(alpha=args.ridge_alpha)
                        model.fit(X[train_idx][ok], y[train_idx, j][ok])
                        fold_pred["ridge_direct"][:, j] = model.predict(Xe).astype(np.float32)
        for name in out:
            ok = np.isfinite(fold_pred[name])
            out[name] = np.where(ok, np.nan_to_num(out[name], nan=0.0) + np.where(ok, fold_pred[name], 0.0), out[name])
            counts[name] += ok.astype(np.float32)
        print(f"light candidates fold {fold} done", flush=True)

    frames = {}
    for name, mat in out.items():
        cnt = counts[name]
        mat = np.divide(mat, cnt, out=np.full_like(mat, np.nan), where=cnt > 0)
        frames[name] = pd.DataFrame(mat, index=x_ext.index, columns=data["phospho_names"])
    return frames


def sample_median_center(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = df.median(axis=1, skipna=True)
    centered = df.sub(med, axis=0).astype(np.float32)
    med_table = pd.DataFrame({"sample_id": df.index.astype(str), "raw_prediction_sample_median": med.to_numpy(dtype=float)})
    return centered, med_table


def parent_matrix(targets: list[str], parent_hat: pd.DataFrame, sample_index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=sample_index, columns=targets, dtype=np.float32)
    for target in targets:
        parent = str(target).split("|", 1)[0]
        if parent in parent_hat.columns:
            out[target] = parent_hat[parent].reindex(sample_index).to_numpy(dtype=np.float32)
    return out


def predict_total_by_context(bridge, rna: pd.DataFrame, meta: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    parts = []
    work = meta.set_index("tcpa_sample_id").loc[rna.index].reset_index()
    for (cancer, study), sub in work.groupby(["cptac_cancer_label", "cptac_study_id"], sort=True):
        ids = sub["tcpa_sample_id"].astype(str).tolist()
        pred = bridge.predict_total_protein(
            rna.loc[ids],
            cancer_label=str(cancer),
            study_label=str(study),
            context_mode="borrowed",
            batch_size=batch_size,
        )
        parts.append(pred)
        print(f"total protein context {cancer}/{study} n={len(ids)} done", flush=True)
    out = pd.concat(parts).loc[rna.index]
    out.index.name = "sample_id"
    return out.astype(np.float32)


def predict_scp68222_head(
    train_mod,
    ckpt: dict,
    rna: pd.DataFrame,
    latent: pd.DataFrame,
    parent_hat: pd.DataFrame,
    v4: pd.DataFrame,
    sample_meta: pd.DataFrame,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    targets = [str(x) for x in ckpt["targets"]]
    genes = [str(x) for x in ckpt["rna_genes"]]
    pathways = [
        train_mod.PathwaySpec(
            name=str(p["name"]),
            gene_idx=np.asarray(p["gene_idx"], dtype=np.int64),
            gene_names=[str(x) for x in p.get("gene_names", [])],
        )
        for p in ckpt["pathways"]
    ]
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    x_std = np.where((x_std > 1.0e-6) & np.isfinite(x_std), x_std, 1.0)
    x = rna.reindex(columns=genes).to_numpy(dtype=np.float32)
    x = np.where(np.isfinite(x), x, x_mean[None, :])
    x = ((x - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
    z = latent.to_numpy(dtype=np.float32)
    z = np.nan_to_num(z, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
    parent = parent_matrix(targets, parent_hat, rna.index).to_numpy(dtype=np.float32)
    v4_in = v4.reindex(index=rna.index, columns=targets).to_numpy(dtype=np.float32)
    v4_in = np.nan_to_num(v4_in, nan=0.0).astype(np.float32)

    group_levels = [str(x) for x in ckpt.get("cancer_group_levels", train_mod.CANCER_GROUP_LEVELS)]
    meta = sample_meta.set_index("tcpa_sample_id").loc[rna.index]
    group_names = [train_mod.cancer_group(x) for x in meta["cptac_cancer_label"].astype(str)]
    group_id = np.array([group_levels.index(g) if g in group_levels else group_levels.index("other") for g in group_names], dtype=np.int64)

    model = train_mod.CancerGroupPathwayResidualHead(
        n_genes=len(genes),
        n_sites=len(targets),
        latent_dim=z.shape[1],
        pathways=pathways,
        n_groups=len(group_levels),
        shrinkage=float(ckpt.get("shrinkage", 0.4)),
        d_model=64,
        dropout=0.12,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    parts = []
    with torch.no_grad():
        for sl in batch_iter(x.shape[0], batch_size):
            xb = torch.tensor(x[sl], dtype=torch.float32, device=device)
            zb = torch.tensor(z[sl], dtype=torch.float32, device=device)
            pb = torch.tensor(parent[sl], dtype=torch.float32, device=device)
            vb = torch.tensor(v4_in[sl], dtype=torch.float32, device=device)
            gb = torch.tensor(group_id[sl], dtype=torch.long, device=device)
            parts.append(model(xb, zb, pb, vb, gb).detach().cpu().numpy().astype(np.float32))
    pred = np.vstack(parts)
    return pd.DataFrame(pred, index=rna.index.astype(str), columns=targets)


def predict_phosphosite(
    args: argparse.Namespace,
    bridge,
    rna: pd.DataFrame,
    sample_meta: pd.DataFrame,
    strict_targets: list[str],
    out: Path,
    device: torch.device,
) -> pd.DataFrame:
    pred_path = out / "predictions/tcga_supported_scp68222_predicted_phosphosite_strict_targets.parquet"
    if args.skip_prediction_if_present and pred_path.exists():
        return pd.read_parquet(pred_path).set_index("sample_id")

    root = Path(args.root)
    bridge_root = Path(args.bridge_root) if args.bridge_root else root / "vm_lsy_parent/lsy/01_data/external_models/SCP682_scGPT_bridge/package/SCP682_scGPT_bridge"
    source = bridge_root / "artifacts/source_code"
    model_base = bridge_root / "artifacts/models/phosphosite_v3_family"
    data_dir = root / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2"

    data = load_base_data(bridge, data_dir, device)
    latent = bridge.encode(rna, batch_size=max(args.batch_size, 128), scaled=True)
    x_ext = pd.concat([rna, latent], axis=1)
    parent_hat = predict_total_by_context(bridge, rna, sample_meta, max(args.batch_size, 128))
    parent_hat.to_parquet(out / "predictions/tcga_supported_predicted_total_protein.parquet")

    specs = [
        ("v3_train_oof", "train_cptac_parent_residual_kinase_cvae_v3_20260502.py", "v3_train_oof", "20260502_cptac_parent_residual_kinase_cvae_experimental_v3"),
        ("v3_1_1_feedback_prior", "train_cptac_parent_residual_kinase_cvae_v3_1_1_feedback_prior_20260502.py", "v3_1_1_feedback_prior", "20260502_cptac_parent_residual_kinase_cvae_experimental_v3_1_1_feedback_prior"),
        ("v3_1_2_target_attention", "train_cptac_parent_residual_kinase_cvae_v3_1_2_target_attention_20260502.py", "v3_1_2_target_attention", "20260502_cptac_parent_residual_kinase_cvae_experimental_v3_1_2_target_attention"),
        ("v3_6_ranking_coverage_loss", "train_cptac_parent_residual_kinase_cvae_v3_6_ranking_coverage_loss_20260502.py", "v3_6_ranking_coverage_loss", "20260502_cptac_parent_residual_kinase_cvae_experimental_v3_6_ranking_coverage_loss"),
    ]
    candidates: dict[str, pd.DataFrame] = {}
    v3_pred = None
    v3_total = None
    for label, script_name, model_subdir, prefix in specs:
        pred, total_pred = predict_v3_family(
            label,
            source / script_name,
            model_base / model_subdir / "models",
            prefix,
            data,
            x_ext,
            sample_meta,
            device,
            args,
            v3_external_for_feedback=v3_pred,
        )
        candidates[label] = pred
        if label == "v3_train_oof":
            v3_pred = pred
            v3_total = total_pred
        pred[strict_targets].to_parquet(out / "predictions" / f"tcga_candidate_{label}_strict_targets.parquet")
        print(f"{label} candidate saved", flush=True)
    assert v3_total is not None
    light = export_light_candidates(data, x_ext, v3_total, sample_meta, args)
    for name, pred in light.items():
        candidates[name] = pred
        pred[strict_targets].to_parquet(out / "predictions" / f"tcga_candidate_{name}_strict_targets.parquet")
        print(f"{name} candidate saved", flush=True)

    formula = pd.read_csv(bridge_root / "artifacts/official_release/tables/v4_0_candidate_formula.tsv", sep="\t")
    weights = formula.set_index("candidate")["final_weight"].astype(float).to_dict()
    common_cols = sorted(set.intersection(*[set(candidates[name].columns) for name in weights]))
    common_index = rna.index
    v4_raw = None
    for name, weight in weights.items():
        part = candidates[name].loc[common_index, common_cols].astype(np.float32) * weight
        v4_raw = part if v4_raw is None else v4_raw.add(part, fill_value=0.0)
    assert v4_raw is not None
    v4_centered, med = sample_median_center(v4_raw)
    med.to_csv(out / "tables/tcga_supported_v4_raw_sample_medians.tsv", sep="\t", index=False)
    v4_centered[strict_targets].to_parquet(out / "predictions/tcga_supported_v4_predicted_phosphosite_strict_targets.parquet")

    if args.prediction_mode == "v4":
        final = v4_centered
    else:
        train_script = Path(args.scp68222_train_script) if args.scp68222_train_script else root / "SCP682-22/frozen_release/SCP682_22_paper_package_20260520/source_code/scp682_22_scripts/train_scp682_22_cancer_group_pathway_residual.py"
        model_dir = Path(args.scp68222_model_dir) if args.scp68222_model_dir else root / "vm_lsy_parent/lsy/01_data/single_cell/intermediate/scp682_22_full_transfer_prior_v1/models"
        train_mod = import_module("scp68222_train_deploy", train_script)
        ckpt_paths = sorted(model_dir.glob("scp682_22_cancer_group_pathway_residual_*_fold*.pt"))
        if not ckpt_paths:
            raise FileNotFoundError(f"No SCP682-22 checkpoints found: {model_dir}")
        head_parts = []
        used = []
        for ckpt_path in ckpt_paths:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            head_parts.append(predict_scp68222_head(train_mod, ckpt, rna, latent, parent_hat, v4_centered, sample_meta, args.batch_size, device))
            used.append(str(ckpt_path))
            print(f"SCP682-22 head {ckpt_path.name} done", flush=True)
        common = sorted(set.intersection(*[set(x.columns) for x in head_parts]))
        final_raw = sum(x[common] for x in head_parts) / float(len(head_parts))
        final, med22 = sample_median_center(final_raw)
        med22.to_csv(out / "tables/tcga_supported_scp68222_raw_sample_medians.tsv", sep="\t", index=False)
        pd.DataFrame({"checkpoint": used}).to_csv(out / "tables/scp68222_checkpoints_used.tsv", sep="\t", index=False)

    final_strict = final.reindex(columns=strict_targets)
    final_out = final_strict.copy()
    final_out.insert(0, "sample_id", final_out.index.astype(str))
    final_out.to_parquet(pred_path, index=False)
    return final_strict.astype(np.float32)


def as_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("[^0-9eE+.-]", "", regex=True), errors="coerce")


def fetch_cbioportal_patient_clinical(projects: list[str], cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path, sep="\t")
    rows: list[dict] = []
    for project in projects:
        study = STUDY_BY_PROJECT.get(project)
        if study is None:
            continue
        response = requests.get(
            f"{CBIOPORTAL_BASE}/studies/{study}/clinical-data",
            params={"clinicalDataType": "PATIENT", "projection": "SUMMARY"},
            timeout=120,
        )
        response.raise_for_status()
        for item in response.json():
            rows.append(
                {
                    "project": project,
                    "study_id": study,
                    "patient_id": item.get("patientId"),
                    "clinicalAttributeId": item.get("clinicalAttributeId"),
                    "value": item.get("value"),
                }
            )
    long = pd.DataFrame(rows)
    if long.empty:
        raise RuntimeError("No cBioPortal clinical records were downloaded.")
    wide = (
        long.pivot_table(
            index=["project", "study_id", "patient_id"],
            columns="clinicalAttributeId",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    wide.to_csv(cache_path, sep="\t", index=False)
    return wide


def standardize_clinical(clinical: pd.DataFrame) -> pd.DataFrame:
    out = clinical.copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out["os_days"] = as_number(out.get("OS_MONTHS", pd.Series(np.nan, index=out.index))) * 30.4375
    status = out.get("OS_STATUS", pd.Series("", index=out.index)).astype(str).str.upper()
    out["os_event"] = np.where(
        status.str.contains("1:|DECEASED|DEAD"),
        1,
        np.where(status.str.contains("0:|LIVING|ALIVE"), 0, np.nan),
    )
    out["age_at_diagnosis"] = as_number(out.get("AGE", pd.Series(np.nan, index=out.index)))
    return out[["project", "patient_id", "os_days", "os_event", "age_at_diagnosis"]].copy()


def fit_cox(time: np.ndarray, event: np.ndarray, x: np.ndarray, age: np.ndarray | None = None) -> dict:
    ok = np.isfinite(time) & np.isfinite(event) & np.isfinite(x) & (time > 0)
    cols = [x]
    names = ["site"]
    if age is not None:
        ok = ok & np.isfinite(age)
        cols.append(age)
        names.append("age")
    n = int(ok.sum())
    events = int(event[ok].sum()) if n else 0
    if n < 10 or events < 3 or np.nanstd(x[ok]) < 1e-9:
        return {"n": n, "events": events, "beta": np.nan, "hr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}
    xx = x[ok].astype(float)
    xx = (xx - xx.mean()) / max(xx.std(ddof=0), 1e-9)
    exog_cols = [xx]
    if age is not None:
        aa = age[ok].astype(float)
        aa = (aa - aa.mean()) / max(aa.std(ddof=0), 1e-9)
        exog_cols.append(aa)
    exog = np.vstack(exog_cols).T
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = PHReg(time[ok].astype(float), exog, status=event[ok].astype(int), ties="efron").fit(disp=0)
        beta = float(res.params[0])
        se = float(res.bse[0])
        return {
            "n": n,
            "events": events,
            "beta": beta,
            "hr": float(np.exp(beta)),
            "ci_low": float(np.exp(beta - 1.96 * se)),
            "ci_high": float(np.exp(beta + 1.96 * se)),
            "p": float(res.pvalues[0]),
        }
    except Exception:
        return {"n": n, "events": events, "beta": np.nan, "hr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}


def run_tcga_site_survival(
    strict: pd.DataFrame,
    pred: pd.DataFrame,
    sample_meta: pd.DataFrame,
    clinical: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    meta = sample_meta.merge(clinical, left_on=["project", "patient_id"], right_on=["project", "patient_id"], how="left")
    meta = meta.set_index("tcpa_sample_id").loc[pred.index]
    rows = []
    for _, row in strict.iterrows():
        target = str(row["target"])
        cancer = str(row["cancer_label"])
        project = CPTAC_TO_TCGA_PROJECT.get(cancer)
        if project is None or target not in pred.columns:
            rec = row.to_dict()
            rec.update({"tcga_project": project, "tcga_n": 0, "tcga_events": 0, "tcga_hr_per_sd": np.nan, "tcga_p": np.nan})
            rows.append(rec)
            continue
        ids = meta.index[meta["project"].eq(project)]
        sub_meta = meta.loc[ids]
        values = pred.loc[ids, target].to_numpy(dtype=float)
        uni = fit_cox(
            sub_meta["os_days"].to_numpy(dtype=float),
            sub_meta["os_event"].to_numpy(dtype=float),
            values,
            None,
        )
        age_adj = fit_cox(
            sub_meta["os_days"].to_numpy(dtype=float),
            sub_meta["os_event"].to_numpy(dtype=float),
            values,
            sub_meta["age_at_diagnosis"].to_numpy(dtype=float),
        )
        rec = row.to_dict()
        rec.update(
            {
                "tcga_project": project,
                "tcga_n": uni["n"],
                "tcga_events": uni["events"],
                "tcga_beta": uni["beta"],
                "tcga_hr_per_sd": uni["hr"],
                "tcga_ci95_low": uni["ci_low"],
                "tcga_ci95_high": uni["ci_high"],
                "tcga_p": uni["p"],
                "tcga_age_adjusted_n": age_adj["n"],
                "tcga_age_adjusted_events": age_adj["events"],
                "tcga_age_adjusted_hr_per_sd": age_adj["hr"],
                "tcga_age_adjusted_p": age_adj["p"],
                "direction_concordant": bool(np.isfinite(uni["beta"]) and np.sign(float(row["site_beta"])) == np.sign(uni["beta"])),
            }
        )
        rows.append(rec)
    out = pd.DataFrame(rows)
    ok = out["tcga_p"].notna()
    out["tcga_fdr_within_strict_pool"] = np.nan
    if ok.any():
        out.loc[ok, "tcga_fdr_within_strict_pool"] = false_discovery_control(out.loc[ok, "tcga_p"].to_numpy(dtype=float), method="bh")
    out["tcga_predicted_site_significant"] = (
        (out["tcga_n"] >= args.min_tcga_n)
        & (out["tcga_events"] >= args.min_tcga_events)
        & (out["tcga_p"] < args.tcga_p)
    )
    out["cptac_measured_and_tcga_predicted_concordant"] = out["strict_site_specific_candidate"].astype(bool) & out["tcga_predicted_site_significant"] & out["direction_concordant"].astype(bool)
    return out.sort_values(["cptac_measured_and_tcga_predicted_concordant", "tcga_p", "site_p"], ascending=[False, True, True])


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out_dir) if args.out_dir else root / "02_results/model_validation/20260527_cptac_measured_tcga_predicted_site_survival_concordance_v1"
    ensure_dirs(out)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)

    strict_path = Path(args.strict_candidates) if args.strict_candidates else root / "02_results/model_validation/20260527_measured_cptac_site_specific_survival_rescreen_v1/tables/strict_site_specific_candidates_measured_site_rna_total_null.tsv"
    manifest_path = Path(args.tcpa_manifest) if args.tcpa_manifest else root / "_codex_inputs/tcpa_supported_sample_manifest.tsv"
    strict = pd.read_csv(strict_path, sep="\t")
    sample_meta = pd.read_csv(manifest_path, sep="\t")
    strict_targets = [t for t in strict["target"].astype(str).drop_duplicates().tolist()]

    bridge_root = Path(args.bridge_root) if args.bridge_root else root / "vm_lsy_parent/lsy/01_data/external_models/SCP682_scGPT_bridge/package/SCP682_scGPT_bridge"
    sys.path.insert(0, str(bridge_root))
    from scp682_scgpt_bridge.api import SCP682Bridge

    bridge = SCP682Bridge.from_package(bridge_root, device=device)
    gene_order = bridge.gene_order
    rna_dir = root / "01_data/tcga_tcpa/intermediate/tcga_xena_rna_for_scp682_20260527"
    rna_dir.mkdir(parents=True, exist_ok=True)
    rna = build_tcga_rna_from_xena(sample_meta, gene_order, rna_dir, out)
    keep = [s for s in sample_meta["tcpa_sample_id"].astype(str) if s in rna.index]
    sample_meta = sample_meta.loc[sample_meta["tcpa_sample_id"].astype(str).isin(keep)].copy()
    rna = rna.loc[keep]
    sample_meta.to_csv(out / "tables/tcga_supported_prediction_manifest.tsv", sep="\t", index=False)

    pred = predict_phosphosite(args, bridge, rna, sample_meta, strict_targets, out, device)
    clinical_raw = fetch_cbioportal_patient_clinical(sorted(sample_meta["project"].unique()), out / "tables/tcga_cbioportal_patient_clinical_wide.tsv")
    clinical = standardize_clinical(clinical_raw)
    clinical.to_csv(out / "tables/tcga_cbioportal_patient_clinical_standardized.tsv", sep="\t", index=False)
    survival = run_tcga_site_survival(strict, pred, sample_meta, clinical, args)
    survival.to_csv(out / "tables/cptac_strict_candidates_with_tcga_predicted_survival.tsv", sep="\t", index=False)
    confirmed = survival.loc[survival["cptac_measured_and_tcga_predicted_concordant"].astype(bool)].copy()
    confirmed.to_csv(out / "tables/tcga_confirmed_site_specific_candidates.tsv", sep="\t", index=False)

    review_path = strict_path.parent / "manual_antibody_novelty_biology_review.tsv"
    if review_path.exists():
        review = pd.read_csv(review_path, sep="\t")
        key_cols = [c for c in ["cancer_label", "target"] if c in review.columns]
        if len(key_cols) == 2:
            merged = confirmed.merge(review, on=key_cols, how="left", suffixes=("", "_manual_review"))
            merged.to_csv(out / "tables/tcga_confirmed_manual_antibody_novelty_biology_review.tsv", sep="\t", index=False)

    summary = {
        "strict_candidate_file": str(strict_path),
        "tcpa_manifest_file": str(manifest_path),
        "bridge_root": str(bridge_root),
        "prediction_mode": args.prediction_mode,
        "device": str(device),
        "n_strict_cptac_candidates": int(len(strict)),
        "n_unique_strict_targets": int(len(strict_targets)),
        "n_tcga_prediction_samples": int(pred.shape[0]),
        "n_tcga_projects": int(sample_meta["project"].nunique()),
        "n_tcga_evaluable_candidates": int(survival["tcga_p"].notna().sum()),
        "n_tcga_predicted_site_p_lt_threshold": int(survival["tcga_predicted_site_significant"].sum()),
        "n_direction_concordant_confirmed": int(confirmed.shape[0]),
        "tcga_p_threshold": float(args.tcga_p),
        "min_tcga_n": int(args.min_tcga_n),
        "min_tcga_events": int(args.min_tcga_events),
    }
    (out / "logs/run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not confirmed.empty:
        cols = ["cancer_label", "target", "site_hr_per_sd", "site_p", "tcga_project", "tcga_hr_per_sd", "tcga_p", "tcga_fdr_within_strict_pool"]
        print(confirmed[cols].head(30).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
