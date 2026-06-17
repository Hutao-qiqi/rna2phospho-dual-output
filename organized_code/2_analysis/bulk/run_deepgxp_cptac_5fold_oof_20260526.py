#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATASETS = [
    "CPTAC_all",
    "CPTAC_kidney",
    "CPTAC_pancreas_HN",
    "CPTAC_gynecologic",
    "CPTAC_gi_hepato",
    "CPTAC_lung",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DeepGxPBulkTorch(nn.Module):
    def __init__(self, n_genes: int, n_outputs: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1024,
            kernel_size=50,
            stride=50,
            padding=25,
            bias=True,
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_genes)
            flat = int(self.pool(self.conv(dummy)).reshape(1, -1).shape[1])
        self.dense = nn.Linear(flat, 512, bias=True)
        self.out = nn.Linear(512, n_outputs, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        return self.out(x)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return (diff * diff).sum() / denom


def make_cancer_groups(meta: pd.DataFrame) -> pd.Series:
    mapping = {
        "CCRCC": "kidney",
        "NON_CCRCC": "kidney",
        "PDA": "pancreas_headneck",
        "HNSCC": "pancreas_headneck",
        "UCEC": "gynecologic",
        "UCEC_CONFIRM": "gynecologic",
        "STAD": "gi_hepatobiliary",
        "LUAD": "lung",
        "LUAD_CONFIRM": "lung",
        "LSCC": "lung",
    }
    return meta["cancer_label"].astype(str).map(mapping).fillna("other")


def dataset_masks(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    g = meta["cancer_group"].astype(str).to_numpy()
    return {
        "CPTAC_all": np.ones(len(meta), dtype=bool),
        "CPTAC_kidney": g == "kidney",
        "CPTAC_pancreas_HN": g == "pancreas_headneck",
        "CPTAC_gynecologic": g == "gynecologic",
        "CPTAC_gi_hepato": g == "gi_hepatobiliary",
        "CPTAC_lung": g == "lung",
    }


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    package = Path(args.package_dir)
    y = pd.read_parquet(package / "training_set" / "observed_phosphosite.parquet")
    y.index = y.index.astype(str)
    y.columns = y.columns.astype(str)

    meta = pd.read_csv(args.sample_manifest_path, sep="\t")
    if "sample_id" not in meta.columns and "index" in meta.columns:
        meta = meta.rename(columns={"index": "sample_id"})
    meta["sample_id"] = meta["sample_id"].astype(str)
    sample_ids = [s for s in meta["sample_id"] if s in y.index]
    meta = meta.set_index("sample_id").loc[sample_ids].reset_index()
    meta["cancer_group"] = make_cancer_groups(meta)

    rna = pd.read_parquet(args.rna_path)
    rna.index = rna.index.astype(str)
    rna.columns = rna.columns.astype(str)
    rna = rna.loc[sample_ids].astype(np.float32)

    target_list = pd.read_csv(args.target_list_path, sep="\t")
    if "spearman" in target_list.columns:
        val = pd.to_numeric(target_list["spearman"], errors="coerce")
        target_list = target_list[np.isfinite(val)]
    targets = [t for t in target_list["target"].astype(str).tolist() if t in y.columns]
    if args.max_sites > 0:
        targets = targets[: args.max_sites]
    y = y.loc[sample_ids, targets]
    return rna, y, meta, targets, sample_ids


def load_folds(path: str, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    dat = np.load(path, allow_pickle=True)
    folds = []
    for k in range(1, n_splits + 1):
        folds.append((dat[f"train_idx_fold{k}"].astype(np.int64), dat[f"test_idx_fold{k}"].astype(np.int64)))
    return folds


def make_val_split(train_full: np.ndarray, strata: np.ndarray, seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    try:
        tr_local, val_local = next(splitter.split(np.zeros(len(train_full)), strata[train_full]))
        return train_full[tr_local], train_full[val_local]
    except ValueError:
        rng = np.random.default_rng(seed)
        order = rng.permutation(train_full)
        n_val = max(1, int(math.ceil(len(train_full) * val_fraction)))
        return order[n_val:], order[:n_val]


def pick_top_variable_genes(x: np.ndarray, train_idx: np.ndarray, n_genes: int) -> np.ndarray:
    var = np.nanvar(x[train_idx], axis=0)
    var = np.nan_to_num(var, nan=-1.0, posinf=-1.0, neginf=-1.0)
    n_keep = min(n_genes, x.shape[1])
    idx = np.argpartition(var, -n_keep)[-n_keep:]
    return idx[np.argsort(var[idx])[::-1]].astype(np.int64)


def scale_x(x: np.ndarray, train_idx: np.ndarray, feature_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = x[:, feature_idx]
    mean = np.nanmean(values[train_idx], axis=0)
    std = np.nanstd(values[train_idx], axis=0)
    std = np.where((std < 1e-6) | ~np.isfinite(std), 1.0, std)
    scaled = (values - mean) / std
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return scaled, mean.astype(np.float32), std.astype(np.float32)


def scale_y(y: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(y).astype(np.float32)
    mean = np.nanmean(y[train_idx], axis=0)
    std = np.nanstd(y[train_idx], axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where((std < 1e-6) | ~np.isfinite(std), 1.0, std)
    scaled = (y - mean) / std
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return scaled, mask, mean.astype(np.float32), std.astype(np.float32)


def train_fold(
    x_np: np.ndarray,
    y_np: np.ndarray,
    train_full: np.ndarray,
    test_idx: np.ndarray,
    strata: np.ndarray,
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
    out: Path,
    genes: np.ndarray,
) -> tuple[np.ndarray, dict[str, object], pd.DataFrame]:
    train_idx, val_idx = make_val_split(train_full, strata, args.seed + fold, args.val_fraction)
    feature_idx = pick_top_variable_genes(x_np, train_idx, args.n_genes)
    x_scaled, x_mean, x_std = scale_x(x_np, train_idx, feature_idx)
    y_scaled, y_mask, y_mean, y_std = scale_y(y_np, train_idx)

    feature_table = pd.DataFrame({
        "fold": fold,
        "rank": np.arange(1, len(feature_idx) + 1),
        "gene": genes[feature_idx],
    })

    model = DeepGxPBulkTorch(x_scaled.shape[1], y_scaled.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_ds = TensorDataset(
        torch.from_numpy(x_scaled[train_idx]).unsqueeze(1),
        torch.from_numpy(y_scaled[train_idx]),
        torch.from_numpy(y_mask[train_idx]),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_x = torch.from_numpy(x_scaled[val_idx]).unsqueeze(1).to(device)
    val_y = torch.from_numpy(y_scaled[val_idx]).to(device)
    val_m = torch.from_numpy(y_mask[val_idx]).to(device)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        mask_sum = 0.0
        for xb, yb, mb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(xb), yb, mb)
            loss.backward()
            optimizer.step()
            count = float(mb.sum().detach().cpu())
            loss_sum += float(loss.detach().cpu()) * count
            mask_sum += count

        model.eval()
        with torch.no_grad():
            val_loss = float(masked_mse(model(val_x), val_y, val_m).detach().cpu())
        train_loss = loss_sum / max(mask_sum, 1.0)
        history.append({"fold": fold, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if epoch == 1 or epoch % 10 == 0:
            log(f"DeepGxP fold {fold} epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(best_state, out / "models" / f"deepgxp_fold{fold}.pt")

    model.eval()
    pred_chunks = []
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_scaled[test_idx]).unsqueeze(1)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    with torch.no_grad():
        for (xb,) in test_loader:
            pred_chunks.append(model(xb.to(device, non_blocking=True)).detach().cpu().numpy())
    pred_scaled = np.vstack(pred_chunks)
    pred = pred_scaled * y_std + y_mean

    info = {
        "fold": fold,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_genes": int(len(feature_idx)),
        "n_outputs": int(y_np.shape[1]),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "target_train_coverage_mean": float(np.mean(y_mask[train_idx])),
        "target_test_coverage_mean": float(np.mean(y_mask[test_idx])),
        "x_mean_mean": float(np.mean(x_mean)),
        "x_std_mean": float(np.mean(x_std)),
        "history": history,
    }
    return pred.astype(np.float32), info, feature_table


def corr_or_nan(true: np.ndarray, pred: np.ndarray, method: str, min_n: int) -> tuple[float, float, int]:
    mask = np.isfinite(true) & np.isfinite(pred)
    n = int(mask.sum())
    if n < min_n or np.nanstd(true[mask]) < 1e-12 or np.nanstd(pred[mask]) < 1e-12:
        return float("nan"), float("nan"), n
    if method == "pearson":
        r = pearsonr(pred[mask], true[mask])
    else:
        r = spearmanr(pred[mask], true[mask])
    return float(r.statistic), float(r.pvalue), n


def evaluate(method: str, pred_df: pd.DataFrame, y_df: pd.DataFrame, meta: pd.DataFrame, targets: list[str], min_n: int) -> pd.DataFrame:
    masks = dataset_masks(meta)
    yv = y_df[targets].to_numpy(dtype=np.float32)
    pv = pred_df[targets].to_numpy(dtype=np.float32)
    rows = []
    for dataset, mask in masks.items():
        idx = np.where(mask)[0]
        for j, target in enumerate(targets):
            rho, pval, n = corr_or_nan(yv[idx, j], pv[idx, j], "spearman", min_n)
            pr, pp, _ = corr_or_nan(yv[idx, j], pv[idx, j], "pearson", min_n)
            rows.append({
                "method": method,
                "dataset": dataset,
                "target": target,
                "n_samples_used": n,
                "spearman": rho,
                "rho_p_value": pval,
                "pearson": pr,
                "pearson_p_value": pp,
            })
    return pd.DataFrame(rows)


def write_summary(df: pd.DataFrame, out: Path) -> None:
    med = df.groupby(["method", "dataset"])["spearman"].median().unstack("dataset")
    pear = df.groupby(["method", "dataset"])["pearson"].median().unstack("dataset")
    na = df.assign(is_na=df["spearman"].isna()).groupby(["method", "dataset"])["is_na"].mean().unstack("dataset")
    med.to_csv(out / "median_spearman_matrix.tsv", sep="\t")
    pear.to_csv(out / "median_pearson_matrix.tsv", sep="\t")
    na.to_csv(out / "na_ratio_matrix.tsv", sep="\t")
    log("median Spearman matrix")
    print(med.to_string(), flush=True)
    log("median Pearson matrix")
    print(pear.to_string(), flush=True)
    log("NA ratio matrix")
    print(na.to_string(), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--package-dir", default="/data/lsy/Infinite_Stream/SCP682-22/frozen_release/SCP682_22_paper_package_20260520")
    p.add_argument("--rna-path", default="/data/lsy/Infinite_Stream/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/rna_log2_tpm_paired.parquet")
    p.add_argument("--sample-manifest-path", default="/data/lsy/Infinite_Stream/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/sample_manifest.tsv")
    p.add_argument("--target-list-path", default="/data/lsy/Infinite_Stream/SCP682-main/results/20260523_general_graph_residual_e160/tables/per_site_pseudo_external_spearman_best.tsv")
    p.add_argument("--fold-indices-path", default="/data/lsy/Infinite_Stream/SCP682-main/results/fast_fullsite_baselines/fold_indices.npz")
    p.add_argument("--out-dir", default="/data/lsy/Infinite_Stream/SCP682-main/results/20260526_deepgxp_cptac_5fold_oof")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-genes", type=int, default=13995)
    p.add_argument("--max-sites", type=int, default=0)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=1e-7)
    p.add_argument("--val-fraction", type=float, default=0.125)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=682)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--min-n", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.out_dir)
    for sub in ["predictions", "tables", "logs", "models"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    rna, y, meta, targets, sample_ids = load_inputs(args)
    folds = load_folds(args.fold_indices_path, args.n_splits)
    x_np = rna.to_numpy(dtype=np.float32, copy=True)
    y_np = y.to_numpy(dtype=np.float32, copy=True)
    genes = rna.columns.astype(str).to_numpy()
    strata = meta["cancer_group"].astype(str).to_numpy()

    oof = pd.DataFrame(np.nan, index=sample_ids, columns=targets, dtype=np.float32)
    fold_infos = []
    feature_tables = []
    for fold, (train_full, test_idx) in enumerate(folds, start=1):
        pred_path = out / "predictions" / f"deepgxp_fold{fold}_predictions.parquet"
        if pred_path.exists() and args.resume:
            log(f"fold {fold}: reuse prediction")
            part = pd.read_parquet(pred_path)
            info_path = out / "logs" / f"fold{fold}_summary.json"
            if info_path.exists():
                with open(info_path, encoding="utf-8") as f:
                    fold_infos.append(json.load(f))
            ids = [sample_ids[i] for i in test_idx]
            oof.loc[ids, targets] = part.loc[ids, targets].to_numpy(dtype=np.float32)
            continue
        log(f"DeepGxP fold {fold}: start")
        fold_pred, info, feature_table = train_fold(
            x_np, y_np, train_full, test_idx, strata, fold, args, device, out, genes
        )
        ids = [sample_ids[i] for i in test_idx]
        part = pd.DataFrame(fold_pred, index=ids, columns=targets)
        part.to_parquet(pred_path)
        oof.loc[ids, targets] = fold_pred
        pd.DataFrame(info["history"]).to_csv(out / "logs" / f"fold{fold}_history.tsv", sep="\t", index=False)
        with open(out / "logs" / f"fold{fold}_summary.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in info.items() if k != "history"}, f, indent=2, ensure_ascii=False)
        fold_infos.append({k: v for k, v in info.items() if k != "history"})
        feature_tables.append(feature_table)
        log(f"DeepGxP fold {fold}: done best_epoch={info['best_epoch']} best_val={info['best_val_loss']:.5f}")

    oof.to_parquet(out / "predictions" / "deepgxp_5fold_oof_predictions.parquet")
    if feature_tables:
        pd.concat(feature_tables, ignore_index=True).to_csv(out / "tables" / "selected_genes_by_fold.tsv", sep="\t", index=False)

    metrics = evaluate("DeepGxP_5fold", oof, y, meta, targets, args.min_n)
    metrics.to_csv(out / "tables" / "deepgxp_5fold_per_site_spearman.tsv", sep="\t", index=False)
    write_summary(metrics, out)

    summary = {
        "model": "DeepGxP_bulk_CNN_5fold_OOF_on_CPTAC_PDC_phosphosites",
        "architecture": "Conv1D(1024,kernel=50,stride=50,padding=25)-MaxPool1D-Dense512-linear multi-target phosphosite head",
        "source_reference": "DeepGxP iScience 2026 DOI 10.1016/j.isci.2026.114815",
        "split": "reuse SCP682 baseline StratifiedKFold by cancer_group random_state=42",
        "n_samples": int(len(sample_ids)),
        "n_targets": int(len(targets)),
        "n_input_genes_available": int(rna.shape[1]),
        "n_input_genes_per_fold": int(args.n_genes),
        "folds": fold_infos,
    }
    with open(out / "logs" / "deepgxp_5fold_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    (out / "done.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
