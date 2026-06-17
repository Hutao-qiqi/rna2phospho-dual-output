#!/usr/bin/env python3
"""Retrain a DeepGxP-style bulk CNN on half of CPTAC/PDC and test on the held-out half.

The original DeepGxP bulk model uses whole-transcriptome input, one Conv1D
layer, max pooling, a 512-unit dense layer, and a multi-target output head.
This script keeps that architecture but replaces the TCGA-TCPA RPPA target
matrix with CPTAC/PDC total protein or phosphosite matrices.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/data/lsy/Infinite_Stream/01_data/multi_omics/processed/pancancer_multi_task_locked_v2",
    )
    parser.add_argument(
        "--out-dir",
        default="/data/lsy/Infinite_Stream/02_results/model_validation/20260511_deepgxp_cptac_half_retrain",
    )
    parser.add_argument("--target", choices=["total", "phosphosite", "both"], default="both")
    parser.add_argument("--n-genes", type=int, default=13995)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--val-size-within-train", type=float, default=0.125)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=682)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-n", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


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


def load_matrices(data_dir: Path, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = pd.read_parquet(data_dir / "rna_log2_tpm_paired.parquet")
    if target == "total":
        y = pd.read_parquet(data_dir / "total_protein_gene_study_zscore_min20pct.parquet")
    elif target == "phosphosite":
        y = pd.read_parquet(data_dir / "phosphosite_gene_site_study_zscore_min20pct_targets.parquet")
    else:
        raise ValueError(target)
    manifest = pd.read_csv(data_dir / "sample_manifest.tsv", sep="\t")
    manifest = manifest.set_index("sample_id")
    common = x.index.astype(str).intersection(y.index.astype(str)).intersection(manifest.index.astype(str))
    x = x.loc[common].copy()
    y = y.loc[common].copy()
    manifest = manifest.loc[common].reset_index()
    return x, y, manifest


def make_split(strata: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(strata))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    try:
        train_full, test_idx = next(splitter.split(np.zeros(len(strata)), strata))
    except ValueError:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(all_idx)
        n_test = int(round(len(perm) * args.test_size))
        test_idx = perm[:n_test]
        train_full = perm[n_test:]

    val_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=args.val_size_within_train,
        random_state=args.seed + 1,
    )
    try:
        train_local, val_local = next(val_splitter.split(np.zeros(len(train_full)), strata[train_full]))
        train_idx = train_full[train_local]
        val_idx = train_full[val_local]
    except ValueError:
        rng = np.random.default_rng(args.seed + 1)
        perm = rng.permutation(train_full)
        n_val = max(1, int(math.ceil(len(train_full) * args.val_size_within_train)))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
    return train_idx, val_idx, test_idx


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


def train_model(
    x_np: np.ndarray,
    y_np: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    feature_idx = pick_top_variable_genes(x_np, train_idx, args.n_genes)
    x_scaled, x_mean, x_std = scale_x(x_np, train_idx, feature_idx)
    y_scaled, y_mask, y_mean, y_std = scale_y(y_np, train_idx)

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
        train_loss_sum = 0.0
        train_mask_sum = 0.0
        for xb, yb, mb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = masked_mse(pred, yb, mb)
            loss.backward()
            optimizer.step()
            mask_count = float(mb.sum().detach().cpu())
            train_loss_sum += float(loss.detach().cpu()) * mask_count
            train_mask_sum += mask_count

        model.eval()
        with torch.no_grad():
            val_loss = float(masked_mse(model(val_x), val_y, val_m).detach().cpu())
        train_loss = train_loss_sum / max(train_mask_sum, 1.0)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-7:
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
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_genes": int(len(feature_idx)),
        "n_outputs": int(y_np.shape[1]),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "x_mean_mean": float(np.mean(x_mean)),
        "x_std_mean": float(np.mean(x_std)),
        "target_train_coverage_mean": float(np.mean(y_mask[train_idx])),
        "target_test_coverage_mean": float(np.mean(y_mask[test_idx])),
        "history": history,
    }
    return pred.astype(np.float32), info


def corr_or_nan(a: np.ndarray, b: np.ndarray, method: str, min_n: int) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < min_n:
        return float("nan")
    if np.nanstd(a[mask]) < 1e-12 or np.nanstd(b[mask]) < 1e-12:
        return float("nan")
    if method == "pearson":
        return float(pearsonr(a[mask], b[mask]).statistic)
    return float(spearmanr(a[mask], b[mask]).statistic)


def build_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    for col in y_true.columns:
        true = y_true[col].to_numpy(dtype=np.float64)
        pred = y_pred[col].to_numpy(dtype=np.float64)
        finite = np.isfinite(true)
        rows.append(
            {
                "target": col,
                "n_test_observed": int(finite.sum()),
                "pearson": corr_or_nan(true, pred, "pearson", min_n),
                "spearman": corr_or_nan(true, pred, "spearman", min_n),
            }
        )
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> dict[str, object]:
    pearson = metrics["pearson"].to_numpy(dtype=float)
    spearman = metrics["spearman"].to_numpy(dtype=float)
    return {
        "n_targets": int(len(metrics)),
        "n_evaluable_pearson": int(np.isfinite(pearson).sum()),
        "median_pearson": float(np.nanmedian(pearson)),
        "median_spearman": float(np.nanmedian(spearman)),
        "iqr_pearson": [
            float(np.nanpercentile(pearson, 25)),
            float(np.nanpercentile(pearson, 75)),
        ],
        "r_gt_0_5": int(np.nansum(pearson > 0.5)),
        "r_gt_0_3": int(np.nansum(pearson > 0.3)),
        "r_gt_0_2": int(np.nansum(pearson > 0.2)),
    }


def run_target(target: str, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    x_df, y_df, manifest = load_matrices(Path(args.data_dir), target)
    if args.smoke:
        x_df = x_df.iloc[:256, : min(2000, x_df.shape[1])]
        y_df = y_df.iloc[:256, : min(256, y_df.shape[1])]
        manifest = manifest.iloc[:256]
        args.epochs = min(args.epochs, 3)
        args.n_genes = min(args.n_genes, 2000)

    strata = manifest["cancer_label"].astype(str).to_numpy()
    train_idx, val_idx, test_idx = make_split(strata, args)
    if args.max_targets and args.max_targets < y_df.shape[1]:
        train_coverage = y_df.iloc[train_idx].notna().sum(axis=0).to_numpy()
        train_variance = np.nanvar(y_df.iloc[train_idx].to_numpy(dtype=np.float32), axis=0)
        order = np.lexsort((-np.nan_to_num(train_variance, nan=-1.0), -train_coverage))
        keep_cols = y_df.columns[order[: args.max_targets]]
        y_df = y_df.loc[:, keep_cols]
    x_np = x_df.to_numpy(dtype=np.float32, copy=True)
    y_np = y_df.to_numpy(dtype=np.float32, copy=True)

    pred, info = train_model(x_np, y_np, train_idx, val_idx, test_idx, args, device)
    test_y = y_df.iloc[test_idx].copy()
    test_pred = pd.DataFrame(pred, index=test_y.index, columns=test_y.columns)

    metrics = build_metrics(test_y, test_pred, args.min_n)
    summary = summarize(metrics)
    summary["target_layer"] = target
    summary["run"] = {
        "model": "DeepGxP_bulk_CNN_retrained_on_half_CPTAC_PDC",
        "source_repo": "https://github.com/hmtsai2024/DeepGxP_manuscript",
        "article": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12972970/",
        "implementation": "PyTorch equivalent of published Keras bulk CNN with masked target loss for CPTAC missingness",
        "input_matrix": str(Path(args.data_dir) / "rna_log2_tpm_paired.parquet"),
        "output_matrix": str(
            Path(args.data_dir)
            / (
                "total_protein_gene_study_zscore_min20pct.parquet"
                if target == "total"
                else "phosphosite_gene_site_study_zscore_min20pct_targets.parquet"
            )
        ),
        "n_samples": int(len(y_df)),
        "n_input_genes_available": int(x_df.shape[1]),
        "n_input_genes_selected": int(info["n_genes"]),
        "split": "single random 50:50 cancer-stratified train/test split with train-internal validation",
        "device": str(device),
        "epochs_max": int(args.epochs),
        "patience": int(args.patience),
    }
    summary["fit"] = {k: v for k, v in info.items() if k != "history"}

    metrics.to_csv(out_dir / "tables" / f"deepgxp_cptac_half_{target}_test_metrics.tsv", sep="\t", index=False)
    test_pred.to_parquet(out_dir / "predictions" / f"deepgxp_cptac_half_{target}_test_predictions.parquet")
    pd.DataFrame(info["history"]).to_csv(out_dir / "logs" / f"deepgxp_cptac_half_{target}_history.tsv", sep="\t", index=False)
    with open(out_dir / "logs" / f"deepgxp_cptac_half_{target}_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    targets = ["total", "phosphosite"] if args.target == "both" else [args.target]
    summaries = {target: run_target(target, args, device) for target in targets}
    with open(out_dir / "logs" / "deepgxp_cptac_half_retrain_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
