#!/usr/bin/env python3
"""Reproduce the bulk DeepGxP RPPA model on the TCGA-TCPA matrix.

This script keeps the published bulk DeepGxP architecture as the comparator:
one 1D convolution, max pooling, a 512-unit dense layer, and a multi-antibody
linear output. TensorFlow is not available in the remote environment, so the
architecture is implemented in PyTorch with the same layer dimensions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/data/lsy/Infinite_Stream/01_data/tcga_tcpa/processed/tcpa_32_project_rna_rppa_20260501",
    )
    parser.add_argument(
        "--out-dir",
        default="/data/lsy/Infinite_Stream/02_results/model_validation/20260507_deepgxp_tcpa_reproduction",
    )
    parser.add_argument("--n-genes", type=int, default=13995)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
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


def load_tcpa(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_path = data_dir / "matrices" / "X_tcpa_32.symbols.parquet"
    y_path = data_dir / "matrices" / "Y_tcpa_32.rppa.parquet"
    panel_path = data_dir / "tables" / "tcpa_32_antibody_panel.tsv"
    manifest_path = data_dir / "tables" / "tcpa_32_sample_manifest.tsv"

    x_raw = pd.read_parquet(x_path)
    gene_symbol = x_raw["gene_symbol"].astype(str).str.upper()
    x_raw = x_raw.drop(columns=["gene_symbol"])
    x_raw.index = gene_symbol
    x_raw = x_raw[~x_raw.index.duplicated(keep="first")]
    x = x_raw.T
    x.index = x.index.astype(str)
    del x_raw

    y = pd.read_parquet(y_path)
    y = y.set_index("rna_sample_id")
    y.index = y.index.astype(str)
    y = y.apply(pd.to_numeric, errors="coerce")

    panel = pd.read_csv(panel_path, sep="\t")
    manifest = pd.read_csv(manifest_path, sep="\t")

    common = y.index.intersection(x.index)
    y = y.loc[common]
    x = x.loc[common]
    manifest = manifest.set_index("rna_sample_id").loc[common].reset_index()
    return x, y, panel, manifest


def pick_top_variable_genes(x_np: np.ndarray, train_idx: np.ndarray, n_genes: int) -> np.ndarray:
    var = np.nanvar(x_np[train_idx], axis=0)
    var = np.nan_to_num(var, nan=-1.0, posinf=-1.0, neginf=-1.0)
    n_keep = min(n_genes, x_np.shape[1])
    idx = np.argpartition(var, -n_keep)[-n_keep:]
    idx = idx[np.argsort(var[idx])[::-1]]
    return idx.astype(np.int64)


def zscore_by_train(
    values: np.ndarray,
    train_idx: np.ndarray,
    feature_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if feature_idx is not None:
        values = values[:, feature_idx]
    mean = np.nanmean(values[train_idx], axis=0)
    std = np.nanstd(values[train_idx], axis=0)
    std = np.where((std < 1e-6) | ~np.isfinite(std), 1.0, std)
    out = (values - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out, mean.astype(np.float32), std.astype(np.float32)


def make_val_split(train_full: np.ndarray, strata: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    local_y = strata[train_full]
    try:
        tr_local, val_local = next(splitter.split(np.zeros(len(train_full)), local_y))
    except ValueError:
        rng = np.random.default_rng(seed)
        order = rng.permutation(train_full)
        n_val = max(1, int(math.ceil(len(train_full) * 0.125)))
        return order[n_val:], order[:n_val]
    return train_full[tr_local], train_full[val_local]


def train_one_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_full: np.ndarray,
    test_idx: np.ndarray,
    strata: np.ndarray,
    fold: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    train_idx, val_idx = make_val_split(train_full, strata, args.seed + fold)
    feature_idx = pick_top_variable_genes(x, train_idx, args.n_genes)
    x_scaled, x_mean, x_std = zscore_by_train(x, train_idx, feature_idx)

    y_mean = np.nanmean(y[train_idx], axis=0)
    y_std = np.nanstd(y[train_idx], axis=0)
    y_std = np.where((y_std < 1e-6) | ~np.isfinite(y_std), 1.0, y_std)
    y_scaled = ((y - y_mean) / y_std).astype(np.float32)
    y_scaled = np.nan_to_num(y_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    model = DeepGxPBulkTorch(x_scaled.shape[1], y_scaled.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(x_scaled[train_idx]).unsqueeze(1),
        torch.from_numpy(y_scaled[train_idx]),
    )
    val_x = torch.from_numpy(x_scaled[val_idx]).unsqueeze(1).to(device)
    val_y = torch.from_numpy(y_scaled[val_idx]).to(device)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    best_epoch = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * xb.shape[0]
            n_seen += xb.shape[0]

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).detach().cpu())
        train_loss = train_loss_sum / max(n_seen, 1)
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
    test_ds = TensorDataset(torch.from_numpy(x_scaled[test_idx]).unsqueeze(1))
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device, non_blocking=True)
            pred_chunks.append(model(xb).detach().cpu().numpy())
    pred_scaled = np.vstack(pred_chunks)
    pred = pred_scaled * y_std + y_mean

    info = {
        "fold": fold,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_genes": int(len(feature_idx)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "history": history,
        "x_mean_mean": float(np.mean(x_mean)),
        "x_std_mean": float(np.mean(x_std)),
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


def build_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    panel: pd.DataFrame,
    min_n: int,
) -> pd.DataFrame:
    rows = []
    panel_map = panel.set_index("antibody")["type"].to_dict()
    for antibody in y_true.columns:
        true = y_true[antibody].to_numpy(dtype=np.float64)
        pred = y_pred[antibody].to_numpy(dtype=np.float64)
        rows.append(
            {
                "antibody": antibody,
                "type": panel_map.get(antibody, "unknown"),
                "n": int(np.isfinite(true).sum()),
                "pearson": corr_or_nan(true, pred, "pearson", min_n),
                "spearman": corr_or_nan(true, pred, "spearman", min_n),
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {}
    for label, sub in [("all", metrics), ("total", metrics[metrics["type"] == "total"]), ("phospho", metrics[metrics["type"] == "phospho"])]:
        out[label] = {
            "n_antibodies": int(len(sub)),
            "n_evaluable_pearson": int(sub["pearson"].notna().sum()),
            "median_pearson": float(np.nanmedian(sub["pearson"])) if len(sub) else float("nan"),
            "median_spearman": float(np.nanmedian(sub["spearman"])) if len(sub) else float("nan"),
            "iqr_pearson": [
                float(np.nanpercentile(sub["pearson"], 25)) if len(sub) else float("nan"),
                float(np.nanpercentile(sub["pearson"], 75)) if len(sub) else float("nan"),
            ],
        }
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    x_df, y_df, panel, manifest = load_tcpa(Path(args.data_dir))
    if args.smoke:
        keep = min(512, len(x_df))
        x_df = x_df.iloc[:keep]
        y_df = y_df.iloc[:keep]
        manifest = manifest.iloc[:keep]
        args.epochs = min(args.epochs, 3)
        args.n_genes = min(args.n_genes, 2000)
        args.max_folds = 1

    x_np = x_df.to_numpy(dtype=np.float32, copy=True)
    y_np = y_df.to_numpy(dtype=np.float32, copy=True)
    strata = manifest["project_id"].astype(str).to_numpy()

    splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    pred = np.full_like(y_np, np.nan, dtype=np.float32)
    fold_infos = []

    for fold, (train_full, test_idx) in enumerate(splitter.split(np.zeros(len(strata)), strata), start=1):
        if args.max_folds and fold > args.max_folds:
            break
        fold_pred, info = train_one_fold(x_np, y_np, train_full, test_idx, strata, fold, args, device)
        pred[test_idx] = fold_pred
        fold_infos.append(info)
        pd.DataFrame(info["history"]).to_csv(out_dir / "logs" / f"fold{fold}_history.tsv", sep="\t", index=False)
        print(json.dumps({k: v for k, v in info.items() if k != "history"}, ensure_ascii=False), flush=True)

    y_pred = pd.DataFrame(pred, index=y_df.index, columns=y_df.columns)
    metrics = build_metrics(y_df, y_pred, panel, args.min_n)
    summary = summarize_metrics(metrics)
    summary["run"] = {
        "model": "DeepGxP_bulk_reproduction_on_TCGA_TCPA",
        "source_repo": "https://github.com/hmtsai2024/DeepGxP_manuscript",
        "source_commit": "9814cd2dbc734b8c2d1d6d69d7ac4dbc86e1ec00",
        "implementation": "PyTorch equivalent of published Keras bulk CNN",
        "input_matrix": str(Path(args.data_dir) / "matrices" / "X_tcpa_32.symbols.parquet"),
        "output_matrix": str(Path(args.data_dir) / "matrices" / "Y_tcpa_32.rppa.parquet"),
        "n_samples": int(len(y_df)),
        "n_input_genes_available": int(x_df.shape[1]),
        "n_input_genes_per_fold": int(args.n_genes),
        "n_outputs": int(y_df.shape[1]),
        "split": f"{args.n_folds}-fold project-stratified OOF with train-internal validation for early stopping",
        "device": str(device),
        "folds_completed": int(len(fold_infos)),
        "epochs_max": int(args.epochs),
        "patience": int(args.patience),
    }
    summary["folds"] = [{k: v for k, v in info.items() if k != "history"} for info in fold_infos]

    metrics.to_csv(out_dir / "tables" / "metrics_by_antibody.tsv", sep="\t", index=False)
    y_pred.to_parquet(out_dir / "predictions" / "deepgxp_tcpa_oof_predictions.parquet")
    with open(out_dir / "logs" / "deepgxp_tcpa_reproduction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
