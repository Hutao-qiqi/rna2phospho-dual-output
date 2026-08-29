from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scp682_sc14_profile_benchmark import DATASET, detailed_metrics, finite_corr, row_minmax


def normalize_log1p(matrix):
    matrix = matrix.tocsr().astype(np.float32)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    factors = 1e4 / np.maximum(totals, 1.0)
    matrix = sp.diags(factors.astype(np.float32)) @ matrix
    matrix.data = np.log1p(matrix.data)
    return matrix.tocsr()


def top_variable_genes(matrix, train_rows, n_genes):
    current = matrix[train_rows]
    mean = np.asarray(current.mean(axis=0)).ravel()
    second = np.asarray(current.power(2).mean(axis=0)).ravel()
    variance = np.maximum(second - mean**2, 0)
    n = min(n_genes, int((variance > 0).sum()))
    return np.argsort(variance)[-n:].astype(np.int64)


class CTPNet(nn.Module):
    def __init__(self, n_input, n_output, target_mean):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(n_input, 1000), nn.ReLU(), nn.Linear(1000, 128), nn.ReLU())
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)) for _ in range(n_output)])
        for j, head in enumerate(self.heads):
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, float(target_mean[j]))

    def forward(self, x):
        state = self.shared(x)
        return torch.cat([head(state) for head in self.heads], dim=1)


class InputBlock(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_input),
            nn.Dropout(0.25),
            nn.Linear(n_input, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(0.25),
        )

    def forward(self, x):
        return self.net(x)


class FFBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.PReLU(), nn.Dropout(0.25))

    def forward(self, x):
        return self.net(x)


class SciPENN(nn.Module):
    def __init__(self, n_input, n_output, target_mean):
        super().__init__()
        self.input = InputBlock(n_input)
        self.rnn = nn.RNNCell(512, 512)
        self.blocks = nn.ModuleList([FFBlock() for _ in range(3)])
        self.output = nn.Linear(512, n_output)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias.copy_(torch.as_tensor(target_mean, dtype=torch.float32))

    def forward(self, x):
        state_input = self.input(x)
        state = self.rnn(state_input, torch.zeros_like(state_input))
        for block in self.blocks:
            state_input = block(state_input)
            state = self.rnn(state_input, state)
        return self.output(state)


class BABEL(nn.Module):
    def __init__(self, n_input, n_output, target_mean):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Linear(64, n_output),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        with torch.no_grad():
            self.decoder[-1].bias.copy_(torch.as_tensor(target_mean, dtype=torch.float32))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def dense_rows(matrix, rows, genes, mean, std):
    values = matrix[rows][:, genes].toarray().astype(np.float32)
    return (values - mean) / std


@torch.no_grad()
def predict(model, matrix, rows, genes, mean, std, device, batch_size):
    model.eval()
    outputs = []
    for start in range(0, len(rows), batch_size):
        x = dense_rows(matrix, rows[start : start + batch_size], genes, mean, std)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs.append(model(torch.as_tensor(x, device=device)).float().cpu().numpy())
    return np.vstack(outputs).astype(np.float32)


def median_site_spearman(prediction, truth, mask):
    values = []
    for j in range(truth.shape[1]):
        valid = mask[:, j] & np.isfinite(prediction[:, j]) & np.isfinite(truth[:, j])
        if valid.sum() >= 3:
            value = finite_corr(truth[valid, j], prediction[valid, j], "spearman")
            if np.isfinite(value):
                values.append(value)
    return float(np.median(values)) if values else -np.inf


def fit(architecture, matrix, y, mask, opt, val, device, seed, n_genes, batch_size, epochs):
    genes = top_variable_genes(matrix, opt, n_genes)
    train_values = matrix[opt][:, genes]
    mean = np.asarray(train_values.mean(axis=0)).ravel().astype(np.float32)
    second = np.asarray(train_values.power(2).mean(axis=0)).ravel().astype(np.float32)
    std = np.sqrt(np.maximum(second - mean**2, 1e-6)).astype(np.float32)
    target_mean = np.nanmean(np.where(mask[opt], y[opt], np.nan), axis=0).astype(np.float32)
    cls = {"ctpnet": CTPNet, "scipenn": SciPENN, "babel": BABEL}[architecture]
    model = cls(len(genes), y.shape[1], target_mean).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial = predict(model, matrix, val, genes, mean, std, device, batch_size)
    best_score = median_site_spearman(initial, y[val], mask[val])
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = [{"epoch": 0, "train_loss": math.nan, "validation_median_site_spearman": best_score}]
    stale = 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(opt)
        losses = []
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            x = dense_rows(matrix, rows, genes, mean, std)
            target = torch.as_tensor(y[rows], dtype=torch.float32, device=device)
            valid = torch.as_tensor(mask[rows], dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                estimate = model(torch.as_tensor(x, device=device))
                mse = (estimate.sub(target).square()[valid]).mean()
                estimate_masked = torch.where(valid, estimate, torch.zeros_like(estimate))
                target_masked = torch.where(valid, target, torch.zeros_like(target))
                loss = mse + 0.25 * (1.0 - F.cosine_similarity(estimate_masked, target_masked, dim=1).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        estimate = predict(model, matrix, val, genes, mean, std, device, batch_size)
        score = median_site_spearman(estimate, y[val], mask[val])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_median_site_spearman": score})
        if score > best_score + 1e-5:
            best_score = score
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(best)
    return model, genes, mean, std, pd.DataFrame(history), best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=["ctpnet", "scipenn", "babel"], required=True)
    parser.add_argument("--seeds", default=",".join(str(x) for x in range(68301, 68311)))
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-genes", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--target-mode", choices=["profile_minmax", "site_zscore"], default="profile_minmax")
    args = parser.parse_args()
    for sub in ("tables", "logs", "predictions"):
        (args.output_dir / sub).mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.contract_dir / "gse300551_cell_manifest.tsv", sep="\t", low_memory=False)
    panel = pd.read_csv(args.contract_dir / "full20_panel.tsv", sep="\t")
    contract = pd.read_csv(args.contract_dir / "target_split_contract.tsv", sep="\t")
    target_indices = panel["target_index"].astype(int).to_numpy()
    global_indices = manifest["global_cell_index"].astype(int).to_numpy()
    input_dir = args.root / r"01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_gse300551_signal_seq_multidomain_v1"
    raw = np.asarray(np.load(input_dir / "targets.npy", mmap_mode="r")[np.ix_(global_indices, target_indices)], dtype=np.float32)
    mask = np.asarray(np.load(input_dir / "target_mask.npy", mmap_mode="r")[np.ix_(global_indices, target_indices)], dtype=bool) & np.isfinite(raw)
    rna_path = args.root / r"01_data\single_cell\intermediate\foundation_model_h5ad_inputs_v1\gse300551_iccite_plex_kinase_2025.h5ad"
    rna = ad.read_h5ad(rna_path)
    lookup = {str(cell): i for i, cell in enumerate(rna.obs_names.astype(str))}
    positions = np.asarray([lookup[str(cell)] for cell in manifest["cell_id"].astype(str)], dtype=np.int64)
    matrix = normalize_log1p(rna.X[positions])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    summaries, sites_all, cells_all = [], [], []
    started = time.time()
    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        split_seed = seed if args.split_seed is None else args.split_seed
        subset = contract[(contract["dataset_id"] == DATASET) & (contract["seed"] == split_seed)]
        opt = subset.loc[subset["split"] == "optimization", "dataset_cell_index"].to_numpy(np.int64)
        val = subset.loc[subset["split"] == "validation", "dataset_cell_index"].to_numpy(np.int64)
        test = subset.loc[subset["split"] == "test", "dataset_cell_index"].to_numpy(np.int64)
        if args.target_mode == "site_zscore":
            train_values = np.where(mask[opt], raw[opt], np.nan)
            target_mean = np.nanmean(train_values, axis=0).astype(np.float32)
            target_scale = np.nanstd(train_values, axis=0).astype(np.float32)
            target_mean[~np.isfinite(target_mean)] = 0.0
            target_scale[~np.isfinite(target_scale) | (target_scale < 1e-6)] = 1.0
            y = ((raw - target_mean) / target_scale).astype(np.float32)
        else:
            target_mean = np.zeros(raw.shape[1], dtype=np.float32)
            target_scale = np.ones(raw.shape[1], dtype=np.float32)
            y = row_minmax(raw, mask)
        model, genes, mean, std, history, val_score = fit(
            args.architecture, matrix, y, mask, opt, val, device, seed, args.n_genes, args.batch_size, args.epochs
        )
        prediction = predict(model, matrix, test, genes, mean, std, device, args.batch_size)
        if args.target_mode == "site_zscore":
            prediction_native = prediction * target_scale + target_mean
            truth_native = raw[test]
        else:
            prediction_native = prediction
            truth_native = y[test]
        summary, cells, sites, prediction_n, truth_n = detailed_metrics(
            prediction_native, truth_native, mask[test], panel
        )
        summary.update(method=args.architecture, model_type=args.architecture, seed=seed, split_seed=split_seed, validation_best_official_cosine=val_score)
        summaries.append(summary)
        cells.insert(0, "seed", seed); cells.insert(0, "method", args.architecture)
        sites.insert(0, "seed", seed); sites.insert(0, "method", args.architecture)
        cells_all.append(cells); sites_all.append(sites)
        history.to_csv(args.output_dir / "logs" / f"seed_{seed}_training.tsv", sep="\t", index=False)
        np.save(args.output_dir / "predictions" / f"seed_{seed}_hvg_indices.npy", genes)
        np.savez_compressed(
            args.output_dir / "predictions" / f"seed_{seed}_test_predictions.npz",
            dataset_cell_index=test,
            prediction=prediction_n,
            truth=truth_n,
            prediction_native=prediction_native,
            truth_native=truth_native,
            mask=mask[test],
        )
        (args.output_dir / f"seed_{seed}_SUCCESS").write_text("ok\n", encoding="utf-8")
        print(json.dumps(summary), flush=True)
    frame = pd.DataFrame(summaries)
    frame.to_csv(args.output_dir / "tables" / "summary_by_seed.tsv", sep="\t", index=False)
    pd.concat(cells_all, ignore_index=True).to_csv(args.output_dir / "tables" / "per_cell.tsv", sep="\t", index=False)
    pd.concat(sites_all, ignore_index=True).to_csv(args.output_dir / "tables" / "per_site.tsv", sep="\t", index=False)
    aggregate = {"method": args.architecture, "model_type": args.architecture, "n_seeds": len(frame), "elapsed_seconds": time.time() - started}
    for column in ["official_batch4_flatten_cosine", "official_batch4_flatten_mse", "official_batch4_flatten_mae", "median_cell_cosine", "median_cell_pearson", "mean_normalized_mse", "median_site_spearman", "median_site_pearson", "median_site_spearman_after_cell_minmax", "median_site_pearson_after_cell_minmax"]:
        aggregate[column] = float(frame[column].mean())
        aggregate[column + "_sd"] = float(frame[column].std(ddof=1))
    (args.output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (args.output_dir / "SUCCESS").write_text("ok\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
