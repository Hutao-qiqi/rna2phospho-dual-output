from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn


DATASET = "gse300551_iccite_plex_kinase_2025"


def normalize_log1p(matrix):
    matrix = matrix.tocsr().astype(np.float32)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    matrix = sp.diags((1e4 / np.maximum(totals, 1.0)).astype(np.float32)) @ matrix
    matrix.data = np.log1p(matrix.data)
    return matrix.tocsr()


def finite_spearman(x, y):
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.std(x[keep]) == 0 or np.std(y[keep]) == 0:
        return np.nan
    return float(spearmanr(x[keep], y[keep]).statistic)


def residualize(values, groups):
    result = values.copy()
    for group in np.unique(groups):
        rows = groups == group
        result[rows] -= np.mean(result[rows], axis=0)
    return result


def metrics(prediction, truth, groups, targets):
    residual_prediction = residualize(prediction, groups)
    residual_truth = residualize(truth, groups)
    group_counts = pd.Series(groups).value_counts()
    valid_groups = group_counts[group_counts >= 5].index.to_numpy()
    pseudo_prediction = np.vstack([prediction[groups == group].mean(0) for group in valid_groups])
    pseudo_truth = np.vstack([truth[groups == group].mean(0) for group in valid_groups])
    rows = []
    for j, target in enumerate(targets):
        rows.append(
            {
                "target_id": target,
                "overall_spearman": finite_spearman(truth[:, j], prediction[:, j]),
                "within_condition_spearman": finite_spearman(
                    residual_truth[:, j], residual_prediction[:, j]
                ),
                "pseudobulk_spearman": finite_spearman(
                    pseudo_truth[:, j], pseudo_prediction[:, j]
                ),
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "median_overall_spearman": float(table["overall_spearman"].median()),
        "median_within_condition_spearman": float(table["within_condition_spearman"].median()),
        "median_pseudobulk_spearman": float(table["pseudobulk_spearman"].median()),
    }
    return summary, table


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout=0.10):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class DirectModel(nn.Module):
    def __init__(self, n_genes, embedding_dim, n_targets, variant):
        super().__init__()
        self.variant = variant
        self.hvg = nn.Sequential(
            nn.Linear(n_genes, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            ResidualBlock(2048),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            ResidualBlock(1024),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            ResidualBlock(512),
        )
        if variant in {"A2", "A3"}:
            self.embedding = nn.Sequential(
                nn.Linear(embedding_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                ResidualBlock(512),
            )
            self.fusion = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                ResidualBlock(512),
            )
        if variant == "A3":
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(512, 256),
                        nn.GELU(),
                        nn.Linear(256, 64),
                        nn.GELU(),
                        nn.Linear(64, 1),
                    )
                    for _ in range(n_targets)
                ]
            )
        else:
            self.shared_head = nn.Sequential(
                nn.Linear(512, 256), nn.GELU(), nn.Linear(256, n_targets)
            )

    def forward(self, genes, embedding):
        latent = self.hvg(genes)
        if self.variant in {"A2", "A3"}:
            latent = self.fusion(torch.cat([latent, self.embedding(embedding)], dim=1))
        if self.variant == "A3":
            output = torch.cat([head(latent) for head in self.heads], dim=1)
        else:
            output = self.shared_head(latent)
        return output, latent


class ConditionBatcher:
    def __init__(self, rows, conditions, plates, seed, n_conditions=24, cells_per_plate=4):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.conditions = conditions
        self.plates = plates
        self.rng = np.random.default_rng(seed)
        self.n_conditions = n_conditions
        self.cells_per_plate = cells_per_plate
        self.groups = {}
        for condition in np.unique(conditions[self.rows]):
            by_plate = {}
            condition_rows = self.rows[conditions[self.rows] == condition]
            for plate in np.unique(plates[condition_rows]):
                selected = condition_rows[plates[condition_rows] == plate]
                if len(selected):
                    by_plate[plate] = selected
            if by_plate:
                self.groups[condition] = by_plate
        self.keys = np.asarray(list(self.groups))

    def sample(self):
        chosen = self.rng.choice(
            self.keys, size=min(self.n_conditions, len(self.keys)), replace=False
        )
        batch = []
        for condition in chosen:
            for rows in self.groups[condition].values():
                batch.extend(
                    self.rng.choice(
                        rows, size=self.cells_per_plate, replace=len(rows) < self.cells_per_plate
                    ).tolist()
                )
        self.rng.shuffle(batch)
        return np.asarray(batch, dtype=np.int64)


def correlation_loss(prediction, truth):
    p = prediction - prediction.mean(0, keepdim=True)
    y = truth - truth.mean(0, keepdim=True)
    corr = (p * y).sum(0) / (p.square().sum(0).sqrt() * y.square().sum(0).sqrt()).clamp_min(1e-6)
    return (1.0 - corr).mean()


def rank_loss(prediction, truth):
    permutation = torch.randperm(len(prediction), device=prediction.device)
    truth_delta = truth - truth[permutation]
    prediction_delta = prediction - prediction[permutation]
    direction = torch.sign(truth_delta)
    valid = truth_delta.abs() > 0.05
    if not bool(valid.any()):
        return prediction.sum() * 0
    return F.softplus(-direction[valid] * prediction_delta[valid] / 0.5).mean()


def grouped_losses(prediction, truth, latent, conditions, plates):
    pseudo, centered_prediction, centered_truth, alignment = [], prediction.clone(), truth.clone(), []
    for condition in torch.unique(conditions):
        rows = conditions == condition
        pseudo.append(F.huber_loss(prediction[rows].mean(0), truth[rows].mean(0), delta=0.5))
        centered_prediction[rows] -= prediction[rows].mean(0, keepdim=True)
        centered_truth[rows] -= truth[rows].mean(0, keepdim=True)
        condition_latent = latent[rows]
        condition_plates = plates[rows]
        if len(torch.unique(condition_plates)) > 1:
            center = condition_latent.mean(0)
            for plate in torch.unique(condition_plates):
                alignment.append((condition_latent[condition_plates == plate].mean(0) - center).square().mean())
    pseudo_loss = torch.stack(pseudo).mean()
    within_loss = correlation_loss(centered_prediction, centered_truth)
    alignment_loss = torch.stack(alignment).mean() if alignment else prediction.sum() * 0
    return pseudo_loss, within_loss, alignment_loss


def top_hvgs(matrix, train, n_genes):
    current = matrix[train]
    mean = np.asarray(current.mean(0)).ravel()
    second = np.asarray(current.power(2).mean(0)).ravel()
    variance = np.maximum(second - mean**2, 0)
    return np.argsort(variance)[-min(n_genes, int((variance > 0).sum())) :].astype(np.int64)


def dense_batch(matrix, rows, genes, mean, std):
    values = matrix[rows][:, genes].toarray().astype(np.float32)
    return (values - mean) / std


@torch.no_grad()
def predict(model, matrix, embedding, rows, genes, mean, std, device, batch_size=512):
    model.eval()
    result = []
    for start in range(0, len(rows), batch_size):
        take = rows[start : start + batch_size]
        x = dense_batch(matrix, take, genes, mean, std)
        e = np.asarray(embedding[take], dtype=np.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output, _ = model(torch.as_tensor(x, device=device), torch.as_tensor(e, device=device))
        result.append(output.float().cpu().numpy())
    return np.vstack(result).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["A1", "A2", "A3"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--n-genes", type=int, default=4000)
    args = parser.parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(0.68, device)
    manifest = pd.read_csv(args.contract_dir / "gse300551_cell_manifest.tsv", sep="\t")
    panel = pd.read_csv(args.contract_dir / "full20_panel.tsv", sep="\t")
    contract = pd.read_csv(args.contract_dir / "target_split_contract.tsv", sep="\t")
    split_seed = args.seed if args.split_seed is None else args.split_seed
    split = contract[contract["seed"].eq(split_seed)]
    opt = split.loc[split["split"].eq("optimization"), "dataset_cell_index"].to_numpy(np.int64)
    val = split.loc[split["split"].eq("validation"), "dataset_cell_index"].to_numpy(np.int64)
    test = split.loc[split["split"].eq("test"), "dataset_cell_index"].to_numpy(np.int64)
    input_dir = args.root / r"01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_gse300551_signal_seq_multidomain_v1"
    global_indices = manifest["global_cell_index"].to_numpy(np.int64)
    target_indices = panel["target_index"].to_numpy(np.int64)
    raw = np.asarray(
        np.load(input_dir / "targets.npy", mmap_mode="r")[np.ix_(global_indices, target_indices)],
        dtype=np.float32,
    )
    target_mean = np.mean(raw[opt], axis=0).astype(np.float32)
    target_std = np.std(raw[opt], axis=0).astype(np.float32)
    target_std[target_std < 1e-6] = 1.0
    target = ((raw - target_mean) / target_std).astype(np.float32)
    embedding = np.load(input_dir / "embeddings.npy", mmap_mode="r")[global_indices]
    rna = ad.read_h5ad(
        args.root / r"01_data\single_cell\intermediate\foundation_model_h5ad_inputs_v1\gse300551_iccite_plex_kinase_2025.h5ad"
    )
    lookup = {str(cell): i for i, cell in enumerate(rna.obs_names.astype(str))}
    positions = np.asarray([lookup[str(cell)] for cell in manifest["cell_id"].astype(str)], dtype=np.int64)
    matrix = normalize_log1p(rna.X[positions])
    genes = top_hvgs(matrix, opt, args.n_genes)
    train_matrix = matrix[opt][:, genes]
    gene_mean = np.asarray(train_matrix.mean(0)).ravel().astype(np.float32)
    gene_second = np.asarray(train_matrix.power(2).mean(0)).ravel().astype(np.float32)
    gene_std = np.sqrt(np.maximum(gene_second - gene_mean**2, 1e-6)).astype(np.float32)
    merged = pd.read_csv(args.metadata, sep="\t", low_memory=False).set_index("dataset_cell_index")
    merged = merged.loc[np.arange(len(manifest))]
    condition_codes = pd.factorize(merged["condition_group"].astype(str))[0].astype(np.int64)
    plate_codes = pd.factorize(merged["sequencing_plate"].astype(str))[0].astype(np.int64)
    model = DirectModel(len(genes), embedding.shape[1], len(target_indices), args.variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    batcher = ConditionBatcher(opt, condition_codes, plate_codes, args.seed)
    steps = args.steps_per_epoch or max(20, math.ceil(len(opt) / 288))
    best, best_score, stale, history = None, -np.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(steps):
            rows = batcher.sample()
            x = dense_batch(matrix, rows, genes, gene_mean, gene_std)
            e = np.asarray(embedding[rows], dtype=np.float32)
            y = torch.as_tensor(target[rows], device=device)
            c = torch.as_tensor(condition_codes[rows], device=device)
            p = torch.as_tensor(plate_codes[rows], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction, latent = model(torch.as_tensor(x, device=device), torch.as_tensor(e, device=device))
                huber = F.huber_loss(prediction, y, delta=0.5)
                corr_loss = correlation_loss(prediction.float(), y)
                ordering = rank_loss(prediction.float(), y)
                pseudo, within, alignment = grouped_losses(prediction.float(), y, latent.float(), c, p)
                loss = huber + 0.25 * corr_loss + 0.15 * ordering + 0.20 * pseudo + 0.15 * within + 0.02 * alignment
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_prediction = predict(model, matrix, embedding, val, genes, gene_mean, gene_std, device)
        val_summary, _ = metrics(
            val_prediction, target[val], condition_codes[val], panel["target_id"].astype(str).tolist()
        )
        score = val_summary["median_overall_spearman"]
        history.append({"epoch": epoch, "train_loss": np.mean(losses), **{f"validation_{k}": v for k, v in val_summary.items()}})
        if score > best_score + 1e-5:
            best_score, stale = score, 0
            best = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if epoch >= 15 and stale >= 8:
                break
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best)
    validation_prediction = predict(model, matrix, embedding, val, genes, gene_mean, gene_std, device)
    test_prediction = predict(model, matrix, embedding, test, genes, gene_mean, gene_std, device)
    test_summary, test_table = metrics(
        test_prediction, target[test], condition_codes[test], panel["target_id"].astype(str).tolist()
    )
    pd.DataFrame(history).to_csv(args.output_dir / "training.tsv", sep="\t", index=False)
    test_table.to_csv(args.output_dir / "test_per_target.tsv", sep="\t", index=False)
    np.savez_compressed(
        args.output_dir / "validation_predictions.npz",
        dataset_cell_index=val,
        prediction=validation_prediction,
        truth=target[val],
        target_ids=panel["target_id"].astype(str).to_numpy(),
    )
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        dataset_cell_index=test,
        prediction=test_prediction,
        truth=target[test],
        target_ids=panel["target_id"].astype(str).to_numpy(),
    )
    np.save(args.output_dir / "hvg_indices.npy", genes)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "variant": args.variant,
            "seed": args.seed,
            "split_seed": split_seed,
        },
        args.output_dir / "best_direct_checkpoint.pt",
    )
    report = {
        "variant": args.variant,
        "seed": args.seed,
        "split_seed": split_seed,
        "best_validation_spearman": best_score,
        **test_summary,
        "epochs": len(history),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "SUCCESS").write_text("ok\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
