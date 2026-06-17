#!/usr/bin/env python3
"""Train CPTAC total proteome predictor with RNA, VAE z, FiLM, and direct RNA residuals."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path("/data/lsy/Infinite_Stream")
DATA_DIR = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2"
OUT = ROOT / "02_results/model_validation/20260429_cptac_total_proteome_film_vae_z_direct_residual_v2"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, latent_dim)
        self.logvar_head = nn.Linear(prev, latent_dim)
        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec.append(nn.Linear(prev, h))
            dec.append(nn.ReLU())
            if dropout > 0:
                dec.append(nn.Dropout(dropout))
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), torch.clamp(self.logvar_head(h), min=-6.0, max=2.0)


class FiLMBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cond_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.film = nn.Linear(cond_dim, out_dim * 2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.linear(x))
        gamma, beta = self.film(cond).chunk(2, dim=1)
        h = h * (1.0 + 0.12 * torch.tanh(gamma)) + 0.12 * beta
        return self.drop(self.act(h))


class TotalProteomeFiLMDirectResidual(nn.Module):
    def __init__(self, n_input: int, n_target: int, n_cancer: int, n_study: int, hidden: int, cond_dim: int, dropout: float) -> None:
        super().__init__()
        self.cancer_emb = nn.Embedding(n_cancer, cond_dim // 2)
        self.study_emb = nn.Embedding(n_study, cond_dim // 2)
        self.in_block = FiLMBlock(n_input, hidden, cond_dim, dropout)
        self.block2 = FiLMBlock(hidden, hidden, cond_dim, dropout)
        self.block3 = FiLMBlock(hidden, hidden, cond_dim, dropout)
        self.block4 = FiLMBlock(hidden, hidden // 2, cond_dim, dropout)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_target),
        )
        self.direct_global_scale = nn.Parameter(torch.zeros(n_target))
        self.direct_cancer_scale = nn.Embedding(n_cancer, n_target)
        self.direct_bias = nn.Parameter(torch.zeros(n_target))
        nn.init.zeros_(self.direct_cancer_scale.weight)

    def cond(self, cancer: torch.Tensor, study: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.cancer_emb(cancer), self.study_emb(study)], dim=1)

    def forward(self, x: torch.Tensor, cancer: torch.Tensor, study: torch.Tensor, direct_rna: torch.Tensor, direct_mask: torch.Tensor) -> torch.Tensor:
        cond = self.cond(cancer, study)
        h1 = self.in_block(x, cond)
        h2 = self.block2(h1, cond) + h1
        h3 = self.block3(h2, cond) + h2
        h4 = self.block4(h3, cond)
        base = self.out(h4)
        scale = self.direct_global_scale[None, :] + 0.12 * self.direct_cancer_scale(cancer)
        return base + direct_mask * (direct_rna * scale + self.direct_bias[None, :])


def encode_vae_z(rna_df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    payload = torch.load(ROOT / "models_vae/vae_weights.pt", map_location=device, weights_only=False)
    tcga_gene_order = pd.read_parquet(ROOT / "data/processed/X_all.symbols.parquet", columns=["gene_symbol"])
    tcga_gene_order = tcga_gene_order["gene_symbol"].drop_duplicates().astype(str).tolist()
    aligned = rna_df.reindex(columns=tcga_gene_order).fillna(0.0)

    mean = np.asarray(payload["input_scaler_mean"], dtype=np.float32)
    scale = np.asarray(payload["input_scaler_scale"], dtype=np.float32)
    scale = np.where(scale == 0, 1.0, scale)
    x = (aligned.to_numpy(dtype=np.float32) - mean) / scale
    x = np.nan_to_num(x, nan=0.0, posinf=8.0, neginf=-8.0)
    x = np.clip(x, -8.0, 8.0)

    model = VAE(
        input_dim=int(payload["input_dim"]),
        latent_dim=int(payload["latent_dim"]),
        hidden_dims=list(payload["hidden_dims"]),
        dropout=float(payload["dropout"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    latents = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 256):
            xb = torch.tensor(x[start : start + 256], dtype=torch.float32, device=device)
            mu, _ = model.encode(xb)
            latents.append(mu.detach().cpu().numpy())
    z = np.vstack(latents).astype(np.float32)
    latent_mean = np.asarray(payload["latent_scaler_mean"], dtype=np.float32)
    latent_scale = np.asarray(payload["latent_scaler_scale"], dtype=np.float32)
    latent_scale = np.where(latent_scale == 0, 1.0, latent_scale)
    z = (z - latent_mean) / latent_scale
    return pd.DataFrame(z, index=rna_df.index, columns=[f"VAE__vae_z_{i:03d}" for i in range(z.shape[1])])


def standardize_x(x_train: np.ndarray, x_all: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.where(np.isfinite(x_train), x_train, np.nan)
    all_x = np.where(np.isfinite(x_all), x_all, np.nan)
    fill = np.nanmedian(train, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    all_filled = np.where(np.isfinite(all_x), all_x, fill)
    mean = np.mean(np.where(np.isfinite(train), train, fill), axis=0)
    std = np.std(np.where(np.isfinite(train), train, fill), axis=0)
    std = np.where((std > 1e-6) & np.isfinite(std), std, 1.0)
    return ((all_filled - mean) / std).astype(np.float32), fill.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def standardize_y(y_train: np.ndarray, y_all: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(y_train, axis=0)
    std = np.nanstd(y_train, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where((std > 1e-6) & np.isfinite(std), std, 1.0)
    return ((y_all - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((pred - y) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def masked_centered_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pred_c = pred - (pred * mask).sum(dim=1, keepdim=True) / denom
    y_c = y - (y * mask).sum(dim=1, keepdim=True) / denom
    return masked_mse(pred_c, y_c, mask)


def build_direct_matrix(x_z: np.ndarray, feature_names: list[str], target_names: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feature_idx = {g: i for i, g in enumerate(feature_names)}
    direct = np.zeros((x_z.shape[0], len(target_names)), dtype=np.float32)
    mask = np.zeros((x_z.shape[0], len(target_names)), dtype=np.float32)
    rows = []
    for j, target in enumerate(target_names):
        gene = str(target)
        if gene in feature_idx:
            direct[:, j] = x_z[:, feature_idx[gene]]
            mask[:, j] = 1.0
            mapped = True
        else:
            mapped = False
        rows.append({"protein_gene": gene, "direct_gene_symbol": gene if mapped else None, "mapped": mapped})
    return direct, mask, pd.DataFrame(rows)


def spearman_by_target(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(names):
        ok = np.isfinite(y_true[:, j]) & np.isfinite(y_pred[:, j])
        rho = np.nan
        p = np.nan
        if int(ok.sum()) >= 10:
            r = spearmanr(y_true[ok, j], y_pred[ok, j], nan_policy="omit")
            rho = float(r.correlation)
            p = float(r.pvalue)
        rows.append({"protein_gene": name, "n": int(ok.sum()), "spearman": rho, "p_value": p})
    return pd.DataFrame(rows)


def sample_metrics(y_df: pd.DataFrame, pred_df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid in y_df.index:
        ok = y_df.loc[sid].notna() & pred_df.loc[sid].notna()
        rho = np.nan
        if int(ok.sum()) >= 10:
            rho = y_df.loc[sid, ok].rank().corr(pred_df.loc[sid, ok].rank())
        rows.append({"sample_id": sid, "n_total_proteins": int(ok.sum()), "sample_spearman": rho})
    return pd.DataFrame(rows).merge(manifest.reset_index()[["sample_id", "pdc_study_id", "cancer_label"]], on="sample_id", how="left")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--cond-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.14)
    parser.add_argument("--lr", type=float, default=1.8e-4)
    parser.add_argument("--weight-decay", type=float, default=8e-5)
    parser.add_argument("--center-loss-weight", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seed_all(args.seed)
    for sub in ["models", "predictions", "tables", "logs"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("GPU is required for this run")
    print("device", device, flush=True)
    print("gpu", torch.cuda.get_device_name(device), flush=True)

    rna_df = pd.read_parquet(DATA_DIR / "rna_log2_tpm_paired.parquet")
    protein_df = pd.read_parquet(DATA_DIR / "total_protein_gene_study_zscore_min20pct.parquet").loc[rna_df.index]
    manifest = pd.read_csv(DATA_DIR / "sample_manifest.tsv", sep="\t").set_index("sample_id").loc[rna_df.index]
    manifest.index.name = "sample_id"

    z_df = encode_vae_z(rna_df, device)
    x_df = pd.concat([rna_df, z_df], axis=1)
    feature_names = list(x_df.columns)
    target_names = list(protein_df.columns)
    x = x_df.to_numpy(dtype=np.float32)
    y = protein_df.to_numpy(dtype=np.float32)
    sample_ids = rna_df.index.to_numpy()

    cancer_cat = manifest["cancer_label"].astype("category")
    study_cat = manifest["pdc_study_id"].astype("category")
    cancer_ids = cancer_cat.cat.codes.to_numpy(dtype=np.int64)
    study_ids = study_cat.cat.codes.to_numpy(dtype=np.int64)
    cancer_levels = list(cancer_cat.cat.categories)
    study_levels = list(study_cat.cat.categories)

    oof = np.full_like(y, np.nan, dtype=np.float32)
    fold_rows = []
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(x, cancer_ids), start=1):
        x_z, x_fill, x_mean, x_std = standardize_x(x[train_idx], x)
        direct_x, direct_mask, direct_map = build_direct_matrix(x_z, feature_names, target_names)
        if fold == 1:
            direct_map.to_csv(OUT / "tables" / "direct_rna_residual_total_protein_gene_map.tsv", sep="\t", index=False)
        y_z, y_mean, y_std = standardize_y(y[train_idx], y)
        y_mask = np.isfinite(y_z).astype(np.float32)
        y_z = np.nan_to_num(y_z, nan=0.0).astype(np.float32)

        train_ds = TensorDataset(
            torch.from_numpy(x_z[train_idx]),
            torch.from_numpy(direct_x[train_idx]),
            torch.from_numpy(direct_mask[train_idx]),
            torch.from_numpy(cancer_ids[train_idx]),
            torch.from_numpy(study_ids[train_idx]),
            torch.from_numpy(y_z[train_idx]),
            torch.from_numpy(y_mask[train_idx]),
        )
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_x = torch.from_numpy(x_z[val_idx]).to(device)
        val_direct = torch.from_numpy(direct_x[val_idx]).to(device)
        val_direct_mask = torch.from_numpy(direct_mask[val_idx]).to(device)
        val_c = torch.from_numpy(cancer_ids[val_idx]).to(device)
        val_s = torch.from_numpy(study_ids[val_idx]).to(device)
        val_y = torch.from_numpy(y_z[val_idx]).to(device)
        val_m = torch.from_numpy(y_mask[val_idx]).to(device)

        model = TotalProteomeFiLMDirectResidual(
            n_input=x.shape[1],
            n_target=y.shape[1],
            n_cancer=len(cancer_levels),
            n_study=len(study_levels),
            hidden=args.hidden,
            cond_dim=args.cond_dim,
            dropout=args.dropout,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        best_state = None
        best_val = float("inf")
        stale = 0
        patience = 22
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for bx, bd, bdm, bc, bs, by, bm in loader:
                bx, bd, bdm, bc, bs, by, bm = bx.to(device), bd.to(device), bdm.to(device), bc.to(device), bs.to(device), by.to(device), bm.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=True):
                    pred = model(bx, bc, bs, bd, bdm)
                    loss = masked_mse(pred, by, bm) + args.center_loss_weight * masked_centered_mse(pred, by, bm)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                val_pred = model(val_x, val_c, val_s, val_direct, val_direct_mask)
                val_loss = float((masked_mse(val_pred, val_y, val_m) + args.center_loss_weight * masked_centered_mse(val_pred, val_y, val_m)).detach().cpu())
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if epoch == 1 or epoch % 10 == 0:
                print(f"fold {fold} epoch {epoch} train {np.mean(losses):.4f} val {val_loss:.4f}", flush=True)
            if stale >= patience:
                print(f"fold {fold} early_stop epoch {epoch}", flush=True)
                break

        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(val_idx), args.batch_size):
                xb = torch.from_numpy(x_z[val_idx][start:start + args.batch_size]).to(device)
                db = torch.from_numpy(direct_x[val_idx][start:start + args.batch_size]).to(device)
                dmb = torch.from_numpy(direct_mask[val_idx][start:start + args.batch_size]).to(device)
                cb = torch.from_numpy(cancer_ids[val_idx][start:start + args.batch_size]).to(device)
                sb = torch.from_numpy(study_ids[val_idx][start:start + args.batch_size]).to(device)
                preds.append(model(xb, cb, sb, db, dmb).detach().cpu().numpy())
        pred_z = np.vstack(preds)
        pred = pred_z * y_std + y_mean
        pred[~np.isfinite(y[val_idx])] = np.nan
        oof[val_idx] = pred.astype(np.float32)
        torch.save(
            {
                "state_dict": best_state,
                "x_fill": x_fill,
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "feature_names": feature_names,
                "rna_gene_features": list(rna_df.columns),
                "vae_features": list(z_df.columns),
                "target_names": target_names,
                "direct_rna_residual_map": direct_map.to_dict(orient="records"),
                "cancer_levels": cancer_levels,
                "study_levels": study_levels,
                "args": vars(args),
                "fold": fold,
                "best_val_loss": best_val,
                "model_class": "TotalProteomeFiLMDirectResidual",
            },
            OUT / "models" / f"cptac_total_proteome_film_vae_z_direct_residual_v1_fold{fold}.pt",
        )
        fold_rows.append({"fold": fold, "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "best_val_loss": best_val})

    pred_df = pd.DataFrame(oof, index=sample_ids, columns=target_names)
    pred_df.to_parquet(OUT / "predictions" / "oof_total_protein_predictions.parquet")
    metrics = spearman_by_target(y, oof, target_names)
    metrics.to_csv(OUT / "tables" / "metrics_by_total_protein.tsv", sep="\t", index=False)
    sm = sample_metrics(protein_df, pred_df, manifest)
    sm.to_csv(OUT / "tables" / "metrics_by_sample.tsv", sep="\t", index=False)
    cancer_rows = []
    for cancer, ids in manifest.groupby("cancer_label").groups.items():
        yy = protein_df.loc[list(ids)]
        pp = pred_df.loc[list(ids)]
        vals = []
        for col in yy.columns:
            ok = yy[col].notna() & pp[col].notna()
            if int(ok.sum()) >= 10:
                rho = yy.loc[ok, col].rank().corr(pp.loc[ok, col].rank())
                if pd.notna(rho):
                    vals.append(float(rho))
        arr = np.array(vals)
        cancer_rows.append({
            "cancer_label": cancer,
            "n_samples": len(ids),
            "n_targets_evaluable": int(len(arr)),
            "median_target_spearman": float(np.nanmedian(arr)) if len(arr) else np.nan,
            "mean_target_spearman": float(np.nanmean(arr)) if len(arr) else np.nan,
            "targets_ge_0_5": int(np.nansum(arr >= 0.5)) if len(arr) else 0,
            "targets_ge_0_7": int(np.nansum(arr >= 0.7)) if len(arr) else 0,
        })
    pd.DataFrame(cancer_rows).to_csv(OUT / "tables" / "metrics_by_cancer_targetwise.tsv", sep="\t", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "tables" / "fold_losses.tsv", sep="\t", index=False)

    old_path = ROOT / "02_results/model_validation/20260426_cptac_multitask_total_residual_locked_v1/tables/metrics_by_total_protein.tsv"
    if old_path.exists():
        old = pd.read_csv(old_path, sep="\t").rename(columns={"feature_id": "protein_gene", "spearman": "old_multitask_total_spearman"})
        comp = metrics.merge(old[["protein_gene", "old_multitask_total_spearman"]], on="protein_gene", how="left")
        comp["delta_new_minus_old"] = comp["spearman"] - comp["old_multitask_total_spearman"]
        comp.to_csv(OUT / "tables" / "compare_old_multitask_vs_new_total_by_protein.tsv", sep="\t", index=False)

    vals = metrics["spearman"].dropna()
    summary = {
        "n_samples": int(x.shape[0]),
        "n_rna_gene_features": int(rna_df.shape[1]),
        "n_vae_features": int(z_df.shape[1]),
        "n_input_features": int(x.shape[1]),
        "n_total_protein_targets": int(y.shape[1]),
        "n_direct_residual_mapped_targets": int(pd.read_csv(OUT / "tables" / "direct_rna_residual_total_protein_gene_map.tsv", sep="\t")["mapped"].sum()),
        "target_spearman_median": float(vals.median()),
        "target_spearman_mean": float(vals.mean()),
        "target_spearman_q25": float(vals.quantile(0.25)),
        "target_spearman_q75": float(vals.quantile(0.75)),
        "targets_ge_0_5": int((vals >= 0.5).sum()),
        "targets_ge_0_7": int((vals >= 0.7).sum()),
        "sample_spearman_median": float(np.nanmedian(sm["sample_spearman"])),
        "sample_spearman_mean": float(np.nanmean(sm["sample_spearman"])),
        "device": str(device),
        "model": "CPTAC total proteome predictor with RNA, VAE z, cancer/study FiLM, and direct RNA residual head",
    }
    (OUT / "logs" / "cptac_total_proteome_film_vae_z_direct_residual_v1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
