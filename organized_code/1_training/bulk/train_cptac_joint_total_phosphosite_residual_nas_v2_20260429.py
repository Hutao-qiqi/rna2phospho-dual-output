#!/usr/bin/env python3
"""Search joint total-proteome and phosphosite residual architectures."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path("/data/lsy/Infinite_Stream")
DATA = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2"
OUT = ROOT / "02_results/model_validation/20260429_cptac_joint_total_phosphosite_residual_nas_v2"
TOTAL_SCRIPT = ROOT / "03_code/model_validation/train_cptac_total_proteome_film_vae_z_direct_residual_20260428.py"


@dataclass(frozen=True)
class Config:
    config_id: str
    hidden: int
    depth: int
    bottleneck: int
    cond_dim: int
    dropout: float
    lr: float
    weight_decay: float
    bank_k: int
    total_loss_weight: float
    center_loss_weight: float
    pda_weight: float
    lamb1_weight: float
    inject_mode: str


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


class JointResidualModel(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_total: int,
        n_phospho: int,
        n_cancer: int,
        n_study: int,
        parent_total_idx: np.ndarray,
        parent_total_mask: np.ndarray,
        cfg: Config,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.cancer_emb = nn.Embedding(n_cancer, cfg.cond_dim // 2)
        self.study_emb = nn.Embedding(n_study, cfg.cond_dim // 2)
        blocks = []
        in_dim = n_input
        for i in range(cfg.depth):
            out_dim = cfg.hidden if i < cfg.depth - 1 else cfg.bottleneck
            blocks.append(FiLMBlock(in_dim, out_dim, cfg.cond_dim, cfg.dropout))
            in_dim = out_dim
        self.blocks = nn.ModuleList(blocks)
        self.total_head = nn.Sequential(nn.LayerNorm(cfg.bottleneck), nn.Linear(cfg.bottleneck, n_total))
        self.phospho_head = nn.Sequential(
            nn.LayerNorm(cfg.bottleneck),
            nn.Linear(cfg.bottleneck, cfg.bottleneck),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.bottleneck, n_phospho),
        )
        self.total_rna_scale = nn.Parameter(torch.zeros(n_total))
        self.total_rna_bias = nn.Parameter(torch.zeros(n_total))
        self.phospho_rna_scale = nn.Parameter(torch.zeros(n_phospho))
        self.phospho_bank_scale = nn.Parameter(torch.zeros(n_phospho))
        self.phospho_total_scale = nn.Parameter(torch.zeros(n_phospho))
        self.phospho_bias = nn.Parameter(torch.zeros(n_phospho))
        self.register_buffer("parent_total_idx", torch.tensor(parent_total_idx, dtype=torch.long))
        self.register_buffer("parent_total_mask", torch.tensor(parent_total_mask, dtype=torch.float32))

    def cond(self, cancer: torch.Tensor, study: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.cancer_emb(cancer), self.study_emb(study)], dim=1)

    def trunk(self, x: torch.Tensor, cancer: torch.Tensor, study: torch.Tensor) -> torch.Tensor:
        cond = self.cond(cancer, study)
        h = x
        for i, block in enumerate(self.blocks):
            h_new = block(h, cond)
            if i > 0 and h_new.shape == h.shape:
                h = h_new + h
            else:
                h = h_new
        return h

    def forward(
        self,
        x: torch.Tensor,
        cancer: torch.Tensor,
        study: torch.Tensor,
        total_direct: torch.Tensor,
        total_direct_mask: torch.Tensor,
        phospho_direct: torch.Tensor,
        phospho_direct_mask: torch.Tensor,
        phospho_bank: torch.Tensor,
        phospho_bank_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x, cancer, study)
        total = self.total_head(h)
        if self.cfg.inject_mode in {"direct", "all"}:
            total = total + total_direct_mask * (total_direct * self.total_rna_scale[None, :] + self.total_rna_bias[None, :])
        phospho = self.phospho_head(h)
        parent_total = total[:, self.parent_total_idx] * self.parent_total_mask[None, :]
        if self.cfg.inject_mode in {"direct", "all"}:
            phospho = phospho + phospho_direct_mask * (phospho_direct * self.phospho_rna_scale[None, :])
        if self.cfg.inject_mode in {"bank", "all"}:
            phospho = phospho + phospho_bank_mask * (phospho_bank * self.phospho_bank_scale[None, :])
        if self.cfg.inject_mode in {"total", "all"}:
            phospho = phospho + self.parent_total_mask[None, :] * (parent_total * self.phospho_total_scale[None, :])
        phospho = phospho + self.phospho_bias[None, :]
        return total, phospho


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_total_module():
    spec = importlib.util.spec_from_file_location("total_module", TOTAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def standardize_x(x_train: np.ndarray, x_all: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.where(np.isfinite(x_train), x_train, np.nan)
    all_x = np.where(np.isfinite(x_all), x_all, np.nan)
    fill = np.nanmedian(train, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    train_f = np.where(np.isfinite(train), train, fill)
    all_f = np.where(np.isfinite(all_x), all_x, fill)
    mean = train_f.mean(axis=0)
    std = train_f.std(axis=0)
    std = np.where((std > 1e-6) & np.isfinite(std), std, 1.0)
    return ((all_f - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def standardize_y(y_train: np.ndarray, y_all: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(y_train, axis=0)
    std = np.nanstd(y_train, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where((std > 1e-6) & np.isfinite(std), std, 1.0)
    return ((y_all - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def weighted_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, target_weight: torch.Tensor | None = None) -> torch.Tensor:
    w = mask if target_weight is None else mask * target_weight[None, :]
    return (((pred - y) ** 2) * w).sum() / w.sum().clamp_min(1.0)


def centered_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pred_c = pred - (pred * mask).sum(dim=1, keepdim=True) / denom
    y_c = y - (y * mask).sum(dim=1, keepdim=True) / denom
    return weighted_mse(pred_c, y_c, mask)


def spearman_vec(y: np.ndarray, p: np.ndarray, names: list[str], min_n: int = 10) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(names):
        ok = np.isfinite(y[:, j]) & np.isfinite(p[:, j])
        rho = np.nan
        if int(ok.sum()) >= min_n:
            rho = spearmanr(y[ok, j], p[ok, j], nan_policy="omit").correlation
        rows.append({"target": name, "n": int(ok.sum()), "spearman": float(rho) if np.isfinite(rho) else np.nan})
    return pd.DataFrame(rows)


def sample_spearman(y: np.ndarray, p: np.ndarray) -> float:
    vals = []
    for i in range(y.shape[0]):
        ok = np.isfinite(y[i]) & np.isfinite(p[i])
        if int(ok.sum()) >= 10:
            vals.append(pd.Series(y[i, ok]).rank().corr(pd.Series(p[i, ok]).rank()))
    return float(np.nanmedian(vals)) if vals else np.nan


def make_direct_total(x_z: np.ndarray, feature_names: list[str], total_names: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    idx = {g: i for i, g in enumerate(feature_names)}
    direct = np.zeros((x_z.shape[0], len(total_names)), dtype=np.float32)
    mask = np.zeros_like(direct)
    rows = []
    for j, gene in enumerate(total_names):
        mapped = gene in idx
        if mapped:
            direct[:, j] = x_z[:, idx[gene]]
            mask[:, j] = 1.0
        rows.append({"total_gene": gene, "direct_gene_symbol": gene if mapped else "", "mapped": bool(mapped)})
    return direct, mask, pd.DataFrame(rows)


def make_phospho_parent_maps(
    x_z: np.ndarray,
    feature_names: list[str],
    phospho_names: list[str],
    total_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    feature_idx = {g: i for i, g in enumerate(feature_names)}
    total_idx = {g: i for i, g in enumerate(total_names)}
    direct = np.zeros((x_z.shape[0], len(phospho_names)), dtype=np.float32)
    direct_mask = np.zeros_like(direct)
    parent_idx = np.zeros(len(phospho_names), dtype=np.int64)
    parent_mask = np.zeros(len(phospho_names), dtype=np.float32)
    rows = []
    for j, target in enumerate(phospho_names):
        gene = str(target).split("|", 1)[0]
        if gene in feature_idx:
            direct[:, j] = x_z[:, feature_idx[gene]]
            direct_mask[:, j] = 1.0
        if gene in total_idx:
            parent_idx[j] = total_idx[gene]
            parent_mask[j] = 1.0
        rows.append({
            "phosphosite": target,
            "protein_gene": gene,
            "direct_rna_mapped": bool(gene in feature_idx),
            "parent_total_mapped": bool(gene in total_idx),
        })
    return direct, direct_mask, parent_idx, parent_mask, pd.DataFrame(rows)


def make_target_weights(names: list[str], manifest: pd.DataFrame, cfg: Config) -> np.ndarray:
    w = np.ones(len(names), dtype=np.float32)
    pda_mask = manifest["cancer_label"].to_numpy() == "PDA"
    # Increase weight for targets observed in PDA so the search does not optimize only large cancers.
    w *= 1.0
    for j, name in enumerate(names):
        if str(name).startswith("LAMB1|S1666"):
            w[j] *= cfg.lamb1_weight
    return w


def build_corr_bank(
    x_z: np.ndarray,
    y_z: np.ndarray,
    y_mask: np.ndarray,
    train_idx: np.ndarray,
    k: int,
    chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        return np.zeros_like(y_z, dtype=np.float32), np.zeros_like(y_z, dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.tensor(x_z[train_idx], dtype=torch.float32, device=device)
    x_all = torch.tensor(x_z, dtype=torch.float32, device=device)
    y_train = torch.tensor(np.nan_to_num(y_z[train_idx], nan=0.0), dtype=torch.float32, device=device)
    m_train = torch.tensor(y_mask[train_idx], dtype=torch.float32, device=device)
    bank_chunks = []
    mask_chunks = []
    with torch.no_grad():
        for start in range(0, y_z.shape[1], chunk):
            end = min(start + chunk, y_z.shape[1])
            yt = y_train[:, start:end] * m_train[:, start:end]
            counts = m_train[:, start:end].sum(dim=0).clamp_min(1.0)
            corr = x_train.T @ yt / counts[None, :]
            top = torch.topk(torch.abs(corr), k=min(k, corr.shape[0]), dim=0)
            idx = top.indices
            weights = torch.gather(corr, 0, idx)
            weights = weights / torch.abs(weights).sum(dim=0, keepdim=True).clamp_min(1e-6)
            vals = []
            for t in range(end - start):
                vals.append((x_all[:, idx[:, t]] * weights[:, t][None, :]).sum(dim=1))
            bank = torch.stack(vals, dim=1).detach().cpu().numpy().astype(np.float32)
            valid = (counts.detach().cpu().numpy() >= 10).astype(np.float32)
            bank_chunks.append(bank)
            mask_chunks.append(np.repeat(valid[None, :], x_z.shape[0], axis=0))
    return np.concatenate(bank_chunks, axis=1), np.concatenate(mask_chunks, axis=1).astype(np.float32)


def generate_configs(limit: int, seed: int) -> list[Config]:
    base = []
    i = 0
    for hidden, depth, bottleneck, dropout, bank_k, inject_mode, total_w, center_w, lr in itertools.product(
        [1024, 1536, 2048],
        [3, 4, 5],
        [512, 768, 1024],
        [0.10, 0.18, 0.28, 0.40],
        [0, 4, 8, 16],
        ["all", "direct", "total", "bank"],
        [0.10, 0.25, 0.45],
        [0.15, 0.30],
        [1.2e-4, 2.0e-4],
    ):
        if bottleneck > hidden:
            continue
        if inject_mode == "bank" and bank_k == 0:
            continue
        i += 1
        base.append(Config(
            config_id=f"nas_{i:04d}",
            hidden=hidden,
            depth=depth,
            bottleneck=bottleneck,
            cond_dim=96,
            dropout=dropout,
            lr=lr,
            weight_decay=1e-4,
            bank_k=bank_k,
            total_loss_weight=total_w,
            center_loss_weight=center_w,
            pda_weight=1.5,
            lamb1_weight=8.0,
            inject_mode=inject_mode,
        ))
    rng = random.Random(seed)
    rng.shuffle(base)
    must = [
        Config("forced_all_k16_h2048_d5", 2048, 5, 1024, 96, 0.18, 1.2e-4, 1e-4, 16, 0.25, 0.30, 1.5, 10.0, "all"),
        Config("forced_all_k8_h1536_d4", 1536, 4, 768, 96, 0.18, 2.0e-4, 1e-4, 8, 0.25, 0.30, 1.5, 10.0, "all"),
        Config("forced_total_direct_h1536", 1536, 4, 768, 96, 0.18, 2.0e-4, 1e-4, 0, 0.25, 0.30, 1.5, 10.0, "total"),
    ]
    return (must + base)[:limit]


def fit_one(
    cfg: Config,
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    data: dict,
    args: argparse.Namespace,
    save_model: bool = False,
) -> dict:
    device = data["device"]
    x_z, x_mean, x_std = standardize_x(data["x"][train_idx], data["x"])
    total_z, total_mean, total_std = standardize_y(data["total"][train_idx], data["total"])
    phospho_z, phospho_mean, phospho_std = standardize_y(data["phospho"][train_idx], data["phospho"])
    total_mask = np.isfinite(total_z).astype(np.float32)
    phospho_mask = np.isfinite(phospho_z).astype(np.float32)
    total_z = np.nan_to_num(total_z, nan=0.0).astype(np.float32)
    phospho_z = np.nan_to_num(phospho_z, nan=0.0).astype(np.float32)

    total_direct, total_direct_mask, total_map = make_direct_total(x_z, data["feature_names"], data["total_names"])
    phospho_direct, phospho_direct_mask, parent_idx, parent_mask, phospho_map = make_phospho_parent_maps(
        x_z, data["feature_names"], data["phospho_names"], data["total_names"]
    )
    bank, bank_mask = build_corr_bank(x_z, phospho_z, phospho_mask, train_idx, cfg.bank_k, chunk=args.bank_chunk)
    target_weight = make_target_weights(data["phospho_names"], data["manifest"], cfg)

    train_ds = TensorDataset(
        torch.from_numpy(x_z[train_idx]),
        torch.from_numpy(data["cancer_ids"][train_idx]),
        torch.from_numpy(data["study_ids"][train_idx]),
        torch.from_numpy(total_direct[train_idx]),
        torch.from_numpy(total_direct_mask[train_idx]),
        torch.from_numpy(phospho_direct[train_idx]),
        torch.from_numpy(phospho_direct_mask[train_idx]),
        torch.from_numpy(bank[train_idx]),
        torch.from_numpy(bank_mask[train_idx]),
        torch.from_numpy(total_z[train_idx]),
        torch.from_numpy(total_mask[train_idx]),
        torch.from_numpy(phospho_z[train_idx]),
        torch.from_numpy(phospho_mask[train_idx]),
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = JointResidualModel(
        n_input=x_z.shape[1],
        n_total=len(data["total_names"]),
        n_phospho=len(data["phospho_names"]),
        n_cancer=len(data["cancer_levels"]),
        n_study=len(data["study_levels"]),
        parent_total_idx=parent_idx,
        parent_total_mask=parent_mask,
        cfg=cfg,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    tw = torch.tensor(target_weight, dtype=torch.float32, device=device)

    val_tensors = [
        torch.tensor(arr[val_idx], dtype=torch.float32, device=device)
        for arr in [x_z, total_direct, total_direct_mask, phospho_direct, phospho_direct_mask, bank, bank_mask, total_z, total_mask, phospho_z, phospho_mask]
    ]
    val_c = torch.tensor(data["cancer_ids"][val_idx], dtype=torch.long, device=device)
    val_s = torch.tensor(data["study_ids"][val_idx], dtype=torch.long, device=device)
    best_state = None
    best_val = math.inf
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for bx, bc, bs, btd, btdm, bpd, bpdm, bb, bbm, byt, bmt, byp, bmp in loader:
            bx = bx.to(device); bc = bc.to(device); bs = bs.to(device)
            btd = btd.to(device); btdm = btdm.to(device); bpd = bpd.to(device); bpdm = bpdm.to(device)
            bb = bb.to(device); bbm = bbm.to(device); byt = byt.to(device); bmt = bmt.to(device); byp = byp.to(device); bmp = bmp.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred_t, pred_p = model(bx, bc, bs, btd, btdm, bpd, bpdm, bb, bbm)
                loss_p = weighted_mse(pred_p, byp, bmp, tw) + cfg.center_loss_weight * centered_mse(pred_p, byp, bmp)
                loss_t = weighted_mse(pred_t, byt, bmt)
                loss = loss_p + cfg.total_loss_weight * loss_t
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            vx, vtd, vtdm, vpd, vpdm, vb, vbm, vyt, vmt, vyp, vmp = val_tensors
            pred_t, pred_p = model(vx, val_c, val_s, vtd, vtdm, vpd, vpdm, vb, vbm)
            val_loss = float((weighted_mse(pred_p, vyp, vmp, tw) + cfg.total_loss_weight * weighted_mse(pred_t, vyt, vmt)).detach().cpu())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{cfg.config_id} fold {fold} epoch {epoch} train {np.mean(losses):.4f} val {val_loss:.4f}", flush=True)
        if stale >= args.patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    pred_t_batches = []
    pred_p_batches = []
    with torch.no_grad():
        for start in range(0, len(val_idx), args.batch_size):
            idx = val_idx[start:start + args.batch_size]
            tensors = [torch.tensor(arr[idx], dtype=torch.float32, device=device) for arr in [x_z, total_direct, total_direct_mask, phospho_direct, phospho_direct_mask, bank, bank_mask]]
            cb = torch.tensor(data["cancer_ids"][idx], dtype=torch.long, device=device)
            sb = torch.tensor(data["study_ids"][idx], dtype=torch.long, device=device)
            pt, pp = model(tensors[0], cb, sb, tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], tensors[6])
            pred_t_batches.append((pt.detach().cpu().numpy() * total_std + total_mean).astype(np.float32))
            pred_p_batches.append((pp.detach().cpu().numpy() * phospho_std + phospho_mean).astype(np.float32))
    pred_t = np.vstack(pred_t_batches)
    pred_p = np.vstack(pred_p_batches)
    true_t = data["total"][val_idx]
    true_p = data["phospho"][val_idx]
    phospho_metrics = spearman_vec(true_p, pred_p, data["phospho_names"])
    total_metrics = spearman_vec(true_t, pred_t, data["total_names"])
    pda_sample_mask = data["manifest"].iloc[val_idx]["cancer_label"].to_numpy() == "PDA"
    pda_metrics = spearman_vec(true_p[pda_sample_mask], pred_p[pda_sample_mask], data["phospho_names"]) if pda_sample_mask.sum() >= 10 else pd.DataFrame()
    lamb = phospho_metrics.loc[phospho_metrics["target"].eq("LAMB1|S1666"), "spearman"]
    lamb_pda = pda_metrics.loc[pda_metrics["target"].eq("LAMB1|S1666"), "spearman"] if not pda_metrics.empty else pd.Series(dtype=float)
    score = (
        float(phospho_metrics["spearman"].median(skipna=True))
        + 0.30 * float(total_metrics["spearman"].median(skipna=True))
        + 0.25 * float(pda_metrics["spearman"].median(skipna=True) if not pda_metrics.empty else 0.0)
        + 0.25 * float(lamb.iloc[0] if len(lamb) and np.isfinite(lamb.iloc[0]) else 0.0)
        + 0.15 * float(lamb_pda.iloc[0] if len(lamb_pda) and np.isfinite(lamb_pda.iloc[0]) else 0.0)
        + 0.10 * sample_spearman(true_p, pred_p)
    )
    rec = {
        **asdict(cfg),
        "fold": fold,
        "epochs_run": epoch,
        "val_loss": best_val,
        "score": score,
        "phospho_median_spearman": float(phospho_metrics["spearman"].median(skipna=True)),
        "phospho_ge_0_5": int((phospho_metrics["spearman"] >= 0.5).sum()),
        "total_median_spearman": float(total_metrics["spearman"].median(skipna=True)),
        "total_ge_0_5": int((total_metrics["spearman"] >= 0.5).sum()),
        "pda_phospho_median_spearman": float(pda_metrics["spearman"].median(skipna=True)) if not pda_metrics.empty else np.nan,
        "lamb1_s1666_spearman": float(lamb.iloc[0]) if len(lamb) else np.nan,
        "pda_lamb1_s1666_spearman": float(lamb_pda.iloc[0]) if len(lamb_pda) else np.nan,
        "sample_phospho_median_spearman": sample_spearman(true_p, pred_p),
    }
    if save_model:
        model_path = OUT / "models" / f"{cfg.config_id}_fold{fold}.pt"
        torch.save({
            "config": asdict(cfg),
            "state_dict": best_state,
            "feature_names": data["feature_names"],
            "total_names": data["total_names"],
            "phospho_names": data["phospho_names"],
            "cancer_levels": data["cancer_levels"],
            "study_levels": data["study_levels"],
            "x_mean": x_mean,
            "x_std": x_std,
            "total_mean": total_mean,
            "total_std": total_std,
            "phospho_mean": phospho_mean,
            "phospho_std": phospho_std,
            "parent_total_idx": parent_idx,
            "parent_total_mask": parent_mask,
            "total_direct_map": total_map,
            "phospho_direct_map": phospho_map,
        }, model_path)
    return rec


def load_data(device: torch.device) -> dict:
    rna = pd.read_parquet(DATA / "rna_log2_tpm_paired.parquet")
    total = pd.read_parquet(DATA / "total_protein_gene_study_zscore_min20pct.parquet").loc[rna.index]
    phospho = pd.read_parquet(DATA / "phosphosite_gene_site_study_zscore_min20pct_targets.parquet").loc[rna.index]
    manifest = pd.read_csv(DATA / "sample_manifest.tsv", sep="\t").set_index("sample_id").loc[rna.index]
    total_module = import_total_module()
    z = total_module.encode_vae_z(rna, device)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["search", "full"], default="search")
    parser.add_argument("--configs", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--full-epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bank-chunk", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260428)
    args = parser.parse_args()
    seed_all(args.seed)
    for sub in ["models", "tables", "logs", "predictions"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("GPU is required")
    print("device", device, torch.cuda.get_device_name(device), flush=True)
    data = load_data(device)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = list(skf.split(data["x"], data["cancer_ids"]))
    if args.mode == "search":
        configs = generate_configs(args.configs, args.seed)
        rows = []
        out_file = OUT / "tables" / "nas_search_results.tsv"
        for i, cfg in enumerate(configs, start=1):
            print(f"CONFIG {i}/{len(configs)} {cfg}", flush=True)
            train_idx, val_idx = folds[0]
            try:
                rec = fit_one(cfg, 1, train_idx, val_idx, data, args, save_model=False)
            except Exception as exc:
                rec = {**asdict(cfg), "fold": 1, "error": repr(exc)}
                print("ERROR", cfg.config_id, repr(exc), flush=True)
            rows.append(rec)
            pd.DataFrame(rows).to_csv(out_file, sep="\t", index=False)
        best = pd.DataFrame(rows).sort_values("score", ascending=False, na_position="last").head(10)
        best.to_csv(OUT / "tables" / "nas_search_top10.tsv", sep="\t", index=False)
        print(best.to_string(index=False), flush=True)
        return 0

    search = pd.read_csv(OUT / "tables" / "nas_search_results.tsv", sep="\t").sort_values("score", ascending=False)
    row = search.iloc[0].to_dict()
    cfg = Config(
        config_id=str(row["config_id"]) + "_full5fold",
        hidden=int(row["hidden"]),
        depth=int(row["depth"]),
        bottleneck=int(row["bottleneck"]),
        cond_dim=int(row["cond_dim"]),
        dropout=float(row["dropout"]),
        lr=float(row["lr"]),
        weight_decay=float(row["weight_decay"]),
        bank_k=int(row["bank_k"]),
        total_loss_weight=float(row["total_loss_weight"]),
        center_loss_weight=float(row["center_loss_weight"]),
        pda_weight=float(row["pda_weight"]),
        lamb1_weight=float(row["lamb1_weight"]),
        inject_mode=str(row["inject_mode"]),
    )
    args.epochs = args.full_epochs
    rows = []
    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        rec = fit_one(cfg, fold, train_idx, val_idx, data, args, save_model=True)
        rows.append(rec)
        pd.DataFrame(rows).to_csv(OUT / "tables" / "best_config_full5fold_metrics.tsv", sep="\t", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
