#!/usr/bin/env python3
"""Train a deployable dual-output RNA-to-phosphorylation model.

Input:
  bulk RNA matrix aligned by gene symbol.

Outputs:
  1. CPTAC/PDC mass-spectrometry phosphosite predictions.
  2. TCGA/TCPA phospho-RPPA antibody predictions.

The two supervision sources are intentionally kept as separate heads because
mass-spectrometry sites and RPPA antibody readouts are not the same measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path("/data/lsy/Infinite_Stream")
CPTAC_DATA = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2"
TCPA_TRAIN_SCRIPT = ROOT / "03_code/model_validation/train_tcpa_pancancer_rppa_film_vae_z_direct_residual_20260428.py"
OUT = ROOT / "02_results/model_validation/20260429_rna2phospho_dual_output_final_v1"


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 1536
    bottleneck: int = 768
    depth: int = 4
    cond_dim: int = 96
    dropout: float = 0.22
    lr: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 80
    patience: int = 14
    rppa_loss_weight: float = 1.0
    center_loss_weight: float = 0.20


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


class DualOutputPhosphoModel(nn.Module):
    def __init__(self, n_gene: int, n_cancer: int, n_site: int, n_rppa: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.cancer_emb = nn.Embedding(n_cancer, cfg.cond_dim)
        blocks: list[nn.Module] = []
        in_dim = n_gene
        for i in range(cfg.depth):
            out_dim = cfg.hidden if i < cfg.depth - 1 else cfg.bottleneck
            blocks.append(FiLMBlock(in_dim, out_dim, cfg.cond_dim, cfg.dropout))
            in_dim = out_dim
        self.blocks = nn.ModuleList(blocks)
        self.site_head = nn.Sequential(
            nn.LayerNorm(cfg.bottleneck),
            nn.Linear(cfg.bottleneck, cfg.bottleneck),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.bottleneck, n_site),
        )
        self.rppa_head = nn.Sequential(
            nn.LayerNorm(cfg.bottleneck),
            nn.Linear(cfg.bottleneck, cfg.bottleneck // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.bottleneck // 2, n_rppa),
        )

    def forward(self, x: torch.Tensor, cancer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cond = self.cancer_emb(cancer)
        h = x
        for block in self.blocks:
            h_new = block(h, cond)
            if h_new.shape == h.shape:
                h = h + h_new
            else:
                h = h_new
        return self.site_head(h), self.rppa_head(h)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_antibody(name: str) -> str:
    if name.startswith("X") and len(name) > 1 and name[1].isdigit():
        return name[1:]
    return name


def is_phospho_antibody(name: str) -> bool:
    key = str(name).upper().replace("_", "")
    if "CLEAVED" in key:
        return False
    return bool(re.search(r"P[STY][0-9]", key))


def import_tcpa_loader():
    spec = importlib.util.spec_from_file_location("tcpa_loader", TCPA_TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def standardize_x(x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = x.to_numpy(dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    fill = np.nanmedian(arr, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    arr = np.where(np.isfinite(arr), arr, fill)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where((std > 1e-6) & np.isfinite(std), std, 1.0)
    return ((arr - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def standardize_y(y: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = y.to_numpy(dtype=np.float32)
    mask = np.isfinite(arr).astype(np.float32)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where((std > 1e-6) & np.isfinite(std), std, 1.0)
    z = (arr - mean) / std
    z = np.where(np.isfinite(z), z, 0.0).astype(np.float32)
    return z, mask, mean.astype(np.float32), std.astype(np.float32)


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum()
    if denom.item() <= 0:
        return pred.sum() * 0.0
    return (((pred - y) ** 2) * mask).sum() / denom


def centered_loss(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    row_n = mask.sum(dim=1, keepdim=True)
    valid = row_n.squeeze(1) > 1
    if valid.sum().item() <= 0:
        return pred.sum() * 0.0
    pred = pred[valid]
    y = y[valid]
    mask = mask[valid]
    row_n = row_n[valid].clamp_min(1.0)
    pred_c = pred - (pred * mask).sum(dim=1, keepdim=True) / row_n
    y_c = y - (y * mask).sum(dim=1, keepdim=True) / row_n
    return masked_mse(pred_c, y_c, mask)


def spearman_by_target(y: np.ndarray, p: np.ndarray, mask: np.ndarray, names: list[str], min_n: int = 10) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(names):
        ok = mask[:, j] > 0
        rho = np.nan
        if int(ok.sum()) >= min_n:
            rho = spearmanr(y[ok, j], p[ok, j], nan_policy="omit").correlation
        rows.append({"target": name, "n": int(ok.sum()), "spearman": float(rho) if np.isfinite(rho) else np.nan})
    return pd.DataFrame(rows)


def load_inputs() -> dict[str, object]:
    cptac_x = pd.read_parquet(CPTAC_DATA / "rna_log2_tpm_paired.parquet")
    cptac_y = pd.read_parquet(CPTAC_DATA / "phosphosite_gene_site_study_zscore_min20pct_targets.parquet")
    cptac_manifest = pd.read_csv(CPTAC_DATA / "sample_manifest.tsv", sep="\t").set_index("sample_id").loc[cptac_x.index]
    cptac_cancer = cptac_manifest["cancer_label"].astype(str)

    tcpa_module = import_tcpa_loader()
    tcga_x_full, tcga_y_all, tcga_meta, gene_features, vae_features, antibody_cols = tcpa_module.load_data()
    gene_cols = [c for c in tcga_x_full.columns if not str(c).startswith("VAE__")]
    tcga_x = tcga_x_full[gene_cols].copy()
    tcga_y_all = tcga_y_all.rename(columns={c: clean_antibody(c) for c in tcga_y_all.columns})
    phospho_antibodies = [c for c in tcga_y_all.columns if is_phospho_antibody(c)]
    tcga_y = tcga_y_all[phospho_antibodies].copy()
    tcga_cancer = tcga_meta["project"].astype(str).str.replace("^TCGA-", "", regex=True)

    genes = sorted(set(map(str, cptac_x.columns)) | set(map(str, tcga_x.columns)))
    cptac_x = cptac_x.reindex(columns=genes)
    tcga_x = tcga_x.reindex(columns=genes)

    all_x = pd.concat([cptac_x, tcga_x], axis=0)
    all_source = np.array(["CPTAC_PDC"] * len(cptac_x) + ["TCGA_TCPA"] * len(tcga_x), dtype=object)
    all_cancer = pd.concat([cptac_cancer, tcga_cancer], axis=0).astype(str).to_numpy()
    cancer_levels = sorted(pd.unique(pd.Series(all_cancer)).tolist()) + ["UNK_CANCER"]
    cancer_to_id = {c: i for i, c in enumerate(cancer_levels)}
    cancer_id = np.array([cancer_to_id.get(c, cancer_to_id["UNK_CANCER"]) for c in all_cancer], dtype=np.int64)

    site_names = list(cptac_y.columns)
    rppa_names = list(tcga_y.columns)
    y_site = pd.DataFrame(np.nan, index=all_x.index, columns=site_names, dtype=np.float32)
    y_rppa = pd.DataFrame(np.nan, index=all_x.index, columns=rppa_names, dtype=np.float32)
    y_site.loc[cptac_y.index, :] = cptac_y
    y_rppa.loc[tcga_y.index, :] = tcga_y

    return {
        "x": all_x,
        "source": all_source,
        "cancer": all_cancer,
        "cancer_id": cancer_id,
        "cancer_levels": cancer_levels,
        "y_site": y_site,
        "y_rppa": y_rppa,
        "genes": genes,
        "site_names": site_names,
        "rppa_names": rppa_names,
        "cptac_n": len(cptac_x),
        "tcga_n": len(tcga_x),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--val-size", type=float, default=0.10)
    args = parser.parse_args()

    seed_all(args.seed)
    cfg = ModelConfig(epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)
    for sub in ["models", "tables", "logs", "manifests", "scripts"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    data = load_inputs()
    x_z, x_mean, x_std = standardize_x(data["x"])
    y_site, site_mask, site_mean, site_std = standardize_y(data["y_site"])
    y_rppa, rppa_mask, rppa_mean, rppa_std = standardize_y(data["y_rppa"])
    cancer_id = data["cancer_id"]

    idx = np.arange(x_z.shape[0])
    strat = pd.Series(data["source"]).astype(str) + "::" + pd.Series(data["cancer"]).astype(str)
    counts = strat.value_counts()
    strat_safe = strat.where(strat.map(counts) >= 2, data["source"])
    train_idx, val_idx = train_test_split(
        idx,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=strat_safe,
    )

    tensors = [
        torch.tensor(x_z, dtype=torch.float32),
        torch.tensor(cancer_id, dtype=torch.long),
        torch.tensor(y_site, dtype=torch.float32),
        torch.tensor(site_mask, dtype=torch.float32),
        torch.tensor(y_rppa, dtype=torch.float32),
        torch.tensor(rppa_mask, dtype=torch.float32),
    ]
    ds = TensorDataset(*tensors)
    train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(torch.utils.data.Subset(ds, val_idx), batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualOutputPhosphoModel(
        n_gene=len(data["genes"]),
        n_cancer=len(data["cancer_levels"]),
        n_site=len(data["site_names"]),
        n_rppa=len(data["rppa_names"]),
        cfg=cfg,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    def run_eval() -> tuple[float, dict[str, np.ndarray]]:
        model.eval()
        total_loss = 0.0
        total_n = 0
        pred_site = []
        pred_rppa = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            for xb, cb, ys, ms, yr, mr in val_loader:
                xb, cb = xb.to(device), cb.to(device)
                ys, ms, yr, mr = ys.to(device), ms.to(device), yr.to(device), mr.to(device)
                ps, pr = model(xb, cb)
                loss_site = masked_mse(ps, ys, ms)
                loss_rppa = masked_mse(pr, yr, mr)
                loss_center = centered_loss(ps, ys, ms)
                loss = loss_site + cfg.rppa_loss_weight * loss_rppa + cfg.center_loss_weight * loss_center
                total_loss += float(loss.detach().cpu()) * xb.shape[0]
                total_n += xb.shape[0]
                pred_site.append(ps.float().cpu().numpy())
                pred_rppa.append(pr.float().cpu().numpy())
        return total_loss / max(total_n, 1), {
            "site": np.vstack(pred_site),
            "rppa": np.vstack(pred_rppa),
        }

    history = []
    best_loss = math.inf
    best_epoch = 0
    best_path = OUT / "models/rna2phospho_dual_output_final_v1.pt"
    print("device", device, flush=True)
    if device.type == "cuda":
        print("gpu", torch.cuda.get_device_name(0), flush=True)
    print(
        json.dumps(
            {
                "n_samples": int(x_z.shape[0]),
                "n_cptac_samples": int(data["cptac_n"]),
                "n_tcga_tcpa_samples": int(data["tcga_n"]),
                "n_genes": len(data["genes"]),
                "n_cptac_phosphosites": len(data["site_names"]),
                "n_tcpa_phospho_antibodies": len(data["rppa_names"]),
                "n_cancers": len(data["cancer_levels"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    bad_epochs = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for xb, cb, ys, ms, yr, mr in train_loader:
            xb, cb = xb.to(device), cb.to(device)
            ys, ms, yr, mr = ys.to(device), ms.to(device), yr.to(device), mr.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                ps, pr = model(xb, cb)
                loss_site = masked_mse(ps, ys, ms)
                loss_rppa = masked_mse(pr, yr, mr)
                loss_center = centered_loss(ps, ys, ms)
                loss = loss_site + cfg.rppa_loss_weight * loss_rppa + cfg.center_loss_weight * loss_center
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            train_loss += float(loss.detach().cpu()) * xb.shape[0]
            train_n += xb.shape[0]
        val_loss, _ = run_eval()
        row = {"epoch": epoch, "train_loss": train_loss / max(train_n, 1), "val_loss": val_loss}
        history.append(row)
        print(f"epoch {epoch} train {row['train_loss']:.5f} val {val_loss:.5f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "genes": data["genes"],
                    "cancer_levels": data["cancer_levels"],
                    "cptac_phosphosite_targets": data["site_names"],
                    "tcpa_phospho_antibodies": data["rppa_names"],
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "site_mean": site_mean,
                    "site_std": site_std,
                    "rppa_mean": rppa_mean,
                    "rppa_std": rppa_std,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_loss,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"early_stop epoch {epoch}", flush=True)
                break

    pd.DataFrame(history).to_csv(OUT / "tables/training_history.tsv", sep="\t", index=False)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_loss, preds = run_eval()
    val_site = y_site[val_idx]
    val_site_mask = site_mask[val_idx]
    val_rppa = y_rppa[val_idx]
    val_rppa_mask = rppa_mask[val_idx]
    site_metrics = spearman_by_target(val_site, preds["site"], val_site_mask, data["site_names"])
    rppa_metrics = spearman_by_target(val_rppa, preds["rppa"], val_rppa_mask, data["rppa_names"])
    site_metrics.to_csv(OUT / "tables/validation_cptac_phosphosite_spearman.tsv", sep="\t", index=False)
    rppa_metrics.to_csv(OUT / "tables/validation_tcpa_phospho_antibody_spearman.tsv", sep="\t", index=False)

    pd.Series(data["genes"], name="gene_symbol").to_csv(OUT / "manifests/input_gene_order.tsv", sep="\t", index=False)
    pd.Series(data["site_names"], name="cptac_phosphosite_target").to_csv(OUT / "manifests/cptac_phosphosite_output_order.tsv", sep="\t", index=False)
    pd.Series(data["rppa_names"], name="tcpa_phospho_antibody").to_csv(OUT / "manifests/tcpa_phospho_antibody_output_order.tsv", sep="\t", index=False)
    pd.Series(data["cancer_levels"], name="cancer_level").to_csv(OUT / "manifests/cancer_levels.tsv", sep="\t", index=False)

    summary = {
        "model_name": "rna2phospho_dual_output_final_v1",
        "output_dir": str(OUT),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_loss),
        "final_val_loss": float(val_loss),
        "n_samples": int(x_z.shape[0]),
        "n_cptac_samples": int(data["cptac_n"]),
        "n_tcga_tcpa_samples": int(data["tcga_n"]),
        "n_genes": len(data["genes"]),
        "n_cptac_phosphosites": len(data["site_names"]),
        "n_tcpa_phospho_antibodies": len(data["rppa_names"]),
        "cptac_site_val_median_spearman": float(site_metrics["spearman"].median(skipna=True)),
        "tcpa_phospho_antibody_val_median_spearman": float(rppa_metrics["spearman"].median(skipna=True)),
        "checkpoint_sha256": sha256(best_path),
        "data_sources": {
            "cptac_pdc": str(CPTAC_DATA),
            "tcga_tcpa_loader": str(TCPA_TRAIN_SCRIPT),
        },
    }
    (OUT / "logs/final_model_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(
        [
            {"path": str(best_path), "sha256": summary["checkpoint_sha256"]},
            {"path": str(CPTAC_DATA / "LOCKED_FILE_HASHES.tsv"), "sha256": sha256(CPTAC_DATA / "LOCKED_FILE_HASHES.tsv")},
        ]
    ).to_csv(OUT / "LOCKED_MODEL_HASHES.tsv", sep="\t", index=False)
    (OUT / "MODEL_CARD.md").write_text(
        "\n".join(
            [
                "# RNA2Phospho dual-output final v1",
                "",
                "Input: bulk RNA expression matrix with gene symbols.",
                "",
                "Outputs:",
                "- CPTAC/PDC mass-spectrometry phosphosite predictions.",
                "- TCGA/TCPA phospho-RPPA antibody predictions.",
                "",
                "The two output heads are separate because phosphosite mass spectrometry and RPPA antibody signals have different measurement semantics.",
                "",
                f"Samples: {summary['n_samples']} total, {summary['n_cptac_samples']} CPTAC/PDC, {summary['n_tcga_tcpa_samples']} TCGA/TCPA.",
                f"Input genes: {summary['n_genes']}.",
                f"CPTAC phosphosite outputs: {summary['n_cptac_phosphosites']}.",
                f"TCPA phospho-antibody outputs: {summary['n_tcpa_phospho_antibodies']}.",
                f"Checkpoint SHA256: {summary['checkpoint_sha256']}.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
