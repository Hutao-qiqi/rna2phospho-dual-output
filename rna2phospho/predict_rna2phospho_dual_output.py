#!/usr/bin/env python3
"""Run a sealed RNA2Phospho dual-output model on a bulk RNA matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


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
        blocks = []
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
            h = h + h_new if h_new.shape == h.shape else h_new
        return self.site_head(h), self.rppa_head(h)


def read_matrix(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep="\t")
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-rna", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cancer", default="UNK_CANCER")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**checkpoint["config"])
    genes = list(checkpoint["genes"])
    cancer_levels = list(checkpoint["cancer_levels"])
    site_targets = list(checkpoint["cptac_phosphosite_targets"])
    rppa_targets = list(checkpoint["tcpa_phospho_antibodies"])

    x = read_matrix(Path(args.input_rna)).reindex(columns=genes)
    arr = x.to_numpy(dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, checkpoint["x_mean"])
    arr = (arr - checkpoint["x_mean"]) / checkpoint["x_std"]
    arr = arr.astype(np.float32)

    cancer = args.cancer if args.cancer in cancer_levels else "UNK_CANCER"
    cancer_id = cancer_levels.index(cancer)
    cancer_vec = np.full(arr.shape[0], cancer_id, dtype=np.int64)

    model = DualOutputPhosphoModel(
        n_gene=len(genes),
        n_cancer=len(cancer_levels),
        n_site=len(site_targets),
        n_rppa=len(rppa_targets),
        cfg=cfg,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    site_chunks = []
    rppa_chunks = []
    with torch.no_grad():
        for start in range(0, arr.shape[0], args.batch_size):
            stop = min(start + args.batch_size, arr.shape[0])
            xb = torch.tensor(arr[start:stop], dtype=torch.float32)
            cb = torch.tensor(cancer_vec[start:stop], dtype=torch.long)
            site_z, rppa_z = model(xb, cb)
            site = site_z.numpy() * checkpoint["site_std"] + checkpoint["site_mean"]
            rppa = rppa_z.numpy() * checkpoint["rppa_std"] + checkpoint["rppa_mean"]
            site_chunks.append(site)
            rppa_chunks.append(rppa)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(np.vstack(site_chunks), index=x.index, columns=site_targets).to_csv(
        out_dir / "predicted_cptac_phosphosites.tsv", sep="\t"
    )
    pd.DataFrame(np.vstack(rppa_chunks), index=x.index, columns=rppa_targets).to_csv(
        out_dir / "predicted_tcpa_phospho_antibodies.tsv", sep="\t"
    )


if __name__ == "__main__":
    main()
