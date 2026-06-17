#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


def _safe_row_normalize(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=1, keepdim=True)
    den = torch.sqrt((x * x).sum(dim=1, keepdim=True)).clamp_min(1e-6)
    return x / den


class SCP682GraphDecoder(nn.Module):
    def __init__(self, hidden: int = 64, latent: int = 32, embd_dim: int = 32, shrinkage: float = 0.3):
        super().__init__()
        self.shrinkage = float(shrinkage)
        self.site_prior_proj = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.baseline_proj = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.site_proj = nn.Sequential(nn.Linear(embd_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.sample_proj = nn.Sequential(nn.Linear(embd_dim, latent), nn.GELU(), nn.LayerNorm(latent))
        self.prior_attention = nn.Sequential(
            nn.LayerNorm(hidden + hidden + latent + 1),
            nn.Linear(hidden + hidden + latent + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.graph_decoder = nn.Sequential(
            nn.LayerNorm(hidden + hidden + latent + 1),
            nn.Linear(hidden + hidden + latent + 1, hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, 1),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(hidden + hidden + latent),
            nn.Linear(hidden + hidden + latent, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )
        self.graph_scale = nn.Parameter(torch.tensor(-0.2))
        self.residual_scale = nn.Parameter(torch.tensor(-1.4))

    def decode(self, row_embed: torch.Tensor, col_embed: torch.Tensor, baseline: torch.Tensor, mask: torch.Tensor, site_prior: torch.Tensor):
        base_h = self.baseline_proj(torch.stack([baseline, mask.float()], dim=-1))
        site_h = self.site_proj(row_embed).unsqueeze(0).expand(baseline.shape[0], -1, -1)
        sample_z = self.sample_proj(col_embed).unsqueeze(1).expand(-1, baseline.shape[1], -1)
        prior = site_prior.unsqueeze(0).expand(baseline.shape[0], -1)
        prior_h = self.site_prior_proj(prior.unsqueeze(-1))
        graph_input = torch.cat([base_h + prior_h, site_h, sample_z, prior.unsqueeze(-1)], dim=-1)
        attention = torch.sigmoid(self.prior_attention(graph_input)).squeeze(-1)
        graph_delta = attention * self.graph_decoder(graph_input).squeeze(-1) * torch.sigmoid(self.graph_scale)
        residual_delta = self.residual(torch.cat([base_h, site_h, sample_z], dim=-1)).squeeze(-1) * torch.sigmoid(self.residual_scale)
        delta = graph_delta + residual_delta
        pred = baseline + self.shrinkage * delta
        return pred, delta, graph_delta, residual_delta, attention


class SCP682GraphRuntime:
    def __init__(self, package_dir: Path, device: str = "auto", knn: int = 25, temperature: float = 0.08, batch_size: int = 4):
        self.package_dir = Path(package_dir).resolve()
        self.device = torch.device("cuda:0" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.knn = int(knn)
        self.temperature = float(temperature)
        self.batch_size = int(batch_size)
        payload = torch.load(self.package_dir / "models" / "scp682_graph_runtime_state.pt", map_location="cpu", weights_only=False)
        self.targets = [str(x) for x in payload["targets"]]
        self.samples = [str(x) for x in payload["samples"]]
        self.row_embed = payload["row_embed"].float().to(self.device)
        self.train_col_embed = payload["train_col_embed"].float().to(self.device)
        self.site_prior = payload["site_prior"].float().to(self.device)
        self.train_baseline = payload["train_v4_baseline"].float().to(self.device)
        self.train_mask = torch.isfinite(self.train_baseline)
        self.train_baseline = torch.nan_to_num(self.train_baseline, nan=0.0)
        self.train_norm = _safe_row_normalize(self.train_baseline)
        cfg = payload.get("decoder_config", {"hidden": 64, "latent": 32, "embd_dim": 32, "shrinkage": 0.3})
        self.decoder = SCP682GraphDecoder(**cfg).to(self.device)
        self.decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
        self.decoder.eval()
        self.meta = payload.get("meta", {})

    def _external_col_embed(self, baseline: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(baseline, nan=0.0)
        x_norm = _safe_row_normalize(x)
        embeds = []
        for start in range(0, x_norm.shape[0], max(1, self.batch_size)):
            xb = x_norm[start : start + self.batch_size]
            sim = xb @ self.train_norm.T
            k = min(self.knn, sim.shape[1])
            vals, idx = torch.topk(sim, k=k, dim=1)
            weight = torch.softmax(vals / max(self.temperature, 1e-6), dim=1)
            neigh = self.train_col_embed.index_select(0, idx.reshape(-1)).reshape(idx.shape[0], idx.shape[1], -1)
            embeds.append((weight.unsqueeze(-1) * neigh).sum(dim=1))
        return torch.cat(embeds, dim=0)

    def predict(self, v4_baseline: pd.DataFrame) -> dict[str, pd.DataFrame]:
        baseline_df = v4_baseline.reindex(columns=self.targets).copy()
        baseline_np = baseline_df.to_numpy(dtype=np.float32)
        mask_np = np.isfinite(baseline_np)
        baseline_np = np.nan_to_num(baseline_np, nan=0.0).astype(np.float32)
        baseline = torch.tensor(baseline_np, dtype=torch.float32, device=self.device)
        mask = torch.tensor(mask_np, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            col_embed = self._external_col_embed(baseline)
            pred_chunks = []
            delta_chunks = []
            attention_chunks = []
            for start in range(0, baseline.shape[0], max(1, self.batch_size)):
                end = min(baseline.shape[0], start + self.batch_size)
                pred, delta, _, _, attention = self.decoder.decode(
                    self.row_embed,
                    col_embed[start:end],
                    baseline[start:end],
                    mask[start:end],
                    self.site_prior,
                )
                pred_chunks.append(pred.detach().cpu())
                delta_chunks.append(delta.detach().cpu())
                attention_chunks.append(attention.detach().cpu())
        pred_df = pd.DataFrame(torch.cat(pred_chunks, dim=0).numpy().astype(np.float32), index=baseline_df.index, columns=self.targets)
        delta_df = pd.DataFrame(torch.cat(delta_chunks, dim=0).numpy().astype(np.float32), index=baseline_df.index, columns=self.targets)
        attention_df = pd.DataFrame(torch.cat(attention_chunks, dim=0).numpy().astype(np.float32), index=baseline_df.index, columns=self.targets)
        pred_df.index.name = "sample_id"
        delta_df.index.name = "sample_id"
        attention_df.index.name = "sample_id"
        return {"scp682": pred_df, "graph_delta": delta_df, "graph_attention": attention_df}
