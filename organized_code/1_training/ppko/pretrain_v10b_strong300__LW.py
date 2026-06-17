import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
DEFAULT_INPUT = ROOT / r"01_data\single_cell\intermediate\phospho_perturb\decryptm_comparison_delta_v8"
DEFAULT_GRAPH = ROOT / r"01_data\pathway_prior\intermediate\global_phosphoprotein_heterograph_v10_measured_string700_top50"
DEFAULT_OUT = ROOT / r"02_results\single_cell\20260520_scp682_ppko_1_attention_prior_v10b_strong_contrast"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def split_genes(value):
    genes = []
    for part in str(value).replace(",", ";").replace("/", ";").split(";"):
        g = "".join(ch for ch in part.upper().strip() if ch.isalnum() or ch in {"-", "."})
        if g and g != "NAN":
            genes.append(g)
    return sorted(set(genes))


def row_normalize(mat):
    mat = mat.astype(np.float32, copy=True)
    denom = np.maximum(1.0, np.abs(mat).sum(axis=1, keepdims=True))
    return mat / denom


def cosine_loss(pred, true, mask, eps=1e-8):
    m = mask.float()
    pred = pred * m
    true = true * m
    num = (pred * true).sum(dim=1)
    den = torch.sqrt(pred.square().sum(dim=1) + eps) * torch.sqrt(true.square().sum(dim=1) + eps)
    return 1.0 - (num / den.clamp_min(eps)).mean()


def masked_huber(pred, true, mask):
    loss = F.smooth_l1_loss(pred, true, reduction="none")
    return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def masked_huber_per_sample(pred, true, mask):
    loss = F.smooth_l1_loss(pred, true, reduction="none") * mask.float()
    return loss.sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)


def cosine_loss_per_sample(pred, true, mask, eps=1e-8):
    m = mask.float()
    pred = pred * m
    true = true * m
    num = (pred * true).sum(dim=1)
    den = torch.sqrt(pred.square().sum(dim=1) + eps) * torch.sqrt(true.square().sum(dim=1) + eps)
    return 1.0 - num / den.clamp_min(eps)


def cosine_np(a, b, mask):
    ok = mask.astype(bool) & np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return np.nan
    den = np.linalg.norm(a[ok]) * np.linalg.norm(b[ok])
    return float(np.dot(a[ok], b[ok]) / den) if den > 0 else np.nan


def direction_np(a, b, mask):
    ok = mask.astype(bool) & np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-6)
    if ok.sum() == 0:
        return np.nan
    return float((np.sign(a[ok]) == np.sign(b[ok])).mean())


def build_global_signed_inputs(comp, graph_dir):
    graph_dir = Path(graph_dir)
    proteins = pd.read_csv(graph_dir / "tables" / "protein_nodes.tsv", sep="\t")
    gene_to_i = {g: int(i) for i, g in zip(proteins["protein_index"], proteins["gene"].astype(str))}
    n_protein = len(proteins)

    signed_pp = np.load(graph_dir / "arrays" / "signed_protein_protein_matrix.npy").astype(np.float32)
    unsigned_pp = np.load(graph_dir / "arrays" / "unsigned_protein_protein_matrix.npy").astype(np.float32)
    signed_rs = np.load(graph_dir / "arrays" / "signed_regulator_site_matrix.npy").astype(np.float32)
    membership = np.load(graph_dir / "arrays" / "protein_site_membership.npy").astype(np.float32)
    signed_pp = row_normalize(signed_pp)
    unsigned_pp = row_normalize(unsigned_pp)
    signed_rs = row_normalize(signed_rs)
    membership = row_normalize(membership)

    signed_protein_seed = np.zeros((len(comp), n_protein), dtype=np.float32)
    for i, row in comp.iterrows():
        sign = 1.0 if str(row.get("action_type", "inhibition")).lower() == "activation" else -1.0
        genes = split_genes(row.get("target_genes", ""))
        matched = [g for g in genes if g in gene_to_i]
        for g in matched:
            signed_protein_seed[i, gene_to_i[g]] = sign / max(1, len(matched))

    p0 = signed_protein_seed
    p1 = p0 @ signed_pp + 0.35 * (p0 @ unsigned_pp)
    p2 = p1 @ signed_pp + 0.20 * (p1 @ unsigned_pp)
    protein_context = p0 + 0.75 * p1 + 0.45 * p2
    site_graph_prior = protein_context @ signed_rs + 0.10 * (protein_context @ membership)
    site_graph_prior = site_graph_prior.astype(np.float32)
    for i in range(site_graph_prior.shape[0]):
        mx = float(np.nanmax(np.abs(site_graph_prior[i]))) if np.any(site_graph_prior[i]) else 0.0
        if mx > 0:
            site_graph_prior[i] = np.clip(site_graph_prior[i] / mx, -1.0, 1.0)

    return signed_protein_seed.astype(np.float32), protein_context.astype(np.float32), site_graph_prior, proteins


class AttentionPriorManifoldV10(nn.Module):
    def __init__(self, n_sites, n_proteins, hidden=192, latent=96, protein_latent=128):
        super().__init__()
        self.n_sites = n_sites
        self.site_embedding = nn.Parameter(torch.randn(n_sites, hidden) * 0.015)
        self.baseline_proj = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.graph_prior_proj = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.encoder = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, latent), nn.GELU())
        self.protein_encoder = nn.Sequential(nn.Linear(n_proteins, protein_latent), nn.GELU(), nn.LayerNorm(protein_latent), nn.Linear(protein_latent, hidden), nn.GELU())
        self.prior_attention = nn.Sequential(nn.LayerNorm(hidden + hidden + latent + 1), nn.Linear(hidden + hidden + latent + 1, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.graph_decoder = nn.Sequential(nn.LayerNorm(hidden + hidden + latent + 1), nn.Linear(hidden + hidden + latent + 1, hidden), nn.GELU(), nn.Dropout(0.08), nn.Linear(hidden, 1))
        self.prior_scale = nn.Parameter(torch.tensor(-0.5))
        self.learned_graph_scale = nn.Parameter(torch.tensor(-0.2))
        self.vector_field = nn.Sequential(nn.Linear(latent + hidden, hidden), nn.GELU(), nn.Dropout(0.10), nn.Linear(hidden, latent))
        self.latent_decoder = nn.Sequential(nn.LayerNorm(hidden + hidden + latent), nn.Linear(hidden + hidden + latent, hidden), nn.GELU(), nn.Dropout(0.10), nn.Linear(hidden, 1))
        self.residual = nn.Sequential(nn.LayerNorm(hidden + hidden), nn.Linear(hidden + hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.latent_scale = nn.Parameter(torch.tensor(-0.9))
        self.residual_scale = nn.Parameter(torch.tensor(-1.8))

    def encode_state(self, baseline, mask, graph_prior):
        x = torch.where(mask, baseline, torch.zeros_like(baseline))
        h = self.baseline_proj(torch.stack([x, mask.float()], dim=-1)) + self.site_embedding.unsqueeze(0) + self.graph_prior_proj(graph_prior.unsqueeze(-1))
        denom = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        z = (self.encoder(h) * mask.float().unsqueeze(-1)).sum(dim=1) / denom
        return h, z

    def forward(self, baseline, mask, protein_context, graph_prior):
        h, z = self.encode_state(baseline, mask, graph_prior)
        ctx = self.protein_encoder(protein_context)
        strength = protein_context.abs().sum(dim=1, keepdim=True).clamp(max=5.0) / 5.0
        ctx_site = ctx.unsqueeze(1).expand(-1, self.n_sites, -1)
        z_site = z.unsqueeze(1).expand(-1, self.n_sites, -1)
        prior_abs = graph_prior.abs().unsqueeze(-1)
        graph_input = torch.cat([h, ctx_site, z_site, prior_abs], dim=-1)
        attention = torch.sigmoid(self.prior_attention(graph_input)).squeeze(-1)
        graph_base = self.graph_decoder(graph_input).squeeze(-1)
        graph_delta = attention * (graph_prior * torch.sigmoid(self.prior_scale) + graph_base * torch.sigmoid(self.learned_graph_scale))
        dz = self.vector_field(torch.cat([z, ctx], dim=1)) * strength
        latent_delta = self.latent_decoder(torch.cat([h, ctx_site, dz.unsqueeze(1).expand(-1, self.n_sites, -1)], dim=-1)).squeeze(-1)
        latent_delta = latent_delta * torch.sigmoid(self.latent_scale) * strength
        residual_delta = self.residual(torch.cat([h, ctx_site], dim=-1)).squeeze(-1) * torch.sigmoid(self.residual_scale) * strength
        pred = graph_delta + latent_delta + residual_delta
        return pred, graph_delta, latent_delta, residual_delta, dz, attention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    ap.add_argument("--graph-dir", default=str(DEFAULT_GRAPH))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=620)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--latent", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20260519)
    ap.add_argument("--contrast-margin", type=float, default=0.08)
    ap.add_argument("--contrast-weight", type=float, default=1.60)
    ap.add_argument("--zero-weight", type=float, default=0.08)
    ap.add_argument("--attention-l1", type=float, default=0.01)
    ap.add_argument("--negative-count", type=int, default=5)
    ap.add_argument("--negative-direction-weight", type=float, default=0.18)
    ap.add_argument("--negative-amplitude-weight", type=float, default=0.04)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    for sub in ("models", "tables", "reports", "logs"):
        ensure_dir(out / sub)
    inp = Path(args.input_dir)
    delta = np.load(inp / "arrays" / "delta_matrix.npy").astype(np.float32)
    baseline = np.load(inp / "arrays" / "baseline_matrix.npy").astype(np.float32)
    valid = np.load(inp / "arrays" / "valid_mask.npy").astype(bool)
    comp = pd.read_csv(inp / "tables" / "comparison_table.tsv", sep="\t")
    sites = pd.read_csv(inp / "tables" / "site_table.tsv", sep="\t")
    protein_seed, protein_context, graph_prior, proteins = build_global_signed_inputs(comp, args.graph_dir)

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    model = AttentionPriorManifoldV10(delta.shape[1], protein_context.shape[1], hidden=args.hidden, latent=args.latent).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tensors = {
        "baseline": torch.as_tensor(baseline, dtype=torch.float32, device=device),
        "delta": torch.as_tensor(delta, dtype=torch.float32, device=device),
        "valid": torch.as_tensor(valid, dtype=torch.bool, device=device),
        "protein_context": torch.as_tensor(protein_context, dtype=torch.float32, device=device),
        "graph_prior": torch.as_tensor(graph_prior, dtype=torch.float32, device=device),
    }
    n = len(comp)
    idx_all = np.arange(n)
    best = {"site_cosine": -1e9, "epoch": 0}
    logs = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(idx_all)
        model.train()
        losses = []
        for st in range(0, n, args.batch_size):
            idx = idx_all[st : st + args.batch_size]
            b = tensors["baseline"][idx]
            y = tensors["delta"][idx]
            m = tensors["valid"][idx]
            pc = tensors["protein_context"][idx]
            gp = tensors["graph_prior"][idx]
            pred, graph, latent, resid, dz, attention = model(b, m, pc, gp)
            negative_ranks = []
            negative_direction_terms = []
            negative_amp_terms = []
            for neg_i in range(max(1, args.negative_count)):
                if len(idx) > 1:
                    perm = torch.randperm(len(idx), device=device)
                    if torch.equal(perm, torch.arange(len(idx), device=device)):
                        perm = perm.roll(1)
                else:
                    perm = torch.arange(len(idx), device=device)
                shuf_pred, _, _, _, _, _ = model(b, m, pc[perm], gp[perm])
                negative_ranks.append(masked_huber_per_sample(shuf_pred, y, m) + 0.25 * cosine_loss_per_sample(shuf_pred, y, m))
                neg_dir = torch.relu((shuf_pred * y) * m.float()).sum(dim=1) / m.float().sum(dim=1).clamp_min(1.0)
                negative_direction_terms.append(neg_dir)
                negative_amp_terms.append((shuf_pred.abs() * m.float()).sum(dim=1) / m.float().sum(dim=1).clamp_min(1.0))
            zero_pred, _, _, _, _, _ = model(b, m, pc * 0, gp * 0)
            pred_loss = masked_huber(pred, y, m) + 0.35 * cosine_loss(pred, y, m)
            true_rank = masked_huber_per_sample(pred, y, m) + 0.25 * cosine_loss_per_sample(pred, y, m)
            negative_rank = torch.stack(negative_ranks, dim=0).min(dim=0).values
            contrast_loss = torch.relu(args.contrast_margin + true_rank - negative_rank).mean()
            negative_direction_loss = torch.stack(negative_direction_terms, dim=0).mean()
            true_amp = (pred.abs() * m.float()).sum(dim=1) / m.float().sum(dim=1).clamp_min(1.0)
            negative_amp = torch.stack(negative_amp_terms, dim=0).mean(dim=0)
            negative_amplitude_loss = torch.relu(negative_amp - true_amp.detach() * 0.75).mean()
            zero_loss = (zero_pred.square() * m.float()).sum() / m.float().sum().clamp_min(1.0)
            latent_penalty = latent.abs().mean() + 0.002 * dz.square().mean()
            residual_penalty = resid.abs().mean()
            attention_penalty = (attention * m.float()).sum() / m.float().sum().clamp_min(1.0)
            loss = (
                pred_loss
                + args.contrast_weight * contrast_loss
                + args.zero_weight * zero_loss
                + args.negative_direction_weight * negative_direction_loss
                + args.negative_amplitude_weight * negative_amplitude_loss
                + 0.04 * latent_penalty
                + 0.02 * residual_penalty
                + args.attention_l1 * attention_penalty
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                pred, graph, latent, resid, dz, attention = model(tensors["baseline"], tensors["valid"], tensors["protein_context"], tensors["graph_prior"])
                perm = torch.arange(n, device=device).roll(7)
                shuf_pred, _, _, _, _, _ = model(tensors["baseline"], tensors["valid"], tensors["protein_context"][perm], tensors["graph_prior"][perm])
            pred_np = pred.detach().cpu().numpy()
            shuf_np = shuf_pred.detach().cpu().numpy()
            graph_np = graph.detach().cpu().numpy()
            latent_np = latent.detach().cpu().numpy()
            resid_np = resid.detach().cpu().numpy()
            att_np = attention.detach().cpu().numpy()
            cos = np.nanmean([cosine_np(pred_np[i], delta[i], valid[i]) for i in range(n)])
            shuf_cos = np.nanmean([cosine_np(shuf_np[i], delta[i], valid[i]) for i in range(n)])
            direc = np.nanmean([direction_np(pred_np[i], delta[i], valid[i]) for i in range(n)])
            shuf_direc = np.nanmean([direction_np(shuf_np[i], delta[i], valid[i]) for i in range(n)])
            row = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "site_cosine": float(cos),
                "shuffled_site_cosine": float(shuf_cos),
                "target_cosine_margin": float(cos - shuf_cos),
                "direction_accuracy": float(direc),
                "shuffled_direction_accuracy": float(shuf_direc),
                "target_direction_margin": float(direc - shuf_direc),
                "pred_abs": float(np.nanmean(np.abs(pred_np[valid]))),
                "graph_abs": float(np.nanmean(np.abs(graph_np[valid]))),
                "latent_abs": float(np.nanmean(np.abs(latent_np[valid]))),
                "residual_abs": float(np.nanmean(np.abs(resid_np[valid]))),
                "attention_mean": float(np.nanmean(att_np[valid])),
                "attention_top10_mean": float(np.nanmean(np.sort(att_np[valid])[-max(1, int(valid.sum() * 0.10)):])),
                "elapsed_min": (time.time() - start) / 60.0,
            }
            logs.append(row)
            print(json.dumps(row), flush=True)
            if row["site_cosine"] > best["site_cosine"]:
                best = row
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "proteins": proteins.to_dict(orient="records"),
                    "sites": sites.to_dict(orient="records"),
                    "comparisons": comp.to_dict(orient="records"),
                    "args": vars(args),
                    "best": best,
                }, out / "models" / "scp682_ppko_attention_prior_v10_best.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "proteins": proteins.to_dict(orient="records"),
        "sites": sites.to_dict(orient="records"),
        "comparisons": comp.to_dict(orient="records"),
        "args": vars(args),
        "best": best,
    }, out / "models" / "scp682_ppko_attention_prior_v10_final.pt")
    pd.DataFrame(logs).to_csv(out / "tables" / "training_log.tsv", sep="\t", index=False)
    summary = {
        "model": "attention-prior phosphoprotein manifold V10",
        "n_comparisons": int(n),
        "n_sites": int(delta.shape[1]),
        "n_proteins": int(protein_context.shape[1]),
        "best": best,
        "final": logs[-1],
        "graph_dir": str(args.graph_dir),
    }
    (out / "reports" / "final_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
