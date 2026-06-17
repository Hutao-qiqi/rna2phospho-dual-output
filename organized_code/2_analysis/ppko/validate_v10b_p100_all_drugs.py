import json
import os
import argparse
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

from pretrain_v10b_strong300 import AttentionPriorManifoldV10, build_global_signed_inputs


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "SCP682_DATA_ROOT",
        r"D:\data\lsy\vm_lsy_parent\lsy" if os.name == "nt" else "/mnt/d/data/lsy/vm_lsy_parent/lsy",
    )
)
MODEL = PACKAGE_ROOT / "models" / "scp682_ppko_v10b_strong300_best.pt"
GRAPH = DEFAULT_DATA_ROOT / "01_data" / "pathway_prior" / "intermediate" / "global_phosphoprotein_heterograph_v10_measured_string700_top50"
TRAIN = DEFAULT_DATA_ROOT / "01_data" / "single_cell" / "intermediate" / "phospho_perturb" / "decryptm_comparison_delta_v8"
P100 = DEFAULT_DATA_ROOT / "01_data" / "single_cell" / "intermediate" / "phospho_perturb" / "lincs_p100_comparison_delta_v7"
MAP_DIR = DEFAULT_DATA_ROOT / "02_results" / "single_cell" / "20260519_scp682_ppko_1_decryptm_network_v8_full_p100_shared_site_validation" / "tables"
OUT = PACKAGE_ROOT / "validation_outputs" / "p100_all_drugs"

TARGETS = {
    "Gefitinib": "EGFR",
    "Erlotinib": "EGFR",
    "Lapatinib": "EGFR;ERBB2",
    "Dasatinib": "ABL1;SRC;LYN;LCK;HCK;KIT;PDGFRB;YES1",
    "Bosutinib": "ABL1;SRC;LYN;HCK",
    "Imatinib": "ABL1;KIT;PDGFRB",
    "Nilotinib": "ABL1;KIT;PDGFRB",
    "Ponatinib": "ABL1;KIT",
    "Trametinib": "MAP2K1;MAP2K2",
    "Selumetinib": "MAP2K1;MAP2K2",
    "dactolisib": "PIK3CA;PIK3CB;MTOR",
    "Bortezomib": "PSMB5",
    "Carfilzomib": "PSMB5",
    "Vorinostat": "HDAC1;HDAC2;HDAC3;HDAC6",
    "vorinostat": "HDAC1;HDAC2;HDAC3;HDAC6",
    "curcumin": "CREBBP;EP300",
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def cosine(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return np.nan
    den = np.linalg.norm(a[ok]) * np.linalg.norm(b[ok])
    return float(np.dot(a[ok], b[ok]) / den) if den > 0 else np.nan


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return float(pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman"))


def direction(a, b):
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-6)
    if ok.sum() == 0:
        return np.nan
    return float((np.sign(a[ok]) == np.sign(b[ok])).mean())


def responsive_mask(real, frac=0.2):
    ok = np.isfinite(real)
    if ok.sum() < 5:
        return np.zeros_like(real, dtype=bool)
    k = max(1, int(np.ceil(ok.sum() * frac)))
    idx = np.where(ok)[0]
    top = idx[np.argsort(-np.abs(real[idx]))[:k]]
    m = np.zeros_like(real, dtype=bool)
    m[top] = True
    return m


def topk_recall(pred, real, frac=0.2):
    ok = np.isfinite(pred) & np.isfinite(real)
    pred = pred[ok]
    real = real[ok]
    if len(real) < 5:
        return np.nan, np.nan
    k = max(1, int(np.ceil(len(real) * frac)))
    t = set(np.argsort(-np.abs(real))[:k].tolist())
    p = set(np.argsort(-np.abs(pred))[:k].tolist())
    hit = sorted(t & p)
    return float(len(hit) / k), direction(pred[hit], real[hit]) if hit else np.nan


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    ok = np.isfinite(scores)
    scores = scores[ok]
    labels = labels[ok]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def eval_pred(pred, real):
    ok = np.isfinite(pred) & np.isfinite(real)
    resp = responsive_mask(real)
    recall, topdir = topk_recall(pred, real)
    bottom = np.zeros_like(real, dtype=bool)
    idx = np.where(ok)[0]
    if len(idx) >= 5:
        bottom[idx[np.argsort(np.abs(real[idx]))[: max(1, int(len(idx) * 0.5))]]] = True
    return {
        "all_cosine": cosine(pred[ok], real[ok]),
        "all_spearman": spearman(pred[ok], real[ok]),
        "all_direction": direction(pred[ok], real[ok]),
        "responsive20_cosine": cosine(pred[resp], real[resp]),
        "responsive20_spearman": spearman(pred[resp], real[resp]),
        "responsive20_direction": direction(pred[resp], real[resp]),
        "topk_recall": recall,
        "topk_direction": topdir,
        "response_auroc_abs_pred": auroc(np.abs(pred[ok]), resp[ok]),
        "pred_abs": float(np.nanmean(np.abs(pred[ok]))),
        "real_abs": float(np.nanmean(np.abs(real[ok]))),
        "responsive20_pred_abs": float(np.nanmean(np.abs(pred[resp]))) if resp.any() else np.nan,
        "bottom50_pred_abs": float(np.nanmean(np.abs(pred[bottom]))) if bottom.any() else np.nan,
    }


def make_context_cache(targets, graph_dir):
    rows = [{"target_genes": t, "action_type": "inhibition"} for t in targets]
    df = pd.DataFrame(rows)
    _, pc, gp, _ = build_global_signed_inputs(df, graph_dir)
    return {t: (pc[i], gp[i]) for i, t in enumerate(targets)}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--graph-dir", default=str(GRAPH))
    ap.add_argument("--train-dir", default=str(TRAIN))
    ap.add_argument("--p100-dir", default=str(P100))
    ap.add_argument("--map-dir", default=str(MAP_DIR))
    ap.add_argument("--output-dir", default=str(OUT))
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    graph_dir = Path(args.graph_dir)
    train_dir = Path(args.train_dir)
    p100_dir = Path(args.p100_dir)
    map_dir = Path(args.map_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir / "tables")
    ensure_dir(out_dir / "reports")
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    sites = pd.DataFrame(ckpt["sites"])
    proteins = pd.DataFrame(ckpt["proteins"])
    model = AttentionPriorManifoldV10(len(sites), len(proteins), hidden=ckpt["args"]["hidden"], latent=ckpt["args"]["latent"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    p_delta = np.load(p100_dir / "arrays" / "delta_matrix.npy").astype(np.float32)
    p_base = np.load(p100_dir / "arrays" / "baseline_matrix.npy").astype(np.float32)
    p_valid = np.load(p100_dir / "arrays" / "valid_mask.npy").astype(bool)
    p_comp = pd.read_csv(p100_dir / "tables" / "comparison_table.tsv", sep="\t")
    mapping = pd.read_csv(map_dir / "p100_to_decryptm_shared_site_map.tsv", sep="\t")
    pidx = mapping["p100_target_index"].to_numpy(dtype=int)
    didx = mapping["decryptm_target_index"].to_numpy(dtype=int)
    n_sites = len(sites)

    train_base = np.load(train_dir / "arrays" / "baseline_matrix.npy").astype(np.float32)
    train_valid = np.load(train_dir / "arrays" / "valid_mask.npy").astype(bool)
    generic = np.zeros(n_sites, dtype=np.float32)
    for j in range(n_sites):
        vals = train_base[train_valid[:, j], j]
        generic[j] = float(np.nanmedian(vals)) if len(vals) else 0.0

    unique_targets = sorted(set(TARGETS.values()) - {""})
    shuffled = {t: unique_targets[(i + 3) % len(unique_targets)] for i, t in enumerate(unique_targets)}
    cache = make_context_cache([""] + unique_targets, graph_dir)
    rows = []
    for i, row in p_comp.iterrows():
        drug = str(row.get("perturbation", ""))
        target = TARGETS.get(drug, "")
        use = p_valid[i, pidx]
        if int(use.sum()) < 5:
            continue
        base = generic.copy()
        valid = np.zeros(n_sites, dtype=bool)
        real = np.full(n_sites, np.nan, dtype=np.float32)
        base[didx[use]] = p_base[i, pidx[use]]
        valid[didx[use]] = True
        real[didx[use]] = p_delta[i, pidx[use]]
        modes = {"true": target, "zero": "", "shuffled": shuffled.get(target, unique_targets[i % len(unique_targets)])}
        for mode, genes in modes.items():
            pc, gp = cache[genes]
            with torch.no_grad():
                pred, graph, latent, resid, dz, attention = model(
                    torch.as_tensor(base, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.as_tensor(valid, dtype=torch.bool, device=device).unsqueeze(0),
                    torch.as_tensor(pc, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.as_tensor(gp, dtype=torch.float32, device=device).unsqueeze(0),
                )
            pred_np = pred.detach().cpu().numpy()[0, didx[use]]
            real_np = real[didx[use]]
            metric = eval_pred(pred_np, real_np)
            metric.update({
                "mode": mode,
                "perturbation": drug,
                "target_genes": genes,
                "n_shared_sites": int(use.sum()),
                "graph_abs": float(np.nanmean(np.abs(graph.detach().cpu().numpy()[0, didx[use]]))),
                "latent_abs": float(np.nanmean(np.abs(latent.detach().cpu().numpy()[0, didx[use]]))),
                "residual_abs": float(np.nanmean(np.abs(resid.detach().cpu().numpy()[0, didx[use]]))),
                "attention_mean": float(np.nanmean(attention.detach().cpu().numpy()[0, didx[use]])),
            })
            rows.append(metric)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "tables" / "v10b_p100_all_drug_metrics.tsv", sep="\t", index=False)
    agg_cols = [c for c in metrics.columns if c not in {"mode", "perturbation", "target_genes"}]
    numeric = [c for c in agg_cols if pd.api.types.is_numeric_dtype(metrics[c])]
    mode_summary = metrics.groupby("mode", as_index=False).agg(n=("perturbation", "count"), **{c: (c, "mean") for c in numeric if c != "n_shared_sites"})
    drug_summary = metrics.groupby(["mode", "perturbation", "target_genes"], as_index=False).agg(n=("perturbation", "count"), **{c: (c, "mean") for c in numeric if c != "n_shared_sites"})
    mode_summary.to_csv(out_dir / "tables" / "v10b_p100_all_drug_mode_summary.tsv", sep="\t", index=False)
    drug_summary.to_csv(out_dir / "tables" / "v10b_p100_all_drug_drug_summary.tsv", sep="\t", index=False)
    report = {
        "model": "SCP682-PPKO V10B strong300 transferable model",
        "model_path": str(model_path),
        "graph_dir": str(graph_dir),
        "train_dir": str(train_dir),
        "p100_dir": str(p100_dir),
        "map_dir": str(map_dir),
        "mode_summary": mode_summary.to_dict(orient="records"),
    }
    (out_dir / "reports" / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
