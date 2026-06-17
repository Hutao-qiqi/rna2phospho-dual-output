from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch


DRUG_TARGETS = {
    "ADAGRASIB": "KRAS",
    "ARS1620": "KRAS",
    "BI3406": "SOS1",
    "MRTX1133": "KRAS",
    "MRTX1257": "KRAS",
    "RMC4630": "PTPN11",
    "SOTORASIB": "KRAS",
    "TEMUTERKIB": "MAPK1;MAPK3",
    "TRAMETINIB": "MAP2K1;MAP2K2",
}

DRUG_CLASS = {
    "ADAGRASIB": "KRAS",
    "ARS1620": "KRAS",
    "MRTX1133": "KRAS",
    "MRTX1257": "KRAS",
    "SOTORASIB": "KRAS",
    "BI3406": "SOS1",
    "RMC4630": "SHP2",
    "TEMUTERKIB": "ERK",
    "TRAMETINIB": "MEK",
}


def clean_sequence(value: object) -> str:
    text = str(value)
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.replace("_", "")
    return re.sub(r"[^A-Za-z]", "", text).upper()


def split_genes(value: object) -> list[str]:
    out = []
    for part in str(value).replace(",", ";").replace("/", ";").split(";"):
        gene = "".join(ch for ch in part.upper().strip() if ch.isalnum() or ch in {"-", "."})
        if gene and gene != "NAN":
            out.append(gene)
    return sorted(set(out))


def target_gene(row: pd.Series) -> str:
    molecule = str(row.get("molecule", ""))
    target_id = str(row.get("target_id", ""))
    if molecule and molecule.lower() != "nan":
        return molecule
    if "__" in target_id:
        return target_id.split("__", 1)[0]
    if "_" in target_id:
        return target_id.split("_", 1)[0]
    return ""


def build_sequence_index(sites: pd.DataFrame) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for _, row in sites.iterrows():
        seq = clean_sequence(row.get("modified_peptide", ""))
        if not seq:
            seq = clean_sequence(row.get("sequence", ""))
        if seq:
            out.setdefault(seq, []).append(int(row["target_index"]))
    return out


def robust_normalize(values: np.ndarray, scale: float, clip: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float32)
    if finite.sum() < 3:
        return out
    v = values[finite]
    center = float(np.nanmedian(v))
    q25, q75 = np.nanpercentile(v, [25, 75])
    spread = float((q75 - q25) / 1.349)
    if not np.isfinite(spread) or spread < 1e-6:
        spread = float(np.nanstd(v))
    if not np.isfinite(spread) or spread < 1e-6:
        spread = 1.0
    out[finite] = np.clip((values[finite] - center) / spread * scale, -clip, clip)
    return out.astype(np.float32)


def build_external_baseline(sub: pd.DataFrame, sequence_index: dict[str, list[int]], n_sites: int, scale: float, clip: float) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.zeros(n_sites, dtype=np.float32)
    mask = np.zeros(n_sites, dtype=bool)
    rows = []
    for _, row in sub.iterrows():
        raw = row.get("baseline_intensity", np.nan)
        if not np.isfinite(raw) or raw <= 0:
            continue
        for idx in sequence_index.get(str(row["norm_sequence"]), []):
            rows.append((idx, float(raw)))
    if not rows:
        return baseline, mask
    tmp = pd.DataFrame(rows, columns=["idx", "raw"])
    agg = tmp.groupby("idx", as_index=False)["raw"].median()
    values = robust_normalize(agg["raw"].to_numpy(dtype=np.float32), scale, clip)
    idx = agg["idx"].to_numpy(dtype=int)
    baseline[idx] = values
    mask[idx] = True
    return baseline, mask


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return np.nan
    den = np.linalg.norm(a[ok]) * np.linalg.norm(b[ok])
    if den <= 0:
        return np.nan
    return float(np.dot(a[ok], b[ok]) / den)


def direction(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-9)
    if ok.sum() == 0:
        return np.nan
    return float((np.sign(a[ok]) == np.sign(b[ok])).mean())


def top_fraction_mask(values: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    mask = np.zeros(len(values), dtype=bool)
    n = int(ok.sum())
    if n == 0:
        return mask
    k = max(1, int(math.ceil(n * fraction)))
    idx = np.where(ok)[0]
    mask[idx[np.argsort(-np.abs(values[idx]))[:k]]] = True
    return mask


def metric_row(comp: str, subset: str, sub: pd.DataFrame) -> dict[str, object]:
    pred = sub["predicted_delta"].to_numpy(dtype=np.float64)
    obs = sub["observed_delta"].to_numpy(dtype=np.float64)
    spear = pd.Series(pred).corr(pd.Series(obs), method="spearman") if len(sub) >= 3 else np.nan
    return {
        "comparison": comp,
        "subset": subset,
        "cell_line": sub["cell_line"].iloc[0],
        "drug_code": sub["drug_code"].iloc[0],
        "drug": sub["drug"].iloc[0],
        "drug_class": sub["drug_class"].iloc[0],
        "target_genes": sub["target_genes"].iloc[0],
        "n_overlap_sites": int(len(sub)),
        "site_cosine": cosine(pred, obs),
        "spearman": float(spear) if pd.notna(spear) else np.nan,
        "direction_accuracy": direction(pred, obs),
        "observed_delta_mean_abs": float(np.mean(np.abs(obs))),
        "predicted_delta_mean_abs": float(np.mean(np.abs(pred))),
        "observed_positive_fraction": float(np.mean(obs > 0)),
        "predicted_positive_fraction": float(np.mean(pred > 0)),
    }


def evaluate(site: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    for comp, sub in site.groupby("comparison", sort=True):
        metrics.append(metric_row(comp, "all", sub))
        reg = sub[sub["regulated"].astype(bool)]
        if len(reg) >= 3:
            metrics.append(metric_row(comp, "regulated", reg))
        resp = sub[sub["is_responsive20"].astype(bool)]
        if len(resp) >= 3:
            metrics.append(metric_row(comp, "responsive20", resp))
        pred_top = sub[sub["is_predicted20"].astype(bool)]
        if len(pred_top) >= 3:
            metrics.append(metric_row(comp, "predicted20", pred_top))
    metric = pd.DataFrame(metrics)
    if metric.empty:
        return metric, pd.DataFrame()
    drug_summary = (
        metric.groupby(["subset", "drug_class", "drug_code", "drug", "target_genes"], as_index=False)
        .agg(
            n_comparisons=("comparison", "count"),
            mean_n_sites=("n_overlap_sites", "mean"),
            mean_site_cosine=("site_cosine", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_direction_accuracy=("direction_accuracy", "mean"),
            mean_observed_abs=("observed_delta_mean_abs", "mean"),
            mean_predicted_abs=("predicted_delta_mean_abs", "mean"),
        )
        .sort_values(["subset", "drug_class", "drug_code"])
    )
    return metric, drug_summary


def parse_args() -> argparse.Namespace:
    root_default = r"D:\data\lsy\vm_lsy_parent\lsy" if os.name == "nt" else "/mnt/d/data/lsy/vm_lsy_parent/lsy"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("SCP682_DATA_ROOT", root_default))
    parser.add_argument("--package-root", default=None)
    parser.add_argument("--observed", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--external-baseline-scale", type=float, default=0.50)
    parser.add_argument("--external-baseline-clip", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    package_root = Path(args.package_root) if args.package_root else data_root / "SCP682_PPKO_V10B_transferable"
    scripts_dir = package_root / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from pretrain_v10b_strong300 import AttentionPriorManifoldV10, build_global_signed_inputs

    observed_path = Path(args.observed) if args.observed else data_root / "02_results" / "single_cell" / "20260519_scp682_ppko_1_external_bulk_validation_v6_cophee_atlas" / "tables" / "pxd063604_observed_long.tsv"
    model_path = package_root / "models" / "scp682_ppko_v10b_strong300_best.pt"
    graph_dir = data_root / "01_data" / "pathway_prior" / "intermediate" / "global_phosphoprotein_heterograph_v10_measured_string700_top50"
    out_dir = Path(args.output_dir) if args.output_dir else data_root / "02_results" / "single_cell" / "20260531_scp682_ppko_v10b_pxd063604_sitelevel"
    tables = out_dir / "tables"
    reports = out_dir / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    sites = pd.DataFrame(ckpt["sites"])
    proteins = pd.DataFrame(ckpt["proteins"])
    model = AttentionPriorManifoldV10(len(sites), len(proteins), hidden=ckpt["args"]["hidden"], latent=ckpt["args"]["latent"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    observed = pd.read_csv(observed_path, sep="\t")
    observed["drug_code"] = observed["drug_code"].astype(str).str.upper()
    observed = observed[observed["drug_code"].isin(DRUG_TARGETS)].copy()
    observed["target_genes"] = observed["drug_code"].map(DRUG_TARGETS)
    observed["drug_class"] = observed["drug_code"].map(DRUG_CLASS)
    observed["norm_sequence"] = observed["norm_sequence"].map(clean_sequence)
    sequence_index = build_sequence_index(sites)

    context_rows = [{"target_genes": g, "action_type": "inhibition"} for g in sorted(set(DRUG_TARGETS.values()))]
    context_df = pd.DataFrame(context_rows)
    _, protein_context, graph_prior, _ = build_global_signed_inputs(context_df, graph_dir)
    context_cache = {g: (protein_context[i], graph_prior[i]) for i, g in enumerate(context_df["target_genes"])}

    pred_cache: dict[str, dict[str, np.ndarray | int | float]] = {}
    descriptor_rows = []
    site_rows = []

    with torch.no_grad():
        for comp, sub in observed.groupby("comparison", sort=True):
            drug_code = str(sub["drug_code"].iloc[0]).upper()
            target_genes = DRUG_TARGETS[drug_code]
            baseline, mask = build_external_baseline(
                sub,
                sequence_index,
                len(sites),
                args.external_baseline_scale,
                args.external_baseline_clip,
            )
            if not mask.any():
                continue
            pc, gp = context_cache[target_genes]
            pred, graph, latent, residual, _, attention = model(
                torch.as_tensor(baseline, dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0),
                torch.as_tensor(pc, dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(gp, dtype=torch.float32, device=device).unsqueeze(0),
            )
            pred_np = pred.detach().cpu().numpy()[0]
            graph_np = graph.detach().cpu().numpy()[0]
            latent_np = latent.detach().cpu().numpy()[0]
            residual_np = residual.detach().cpu().numpy()[0]
            attention_np = attention.detach().cpu().numpy()[0]
            pred_cache[comp] = {
                "pred": pred_np,
                "graph": graph_np,
                "latent": latent_np,
                "residual": residual_np,
                "attention": attention_np,
                "n_external_baseline_sites": int(mask.sum()),
            }
            descriptor_rows.append(
                {
                    "comparison": comp,
                    "cell_line": sub["cell_line"].iloc[0],
                    "drug_code": drug_code,
                    "drug": sub["drug"].iloc[0],
                    "drug_class": DRUG_CLASS[drug_code],
                    "target_genes": target_genes,
                    "action_type": "inhibition",
                    "baseline_source": "external_low_dose_or_first_channel",
                    "n_external_baseline_sites": int(mask.sum()),
                    "mean_abs_pred_delta": float(np.mean(np.abs(pred_np))),
                    "mean_abs_graph_delta": float(np.mean(np.abs(graph_np))),
                    "mean_abs_latent_delta": float(np.mean(np.abs(latent_np))),
                    "mean_abs_residual_delta": float(np.mean(np.abs(residual_np))),
                    "attention_mean": float(np.mean(attention_np)),
                }
            )

    for _, row in observed.iterrows():
        comp = row["comparison"]
        cached = pred_cache.get(comp)
        if cached is None:
            continue
        matches = sequence_index.get(row["norm_sequence"], [])
        if not matches:
            continue
        pred_np = cached["pred"]
        graph_np = cached["graph"]
        latent_np = cached["latent"]
        residual_np = cached["residual"]
        attention_np = cached["attention"]
        site_rows.append(
            {
                **row.to_dict(),
                "n_model_matches": int(len(matches)),
                "model_target_indices": ";".join(str(i) for i in matches),
                "predicted_delta": float(np.mean(pred_np[matches])),
                "graph_delta": float(np.mean(graph_np[matches])),
                "latent_delta": float(np.mean(latent_np[matches])),
                "residual_delta": float(np.mean(residual_np[matches])),
                "attention": float(np.mean(attention_np[matches])),
                "predicted_abs_delta": float(abs(np.mean(pred_np[matches]))),
                "direction_match": bool(np.sign(np.mean(pred_np[matches])) == np.sign(float(row["observed_delta"]))),
            }
        )

    site = pd.DataFrame(site_rows)
    if len(site):
        site["is_responsive20"] = False
        site["is_predicted20"] = False
        for _, idx in site.groupby("comparison", sort=False).groups.items():
            idx = list(idx)
            real = site.loc[idx, "observed_delta"].to_numpy(dtype=float)
            pred_values = site.loc[idx, "predicted_delta"].to_numpy(dtype=float)
            site.loc[idx, "is_responsive20"] = top_fraction_mask(real, 0.2)
            site.loc[idx, "is_predicted20"] = top_fraction_mask(pred_values, 0.2)

    metrics, drug_summary = evaluate(site)
    descriptors = pd.DataFrame(descriptor_rows)
    observed.to_csv(tables / "pxd063604_v10b_observed_long.tsv", sep="\t", index=False)
    descriptors.to_csv(tables / "pxd063604_v10b_drug_descriptors.tsv", sep="\t", index=False)
    site.to_csv(tables / "pxd063604_v10b_sitelevel_delta_long.tsv", sep="\t", index=False)
    metrics.to_csv(tables / "pxd063604_v10b_comparison_metrics.tsv", sep="\t", index=False)
    drug_summary.to_csv(tables / "pxd063604_v10b_drug_summary.tsv", sep="\t", index=False)

    all_metrics = metrics[metrics["subset"].eq("all")] if len(metrics) else metrics
    reg_metrics = metrics[metrics["subset"].eq("regulated")] if len(metrics) else metrics
    summary = {
        "analysis": "SCP682-PPKO V10B frozen model PXD063604 site-level export",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(model_path),
        "observed_path": str(observed_path),
        "device": str(device),
        "n_model_sites": int(len(sites)),
        "n_observed_rows": int(len(observed)),
        "n_overlap_rows": int(len(site)),
        "n_comparisons": int(site["comparison"].nunique()) if len(site) else 0,
        "mean_site_cosine_all": float(all_metrics["site_cosine"].mean()) if len(all_metrics) else float("nan"),
        "mean_spearman_all": float(all_metrics["spearman"].mean()) if len(all_metrics) else float("nan"),
        "mean_direction_accuracy_all": float(all_metrics["direction_accuracy"].mean()) if len(all_metrics) else float("nan"),
        "mean_site_cosine_regulated": float(reg_metrics["site_cosine"].mean()) if len(reg_metrics) else float("nan"),
        "mean_spearman_regulated": float(reg_metrics["spearman"].mean()) if len(reg_metrics) else float("nan"),
        "mean_direction_accuracy_regulated": float(reg_metrics["direction_accuracy"].mean()) if len(reg_metrics) else float("nan"),
    }
    (reports / "pxd063604_v10b_sitelevel_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
