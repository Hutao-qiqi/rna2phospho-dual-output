from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch


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


def clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def drug_class(perturbation: str, target_genes: str) -> str:
    p = str(perturbation).strip().lower()
    t = str(target_genes).upper()
    if "HDAC" in t or p == "vorinostat":
        return "HDAC"
    if "PSMB" in t or p in {"bortezomib", "carfilzomib"}:
        return "proteasome"
    if "MAP2K" in t or p in {"trametinib", "selumetinib"}:
        return "MEK"
    if "EGFR" in t or "ERBB2" in t:
        return "EGFR_HER2"
    if any(g in t for g in ["ABL1", "SRC", "LYN", "HCK"]):
        return "ABL_SRC"
    if "MTOR" in t:
        return "mTOR"
    return "other"


def top_fraction_mask(values: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    ok = np.isfinite(values)
    out = np.zeros(len(values), dtype=bool)
    n = int(ok.sum())
    if n == 0:
        return out
    k = max(1, int(math.ceil(n * fraction)))
    idx = np.where(ok)[0]
    out[idx[np.argsort(-np.abs(values[idx]))[:k]]] = True
    return out


def mean_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    if not ok.any():
        return np.nan
    return float(values[ok].mean())


def parse_args() -> argparse.Namespace:
    root_default = r"D:\data\lsy\vm_lsy_parent\lsy" if os.name == "nt" else "/mnt/d/data/lsy/vm_lsy_parent/lsy"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("SCP682_DATA_ROOT", root_default))
    parser.add_argument("--package-root", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    package_root = Path(args.package_root) if args.package_root else data_root / "SCP682_PPKO_V10B_transferable"
    scripts_dir = package_root / "scripts"
    sys.path.insert(0, str(scripts_dir))

    from pretrain_v10b_strong300 import AttentionPriorManifoldV10, build_global_signed_inputs

    model_path = Path(args.model) if args.model else package_root / "models" / "scp682_ppko_v10b_strong300_best.pt"
    graph_dir = data_root / "01_data" / "pathway_prior" / "intermediate" / "global_phosphoprotein_heterograph_v10_measured_string700_top50"
    train_dir = data_root / "01_data" / "single_cell" / "intermediate" / "phospho_perturb" / "decryptm_comparison_delta_v8"
    p100_dir = data_root / "01_data" / "single_cell" / "intermediate" / "phospho_perturb" / "lincs_p100_comparison_delta_v7"
    map_dir = data_root / "02_results" / "single_cell" / "20260519_scp682_ppko_1_decryptm_network_v8_full_p100_shared_site_validation" / "tables"
    out_dir = Path(args.output_dir) if args.output_dir else data_root / "02_results" / "single_cell" / "20260531_scp682_ppko_v10b_p100_sitelevel_all125"
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

    p_delta = np.load(p100_dir / "arrays" / "delta_matrix.npy").astype(np.float32)
    p_base = np.load(p100_dir / "arrays" / "baseline_matrix.npy").astype(np.float32)
    p_valid = np.load(p100_dir / "arrays" / "valid_mask.npy").astype(bool)
    p_comp = pd.read_csv(p100_dir / "tables" / "comparison_table.tsv", sep="\t")
    p_sites = pd.read_csv(p100_dir / "tables" / "site_table.tsv", sep="\t")
    mapping = pd.read_csv(map_dir / "p100_to_decryptm_shared_site_map.tsv", sep="\t")
    pidx = mapping["p100_target_index"].to_numpy(dtype=int)
    didx = mapping["decryptm_target_index"].to_numpy(dtype=int)

    train_base = np.load(train_dir / "arrays" / "baseline_matrix.npy").astype(np.float32)
    train_valid = np.load(train_dir / "arrays" / "valid_mask.npy").astype(bool)
    generic = np.zeros(len(sites), dtype=np.float32)
    for site_i in range(len(sites)):
        vals = train_base[train_valid[:, site_i], site_i]
        generic[site_i] = float(np.nanmedian(vals)) if len(vals) else 0.0

    targets = sorted(set(TARGETS.values()) - {""})
    context_df = pd.DataFrame([{"target_genes": t, "action_type": "inhibition"} for t in targets])
    _, protein_context, graph_prior, _ = build_global_signed_inputs(context_df, graph_dir)
    context_cache = {t: (protein_context[i], graph_prior[i]) for i, t in enumerate(targets)}

    mapping_rows: list[dict[str, object]] = []
    unique_rows: list[dict[str, object]] = []
    grouped_mapping = list(mapping.groupby("p100_target_index", sort=False))

    for comp_i, comp in p_comp.iterrows():
        perturbation = clean(comp.get("perturbation", ""))
        target_genes = TARGETS.get(perturbation, clean(comp.get("target_genes", "")))
        if not target_genes:
            continue
        use = p_valid[comp_i, pidx]
        if int(use.sum()) < 5:
            continue

        base = generic.copy()
        valid = np.zeros(len(sites), dtype=bool)
        base[didx[use]] = p_base[comp_i, pidx[use]]
        valid[didx[use]] = True

        pc, gp = context_cache[target_genes]
        with torch.no_grad():
            pred, graph, latent, residual, _, attention = model(
                torch.as_tensor(base, dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(valid, dtype=torch.bool, device=device).unsqueeze(0),
                torch.as_tensor(pc, dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(gp, dtype=torch.float32, device=device).unsqueeze(0),
            )
        pred_all = pred.detach().cpu().numpy()[0]
        graph_all = graph.detach().cpu().numpy()[0]
        latent_all = latent.detach().cpu().numpy()[0]
        residual_all = residual.detach().cpu().numpy()[0]
        attention_all = attention.detach().cpu().numpy()[0]

        comparison_id = clean(comp.get("comparison_id", f"p100_delta_{comp_i:05d}"))
        shared_unique: list[dict[str, object]] = []
        for p100_target_index, sub in grouped_mapping:
            p100_target_index = int(p100_target_index)
            if not p_valid[comp_i, p100_target_index]:
                continue
            site_meta = p_sites.iloc[p100_target_index] if p100_target_index < len(p_sites) else pd.Series(dtype=object)
            decryptm_indices = sub["decryptm_target_index"].astype(int).to_numpy()
            valid_didx = decryptm_indices[np.isfinite(pred_all[decryptm_indices])]
            if len(valid_didx) == 0:
                continue
            first = sub.iloc[0]
            common = {
                "comparison_id": comparison_id,
                "comparison_index": int(comp_i),
                "dataset_id": clean(comp.get("dataset_id", "")),
                "cell_line": clean(comp.get("cell_type", "")),
                "perturbation": perturbation,
                "perturbation_id": clean(comp.get("perturbation_id", "")),
                "target_genes": target_genes,
                "context_target_genes": target_genes,
                "action_type": clean(comp.get("action_type", "inhibition")),
                "dose_molar": comp.get("dose_molar", np.nan),
                "time_hours": comp.get("time_hours", np.nan),
                "drug_class": drug_class(perturbation, target_genes),
                "p100_target_index": p100_target_index,
                "phosphosite": clean(first.get("p100_target_id", site_meta.get("target_id", ""))),
                "gene": clean(first.get("p100_molecule", site_meta.get("molecule", ""))),
                "site": clean(first.get("p100_site", site_meta.get("site", ""))),
                "uniprot": clean(first.get("p100_uniprot", site_meta.get("uniprot_id", ""))),
                "modified_peptide": clean(first.get("p100_modified_peptide", site_meta.get("modified_peptide", ""))),
                "observed_delta": float(p_delta[comp_i, p100_target_index]),
                "baseline_value": float(p_base[comp_i, p100_target_index]),
            }
            unique_rec = {
                **common,
                "predicted_delta": mean_or_nan(pred_all[valid_didx]),
                "graph_delta": mean_or_nan(graph_all[valid_didx]),
                "latent_delta": mean_or_nan(latent_all[valid_didx]),
                "residual_delta": mean_or_nan(residual_all[valid_didx]),
                "attention": mean_or_nan(attention_all[valid_didx]),
                "n_decryptm_mapped_sites": int(len(set(valid_didx.tolist()))),
                "decryptm_target_indices": ";".join(map(str, sorted(set(valid_didx.tolist())))),
                "decryptm_target_ids": ";".join(sorted(set(sub["decryptm_target_id"].astype(str)))),
                "mapping_methods": ";".join(sorted(set(sub["mapping_method"].astype(str)))),
            }
            shared_unique.append(unique_rec)
            for _, map_row in sub.iterrows():
                decryptm_index = int(map_row["decryptm_target_index"])
                if not np.isfinite(pred_all[decryptm_index]):
                    continue
                mapping_rows.append(
                    {
                        **common,
                        "decryptm_target_index": decryptm_index,
                        "decryptm_target_id": clean(map_row.get("decryptm_target_id", "")),
                        "decryptm_uniprot": clean(map_row.get("decryptm_uniprot", "")),
                        "decryptm_modified_peptide": clean(map_row.get("decryptm_modified_peptide", "")),
                        "mapping_method": clean(map_row.get("mapping_method", "")),
                        "predicted_delta": float(pred_all[decryptm_index]),
                        "graph_delta": float(graph_all[decryptm_index]),
                        "latent_delta": float(latent_all[decryptm_index]),
                        "residual_delta": float(residual_all[decryptm_index]),
                        "attention": float(attention_all[decryptm_index]),
                    }
                )

        if shared_unique:
            real = np.asarray([r["observed_delta"] for r in shared_unique], dtype=float)
            pred_vec = np.asarray([r["predicted_delta"] for r in shared_unique], dtype=float)
            responsive = top_fraction_mask(real, 0.2)
            predicted = top_fraction_mask(pred_vec, 0.2)
            for flag_i, rec in enumerate(shared_unique):
                rec["is_responsive20"] = bool(responsive[flag_i])
                rec["is_predicted20"] = bool(predicted[flag_i])
                unique_rows.append(rec)

    mapping_df = pd.DataFrame(mapping_rows)
    unique_df = pd.DataFrame(unique_rows)
    mapping_df.to_csv(tables / "p100_v10b_all125_mapping_row_sitelevel_delta_long.tsv", sep="\t", index=False)
    unique_df.to_csv(tables / "p100_v10b_all125_unique_sitelevel_delta_long.tsv", sep="\t", index=False)

    report = {
        "model": "SCP682-PPKO V10B strong300",
        "model_path": str(model_path),
        "output_dir": str(out_dir),
        "n_comparisons": int(unique_df["comparison_id"].nunique()) if len(unique_df) else 0,
        "n_unique_p100_sites": int(unique_df["p100_target_index"].nunique()) if len(unique_df) else 0,
        "n_unique_rows": int(len(unique_df)),
        "n_mapping_rows": int(len(mapping_df)),
        "tables": [
            "p100_v10b_all125_unique_sitelevel_delta_long.tsv",
            "p100_v10b_all125_mapping_row_sitelevel_delta_long.tsv",
        ],
    }
    (reports / "p100_v10b_all125_sitelevel_export_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
