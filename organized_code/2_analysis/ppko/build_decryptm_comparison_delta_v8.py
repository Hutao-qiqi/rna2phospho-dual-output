import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
DEFAULT_INPUT = ROOT / r"01_data\single_cell\intermediate\phospho_perturb\decryptm_pxd037285_curve_v1"
DEFAULT_OUT = ROOT / r"01_data\single_cell\intermediate\phospho_perturb\decryptm_comparison_delta_v8"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_genes(value):
    out = []
    for part in re.split(r"[/,;| ]+", str(value)):
        gene = re.sub(r"[^A-Za-z0-9-]", "", part).upper()
        if gene and gene != "NAN":
            out.append(gene)
    return sorted(set(out))


def infer_action_type(row):
    ptype = str(row.get("perturbation_type", "")).lower()
    name = str(row.get("perturbation", "")).lower()
    if "stim" in name or "egf" == name:
        return "activation"
    if ptype == "td":
        return "activation"
    if ptype == "dd":
        return "inhibition"
    if ptype == "control":
        return "none"
    return "inhibition"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--min-control-n", type=int, default=2)
    ap.add_argument("--min-treatment-n", type=int, default=2)
    ap.add_argument("--min-valid-sites", type=int, default=200)
    args = ap.parse_args()

    inp = Path(args.input_dir)
    out = ensure_dir(args.output_dir)
    for sub in ("arrays", "tables", "reports"):
        ensure_dir(out / sub)

    x = np.load(inp / "phospho_values.npy").astype(np.float32)
    raw = np.load(inp / "phospho_raw.npy").astype(np.float32)
    mask = np.load(inp / "target_mask.npy").astype(bool)
    meta = pd.read_csv(inp / "cell_metadata.tsv", sep="\t")
    cond = pd.read_csv(inp / "condition_table.tsv", sep="\t")
    targets = pd.read_csv(inp / "phospho_target_table.tsv", sep="\t")
    manifest = pd.read_csv(inp / "pathway_target_manifest.tsv", sep="\t")
    priors = pd.read_csv(inp / "condition_pathway_prior.tsv", sep="\t")

    by_condition = {c: sub.index.to_numpy(dtype=np.int64) for c, sub in meta.groupby("condition")}
    pathway_names = list(dict.fromkeys(manifest["pathway"].astype(str).tolist()))
    if "global_phospho" not in pathway_names:
        pathway_names.append("global_phospho")
    p_to_i = {p: i for i, p in enumerate(pathway_names)}
    prior_by_cond = {}
    for condition, sub in priors.groupby("condition"):
        v = np.zeros(len(pathway_names), dtype=np.float32)
        for _, row in sub.iterrows():
            p = str(row["pathway"])
            if p in p_to_i:
                v[p_to_i[p]] = max(v[p_to_i[p]], float(row.get("prior", 1.0)))
        if "global_phospho" in p_to_i:
            v[p_to_i["global_phospho"]] = 1.0
        prior_by_cond[str(condition)] = v

    comparisons = []
    delta_rows = []
    baseline_rows = []
    valid_rows = []
    pathway_rows = []
    for _, row in cond.iterrows():
        condition = str(row["condition"])
        control = str(row["control_condition"])
        if condition == control or str(row.get("perturbation_type", "")).lower() == "control":
            continue
        trt_idx = by_condition.get(condition, np.array([], dtype=np.int64))
        ctrl_idx = by_condition.get(control, np.array([], dtype=np.int64))
        if len(trt_idx) < args.min_treatment_n or len(ctrl_idx) < args.min_control_n:
            continue
        trt = np.nanmean(np.where(mask[trt_idx], x[trt_idx], np.nan), axis=0)
        ctrl = np.nanmean(np.where(mask[ctrl_idx], x[ctrl_idx], np.nan), axis=0)
        valid = np.isfinite(trt) & np.isfinite(ctrl)
        if int(valid.sum()) < args.min_valid_sites:
            continue
        delta = (trt - ctrl).astype(np.float32)
        baseline = ctrl.astype(np.float32)
        delta[~valid] = 0.0
        baseline[~np.isfinite(baseline)] = 0.0
        genes = split_genes(row.get("target_genes", ""))
        prior = prior_by_cond.get(condition, np.zeros(len(pathway_names), dtype=np.float32))
        if "global_phospho" in p_to_i:
            prior[p_to_i["global_phospho"]] = 1.0
        comp_id = f"decryptm_delta_{len(comparisons):05d}"
        comparisons.append({
            "comparison_id": comp_id,
            "dataset_id": "decryptm_pxd037285_curve",
            "cell_type": row.get("cell_type", ""),
            "control_condition": control,
            "treatment_condition": condition,
            "perturbation": row.get("perturbation", ""),
            "perturbation_id": row.get("perturbation_id", ""),
            "perturbation_type": row.get("perturbation_type", ""),
            "target_genes": ";".join(genes),
            "action_type": infer_action_type(row),
            "dose": row.get("dose", ""),
            "dose_unit": row.get("dose_unit", ""),
            "dose_molar": row.get("dose_molar", np.nan),
            "time": row.get("time", ""),
            "time_unit": row.get("time_unit", ""),
            "time_hours": row.get("time_hours", np.nan),
            "control_n": int(len(ctrl_idx)),
            "treatment_n": int(len(trt_idx)),
            "valid_site_count": int(valid.sum()),
            "delta_vector_id": len(delta_rows),
        })
        delta_rows.append(delta)
        baseline_rows.append(baseline)
        valid_rows.append(valid)
        pathway_rows.append(prior)

    if not delta_rows:
        raise RuntimeError("no decryptM comparisons")
    delta_mat = np.vstack(delta_rows).astype(np.float32)
    baseline_mat = np.vstack(baseline_rows).astype(np.float32)
    valid_mat = np.vstack(valid_rows).astype(bool)
    pathway_mat = np.vstack(pathway_rows).astype(np.float32)

    np.save(out / "arrays" / "delta_matrix.npy", delta_mat)
    np.save(out / "arrays" / "baseline_matrix.npy", baseline_mat)
    np.save(out / "arrays" / "valid_mask.npy", valid_mat)
    np.save(out / "arrays" / "pathway_prior_matrix.npy", pathway_mat)
    pd.DataFrame(comparisons).to_csv(out / "tables" / "comparison_table.tsv", sep="\t", index=False)
    targets.to_csv(out / "tables" / "site_table.tsv", sep="\t", index=False)
    pd.DataFrame({"pathway_index": range(len(pathway_names)), "pathway": pathway_names}).to_csv(out / "tables" / "pathway_table.tsv", sep="\t", index=False)
    summary = {
        "source": "decryptM PXD037285 only, comparison-level treatment-control delta",
        "excluded": "P100",
        "n_comparisons": int(delta_mat.shape[0]),
        "n_sites": int(delta_mat.shape[1]),
        "finite_fraction": float(valid_mat.mean()),
        "n_drugs": int(pd.DataFrame(comparisons)["perturbation"].nunique()),
        "n_cell_types": int(pd.DataFrame(comparisons)["cell_type"].nunique()),
        "n_pathways": int(len(pathway_names)),
        "dose_molar_present": int(pd.to_numeric(pd.DataFrame(comparisons)["dose_molar"], errors="coerce").notna().sum()),
        "time_hours_present": int(pd.to_numeric(pd.DataFrame(comparisons)["time_hours"], errors="coerce").notna().sum()),
    }
    with (out / "reports" / "build_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
