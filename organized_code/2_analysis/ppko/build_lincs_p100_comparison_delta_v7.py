import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
DEFAULT_P100 = ROOT / r"01_data\single_cell\intermediate\phospho_perturb\lincs_p100_lvl4_v1"
DEFAULT_JOINT = ROOT / r"01_data\single_cell\intermediate\phospho_perturb\decryptm_p100_joint_v1"
DEFAULT_OUT = ROOT / r"01_data\single_cell\intermediate\phospho_perturb\lincs_p100_comparison_delta_v7"


PATHWAY_GENES = {
    "RTK_SRC_axis": {"EGFR", "ERBB2", "ERBB3", "ERBB4", "FGFR1", "FGFR2", "FGFR3", "FGFR4", "KIT", "PDGFRA", "PDGFRB", "SRC", "YES1", "LYN", "LCK", "HCK", "ABL1"},
    "RAS_MAPK_axis": {"KRAS", "NRAS", "HRAS", "BRAF", "RAF1", "ARAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3"},
    "PI3K_AKT_mTOR_axis": {"PIK3CA", "PIK3CB", "PIK3CD", "AKT1", "AKT2", "AKT3", "MTOR", "RPTOR", "RICTOR", "PDPK1"},
    "epigenetic_chromatin": {"HDAC1", "HDAC2", "HDAC3", "HDAC6", "CREBBP", "EP300", "EZH2", "BRD4"},
    "cell_cycle_DNA_damage": {"CDK1", "CDK2", "CDK4", "CDK6", "AURKA", "AURKB", "CHEK1", "CHEK2", "ATM", "ATR"},
    "global_phospho": set(),
}


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_float(value):
    try:
        out = float(value)
        return out if np.isfinite(out) else np.nan
    except Exception:
        return np.nan


def dose_to_molar(dose, unit):
    value = parse_float(dose)
    if not np.isfinite(value):
        return np.nan
    unit = str(unit).strip().lower()
    if unit in {"m", "mol", "molar"}:
        return value
    if unit in {"um", "µm", "μm", "micromolar"}:
        return value * 1e-6
    if unit in {"nm", "nanomolar"}:
        return value * 1e-9
    if unit in {"mm", "millimolar"}:
        return value * 1e-3
    return np.nan


def time_to_hours(value, unit):
    number = parse_float(value)
    if not np.isfinite(number):
        return np.nan
    unit = str(unit).strip().lower()
    if unit in {"h", "hr", "hour", "hours"}:
        return number
    if unit in {"m", "min", "minute", "minutes"}:
        return number / 60.0
    if unit in {"s", "sec", "second", "seconds"}:
        return number / 3600.0
    return number


def split_genes(value):
    out = []
    for part in re.split(r"[/,;| ]+", str(value)):
        gene = re.sub(r"[^A-Za-z0-9-]", "", part).upper()
        if gene and gene != "NAN":
            out.append(gene)
    return sorted(set(out))


def infer_action_type(row):
    name = str(row.get("perturbation", "")).lower()
    ptype = str(row.get("perturbation_type", "")).lower()
    if any(x in name for x in ("agonist", "stim", "egf", "bcr")):
        return "activation"
    if "degrader" in name or "protac" in name:
        return "degrader"
    if ptype == "trt_cp":
        return "inhibition"
    return "unknown"


def pathway_prior_for_targets(genes):
    prior = {p: 0.0 for p in PATHWAY_GENES}
    gset = set(genes)
    for pathway, members in PATHWAY_GENES.items():
        if pathway == "global_phospho":
            continue
        if gset & members:
            prior[pathway] = 1.0
    prior["global_phospho"] = 1.0
    return prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p100-dir", default=str(DEFAULT_P100))
    ap.add_argument("--joint-dir", default=str(DEFAULT_JOINT))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--min-control-n", type=int, default=2)
    ap.add_argument("--min-treatment-n", type=int, default=2)
    ap.add_argument("--min-valid-sites", type=int, default=20)
    ap.add_argument("--require-target-genes", action="store_true", default=True)
    args = ap.parse_args()

    p100_dir = Path(args.p100_dir)
    out = ensure_dir(args.output_dir)
    for sub in ("tables", "arrays", "reports"):
        ensure_dir(out / sub)

    raw = np.load(p100_dir / "phospho_raw.npy").astype(np.float32)
    z = np.load(p100_dir / "phospho_values.npy").astype(np.float32)
    mask = np.load(p100_dir / "target_mask.npy").astype(bool)
    meta = pd.read_csv(p100_dir / "cell_metadata.tsv", sep="\t")
    targets = pd.read_csv(p100_dir / "phospho_target_table.tsv", sep="\t")
    cond = pd.read_csv(p100_dir / "condition_table.tsv", sep="\t")

    joint_cond = pd.read_csv(Path(args.joint_dir) / "condition_table.tsv", sep="\t")
    ann = joint_cond[joint_cond["dataset_id"].eq("lincs_p100_lvl4")][["perturbation_id", "perturbation", "target_genes"]].dropna()
    ann = ann.drop_duplicates("perturbation_id")
    target_by_brd = dict(zip(ann["perturbation_id"].astype(str), ann["target_genes"].astype(str)))

    condition_to_indices = {c: sub.index.to_numpy(dtype=np.int64) for c, sub in meta.groupby("condition")}
    comparisons = []
    delta_rows = []
    baseline_rows = []
    valid_rows = []
    pathway_rows = []
    drug_rows = []

    pathways = list(PATHWAY_GENES)
    pathway_to_idx = {p: i for i, p in enumerate(pathways)}
    target_gene_sets = []
    for _, row in targets.iterrows():
        target_gene_sets.append(set(split_genes(row.get("molecule", ""))))

    for _, row in cond.iterrows():
        condition = str(row["condition"])
        control = str(row["control_condition"])
        if condition == control:
            continue
        trt_idx = condition_to_indices.get(condition, np.array([], dtype=np.int64))
        ctrl_idx = condition_to_indices.get(control, np.array([], dtype=np.int64))
        if len(trt_idx) < args.min_treatment_n or len(ctrl_idx) < args.min_control_n:
            continue
        brd = str(row.get("perturbation_id", ""))
        genes = split_genes(target_by_brd.get(brd, ""))
        if args.require_target_genes and not genes:
            continue
        trt_mean = np.nanmean(np.where(mask[trt_idx], z[trt_idx], np.nan), axis=0)
        ctrl_mean = np.nanmean(np.where(mask[ctrl_idx], z[ctrl_idx], np.nan), axis=0)
        trt_raw = np.nanmean(np.where(mask[trt_idx], raw[trt_idx], np.nan), axis=0)
        ctrl_raw = np.nanmean(np.where(mask[ctrl_idx], raw[ctrl_idx], np.nan), axis=0)
        valid = np.isfinite(trt_mean) & np.isfinite(ctrl_mean)
        if int(valid.sum()) < args.min_valid_sites:
            continue
        delta = (trt_mean - ctrl_mean).astype(np.float32)
        baseline = ctrl_mean.astype(np.float32)
        delta[~valid] = 0.0
        baseline[~np.isfinite(baseline)] = 0.0
        dose_molar = dose_to_molar(row.get("dose"), row.get("dose_unit"))
        time_hours = time_to_hours(row.get("time"), row.get("time_unit"))
        action_type = infer_action_type(row)
        prior = pathway_prior_for_targets(genes)
        comp_id = f"p100_delta_{len(comparisons):05d}"
        anchor = np.zeros(len(targets), dtype=np.float32)
        for i, gset in enumerate(target_gene_sets):
            if gset & set(genes):
                anchor[i] = -1.0 if action_type in {"inhibition", "degrader"} else 1.0
        comparisons.append(
            {
                "comparison_id": comp_id,
                "dataset_id": "lincs_p100_lvl4",
                "cell_type": row.get("cell_type", ""),
                "plate": str(condition).split("_")[2] if len(str(condition).split("_")) > 2 else "",
                "control_condition": control,
                "treatment_condition": condition,
                "perturbation": row.get("perturbation", ""),
                "perturbation_id": brd,
                "target_genes": ";".join(genes),
                "target_weights": ";".join(f"{g}:1" for g in genes),
                "action_type": action_type,
                "dose": row.get("dose", ""),
                "dose_unit": row.get("dose_unit", ""),
                "dose_molar": dose_molar,
                "time": row.get("time", ""),
                "time_unit": row.get("time_unit", ""),
                "time_hours": time_hours,
                "control_n": int(len(ctrl_idx)),
                "treatment_n": int(len(trt_idx)),
                "valid_site_count": int(valid.sum()),
                "anchor_site_count": int((anchor != 0).sum()),
                "delta_vector_id": len(delta_rows),
            }
        )
        delta_rows.append(delta)
        baseline_rows.append(baseline)
        valid_rows.append(valid)
        pathway_rows.append([prior[p] for p in pathways])
        drug_rows.append(anchor)

    if not delta_rows:
        raise RuntimeError("no valid P100 comparisons after filtering")

    delta_mat = np.vstack(delta_rows).astype(np.float32)
    baseline_mat = np.vstack(baseline_rows).astype(np.float32)
    valid_mat = np.vstack(valid_rows).astype(bool)
    pathway_mat = np.asarray(pathway_rows, dtype=np.float32)
    anchor_mat = np.asarray(drug_rows, dtype=np.float32)

    np.save(out / "arrays" / "delta_matrix.npy", delta_mat)
    np.save(out / "arrays" / "baseline_matrix.npy", baseline_mat)
    np.save(out / "arrays" / "valid_mask.npy", valid_mat)
    np.save(out / "arrays" / "pathway_prior_matrix.npy", pathway_mat)
    np.save(out / "arrays" / "target_anchor_matrix.npy", anchor_mat)
    pd.DataFrame(comparisons).to_csv(out / "tables" / "comparison_table.tsv", sep="\t", index=False)
    targets.to_csv(out / "tables" / "site_table.tsv", sep="\t", index=False)
    pd.DataFrame({"pathway_index": range(len(pathways)), "pathway": pathways}).to_csv(out / "tables" / "pathway_table.tsv", sep="\t", index=False)
    ann.to_csv(out / "tables" / "drug_target_table.tsv", sep="\t", index=False)

    summary = {
        "source": "LINCS P100 only, comparison-level treatment-control delta",
        "excluded": "decryptM",
        "n_comparisons": int(delta_mat.shape[0]),
        "n_sites": int(delta_mat.shape[1]),
        "finite_fraction": float(valid_mat.mean()),
        "n_drugs": int(pd.DataFrame(comparisons)["perturbation_id"].nunique()),
        "n_cell_types": int(pd.DataFrame(comparisons)["cell_type"].nunique()),
        "n_with_anchor_sites": int((anchor_mat != 0).any(axis=1).sum()),
        "mean_anchor_sites": float((anchor_mat != 0).sum(axis=1).mean()),
        "dose_molar_present": int(np.isfinite(pd.DataFrame(comparisons)["dose_molar"].to_numpy(dtype=float)).sum()),
    }
    with (out / "reports" / "build_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
