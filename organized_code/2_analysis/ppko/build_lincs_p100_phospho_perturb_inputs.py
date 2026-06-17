import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
DEFAULT_RAW = DEFAULT_ROOT / r"01_data\single_cell\raw\lincs_p100_lvl4_v1"
DEFAULT_OUT = DEFAULT_ROOT / r"01_data\single_cell\intermediate\phospho_perturb\lincs_p100_lvl4_v1"


PATHWAY_GENES = {
    "MAPK_axis": {"RAF1", "BRAF", "ARAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "RPS6KA1", "RPS6KA2", "RPS6KA3", "ELK1", "DUSP6", "FOS", "FOSL2", "JUN", "JUND"},
    "stress_MAPK_axis": {"MAPK8", "MAPK9", "MAPK14", "MAPKAPK2", "HSPB1", "ATF2", "JUN", "JUND"},
    "PI3K_AKT_mTOR_axis": {"PIK3CA", "PIK3CB", "PIK3CD", "AKT1", "AKT2", "AKT3", "PDPK1", "MTOR", "RPTOR", "RICTOR", "RPS6", "RPS6KB1", "EIF4EBP1", "GSK3A", "GSK3B", "FOXO1", "FOXO3"},
    "PLCG_PKC_axis": {"PLCG1", "PLCG2", "PRKCA", "PRKCB", "PRKCD", "PRKCE", "MARCKS", "PAK1", "PAK2"},
    "cell_cycle_DNA_damage": {"CDK1", "CDK2", "CDK4", "CDK6", "RB1", "TP53", "CHEK1", "CHEK2", "ATM", "ATR", "ATRIP", "BRD4", "ZC3HC1", "TPX2"},
    "cytoskeleton_adhesion": {"ABI1", "AHNAK", "LIMA1", "PLEC1", "MAP4", "KIF4A", "TMPO", "SH3KBP1"},
}


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_token(value):
    value = str(value)
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "NA"


def split_genes(value):
    genes = []
    for part in re.split(r"[/,;| ]+", str(value)):
        part = clean_token(part).upper()
        if part and part != "NA":
            genes.append(part)
    return genes


def make_target_id(row):
    genes = clean_token(row.get("pr_gene_symbol", "NA")).upper()
    site = clean_token(row.get("pr_p100_phosphosite", "NA")).upper()
    if genes != "NA" and site != "NA":
        return f"{genes}_{site}"
    return clean_token(row.get("id", row.get("pr_p100_modified_peptide_code", "target"))).upper()


def parse_gct(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        version = fh.readline().strip()
        if version != "#1.3":
            raise ValueError(f"{path} is not GCT 1.3")
        n_rows, n_cols, n_row_meta, n_col_meta = [int(x) for x in fh.readline().strip().split("\t")]
        header = fh.readline().rstrip("\n").split("\t")
        sample_start = 1 + n_row_meta
        sample_ids = header[sample_start:]
        col_meta = {}
        for _ in range(n_col_meta):
            parts = fh.readline().rstrip("\n").split("\t")
            col_meta[parts[0]] = parts[sample_start:]
        row_meta = []
        values = []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            row_meta.append({header[i]: parts[i] if i < len(parts) else "NA" for i in range(sample_start)})
            vals = []
            for x in parts[sample_start:]:
                if x in ("", "NA", "NaN", "nan"):
                    vals.append(np.nan)
                else:
                    try:
                        vals.append(float(x))
                    except ValueError:
                        vals.append(np.nan)
            values.append(vals)
    if len(row_meta) != n_rows or len(sample_ids) != n_cols:
        raise ValueError(f"dimension mismatch in {path}")
    return pd.DataFrame(row_meta), np.asarray(values, dtype=np.float32), pd.DataFrame(col_meta, index=sample_ids)


def pathway_membership(genes):
    hits = []
    gene_set = set(genes)
    for pathway, members in PATHWAY_GENES.items():
        if gene_set & members:
            hits.append(pathway)
    hits.append("global_phospho")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--min-target-finite", type=int, default=20)
    ap.add_argument("--min-condition-replicates", type=int, default=2)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out = ensure_dir(args.output_dir)
    files = sorted(raw_dir.glob("*_LVL4.gct"))
    if not files:
        raise FileNotFoundError(f"no LVL4 GCT files in {raw_dir}")

    target_info = {}
    parsed_files = []
    for gct in files:
        row_meta, mat, col_meta = parse_gct(gct)
        row_meta = row_meta.copy()
        row_meta["target_id"] = row_meta.apply(make_target_id, axis=1)
        for _, row in row_meta.iterrows():
            tid = row["target_id"]
            if tid not in target_info:
                target_info[tid] = row.to_dict()
        parsed_files.append((gct, row_meta, mat, col_meta))

    samples = []
    sample_vectors = []
    sample_masks = []
    condition_counts = defaultdict(int)

    for gct, row_meta, mat, col_meta in parsed_files:
        local_ids = row_meta["target_id"].tolist()
        local_to_rows = defaultdict(list)
        for i, tid in enumerate(local_ids):
            local_to_rows[tid].append(i)

        plate = str(col_meta.get("det_plate", pd.Series(["NA"])).iloc[0])
        cell_line = str(col_meta.get("cell_id", pd.Series(["NA"])).iloc[0])
        if cell_line == "NA":
            cell_line = str(col_meta.get("cell_line", pd.Series(["NA"])).iloc[0])
        control_condition = clean_token(f"lincs_p100__{plate}__{cell_line}__DMSO")

        local_master = []
        for tid in target_info:
            rows = local_to_rows.get(tid, [])
            if rows:
                block = mat[rows, :]
                count = np.isfinite(block).sum(axis=0)
                summed = np.nansum(block, axis=0)
                local_master.append(np.where(count > 0, summed / np.maximum(count, 1), np.nan))
            else:
                local_master.append(np.full(mat.shape[1], np.nan, dtype=np.float32))
        aligned = np.vstack(local_master).astype(np.float32)

        for sample_i, sample_id in enumerate(col_meta.index.tolist()):
            pert_type = str(col_meta.loc[sample_id].get("pert_type", "NA"))
            pert_iname = str(col_meta.loc[sample_id].get("pert_iname", "NA"))
            pert_id = str(col_meta.loc[sample_id].get("pert_id", "NA"))
            dose = str(col_meta.loc[sample_id].get("pert_dose", "NA"))
            dose_unit = str(col_meta.loc[sample_id].get("pert_dose_unit", "NA"))
            time_value = str(col_meta.loc[sample_id].get("pert_time", "NA"))
            time_unit = str(col_meta.loc[sample_id].get("pert_time_unit", "NA"))
            if pert_type == "ctrl_vehicle" or pert_iname.upper() == "DMSO":
                condition = control_condition
                control = control_condition
            else:
                condition = clean_token(f"lincs_p100__{plate}__{cell_line}__{pert_id}__{dose}{dose_unit}__{time_value}{time_unit}")
                control = control_condition
            vec = aligned[:, sample_i]
            mask = np.isfinite(vec)
            if int(mask.sum()) < args.min_target_finite:
                continue
            row = {
                "cell_id": f"lincs_p100:{gct.stem}:{sample_id}",
                "dataset_id": "lincs_p100_lvl4",
                "plate": plate,
                "sample_id": sample_id,
                "condition": condition,
                "condition_label": condition,
                "perturbation": pert_iname,
                "perturbation_id": pert_id,
                "perturbation_type": pert_type,
                "control_condition": control,
                "time": time_value,
                "time_unit": time_unit,
                "dose": dose,
                "dose_unit": dose_unit,
                "cell_type": cell_line,
                "is_simulated": False,
                "source_file": gct.name,
            }
            samples.append(row)
            sample_vectors.append(vec)
            sample_masks.append(mask)
            condition_counts[condition] += 1

    keep_conditions = {k for k, v in condition_counts.items() if v >= args.min_condition_replicates}
    keep = [i for i, row in enumerate(samples) if row["condition"] in keep_conditions and row["control_condition"] in keep_conditions]
    samples = [samples[i] for i in keep]
    raw = np.vstack([sample_vectors[i] for i in keep]).astype(np.float32)
    mask = np.vstack([sample_masks[i] for i in keep]).astype(bool)

    mean = np.nanmean(raw, axis=0, keepdims=True)
    sd = np.nanstd(raw, axis=0, keepdims=True)
    mean[~np.isfinite(mean)] = 0.0
    sd[(~np.isfinite(sd)) | (sd <= 1e-8)] = 1.0
    z = ((raw - mean) / sd).astype(np.float32)
    z[~np.isfinite(z)] = 0.0

    target_rows = []
    target_ids = list(target_info)
    for idx, tid in enumerate(target_ids):
        info = target_info[tid]
        genes = split_genes(info.get("pr_gene_symbol", "NA"))
        site = str(info.get("pr_p100_phosphosite", "NA"))
        target_rows.append(
            {
                "target_index": idx,
                "raw_name": info.get("id", tid),
                "target_id": tid,
                "molecule": info.get("pr_gene_symbol", "NA"),
                "site": site,
                "uniprot_id": info.get("pr_uniprot_id", "NA"),
                "modified_peptide": info.get("pr_p100_modified_peptide_code", "NA"),
                "primary_pathway": pathway_membership(genes)[0],
            }
        )

    condition_rows = []
    meta = pd.DataFrame(samples)
    for cond, sub in meta.groupby("condition", sort=False):
        first = sub.iloc[0]
        condition_rows.append(
            {
                "condition": cond,
                "condition_label": cond,
                "perturbation": first["perturbation"],
                "perturbation_id": first["perturbation_id"],
                "perturbation_type": first["perturbation_type"],
                "control_condition": first["control_condition"],
                "time": first["time"],
                "time_unit": first["time_unit"],
                "dose": first["dose"],
                "dose_unit": first["dose_unit"],
                "cell_type": first["cell_type"],
                "n_cells": int(len(sub)),
                "is_simulated": False,
            }
        )

    pathway_rows = []
    for idx, row in enumerate(target_rows):
        genes = split_genes(row["molecule"])
        for pathway in pathway_membership(genes):
            pathway_rows.append({"pathway": pathway, "raw_name": row["raw_name"], "target_index": idx, "target_id": row["target_id"]})

    np.save(out / "phospho_values.npy", z)
    np.save(out / "phospho_raw.npy", raw.astype(np.float32))
    np.save(out / "target_mask.npy", mask)
    meta.to_csv(out / "cell_metadata.tsv", sep="\t", index=False)
    pd.DataFrame(condition_rows).to_csv(out / "condition_table.tsv", sep="\t", index=False)
    pd.DataFrame(target_rows).to_csv(out / "phospho_target_table.tsv", sep="\t", index=False)
    pd.DataFrame(pathway_rows).to_csv(out / "pathway_target_manifest.tsv", sep="\t", index=False)
    with (out / "transform_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "source": "LINCS P100 Level 4 GCT",
                "transform": "per-target zscore over finite Level 4 values",
                "n_samples": int(z.shape[0]),
                "n_targets": int(z.shape[1]),
                "n_conditions": int(len(condition_rows)),
                "n_files": int(len(files)),
                "finite_fraction": float(mask.mean()),
            },
            fh,
            indent=2,
        )
    print(f"wrote {out} samples={z.shape[0]} targets={z.shape[1]} conditions={len(condition_rows)} finite={mask.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
