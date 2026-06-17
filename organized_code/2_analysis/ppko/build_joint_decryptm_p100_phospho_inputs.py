import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_decryptm_pxd037285_phospho_curve_inputs import (
    PATHWAY_GENES,
    clean_token,
    parse_float,
    parse_time_hours,
    pathway_membership,
    split_genes,
    target_genes_for_drug,
)


DEFAULT_ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
DEFAULT_DECRYPTM = DEFAULT_ROOT / r"01_data\single_cell\intermediate\phospho_perturb\decryptm_pxd037285_curve_v1"
DEFAULT_P100 = DEFAULT_ROOT / r"01_data\single_cell\intermediate\phospho_perturb\lincs_p100_lvl4_v1"
DEFAULT_OUT = DEFAULT_ROOT / r"01_data\single_cell\intermediate\phospho_perturb\decryptm_p100_joint_v1"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_bool(values):
    return pd.Series(values).astype(str).str.lower().isin(["true", "1", "yes", "y"])


def load_input_dir(input_dir, dataset_id):
    input_dir = Path(input_dir)
    x = np.load(input_dir / "phospho_values.npy").astype(np.float32)
    mask = np.load(input_dir / "target_mask.npy").astype(bool)
    meta = pd.read_csv(input_dir / "cell_metadata.tsv", sep="\t")
    targets = pd.read_csv(input_dir / "phospho_target_table.tsv", sep="\t")
    manifest = pd.read_csv(input_dir / "pathway_target_manifest.tsv", sep="\t")
    conditions = pd.read_csv(input_dir / "condition_table.tsv", sep="\t")
    priors_path = input_dir / "condition_pathway_prior.tsv"
    priors = pd.read_csv(priors_path, sep="\t") if priors_path.exists() else pd.DataFrame(columns=["condition", "pathway", "prior", "target_genes"])
    if "dataset_id" not in meta:
        meta["dataset_id"] = dataset_id
    else:
        meta["dataset_id"] = meta["dataset_id"].fillna(dataset_id)
    if "dataset_id" not in conditions:
        conditions["dataset_id"] = dataset_id
    else:
        conditions["dataset_id"] = conditions["dataset_id"].fillna(dataset_id)
    targets["source_dataset_id"] = dataset_id
    return {
        "input_dir": input_dir,
        "dataset_id": dataset_id,
        "x": x,
        "mask": mask,
        "meta": meta,
        "targets": targets,
        "manifest": manifest,
        "conditions": conditions,
        "priors": priors,
    }


def normalize_condition_table(conditions, dataset_id):
    out = conditions.copy()
    for col in ["condition", "condition_label", "perturbation", "perturbation_id", "perturbation_type", "control_condition", "time", "time_unit", "dose", "dose_unit", "cell_type", "target_genes"]:
        if col not in out:
            out[col] = ""
    if "time_hours" not in out:
        out["time_hours"] = [
            parse_time_hours(f"{t}{u}" if str(u).lower() not in {"h", "hour", "hours"} else t)
            for t, u in zip(out["time"], out["time_unit"])
        ]
    if "dose_molar" not in out:
        out["dose_molar"] = math.nan
    out["dataset_id"] = dataset_id
    out["target_genes"] = [
        value if str(value).strip() else ";".join(target_genes_for_drug(drug))
        for value, drug in zip(out["target_genes"], out["perturbation"])
    ]
    return out


def build_target_union(datasets):
    records = {}
    memberships = defaultdict(set)
    for ds in datasets:
        targets = ds["targets"].copy()
        manifest = ds["manifest"].copy()
        old_to_id = dict(zip(targets["target_index"].astype(int), targets["target_id"].astype(str)))
        for _, row in targets.iterrows():
            tid = str(row["target_id"])
            if tid not in records:
                records[tid] = {
                    "target_id": tid,
                    "raw_name": row.get("raw_name", tid),
                    "molecule": row.get("molecule", "NA"),
                    "site": row.get("site", "NA"),
                    "uniprot_id": row.get("uniprot_id", "NA"),
                    "modified_peptide": row.get("modified_peptide", "NA"),
                    "primary_pathway": row.get("primary_pathway", "global_phospho"),
                    "source_datasets": set(),
                }
            records[tid]["source_datasets"].add(ds["dataset_id"])
        for _, row in manifest.iterrows():
            old_idx = int(row["target_index"])
            tid = old_to_id.get(old_idx)
            if tid:
                memberships[tid].add(str(row["pathway"]))
        for _, row in targets.iterrows():
            tid = str(row["target_id"])
            genes = split_genes(row.get("molecule", ""))
            for pathway in pathway_membership(genes):
                memberships[tid].add(pathway)
    ordered = sorted(records)
    target_rows = []
    tid_to_new = {}
    for i, tid in enumerate(ordered):
        rec = records[tid]
        tid_to_new[tid] = i
        target_rows.append(
            {
                "target_index": i,
                "raw_name": rec["raw_name"],
                "target_id": tid,
                "molecule": rec["molecule"],
                "site": rec["site"],
                "uniprot_id": rec["uniprot_id"],
                "modified_peptide": rec["modified_peptide"],
                "primary_pathway": rec["primary_pathway"],
                "source_datasets": ";".join(sorted(rec["source_datasets"])),
            }
        )
    manifest_rows = []
    for tid, pathways in memberships.items():
        if tid not in tid_to_new:
            continue
        for pathway in sorted(pathways):
            manifest_rows.append({"pathway": pathway, "target_index": tid_to_new[tid], "target_id": tid, "raw_name": records[tid]["raw_name"]})
    return pd.DataFrame(target_rows), pd.DataFrame(manifest_rows), tid_to_new

def align_dataset(ds, tid_to_new, n_targets):
    targets = ds["targets"]
    old_tids = targets.sort_values("target_index")["target_id"].astype(str).tolist()
    old_to_new = np.array([tid_to_new[tid] for tid in old_tids], dtype=np.int64)
    x_out = np.zeros((ds["x"].shape[0], n_targets), dtype=np.float32)
    m_out = np.zeros((ds["x"].shape[0], n_targets), dtype=bool)
    x_out[:, old_to_new] = ds["x"]
    m_out[:, old_to_new] = ds["mask"]
    return x_out, m_out


def condition_priors_for_dataset(ds, condition_table):
    priors = ds["priors"].copy()
    rows = []
    if len(priors):
        for _, row in priors.iterrows():
            rows.append(
                {
                    "condition": row["condition"],
                    "pathway": row["pathway"],
                    "prior": float(row.get("prior", 1.0)),
                    "target_genes": row.get("target_genes", ""),
                    "dataset_id": ds["dataset_id"],
                }
            )
    existing = {(r["condition"], r["pathway"]) for r in rows}
    for _, row in condition_table.iterrows():
        genes = split_genes(row.get("target_genes", "")) or target_genes_for_drug(row.get("perturbation", ""))
        gene_set = set(genes)
        for pathway, members in PATHWAY_GENES.items():
            key = (row["condition"], pathway)
            if key in existing:
                continue
            if gene_set & {m.upper() for m in members}:
                rows.append(
                    {
                        "condition": row["condition"],
                        "pathway": pathway,
                        "prior": 1.0,
                        "target_genes": ";".join(sorted(gene_set)),
                        "dataset_id": ds["dataset_id"],
                    }
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decryptm-dir", default=str(DEFAULT_DECRYPTM))
    parser.add_argument("--p100-dir", default=str(DEFAULT_P100))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--exclude-p100", action="store_true")
    args = parser.parse_args()

    datasets = [load_input_dir(args.decryptm_dir, "decryptm_pxd037285_curve")]
    if not args.exclude_p100:
        datasets.append(load_input_dir(args.p100_dir, "lincs_p100_lvl4"))

    target_table, manifest, tid_to_new = build_target_union(datasets)
    x_parts = []
    mask_parts = []
    meta_parts = []
    cond_parts = []
    prior_parts = []
    offset = 0
    for ds in datasets:
        x_aligned, m_aligned = align_dataset(ds, tid_to_new, target_table.shape[0])
        x_parts.append(x_aligned)
        mask_parts.append(m_aligned)
        meta = ds["meta"].copy()
        meta["joint_row_index"] = np.arange(offset, offset + len(meta))
        meta_parts.append(meta)
        cond = normalize_condition_table(ds["conditions"], ds["dataset_id"])
        cond_parts.append(cond)
        prior_parts.append(condition_priors_for_dataset(ds, cond))
        offset += len(meta)

    x = np.vstack(x_parts).astype(np.float32)
    mask = np.vstack(mask_parts).astype(bool)
    meta = pd.concat(meta_parts, ignore_index=True, sort=False)
    conditions = pd.concat(cond_parts, ignore_index=True, sort=False).drop_duplicates("condition")
    priors = pd.concat(prior_parts, ignore_index=True, sort=False) if prior_parts else pd.DataFrame()

    out = ensure_dir(args.output_dir)
    np.save(out / "phospho_values.npy", x)
    np.save(out / "target_mask.npy", mask)
    meta.to_csv(out / "cell_metadata.tsv", sep="\t", index=False)
    conditions.to_csv(out / "condition_table.tsv", sep="\t", index=False)
    target_table.to_csv(out / "phospho_target_table.tsv", sep="\t", index=False)
    manifest.to_csv(out / "pathway_target_manifest.tsv", sep="\t", index=False)
    if len(priors):
        priors.to_csv(out / "condition_pathway_prior.tsv", sep="\t", index=False)
    with (out / "transform_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": "decryptM PXD037285 curve inputs plus LINCS P100 inputs",
                "n_samples": int(x.shape[0]),
                "n_targets": int(x.shape[1]),
                "n_conditions": int(conditions.shape[0]),
                "finite_fraction": float(mask.mean()),
                "datasets": [
                    {
                        "dataset_id": ds["dataset_id"],
                        "n_samples": int(ds["x"].shape[0]),
                        "n_targets": int(ds["x"].shape[1]),
                        "input_dir": str(ds["input_dir"]),
                    }
                    for ds in datasets
                ],
            },
            handle,
            indent=2,
        )
    print(f"wrote {out} samples={x.shape[0]} targets={x.shape[1]} conditions={conditions.shape[0]} finite={mask.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
