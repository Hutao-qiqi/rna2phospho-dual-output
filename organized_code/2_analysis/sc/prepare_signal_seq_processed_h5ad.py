import gzip
import json
import shutil
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
RAW_DIR = ROOT / r"01_data\single_cell\raw\external_single_cell_phospho_validation_v1\SIGNAL-seq_GSE256405\processed_h5ad"
UNPACK_DIR = ROOT / r"01_data\single_cell\intermediate\SIGNAL-seq_GSE256405\processed_h5ad_unpacked"
PAIRED_DIR = ROOT / r"01_data\single_cell\intermediate\paired_matrices\signal_seq_gse256405_processed_h5ad_v1"
OUT_DIR = ROOT / r"02_results\single_cell\20260519_signal_seq_h5ad_integrity_v1"


DATASETS = [
    {
        "dataset_id": "signal_seq_gse256403_hela_2024",
        "sample_label": "HeLa",
        "rna_gz": "GSE256403_ex0003_hela_rna_adata.h5ad.gz",
        "adt_gz": "GSE256403_ex0003_hela_adt_adata.h5ad.gz",
        "feature_reference_gz": "GSE256403_ex0003_adt_feature_reference.csv.gz",
    },
    {
        "dataset_id": "signal_seq_gse256404_pdo_caf_2024",
        "sample_label": "PDO_CAF",
        "rna_gz": "GSE256404_ex0015_pdo_rna_adata.h5ad.gz",
        "adt_gz": "GSE256404_ex0015_pdo_adt_adata.h5ad.gz",
        "feature_reference_gz": "GSE256404_ex0015_feature_reference.csv.gz",
    },
]


TARGET_HINTS = {
    "pS6": "RPS6_pSitePending",
    "pRB": "p-Rb",
    "pHistone H3": "HISTONE_H3_S28",
    "pP38": "MAPK14_pSitePending",
    "pP38 MAPK": "MAPK14_pSitePending",
    "pAKT_T308": "AKT_pSitePending",
    "p4E_BP1": "EIF4EBP1_T37_T46",
    "pBTK": "BTK_Y551",
    "pNF_kB_p65": "p-p65",
    "pPDPK1": "PDPK1_S241",
    "pMKK4": "MAP2K4_S257",
    "pNDRG1": "NDRG1_T346",
    "pP120": "CTNND1_T310",
    "pHistone H2A": "H2AFX_S139",
}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def unpack_gzip(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        log(f"use unpacked {dst.name} bytes={dst.stat().st_size}")
        return dst
    ensure_dir(dst.parent)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    log(f"unpack {src.name} -> {dst}")
    with gzip.open(src, "rb") as fin, tmp.open("wb") as fout:
        shutil.copyfileobj(fin, fout, length=64 * 1024 * 1024)
    tmp.replace(dst)
    log(f"unpack done {dst.name} bytes={dst.stat().st_size}")
    return dst


def h5ad_summary(path, dataset_id, modality):
    a = ad.read_h5ad(path, backed="r")
    x = a.X
    if sparse.issparse(x):
        x_type = type(x).__name__
    else:
        x_type = type(x).__name__
    row = {
        "dataset_id": dataset_id,
        "modality": modality,
        "path": str(path),
        "n_obs": int(a.n_obs),
        "n_vars": int(a.n_vars),
        "x_type": x_type,
        "obs_columns": ";".join(map(str, a.obs.columns.tolist())),
        "var_columns": ";".join(map(str, a.var.columns.tolist())),
        "obs_name_preview": ";".join(map(str, a.obs_names[:5].tolist())),
        "var_name_preview": ";".join(map(str, a.var_names[:10].tolist())),
    }
    obs = a.obs.head(200).copy()
    obs.insert(0, "obs_name", a.obs_names[: len(obs)].astype(str))
    var = a.var.copy()
    var.insert(0, "var_name", a.var_names.astype(str))
    a.file.close()
    return row, obs, var


def read_adt_matrix(path):
    a = ad.read_h5ad(path)
    obs = a.obs.copy()
    obs.index = obs.index.astype(str)
    var = a.var.copy()
    var.index = var.index.astype(str)
    x = a.X
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    matrix = pd.DataFrame(x, index=obs.index.astype(str), columns=var.index.astype(str))
    return obs, var, matrix


def obs_frame_with_name(adata):
    obs = adata.obs.copy()
    obs.insert(0, "obs_name", adata.obs_names.astype(str))
    return obs


def key_candidates(obs):
    out = {}
    obs_name = obs["obs_name"].astype(str)
    out["obs_name"] = obs_name
    if "barcode_seq" in obs.columns:
        out["barcode_seq"] = obs["barcode_seq"].astype(str)
    if "sublib_index" in obs.columns:
        sub = obs["sublib_index"].astype(str)
        out["obs_name_sublib_index"] = obs_name + "_" + sub
        if "barcode_seq" in obs.columns:
            out["barcode_seq_sublib_index"] = obs["barcode_seq"].astype(str) + "_" + sub
    stripped = obs_name.str.replace(r"_[0-9]+$", "", regex=True)
    if not stripped.equals(obs_name):
        out["obs_name_strip_numeric_suffix"] = stripped
    return out


def choose_pairing_keys(rna_obs, adt_obs):
    rna_keys = key_candidates(rna_obs)
    adt_keys = key_candidates(adt_obs)
    best = None
    for rna_name, rna_values in rna_keys.items():
        rna_set = set(rna_values.astype(str))
        for adt_name, adt_values in adt_keys.items():
            n = len(rna_set.intersection(set(adt_values.astype(str))))
            row = (n, rna_name, adt_name, rna_values.astype(str), adt_values.astype(str))
            if best is None or n > best[0]:
                best = row
    if best is None:
        raise RuntimeError("no pairing key candidates")
    return {
        "n_overlap_key_values": int(best[0]),
        "rna_key_name": best[1],
        "adt_key_name": best[2],
        "rna_key": best[3].reset_index(drop=True),
        "adt_key": best[4].reset_index(drop=True),
    }


def normalize_feature_name(name):
    return (
        str(name)
        .replace(" ", "_")
        .replace("[", "")
        .replace("]", "")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def propose_target_id(antigen):
    antigen = str(antigen)
    for key, value in TARGET_HINTS.items():
        if key in antigen:
            return value
    return ""


def main():
    ensure_dir(UNPACK_DIR)
    ensure_dir(PAIRED_DIR)
    ensure_dir(OUT_DIR / "tables")
    ensure_dir(OUT_DIR / "reports")
    ensure_dir(OUT_DIR / "logs")

    file_rows = []
    obs_preview_rows = []
    var_rows = []
    pair_rows = []
    feature_rows = []
    cell_meta_rows = []
    phospho_tables = []
    manifest_rows = []
    mapping_rows = []

    for ds in DATASETS:
        dataset_id = ds["dataset_id"]
        log(f"process {dataset_id}")
        rna_h5ad = unpack_gzip(RAW_DIR / ds["rna_gz"], UNPACK_DIR / ds["rna_gz"].replace(".gz", ""))
        adt_h5ad = unpack_gzip(RAW_DIR / ds["adt_gz"], UNPACK_DIR / ds["adt_gz"].replace(".gz", ""))

        for modality, path in [("RNA", rna_h5ad), ("ADT", adt_h5ad)]:
            row, obs_preview, var = h5ad_summary(path, dataset_id, modality)
            file_rows.append(row)
            obs_preview.insert(0, "modality", modality)
            obs_preview.insert(0, "dataset_id", dataset_id)
            obs_preview_rows.append(obs_preview)
            var.insert(0, "modality", modality)
            var.insert(0, "dataset_id", dataset_id)
            var_rows.append(var)

        rna = ad.read_h5ad(rna_h5ad, backed="r")
        rna_obs_df = obs_frame_with_name(rna)
        rna_obs = pd.Index(rna_obs_df["obs_name"].astype(str))
        rna.file.close()
        adt_obs, adt_var, adt_matrix = read_adt_matrix(adt_h5ad)
        adt_obs_df = adt_obs.copy()
        adt_obs_df.insert(0, "obs_name", adt_obs.index.astype(str))
        adt_index = pd.Index(adt_obs_df["obs_name"].astype(str))
        direct_overlap = rna_obs.intersection(adt_index)
        key_info = choose_pairing_keys(rna_obs_df, adt_obs_df)
        adt_first = {}
        for i, key in enumerate(key_info["adt_key"].tolist()):
            if key not in adt_first:
                adt_first[key] = i
        rna_pos = []
        adt_pos = []
        overlap_keys = []
        for i, key in enumerate(key_info["rna_key"].tolist()):
            if key in adt_first:
                rna_pos.append(i)
                adt_pos.append(adt_first[key])
                overlap_keys.append(key)
        pair_rows.append(
            {
                "dataset_id": dataset_id,
                "sample_label": ds["sample_label"],
                "n_rna_cells": int(len(rna_obs_df)),
                "n_adt_cells": int(len(adt_obs_df)),
                "n_direct_obs_name_overlap": int(len(direct_overlap)),
                "pairing_rna_key": key_info["rna_key_name"],
                "pairing_adt_key": key_info["adt_key_name"],
                "n_overlap_cells": int(len(overlap_keys)),
                "rna_overlap_fraction": float(len(overlap_keys) / max(len(rna_obs_df), 1)),
                "adt_overlap_fraction": float(len(overlap_keys) / max(len(adt_obs_df), 1)),
                "n_duplicate_rna_pair_keys": int(key_info["rna_key"].duplicated().sum()),
                "n_duplicate_adt_pair_keys": int(key_info["adt_key"].duplicated().sum()),
            }
        )

        feature_ref = pd.read_csv(RAW_DIR / ds["feature_reference_gz"])
        feature_ref.insert(0, "dataset_id", dataset_id)
        feature_ref.insert(1, "sample_label", ds["sample_label"])
        feature_ref["proposed_target_id"] = feature_ref["Antigen"].map(propose_target_id)
        feature_ref["normalized_feature_id"] = feature_ref["Antigen"].map(normalize_feature_name)
        feature_rows.append(feature_ref)

        paired_adt = adt_matrix.iloc[adt_pos].copy()
        rename = {}
        for col in paired_adt.columns:
            match = feature_ref[
                feature_ref["ADT_identifier"].astype(str).eq(str(col))
                | feature_ref["Antigen"].astype(str).eq(str(col))
                | feature_ref["normalized_feature_id"].astype(str).eq(str(col))
            ]
            if len(match):
                rename[col] = str(match.iloc[0]["normalized_feature_id"])
            else:
                rename[col] = normalize_feature_name(col)
        paired_adt = paired_adt.rename(columns=rename)
        selected_rna_obs = rna_obs_df.iloc[rna_pos].reset_index(drop=True)
        prefixed_index = [f"{dataset_id}:{x}" for x in overlap_keys]
        paired_adt.index = prefixed_index
        phospho_tables.append(paired_adt)

        raw_barcode = selected_rna_obs["barcode_seq"].astype(str) if "barcode_seq" in selected_rna_obs.columns else selected_rna_obs["obs_name"].astype(str)
        sample_id = selected_rna_obs["sample_id"].astype(str) if "sample_id" in selected_rna_obs.columns else ds["sample_label"]
        cell_type = selected_rna_obs["cell_type"].astype(str) if "cell_type" in selected_rna_obs.columns else ds["sample_label"]
        state = selected_rna_obs["cell_type_condition"].astype(str) if "cell_type_condition" in selected_rna_obs.columns else sample_id
        meta = pd.DataFrame(
            {
                "cell_id": prefixed_index,
                "raw_barcode": raw_barcode.to_numpy(dtype=str),
                "pair_key": np.asarray(overlap_keys, dtype=str),
                "dataset_id": dataset_id,
                "sample_label": ds["sample_label"],
                "cell_type_label": np.asarray(cell_type, dtype=str),
                "state_label": np.asarray(state, dtype=str),
                "sample_id": np.asarray(sample_id, dtype=str),
                "rna_h5ad": str(rna_h5ad),
                "adt_h5ad": str(adt_h5ad),
            }
        )
        cell_meta_rows.append(meta)
        manifest_rows.append(
            {
                "dataset_id": dataset_id,
                "sample_label": ds["sample_label"],
                "rna_h5ad": str(rna_h5ad),
                "adt_h5ad": str(adt_h5ad),
                "feature_reference": str(RAW_DIR / ds["feature_reference_gz"]),
                "n_paired_cells": int(len(meta)),
                "n_adt_features": int(paired_adt.shape[1]),
            }
        )

        for _, row in feature_ref.iterrows():
            mapping_rows.append(
                {
                    "dataset_id": dataset_id,
                    "sample_label": ds["sample_label"],
                    "feature_id": str(row["normalized_feature_id"]),
                    "antigen": str(row["Antigen"]),
                    "clone": str(row.get("Clone", "")),
                    "ag_category": str(row.get("Ag_category", "")),
                    "proposed_target_id": str(row.get("proposed_target_id", "")),
                    "include_for_phospho_validation": bool(str(row.get("Ag_category", "")).lower() == "signalling"),
                }
            )

    pd.DataFrame(file_rows).to_csv(OUT_DIR / "tables" / "h5ad_file_summary.tsv", sep="\t", index=False)
    if obs_preview_rows:
        pd.concat(obs_preview_rows, ignore_index=True).to_csv(OUT_DIR / "tables" / "obs_preview.tsv", sep="\t", index=False)
    if var_rows:
        pd.concat(var_rows, ignore_index=True).to_csv(OUT_DIR / "tables" / "var_table.tsv", sep="\t", index=False)
    pd.DataFrame(pair_rows).to_csv(OUT_DIR / "tables" / "rna_adt_barcode_overlap.tsv", sep="\t", index=False)
    pd.concat(feature_rows, ignore_index=True).to_csv(OUT_DIR / "tables" / "adt_feature_reference.tsv", sep="\t", index=False)
    pd.DataFrame(mapping_rows).to_csv(OUT_DIR / "tables" / "adt_feature_mapping_candidates.tsv", sep="\t", index=False)

    cell_meta = pd.concat(cell_meta_rows, ignore_index=True)
    phospho = pd.concat(phospho_tables, axis=0, sort=False).fillna(0.0).astype(np.float32)
    cell_meta.to_csv(PAIRED_DIR / "cell_metadata.tsv", sep="\t", index=False)
    pd.Series(cell_meta["cell_id"].astype(str)).to_csv(PAIRED_DIR / "barcodes.tsv", sep="\t", index=False, header=False)
    phospho.to_csv(PAIRED_DIR / "phospho_counts.tsv", sep="\t", index_label="cell_id")
    pd.DataFrame(manifest_rows).to_csv(PAIRED_DIR / "rna_adt_h5ad_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(mapping_rows).to_csv(PAIRED_DIR / "target_mapping_candidates.tsv", sep="\t", index=False)

    manifest = {
        "dataset_id": "signal_seq_gse256405_processed_h5ad_v1",
        "raw_dir": str(RAW_DIR),
        "unpack_dir": str(UNPACK_DIR),
        "paired_dir": str(PAIRED_DIR),
        "result_dir": str(OUT_DIR),
        "n_datasets": len(DATASETS),
        "n_cells": int(len(cell_meta)),
        "n_adt_features_union": int(phospho.shape[1]),
        "n_done_files": len(file_rows),
        "pairing": pair_rows,
    }
    (PAIRED_DIR / "paired_matrix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT_DIR / "reports" / "signal_seq_h5ad_integrity_summary.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    log(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
