import argparse
import gzip
import json
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DATASET_ID = "gse300551_iccite_plex_kinase_2025"

PLATES = [
    {
        "plate": "D9",
        "rna": "GSM9064445_Species_Kinase_01_D9_filtered_feature_bc_matrix.h5",
        "tsb": "GSM9064460_Species_Kinase_01_D9_TSB_count_filtered_feature_bc_matrix.h5",
    },
    {
        "plate": "E9",
        "rna": "GSM9064446_Species_Kinase_02_E9_filtered_feature_bc_matrix.h5",
        "tsb": "GSM9064461_Species_Kinase_02_E9_TSB_count_filtered_feature_bc_matrix.h5",
    },
    {
        "plate": "F9",
        "rna": "GSM9064447_Species_Kinase_03_F9_filtered_feature_bc_matrix.h5",
        "tsb": "GSM9064462_Species_Kinase_03_F9_TSB_count_filtered_feature_bc_matrix.h5",
    },
    {
        "plate": "G9",
        "rna": "GSM9064448_Species_Kinase_04_G9_filtered_feature_bc_matrix.h5",
        "tsb": "GSM9064463_Species_Kinase_04_G9_TSB_count_filtered_feature_bc_matrix.h5",
    },
    {
        "plate": "H9",
        "rna": "GSM9064449_Species_Kinase_05_H9_filtered_feature_bc_matrix.h5",
        "tsb": "GSM9064464_Species_Kinase_05_H9_TSB_count_filtered_feature_bc_matrix.h5",
    },
]

TARGET_SPECS = [
    ("intra_B0042_HistoneH31PhoSer28", "HISTONE_H3_S28", "H3F3A;H3F3B", "S28", "strict_site"),
    ("intra_BHM03_NRF2PhoS40", "NFE2L2_S40", "NFE2L2", "S40", "strict_site"),
    ("intra_B1126_RPS6Pho", "RPS6_pSitePending", "RPS6", "pending_antibody_clone", "main"),
    ("intra_BHMIgG1003_phospho_p38_A16016A", "MAPK14_pSitePending", "MAPK14", "pending_antibody_clone", "main"),
    ("intra_BHM0007_phospho_p38_D3F9", "MAPK14_pSitePending_D3F9", "MAPK14", "pending_antibody_clone", "pending_confirmation"),
    ("intra_BHM0008_phospho_p65_93H1", "RELA_pSitePending_93H1", "RELA", "pending_antibody_clone", "pending_confirmation"),
    ("intra_BHM0010_phospho_c_Jun", "JUN_pSitePending", "JUN", "pending_antibody_clone", "pending_confirmation"),
    ("intra_phospho-LAT_pY226_J96-1238.58.93", "LAT_Y226", "LAT", "Y226", "strict_site"),
    ("intra_phospho-CREB_pS133_ATF-1_pS63_J151-21", "CREB1_S133_ATF1_S63", "CREB1;ATF1", "S133;S63", "strict_site"),
    ("intra_phophorylated_Rb", "p-Rb", "RB1", "pending_antibody_clone", "pending_confirmation"),
    ("intra_phospho-Akt_Ser473_D9E", "AKT_S473", "AKT1;AKT2;AKT3", "S473", "strict_site"),
    ("intra_phospho-Stat3_Tyr705_D3A7", "STAT3_Y705", "STAT3", "Y705", "main"),
    ("intra_phospho-4E-BP1_Thr37_46_236B4", "EIF4EBP1_T37_T46", "EIF4EBP1", "T37;T46", "strict_site"),
    ("intra_phospho-SLP-76_Ser376_E3G9U", "LCP2_S376", "LCP2", "S376", "strict_site"),
    ("intra_phospho-MEK1_pS298_J114-64", "MAP2K1_S298", "MAP2K1", "S298", "strict_site"),
    ("intra_B1097_anti-STAT6_Phospho_Tyr641", "STAT6_Y641", "STAT6", "Y641", "strict_site"),
    ("intra_B1100_anti-ZAP70_Phospho_Tyr493", "ZAP70_Y493", "ZAP70", "Y493", "strict_site"),
    ("intra_B1101_anti-Lck_Phospho_Tyr505", "LCK_Y505", "LCK", "Y505", "strict_site"),
    ("intra_B1110_anti-STAT5A_Phospho_Tyr694", "STAT5A_Y694", "STAT5A", "Y694", "strict_site"),
    ("intra_B1115_anti-STAT3_Phospho_Tyr705", "STAT3_Y705", "STAT3", "Y705", "main"),
    ("intra_B1073_anti-STAT3_Phospho_Tyr705", "STAT3_Y705", "STAT3", "Y705", "main"),
    ("intra_PLCg1_Phospho_Tyr783", "PLCG1_Y783", "PLCG1", "Y783", "strict_site"),
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_needed(raw_dir: Path, extract_dir: Path) -> None:
    ensure_dir(extract_dir)
    needed = {item["rna"] for item in PLATES} | {item["tsb"] for item in PLATES}
    missing = [name for name in needed if not (extract_dir / name).exists()]
    if not missing:
        return
    tar_path = raw_dir / "GSE300551_RAW.tar"
    if not tar_path.exists():
        raise FileNotFoundError(tar_path)
    with tarfile.open(tar_path, "r") as tf:
        members = [m for m in tf.getmembers() if Path(m.name).name in set(missing)]
        if len(members) != len(missing):
            found = {Path(m.name).name for m in members}
            raise RuntimeError(f"missing from tar: {sorted(set(missing) - found)}")
        tf.extractall(extract_dir, members=members)


def read_h5_barcodes(path: Path) -> list[str]:
    with h5py.File(path, "r") as h:
        return [x.decode() if isinstance(x, bytes) else str(x) for x in h["matrix"]["barcodes"][:]]


def read_h5_features(path: Path) -> list[str]:
    with h5py.File(path, "r") as h:
        return [x.decode() if isinstance(x, bytes) else str(x) for x in h["matrix"]["features"]["name"][:]]


def read_feature_values(path: Path, feature_names: list[str], wanted_features: list[str], barcode_indices: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    name_to_i = {name: i for i, name in enumerate(feature_names)}
    with h5py.File(path, "r") as h:
        g = h["matrix"]
        shape = tuple(int(x) for x in g["shape"][:])
        from scipy import sparse

        mat = sparse.csc_matrix(
            (g["data"][:], g["indices"][:], g["indptr"][:]),
            shape=shape,
            dtype=np.float32,
        )
        for feature in wanted_features:
            if feature not in name_to_i:
                continue
            vals = np.asarray(mat[name_to_i[feature], barcode_indices].todense()).ravel().astype(np.float32)
            out[feature] = np.log1p(vals).astype(np.float32)
    return out


def load_metadata_xls(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    meta_gz = raw_dir / "GSE300551_Additional_metadata.xls.gz"
    sheets = {}
    if not meta_gz.exists():
        return pd.DataFrame(), sheets
    tmp = raw_dir / "GSE300551_Additional_metadata.xls"
    if not tmp.exists():
        with gzip.open(meta_gz, "rb") as src:
            tmp.write_bytes(src.read())
    xls = pd.ExcelFile(tmp)
    for sheet in xls.sheet_names:
        sheets[sheet] = pd.read_excel(tmp, sheet_name=sheet)
    return sheets.get("Kinase_species_Drug_metadata", pd.DataFrame()), sheets


def normalize_old_target_table(old_tt: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    old_tt = old_tt.copy()
    old_tt["target_index"] = old_tt["target_index"].astype(int)
    target_to_index = {}
    for _, row in old_tt.sort_values("target_index").iterrows():
        target_to_index.setdefault(str(row["target_id"]), int(row["target_index"]))
    return old_tt, target_to_index


def build_target_mapping(old_tt: pd.DataFrame, paired_dir: Path) -> pd.DataFrame:
    old_tt, target_to_index = normalize_old_target_table(old_tt)
    next_index = max(target_to_index.values()) + 1 if target_to_index else 0
    rows = []
    for feature_id, target_id, protein_symbol, residue, tier in TARGET_SPECS:
        if target_id not in target_to_index:
            target_to_index[target_id] = next_index
            next_index += 1
        target_index = target_to_index[target_id]
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "feature_id": feature_id,
                "target_id": target_id,
                "site_id": target_id if residue and "pending" not in residue else "",
                "target_index": target_index,
                "protein_symbol": protein_symbol,
                "residue": residue,
                "canonical_label": f"{protein_symbol}_{residue}",
                "evaluation_tier": tier,
                "include_in_loss": True,
                "value_transform": "log1p_tsb_umi",
            }
        )
    mapping = pd.DataFrame(rows)
    mapping.to_csv(paired_dir / "target_mapping.tsv", sep="\t", index=False)
    return mapping


def build_paired(root: Path) -> None:
    raw_dir = root / "01_data" / "single_cell" / "raw" / "external_single_cell_phospho_validation_v1" / "icCITE-plex_GSE300551"
    extract_dir = root / "01_data" / "single_cell" / "intermediate" / "gse300551_iccite_plex_kinase_2025_extracted"
    paired_dir = ensure_dir(root / "01_data" / "single_cell" / "intermediate" / "paired_matrices" / DATASET_ID)
    extract_needed(raw_dir, extract_dir)

    old_input = root / "01_data" / "single_cell" / "intermediate" / "phospho_model_inputs" / "scfoundation_cap12000_masked_multisite_v1"
    old_tt = pd.read_csv(old_input / "phospho_target_table.tsv", sep="\t")
    mapping = build_target_mapping(old_tt, paired_dir)
    wanted_features = list(dict.fromkeys(mapping["feature_id"].astype(str).tolist()))

    drug_meta, sheets = load_metadata_xls(raw_dir)
    if not drug_meta.empty:
        drug_meta.to_csv(paired_dir / "drug_metadata.tsv", sep="\t", index=False)
    for name, df in sheets.items():
        df.to_csv(paired_dir / f"metadata_sheet_{name.replace(' ', '_')}.tsv", sep="\t", index=False)

    all_cell_ids = []
    all_meta = []
    all_values = {feature: [] for feature in wanted_features}
    rna_manifest = []
    total_rna = 0
    total_tsb = 0

    for item in PLATES:
        plate = item["plate"]
        rna_path = extract_dir / item["rna"]
        tsb_path = extract_dir / item["tsb"]
        rna_barcodes = read_h5_barcodes(rna_path)
        tsb_barcodes = read_h5_barcodes(tsb_path)
        total_rna += len(rna_barcodes)
        total_tsb += len(tsb_barcodes)
        tsb_index = {bc: i for i, bc in enumerate(tsb_barcodes)}
        overlap_raw = [bc for bc in rna_barcodes if bc in tsb_index]
        tsb_pos = np.asarray([tsb_index[bc] for bc in overlap_raw], dtype=np.int64)
        cell_ids = [f"{plate}_{bc}" for bc in overlap_raw]
        feature_names = read_h5_features(tsb_path)
        vals = read_feature_values(tsb_path, feature_names, wanted_features, tsb_pos)
        for feature in wanted_features:
            if feature in vals:
                all_values[feature].append(vals[feature])
            else:
                all_values[feature].append(np.full(len(cell_ids), np.nan, dtype=np.float32))
        all_cell_ids.extend(cell_ids)
        all_meta.append(
            pd.DataFrame(
                {
                    "cell_id": cell_ids,
                    "raw_barcode": overlap_raw,
                    "plate": plate,
                    "dataset_id": DATASET_ID,
                    "cell_type_label": "T_cell_mixed_species",
                    "state_label": "kinase_screen",
                    "time_min": np.nan,
                    "ibrutinib": False,
                    "sample_key": plate,
                    "split_group": plate,
                    "rna_h5": item["rna"],
                    "tsb_h5": item["tsb"],
                }
            )
        )
        rna_manifest.append(
            {
                "plate": plate,
                "rna_h5": str(rna_path),
                "tsb_h5": str(tsb_path),
                "n_rna_barcodes": len(rna_barcodes),
                "n_tsb_barcodes": len(tsb_barcodes),
                "n_paired_barcodes": len(cell_ids),
            }
        )

    meta = pd.concat(all_meta, ignore_index=True)
    phospho = pd.DataFrame(index=all_cell_ids)
    for feature, parts in all_values.items():
        phospho[feature] = np.concatenate(parts).astype(np.float32)

    meta.to_csv(paired_dir / "cell_metadata.tsv", sep="\t", index=False)
    pd.Series(all_cell_ids).to_csv(paired_dir / "barcodes.tsv", sep="\t", index=False, header=False)
    phospho.to_csv(paired_dir / "phospho_counts.tsv", sep="\t", index_label="cell_id")
    pd.DataFrame(rna_manifest).to_csv(paired_dir / "rna_h5_manifest.tsv", sep="\t", index=False)
    genes = read_h5_features(extract_dir / PLATES[0]["rna"])
    pd.Series(genes).to_csv(paired_dir / "genes.tsv", sep="\t", index=False, header=False)

    manifest = {
        "dataset_id": DATASET_ID,
        "n_cells": int(len(meta)),
        "n_rna_barcodes_total": int(total_rna),
        "n_tsb_barcodes_total": int(total_tsb),
        "n_phospho_features": int(phospho.shape[1]),
        "value_transform": "log1p_tsb_umi",
        "raw_dir": str(raw_dir),
        "extract_dir": str(extract_dir),
        "paired_dir": str(paired_dir),
    }
    (paired_dir / "paired_matrix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def build_model_input(root: Path, output_name: str) -> None:
    old_input = root / "01_data" / "single_cell" / "intermediate" / "phospho_model_inputs" / "scfoundation_cap12000_masked_multisite_v1"
    paired_dir = root / "01_data" / "single_cell" / "intermediate" / "paired_matrices" / DATASET_ID
    out_dir = ensure_dir(root / "01_data" / "single_cell" / "intermediate" / "phospho_model_inputs" / output_name)

    old_meta = pd.read_csv(old_input / "cell_metadata.tsv", sep="\t", low_memory=False)
    old_y = np.load(old_input / "targets.npy", mmap_mode="r")
    old_mask = np.load(old_input / "target_mask.npy", mmap_mode="r")
    old_tt = pd.read_csv(old_input / "phospho_target_table.tsv", sep="\t")
    mapping = pd.read_csv(paired_dir / "target_mapping.tsv", sep="\t")
    phospho = pd.read_csv(paired_dir / "phospho_counts.tsv", sep="\t", index_col=0)
    new_meta = pd.read_csv(paired_dir / "cell_metadata.tsv", sep="\t", low_memory=False)

    max_target = int(max(old_tt["target_index"].max(), mapping["target_index"].max())) + 1
    old_y_full = np.full((old_y.shape[0], max_target), np.nan, dtype=np.float32)
    old_m_full = np.zeros((old_mask.shape[0], max_target), dtype=bool)
    old_y_full[:, : old_y.shape[1]] = np.asarray(old_y, dtype=np.float32)
    old_m_full[:, : old_mask.shape[1]] = np.asarray(old_mask, dtype=bool)

    new_y = np.full((len(new_meta), max_target), np.nan, dtype=np.float32)
    new_m = np.zeros((len(new_meta), max_target), dtype=bool)
    for target_index, sub in mapping.groupby("target_index"):
        cols = [c for c in sub["feature_id"].astype(str).tolist() if c in phospho.columns]
        if not cols:
            continue
        vals = phospho.loc[new_meta["cell_id"].astype(str).tolist(), cols].to_numpy(dtype=np.float32)
        with np.errstate(invalid="ignore"):
            merged = np.nanmean(vals, axis=1).astype(np.float32)
        ok = np.isfinite(merged)
        new_y[ok, int(target_index)] = merged[ok]
        new_m[ok, int(target_index)] = True

    row_offset = len(old_meta)
    new_meta = new_meta.copy()
    new_meta["row_index"] = np.arange(row_offset, row_offset + len(new_meta), dtype=np.int64)
    meta = pd.concat([old_meta, new_meta], ignore_index=True)
    y = np.vstack([old_y_full, new_y]).astype(np.float32)
    mask = np.vstack([old_m_full, new_m]).astype(bool)

    rows = [old_tt]
    gse_rows = []
    for _, row in mapping.iterrows():
        vals = new_y[new_m[:, int(row["target_index"])], int(row["target_index"])]
        gse_rows.append(
            {
                **row.to_dict(),
                "n_observed": int(np.isfinite(vals).sum()),
                "variance": float(np.nanvar(vals)) if len(vals) else np.nan,
                "n_nonzero": int(np.sum(vals != 0)) if len(vals) else 0,
                "n_datasets": 1,
                "n_observed_feature": int(np.isfinite(vals).sum()),
                "variance_feature": float(np.nanvar(vals)) if len(vals) else np.nan,
                "n_nonzero_feature": int(np.sum(vals != 0)) if len(vals) else 0,
                "n_cells_feature": int(len(new_meta)),
            }
        )
    rows.append(pd.DataFrame(gse_rows))
    target_table = pd.concat(rows, ignore_index=True, sort=False)
    target_table = target_table.sort_values(["target_index", "dataset_id", "feature_id"]).reset_index(drop=True)

    np.save(out_dir / "targets.npy", y)
    np.save(out_dir / "target_mask.npy", mask)
    meta.to_csv(out_dir / "cell_metadata.tsv", sep="\t", index=False)
    target_table.to_csv(out_dir / "phospho_target_table.tsv", sep="\t", index=False)
    manifest = {
        "model_input": str(out_dir),
        "source_input": str(old_input),
        "added_dataset": DATASET_ID,
        "n_cells": int(y.shape[0]),
        "n_targets": int(y.shape[1]),
        "n_observed_values": int(mask.sum()),
        "note": "No embeddings.npy is written here; SCP682-SC3 uses Geneformer pathway features built from paired RNA matrices.",
    }
    (out_dir / "model_input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    ap.add_argument("--output-name", default="geneformer_gse300551_iccite_v1")
    ap.add_argument("--paired-only", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    build_paired(root)
    if not args.paired_only:
        build_model_input(root, args.output_name)


if __name__ == "__main__":
    main()
