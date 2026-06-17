import argparse
import gzip
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


SAMPLES = {
    "1a": {
        "rna_gsm": "GSM4952447",
        "protein_gsm": "GSM4952455",
        "title": "BJABs 0 min aIg",
        "treatment": "Untreated",
        "time_min": 0,
        "bcr_stim": False,
        "ibrutinib": False,
    },
    "3a": {
        "rna_gsm": "GSM4952448",
        "protein_gsm": "GSM4952456",
        "title": "BJABs 2 min aIg",
        "treatment": "2 min polyclonal anti-immunoglobulin antibody",
        "time_min": 2,
        "bcr_stim": True,
        "ibrutinib": False,
    },
    "4a": {
        "rna_gsm": "GSM4952449",
        "protein_gsm": "GSM4952457",
        "title": "BJABs 4 min aIg",
        "treatment": "4 min polyclonal anti-immunoglobulin antibody",
        "time_min": 4,
        "bcr_stim": True,
        "ibrutinib": False,
    },
    "5a": {
        "rna_gsm": "GSM4952450",
        "protein_gsm": "GSM4952458",
        "title": "BJABs 6 min aIg",
        "treatment": "6 min polyclonal anti-immunoglobulin antibody",
        "time_min": 6,
        "bcr_stim": True,
        "ibrutinib": False,
    },
    "6a": {
        "rna_gsm": "GSM4952451",
        "protein_gsm": "GSM4952459",
        "title": "BJABs 6 min aIg + Ibrutinib",
        "treatment": "6 min polyclonal anti-immunoglobulin antibody + Ibrutinib",
        "time_min": 6,
        "bcr_stim": True,
        "ibrutinib": True,
    },
    "10a": {
        "rna_gsm": "GSM4952452",
        "protein_gsm": "GSM4952460",
        "title": "BJABs 60 min aIg",
        "treatment": "60 min polyclonal anti-immunoglobulin antibody",
        "time_min": 60,
        "bcr_stim": True,
        "ibrutinib": False,
    },
    "11a": {
        "rna_gsm": "GSM4952453",
        "protein_gsm": "GSM4952461",
        "title": "BJABs 180 min aIg",
        "treatment": "180 min polyclonal anti-immunoglobulin antibody",
        "time_min": 180,
        "bcr_stim": True,
        "ibrutinib": False,
    },
    "12a": {
        "rna_gsm": "GSM4952454",
        "protein_gsm": "GSM4952462",
        "title": "BJABs 180 min aIg + Ibrutinib",
        "treatment": "180 min polyclonal anti-immunoglobulin antibody + Ibrutinib",
        "time_min": 180,
        "bcr_stim": True,
        "ibrutinib": True,
    },
}


PHOSPHO_CANONICAL = {
    "p-AMPK-a1/2": "PRKAA1_or_PRKAA2",
    "p-AMPK-b1": "PRKAB1",
    "p-Akt": "AKT1_or_AKT2_or_AKT3",
    "p-BLNK": "BLNK",
    "p-Btk": "BTK",
    "p-CD79a": "CD79A",
    "p-CDK1": "CDK1",
    "p-CDK4": "CDK4",
    "p-CDK6": "CDK6",
    "p-Erk1/2": "MAPK1_or_MAPK3",
    "p-HistonH2A.X": "H2AX",
    "p-HistonH3": "H3",
    "p-IKKa/b": "CHUK_or_IKBKB",
    "p-IRAK4": "IRAK4",
    "p-JAK1": "JAK1",
    "p-p65": "RELA_or_NFKB_p65",
    "p-SYK": "SYK",
    "p-Syk": "SYK",
    "p-JNK": "JNK_MAPK8_9",
    "p-p38": "p38_MAPK11_14",
    "p-PKC-b1": "PRKCB",
    "p-PLC-y2": "PLCG2",
    "p-PLC-y2Y759": "PLCG2",
    "p-BTK": "BTK",
    "p-Rb": "RB1",
    "p-S6": "RPS6",
    "p-SHP-2": "PTPN11",
    "p-SHP-1": "PTPN6",
    "p-STAT1": "STAT1",
    "p-STAT3": "STAT3",
    "p-STAT5": "STAT5",
    "p-STAT6": "STAT6",
    "p-Src": "SRC",
    "p-TOR": "MTOR",
    "p-RB": "RB1",
    "p-c-JUN": "JUN",
    "p-c-Jun": "JUN",
}

PHOSPHO_RESIDUE = {
    "p-PLC-y2Y759": "Y759",
}


def strip_protein_barcode(value: str) -> str:
    match = re.search(r"_(bc[A-Z]+)$", value)
    return match.group(1) if match else value


def find_rna_file(extracted: Path, key: str) -> Path:
    hits = sorted(extracted.glob(f"*_{key}mRNA*.gz")) + sorted(extracted.glob(f"*_{key}mrna*.gz"))
    hits = sorted(set(hits))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected one RNA file for {key}, found {len(hits)}")
    return hits[0]


def find_protein_file(extracted: Path, key: str) -> Path:
    hits = sorted(extracted.glob(f"*_{key}prot*.gz"))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected one protein file for {key}, found {len(hits)}")
    return hits[0]


def extract_tar(raw_dir: Path, extracted: Path) -> None:
    extracted.mkdir(parents=True, exist_ok=True)
    sentinel = extracted / "GSM4952447_1amRNA.counts.tsv.gz"
    if sentinel.exists():
        return
    shutil.unpack_archive(str(raw_dir / "GSE162461_RAW.tar"), str(extracted))


def read_count_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", compression="gzip", index_col=0)
    table.index = table.index.astype(str)
    return table.astype(np.float32, copy=False)


def read_antibody_annotation(raw_dir: Path, matrix_features: list[str]) -> pd.DataFrame:
    xlsx = raw_dir / "GSE162461_Ab_barcodes.xlsx"
    raw = pd.read_excel(xlsx, header=None)
    header_idx = None
    for idx, row in raw.iterrows():
        if str(row.iloc[0]).strip() == "Target":
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("Could not find Target row in antibody barcode file")

    table = raw.iloc[header_idx + 1 :, :2].copy()
    table.columns = ["feature", "barcode_sequence"]
    table = table.dropna(subset=["feature"])
    table["feature"] = table["feature"].astype(str).str.strip()
    table["barcode_sequence"] = table["barcode_sequence"].astype(str).str.strip()

    by_feature = table.set_index("feature")["barcode_sequence"].to_dict()
    rows = []
    for feature in matrix_features:
        is_phospho = feature.startswith("p-")
        rows.append(
            {
                "feature": feature,
                "barcode_sequence": by_feature.get(feature, ""),
                "is_phospho": bool(is_phospho),
                "feature_class": "phospho" if is_phospho else "total_or_marker",
                "canonical_target": PHOSPHO_CANONICAL.get(feature, feature),
                "residue": PHOSPHO_RESIDUE.get(feature, "unknown"),
            }
        )
    return pd.DataFrame(rows)


def write_mtx_gz(matrix: sp.csr_matrix, path: Path) -> None:
    tmp = path.with_suffix("")
    scipy.io.mmwrite(str(tmp), matrix.tocoo())
    with open(tmp, "rb") as src, gzip.open(path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    tmp.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    parser.add_argument("--dataset-id", default="qurie_seq_bjab_2021")
    args = parser.parse_args()

    root = Path(args.root)
    raw_dir = root / "01_data" / "single_cell" / "raw" / args.dataset_id
    extracted = raw_dir / "extracted"
    out_dir = root / "01_data" / "single_cell" / "intermediate" / "paired_matrices" / args.dataset_id
    result_dir = root / "02_results" / "single_cell" / "20260511_qurie_seq_import"
    table_dir = result_dir / "tables"

    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    extract_tar(raw_dir, extracted)

    genes = None
    protein_features = None
    rna_blocks = []
    protein_blocks = []
    metadata_rows = []
    sample_rows = []

    for sample_key, meta in SAMPLES.items():
        rna_file = find_rna_file(extracted, sample_key)
        protein_file = find_protein_file(extracted, sample_key)

        rna_df = read_count_table(rna_file)
        protein_df = read_count_table(protein_file)
        protein_df.columns = [strip_protein_barcode(c) for c in protein_df.columns]

        if genes is None:
            genes = list(rna_df.columns)
        elif genes != list(rna_df.columns):
            raise ValueError(f"Gene columns differ in {sample_key}")

        if protein_features is None:
            protein_features = list(protein_df.index)
        elif protein_features != list(protein_df.index):
            raise ValueError(f"Protein features differ in {sample_key}")

        paired_barcodes = [bc for bc in rna_df.index if bc in protein_df.columns]
        cell_ids = [f"{sample_key}_{bc}" for bc in paired_barcodes]

        rna_block = sp.csr_matrix(rna_df.loc[paired_barcodes].to_numpy(dtype=np.float32))
        protein_block = protein_df.loc[:, paired_barcodes].T
        protein_block.index = cell_ids

        rna_blocks.append(rna_block)
        protein_blocks.append(protein_block)

        sample_rows.append(
            {
                "sample_key": sample_key,
                "rna_file": rna_file.name,
                "protein_file": protein_file.name,
                "rna_cells": int(rna_df.shape[0]),
                "protein_cells": int(protein_df.shape[1]),
                "paired_cells": int(len(paired_barcodes)),
                "time_min": meta["time_min"],
                "bcr_stim": meta["bcr_stim"],
                "ibrutinib": meta["ibrutinib"],
                "treatment": meta["treatment"],
            }
        )

        for raw_bc, cell_id in zip(paired_barcodes, cell_ids):
            metadata_rows.append(
                {
                    "dataset_id": args.dataset_id,
                    "cell_id": cell_id,
                    "raw_barcode": raw_bc,
                    "sample_key": sample_key,
                    "rna_gsm": meta["rna_gsm"],
                    "protein_gsm": meta["protein_gsm"],
                    "cell_line": "BJAB",
                    "cell_type_label": "B lymphocyte cell line",
                    "species": "human",
                    "time_min": meta["time_min"],
                    "bcr_stim": meta["bcr_stim"],
                    "ibrutinib": meta["ibrutinib"],
                    "treatment": meta["treatment"],
                    "condition_label": meta["title"],
                }
            )

        del rna_df, protein_df

    rna_counts = sp.vstack(rna_blocks, format="csr")
    protein_counts = pd.concat(protein_blocks, axis=0)
    cell_metadata = pd.DataFrame(metadata_rows)
    sample_summary = pd.DataFrame(sample_rows)
    feature_annotation = read_antibody_annotation(raw_dir, list(protein_features))
    phospho_features = feature_annotation.loc[feature_annotation["is_phospho"], "feature"].tolist()
    phospho_counts = protein_counts.loc[:, phospho_features]

    write_mtx_gz(rna_counts, out_dir / "rna_counts.mtx.gz")
    pd.Series(genes).to_csv(out_dir / "genes.tsv", sep="\t", index=False, header=False)
    cell_metadata["cell_id"].to_csv(out_dir / "barcodes.tsv", sep="\t", index=False, header=False)
    protein_counts.to_csv(out_dir / "protein_counts.tsv.gz", sep="\t", compression="gzip", index_label="cell_id")
    phospho_counts.to_csv(out_dir / "phospho_counts.tsv.gz", sep="\t", compression="gzip", index_label="cell_id")
    cell_metadata.to_csv(out_dir / "cell_metadata.tsv", sep="\t", index=False)
    feature_annotation.to_csv(out_dir / "protein_feature_annotation.tsv", sep="\t", index=False)
    sample_summary.to_csv(out_dir / "sample_summary.tsv", sep="\t", index=False)

    feature_summary = []
    for feature in protein_counts.columns:
        values = protein_counts[feature].to_numpy(dtype=np.float32)
        anno = feature_annotation.loc[feature_annotation["feature"] == feature].iloc[0].to_dict()
        feature_summary.append(
            {
                "feature": feature,
                "is_phospho": anno["is_phospho"],
                "feature_class": anno["feature_class"],
                "canonical_target": anno["canonical_target"],
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "nonzero_rate": float(np.mean(values > 0)),
            }
        )
    feature_summary = pd.DataFrame(feature_summary)
    feature_summary.to_csv(table_dir / "qurie_feature_summary.tsv", sep="\t", index=False)
    sample_summary.to_csv(table_dir / "qurie_sample_summary.tsv", sep="\t", index=False)

    manifest = {
        "dataset_id": args.dataset_id,
        "n_cells": int(rna_counts.shape[0]),
        "n_genes": int(rna_counts.shape[1]),
        "n_protein_features": int(protein_counts.shape[1]),
        "n_phospho_features": int(len(phospho_features)),
        "phospho_features": phospho_features,
        "output_dir": str(out_dir),
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (table_dir / "qurie_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
