import json
import tarfile
import tempfile
from pathlib import Path

import h5py
import pandas as pd


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
RAW_DIR = ROOT / r"01_data\single_cell\raw\external_single_cell_phospho_validation_v1\icCITE-plex_GSE300551"
OUT_DIR = ROOT / r"02_results\single_cell\20260518_gse300551_iccite_plex_inspect"
RAW_TAR = RAW_DIR / "GSE300551_RAW.tar"


def sample_from_name(name):
    for sample in ["D9", "E9", "F9", "G9", "H9"]:
        if sample in name:
            return sample
    return ""


def read_h5_barcodes(path):
    with h5py.File(path, "r") as h5:
        barcodes = h5["matrix"]["barcodes"][:]
    return {x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x) for x in barcodes}


def read_h5_features(path):
    with h5py.File(path, "r") as h5:
        names = h5["matrix"]["features"]["name"][:]
    return [x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x) for x in names]


def main():
    rows = []
    tsb_features = {}
    with tarfile.open(RAW_TAR, "r") as tar, tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        members = {m.name: m for m in tar.getmembers() if m.name.endswith(".h5")}
        by_sample = {}
        for name, member in members.items():
            sample = sample_from_name(name)
            if not sample:
                continue
            if name.endswith("_TSB_count_filtered_feature_bc_matrix.h5"):
                by_sample.setdefault(sample, {})["tsb"] = member
            elif name.endswith("_filtered_feature_bc_matrix.h5"):
                by_sample.setdefault(sample, {})["rna"] = member

        for sample in sorted(by_sample):
            pair = by_sample[sample]
            extracted = {}
            for assay in ["rna", "tsb"]:
                member = pair[assay]
                out_path = temp_dir / f"{sample}_{assay}.h5"
                with tar.extractfile(member) as src, out_path.open("wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                extracted[assay] = out_path

            rna_barcodes = read_h5_barcodes(extracted["rna"])
            tsb_barcodes = read_h5_barcodes(extracted["tsb"])
            overlap = rna_barcodes & tsb_barcodes
            rows.append(
                {
                    "sample": sample,
                    "rna_barcodes": len(rna_barcodes),
                    "tsb_barcodes": len(tsb_barcodes),
                    "overlap_barcodes": len(overlap),
                    "tsb_in_rna_fraction": len(overlap) / len(tsb_barcodes) if tsb_barcodes else 0.0,
                }
            )
            tsb_features[sample] = read_h5_features(extracted["tsb"])

    tables = OUT_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    overlap_df = pd.DataFrame(rows)
    overlap_df.to_csv(tables / "gse300551_rna_tsb_barcode_overlap.tsv", sep="\t", index=False)
    pd.DataFrame({"feature": tsb_features[sorted(tsb_features)[0]]}).to_csv(
        tables / "gse300551_tsb_h5_features.tsv", sep="\t", index=False
    )
    summary = {
        "rows": rows,
        "total_rna": int(overlap_df["rna_barcodes"].sum()),
        "total_tsb": int(overlap_df["tsb_barcodes"].sum()),
        "total_overlap": int(overlap_df["overlap_barcodes"].sum()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
