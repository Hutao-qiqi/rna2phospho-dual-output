from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
DATASET = "iccite_seq_tcell_2025"
PAIRED = ROOT / "01_data" / "single_cell" / "intermediate" / "paired_matrices" / DATASET
MODEL_INPUT = ROOT / "01_data" / "single_cell" / "intermediate" / "phospho_model_inputs" / "scfoundation_cap12000_masked_multisite_v1"


def read_lines(path: Path) -> list[str]:
    return [x.rstrip("\n").split("\t")[0] for x in path.open("r", encoding="utf-8", errors="replace") if x.strip()]


def print_series(title: str, series: pd.Series, n: int = 30) -> None:
    print(title)
    if series is None:
        print("NA")
        return
    print(series.value_counts(dropna=False).head(n).to_string())


def mtx_shape(path: Path) -> tuple[int, int, int]:
    mat = scipy.io.mmread(str(path)).tocsr()
    return int(mat.shape[0]), int(mat.shape[1]), int(mat.nnz)


def main() -> None:
    print("PAIRED_DIR", PAIRED)
    meta_pair = pd.read_csv(PAIRED / "cell_metadata.tsv", sep="\t", low_memory=False)
    print("PAIR_META_SHAPE", meta_pair.shape)
    print("PAIR_META_COLUMNS", "\t".join(meta_pair.columns.astype(str)))

    rna_dir = PAIRED / "rna_full_counts"
    rna_cells = read_lines(rna_dir / "rna_full_counts_barcodes.tsv")
    rna_genes = read_lines(rna_dir / "rna_full_counts_features.tsv")
    rna_shape = mtx_shape(rna_dir / "rna_full_counts.mtx")
    print("RNA_CELLS", len(rna_cells))
    print("RNA_GENES", len(rna_genes))
    print("RNA_MTX_SHAPE_NNZ", rna_shape)

    phospho_dir = PAIRED / "phospho_counts"
    phospho_features = read_lines(phospho_dir / "phospho_counts_features.tsv")
    phospho_cells = read_lines(phospho_dir / "phospho_counts_barcodes.tsv")
    phospho_shape = mtx_shape(phospho_dir / "phospho_counts.mtx")
    print("PHOSPHO_CELLS", len(phospho_cells))
    print("PHOSPHO_FEATURES", len(phospho_features))
    print("PHOSPHO_MTX_SHAPE_NNZ", phospho_shape)
    print("PHOSPHO_FEATURE_LIST")
    print("\n".join(phospho_features))

    intra_dir = PAIRED / "intra_counts"
    if intra_dir.exists():
        intra_features = read_lines(intra_dir / "intra_counts_features.tsv")
        intra_cells = read_lines(intra_dir / "intra_counts_barcodes.tsv")
        print("INTRA_CELLS", len(intra_cells))
        print("INTRA_FEATURES", len(intra_features))
        print("INTRA_FEATURE_LIST")
        print("\n".join(intra_features))

    bg_dir = PAIRED / "iccite_background_corrected_strict"
    if bg_dir.exists():
        bg_features = read_lines(bg_dir / "phospho_counts_control_mean_subtracted_features.tsv")
        bg_cells = read_lines(bg_dir / "phospho_counts_control_mean_subtracted_barcodes.tsv")
        bg_shape = mtx_shape(bg_dir / "phospho_counts_control_mean_subtracted.mtx")
        print("BACKGROUND_CORRECTED_CELLS", len(bg_cells))
        print("BACKGROUND_CORRECTED_FEATURES", len(bg_features))
        print("BACKGROUND_CORRECTED_MTX_SHAPE_NNZ", bg_shape)
        print("BACKGROUND_CORRECTED_FEATURE_LIST")
        print("\n".join(bg_features))

    meta = pd.read_csv(MODEL_INPUT / "cell_metadata.tsv", sep="\t", low_memory=False)
    tt = pd.read_csv(MODEL_INPUT / "phospho_target_table.tsv", sep="\t")
    y = np.load(MODEL_INPUT / "targets.npy", mmap_mode="r")
    mask = np.load(MODEL_INPUT / "target_mask.npy", mmap_mode="r")

    sub = meta[meta["dataset_id"].astype(str).eq(DATASET)].copy()
    idx = sub.index.to_numpy(dtype=np.int64)
    print("MODEL_INPUT_ICCITE_CELLS", len(sub))
    print("MODEL_INPUT_META_COLUMNS", "\t".join(sub.columns.astype(str)))
    for col in [
        "cell_type_label",
        "state_label",
        "Donor",
        "PerturbedGene",
        "gRNA",
        "orig.ident",
        "sample_key",
        "split_group",
        "time_min",
        "nCount_RNA",
        "nFeature_RNA",
        "nCount_intra",
        "nFeature_intra",
    ]:
        if col in sub.columns:
            if pd.api.types.is_numeric_dtype(sub[col]):
                vals = pd.to_numeric(sub[col], errors="coerce")
                print(
                    f"NUMERIC_{col}",
                    "n",
                    int(vals.notna().sum()),
                    "mean",
                    float(vals.mean()),
                    "median",
                    float(vals.median()),
                    "min",
                    float(vals.min()),
                    "max",
                    float(vals.max()),
                )
            else:
                print_series(f"COUNT_{col}", sub[col], 30)

    rows = []
    for _, row in tt[tt["dataset_id"].astype(str).eq(DATASET)].sort_values("target_index").iterrows():
        j = int(row["target_index"])
        vals = np.asarray(y[idx, j], dtype=np.float32)
        obs = np.asarray(mask[idx, j], dtype=bool) & np.isfinite(vals)
        x = vals[obs]
        rows.append(
            {
                "target_index": j,
                "target_id": row["target_id"],
                "feature_id": row["feature_id"],
                "evaluation_tier": row["evaluation_tier"],
                "include_in_loss": row["include_in_loss"],
                "n_obs": int(obs.sum()),
                "coverage": float(obs.mean()),
                "mean": float(np.mean(x)) if len(x) else np.nan,
                "sd": float(np.std(x)) if len(x) else np.nan,
                "min": float(np.min(x)) if len(x) else np.nan,
                "max": float(np.max(x)) if len(x) else np.nan,
                "n_nonzero": int(np.sum(x != 0)) if len(x) else 0,
            }
        )
    target_df = pd.DataFrame(rows)
    print("TARGET_COVERAGE_TABLE")
    print(target_df.to_string(index=False))


if __name__ == "__main__":
    main()
