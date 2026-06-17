import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error


DATASET_IDS = {
    "iccite": "iccite_seq_tcell_2025",
    "qurie": "qurie_seq_bjab_2021",
    "blair": "phospho_seq_blair_2025_phospho_multi",
    "vivo": "vivo_seq_th17_2025",
}

TRAIN_DATASETS = ["iccite_seq_tcell_2025", "qurie_seq_bjab_2021"]
HOLDOUT_DATASETS = ["phospho_seq_blair_2025_phospho_multi", "vivo_seq_th17_2025"]

MAIN_TARGETS = {"RPS6_pSitePending", "STAT3_Y705", "MAPK14_pSitePending"}
STRICT_RESIDUE_TARGETS = {
    "STAT3_Y705",
    "LAT_Y226",
    "CREB1_S133_ATF1_S63",
    "LCP2_S376",
    "MAP2K1_S298",
    "STAT5A_STAT5B_Y694_Y699",
    "MAPK1_MAPK3_T202_Y204",
    "FOS_S32",
    "RELA_S536",
}
DYNAMIC_PROTEINS = {
    "SYK", "BTK", "PLCG2", "MAPK1", "MAPK3", "MAPK14", "JNK_MAPK8_9",
    "AKT1", "AKT2", "AKT3", "RPS6", "RELA", "JUN", "SRC", "STAT1",
    "STAT3", "STAT5A", "STAT5B",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_lines(path: Path) -> list[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n").split("\t")[0] for line in handle if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_mtx(path: Path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return scipy.io.mmread(handle).tocsr().astype(np.float32)
    return scipy.io.mmread(str(path)).tocsr().astype(np.float32)


def read_dense_tsv(path: Path) -> pd.DataFrame:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return pd.read_csv(handle, sep="\t", index_col=0)


def safe_spearman(y, pred) -> float:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() < 3 or np.std(y[ok]) == 0 or np.std(pred[ok]) == 0:
        return float("nan")
    return float(spearmanr(y[ok], pred[ok]).statistic)


def score_prediction(y, pred, extra: dict) -> dict:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(pred)
    row = {
        "n": int(ok.sum()),
        "spearman": safe_spearman(y, pred),
        "pearson": float(np.corrcoef(y[ok], pred[ok])[0, 1]) if ok.sum() >= 3 and np.std(y[ok]) > 0 and np.std(pred[ok]) > 0 else float("nan"),
        "rmse": float(math.sqrt(mean_squared_error(y[ok], pred[ok]))) if ok.sum() else float("nan"),
    }
    row.update(extra)
    return row


def dataset_dir(root: Path, dataset_id: str) -> Path:
    return root / "01_data" / "single_cell" / "intermediate" / "paired_matrices" / dataset_id


def load_dataset_cells(root: Path, dataset_id: str) -> list[str]:
    base = dataset_dir(root, dataset_id)
    if dataset_id == "iccite_seq_tcell_2025":
        return read_lines(base / "rna_full_counts" / "rna_full_counts_barcodes.tsv")
    return read_lines(base / "barcodes.tsv")


def load_metadata(root: Path, dataset_id: str) -> pd.DataFrame:
    base = dataset_dir(root, dataset_id)
    path = base / "cell_metadata.tsv"
    if path.exists():
        meta = pd.read_csv(path, sep="\t")
    else:
        meta = pd.DataFrame({"cell_id": load_dataset_cells(root, dataset_id)})
    if "cell_id" not in meta.columns:
        meta.insert(0, "cell_id", load_dataset_cells(root, dataset_id))
    meta["dataset_id"] = dataset_id
    if "cell_type_label" not in meta.columns:
        meta["cell_type_label"] = dataset_id
    if "state_label" not in meta.columns:
        meta["state_label"] = meta["cell_type_label"].astype(str)
    if "time_min" not in meta.columns:
        meta["time_min"] = 0
    if "ibrutinib" not in meta.columns:
        meta["ibrutinib"] = False
    if "sample_key" not in meta.columns:
        for col in ["orig.ident", "Donor", "stim", "panel", "culture", "cell_type_label"]:
            if col in meta.columns:
                meta["sample_key"] = meta[col].astype(str)
                break
        if "sample_key" not in meta.columns:
            meta["sample_key"] = dataset_id
    meta["split_group"] = meta["sample_key"].astype(str)
    return meta


def load_phospho_vector(root: Path, dataset_id: str, feature_id: str) -> tuple[list[str], np.ndarray]:
    base = dataset_dir(root, dataset_id)
    if dataset_id == "iccite_seq_tcell_2025":
        bg = base / "iccite_background_corrected_strict"
        matrix = bg / "phospho_counts_control_mean_subtracted.mtx"
        features_path = bg / "phospho_counts_control_mean_subtracted_features.tsv"
        barcodes_path = bg / "phospho_counts_control_mean_subtracted_barcodes.tsv"
        if not matrix.exists():
            matrix = base / "phospho_counts" / "phospho_counts.mtx"
            features_path = base / "phospho_counts" / "phospho_counts_features.tsv"
            barcodes_path = base / "phospho_counts" / "phospho_counts_barcodes.tsv"
        features = read_lines(features_path)
        cells = read_lines(barcodes_path)
        if feature_id not in features:
            raise KeyError(f"{feature_id} not found in {features_path}")
        mat = load_mtx(matrix)
        if mat.shape == (len(features), len(cells)):
            values = np.asarray(mat[features.index(feature_id), :].todense()).ravel().astype(np.float32)
        else:
            values = np.asarray(mat[:, features.index(feature_id)].todense()).ravel().astype(np.float32)
        return cells, values

    if dataset_id == "phospho_seq_blair_2025_phospho_multi":
        table = read_dense_tsv(base / "adt_counts.tsv")
    elif dataset_id == "qurie_seq_bjab_2021":
        table = read_dense_tsv(base / "phospho_counts.tsv.gz")
    elif dataset_id == "vivo_seq_th17_2025":
        table = read_dense_tsv(base / "phospho_counts.tsv")
    else:
        raise ValueError(dataset_id)
    if feature_id not in table.columns:
        raise KeyError(f"{feature_id} not found for {dataset_id}")
    values = pd.to_numeric(table[feature_id], errors="coerce").to_numpy(dtype=np.float32)
    return table.index.astype(str).tolist(), values


def infer_evaluation_tier(target_id: str, protein_symbol: str, residue: str, dataset_id: str) -> str:
    proteins = set(str(protein_symbol).replace("_or_", ";").replace(",", ";").split(";"))
    residue = str(residue)
    if target_id in MAIN_TARGETS:
        return "main"
    if target_id in STRICT_RESIDUE_TARGETS or (residue not in {"", "unknown", "pending", "pending_antibody_clone", "none"} and "pending" not in residue):
        return "strict_site"
    if dataset_id == "qurie_seq_bjab_2021" and (proteins & DYNAMIC_PROTEINS):
        return "dynamic"
    if "pSitePending" in target_id or "pending" in residue or residue == "unknown":
        return "pending_confirmation"
    return "auxiliary"


def load_model_input(input_dir: Path) -> dict:
    data = {
        "x": np.load(input_dir / "embeddings.npy").astype(np.float32),
        "y": np.load(input_dir / "targets.npy").astype(np.float32),
        "mask": np.load(input_dir / "target_mask.npy").astype(bool),
        "meta": pd.read_csv(input_dir / "cell_metadata.tsv", sep="\t"),
        "targets": pd.read_csv(input_dir / "phospho_target_table.tsv", sep="\t"),
    }
    return data


def save_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
