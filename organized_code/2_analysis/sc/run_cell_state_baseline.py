import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import OneHotEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phospho_model_common import TRAIN_DATASETS, ensure_dir, load_model_input, score_prediction, write_rows


def make_state_features(meta: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
    cols = [c for c in ["dataset_id", "cell_type_label", "state_label", "time_min", "ibrutinib"] if c in meta.columns]
    train = meta.iloc[train_idx][cols].astype(str)
    test = meta.iloc[test_idx][cols].astype(str)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_train = enc.fit_transform(train).astype(np.float32)
    x_test = enc.transform(test).astype(np.float32)
    return x_train, x_test


def split_indices(meta: pd.DataFrame, idx: np.ndarray, seed: int):
    groups = meta.iloc[idx]["split_group"].astype(str).to_numpy() if "split_group" in meta.columns else None
    if groups is not None and len(set(groups)) >= 5:
        splitter = GroupKFold(n_splits=5)
        return splitter.split(idx, groups=groups)
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    return splitter.split(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=r"D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\phospho_model_inputs\masked_multisite_v1")
    ap.add_argument("--output", default=r"D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260511_model_minus1_cell_state_baseline")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--min-train", type=int, default=50)
    ap.add_argument("--min-test", type=int, default=20)
    args = ap.parse_args()

    data = load_model_input(Path(args.input_dir))
    meta = data["meta"].copy()
    y = data["y"]
    mask = data["mask"]
    targets = data["targets"].drop_duplicates("target_index").sort_values("target_index")
    out = Path(args.output)
    rows = []

    for dataset_id in TRAIN_DATASETS:
        idx = np.flatnonzero(meta["dataset_id"].to_numpy() == dataset_id)
        if len(idx) < 10:
            continue
        for fold, (tr_rel, te_rel) in enumerate(split_indices(meta, idx, args.seed), start=1):
            train_idx = idx[tr_rel]
            test_idx = idx[te_rel]
            x_train, x_test = make_state_features(meta, train_idx, test_idx)
            for _, target in targets.iterrows():
                j = int(target["target_index"])
                tr = mask[train_idx, j]
                te = mask[test_idx, j]
                if tr.sum() < args.min_train or te.sum() < args.min_test:
                    continue
                model = RidgeCV(alphas=np.logspace(-3, 3, 13))
                model.fit(x_train[tr], y[train_idx][tr, j])
                pred = model.predict(x_test[te])
                rows.append(score_prediction(y[test_idx][te, j], pred, {
                    "method": "cell_state_onehot_ridge",
                    "evaluation": "internal_5fold",
                    "dataset_id": dataset_id,
                    "fold": fold,
                    "target_id": target["target_id"],
                    "target_index": j,
                    "alpha": float(model.alpha_),
                }))

    write_rows(out / "tables" / "cell_state_baseline_performance.tsv", rows)
    print(f"rows={len(rows)} output={out}")


if __name__ == "__main__":
    main()
