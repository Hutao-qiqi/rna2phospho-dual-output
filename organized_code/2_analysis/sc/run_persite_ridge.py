import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phospho_model_common import HOLDOUT_DATASETS, TRAIN_DATASETS, load_model_input, score_prediction, write_rows


def fit_predict(x_train, y_train, x_test):
    scaler = StandardScaler()
    xs_train = scaler.fit_transform(x_train).astype(np.float32)
    xs_test = scaler.transform(x_test).astype(np.float32)
    model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model.fit(xs_train, y_train)
    return model.predict(xs_test).astype(np.float32), float(model.alpha_)


def split_indices(meta, idx, seed):
    groups = meta.iloc[idx]["split_group"].astype(str).to_numpy() if "split_group" in meta.columns else None
    if groups is not None and len(set(groups)) >= 5:
        splitter = GroupKFold(n_splits=5)
        return splitter.split(idx, groups=groups)
    return KFold(n_splits=5, shuffle=True, random_state=seed).split(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=r"D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\phospho_model_inputs\masked_multisite_v1")
    ap.add_argument("--output", default=r"D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260511_model0_persite_ridge")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--min-train", type=int, default=50)
    ap.add_argument("--min-test", type=int, default=20)
    ap.add_argument("--method-name", default="scgpt_frozen_persite_ridge")
    args = ap.parse_args()

    data = load_model_input(Path(args.input_dir))
    x = data["x"]
    y = data["y"]
    mask = data["mask"]
    meta = data["meta"]
    targets = data["targets"].drop_duplicates("target_index").sort_values("target_index")
    rows = []

    for dataset_id in TRAIN_DATASETS:
        idx = np.flatnonzero(meta["dataset_id"].to_numpy() == dataset_id)
        if len(idx) < 10:
            continue
        for fold, (tr_rel, te_rel) in enumerate(split_indices(meta, idx, args.seed), start=1):
            train_idx = idx[tr_rel]
            test_idx = idx[te_rel]
            for _, target in targets.iterrows():
                j = int(target["target_index"])
                tr = mask[train_idx, j]
                te = mask[test_idx, j]
                if tr.sum() < args.min_train or te.sum() < args.min_test:
                    continue
                pred, alpha = fit_predict(x[train_idx][tr], y[train_idx][tr, j], x[test_idx][te])
                rows.append(score_prediction(y[test_idx][te, j], pred, {
                    "method": args.method_name,
                    "evaluation": "internal_5fold",
                    "train_dataset": dataset_id,
                    "test_dataset": dataset_id,
                    "fold": fold,
                    "target_id": target["target_id"],
                    "target_index": j,
                    "alpha": alpha,
                }))

    train_idx_all = np.flatnonzero(meta["dataset_id"].isin(TRAIN_DATASETS).to_numpy())
    for test_dataset in HOLDOUT_DATASETS:
        test_idx_all = np.flatnonzero(meta["dataset_id"].to_numpy() == test_dataset)
        if len(test_idx_all) == 0:
            continue
        for _, target in targets.iterrows():
            j = int(target["target_index"])
            tr = mask[train_idx_all, j]
            te = mask[test_idx_all, j]
            if tr.sum() < args.min_train or te.sum() < args.min_test:
                continue
            pred, alpha = fit_predict(x[train_idx_all][tr], y[train_idx_all][tr, j], x[test_idx_all][te])
            rows.append(score_prediction(y[test_idx_all][te, j], pred, {
                "method": args.method_name,
                "evaluation": "zero_shot_external",
                "train_dataset": ";".join(TRAIN_DATASETS),
                "test_dataset": test_dataset,
                "fold": "all",
                "target_id": target["target_id"],
                "target_index": j,
                "alpha": alpha,
            }))

    out = Path(args.output)
    write_rows(out / "tables" / "persite_ridge_performance.tsv", rows)
    print(f"rows={len(rows)} output={out}")


if __name__ == "__main__":
    main()
