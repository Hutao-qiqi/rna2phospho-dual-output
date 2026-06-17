import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phospho_model_common import load_model_input, score_prediction, write_rows


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def fit_predict(x_train, y_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model.fit(x_train, y_train)
    return model.predict(x_test).astype(np.float32), float(model.alpha_)


def finite_median(values):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if len(vals) else math.nan


def finite_mean(values):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if len(vals) else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-datasets", default="iccite_seq_tcell_2025,qurie_seq_bjab_2021")
    parser.add_argument(
        "--test-datasets",
        default="gse300551_iccite_plex_kinase_2025,signal_seq_gse256403_hela_2024,signal_seq_gse256404_pdo_caf_2024,phospho_seq_blair_2025_phospho_multi,vivo_seq_th17_2025",
    )
    parser.add_argument("--min-train", type=int, default=50)
    parser.add_argument("--min-test", type=int, default=20)
    parser.add_argument("--method-name", default="scfoundation_cap12000_persite_ridge_iccite_qurie")
    args = parser.parse_args()

    train_datasets = parse_csv(args.train_datasets)
    test_datasets = parse_csv(args.test_datasets)
    data = load_model_input(Path(args.input_dir))
    x = data["x"]
    y = data["y"]
    mask = data["mask"]
    meta = data["meta"]
    targets = data["targets"].drop_duplicates("target_index").sort_values("target_index")
    rows = []
    train_idx = np.flatnonzero(meta["dataset_id"].astype(str).isin(train_datasets).to_numpy())

    for test_dataset in test_datasets:
        test_idx = np.flatnonzero(meta["dataset_id"].astype(str).to_numpy() == test_dataset)
        if len(test_idx) == 0:
            continue
        print(f"evaluate {test_dataset}: n={len(test_idx)}", flush=True)
        for _, target in targets.iterrows():
            j = int(target["target_index"])
            if j >= y.shape[1]:
                continue
            tr = mask[train_idx, j]
            te = mask[test_idx, j]
            if int(tr.sum()) < args.min_train or int(te.sum()) < args.min_test:
                continue
            pred, alpha = fit_predict(x[train_idx][tr], y[train_idx][tr, j], x[test_idx][te])
            rows.append(score_prediction(y[test_idx][te, j], pred, {
                "method": args.method_name,
                "evaluation": "zero_shot_external",
                "train_dataset": ";".join(train_datasets),
                "test_dataset": test_dataset,
                "target_id": str(target["target_id"]),
                "target_index": j,
                "protein_symbol": str(target.get("protein_symbol", "")),
                "residue": str(target.get("residue", "")),
                "evaluation_tier": str(target.get("evaluation_tier", "")),
                "alpha": alpha,
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
            }))

    out = Path(args.output)
    write_rows(out / "tables" / "persite_ridge_performance.tsv", rows)
    perf = pd.DataFrame(rows)
    summaries = []
    if not perf.empty:
        for dataset_id, sub in perf.groupby("test_dataset"):
            vals = pd.to_numeric(sub["spearman"], errors="coerce")
            summaries.append({
                "test_dataset": dataset_id,
                "n_targets": int(vals.notna().sum()),
                "median_spearman": finite_median(vals),
                "mean_spearman": finite_mean(vals),
                "best_target": str(sub.loc[vals.idxmax(), "target_id"]) if vals.notna().any() else "",
                "best_spearman": float(vals.max()) if vals.notna().any() else math.nan,
                "worst_target": str(sub.loc[vals.idxmin(), "target_id"]) if vals.notna().any() else "",
                "worst_spearman": float(vals.min()) if vals.notna().any() else math.nan,
            })
    write_rows(out / "tables" / "external_summary_by_dataset.tsv", summaries)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "reports" / "run_manifest.json").write_text(json.dumps({
        "input_dir": str(args.input_dir),
        "output": str(args.output),
        "train_datasets": train_datasets,
        "test_datasets": test_datasets,
        "min_train": int(args.min_train),
        "min_test": int(args.min_test),
        "n_rows": int(len(rows)),
    }, indent=2), encoding="utf-8")
    print(f"rows={len(rows)} output={out}", flush=True)


if __name__ == "__main__":
    main()
