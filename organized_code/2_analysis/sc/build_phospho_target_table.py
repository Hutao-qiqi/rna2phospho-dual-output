import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "modeling"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phospho_model_common import (
    MAIN_TARGETS,
    ensure_dir,
    infer_evaluation_tier,
    load_phospho_vector,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    ap.add_argument("--coverage-dir", default=r"02_results\single_cell\20260511_data_processing_inventory\tables")
    ap.add_argument("--min-observed", type=int, default=20)
    ap.add_argument("--variance-quantile", type=float, default=0.10)
    args = ap.parse_args()

    root = Path(args.root)
    coverage = root / args.coverage_dir / "phospho_feature_long.tsv"
    rows = pd.read_csv(coverage, sep="\t")
    if "feature_id" not in rows.columns and "feature_name" in rows.columns:
        rows = rows.rename(columns={"feature_name": "feature_id"})

    stats = []
    for _, row in rows.iterrows():
        dataset_id = str(row["dataset_id"])
        feature_id = str(row["feature_id"])
        cells, values = load_phospho_vector(root, dataset_id, feature_id)
        ok = np.isfinite(values)
        observed = int(ok.sum())
        variance = float(np.nanvar(values[ok])) if observed else 0.0
        stats.append(
            {
                "dataset_id": dataset_id,
                "feature_id": feature_id,
                "n_observed_feature": observed,
                "variance_feature": variance,
                "n_nonzero_feature": int((np.nan_to_num(values, nan=0.0) != 0).sum()),
                "n_cells_feature": len(cells),
            }
        )
    stats_df = pd.DataFrame(stats)
    rows = rows.merge(stats_df, on=["dataset_id", "feature_id"], how="left")

    target_stats = (
        rows.groupby("target_id", dropna=False)
        .agg(
            n_observed=("n_observed_feature", "sum"),
            n_nonzero=("n_nonzero_feature", "sum"),
            variance=("variance_feature", "max"),
            n_datasets=("dataset_id", "nunique"),
        )
        .reset_index()
    )
    positive_var = target_stats.loc[target_stats["variance"] > 0, "variance"]
    var_threshold = float(positive_var.quantile(args.variance_quantile)) if len(positive_var) else 0.0
    target_stats["include_in_loss"] = (
        (target_stats["n_observed"] >= args.min_observed)
        & ((target_stats["variance"] >= var_threshold) | target_stats["target_id"].isin(MAIN_TARGETS))
    )

    def priority(target_id: str) -> tuple[int, str]:
        if target_id == "RPS6_pSitePending":
            return (0, target_id)
        if target_id == "STAT3_Y705":
            return (1, target_id)
        if target_id == "MAPK14_pSitePending":
            return (2, target_id)
        return (10, target_id)

    target_order = sorted(target_stats["target_id"].astype(str).tolist(), key=priority)
    target_index = {target: i for i, target in enumerate(target_order)}

    rows = rows.merge(target_stats, on="target_id", how="left", suffixes=("", "_target"))
    rows["target_index"] = rows["target_id"].map(target_index)
    rows["site_id"] = rows.apply(
        lambda r: r["target_id"]
        if str(r.get("residue", "")) not in {"", "unknown", "pending", "pending_antibody_clone", "none"}
        and "pending" not in str(r.get("residue", ""))
        else "",
        axis=1,
    )
    rows["evaluation_tier"] = rows.apply(
        lambda r: infer_evaluation_tier(
            str(r["target_id"]),
            str(r.get("protein_symbol", "")),
            str(r.get("residue", "")),
            str(r["dataset_id"]),
        ),
        axis=1,
    )
    rows["feature_id"] = rows["feature_id"].astype(str)

    keep_cols = [
        "dataset_id",
        "feature_id",
        "target_id",
        "site_id",
        "target_index",
        "protein_symbol",
        "residue",
        "canonical_label",
        "evaluation_tier",
        "include_in_loss",
        "n_observed",
        "variance",
        "n_nonzero",
        "n_datasets",
        "n_observed_feature",
        "variance_feature",
        "n_nonzero_feature",
        "n_cells_feature",
    ]
    target_table = rows[keep_cols].sort_values(["target_index", "dataset_id", "feature_id"])
    target_index_table = (
        target_table.drop_duplicates("target_id")
        .sort_values("target_index")
        [["target_index", "target_id", "site_id", "protein_symbol", "residue", "evaluation_tier", "include_in_loss", "n_observed", "variance", "n_datasets"]]
    )

    metadata_dir = ensure_dir(root / "01_data" / "shared" / "metadata")
    out_dir = ensure_dir(root / "02_results" / "single_cell" / "20260511_phospho_model_setup" / "tables")
    for base in [metadata_dir, out_dir]:
        target_table.to_csv(base / "phospho_target_table.tsv", sep="\t", index=False)
        target_index_table.to_csv(base / "phospho_target_index.tsv", sep="\t", index=False)

    print(f"target_table={metadata_dir / 'phospho_target_table.tsv'}")
    print(f"n_features={target_table.shape[0]} n_targets={target_index_table.shape[0]} variance_threshold={var_threshold:.6g}")
    print(target_index_table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
