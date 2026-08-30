"""Build MODZ replicate consensus and quality metadata for a P100 cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def pair_spearman(left, right, left_mask, right_mask):
    keep = left_mask & right_mask & np.isfinite(left) & np.isfinite(right)
    if keep.sum() < 5:
        return 0.0
    x, y = rankdata(left[keep]), rankdata(right[keep])
    denominator = np.linalg.norm(x - x.mean()) * np.linalg.norm(y - y.mean())
    return float(np.dot(x - x.mean(), y - y.mean()) / denominator) if denominator > 0 else 0.0


def modz(values, mask, rows):
    block, observed = values[rows], mask[rows]
    count = len(rows)
    correlations = np.eye(count, dtype=np.float32)
    pairs = []
    for left in range(count):
        for right in range(left + 1, count):
            value = pair_spearman(block[left], block[right], observed[left], observed[right])
            correlations[left, right] = correlations[right, left] = value
            pairs.append(value)
    positive = np.clip(correlations, 0.0, None)
    np.fill_diagonal(positive, 0.0)
    weights = positive.sum(1)
    if weights.sum() <= 0:
        weights = np.ones(count, dtype=np.float32)
    weights /= weights.sum()
    weighted = np.where(observed, block, 0.0) * weights[:, None]
    denominator = observed.astype(np.float32) * weights[:, None]
    consensus = np.divide(weighted.sum(0), denominator.sum(0), out=np.full(block.shape[1], np.nan), where=denominator.sum(0) > 0)
    cc = float(np.quantile(pairs, 0.75)) if pairs else 0.0
    return consensus.astype(np.float32), weights.astype(np.float32), cc, pairs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate-dir", required=True)
    parser.add_argument("--cohort-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    source, cohort, output = Path(args.replicate_dir), Path(args.cohort_dir), Path(args.output_dir)
    for name in ["arrays", "tables", "reports"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    values = np.load(source / "phospho_raw.npy").astype(np.float32)
    mask = np.load(source / "target_mask.npy").astype(bool)
    metadata = pd.read_csv(source / "cell_metadata.tsv", sep="\t")
    metadata["row_index"] = np.arange(len(metadata))
    comparisons = pd.read_csv(cohort / "tables" / "comparison_table.tsv", sep="\t")
    site_table = pd.read_csv(cohort / "tables" / "site_table.tsv", sep="\t")
    target_indices = site_table.p100_target_index.astype(int).to_numpy()
    condition_rows = {condition: frame.row_index.to_numpy(dtype=int) for condition, frame in metadata.groupby("condition")}

    consensus = {}
    quality_rows, weight_rows, null_norms = [], [], []
    for condition, rows in condition_rows.items():
        profile, weights, cc, pairs = modz(values, mask, rows)
        consensus[condition] = profile
        condition_meta = metadata.loc[rows[0]]
        quality_rows.append({
            "condition": condition, "plate": condition_meta.plate,
            "perturbation_type": condition_meta.perturbation_type,
            "cell_line": condition_meta.cell_type, "n_replicates": len(rows),
            "cc_q75": cc, "cc_median": float(np.median(pairs)) if pairs else 0.0,
            "profile_l2": float(np.linalg.norm(np.nan_to_num(profile))),
        })
        for row, weight in zip(rows, weights):
            weight_rows.append({"condition": condition, "cell_id": metadata.loc[row, "cell_id"], "row_index": row, "modz_weight": weight})
        if str(condition_meta.perturbation_type) == "ctrl_vehicle" and len(rows) >= 3:
            for held_out in range(len(rows)):
                other = np.delete(rows, held_out)
                reference, _, _, _ = modz(values, mask, other)
                keep = mask[rows[held_out]] & np.isfinite(reference)
                difference = values[rows[held_out]] - reference
                null_norms.append(float(np.linalg.norm(difference[keep])))

    baseline_rows, delta_rows, valid_rows, comparison_quality = [], [], [], []
    quality = pd.DataFrame(quality_rows).set_index("condition")
    for row in comparisons.itertuples(index=False):
        control, treated = consensus[row.control_condition], consensus[row.condition]
        valid = np.isfinite(control) & np.isfinite(treated)
        delta = treated - control
        baseline_rows.append(control[target_indices])
        delta_rows.append(delta[target_indices])
        valid_rows.append(valid[target_indices])
        comparison_quality.append({
            "comparison_id": row.comparison_id,
            "condition": row.condition,
            "control_condition": row.control_condition,
            "cc_q75": quality.loc[row.condition, "cc_q75"],
            "signature_l2_full91": float(np.linalg.norm(np.nan_to_num(delta))),
            "signature_l2_45": float(np.linalg.norm(np.nan_to_num(delta[target_indices]))),
            "n_valid_45": int(valid[target_indices].sum()),
        })

    np.save(output / "arrays" / "baseline45_modz.npy", np.asarray(baseline_rows, dtype=np.float32))
    np.save(output / "arrays" / "delta45_modz.npy", np.asarray(delta_rows, dtype=np.float32))
    np.save(output / "arrays" / "valid45_modz.npy", np.asarray(valid_rows, dtype=bool))
    np.save(output / "arrays" / "condition_consensus91.npy", np.row_stack([consensus[value] for value in sorted(consensus)]))
    np.save(output / "arrays" / "dmso_pseudo_null_l2.npy", np.asarray(null_norms, dtype=np.float32))
    pd.DataFrame({"condition": sorted(consensus)}).to_csv(output / "tables" / "condition_order.tsv", sep="\t", index=False)
    pd.DataFrame(quality_rows).to_csv(output / "tables" / "condition_quality.tsv", sep="\t", index=False)
    pd.DataFrame(weight_rows).to_csv(output / "tables" / "replicate_modz_weights.tsv", sep="\t", index=False)
    pd.DataFrame(comparison_quality).to_csv(output / "tables" / "comparison_quality.tsv", sep="\t", index=False)
    comparisons.to_csv(output / "tables" / "comparison_table.tsv", sep="\t", index=False)
    site_table.to_csv(output / "tables" / "site_table.tsv", sep="\t", index=False)
    limma_values = pd.DataFrame(np.where(mask, values, np.nan), columns=[f"analyte_{index}" for index in range(values.shape[1])])
    limma_values.insert(0, "condition", metadata.condition.to_numpy())
    limma_values.to_csv(output / "tables" / "limma_replicate_values91.tsv.gz", sep="\t", index=False, compression="gzip")
    report = {
        "n_comparisons": len(comparisons), "n_conditions": len(consensus),
        "n_dmso_pseudo_comparisons": len(null_norms), "n_sites_model": len(target_indices),
        "median_cc_q75_treatment": float(pd.DataFrame(comparison_quality).cc_q75.median()),
        "median_signature_l2_45": float(pd.DataFrame(comparison_quality).signature_l2_45.median()),
    }
    (output / "reports" / "modz_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
