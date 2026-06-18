#!/usr/bin/env python3
"""Input guard and sample median centering for SCP682 deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SCP682_PROJECT_ROOT", "/data/lsy/Infinite_Stream"))
DEFAULT_FEATURE_REFERENCE = ROOT / "data/processed/X_all.symbols.parquet"


def read_matrix(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep="\t")

    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    elif "Sample" in df.columns:
        df = df.set_index("Sample")
    elif "sample" in df.columns:
        df = df.set_index("sample")
    elif df.shape[1] > 1 and not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        df = df.set_index(df.columns[0])

    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def load_reference_features(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.exists():
        return []
    if path.suffix.lower() in {".txt", ".tsv", ".csv"}:
        sep = "\t" if path.suffix.lower() in {".txt", ".tsv"} else ","
        ref = pd.read_csv(path, sep=sep, header=None)
        return [str(x) for x in ref.iloc[:, 0].dropna().tolist()]
    if path.suffix.lower() == ".parquet":
        ref = pd.read_parquet(path)
        col = "gene_symbol" if "gene_symbol" in ref.columns else ref.columns[0]
        return [str(x) for x in ref[col].dropna().tolist()]
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return []
    for key in ["feature_names", "gene_names"]:
        if key in ckpt:
            return [str(x) for x in ckpt[key]]
    return []


def quantile(values: np.ndarray, q: float) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.nanpercentile(finite, q))


def sample_summary(x: pd.DataFrame) -> dict[str, Any]:
    arr = x.to_numpy(dtype=float)
    sample_median = np.nanmedian(arr, axis=1)
    sample_sum = np.nansum(arr, axis=1)
    return {
        "n_samples": int(x.shape[0]),
        "n_genes": int(x.shape[1]),
        "finite_fraction": float(np.isfinite(arr).mean()),
        "missing_fraction": float(np.isnan(arr).mean()),
        "negative_fraction": float((arr[np.isfinite(arr)] < 0).mean()) if np.isfinite(arr).any() else None,
        "min": quantile(arr, 0),
        "p50": quantile(arr, 50),
        "p95": quantile(arr, 95),
        "p99": quantile(arr, 99),
        "max": quantile(arr, 100),
        "sample_median_p10": quantile(sample_median, 10),
        "sample_median_p50": quantile(sample_median, 50),
        "sample_median_p90": quantile(sample_median, 90),
        "sample_sum_p50": quantile(sample_sum, 50),
        "sample_sum_p90": quantile(sample_sum, 90),
    }


def assess_log2tpm(x: pd.DataFrame, reference_features: list[str], min_gene_overlap: float) -> tuple[bool, list[str], dict[str, Any]]:
    stats = sample_summary(x)
    reasons: list[str] = []

    if stats["finite_fraction"] is None or stats["finite_fraction"] < 0.80:
        reasons.append("finite_fraction_below_0.80")
    if stats["negative_fraction"] is not None and stats["negative_fraction"] > 0.001:
        reasons.append("negative_values_present")
    if stats["p99"] is None:
        reasons.append("no_finite_expression_values")
    else:
        if stats["p99"] > 25:
            reasons.append("p99_too_high_for_log2tpm")
        if stats["max"] is not None and stats["max"] > 40:
            reasons.append("max_too_high_for_log2tpm")
    if stats["sample_sum_p50"] is not None and stats["sample_sum_p50"] > 250000:
        reasons.append("sample_sum_raw_tpm_or_fpkm_like")

    overlap_report: dict[str, Any] = {}
    if reference_features:
        input_genes = set(x.columns.astype(str))
        ref_genes = set(reference_features)
        overlap = len(input_genes.intersection(ref_genes))
        overlap_fraction = overlap / max(len(ref_genes), 1)
        overlap_report = {
            "reference_feature_count": int(len(ref_genes)),
            "input_reference_overlap": int(overlap),
            "input_reference_overlap_fraction": float(overlap_fraction),
            "min_required_overlap_fraction": float(min_gene_overlap),
        }
        if overlap_fraction < min_gene_overlap:
            reasons.append("model_gene_overlap_too_low")

    passed = len(reasons) == 0
    report = {**stats, **overlap_report, "passed": bool(passed), "failure_reasons": reasons}
    return passed, reasons, report


def write_checked_input(x: pd.DataFrame, out_dir: Path) -> str:
    out = out_dir / "rna_log2tpm_checked.parquet"
    x.index.name = "sample_id"
    x.to_parquet(out)
    return out.name


def sample_median_center(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = pred.median(axis=1, skipna=True)
    centered = pred.sub(med, axis=0)
    centered.index.name = pred.index.name or "sample_id"
    med_table = pd.DataFrame({"sample_id": pred.index.astype(str), "prediction_sample_median": med.to_numpy(dtype=float)})
    return centered.astype(np.float32), med_table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-rna", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prediction-phosphosite")
    parser.add_argument("--feature-reference", default=str(DEFAULT_FEATURE_REFERENCE))
    parser.add_argument("--min-gene-overlap", type=float, default=0.60)
    parser.add_argument("--allow-log2tpm-warning", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rna = read_matrix(Path(args.input_rna))
    reference_features = load_reference_features(Path(args.feature_reference) if args.feature_reference else None)
    passed, reasons, input_report = assess_log2tpm(rna, reference_features, args.min_gene_overlap)

    summary: dict[str, Any] = {
        "model_id": "SCP682_S_phi",
        "input_file": str(Path(args.input_rna)),
        "required_input_unit": "log2(TPM+1)",
        "rna_input_action": "check_only_no_sample_median_centering",
        "phosphosite_output_action": "sample_median_centering_if_prediction_file_is_supplied",
        "input_qc": input_report,
        "files": {},
    }

    if not passed and not args.allow_log2tpm_warning:
        (out_dir / "input_qc_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        raise SystemExit(
            "Input RNA did not pass SCP682_S_phi log2TPM guard: "
            + ", ".join(reasons)
            + ". Supply log2(TPM+1) RNA, not raw TPM, FPKM, RSEM, counts, or log2FPKM."
        )

    summary["files"]["checked_rna"] = write_checked_input(rna, out_dir)

    if args.prediction_phosphosite:
        pred = read_matrix(Path(args.prediction_phosphosite))
        centered, med_table = sample_median_center(pred)
        centered_path = out_dir / "SCP682_S_phi_phosphosite_sample_median_centered.parquet"
        med_path = out_dir / "SCP682_S_phi_phosphosite_sample_medians.tsv"
        centered.to_parquet(centered_path)
        med_table.to_csv(med_path, sep="\t", index=False)
        summary["files"]["sample_median_centered_phosphosite"] = centered_path.name
        summary["files"]["phosphosite_sample_medians"] = med_path.name

    (out_dir / "input_qc_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

