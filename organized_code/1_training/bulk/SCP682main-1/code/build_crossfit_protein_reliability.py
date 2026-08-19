"""Build a development protein-reliability matrix from training OOF metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--oof-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    prediction = pd.read_parquet(args.prediction)
    prediction.index = prediction.index.astype(str)
    prediction.columns = prediction.columns.astype(str)
    metrics = pd.read_csv(args.oof_metrics, sep="\t")
    required = {"protein", "n_samples_used", "spearman"}
    if not required.issubset(metrics.columns):
        raise ValueError(f"OOF metric table lacks columns: {sorted(required - set(metrics.columns))}")
    metrics["protein"] = metrics["protein"].astype(str)
    if metrics["protein"].duplicated().any():
        raise ValueError("OOF metric table contains duplicate proteins")
    score = metrics.set_index("protein")["spearman"].reindex(prediction.columns)
    if score.isna().any():
        missing = score[score.isna()].index[:10].tolist()
        raise ValueError(f"OOF reliability is missing prediction proteins: {missing}")
    reliability_vector = np.clip(score.to_numpy(np.float32), 0.0, 1.0)
    reliability = pd.DataFrame(
        np.broadcast_to(reliability_vector, prediction.shape).copy(),
        index=prediction.index,
        columns=prediction.columns,
    )
    # The guarded development-row reader filters on pandas' unnamed-index
    # parquet field and therefore requires this exact storage contract.
    reliability.index.name = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    reliability.to_parquet(args.output)
    audit = {
        "status": "complete",
        "definition": "clip(training_oof_spearman, 0, 1)",
        "shape": list(reliability.shape),
        "minimum": float(reliability_vector.min()),
        "median": float(np.median(reliability_vector)),
        "maximum": float(reliability_vector.max()),
        "zero_count": int((reliability_vector == 0).sum()),
        "prediction_path": str(args.prediction),
        "prediction_sha256": sha256(args.prediction),
        "oof_metrics_path": str(args.oof_metrics),
        "oof_metrics_sha256": sha256(args.oof_metrics),
        "output_path": str(args.output),
        "output_sha256": sha256(args.output),
        "validation_or_sealed_protein_labels_used": False,
        "phosphosite_labels_used": False,
    }
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
