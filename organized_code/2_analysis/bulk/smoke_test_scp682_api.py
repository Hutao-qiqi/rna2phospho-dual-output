#!/usr/bin/env python3
"""Run an end-to-end SCP682 web API smoke test."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path("/data/lsy/Infinite_Stream")
DEFAULT_INPUT = ROOT / "tmp/scp682_smoke_bulk_rna.tsv"
JOBS_DIR = ROOT / "02_results/public_bulk_phosphoproteome_atlas/web_user_jobs"


def make_input(path: Path, n_samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_raw = pd.read_parquet(ROOT / "data/processed/X_all.symbols.parquet")
    x_raw = x_raw.drop_duplicates("gene_symbol", keep="first").set_index("gene_symbol")
    sample_cols = list(x_raw.columns[:n_samples])
    x = x_raw[sample_cols].T
    x.index.name = "sample_id"
    x.to_csv(path, sep="\t")


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as handle:
        return json.loads(handle.read().decode("utf-8"))


def post_job(base_url: str, input_path: Path, cptac_label: str, study_id: str, tcpa_project: str, transform: str) -> dict:
    cmd = [
        "curl",
        "-s",
        "-F",
        f"file=@{input_path}",
        "-F",
        f"cptac_cancer_label={cptac_label}",
        "-F",
        f"cptac_study_id={study_id}",
        "-F",
        f"tcpa_project={tcpa_project}",
        "-F",
        f"transform={transform}",
        f"{base_url}/api/predict",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8866")
    parser.add_argument("--input-rna", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--cptac-label", default="PDA")
    parser.add_argument("--study-id", default="PDC000271")
    parser.add_argument("--tcpa-project", default="TCGA-PAAD")
    parser.add_argument("--transform", default="none")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    make_input(args.input_rna, args.n_samples)
    health = get_json(f"{args.base_url}/api/health")
    options = get_json(f"{args.base_url}/api/options")
    contract = get_json(f"{args.base_url}/api/model-contract")
    downloads = get_json(f"{args.base_url}/api/downloads")

    job = post_job(args.base_url, args.input_rna, args.cptac_label, args.study_id, args.tcpa_project, args.transform)
    job_id = job["job_id"]
    deadline = time.time() + args.timeout
    status = {}
    while time.time() < deadline:
        status = get_json(f"{args.base_url}/api/jobs/{job_id}")
        if status.get("status") in {"done", "failed"}:
            break
        time.sleep(5)
    if status.get("status") != "done":
        raise SystemExit(f"job {job_id} did not finish successfully: {status}")

    out_dir = JOBS_DIR / job_id / "outputs"
    expected = {
        "predicted_cptac_pdc_total_protein.parquet": (args.n_samples, 11312),
        "predicted_cptac_pdc_phosphosite.parquet": (args.n_samples, 18902),
        "predicted_cptac_pdc_phosphosite_raw_before_sample_median_centering.parquet": (args.n_samples, 18902),
        "predicted_tcpa_total_antibody.parquet": (args.n_samples, 375),
        "predicted_tcpa_phospho_antibody.parquet": (args.n_samples, 74),
    }
    shapes = {}
    for name, expected_shape in expected.items():
        path = out_dir / name
        if not path.exists():
            raise SystemExit(f"missing output: {path}")
        shape = tuple(pd.read_parquet(path).shape)
        shapes[name] = shape
        if shape != expected_shape:
            raise SystemExit(f"unexpected shape for {name}: {shape}, expected {expected_shape}")

    summary = {
        "job_id": job_id,
        "status": status["status"],
        "health_download_files": f"{health['download_files_present']}/{health['download_files_total']}",
        "n_cptac_contexts": len(options["contexts"]),
        "n_tcpa_projects": len(options["tcpa_projects"]),
        "model_version": f"{contract.get('model_name', 'SCP682')} {contract.get('model_version', contract.get('version', contract.get('model_id', '')))}",
        "n_download_matrices": len(downloads["items"]),
        "output_shapes": shapes,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
