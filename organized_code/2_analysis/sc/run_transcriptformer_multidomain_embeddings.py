import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import pandas as pd


def has_embedding(path: Path, obsm_key: str) -> bool:
    if not path.exists():
        return False
    try:
        obj = ad.read_h5ad(path, backed="r")
        ok = obsm_key in obj.obsm
        obj.file.close()
        return bool(ok)
    except Exception:
        return False


def repair_obs_names(input_h5ad: Path, output_h5ad: Path, obsm_key: str) -> None:
    src = ad.read_h5ad(input_h5ad, backed="r")
    obs_names = src.obs_names.astype(str).to_list()
    obs = src.obs.copy()
    src.file.close()

    out = ad.read_h5ad(output_h5ad)
    if obsm_key not in out.obsm:
        raise RuntimeError(f"{output_h5ad} missing obsm[{obsm_key}]")
    if out.n_obs != len(obs_names):
        raise RuntimeError(f"cell count mismatch for {output_h5ad}: {out.n_obs} vs {len(obs_names)}")
    out.obs_names = obs_names
    for col in obs.columns:
        if col not in out.obs.columns:
            out.obs[col] = obs[col].to_numpy()
    out.write_h5ad(output_h5ad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    parser.add_argument("--python", default=r"D:\data\lsy\envs\scgpt\python.exe")
    parser.add_argument("--transcriptformer-src", default=r"D:\data\lsy\repos\transcriptformer\src")
    parser.add_argument("--checkpoint-path", default=r"D:\data\lsy\models\transcriptformer\tf_sapiens")
    parser.add_argument(
        "--h5ad-dir",
        default=r"01_data\single_cell\intermediate\foundation_model_h5ad_inputs_v1",
    )
    parser.add_argument(
        "--output-dir",
        default=r"01_data\single_cell\intermediate\foundation_model_embeddings\transcriptformer_tf_sapiens_multidomain_v1",
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    h5ad_dir = root / args.h5ad_dir
    out_root = root / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(h5ad_dir / "foundation_h5ad_manifest.tsv", sep="\t")
    wanted = set(args.datasets) if args.datasets else None
    rows = []

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(args.transcriptformer_src)) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env["OMP_NUM_THREADS"] = str(args.cpu_threads)
    env["MKL_NUM_THREADS"] = str(args.cpu_threads)
    env["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads)

    for row in manifest.to_dict("records"):
        dataset_id = str(row["dataset_id"])
        if wanted is not None and dataset_id not in wanted:
            continue
        h5ad_path = Path(str(row["h5ad"]))
        out_dir = out_root / dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_h5ad = out_dir / f"{h5ad_path.stem}_transcriptformer_tf_sapiens_adata.h5ad"
        log_path = out_dir / "transcriptformer_inference.log"
        if args.skip_existing and has_embedding(out_h5ad, "embeddings"):
            rows.append({
                "dataset_id": dataset_id,
                "status": "skipped_existing",
                "output_h5ad": str(out_h5ad),
            })
            pd.DataFrame(rows).to_csv(out_root / "transcriptformer_embedding_manifest.tsv", sep="\t", index=False)
            continue

        cmd = [
            args.python,
            "-c",
            "from transcriptformer.cli import main; main()",
            "inference",
            "--checkpoint-path",
            str(Path(args.checkpoint_path)),
            "--data-file",
            str(h5ad_path),
            "--output-path",
            str(out_dir),
            "--output-filename",
            out_h5ad.name,
            "--batch-size",
            str(args.batch_size),
            "--gene-col-name",
            "ensembl_id",
            "--precision",
            "16-mixed",
            "--device",
            "cuda",
            "--num-gpus",
            "1",
            "--use-raw",
            "False",
            "--remove-duplicate-genes",
            "--disable-compile-block-mask",
            "--config-override",
            f"model.data_config.n_data_workers={args.num_workers}",
        ]
        print(f"TranscriptFormer start {dataset_id} cells={row['n_cells']}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, env=env, cwd=str(out_dir), stdout=log, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            rows.append({
                "dataset_id": dataset_id,
                "status": "failed",
                "returncode": int(proc.returncode),
                "output_h5ad": str(out_h5ad),
                "log": str(log_path),
            })
            pd.DataFrame(rows).to_csv(out_root / "transcriptformer_embedding_manifest.tsv", sep="\t", index=False)
            raise RuntimeError(f"TranscriptFormer failed for {dataset_id}; see {log_path}")

        repair_obs_names(h5ad_path, out_h5ad, "embeddings")
        rows.append({
            "dataset_id": dataset_id,
            "status": "completed",
            "n_cells": int(row["n_cells"]),
            "output_h5ad": str(out_h5ad),
            "log": str(log_path),
        })
        pd.DataFrame(rows).to_csv(out_root / "transcriptformer_embedding_manifest.tsv", sep="\t", index=False)
        print(f"TranscriptFormer done {dataset_id}", flush=True)

    (out_root / "transcriptformer_embedding_manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
