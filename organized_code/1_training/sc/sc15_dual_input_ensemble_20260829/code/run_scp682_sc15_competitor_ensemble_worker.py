from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--tasks", required=True)
    args = parser.parse_args()
    script = Path(__file__).resolve().parent / "run_scp682_sc14_raw_rna_translators.py"
    for item in args.tasks.split(","):
        architecture, training_seed, split_seed = item.split(":")
        output = args.output_root / architecture / f"split_{split_seed}" / f"seed_{training_seed}"
        command = [
            sys.executable,
            str(script),
            "--root",
            str(args.root),
            "--contract-dir",
            str(args.contract_dir),
            "--output-dir",
            str(output),
            "--architecture",
            architecture,
            "--seeds",
            training_seed,
            "--split-seed",
            split_seed,
            "--device",
            args.device,
            "--n-genes",
            "4000",
            "--batch-size",
            "512",
            "--epochs",
            "100",
            "--target-mode",
            "site_zscore",
        ]
        result = subprocess.run(command)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
