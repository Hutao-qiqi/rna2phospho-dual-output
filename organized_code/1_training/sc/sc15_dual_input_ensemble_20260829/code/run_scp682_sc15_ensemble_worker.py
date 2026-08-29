from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--tasks", required=True)
    args = parser.parse_args()
    script = Path(__file__).resolve().parent / "train_scp682_sc15_direct.py"
    for item in args.tasks.split(","):
        training_seed, split_seed = (int(value) for value in item.split(":"))
        output = args.output_root / f"split_{split_seed}" / f"seed_{training_seed}"
        command = [
            sys.executable,
            str(script),
            "--root",
            str(args.root),
            "--contract-dir",
            str(args.contract_dir),
            "--metadata",
            str(args.metadata),
            "--output-dir",
            str(output),
            "--variant",
            "A2",
            "--seed",
            str(training_seed),
            "--split-seed",
            str(split_seed),
            "--device",
            args.device,
            "--epochs",
            "80",
        ]
        result = subprocess.run(command)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
