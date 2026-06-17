from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="genbio-ai/scFoundation")
    parser.add_argument("--filename", default="models.ckpt")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "download_hf_hub.log"

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"start repo={args.repo_id} filename={args.filename} output_dir={outdir}")
    started = time.time()
    try:
        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=str(outdir),
            force_download=False,
            local_files_only=False,
        )
    except Exception as exc:
        log(f"failed type={type(exc).__name__} message={exc}")
        raise

    elapsed = time.time() - started
    size = Path(path).stat().st_size
    mib = size / 1024 / 1024
    speed = mib / max(elapsed, 1e-6)
    log(f"done path={path}")
    log(f"size_bytes={size} size_mib={mib:.2f} elapsed_sec={elapsed:.2f} avg_mib_s={speed:.2f}")


if __name__ == "__main__":
    main()
