"""Run one decoder/retrieval configuration on two Windows CUDA workers."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import torch
import torch.multiprocessing as mp


def _worker(rank, world_size, training_args, gradient_buffer, metric_buffer, correlation_buffer, barrier):
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["DECODER_RETRIEVAL_MANUAL_DATA_PARALLEL"] = "1"
    torch.cuda.set_device(rank)
    import manual_dp_runtime

    manual_dp_runtime.configure(
        rank, world_size, gradient_buffer, metric_buffer, barrier,
        correlation_buffer=correlation_buffer,
    )
    sys.argv = ["train_decoder_retrieval.py", *training_args]
    import train_decoder_retrieval

    try:
        result = train_decoder_retrieval.main()
        if result:
            raise SystemExit(result)
    except BaseException:
        try:
            position = training_args.index("--output-dir")
            output = Path(training_args[position + 1])
            log_dir = output / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"worker_rank_{rank}_exception.log").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        finally:
            raise


def main() -> int:
    world_size = 2
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError("two CUDA devices are required")
    # Current four configurations remain below ten million trainable parameters.
    gradient_buffer = torch.zeros((world_size, 12_000_000), dtype=torch.float32)
    gradient_buffer.share_memory_()
    metric_buffer = torch.zeros((world_size, 2), dtype=torch.float64)
    metric_buffer.share_memory_()
    # Six Pearson sufficient statistics plus one Huber point-loss sum.
    correlation_buffer = torch.zeros((world_size, 7, 1000), dtype=torch.float64)
    correlation_buffer.share_memory_()
    context = mp.get_context("spawn")
    barrier = context.Barrier(world_size)
    mp.spawn(
        _worker,
        args=(world_size, sys.argv[1:], gradient_buffer, metric_buffer, correlation_buffer, barrier),
        nprocs=world_size,
        join=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
