"""Deterministic smoke test for released ChemState-MODZ checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from chemstate_modz_model import ChemStateMODZ


def test_checkpoint(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = ChemStateMODZ(
        n_sites=config["n_sites"],
        n_cells=config["n_cells"],
        fingerprint_dim=config["fingerprint_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    rng = np.random.default_rng(682)
    inputs = {
        "baseline": torch.as_tensor(rng.normal(size=(4, config["n_sites"])), dtype=torch.float32),
        "input_mask": torch.ones((4, config["n_sites"]), dtype=torch.bool),
        "fingerprint": torch.as_tensor(rng.integers(0, 2, size=(4, config["fingerprint_dim"])), dtype=torch.float32),
        "cell_ids": torch.arange(4, dtype=torch.long) % config["n_cells"],
        "condition": torch.as_tensor(np.column_stack([np.full(4, -6.0), np.full(4, np.log1p(24.0))]), dtype=torch.float32),
    }
    with torch.no_grad():
        first = model(inputs)
        second = model(inputs)
    if first.shape != (4, config["n_sites"]):
        raise AssertionError(f"unexpected output shape for {path}")
    if not torch.isfinite(first).all() or not torch.equal(first, second):
        raise AssertionError(f"determinism test failed for {path}")
    print(f"PASS {path.name} {tuple(first.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    for checkpoint in args.checkpoints:
        test_checkpoint(checkpoint)


if __name__ == "__main__":
    main()
