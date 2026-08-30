"""Inference entry point for SCP682-ChemState-MODZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from chemstate_modz_model import ChemStateMODZ


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-npy", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    config = {name: config[name] for name in ["n_sites", "n_cells", "fingerprint_dim"]}
    model = ChemStateMODZ(**config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    source = np.load(args.input_npz)
    required = ["baseline", "input_mask", "fingerprint", "cell_ids", "condition"]
    missing = [name for name in required if name not in source]
    if missing:
        raise ValueError(f"Missing input arrays: {missing}")
    batch = len(source["baseline"])
    if source["baseline"].shape != (batch, config["n_sites"]):
        raise ValueError("baseline shape does not match checkpoint")
    if source["fingerprint"].shape != (batch, config["fingerprint_dim"]):
        raise ValueError("fingerprint shape does not match checkpoint")
    if np.min(source["cell_ids"]) < 0 or np.max(source["cell_ids"]) >= config["n_cells"]:
        raise ValueError("cell_ids contain values outside the checkpoint vocabulary")

    inputs = {
        "baseline": torch.as_tensor(source["baseline"], dtype=torch.float32, device=device),
        "input_mask": torch.as_tensor(source["input_mask"], dtype=torch.bool, device=device),
        "fingerprint": torch.as_tensor(source["fingerprint"], dtype=torch.float32, device=device),
        "cell_ids": torch.as_tensor(source["cell_ids"], dtype=torch.long, device=device),
        "condition": torch.as_tensor(source["condition"], dtype=torch.float32, device=device),
    }
    with torch.no_grad():
        prediction = model(inputs).cpu().numpy()
    if prediction.shape != (batch, config["n_sites"]) or not np.isfinite(prediction).all():
        raise RuntimeError("invalid prediction")
    output = Path(args.output_npy)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, prediction)
    print(f"saved {prediction.shape} to {output}")


if __name__ == "__main__":
    main()
