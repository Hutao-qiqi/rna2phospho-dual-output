"""Train a frozen SCP682-ChemState-MODZ model on a complete P100 cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from chemstate_modz_model import ChemStateMODZ


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", required=True)
    parser.add_argument("--modz-dir", required=True)
    parser.add_argument("--include-classes", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=682)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_training_data(cohort_dir: str, modz_dir: str, include_classes: list[str]):
    cohort = Path(cohort_dir)
    modz = Path(modz_dir)
    comparisons = pd.read_csv(cohort / "tables" / "comparison_table.tsv", sep="\t")
    parents = pd.read_csv(cohort / "tables" / "parent_drug_table.tsv", sep="\t")
    selected_parents = set(parents.loc[parents.inhibitor_class.isin(include_classes), "parent_key"])
    comparisons = comparisons.loc[comparisons.parent_key.isin(selected_parents)].reset_index(drop=True)

    modz_comparisons = pd.read_csv(modz / "tables" / "comparison_table.tsv", sep="\t")
    modz_index = {value: index for index, value in enumerate(modz_comparisons.comparison_id)}
    selected = np.asarray([modz_index[value] for value in comparisons.comparison_id], dtype=int)
    baseline_raw = np.load(modz / "arrays" / "baseline45_modz.npy")[selected].astype(np.float32)
    truth_raw = np.load(modz / "arrays" / "delta45_modz.npy")[selected].astype(np.float32)
    target_mask = np.load(modz / "arrays" / "valid45_modz.npy")[selected].astype(bool)
    input_mask = np.isfinite(baseline_raw)
    baseline = np.nan_to_num(baseline_raw, nan=0.0, posinf=0.0, neginf=0.0)
    truth = np.nan_to_num(truth_raw, nan=0.0, posinf=0.0, neginf=0.0)

    parent_index = {value: index for index, value in enumerate(parents.parent_key)}
    parent_fingerprints = np.load(cohort / "arrays" / "parent_fingerprints.npy").astype(np.float32)
    fingerprint = np.row_stack([
        parent_fingerprints[parent_index[value]] for value in comparisons.parent_key
    ])
    cells = sorted(comparisons.cell_line.astype(str).unique())
    cell_index = {value: index for index, value in enumerate(cells)}
    cell_ids = comparisons.cell_line.astype(str).map(cell_index).to_numpy(dtype=np.int64)

    dose_factor = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}
    time_factor = {"h": 1.0, "hr": 1.0, "min": 1.0 / 60.0}
    dose_molar = np.asarray([
        float(row.dose) * dose_factor.get(str(row.dose_unit).strip().lower(), 1e-6)
        for row in comparisons.itertuples(index=False)
    ], dtype=np.float32)
    time_hours = np.asarray([
        float(row.time) * time_factor.get(str(row.time_unit).strip().lower(), 1.0)
        for row in comparisons.itertuples(index=False)
    ], dtype=np.float32)
    condition = np.column_stack([
        np.log10(np.clip(dose_molar, 1e-12, None)),
        np.log1p(np.clip(time_hours, 0.0, None)),
    ]).astype(np.float32)
    drugs = comparisons.parent_key.astype(str).to_numpy()
    drug_vocab = sorted(set(drugs))
    drug_index = {value: index for index, value in enumerate(drug_vocab)}
    drug_ids = np.asarray([drug_index[value] for value in drugs], dtype=np.int64)
    arrays = {
        "baseline": baseline,
        "input_mask": input_mask,
        "truth": truth,
        "target_mask": target_mask,
        "fingerprint": fingerprint,
        "cell_ids": cell_ids,
        "condition": condition,
        "drug_ids": drug_ids,
    }
    return arrays, comparisons, cells, drug_vocab


def q1_loss(prediction, truth, target_mask, drug_ids):
    weight = target_mask.float()
    huber = (F.smooth_l1_loss(prediction, truth, reduction="none") * weight).sum(1)
    huber = huber / weight.sum(1).clamp_min(1.0)
    cosine = 1.0 - F.cosine_similarity(
        prediction * weight, truth * weight, dim=1, eps=1e-8,
    )
    drug_losses = []
    for drug in torch.unique(drug_ids):
        keep = drug_ids == drug
        drug_losses.append((huber[keep] + 0.80 * cosine[keep]).mean())
    return torch.stack(drug_losses).mean()


def main() -> None:
    args = parse_args()
    classes = [value.strip() for value in args.include_classes.split(",") if value.strip()]
    arrays, comparisons, cells, drug_vocab = load_training_data(
        args.cohort_dir, args.modz_dir, classes,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tensors = {
        name: torch.as_tensor(
            value,
            dtype=torch.bool if value.dtype == bool else torch.long if name in {"cell_ids", "drug_ids"} else torch.float32,
            device=device,
        )
        for name, value in arrays.items()
    }
    model_inputs = {name: tensors[name] for name in [
        "baseline", "input_mask", "fingerprint", "cell_ids", "condition",
    ]}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = ChemStateMODZ(
        n_sites=arrays["truth"].shape[1],
        n_cells=len(cells),
        fingerprint_dim=arrays["fingerprint"].shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        prediction = model(model_inputs)
        loss = q1_loss(prediction, tensors["truth"], tensors["target_mask"], tensors["drug_ids"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu())})

    output = Path(args.output_dir)
    (output / "models").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "model_config": {
            "n_sites": int(arrays["truth"].shape[1]),
            "n_cells": len(cells),
            "fingerprint_dim": int(arrays["fingerprint"].shape[1]),
            "dropout": 0.05,
        },
        "cells": cells,
        "parent_drugs": drug_vocab,
        "training": {
            "label": "MODZ_Q1",
            "epochs": args.epochs,
            "seed": args.seed,
            "n_comparisons": len(comparisons),
            "n_parent_drugs": len(drug_vocab),
            "include_classes": classes,
        },
    }
    torch.save(checkpoint, output / "models" / "scp682_chemstate_modz.pt")
    pd.DataFrame(history).to_csv(output / "tables" / "training_history.tsv", sep="\t", index=False)
    comparisons.to_csv(output / "tables" / "training_comparisons.tsv", sep="\t", index=False)
    print(json.dumps(checkpoint["training"], indent=2))


if __name__ == "__main__":
    main()
