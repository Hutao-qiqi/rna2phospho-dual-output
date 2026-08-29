from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_scp682_sc15_direct import DirectModel  # noqa: E402


def selected_matrix(matrix, var_names, selected_genes, mean, std):
    lookup = {str(gene): index for index, gene in enumerate(var_names)}
    present_output = []
    present_input = []
    for output_index, gene in enumerate(selected_genes):
        input_index = lookup.get(str(gene))
        if input_index is not None:
            present_output.append(output_index)
            present_input.append(input_index)

    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
    result = np.zeros((matrix.shape[0], len(selected_genes)), dtype=np.float32)
    if present_input:
        values = (sparse.diags(scale) @ matrix[:, present_input]).toarray()
        values = np.log1p(values)
        output = np.asarray(present_output, dtype=np.int64)
        result[:, output] = (values - mean[output]) / std[output]
    return result


def infer_member(model_path, genes, embedding, device, batch_size):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = DirectModel(
        genes.shape[1], embedding.shape[1], 20, checkpoint.get("variant", "A2")
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(genes), batch_size):
            stop = min(start + batch_size, len(genes))
            prediction, _ = model(
                torch.as_tensor(genes[start:stop], device=device),
                torch.as_tensor(embedding[start:stop], device=device),
            )
            predictions.append(prediction.float().cpu().numpy())
    return np.vstack(predictions).astype(np.float32)


def rank_ensemble(predictions):
    transformed = []
    for prediction in predictions:
        ranked = np.column_stack(
            [rankdata(prediction[:, j], method="average") for j in range(prediction.shape[1])]
        )
        quantile = (ranked - 0.5) / len(prediction)
        transformed.append(norm.ppf(np.clip(quantile, 1e-5, 1 - 1e-5)))
    return np.mean(transformed, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--rna-h5ad", type=Path, required=True)
    parser.add_argument("--scfoundation-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.release_dir / "ensemble_config.json").read_text())
    rna = ad.read_h5ad(args.rna_h5ad)
    embedding = np.asarray(np.load(args.scfoundation_embeddings), dtype=np.float32)
    if len(rna) != len(embedding):
        raise ValueError("RNA cells and scFoundation embedding rows differ")
    device = torch.device(args.device)

    fold_predictions = []
    target_ids = None
    for fold in config["folds"]:
        preprocessing = np.load(args.release_dir / fold["preprocessing"], allow_pickle=False)
        if embedding.shape[1] != int(preprocessing["embedding_dim"]):
            raise ValueError("scFoundation embedding dimension differs from locked model")
        genes = selected_matrix(
            rna.X,
            rna.var_names.astype(str),
            preprocessing["gene_names"].astype(str),
            preprocessing["gene_mean"],
            preprocessing["gene_std"],
        )
        members = [
            infer_member(
                args.release_dir / model,
                genes,
                embedding,
                device,
                args.batch_size,
            )
            for model in fold["members"]
        ]
        if fold["aggregation"] == "rank3":
            standardized = rank_ensemble(members)
        elif fold["aggregation"] == "mean3":
            standardized = np.mean(members, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {fold['aggregation']}")
        native = (
            standardized * preprocessing["target_std"] + preprocessing["target_mean"]
        )
        fold_predictions.append(native.astype(np.float32))
        target_ids = preprocessing["target_ids"].astype(str)

    prediction = np.mean(fold_predictions, axis=0).astype(np.float32)
    np.save(args.output_dir / "scp682_sc15_prediction.npy", prediction)
    pd.DataFrame(prediction, index=rna.obs_names, columns=target_ids).to_csv(
        args.output_dir / "scp682_sc15_prediction.tsv", sep="\t"
    )
    np.savez_compressed(
        args.output_dir / "scp682_sc15_prediction_with_folds.npz",
        prediction=prediction,
        fold_predictions=np.stack(fold_predictions),
        cell_ids=rna.obs_names.astype(str).to_numpy(),
        target_ids=target_ids,
    )


if __name__ == "__main__":
    main()
