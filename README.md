# RNA2Phospho dual-output model

This repository contains the deployable code contract for a bulk RNA-to-phosphorylation model.

The model takes a bulk RNA expression matrix as input and produces two separate phosphorylation outputs:

1. CPTAC/PDC mass-spectrometry phosphosite predictions.
2. TCGA/TCPA phospho-RPPA antibody predictions.

These outputs are deliberately kept separate. A mass-spectrometry phosphosite and an RPPA antibody readout are different measurement layers and should not be merged into a single target table.

## Current sealed model

Use `rna2phospho_dual_output_router_v2`.

The earlier single-checkpoint `rna2phospho_dual_output_final_v1` is superseded. It was technically runnable, but its CPTAC phosphosite validation median Spearman was only `0.301`, because it collapsed two mature specialist models into one simplified joint model.

Correct deployed contract:

- CPTAC/PDC mass-spectrometry phosphosite output: use the strongest per-target correlation-weighted CPTAC ensemble.
- TCGA/TCPA phospho-RPPA antibody output: use the RNA + VAE z + direct RNA residual FiLM model.

Correct benchmark values:

| output head | model | median Spearman | stronger-use metric |
|---|---|---:|---:|
| CPTAC/PDC phosphosite | per-target correlation-weighted ensemble | 0.431 | 4506 sites >= 0.5 |
| TCPA phospho-RPPA antibody | RNA + VAE z + direct RNA residual FiLM | 0.636 | 70 phospho antibodies >= 0.5 |

Server directory:

`/data/lsy/Infinite_Stream/02_results/model_validation/20260429_rna2phospho_dual_output_router_v2`

## Superseded interface prototype

Model name:

`rna2phospho_dual_output_final_v1`

Training output directory on the compute server:

`/data/lsy/Infinite_Stream/02_results/model_validation/20260429_rna2phospho_dual_output_final_v1`

Training script:

`rna2phospho/train_rna2phospho_dual_output_final_20260429.py`

Inference script:

`rna2phospho/predict_rna2phospho_dual_output.py`

This checkpoint is retained only as a runnable interface prototype. Do not use it as the main CPTAC phosphosite model.

Sealed model summary:

| field | value |
|---|---:|
| training samples | 7465 |
| CPTAC/PDC samples | 1431 |
| TCGA/TCPA samples | 6034 |
| input genes | 33233 |
| CPTAC phosphosite outputs | 16049 |
| TCPA phospho-antibody outputs | 76 |
| best epoch | 20 |
| CPTAC phosphosite validation median Spearman | 0.301 |
| TCPA phospho-antibody validation median Spearman | 0.648 |

Checkpoint SHA256:

`e30d91a80cd39f57ed463a9a1866a7ea0c31efd42d7f156863f6ae5cd01b6ebf`

## Training data contract

The final model uses two supervised label sources:

- CPTAC/PDC phosphoproteomics: RNA paired with mass-spectrometry phosphosite labels.
- TCGA/TCPA RPPA: RNA paired with phospho-antibody labels.

The shared RNA encoder is trained with missing-label masks. CPTAC/PDC samples update the phosphosite head; TCGA/TCPA samples update the phospho-RPPA head.

## Input format

The inference input should be a matrix with samples as rows and gene symbols as columns. Accepted formats:

- `.tsv`
- `.csv`
- `.parquet`

If a `sample_id` column is present, it is used as the row index.

## Output files

The inference script writes:

- `predicted_cptac_phosphosites.tsv`
- `predicted_tcpa_phospho_antibodies.tsv`

## Minimal inference command

```bash
python rna2phospho/predict_rna2phospho_dual_output.py \
  --checkpoint /path/to/rna2phospho_dual_output_final_v1.pt \
  --input-rna /path/to/bulk_rna_matrix.tsv \
  --out-dir outputs/my_cohort \
  --cancer UNK_CANCER
```

Use a specific cancer label when known. Unknown or unsupported cancer labels are mapped to `UNK_CANCER`.
