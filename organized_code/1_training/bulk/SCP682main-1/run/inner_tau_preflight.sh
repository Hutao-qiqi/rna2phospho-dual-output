#!/usr/bin/env bash
set -eu
root=${SCP682_ROOT:?set SCP682_ROOT to the SCP682-main checkout}
python=${SCP682_PYTHON:?set SCP682_PYTHON to the project python}
input=$root/01_data/bulk/intermediate/random70_reference_rna_protein_logscale_phosphosite_inputs_20260815
cv=$root/01_data/bulk/intermediate/random70_tau_inner_cv_20260819_v2/fold0
code=$(cd "$(dirname "$0")/.." && pwd)/code/train_decoder_retrieval.py
manifest=$root/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/residual_target_manifest.tsv
cophee=$root/01_data/pathway_prior/processed/scp682main1_dynamic_edge_gated_binary_cophee_v1
esm2=$root/01_data/intermediate/esm2_sequence_prior_20260806/esm2_t33_650M_protein_and_site_prior.npz
out=$root/02_results/model_validation/scp682main1_tau_inner_cv_preflight_20260819_v5
compat=$root/00_tools/nvidia_compat_560
cuda_lib_dir=${CUDA_LIB_DIR:-/usr/lib/x86_64-linux-gnu}
mkdir -p "$compat" "$out/logs"
ln -sfn "$cuda_lib_dir/libcuda.so.560.35.03" "$compat/libcuda.so.1"
ln -sfn "$cuda_lib_dir/libnvidia-ml.so.560.35.03" "$compat/libnvidia-ml.so.1"
export LD_LIBRARY_PATH="$compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
"$python" "$code" \
  --configuration parent_direct_separated --training-stage pan \
  --rna "$input/rna_reference_quantile.parquet" --rna-input-transform feature_zscore \
  --protein-prediction "$cv/protein_prediction.parquet" --protein-reliability "$input/protein_reliability.parquet" \
  --protein-provenance "$cv/protein_prediction_provenance.tsv" --protein-input-mode crossfit_prediction \
  --phosphosite "$input/phosphosite_logscale_aligned.parquet" --phosphosite-manifest "$manifest" \
  --split-manifest "$cv/split_manifest.tsv" --sample-metadata "$cv/sample_metadata.tsv" \
  --cophee-prior-bundle "$cophee" --esm2-prior-npz "$esm2" --esm2-fusion residue_residual \
  --device cuda:0 --sample-batch-size 112 --evaluation-batch-size 16 \
  --maximum-kinases-per-site 128 --operator-rank 128 --modality-tokens 32 --phospho-latent-tokens 16 \
  --transformer-layers 2 --site-chunk-size 400 --site-scaling identity_logscale --profile-mse-weight 1 \
  --site-huber-weight 0 --site-pearson-weight 0 --site-rank-weight 0 --profile-pearson-weight 0 \
  --precision bfloat16 --drop-last-training-batch --expected-train-samples 1426 \
  --expected-validation-samples 370 --expected-sealed-samples 0 --early-stopping-metric profile_spearman \
  --validation-interval-updates 16 --epochs 1000 --patience 1000 --seed 20260819 \
  --initial-checkpoint "$root/02_results/model_validation/scp682main1_random70_patient_equal_profile_mse_logscale_lr2e4_u288_20260815/models/decoder_retrieval_profile_spearman_best.pt" \
  --allow-initial-rna-scaler-mismatch \
  --learning-rate 0.00005 --maximum-finetune-updates 1 --target-study-residual --study-residual-tau 10 \
  --study-residual-min-overlap 5 --output-dir "$out" --validate-inputs-only \
  >"$out/logs/training.stdout.log" 2>"$out/logs/training.stderr.log"
test -f "$out/reports/input_validation.json"
touch "$out/SUCCESS"
