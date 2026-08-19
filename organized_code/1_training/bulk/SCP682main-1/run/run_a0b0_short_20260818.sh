#!/usr/bin/env bash
set -eu

root=${SCP682_ROOT:?set SCP682_ROOT to the SCP682-main checkout}
python=${SCP682_PYTHON:?set SCP682_PYTHON to the project python}
input_dir=$root/01_data/bulk/intermediate/random70_reference_rna_protein_logscale_phosphosite_inputs_20260815
code=$(cd "$(dirname "$0")/.." && pwd)/code/train_decoder_retrieval.py
manifest=$root/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/residual_target_manifest.tsv
cophee=$root/01_data/pathway_prior/processed/scp682main1_dynamic_edge_gated_binary_cophee_v1
esm2=$root/01_data/intermediate/esm2_sequence_prior_20260806/esm2_t33_650M_protein_and_site_prior.npz
stage1=$root/02_results/model_validation/scp682main1_random70_patient_equal_profile_mse_logscale_lr2e4_u288_20260815
results=$root/02_results/model_validation
a0_dir=$results/scp682main1_random70_a0_short_u96_20260818
b0_dir=$results/scp682main1_random70_b0_study_residual_u96_20260818
compat=$root/00_tools/nvidia_compat_560
cuda_lib_dir=${CUDA_LIB_DIR:-/usr/lib/x86_64-linux-gnu}

mkdir -p "$compat"
ln -sfn "$cuda_lib_dir/libcuda.so.560.35.03" "$compat/libcuda.so.1"
ln -sfn "$cuda_lib_dir/libnvidia-ml.so.560.35.03" "$compat/libnvidia-ml.so.1"
export LD_LIBRARY_PATH="$compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1

common=(
  --configuration parent_direct_separated --training-stage pan
  --rna "$input_dir/rna_reference_quantile.parquet"
  --rna-input-transform feature_zscore
  --protein-prediction "$input_dir/protein_prediction.parquet"
  --protein-reliability "$input_dir/protein_reliability.parquet"
  --protein-provenance "$input_dir/protein_prediction_provenance.tsv"
  --protein-input-mode crossfit_prediction
  --phosphosite "$input_dir/phosphosite_logscale_aligned.parquet"
  --phosphosite-manifest "$manifest"
  --split-manifest "$input_dir/split_manifest.tsv"
  --sample-metadata "$input_dir/sample_metadata.tsv"
  --cophee-prior-bundle "$cophee"
  --esm2-prior-npz "$esm2" --esm2-fusion residue_residual
  --device cuda:0 --sample-batch-size 112 --evaluation-batch-size 16
  --maximum-kinases-per-site 128 --operator-rank 128 --modality-tokens 32
  --phospho-latent-tokens 16 --transformer-layers 2 --site-chunk-size 400
  --site-scaling identity_logscale
  --profile-mse-weight 1
  --site-huber-weight 0 --site-pearson-weight 0 --site-rank-weight 0
  --profile-pearson-weight 0
  --precision bfloat16 --drop-last-training-batch
  --expected-train-samples 1796 --expected-validation-samples 770
  --expected-sealed-samples 0 --early-stopping-metric profile_spearman
  --validation-interval-updates 16 --epochs 1000 --patience 1000
  --seed 20260810 --initial-checkpoint "$stage1/models/decoder_retrieval_profile_spearman_best.pt"
  --learning-rate 0.00005 --maximum-finetune-updates 96
)

mkdir -p "$a0_dir/logs"
"$python" "$code" "${common[@]}" \
  --output-dir "$a0_dir" \
  >"$a0_dir/logs/training.stdout.log" \
  2>"$a0_dir/logs/training.stderr.log"
test -f "$a0_dir/models/decoder_retrieval_profile_spearman_best.pt"
touch "$a0_dir/SUCCESS"

mkdir -p "$b0_dir/logs"
"$python" "$code" "${common[@]}" \
  --target-study-residual \
  --study-residual-tau 0 \
  --output-dir "$b0_dir" \
  >"$b0_dir/logs/training.stdout.log" \
  2>"$b0_dir/logs/training.stderr.log"
test -f "$b0_dir/models/decoder_retrieval_profile_spearman_best.pt"
touch "$b0_dir/SUCCESS"

echo "A0 and B0 short comparison complete"
