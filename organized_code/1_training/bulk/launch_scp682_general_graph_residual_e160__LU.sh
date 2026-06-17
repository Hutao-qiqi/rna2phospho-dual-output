#!/usr/bin/env bash
set -euo pipefail

OUT=/data/lsy/Infinite_Stream/SCP682-main/results/20260523_general_graph_residual_e160
rm -rf "$OUT"
mkdir -p "$OUT/logs" "$OUT/models" "$OUT/tables" "$OUT/predictions" "$OUT/reports"

cd /data/lsy/Infinite_Stream
nohup env \
  PYTHONPATH=/data/lsy/Infinite_Stream/.python_targets/pyg_torch251 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0 \
  /data/lsy/conda-envs/3ef0fd2916e30222c6bfbc5c753696fe_/bin/python \
  /data/lsy/Infinite_Stream/remote_scripts/train_scp682_general_graph_residual.py \
  --package-dir /data/lsy/Infinite_Stream/SCP682-22/frozen_release/SCP682_22_paper_package_20260520 \
  --prior-root /data/lsy/Infinite_Stream/01_data/pathway_prior \
  --general-baseline-path /data/lsy/Infinite_Stream/SCP682-main/inputs/general_baseline_predictions/general_baseline_internal_cptac_pdc_phosphosite.parquet \
  --output-dir "$OUT" \
  --device cuda:0 \
  --epochs 160 \
  --batch-size 4 \
  --hidden 64 \
  --latent 32 \
  --inter-dim 96 \
  --embd-dim 32 \
  --num-layers 1 \
  --pseudo-weight 0.75 \
  --anchor-k 25 \
  --anchor-temperature 0.08 \
  > "$OUT/logs/stdout.log" 2> "$OUT/logs/stderr.log" &

echo $! > "$OUT/run.pid"
cat "$OUT/run.pid"
