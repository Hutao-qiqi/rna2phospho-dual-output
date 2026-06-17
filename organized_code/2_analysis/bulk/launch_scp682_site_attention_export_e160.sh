#!/usr/bin/env bash
set -euo pipefail

OUT=/data/lsy/Infinite_Stream/SCP682-main/results/20260612_site_attention_export_e160
SCRIPT=/data/lsy/Infinite_Stream/remote_scripts/export_scp682_site_attention.py
PKG=/data/lsy/Infinite_Stream/SCP682/frozen_release/SCP682_main_exact_scnet_gnn_20260522
TRAIN_DIR=/data/lsy/Infinite_Stream/remote_scripts
PY=/data/lsy/conda-envs/3ef0fd2916e30222c6bfbc5c753696fe_/bin/python

mkdir -p "$OUT/logs" "$OUT/tables"
rm -f "$OUT/done.txt" "$OUT/fatal.log"

cd /data/lsy/Infinite_Stream
nohup env \
  SCP682_TRAIN_DIR="$TRAIN_DIR" \
  PYTHONPATH=/data/lsy/Infinite_Stream/.python_targets/pyg_torch251 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$PY" "$SCRIPT" \
    --package-dir "$PKG" \
    --prior-root /data/lsy/Infinite_Stream/01_data/pathway_prior \
    --checkpoint "$PKG/models/scp682_main_v4_exact_scnet_gnn_best.pt" \
    --general-baseline-path "$PKG/training_set/v4_phosphosite_baseline.parquet" \
    --rna-path /data/lsy/Infinite_Stream/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/rna_log2_tpm_paired.parquet \
    --sample-manifest-path /data/lsy/Infinite_Stream/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/sample_manifest.tsv \
    --output-dir "$OUT" \
    --device cuda:0 \
    --hidden 64 \
    --latent 32 \
    --inter-dim 96 \
    --embd-dim 32 \
    --num-layers 1 \
  > "$OUT/logs/export_attention.log" \
  2> "$OUT/logs/export_attention.stderr.log" \
  && echo "done $(date -Is)" > "$OUT/done.txt" \
  || { echo "fatal $(date -Is)" > "$OUT/fatal.log"; exit 1; } &

echo $! > "$OUT/run.pid"
echo "$OUT"
cat "$OUT/run.pid"
