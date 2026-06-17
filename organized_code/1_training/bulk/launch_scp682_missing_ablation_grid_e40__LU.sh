#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/data/lsy/Infinite_Stream
PY=/home/USER/.local/share/mamba/envs/omicverse/bin/python
SCRIPT=$ROOT/remote_scripts/train_scp682_missing_ablation.py
OUT_ROOT=$ROOT/SCP682-main/results/20260523_missing_ablation_grid_e40
PKG=$ROOT/SCP682-22/frozen_release/SCP682_22_paper_package_20260520
PRIOR=$ROOT/01_data/pathway_prior
BASELINE=$ROOT/SCP682-main/inputs/general_baseline_predictions/general_baseline_internal_cptac_pdc_phosphosite.parquet
GPU=${GPU:-0}

mkdir -p "$OUT_ROOT/logs"
echo "start $(date)" > "$OUT_ROOT/run_status.txt"

run_one() {
  local name="$1"
  local axis="$2"
  local edge="$3"
  local out="$OUT_ROOT/$name"
  mkdir -p "$out"
  echo "[$(date)] start $name axis=$axis edge=$edge" | tee -a "$OUT_ROOT/run_status.txt"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$SCRIPT" \
    --package-dir "$PKG" \
    --prior-root "$PRIOR" \
    --output-dir "$out" \
    --general-baseline-path "$BASELINE" \
    --device cuda:0 \
    --epochs 40 \
    --batch-size 8 \
    --hidden 96 \
    --latent 32 \
    --inter-dim 96 \
    --embd-dim 32 \
    --num-layers 1 \
    --axis-mode "$axis" \
    --edge-mode "$edge" \
    > "$OUT_ROOT/logs/${name}.stdout.log" \
    2> "$OUT_ROOT/logs/${name}.stderr.log"
  echo "[$(date)] done $name" | tee -a "$OUT_ROOT/run_status.txt"
}

run_one axis_dual_edge_all dual all
run_one axis_site_only_edge_all site_only all
run_one axis_sample_only_edge_all sample_only all
run_one axis_dual_edge_copheemap dual copheemap
run_one axis_dual_edge_copheeksa dual copheeksa
run_one axis_dual_edge_kstar dual kstar

"$PY" "$ROOT/remote_scripts/summarize_scp682_missing_ablation_grid.py" \
  --grid-dir "$OUT_ROOT" \
  --output "$OUT_ROOT/tables/missing_ablation_grid_summary.tsv"

echo "done $(date)" > "$OUT_ROOT/done.txt"
