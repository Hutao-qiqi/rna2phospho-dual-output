#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/data/lsy/Infinite_Stream
PY=/home/USER/.local/share/mamba/envs/omicverse/bin/python
SCRIPT=$ROOT/remote_scripts/train_scp682_missing_ablation_degree_rewire.py
SUMMARY=$ROOT/remote_scripts/summarize_scp682_missing_ablation_grid.py
OUT_ROOT=$ROOT/SCP682-main/results/20260602_m4_knowledge_graph_controls_e160
PKG=$ROOT/SCP682_PORTABLE
PRIOR=$ROOT/01_data/pathway_prior
BASELINE=$ROOT/SCP682-main/inputs/general_baseline_predictions/general_baseline_internal_cptac_pdc_phosphosite.parquet
RNA=$ROOT/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/rna_log2_tpm_paired.parquet
SAMPLE_MANIFEST=$ROOT/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/sample_manifest.tsv
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
    --rna-path "$RNA" \
    --sample-manifest-path "$SAMPLE_MANIFEST" \
    --group-column cancer_label \
    --device cuda:0 \
    --epochs 160 \
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

run_one axis_dual_edge_rewired_all dual rewired_all
run_one axis_dual_edge_no_copheemap dual no_copheemap
run_one axis_dual_edge_no_copheeksa dual no_copheeksa
run_one axis_dual_edge_no_kstar dual no_kstar

"$PY" "$SUMMARY" \
  --grid-dir "$OUT_ROOT" \
  --output "$OUT_ROOT/tables/m4_knowledge_graph_controls_summary.tsv"

echo "done $(date)" > "$OUT_ROOT/done.txt"

