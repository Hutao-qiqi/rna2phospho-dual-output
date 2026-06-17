#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/mnt/d/data/lsy/vm_lsy_parent/lsy
PY=/home/USER/miniforge3/envs/omicverse/bin/python
SCRIPT=$ROOT/remote_scripts/train_scp682_missing_ablation_degree_rewire.py
SUMMARY=$ROOT/remote_scripts/summarize_scp682_missing_ablation_grid.py
OUT_ROOT=$ROOT/02_results/model_validation/20260602_m4_knowledge_graph_controls_e160_wsl
PKG=$ROOT/SCP682_MAIN
PRIOR=$ROOT/01_data/pathway_prior
BASELINE=$ROOT/SCP682_MAIN/inputs/general_baseline_predictions/general_baseline_internal_cptac_pdc_phosphosite.parquet
RNA=/mnt/d/data/lsy/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/rna_log2_tpm_paired.parquet
SAMPLE_MANIFEST=/mnt/d/data/lsy/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/sample_manifest.tsv

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/tables"
rm -f "$OUT_ROOT/fatal.log" "$OUT_ROOT/done.txt"
echo "start $(date -Is)" > "$OUT_ROOT/run_status.txt"

run_one() {
  local name="$1"
  local axis="$2"
  local edge="$3"
  local gpu="$4"
  local out="$OUT_ROOT/$name"
  mkdir -p "$out"
  echo "[$(date -Is)] start $name axis=$axis edge=$edge gpu=$gpu" | tee -a "$OUT_ROOT/run_status.txt"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" \
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
  echo "[$(date -Is)] done $name" | tee -a "$OUT_ROOT/run_status.txt"
}

run_pair() {
  run_one "$1" "$2" "$3" "$4" &
  local pid1=$!
  run_one "$5" "$6" "$7" "$8" &
  local pid2=$!
  set +e
  wait "$pid1"
  local code1=$?
  wait "$pid2"
  local code2=$?
  set -e
  echo "[$(date -Is)] pair exit pid1=$pid1 code1=$code1 pid2=$pid2 code2=$code2" | tee -a "$OUT_ROOT/run_status.txt"
  if [[ "$code1" -ne 0 || "$code2" -ne 0 ]]; then
    echo "failed pair pid1=$pid1 code1=$code1 pid2=$pid2 code2=$code2 $(date -Is)" | tee "$OUT_ROOT/fatal.log"
    exit 1
  fi
}

run_pair \
  axis_dual_edge_rewired_all dual rewired_all 0 \
  axis_dual_edge_no_copheemap dual no_copheemap 1

run_pair \
  axis_dual_edge_no_copheeksa dual no_copheeksa 0 \
  axis_dual_edge_no_kstar dual no_kstar 1

"$PY" "$SUMMARY" \
  --grid-dir "$OUT_ROOT" \
  --output "$OUT_ROOT/tables/m4_knowledge_graph_controls_summary.tsv"

echo "done $(date -Is)" > "$OUT_ROOT/done.txt"
echo "[$(date -Is)] done all" | tee -a "$OUT_ROOT/run_status.txt"
