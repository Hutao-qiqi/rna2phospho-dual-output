#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/d/data/lsy/vm_lsy_parent/lsy
OUT=$ROOT/02_results/model_validation/20260602_m4_single_rewired_wsl
mkdir -p "$OUT/logs"
rm -f "$OUT"/done.txt "$OUT"/fatal.log
nohup /home/USER/miniforge3/envs/omicverse/bin/python "$ROOT/remote_scripts/train_scp682_missing_ablation_degree_rewire.py" \
  --package-dir "$ROOT/SCP682_MAIN" \
  --prior-root "$ROOT/01_data/pathway_prior" \
  --output-dir "$OUT/axis_dual_edge_rewired_all" \
  --general-baseline-path "$ROOT/SCP682_MAIN/inputs/general_baseline_predictions/general_baseline_internal_cptac_pdc_phosphosite.parquet" \
  --rna-path /mnt/d/data/lsy/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/rna_log2_tpm_paired.parquet \
  --sample-manifest-path /mnt/d/data/lsy/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/sample_manifest.tsv \
  --group-column cancer_label \
  --device cuda:0 \
  --epochs 160 \
  --batch-size 8 \
  --hidden 96 \
  --latent 32 \
  --inter-dim 96 \
  --embd-dim 32 \
  --num-layers 1 \
  --axis-mode dual \
  --edge-mode rewired_all \
  > "$OUT/logs/axis_dual_edge_rewired_all.stdout.log" \
  2> "$OUT/logs/axis_dual_edge_rewired_all.stderr.log" &
echo $! > "$OUT/run.pid"
echo "started pid=$(cat "$OUT/run.pid")"
