#!/usr/bin/env bash
set -euo pipefail

PY=/home/USER/.local/share/mamba/envs/omicverse/bin/python
SCRIPT=/data/lsy/Infinite_Stream/remote_scripts/run_scp682_ml_baseline_benchmark.py
OUT=/data/lsy/Infinite_Stream/SCP682-main/results/20260525_ml_baseline_benchmark_5fold

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "${OUT}"
echo "[launch] $(date)" > "${OUT}/run_status.txt"

"${PY}" "${SCRIPT}" \
  --output-dir "${OUT}" \
  --resume \
  --methods mean_pred mrna_naive lasso elasticnet pls random_forest gbm xgboost mlp \
  --simple-n-jobs 140 \
  --tree-outer-jobs 35 \
  --tree-inner-jobs 4 \
  --mlp-max-epochs 200 \
  --mlp-patience 10 \
  --mlp-batch-size 128 \
  > "${OUT}/baseline_benchmark.log" 2> "${OUT}/baseline_benchmark.stderr.log"

echo "[done] $(date)" >> "${OUT}/run_status.txt"
