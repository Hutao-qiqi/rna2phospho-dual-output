#!/usr/bin/env bash
set -euo pipefail

PY=/home/USER/.local/share/mamba/envs/omicverse/bin/python
SCRIPT=/data/lsy/Infinite_Stream/remote_scripts/run_scp682_fast_fullsite_baselines.py
OUT=/data/lsy/Infinite_Stream/SCP682-main/results/fast_fullsite_baselines
EXT=/data/lsy/Infinite_Stream/SCP682-main/results/fast_fullsite_baselines_extdata

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "${OUT}" "${EXT}" "${OUT}/logs" "${EXT}/logs"
echo "[launch] $(date)" > "${OUT}/run_status.txt"

"${PY}" "${SCRIPT}" --mode trees_cpu --output-dir "${OUT}" --ext-output-dir "${EXT}" --resume \
  > "${EXT}/logs/trees_cpu.log" 2> "${EXT}/logs/trees_cpu.stderr.log" &
CPU_TREE_PID=$!
echo "${CPU_TREE_PID}" > "${EXT}/trees_cpu.pid"
echo "[start trees_cpu] ${CPU_TREE_PID} $(date)" >> "${OUT}/run_status.txt"

"${PY}" "${SCRIPT}" --mode fullsite --output-dir "${OUT}" --ext-output-dir "${EXT}" --resume \
  > "${OUT}/logs/fullsite.log" 2> "${OUT}/logs/fullsite.stderr.log"
echo "[done fullsite] $(date)" >> "${OUT}/run_status.txt"

"${PY}" "${SCRIPT}" --mode trees_xgb --output-dir "${OUT}" --ext-output-dir "${EXT}" --resume \
  > "${EXT}/logs/trees_xgb.log" 2> "${EXT}/logs/trees_xgb.stderr.log"
echo "[done trees_xgb] $(date)" >> "${OUT}/run_status.txt"

wait "${CPU_TREE_PID}"
echo "[done trees_cpu] $(date)" >> "${OUT}/run_status.txt"

echo "[done all] $(date)" >> "${OUT}/run_status.txt"
date > "${OUT}/done.txt"
