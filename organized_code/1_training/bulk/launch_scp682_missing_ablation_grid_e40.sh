# 模型: SCP682
# 作用: 从原始实验脚本提取的最小可复现代码片段，文件名 launch_scp682_missing_ablation_grid_e40.sh
# 输入: ./data_root 下的训练数据、图先验和冻结基线预测
# 输出: ./paper_materials_SCP682 或结果目录中的模型、表格、报告
# 依赖: bash、Python、pandas、numpy、torch、torch_geometric
# 原始路径: remote_scripts/launch_scp682_missing_ablation_grid_e40.sh
# 原始版本: 20260523 结果目录对应脚本

#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=./data_root
PY=./user_home/.local./data_root/placeholder
SCRIPT=$ROOT/remote_scripts/train_scp682_missing_ablation.py
OUT_ROOT=$ROOT/SCP682-main/results/20260523_missing_ablation_grid_e40
PKG=$ROOT/SCP682_PORTABLE
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

