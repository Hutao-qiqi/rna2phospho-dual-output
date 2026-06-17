# 模型: SCP682
# 作用: 从原始实验脚本提取的最小可复现代码片段，文件名 launch_scp682_general_graph_external.sh
# 输入: ./data_root 下的训练数据、图先验和冻结基线预测
# 输出: ./paper_materials_SCP682 或结果目录中的模型、表格、报告
# 依赖: bash、Python、pandas、numpy、torch、torch_geometric
# 原始路径: remote_scripts/launch_scp682_general_graph_external.sh
# 原始版本: 20260523 结果目录对应脚本

#!/usr/bin/env bash
set -euo pipefail

OUT=./data_root/SCP682-main/results/20260523_general_graph_external_fixed_anchor
rm -rf "$OUT"
mkdir -p "$OUT/logs" "$OUT/predictions" "$OUT/tables" "$OUT/reports"

cd ./data_root
nohup env \
  PYTHONPATH=./data_root/.python_targets/pyg_torch251 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0 \
  ./conda_envs/3ef0fd2916e30222c6bfbc5c753696fe_/bin/python \
  ./data_root/remote_scripts/predict_scp682_general_graph_external.py \
  --output-dir "$OUT" \
  --device cuda:0 \
  --batch-size 8 \
  --anchor-k 25 \
  --anchor-temperature 0.08 \
  > "$OUT/logs/stdout.log" 2> "$OUT/logs/stderr.log" &

echo $! > "$OUT/run.pid"
cat "$OUT/run.pid"
