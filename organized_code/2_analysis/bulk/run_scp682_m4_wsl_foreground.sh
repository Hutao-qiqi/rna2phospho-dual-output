#!/usr/bin/env bash
set -euo pipefail
pkill -f train_scp682_missing_ablation_degree_rewire.py 2>/dev/null || true
rm -rf /mnt/d/data/lsy/vm_lsy_parent/lsy/02_results/model_validation/20260602_m4_knowledge_graph_controls_e160_wsl
bash /mnt/d/data/lsy/vm_lsy_parent/lsy/remote_scripts/launch_scp682_m4_wsl_parallel.sh
