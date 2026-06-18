# CURRENT_SCP682_MAIN_AUDIT

日期：2026-06-18

本文件记录 `organized_code` 中与 SCP682 bulk 主模型相关的脚本口径清理结果。

## 当前主入口

当前 bulk 主模型预测入口：

```text
organized_code/2_analysis/bulk/predict_scp682.py
```

该入口依赖同目录运行时：

```text
organized_code/2_analysis/bulk/scp682_v4_engine.py
organized_code/2_analysis/bulk/scp682_graph_runtime.py
```

公开模型公式统一写作：

```text
Y_hat = S_phi + 0.3 * Delta
```

其中 `S_phi` 是冻结磷酸化状态估计器，`Delta` 是图约束残差算子输出。

## 已替换的旧预测入口

以下旧文件名已经替换为当前主入口的兼容副本：

```text
organized_code/2_analysis/bulk/scp682_predict.py
organized_code/2_analysis/bulk/predict_scp682_v4_0_public_bulk_20260508.py
organized_code/2_analysis/bulk/predict_uploaded_bulk.py
```

保留旧文件名是为了兼容已有命令和旧调度脚本。文件内容已经走当前 SCP682 主模型入口。

以下旧 reviewer 脚本原来调用历史训练器，现已改为显式停止并指向当前主训练脚本：

```text
organized_code/2_analysis/bulk/run_scp682_reviewer_strict_holdout.py
organized_code/2_analysis/bulk/run_scp682_reviewer_strict_holdout_extra.py
```

## 已同步的运行时文件

以下文件已从当前 `SCP682_PORTABLE` 包同步：

```text
organized_code/2_analysis/bulk/predict_scp682.py
organized_code/2_analysis/bulk/scp682_graph_runtime.py
organized_code/2_analysis/bulk/scp682_v4_engine.py
```

`scp682_v4_engine.py` 的文件名属于内部组件名。它加载冻结状态估计器所需的历史权重、候选预测矩阵和总蛋白组件。论文和图注中统一称为 `S_phi`，避免把内部文件名当作主模型名称。

## 已清理的公开口径

以下文件已改为 `S_phi` / 当前主模型口径：

```text
organized_code/README.md
organized_code/REPRODUCE.md
organized_code/reproduction_chains/module_bulk.md
organized_code/reproduction_chains/module_atlas.md
organized_code/reproduction_chains/module_framework.md
organized_code/3_plotting/shared/make_fig1_rebuild.py
organized_code/3_plotting/bulk/panel_a.R
organized_code/2_analysis/bulk/extract_scp682_paper_materials.py
organized_code/2_analysis/bulk/run_scp682_consistent_external_from_bphi.py
organized_code/2_analysis/bulk/run_scp682_consistent_gamma_sensitivity.py
organized_code/2_analysis/bulk/run_scp682_lambda_sensitivity_20260602.py
organized_code/2_analysis/bulk/app.py
organized_code/2_analysis/bulk/server_stdlib.py
```

`organized_code/_copy_index.tsv` 的用途列也已同步，将旧主模型描述改为 `SCP682 main` 或 `S_phi` 内部组件描述。路径列保留原始来源，便于追溯。

## 保留的历史组件

以下类型文件仍可出现在仓库中：

```text
v4_engine/
v4_baseline_release/
SCP682_v4_0_* prediction files
train_cptac_*_20260429.py
scp682_22_* transfer-prior files in SC scripts
```

保留原因：

- `v4_engine` 和 `v4_baseline_release` 是 `S_phi` 状态估计器的权重与运行时来源。
- `20260429` 训练脚本产出总蛋白和残差候选组件，属于 `S_phi` 的内部构成。
- SC11 的 `scp682_22_*` transfer-prior 路径对应旧单细胞通路迁移实验。当前 SCP682_PORTABLE 包没有显式 pathway token，因此不能把这部分强行改写成 current main pathway-token 迁移。

## 后续检查命令

用旧关键词扫描 `organized_code` 时，命中结果需要分两类解释：

- 路径或内部权重文件名：可保留。
- 图注、README、论文素材、主预测入口说明：需要改为当前主模型口径。
