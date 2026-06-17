# Fig3 对抗式核验报告（SCP682-SC11）

生成日期：2026-06-17  
核验员：对抗式只读审查（基于 organized_code/ 实际文件内容 + Grep 抽查）  
底座文档：`module_sc.md`（2026-06-16）  
范围：Fig3a–g，7 panels

---

## 总览（每 panel 一行状态）

| Panel | 内容 | 状态 | 核心风险 |
|---|---|---|---|
| **3a** | SC11 架构示意图 | complete | 数字硬编码，无数据追溯风险 |
| **3b** | 内部5折 + 4外部队列 per-site 中位 Spearman | **gap** | Blair CSV→paired_matrices 缺专门脚本；Vivo-seq 预处理脚本缺失 |
| **3c** | CTNND1 T310 HeLa hexbin + STAT3 Y705 跨平台柱 | **risk** | `export_sc11_ctnnd1_hela_scatter.py` 读 `20260522_` 旧路径，非5折正式模型 |
| **3d** | GSE300551 Benchmark（SCP682-SC vs 6 baselines） | complete | 链路完整；SCP682-SC 数据同样来自 20260522 路径（与3c共线） |
| **3e** | per-readout win-scatter（n=11 readouts） | complete | 依赖 3d 同一上游，同行风险 |
| **3f** | HeLa UMAP 空间连贯性（NMF1/2/3 + 3 transfer sites） | **risk+gap** | `export_sc11_hela_umap.py` 也读 `20260522_` 旧路径；NDRG1/MAP2K4/PDPK1 transfer sites 导出脚本缺失 |
| **3g** | 扩展 ScNET 图消融（site-graph ablation） | **risk** | `run_scp682_sc11_m4_graph_controls.bat` 使用 SC7 warm-start + 不同 transfer-dir 参数，与正式训练超参分歧；链路完整但版本一致性存疑 |

---

## Panel 3a — 架构示意图

**状态：complete**

**链路核验**：
- `panel_a.R` → `make_panel_a()` 全程 `ggplot2::annotate` 硬编码绘制，无外部数据文件依赖。
- 关键数字（scFoundation 3,072-d / 9 pathway tokens / 56 readouts / 7,369 nodes / 882,959 edges）注释在 panel_a.R 开头注释行，标注为"verified hyperparameters"。
- `generate_fig1a_scp682_sc_architecture.py` 为 Python 并行版，均为 canonical。

**无缺口**。架构图是示意性信息图，数字一致性由 manifest.json 声明，不涉及权重路径。

---

## Panel 3b — 内部5折 + 4外部队列 per-site 中位 Spearman

**状态：gap**

**链路核验**：
- 训练链：`run_scp682_sc11_formal_internal_5fold.bat` → `train_scp682_sc11_expanded_scnet_site_gnn.py`（MODEL_NAME = "SCP682-SC11-expanded-ScNET-site-GNN"，确认为 SC11 canonical）→ 输出至 `20260529_scp682_sc11_current_main_internal_5fold_v1/fold_1..5`。
- 汇总：`summarize_scp682_sc11_internal_5fold.py` 读 `fold_n/tables/scp682_sc11_reconstruction_performance.tsv`。
- 外部验证：`export_sc11_external_predicted_observed_all.py` → `external_funnel_prep.py` → `panel_b.R` 读 `fig3_panel_b_data.tsv`（路径 `paper_materials_SCP682_SC11/04_figure_source_data/fig3/`）。

**已发现缺口（承继自 module_sc.md，实地核实）**：

1. **Blair phospho-seq CSV→paired_matrices 脚本缺失**：`inspect_blair_csv.py`、`inspect_blair_supp.py`、`list_blair_adt_features.py`、`patch_blair_prps6_schema.py` 均为检视/补丁工具，`_copy_index.tsv` 与 `2_analysis/sc/` 目录中无名为 `prepare_blair_phospho_*.py` 的 canonical 脚本。external_funnel_prep.py 中 `phospho_seq_blair_2025_phospho_multi` 被当作已有外部队列使用，但其 paired_matrices 的构建来源不明。

2. **Vivo-seq Th17 预处理脚本缺失**：`inspect_vivo_h5ad.py` 标为 support/检视工具，无专门 `prepare_vivo_seq_*.py` canonical 脚本。外部验证漏斗中该队列同样被直接引用。

**影响评估**：如审稿人要求复现从原始 GEO 数据起始的 Blair 和 Vivo-seq Th17 外部验证，当前代码包无法独立完成。Panel 3b 的 Blair 与 Vivo-Th17 数据点无法从代码起始端追溯。

---

## Panel 3c — CTNND1 T310 HeLa hexbin + STAT3 Y705 跨平台柱

**状态：risk**

**链路核验**：
- `panel_c.R` 读 `fig3/panel_a_hela_ctnnd1_t310_scatter.tsv`（左图）和 `panel_b_stat3_cross_platform.tsv`（右图柱），路径均在 `paper_materials_SCP682_SC11/04_figure_source_data/fig3/`。
- 上游：`export_sc11_ctnnd1_hela_scatter.py` 生成 `scp682_sc11_hela_ctnnd1_t310_predicted_observed.tsv`。
- **关键 risk**：`export_sc11_ctnnd1_hela_scatter.py` 硬编码加载路径为：
  ```
  D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260522_scp682_sc11_expanded_scnet_site_gnn_v1\models\scp682_sc11_final.pt
  ```
  这是 **2026-05-22 的单次训练模型**，而论文性能基准应来自 **2026-05-29 正式5折训练**（`20260529_scp682_sc11_current_main_internal_5fold_v1`）。

- 两个路径均出现在代码库中（`20260522` 命中 8 个文件，`20260529` 仅命中 3 个）。`assemble_scp682_sc_reviewer_benchmarks_v2.py` 的 `RESULT_DIR` 也指向 `20260522_`。

**结论**：Panel 3c 的 CTNND1 T310 HeLa scatter 数据与 Fig3b 的5折正式评估使用的是不同 checkpoint。两个模型是否超参一致需确认（`run_scp682_sc11_formal.bat` 使用 `scp682_main_pathway_token_transfer_dir` 参数，而5折版 `run_scp682_sc11_formal_internal_5fold.bat` 使用 `scp682-main-transfer-dir`，参数名不同）。

---

## Panel 3d — GSE300551 Benchmark（SCP682-SC vs 6 baselines）

**状态：complete（条件性）**

**链路核验**：
- 6 个 foundation model embedding 脚本（`precompute_scfoundation_embeddings_multidomain.py`、`precompute_scgpt_embeddings_multidomain.py` 等）均在 `2_analysis/sc/` 中为 support，已收录。
- `assemble_scp682_sc_reviewer_benchmarks_v2.py` 汇总至 `reviewer_requested_tables_v2/benchmark_table_reviewer_full_per_target.tsv`。
- `benchmark_leaderboard_prep.py` 做 matched-readout Wilcoxon 并写 `paper_final/fig3/main_figure_biology_v1/source_data/`。
- `render_biology.R` 最终渲染。

**条件性 risk**（与 3c 共线）：  
`assemble_scp682_sc_reviewer_benchmarks_v2.py` 中的 `RESULT_DIR = ... 20260522_...`，即 SCP682-SC 的 benchmark 结果也来自 `20260522` 模型，而非5折正式版。如果 `20260522` 与5折正式版是相同超参的不同跑次，benchmark 数值有效；如超参不同（见下方 3g 分析），benchmark 结论存在一致性风险。

---

## Panel 3e — per-readout win-scatter（SCP682-SC vs n=11 readouts）

**状态：complete（继承 3d 上游风险）**

**链路核验**：
- `winscatter_prep.py` 完全复用 `benchmark_leaderboard_prep.py` 的同一 matched-readout 集（`benchmark_table_reviewer_full_per_target.tsv`）。
- 输出 `fig3_benchmark_gse300551_per_readout.tsv` / `winsummary.tsv` → `render_biology.R` 渲染。
- 链路完整，脚本均为 canonical。

**继承风险**：同 3d，SCP682-SC 的 per-readout Spearman 来自 `20260522` 模型路径。

---

## Panel 3f — HeLa UMAP 空间连贯性（NMF1/2/3 + transfer sites）

**状态：risk + gap**

**链路核验**：
- `make_main_panels.R` 渲染 UMAP；读两个中间文件：
  - `fig9_hela_nmf_W_cell_scores.tsv`（来自 `paper_materials_SCP682_SC11/04_figure_source_data/fig3/`）
  - `scp682_sc11_hela_scfoundation_umap.tsv`（来自 `SRC_CACHE`，推断为 `04_figure_source_data/` 某子目录）
- `export_sc11_hela_umap.py` 生成 UMAP TSV，硬编码读取：
  ```
  D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260522_scp682_sc11_expanded_scnet_site_gnn_v1
  ```
  同样为 `20260522` 旧路径，而非5折正式版。

**已发现 gap**：
1. **NDRG1 T346、MAP2K4 S257、PDPK1 S241 三个 transfer-only 位点导出脚本缺失**：`export_sc11_ctnnd1_hela_scatter.py` 明确只处理 CTNND1 T310（line 49：`idx_target = ... ["CTNND1_T310"]`）。`export_sc11_external_predicted_observed_all.py` 导出全部5个外部队列的逐细胞预测，但其 RESULT_DIR 使用相对路径（`./results/SCP682_SC/`），输出文件名为 `scp682_sc11_predicted_observed_<cohort>.tsv`。需确认 NMF source 数据中是否包含这3个位点的预测值，以及由哪个脚本生成 `fig9_hela_nmf_W_cell_scores.tsv`（该文件在代码包中无生成脚本，为预计算产物）。

2. **`fig9_hela_nmf_W_cell_scores.tsv` 上游脚本未收录**：`module_sc.md` 的 Fig3f 链路未列出 NMF 分析脚本（应在 `run_scp682_sc11_validation_nmf.py` 中，该脚本在 `2_analysis/sc/` 中存在但未被 Fig3f 链路明确引用）。

---

## Panel 3g — 扩展 ScNET 图消融（site-graph ablation）

**状态：risk**

**链路核验**：
- 无-site-graph 消融：`run_scp682_sc11_no_site_graph_matched_ablation.bat` → `train_scp682_sc11_expanded_scnet_site_gnn.py`（`--site-graph-weight 0.0 --site-graph-scale 0.0 --site-graph-candidate-limit 0 --site-graph-max-aux-nodes 0`）。输出路径 `20260529_scp682_sc11_current_main_no_site_graph_ablation_v1`，超参与5折正式版高度一致，canonical。
- 无-pathway-attention 消融：`run_scp682_sc11_no_pathway_attention_ablation.ps1` → `train_scp682_sc11_expanded_scnet_site_gnn_no_attention_ablation.py`（`--disable-pathway-attention`）。
- M4 graph edge-source 消融：`run_scp682_sc11_m4_graph_controls.bat` → `train_scp682_sc11_expanded_scnet_site_gnn.py`（4 种 edge mode：rewired_all / no_copheemap / no_copheeksa / no_kstar）。

**发现 risk（M4 消融超参不一致）**：
`run_scp682_sc11_m4_graph_controls.bat` 与正式5折训练有两项关键分歧：

1. **Transfer 参数不同**：m4 使用 `--scp682-main-pathway-token-transfer-dir`（`scp682_main_pathway_token_transfer_prior_v1`），而正式5折使用 `--scp682-main-transfer-dir`（`scp682_main_sc_transfer_prior_v1`）。两者是不同的 bulk transfer prior 包，训练动力学可能不同。

2. **依赖 SC7 warm-start**：m4 消融强制加载 `20260520_scp682_sc7_scfoundation_scp682_main_transfer_v1\models\scp682_sc7_final.pt` 作为 warm-start，而正式5折训练无 warm-start（`run_scp682_sc11_formal_internal_5fold.bat` 未传 `--warm-start-model`）。m4 消融从 SC7（legacy warm-start 来源）开始训练，可能导致消融比较的基线不对齐。

3. **Val 策略不同**：m4 消融使用 `--val-fraction 0.12 --val-cells 30000`，正式5折使用 `--val-fraction 0 --val-cells 0`（纯 CV fold 划分，无独立 val split）。

- 数据汇总脚本：`assemble_scp682_sc_reviewer_ablation_v2.py` → `reviewer_ed_prep.py` → `panel_d.R`（lollipop）+ `fig22.R`（配对 Δ）。链路存在，但消融的 full-model 基准来自 `20260522_`，与消融模型输出路径 `20260602_scp682_sc11_m4_graph_controls` 不同日期、不同训练配置，配对比较的"full"基准定义需核实。

---

## 共性风险汇总

### Risk 1：模型版本分叉（最高优先级）

全部分析脚本（`export_sc11_ctnnd1_hela_scatter.py`、`export_sc11_hela_umap.py`、`assemble_scp682_sc_reviewer_benchmarks_v2.py`、`assemble_scp682_sc_reviewer_ablation_v2.py`、`build_scp682_sc_reviewer_missing_tables_v2.py`）统一指向 `20260522_scp682_sc11_expanded_scnet_site_gnn_v1`，而文档声明的正式5折训练输出为 `20260529_scp682_sc11_current_main_internal_5fold_v1`。

- Fig3b 的 "Internal (train recon)" 来自5折汇总，但外部验证数字（HeLa/Blair/GSE300551/Vivo-Th17）均来自 `20260522` 单次模型。
- 二者关系不清晰：若 `20260522` 是5折之前的 full-data 模型（作为推理模型）则逻辑自洽；若二者超参不同则存在版本混用。
- `run_scp682_sc11_formal.bat`（指向 `20260531` 输出，使用 pathway-token transfer + SC7 warm-start）与 `run_scp682_sc11_formal_internal_5fold.bat`（指向 `20260529` 输出，使用 sc transfer + 无 warm-start）超参明显不同，表明二者并非同一超参的两次运行。

### Risk 2：m4 消融 warm-start 使 ablation 比较基线不齐

m4 消融从 SC7 checkpoint 继续，而 full model 没有 warm-start。以有 warm-start 的消融对比无 warm-start 的 full，图残差贡献量的正方向可能被 warm-start 效应混淆。

### Gap 1：Blair/Vivo-seq 原始数据处理脚本缺失（影响 Fig3b/c/d 中这两个队列）

### Gap 2：NDRG1/MAP2K4/PDPK1 transfer sites 在 Fig3f 中的数据来源无专门导出脚本

### Gap 3：`fig9_hela_nmf_W_cell_scores.tsv` 生成脚本未在 Fig3f 链路中明确标记

---

## 关键3-5条疑点（给作者行动清单）

1. **澄清 20260522 vs 20260529 模型的定位关系**：两者超参是否相同？20260522 是否是 full-data 训练（5折结束后的最终模型）还是独立跑次？需在 `module_sc.md` 中明确哪个 checkpoint 是用于外部推理的"final model"，并确保所有分析脚本指向同一版本。

2. **补充 Blair phospho-seq CSV→paired_matrices 处理脚本或说明**：如已由手工或上游流程完成，需在复现链中注明数据位置和处理逻辑；否则该外部队列无法独立复现。

3. **说明 Vivo-seq Th17 h5ad 至模型输入的完整流程**：`inspect_vivo_h5ad.py` 仅为检视工具，预处理到 paired_matrices 格式的脚本缺失。

4. **核实 m4 消融（Fig3g）的 warm-start 是否导致比较基准不一致**：如 full-model benchmark（来自 `20260522`，无 warm-start）而 m4 消融（来自 `20260602`，有 SC7 warm-start），paired Δ 可能混入 warm-start 效益，需要提供同配置（有/无 warm-start 均一致）的 full vs ablation 配对。

5. **补充或明确 NDRG1 T346 / MAP2K4 S257 / PDPK1 S241 transfer sites 在 Fig3f UMAP 中的数据来源**：确认是否已在 `export_sc11_external_predicted_observed_all.py` 的输出中覆盖，以及 `fig9_hela_nmf_W_cell_scores.tsv` 的生成脚本路径。
