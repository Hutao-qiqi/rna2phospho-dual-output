# PPKO 模块复现链

模块范围：药物扰动算子 PPKO V10B (strong300)、P100 验证 (n=125)、患者级 TCGA-TCPA 响应 (AUC 0.72)、Fig4 全部 panel (a-h) 及 ED baseline 图。

canonical 模型身份：`SCP682_PPKO_V10B_transferable` / `paper_materials_SCP682_PPKO`，架构 `AttentionPriorManifoldV10`，训练脚本默认 620 epoch，权重 `scp682_ppko_v10b_strong300_best.pt`。

---

## 1. 数据准备（上游，支持训练与验证）

### 1-1. 下载原始数据

| 步骤 | 脚本 | 机器 | 说明 |
|------|------|------|------|
| 下载 DecryptM PXD037285 | `remote_scripts/download_decryptm_pxd037285_curves.py` | LW | PRIDE API 原始 zip |
| 下载 LINCS P100 LVL4 | `remote_scripts/download_lincs_p100_lvl4.py` | LW | PanoramaWeb GCT 文件 |

### 1-2. 构建训练输入

| 步骤 | 脚本 | 机器 | 输出 |
|------|------|------|------|
| DecryptM PXD037285 剂量曲线解析 | `remote_scripts/build_decryptm_pxd037285_phospho_curve_inputs.py` | LW | `decryptm_pxd037285_curve_v1` |
| DecryptM delta v8 计算 | `remote_scripts/build_decryptm_comparison_delta_v8.py` | LW | `comparison_delta_v8/` |
| LINCS P100 LVL4 标准化输入 | `remote_scripts/build_lincs_p100_phospho_perturb_inputs.py` | LW | `lincs_p100_lvl4_v1` |
| P100 delta v7 计算 | `remote_scripts/build_lincs_p100_comparison_delta_v7.py` | LW | `lincs_p100_comparison_delta_v7` |
| 合并 DecryptM+P100 联合输入（remote版） | `remote_scripts/build_joint_decryptm_p100_phospho_inputs.py` | LW | `decryptm_p100_joint_v1` |
| 合并 DecryptM+P100 联合输入（paper版） | `paper_materials_SCP682_PPKO/03_code/preprocessing/build_joint_decryptm_p100_phospho_inputs.py` | L | `decryptm_p100_joint_v1`（论文包可复现版） |

### 1-3. 构建图先验

| 步骤 | 脚本 | 机器 | 输出 |
|------|------|------|------|
| CoPheeMap 先验边 v6 | `remote_scripts/build_copheemap_prior_for_scp682_ppko_v6.py` | LW | `copheemap_v6 edge list` |
| KSTAR 边 v5 | `remote_scripts/build_kstar_edges_for_scp682_ppko_v5.py` | LW | `kstar_kinase_site_edges.tsv` |
| Signed 调控先验 v9 | `remote_scripts/build_signed_phospho_regulatory_prior_v9.py` | LW | `signed_phospho_regulatory_prior_v9` |
| 全局磷蛋白异质图 V10 | `remote_scripts/build_global_phosphoprotein_heterograph_v10.py` | LW | `global_phosphoprotein_heterograph_v10_measured_string700_top50` |
| 图先验质检 | `remote_scripts/inspect_signed_phospho_prior.py` | LW | stdout 统计 |
| PPKO 数据集特征审计 | `remote_scripts/summarize_scp682_ppko_dataset_features.py` | LW | `dataset_feature_overview.json`，支持 Fig4a |

### 1-4. LINCS P100 覆盖率审计（support）

| 步骤 | 脚本 | 机器 |
|------|------|------|
| LINCS P100 覆盖统计 | `remote_scripts/summarize_lincs_p100_local_coverage.py` | LW |

---

## 2. 训练（canonical V10B strong300）

| 步骤 | 脚本 | 机器 | 输入 | 输出 |
|------|------|------|------|------|
| **canonical 训练（transferable版）** | `SCP682_PPKO_V10B_transferable/scripts/pretrain_v10b_strong300.py` | LW | `decryptm_comparison_delta_v8` + `global_phosphoprotein_heterograph_v10_measured_string700_top50` | `scp682_ppko_v10b_strong300_best.pt` |
| canonical 训练（paper_materials 版，同架构） | `paper_materials_SCP682_PPKO/03_code/training/pretrain_v10b_strong300.py` | L | 同上 | 同上 |

**版本漂移说明**：`_pretrain_v10.py`（旧入口）及全部 pre-V10B 版本（v1–v22，包括 cophee_atlas_v6、signed_manifold_v9、context_ppi_v11、scnet_dualview_v12、expert_fusion_v19 等）均已标记为 legacy，不纳入复现集。canonical 以 `pretrain_v10b_strong300.py`（AttentionPriorManifoldV10，默认 620 epoch）为准；训练脚本会输出 `scp682_ppko_v10b_strong300_best.pt`，并保留旧名 checkpoint 作为兼容别名。

---

## 3. 分析（推理 / 统计 / 验证）

### 3-1. P100 n=125 位点级推理（Fig4b/c 源数据）

| 步骤 | 脚本 | 机器 | 输出 |
|------|------|------|------|
| canonical V10B 全125条 P100 delta | `03_code/figure_generation/export_v10b_p100_sitelevel_all125.py` | LW | `p100_v10b_all125_unique_sitelevel_delta_long.tsv` |
| 方向统计量计算 | `make_fig4_locked_p100_direction_stats.py` | L | `panel_b/c/d TSV` |
| 锁定 P100 源数据表构建 | `make_fig4_locked_p100_tables.py` | L | `fig4 panel b/c 源数据 TSV` |

### 3-2. P100 已发表基线对比（Fig4b/c ED benchmark）

| 步骤 | 脚本 | 机器 | 输出 |
|------|------|------|------|
| V10B vs 已发表基线（评估主脚本） | `03_code/model_validation/evaluate_ppko_p100_published_baselines.py` | LW | `p100_baseline_comparison 表` |
| Windows 启动包装 | `remote_scripts/run_ppko_p100_published_baselines_windows.cmd` | LW | 同上 |
| 当前补跑入口 | V10B 冻结包补跑调度 | L | `2_analysis/ppko/rerun_missing_items.py`（support，指向冻结包验证、P100 基线和位点级导出脚本） |

### 3-3. P100 全药物集验证（canonical，Fig4c）

| 步骤 | 脚本 | 机器 | 输出 |
|------|------|------|------|
| V10B P100 全药物集验证 | `SCP682_PPKO_V10B_transferable/scripts/validate_v10b_p100_all_drugs.py` | LW | `p100 drug validation tables` |

### 3-4. PXD063604 外部验证（KRAS 抑制剂，Fig4b/c 机制）

| 步骤 | 脚本 | 机器 | 输出 |
|------|------|------|------|
| PXD063604 位点级推理 | `03_code/figure_generation/export_v10b_pxd063604_sitelevel.py` | LW | `pxd063604_v10b_sitelevel_delta_long.tsv` |
| PXD063604 KRAS 通路节点汇总表 | `03_code/figure_generation/make_pxd063604_reusable_candidate_tables.py` | L | `fig4 汇总表` |

### 3-5. TCGA-TCPA 患者级响应（Fig4g/h，AUC 0.72）

| 步骤 | 脚本 | 机器 | 输出 |
|------|------|------|------|
| V10B 对 TCGA-TCPA 患者推断 + AUC 计算 | `tcga_tcpa_ppko_patient_response_v1.py` | LW | `all_model_patient_predictions.tsv`, `per_drug_auc_*.tsv` |
| PPKO vs 随机/全局对照 AUC | `tcga_tcpa_general_score_controls_v1.py` | LW | `v10b_general_score_control_auc.tsv`, `v10b_random_marker_control_auc.tsv` |
| TCGA-TCPA OS 生存分析（Cox+KM） | `03_code/clinical_validation/run_ppko_tcga_survival_analysis.py` | L | `survival_v10b_300/ 生存分析结果` |
| TCGA-TCPA Fig4d 图表构建 | `make_fig4_tcga_validation_plot_tables.py` | L | `TCGA 药物响应 AUC/ROC 表` |

---

## 4. 画图（Fig4 全部 panel）

Fig4 画图主目录：`02_results/figure_outputs/fig4_v3/scripts/`
出图根目录：`04_figures/fig4_ppko_v10b_mechanism/` 及 `paper_final/fig4/main_figure/`

| panel | 脚本 | 机器 | 输入 |
|-------|------|------|------|
| **Fig4a 架构示意图**（matplotlib）| `03_code/figure_generation/make_scp682_ppko_methods_schematic.py` | L | 无外部数据 |
| Fig4a 占位 ggplot | `02_results/figure_outputs/fig4_v3/scripts/11_panel_a_placeholder.R` | L | support，正式版由 BioRender SVG 替换 |
| **Fig4b P100 整体验证 grouped bar** | `02_results/figure_outputs/fig4_v3/scripts/03_panel_b.R` | L | `panel_b_p100_true_overall_bars_cosine_direction.tsv` |
| **Fig4c per-comparison 雨云图** | `02_results/figure_outputs/fig4_v3/scripts/04_panel_c.R` | L | `panel_c_*_long.tsv` |
| **Fig4d 药物 class 哑铃图** | `02_results/figure_outputs/fig4_v3/scripts/05_panel_d.R` | L | `panel_d_*_drug_class_*.tsv` |
| **Fig4e per-drug 热图** | `02_results/figure_outputs/fig4_v3/scripts/06_panel_e.R` | L | `panel_e_p100_true_drug_heatmap_multi_metrics.tsv` |
| **Fig4f PPKO vs zero 散点+边缘小提琴** | `02_results/figure_outputs/fig4_v3/scripts/07_panel_f.R` | L | `panel_f_p100_true_vs_zero_paired*.tsv` |
| **Fig4g TCGA ROC+boxplot（AUC=0.72）** | `02_results/figure_outputs/fig4_v3/scripts/08_panel_g.R` | L | `panel_g_tcga_v10b_*.tsv` |
| **Fig4h 对照 AUC 棒棒糖图** | `02_results/figure_outputs/fig4_v3/scripts/09_panel_h.R` | L | `panel_h/i 对照 AUC tsv` |
| 新 panel：drug×cell-line 余弦热图 + 位点级散点 | `02_results/figure_outputs/fig4_v3/scripts/20_new_panels.R` | L | `representative_comparison_candidates.tsv` |
| **Fig4 主合图组装** | `02_results/figure_outputs/fig4_v3/scripts/99_assemble.R` | L | panels/panel_*.pdf/png |
| **ED Fig M2 已发表基线对比 bar** | `02_results/figure_outputs/fig4_v3/scripts/40_ed_baselines.R` | L | `p100_v10b_published_baseline_comparison_summary.tsv` |
| 全局主题/路径定义（support） | `02_results/figure_outputs/fig4_v3/scripts/01_config.R` | L | - |
| 统一数据加载函数（support） | `02_results/figure_outputs/fig4_v3/scripts/02_load_data.R` | L | fig4 所有 panel 源表 |
| **Fig4b/c PXD063604 机制面板** | `03_code/figure_generation/make_pxd063604_v10b_mechanism_panels.py` | LW | `pxd063604_v10b_sitelevel_delta_long.tsv` |
| **Fig4b/c/d P100 机制面板** | `03_code/figure_generation/make_fig4_ppko_v10b_mechanism_panels.py` | L | `p100_v10b_all125_unique_sitelevel_delta_long.tsv` |
| 缩略图生成（support） | `02_results/figure_outputs/fig4_v3/scripts/_thumb_final.R` | L | `Fig4_v1.0.png` |
| batch 缩略图（support） | `02_results/figure_outputs/fig4_v3/scripts/_thumb.R` | L | panels/*.png |

---

## 5. 缺口（gap）

| 缺口 | 说明 |
|------|------|
| **Fig4a 正式 SVG** | `11_panel_a_placeholder.R` 只生成占位 ggplot；正式架构示意图由 BioRender SVG 提供，源文件路径未在记录中出现（`make_scp682_ppko_methods_schematic.py` 产出 matplotlib 版，但与论文最终版本的关系需确认） |
| **P100 已发表基线比较汇总表** | `40_ed_baselines.R` 依赖 `p100_v10b_published_baseline_comparison_summary.tsv`；该表由 `evaluate_ppko_p100_published_baselines.py` 生成，固定读取 `SCP682_PPKO_V10B_transferable` 冻结权重，并输出到 `results/p100_published_baselines/`。release-v1 补表脚本已归档到 `organized_code/legacy/ppko/`，不作为 Fig4 基线主链路。 |
| **TCGA 生存分析输入** | `run_ppko_tcga_survival_analysis.py` 需要 `v10b_300_patient_predictions.tsv` 和 `TCGA_survival_data.tsv`，后者来源未在 records 中单独列为一个数据下载/准备脚本（可能直接依赖 TCGA 公开文件） |
| **Fig4a 图统计数（graph_statistics）** | 异质图节点/边统计（8192 site / 8751 protein 节点，用于 Fig4a 标注）无专用导出脚本，需由 `build_global_phosphoprotein_heterograph_v10.py` 产出的 TSV 直接读数 |
| **panel_g/h 数据表锁定路径** | `figure_sources/20260528_fig4_locked_p100_v10b_cosine_direction/` 是 `02_load_data.R` 的硬编码路径，该锁定目录是由哪个分析脚本写入的，records 中无直接对应项（可能是 `make_fig4_locked_p100_tables.py` 写出但路径存在差异） |
