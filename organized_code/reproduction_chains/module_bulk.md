# SCP682 Bulk 模块复现链

生成日期：2026-06-16  
范围：bulk RNA→phospho 主模型 (SCP682-22 v4 exact_scnet) 训练/推理、CPTAC 内部 OOF benchmark、4 个外部队列验证、Fig2 a-d；及 atlas 模块 Fig2 e-f-g (NMF k30 泛癌图谱)。

---

## 模块边界与版本说明

- **Canonical 模型**：SCP682-22 / SCP682_PORTABLE（v4 exact_scnet_gnn，公式 Y=B_phi+0.3*delta）
- **Canonical atlas**：10,023 TCGA 原发瘤，signed-split NMF k=30，模块活性→Cox生存
- **排除（legacy）**：SCP682-7~21、SCP682-23~28（编号 < 22 或 > 22 均为 legacy dev 编号）、film_vae/72-pathway-token/CVAE v3.x/DANN系列、NAS v1/v2 stacking 路线、SCP682-30/31/32 探索版

---

## Fig2a — 架构示意图

**需求**：展示 B_phi 冻结基线 + G_theta 图约束残差架构；420,102 位点图边；21,925 样本图边；α=0.3 shrinkage 系数确认。

| 步骤 | 脚本 | 机器 | category |
|------|------|------|----------|
| 1. 图先验构建（位点图边来源） | `03_code/model_validation/priors/process_copheemap_prior_20260428.py` | LU | 分析/数据准备 |
| 2. 总蛋白基线训练（B_phi 冻结组件） | `03_code/model_validation/training/train_cptac_total_proteome_film_vae_z_direct_residual_v2_20260429.py` | LU | 训练(support) |
| 3. 主训练启动（canonical exact_scnet） | `paper_materials_SCP682/03_code/training/launch_scp682_general_graph_residual_e160.sh` | LU | 训练 |
| 4. α-scan（确认0.3系数） | `paper_materials_SCP682/03_code/training/run_scp682_shrinkage_sensitivity_grid.py` | LU | 分析 |
| 5. 图统计导出（边数数字） | `paper_materials_SCP682/04_figure_source_data/fig2_extensions/make_fig2_extensions.py` | L | 分析/support |
| 6. Panel a 渲染 | `paper_final/fig2/scripts/panels/panel_a.R` | L | 画图 |

**Gap**：graph_statistics.tsv 中 420,102 / 21,925 边数的直接导出脚本未找到专用工具；这些数字嵌在 process_copheemap_prior 和 paper_final/fig2/scripts/panels/panel_a.R 中硬编码。

---

## Fig2b — CPTAC 内部 OOF 跨组织 Benchmark（9方法）

**需求**：SCP682 vs 8 RNA 基线（含 DeepGxP/Ridge/ElasticNet/RF等）；5折OOF；per-site Spearman；配对Wilcoxon P<1e-4。

| 步骤 | 脚本 | 机器 | category |
|------|------|------|----------|
| 1. CPTAC PDC 磷酸化数据下载 | `03_code/model_validation/download_pdc_phosphoproteome_report_files.py` | LU | 分析/support |
| 2. GDC STAR counts 下载 | `03_code/model_validation/download_cptac_gdc_open_star_counts.py` | LU | 分析/support |
| 3. PDC 总蛋白下载 | `03_code/model_validation/download_pdc_proteome_report_files.py` | LU | 分析/support |
| 4. CPTAC pancancer RNA-phospho 配对矩阵构建 | `03_code/model_validation/prepare_cptac_pancancer_rna_phosphoproteome_pairs.py` | LU | 分析/support |
| 5. 磷酸化位点 z-score 标准化 locked v1 | `03_code/model_validation/prepare_cptac_phosphosite_gene_site_locked_v1.py` | LU | 分析/support |
| 6. CPTAC 多任务 locked v2 矩阵构建 | `03_code/model_validation/data_preparation/prepare_cptac_multi_task_locked_v2_20260429.py` | LU | 训练/support |
| 7. 图先验构建 | `03_code/model_validation/priors/process_copheemap_prior_20260428.py` | LU | 分析/support |
| 8. GSEA 通路辅助标签构建 | `03_code/model_validation/prepare_gsea_pathway_aux_labels.py` | LU | 分析/support |
| 9. 总蛋白基线训练 (B_phi) | `03_code/model_validation/training/train_cptac_total_proteome_film_vae_z_direct_residual_v2_20260429.py` | LU | 训练/support |
| 10. canonical 主训练 160 epoch | `paper_materials_SCP682/03_code/training/launch_scp682_general_graph_residual_e160.sh` | LU | 训练 |
| 11. 主训练器代码 | `paper_materials_SCP682/03_code/training/train_scp682_general_graph_residual.py` | LU | 训练 |
| 12. 8 RNA 基线 benchmark（fast baselines） | `remote_scripts/launch_scp682_fast_fullsite_baselines.sh` | LU | 分析/support |
| 13. ML 基线 benchmark（9方法5折） | `remote_scripts/launch_scp682_ml_baseline_benchmark_full.sh` | LU | 分析/support |
| 14. OOF branch benchmark（5种RNA基线） | `paper_materials_SCP682/03_code/training/run_scp682_oof_branch_benchmark.py` | LU | 分析 |
| 15. DeepGxP 基线复现（CPTAC） | `03_code/model_validation/run_deepgxp_cptac_half_retrain_20260511.py` | LU | 分析/support |
| 16. DeepGxP 基线复现（TCGA-TCPA RPPA） | `03_code/model_validation/run_deepgxp_tcpa_reproduction_20260507.py` | LU | 分析/support |
| 17. Srivastava/Lau2022 基线 | `03_code/model_validation/run_srivastava_lau2022_rppa_baseline.py` | LU | 分析/support |
| 18. 输入守卫 & 样本中心化 | `03_code/model_validation/scp682_v4_0_input_guard_and_sample_centering_20260503.py` | LU | 分析/support |
| 19. bulk 主分析面板构建（per-site rho 多基线） | `03_code/model_interpretation/build_scp682_bulk_main_panels.py` | LU | 分析 |
| 20. Panel b 渲染 | `paper_final/fig2/scripts/panels/panel_b.R` | L | 画图 |

---

## Fig2c — 外部多队列冻结 benchmark（4队列）

**需求**：冻结 SCP682 对 FU-iCCA / TU-SCLC / CHCC-HBV 等独立队列推理；6 个 learned baseline 同步评估。

| 步骤 | 脚本 | 机器 | category |
|------|------|------|----------|
| 1. 外部磷酸化队列下载（v1） | `remote_scripts/download_external_bulk_phospho_validation_v1.py` | LW | 分析/support |
| 2. 外部队列大文件下载（v2） | `remote_scripts/download_external_bulk_large_validation_assets_v2.py` | LW | 分析/support |
| 3. 本地下载并推送外部数据 | `remote_scripts/local_download_external_bulk_then_push.py` | L | 分析/support |
| 4. 冻结 SCP682 外部推理启动脚本 | `paper_materials_SCP682/03_code/evaluation/launch_scp682_general_graph_external.sh` | LU | 分析 |
| 5. 冻结 SCP682 外部推理核心代码 | `paper_materials_SCP682/03_code/inference/predict_scp682_general_graph_external.py` | LU | 分析 |
| 6. 外部 9 模型 benchmark 汇总 | `paper_materials_SCP682/04_figure_source_data/fig2_extensions/make_external_9model_benchmark.py` | L | 分析/support |
| 7. Lau 协议总蛋白外部 benchmark | `03_code/model_validation/run_lau_style_external_total_validation_20260505.py` | L | 分析/support |
| 8. Lau 协议控制（logratio vs zscore） | `03_code/model_validation/run_lau2022_total_protein_protocol_controls_20260504.py` | LU | 分析/support |
| 9. Lau 协议控制（Ridge矩阵版） | `03_code/model_validation/run_lau2022_total_protein_protocol_controls_fast_ridge_20260504.py` | LU | 分析/support |
| 10. SCP682-2 Lau 总蛋白 + 外部 benchmark | `03_code/model_validation/run_scp682_2_lau_total_benchmark_and_external_20260505.py` | LU | 分析/support |
| 11. Portable 预测入口 | `portable_src/predict_scp682.py` | LUW | 分析 |
| 12. Panel c 渲染 | `paper_final/fig2/scripts/panels/panel_c.R` | L | 画图 |
| 13. Panel c 渲染入口（build） | `paper_final/fig2/scripts/panels/_build_panel_c.R` | L | 画图/support |

---

## Fig2d — 位点图注意力可解释性

**需求**：top 5% 高注意力边 → same-protein 4.7×/same-pathway 1.8× 富集；Fisher exact；420,102 边总量。

| 步骤 | 脚本 | 机器 | category |
|------|------|------|----------|
| 1. 消融训练网格（轴×边源）— canonical e40 | `paper_materials_SCP682/03_code/training/launch_scp682_missing_ablation_grid_e40.sh` | LU | 训练 |
| 2. 消融训练器代码 | `paper_materials_SCP682/03_code/training/train_scp682_missing_ablation.py` | LU | 训练 |
| 3. 消融网格汇总 | `paper_materials_SCP682/03_code/training/summarize_scp682_missing_ablation_grid.py` | LU | 分析/support |
| 4. 注意力权重导出（e160 canonical 权重） | `paper_final/fig2/scripts/export_scp682_site_attention.py` | LU | 分析 |
| 5. 注意力导出启动脚本 | `remote_scripts/launch_scp682_site_attention_export_e160.sh` | LU | 分析 |
| 6. Fisher exact + 富集分析 | `paper_final/fig2/scripts/analyze_site_attention.py` | L | 分析 |
| 7. Panel d 渲染（paper_final版） | `paper_final/fig2/scripts/panels/panel_d.R` | L | 画图 |
| 8. Panel d 渲染（materials包版，hardcoded数值） | `paper_materials_SCP682/04_figure_source_data/fig2/_scripts/panels/panel_d.R` | L | 画图 |
| 9. 知识图消融（边源：rewire/no_copheemap 等） | `remote_scripts/launch_scp682_m4_knowledge_graph_controls_e160_windows.ps1` | LW | 训练/support |
| 10. 知识图消融（Ubuntu版） | `remote_scripts/launch_scp682_m4_knowledge_graph_controls_e160.sh` | LU | 训练/support |
| 11. 双轴消融 Windows 并行 | `remote_scripts/launch_scp682_consistent_graph_controls_e160_windows.py` | LW | 训练/support |
| 12. Panel d 解释性图渲染 | `paper_final/fig2/scripts/_build_fig2d_interp.R` | L | 画图 |

---

## Fig2e — 泛癌 NMF 磷酸化图谱（atlas）

**需求**：10,023 TCGA 原发瘤；signed-split NMF k=30；30 模块 × 32 癌种活性热图。

| 步骤 | 脚本 | 机器 | category |
|------|------|------|----------|
| 1. TCGA RNA 数据获取 | `src/R/get_tcga_rnaseq.R` | needs_review | 分析/support |
| 2. TCGA 全样本 bulk 预测（portable） | `portable_src/predict_scp682.py` | LUW | 分析 |
| 3. 预测全 TCGA（Fig5 上游，同时产出 atlas 输入） | `04_figures/20260528_fig5_v2/code/22_repredict_tcga_full_scp682_from_raw.py` | LU | 分析/support |
| 4. NMF 运行（k=20/30/40） | `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/run_nmf.py` | L | 分析 |
| 5. 模块分析（ORA 富集、活性矩阵） | `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/analyze_modules.py` | L | 分析 |
| 6. CPTAC 投影验证 | `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/project_to_cptac.py` | L | 分析 |
| 7. 33项目扩展（含LAML） | `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/build_ED7c_33project.py` | L | 分析 |
| 8. 主热图渲染（30模块×32癌种） | `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/build_figB_module_cancer.R` | L | 画图 |
| 9. 带临床注释热图 | `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/build_figA_annotated.R` | L | 画图 |
| 10. 样本概览热图 | `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/build_figA_sample_overview.R` | L | 画图 |
| 11. paper_final 版主热图 | `paper_final/fig2/fig_tcga_pancancer_atlas/scripts/build_figB_module_cancer.R` | L | 画图 |
| 12. paper_final 版带注释热图 | `paper_final/fig2/fig_tcga_pancancer_atlas/scripts/build_figA_annotated.R` | L | 画图 |

---

## Fig2f — NMF 模块生存（Cox，16/30 显著）

| 步骤 | 脚本 | 机器 | category |
|------|------|------|----------|
| 1. NMF 运行（同 Fig2e step 4） | `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/run_nmf.py` | L | 分析 |
| 2. 模块分析（活性矩阵产出） | `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/analyze_modules.py` | L | 分析 |
| 3. Cox + BH-FDR 生存森林图渲染 | `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/build_figD_survival.R` | L | 画图 |

---

## Fig2g — 代表性模块 Hallmark ORA

| 步骤 | 脚本 | 机器 | category |
|------|------|------|----------|
| 1. NMF 运行（同 Fig2e step 4） | `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/run_nmf.py` | L | 分析 |
| 2. 模块分析（富集表产出） | `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/analyze_modules.py` | L | 分析 |
| 3. Hallmark ORA bar 图渲染（9代表模块） | `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/build_figC_pathway_bars.R` | L | 画图 |
| 4. paper_final 版 | `paper_final/fig2/fig_tcga_pancancer_atlas/scripts/build_figC_pathway_bars.R` | L | 画图 |

---

## SCP682_PORTABLE 部署与推理基础设施

| 脚本 | 机器 | category | 说明 |
|------|------|----------|------|
| `portable_src/scp682_graph_runtime.py` | LUW | 训练/engine | 图残差 runtime 引擎（SCP682GraphDecoder + SCP682GraphRuntime） |
| `portable_src/scp682_v4_engine.py` | LUW | 训练/engine | v4 baseline 引擎封装 |
| `portable_src/scripts/export_graph_runtime_state.py` | LU | 分析/support | 从 checkpoint 导出 runtime_state.pt |
| `remote_scripts/package_scp682_exact_scnet_main_release.py` | LU | 分析/support | 打包 canonical 发布包（frozen_release） |
| `03_code/rna2phospho_web/predict_uploaded_bulk.py` | LU | 分析/support | web 服务推理脚本 |
| `03_code/rna2phospho_web/app.py` | LU | 分析/support | FastAPI web 服务主入口 |

---

## 共用图先验审计 & 比较

| 脚本 | 机器 | category |
|------|------|----------|
| `remote_scripts/audit_original_copheemap_local.py` | L | 分析/support |
| `remote_scripts/audit_original_gnn_tables.py` | LU | 分析/support |
| `remote_scripts/inspect_scp682_benchmark_inputs.py` | LU | 分析/support |

---

## 版本漂移说明（为何弃用旧版）

| 脚本模式 | 原因 |
|----------|------|
| `train_cptac_*_dann_locked_v1*.py` | DANN 域对抗系列，非 exact_scnet_gnn 架构，已排除 |
| `train_cptac_parent_residual_kinase_cvae_v*.py` | CVAE v3.x 系列（film_vae 前身），已被 exact_scnet 取代 |
| `SCP682-7~21 / SCP682-23~28` | dev 编号版本，不符合 canonical SCP682-22 |
| `train_scp682_1~4_*.py` / `train_spc682_*.py` | SCP682-1~4 train 系列，legacy 命名模式 |
| `predict_tcga_*_film_vae*.py` | film_vae_z 架构，已被 portable_src 取代 |
| `fit_cptac_phosphosite_oof_stacking*.py` | old stacking 路线（v3.x ridge stacking），已排除 |
| `build_cptac_*_stacking_*.py` | 层叠 stacking 变体，legacy |
| `run_scp682_30*/31*/32*` | dev-30~32 探索分支，未进 canonical |

---

## 缺口（Gap）列表

1. **graph_statistics.tsv 专用生成脚本**：420,102 位点图边 / 21,925 样本图边数字未找到独立输出脚本；数字来源于 process_copheemap_prior_20260428.py 内部计数和 panel_a.R 硬编码。
2. **per_site_spearman_with_deep_learning.tsv 生成链**：build_scp682_bulk_main_panels.py 消费该文件，但原始 merge 逻辑（合并 OOF + 8个基线）分散在多个 launch_* shell 脚本中，无单一入口。
3. **TCGA RNA 全量下载脚本**：用于 Fig2e NMF 输入的 TCGA RNA-seq 批量下载（11,370 样本）未找到专用脚本；可能通过 GDC API 或 cBioPortal 数据集完成，需确认。
4. **DeepGxP 外部 benchmark**：`run_deepgxp_cptac_half_retrain_20260511.py` 覆盖内部基线，外部队列的 DeepGxP baseline 比较来自 `make_external_9model_benchmark.py`，但原始 DeepGxP 外部推理脚本不在 records 中。
5. **NMF 稳健性 ED 图**（k=20/40）：`build_ED7a_stability.R` 和 `build_ED7c_heatmap.R` 存在，但 `build_ED7b_cptac.py`（CPTAC 投影）依赖的 observed_phosphosite.parquet 生成路径已确认（prepare_cptac_pancancer_rna_phosphoproteome_pairs.py），无缺口。
6. **模型合约/SCP682_CURRENT.json 更新**：`package_scp682_exact_scnet_main_release.py` 写入 SCP682_CURRENT.json，但版本号冻结时机需手动确认（RELEASE_NAME=SCP682_main_exact_scnet_gnn_20260522）。
