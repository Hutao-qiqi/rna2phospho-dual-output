# SC 模块复现链（SCP682-SC11 / Fig3）

生成日期：2026-06-16  
canonical 模型：SCP682-SC11（`train_scp682_sc11_expanded_scnet_site_gnn.py`）  
覆盖 panel：Fig3a / Fig3b / Fig3c / Fig3d / Fig3e / Fig3f / Fig3g  

---

## 0. 版本漂移说明（Legacy 排除）

| 版本系列 | canonical 状态 | 排除原因 |
|---|---|---|
| SC1–SC4（Geneformer 系） | legacy | Geneformer backbone，已被 scFoundation 替换 |
| SC5–SC9（过渡版） | legacy | warm-start 来源或探索架构，SC7 仅作为 SC11 的 warm-start 来源，本身不进复现 |
| SC10 | legacy | ScNET site GNN 前身，已被 SC11 扩展图版取代 |
| SC3-BP | legacy | 多域无 QuRIE 旧版 |
| scko3 系 | legacy | drug-operator 旧路线 |
| SC4-delta / SC2p | legacy | 探索路线，非论文模型 |
| scGPT bridge 包 | exploratory | VAE-FiLM 旧架构，film_vae legacy |

---

## 1. Fig3a — SCP682-SC11 架构示意图

### 复现链
```
[无输入数据] 
  → [画图] paper_materials_SCP682_SC11/03_code/visualization/generate_fig1a_scp682_sc_architecture.py (canonical)
  → 04_figure_source_data/fig1_model_architecture/*.svg/*.pdf

[或 R 版]
  → [画图] paper_final/fig3/scripts/panels/panel_a.R (canonical)
  → make_panel_a() ggplot 对象（硬编码 7369节点/882959边/9通路token/56 readout）

[并行：架构示意参数来源]
  → [分析] paper_materials_SCP682_SC11/03_code/preprocessing/export_scp682_main_sc_transfer_prior.py
    (canonical, 训练前置) → scp682_main_sc_transfer_prior_v1（注意力权重+bulk Spearman 表）
```

**无缺口**：架构数值硬编码于 panel_a.R，或由 generate_fig1a 程序绘制。

---

## 2. Fig3b — 内部 5 折重建 + 4 外部队列 per-site 中位 Spearman

### 复现链
```
[数据准备 → 分析]
  1. 下载训练数据：
     - remote_scripts/download_single_cell_phospho_validation_v1.py (support)
     - 03_code/single_cell/download/download_core_geo.ps1 (support)
  2. iccite 数据预处理：
     - 03_code/single_cell/iccite/fetch_iccite_metadata_and_download.ps1 (support)
     - 03_code/single_cell/iccite/export_iccite_matrices.R (support)
     - 03_code/single_cell/iccite/correct_iccite_background_from_mtx.py (support)
  3. SIGNAL-seq 外部验证数据：
     - remote_scripts/prepare_signal_seq_processed_h5ad.py (support)
  4. GSE300551 外部验证数据：
     - remote_scripts/build_gse300551_iccite_plex_inputs.py (support)
  5. Blair / Vivo-seq 数据审计：
     - 03_code/single_cell/data_audit/audit_data_coverage.py (support)
     - 03_code/single_cell/data_audit/build_antibody_and_anchor_audit.py (support)
  6. schema 构建：
     - 03_code/single_cell/schema/summarize_paired_dataset_coverage.py (support)
     - 03_code/single_cell/schema/build_phospho_target_table.py (support)
     - 03_code/single_cell/schema/finalize_iccite_schema.py (support)
     - 03_code/single_cell/schema/patch_blair_prps6_schema.py (support)
     - 03_code/single_cell/schema/update_common_targets_from_anchor.py (support)
     - 03_code/single_cell/preprocessing/prepare_qurie_seq_bjab.py (support)
  7. scFoundation 模型下载：
     - 03_code/single_cell/modeling/download_scfoundation_model.py (support)
  8. scFoundation embedding 计算：
     - 03_code/single_cell/modeling/precompute_scfoundation_embeddings.py (support)
     - 03_code/single_cell/modeling/merge_scfoundation_shards.py (support)
  9. 多域模型输入组装：
     - remote_scripts/build_scfoundation_multidomain_model_input.py (support)

[训练]
  10. bulk teacher prior 导出（SC11 训练必需前置）：
      - paper_materials_SCP682_SC11/03_code/preprocessing/export_scp682_main_sc_transfer_prior.py (canonical)
  11. SC11 正式 5 折训练启动（含 summarize）：
      - paper_materials_SCP682_SC11/03_code/training/run_scp682_sc11_formal_internal_5fold.bat (canonical)
  11a. SC11 核心训练器：
      - paper_materials_SCP682_SC11/03_code/architecture/train_scp682_sc11_expanded_scnet_site_gnn.py (canonical)
      - remote_scripts/train_scp682_sc11_expanded_scnet_site_gnn.py (canonical, 等价副本)
      - remote_scripts/_paper_extract_sources/train_scp682_sc11_expanded_scnet_site_gnn.py (canonical, paper 副本)

[分析]
  12. 内部 5 折结果汇总：
      - paper_materials_SCP682_SC11/03_code/evaluation/summarize_scp682_sc11_internal_5fold.py (canonical)
      - remote_scripts/summarize_scp682_sc11_internal_5fold.py (support)
  13. 外部验证推理：
      - paper_materials_SCP682_SC11/03_code/inference/export_sc11_external_predicted_observed_all.py (canonical)
  14. 外部验证漏斗数据准备：
      - paper_final/fig3/scripts/prep/external_funnel_prep.py (canonical)

[画图]
  15. panel_b.R 渲染：
      - paper_final/fig3/scripts/panels/panel_b.R (canonical)
  16. fig04.R 跨队列 Spearman 矩阵热图：
      - paper_final/fig3/scripts/figures/fig04.R (canonical)
  17. 主渲染入口：
      - paper_final/fig3/scripts/render_biology.R (canonical)
      - paper_final/fig3/scripts/render_all.R (canonical)
```

**缺口**：
- Blair phospho-seq 原始数据预处理脚本（GEO CSV→paired_matrices）未在 canonical/support 中发现专门的 prepare_blair 入口，仅有数据审计工具（audit_data_coverage.py / inspect_blair_csv.py），需确认 paired_matrices 是否已经由上游手工处理或由 remote_scripts 目录中某未收录脚本生成。

---

## 3. Fig3c — 锚点位点跨平台验证（CTNND1 T310 HeLa / STAT3 Y705 跨平台）

### 复现链
```
[分析]
  1. HeLa CTNND1 T310 per-cell 预测导出：
     - remote_scripts/export_sc11_ctnnd1_hela_scatter.py (canonical)
  2. 数据漏斗 + 补充表：
     - paper_final/fig3/scripts/prep/external_funnel_prep.py (canonical)

[画图]
  3. panel_c.R 渲染（CTNND1 hexbin + STAT3 Y705 跨平台柱）：
     - paper_final/fig3/scripts/panels/panel_c.R (canonical)
  4. render_biology.R / render_all.R（同上）
```

**无缺口**。

---

## 4. Fig3d — GSE300551 Benchmark（SCP682-SC vs 6 baselines）

### 复现链
```
[分析 — baseline embedding 计算]
  1. 标准化 h5ad 输入：
     - remote_scripts/build_foundation_model_h5ad_inputs.py (support)
     - 02_results/single_cell/20260531_foundation_model_h5ad_inputs_v1/logs/run_build_h5ad.cmd (support)
  2. 各 foundation model embedding：
     - remote_scripts/precompute_scfoundation_embeddings_multidomain.py (support)
     - remote_scripts/precompute_scgpt_embeddings_multidomain.py (support)
     - remote_scripts/precompute_scimilarity_embeddings_multidomain.py (support)
     - remote_scripts/precompute_tgpt_embeddings_multidomain.py (support)
     - remote_scripts/run_uce_multidomain_embeddings.py (support)
     - remote_scripts/merge_scfoundation_multidomain_shards.py (support)
  3. Geneformer pathway flatten：
     - remote_scripts/prepare_geneformer_pathway_flatten_benchmark_input.py (support)
     - remote_scripts/prepare_transcriptformer_h5ad_inputs.py (support)
     - remote_scripts/run_transcriptformer_multidomain_embeddings.py (support)
  4. embedding 统一格式组装：
     - remote_scripts/assemble_foundation_embeddings_model_input.py (support)
  5. per-site 线性回归基线：
     - remote_scripts/run_foundation_multidomain_persite_linear_regression.py (support)
     - remote_scripts/run_scfoundation_multidomain_persite_ridge.py (support)
     - remote_scripts/run_remaining_foundation_models.cmd (support)
     - remote_scripts/run_transcriptformer_linear_pipeline.cmd (support)
  6. cognate-mRNA Ridge baseline：
     - remote_scripts/run_scp682_sc_raw_expression_selected_gene_ridge.py (canonical)
  7. cell-state Ridge baseline (-1)：
     - 03_code/single_cell/modeling/run_scfoundation_cap12000_model_minus1.ps1 (support)
  8. per-site RidgeCV baseline (model0)：
     - 03_code/single_cell/modeling/run_persite_ridge.py (support)
  9. UCE 单队列：
     - remote_scripts/run_uce_pdo_caf_embedding.cmd (support)
  10. UCE 数据加载器（第三方工具）：
      - remote_scripts/UCE_eval_data.py (thirdparty, 仅供 Fig3d 基线对照)

[分析 — benchmark 汇总]
  11. 审稿人 benchmark 汇总：
      - remote_scripts/assemble_scp682_sc_reviewer_benchmarks_v2.py (support)
      - remote_scripts/build_scp682_sc_reviewer_missing_tables_v2.py (support)
  12. benchmark 擂台数据生成：
      - paper_final/fig3/scripts/prep/benchmark_leaderboard_prep.py (canonical)
  13. win-scatter 数据生成：
      - paper_final/fig3/scripts/prep/winscatter_prep.py (canonical)

[画图]
  → render_biology.R / render_all.R（含 panel_d/panel_e 调用）
```

**无缺口**（UCE thirdparty 已收录）。

---

## 5. Fig3e — per-readout win-scatter（SCP682-SC vs 各 baseline，n=11 readouts）

### 复现链
```
[依赖 Fig3d 相同的 benchmark 数据]
  → paper_final/fig3/scripts/prep/winscatter_prep.py (canonical)
     产出 fig3_benchmark_gse300551_per_readout.tsv / winsummary.tsv

[画图]
  → render_biology.R / render_all.R
```

**无缺口**。

---

## 6. Fig3f — HeLa UMAP 空间连贯性（CTNND1 T310 + 3 transfer sites）

### 复现链
```
[分析]
  1. HeLa UMAP 坐标生成：
     - remote_scripts/export_sc11_hela_umap.py (canonical)
  2. CTNND1 T310 scatter data：
     - remote_scripts/export_sc11_ctnnd1_hela_scatter.py (canonical)

[画图]
  3. panel_e.R (phospho-NMF3 × RNA-NMF01 hexbin)：
     - paper_final/fig3/scripts/panels/panel_e.R (canonical)
  4. make_main_panels.R（UMAP + hexbin + Hallmark + NMF 热图合成）：
     - paper_final/fig3/scripts/make_main_panels.R (canonical)
  5. render_biology.R / render_all.R
```

**缺口**：NDRG1 T346、MAP2K4 S257、PDPK1 S241 三个 transfer-only 位点的 per-cell 预测值导出脚本未在记录中单独出现（export_sc11_ctnnd1_hela_scatter.py 仅明确处理 CTNND1 T310）。Fig3f 可能需要对应脚本或在 export_sc11_external_predicted_observed_all.py 内含括。需确认。

---

## 7. Fig3g — 扩展 ScNET 图消融（site-graph ablation）

### 复现链
```
[训练 — 消融模型]
  1. no-site-graph 消融训练启动：
     - paper_materials_SCP682_SC11/03_code/training/run_scp682_sc11_no_site_graph_matched_ablation.bat (canonical)
  2. no-pathway-attention 消融训练：
     - remote_scripts/run_scp682_sc11_no_pathway_attention_ablation.ps1 (canonical)
     - remote_scripts/train_scp682_sc11_expanded_scnet_site_gnn_no_attention_ablation.py (canonical)
  3. graph edge-source 消融（rewired/无CoPheeMap/无CoPheeKSA/无KSTAR）：
     - remote_scripts/run_scp682_sc11_m4_graph_controls.bat (canonical)

[分析 — 共用工具库]
  4. 审稿人消融表汇总：
     - remote_scripts/assemble_scp682_sc_reviewer_ablation_v2.py (support)
  5. reviewer ED 数据准备（消融配对 Δ 表）：
     - paper_final/fig3/scripts/prep/reviewer_ed_prep.py (canonical)
  6. GSE300551 消融数据：
     - (消融推理由 train script 内部自动产出 per_target TSV)

[画图]
  7. panel_d.R（GSE300551 lollipop）：
     - paper_final/fig3/scripts/panels/panel_d.R (canonical)
  8. fig05.R（残差贡献水平柱）：
     - paper_final/fig3/scripts/figures/fig05.R (canonical)
  9. fig22.R（组件消融配对 Δ，审稿人要求）：
     - paper_final/fig3/scripts/figures/fig22.R (canonical)
  10. render_biology.R / render_all.R
```

**无缺口**。

---

## 8. 共用架构 / 工具库

| 文件 | 角色 |
|---|---|
| 03_code/single_cell/modeling/phospho_model_common.py | SC 共享常量/工具函数（support，训练类） |
| paper_final/fig3/scripts/panels/theme_fig3.R | Fig3 ggplot 主题/调色板（support，画图类） |
| paper_materials_SCP682_SC11/03_code/visualization/generate_scp682_sc11_all_visualizations.py | 全量可视化主入口（support，画图类） |
| paper_materials_SCP682_SC11/03_code/visualization/run_scp682_sc11_validation_nmf.py | 外部验证 NMF 分析（support，分析类） |
| paper_materials_SCP682_SC11/03_code/visualization/run_scp682_sc11_validation_rna_nmf.py | 外部队列 RNA NMF + hallmark（support，分析类） |
| paper_materials_SCP682_SC11/03_code/visualization/build_requested_review_tables.py | 审稿人辅助表（support，分析类） |
| paper_materials_SCP682_SC11/03_code/visualization/compose_fig3_v2.py | Fig3 v2 组合图（canonical，画图类） |
| paper_materials_SCP682_SC11/03_code/generate_scp682_sc11_all_visualizations.py | SC11 候选面板（support，画图类） |
| remote_scripts/_paper_extract_sources/run_scp682_sc11_formal.bat | SC11 正式训练启动（canonical，等价副本） |

---

## 9. 缺口汇总

| 缺口 | 对应 panel | 说明 |
|---|---|---|
| Blair phospho-seq CSV → paired_matrices 格式化脚本 | Fig3b/c/d | 仅有 inspect_blair_csv.py / audit 工具，未见专门 prepare_blair_phospho_multirna.py 类 canonical 脚本 |
| NDRG1/MAP2K4/PDPK1 transfer sites per-cell 导出 | Fig3f | export_sc11_ctnnd1_hela_scatter.py 只明确 CTNND1 T310，其余 3 位点来源需确认（可能在 export_sc11_external_predicted_observed_all.py 内覆盖） |
| Fig3b Fig3c 中 Vivo-seq Th17 数据预处理 | Fig3b/c | inspect_vivo_h5ad.py 为检视工具，未见专门 prepare_vivo_paired_matrices.py 类脚本 |
