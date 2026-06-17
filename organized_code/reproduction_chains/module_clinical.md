# Clinical 模块复现链

模块范围：Fig5（KIRC deep-dive、pS6/RPS6 beyond-parent-mRNA 预后(BH-FDR/联合Cox LRT)、SC 恶性程序、泛癌磷酸化临床图谱）

---

## 总览

Clinical 模块无独立训练步骤。所有预测均使用 canonical bulk SCP682-PORTABLE（exact_scnet_gnn，v4 baseline+0.3*delta）和 canonical SC11 权重做推理；核心链路为：数据准备 → 冻结推理 → 统计分析 → 画图。

---

## Fig5 主复现链

### 链路一：全 TCGA 重预测（Fig5 推理基础）

**数据准备**
- `remote_scripts/download_tcga_cbioportal_clinical_covariates.py` [分析/support] — 从 cBioPortal 批量拉取 TCGA 临床协变量(OS/分期/分级等)，所有 Cox 分析的临床输入

**分析（推理与统计）**
1. `04_figures/20260528_fig5_v2/code/22_repredict_tcga_full_scp682_from_raw.py` [分析/support] — 用 SCP682_PORTABLE canonical 对全 TCGA 重预测，生成预测磷酸化 parquet（Fig5 全部分析的上游输入）
2. `04_figures/20260528_fig5_v2/code/23_rebuild_fig5_support_from_full_tcga.py` [分析/canonical] — 从全 TCGA 重预测重建 Fig5 支撑表（full prediction + baseline + parent-mRNA Cox LRT 分解），**canonical Fig5 主分析脚本**

**画图入口**
- `paper_final/fig5/scripts/prepare_fig5_source_data.py` [分析/canonical] — 汇集各panel上游结果表到 source_data/tables，统一绘图入口
- `paper_final/fig5/scripts/render_fig5_panels.R` [画图/canonical] — 批量渲染 Fig5 全部 panel(a-i)，输出 pdf/png/svg

---

### 链路二：Fig5 泛癌磷酸化临床入口图（Panel a）

**分析**
- `paper_final/fig5/scripts/audit_panel_a_fdr.py` [分析/support] — 审计 BH-FDR 校正前后各 category 计数，验证 flag 定义一致性
- `paper_final/fig5/scripts/panel_a_cancer_entry_data.py` [分析/canonical] — BH-FDR 重定义三种 flag(clinical/beyond-mRNA/graph-residual) 并组织 12行×28列 泛癌入口图数据

**画图**
- `paper_final/fig5/scripts/panel_a_cancer_entry_plot.R` [画图/canonical] — 渲染 Fig5a 泛癌入口图(顶部双层柱+中部数值矩阵+右侧全局总量条)
- `paper_final/fig5/scripts/panels/panel_a.R` [画图/canonical] — 渲染 KIRC predicted pS6 vs OS/Grade/mTOR 状态 raincloud 图及 RPS6 mRNA 对照组(共6子图)

---

### 链路三：Fig5b 通路方向模块热图

**分析**
- `paper_final/fig5/scripts/build_pathway_direction_enrichment.py` [分析/canonical] — 计算 pathway_family × 方向(risk/protective) Fisher 富集，双口径(clinical FDR/beyond parent mRNA FDR)
- `paper_final/fig5/scripts/build_pathway_module_direction.py` [分析/canonical] — 计算 24 pathway_module 方向占比(risk_frac)及 Fisher 富集，产出 panel c 排序轴和代表基因
- `paper_final/fig5/scripts/build_panel_b_module_cancer.py` [分析/canonical] — 计算 24 pathway_module × 28 癌种的 risk-minus-protective 比例矩阵

**画图**
- `paper_final/fig5/scripts/panel_b_module_cancer.R` [画图/canonical] — 渲染 Fig5b 24 pathway_module × 癌种 risk-minus-protective 热图

---

### 链路四：Fig5c 通路方向富集柱图

**画图**
- `paper_final/fig5/scripts/enh_plots.R` [画图/canonical] — 渲染 enh2（Fig5i 前身 panel c 通路方向柱状图）及 enh1(恶性细胞 pS6 vs 血管生成程序)

---

### 链路五：Fig5d pS6 beyond-parent-mRNA 预后（KIRC/CCRCC 核心 KM+Cox）

**分析**
- `03_code/model_validation/rescreen_measured_cptac_site_specific_survival_20260527.py` [分析/support] — 对 CPTAC 实测磷酸化做癌型分层 Cox 生存筛查(RNA/总蛋白双重 null 对照)，产出 Fig5 pS6 候选位点
- `03_code/model_validation/run_tcga_existing_predicted_site_survival_concordance_20260527.py` [分析/support] — 验证 CPTAC 实测位点生存信号在 SCP682 TCGA 预测磷酸化中的一致性
- `03_code/model_validation/run_tcga_predicted_site_survival_concordance_20260527.py` [分析/support] — 在 SCP682 TCGA 预测磷酸化上运行癌症特异性位点 Cox 生存筛查并与 RNA/总蛋白 null 对照（Fig5 KIRC pS6 BH-FDR 主线）

**画图**
- `paper_final/fig5/scripts/panels/panel_b.R` [画图/canonical] — 渲染 KIRC predicted pS6 KM 生存曲线和 mRNA 校正 Cox forest 图（Fig5d）
- `paper_final/fig5/scripts/panels/panel_c.R` [画图/canonical] — 渲染 CPTAC ccRCC predicted vs measured pS6 散点图（Fig5c）
- `paper_final/fig5/scripts/panels/panel_d.R` [画图/canonical] — 渲染 CPTAC ccRCC predicted pS6 与 mTOR 轴磷酸化偏相关气泡图（Fig5d）
- `paper_final/fig5/scripts/render_fig5c_split.R` [画图/canonical] — 将 Fig5 panel d(KM+Cox) 拆成两张独立图输出

---

### 链路六：Fig5e beyond-parent-mRNA 瀑布图

**画图**
- `paper_final/fig5/scripts/panels/panel_e.R` [画图/canonical] — 渲染 KIRC predicted pS6 beyond parent mRNA 残差瀑布图，附 OS 状态颜色条，Fisher 检验标注（Fig5e）

---

### 链路七：Fig5f DepMap ccRCC mTOR 药物相关

**画图**
- `paper_final/fig5/scripts/panels/panel_f.R` [画图/canonical] — 渲染 DepMap ccRCC 细胞系中 predicted pS6 vs mTOR 抑制剂 AUC Spearman 相关图（Fig5f）

---

### 链路八：Fig5g/h 全位点模块 × 癌种热图 & signed PTM-SEA

**分析**
- `paper_final/fig5/scripts/panel_i_full_phospho_specific_modules_data.py` [分析/canonical] — 对全 TCGA SCP682 预测做 FastICA 提取磷酸化特异性模块，并做 PTMSigDB 富集和 Cox 生存（需 Linux 服务器）
- `paper_final/fig5/scripts/panel_i_program_specificity_data.py` [分析/canonical] — 将磷酸化特异性 ICA 模块折叠到 program 级别，计算各 program 特异性指标

**画图**
- `paper_final/fig5/scripts/panels/panel_g_all_site_modules.R` [画图/canonical] — 渲染全癌种位点模块 × 癌种方向热图（Fig5g）
- `paper_final/fig5/scripts/panels/panel_h_signed_ptmsea.R` [画图/canonical] — 渲染 signed PTM-SEA 通路富集分数 × 癌种热图（Fig5h）

---

### 链路九：Fig5i 癌症特异磷酸化位点聚类（含 KEGG 富集）

**分析**
- `paper_final/fig5/scripts/panel_i_site_specificity_kegg_data.py` [分析/canonical] — 对 cancer-specific 磷酸化位点做层次聚类(14 簇)并做 KEGG 超几何富集

**画图**
- `paper_final/fig5/scripts/panels/panel_i_representative_sites.R` [画图/canonical] — 渲染 Fig5i 癌症特异磷酸化位点 14 聚类 × 癌种热图叠加 KEGG 富集热图

---

### 链路十：pS6/RPS6 专项 Panel（KIRC 多角度绘图）

**分析（仅保留 canonical 与 support）**
- `03_code/model_validation/plot_rps6_ps6_bubble_correlation_20260530.py` [画图/canonical] — 绘制 CCRCC RPS6 S235/S236 pS6 与 mTOR 轴磷酸位点气泡相关矩阵图
- `03_code/model_validation/plot_rps6_ps6_fig5_anchor_20260530.py` [画图/canonical] — 绘制 Fig5 pS6-RPS6 beyond-parent-mRNA 核心锚定面板（预测 vs 实测 vs mRNA 散点+KM）
- `03_code/model_validation/plot_rps6_ps6_mtor_phospho_raincloud_ridge_20260531.py` [画图/canonical] — 绘制 CCRCC mTOR 轴磷酸化状态分组 raincloud/ridge 图
- `03_code/model_validation/plot_rps6_ps6_mtor_phospho_state_20260530.py` [画图/canonical] — 绘制 CCRCC RPS6 pS6 高低组在 mTOR 轴磷酸位点上的分布状态图
- `03_code/model_validation/plot_rps6_ps6_mutation_ridgeline_20260530.py` [画图/canonical] — 绘制 TCGA-KIRC pS6 按 mTOR 通路突变分层 ridgeline 图
- `03_code/model_validation/plot_rps6_ps6_patient_slope_20260530.py` [画图/canonical] — 绘制 CCRCC 患者级别预测 pS6 vs 实测 pS6 斜率散点图
- `03_code/model_validation/plot_rps6_ps6_raincloud_20260530.py` [画图/canonical] — 绘制 TCGA-KIRC predicted pS6 按肿瘤分级/分期分层 raincloud 图
- `03_code/model_validation/plot_rps6_ps6_waterfall_20260530.py` [画图/canonical] — 绘制 TCGA-KIRC predicted pS6 与 RPS6 mRNA z-score waterfall 排列对比图
- `03_code/model_validation/plot_tcga_predicted_rps6_mtor_phospho_distributions_20260531.py` [画图/canonical] — 绘制 TCGA-KIRC mTOR 通路相关磷酸位点预测分布多面板图
- `03_code/model_validation/plot_tcga_rps6_ps6_combined_raincloud_20260531.py` [画图/canonical] — 绘制 TCGA-KIRC combined raincloud 图（predicted pS6 + mTOR 突变分层）

---

### 链路十一：KIRC 单细胞恶性程序（Fig5 SC 部分）

**分析（推理 + 验证）**
- `remote_scripts/run_kirc_rps6_validation_full.ps1` [分析/support] — 在 KIRC GSE242299 队列上运行 SC11 对 RPS6 的完整验证（PowerShell 启动器）
- `03_code/single_cell/modeling/run_kirc_rps6_validation.py` [分析/support] — 验证 SCP682-SC 预测 pRPS6 与 KIRC 单细胞 mTOR/增殖/缺氧通路 score 的空间相关性
- `03_code/single_cell/modeling/run_kirc_rps6_clinicopathology.py` [分析/support] — 提取 KIRC 患者 pT 分期/坏死/肿瘤大小并与预测 pS6 均值做 Mann-Whitney 关联

**画图**
- `remote_scripts/plot_kirc_rps6_umap_malignancy.py` [画图/support] — 绘制 KIRC 单细胞 UMAP（恶性/非恶性标注 + 预测 RPS6 pS235/S236），Fig5 SC 恶性程序面板
- `paper_materials_SCP682_SC11/04_figure_source_data/kirc_rps6_single_cell_validation_v1/scripts/plot_kirc_rps6_malignant_meantest.R` [画图/canonical] — KIRC 单细胞 RPS6 预测值按恶性/非恶性/健康分组 violin+患者配对斜线+效应量森林图

---

### 链路十二：共享样式与辅助

- `paper_final/fig5/scripts/panels/panel_common.R` [画图/support] — Fig5 所有 panel 共享常量(颜色/theme_fig5/save_panel/helper 函数)
- `paper_final/fig5/scripts/panels/panel_fig2_style.R` [画图/support] — 从 Fig2 复制视觉语法到 Fig5 的共享样式常量
- `remote_scripts/run_scp682_fig5_v2_exports_20260528.py` [分析/canonical] — 利用冻结 SCP682 PORTABLE 对 TCGA-KIRC/LUAD/PAAD 推理，导出 Fig5 所需 pS6 等靶点预测-观测对照数据
- `remote_scripts/run_scp682_tcga_tcpa_overlap_20260526.py` [分析/support] — 分析 TCGA-TCPA 样本 RNA-RPPA 匹配度并对 TCPA RPPA 抗体做磷酸位点映射

---

## GAP（缺环）列表

1. **Fig5f DepMap 数据准备脚本缺失**：`panel_f_depmap_rcc_mtor_drug_correlations.tsv` 的生成脚本（DepMap ccRCC AUC 提取）在 records 中未出现对应 canonical/support 条目，panel_f.R 直接读取预计算 TSV。
2. **Fig5h signed PTM-SEA 数据准备脚本缺失**：`fig5d_ptmsea_signature_heatmap_matrix.tsv` 的产出脚本未在 records 中出现，panel_h 直接读取该 TSV。
3. **Fig5g all_site_module 数据准备脚本缺失**：`fig5b_all_site_module_risk_minus_protective_fraction.tsv` 等数据源的生成脚本不在 records 中，可能与 build_panel_b_module_cancer.py 产出有重叠但未明确列出。
4. **KIRC SC h5ad 下载/预处理脚本未登记**：GSE242299 数据集的下载和预处理步骤缺乏明确的 records 条目（仅有启动脚本 run_kirc_rps6_validation_full.ps1）。
5. **CGGA 临床验证链路已完全作废**：records 中所有 cgga/* 脚本均为 legacy（依赖旧版 CVAE/MLP 模型），不属于 canonical 复现集，不存在可用替代。
6. **panel_i_full_phospho_specific_modules_data.py 需 Linux 路径**：records 明确标注需要服务器运行（TCGA Linux 路径硬编码），本地复现时需特殊处理。
