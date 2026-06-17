# Fig 5 复现链对抗式核验报告

生成日期：2026-06-16 | 审查范围：Fig5 全 9 panel（a–i）| 审查方法：逐脚本代码阅读 + Grep 文件名印证

---

## 总体判断

| Panel | 状态 | 一句话摘要 |
|---|---|---|
| a（泛癌入口图） | 黄 | FDR 重定义逻辑正确，但 r08–r10 模块计数仍用 nominal-p 构建的 NMF 输入；旧 `fig5a_tcga_cancer_effect_classification.tsv` 作为排序/筛癌骨架但未见其生成脚本 |
| b（通路模块热图） | 黄 | `build_panel_b_module_cancer.py` 链路完整，但热图矩阵的数据直接来自 `23_rebuild` 的 nominal-p flag 产物（`FOUR_CANCERS` 硬编码 LGG/KIRC/SARC/LIHC），未经 BH 重定义 |
| c（通路方向富集柱图） | 黄 | `build_pathway_direction_enrichment.py` / `build_pathway_module_direction.py` 均已有双口径（clinical FDR / beyond parent mRNA FDR），但 `enh_plots.R` 读的是哪张表、是否已对接 BH-q 口径未能在本次 grep 中验证 |
| d（KIRC KM + Cox Forest） | 绿 | `panel_b.R`（对应论文 Fig5d）明确读取 `panel_b_tcga_kirc_rps6_cox_forest.tsv`；`run_tcga_predicted_site_survival_concordance_20260527.py` 内调用 `false_discovery_control(method="bh")`；beyond-parent 用联合 Cox LRT，链路完整 |
| e（site-over-parent 瀑布图） | 绿 | `panel_e.R` 读 `panel_e_tcga_kirc_site_over_parent_waterfall.tsv`，数据由 `prepare_fig5_source_data.py` 从 `20260530_fig5_rps6_ps6_panel_v1/tables/tcga_kirc_rps6_ps6_site_over_parent_waterfall_data.tsv` 复制；但生成该 TSV 的分析脚本不在 canonical 集内（GAP） |
| f（DepMap mTORi AUC） | 红 | `panel_f.R` 直接读取预计算 TSV `panel_f_depmap_rcc_mtor_drug_correlations.tsv`；生成该文件的 DepMap 提取脚本完全缺失（已在 module_clinical.md GAP #1 标注），无法追溯 Spearman p 口径与样本选择 |
| g（全位点模块 × 癌种热图） | 红 | `panel_g_all_site_modules.R` 读三张 `fig5b_all_site_module_*.tsv`；生成这些文件的脚本不在 canonical 集（module_clinical.md GAP #3 可能部分对应 `build_panel_b_module_cancer.py`，但未明确）；且文件名前缀 `fig5b_` 而 R 脚本是 panel_g，命名体系错位 |
| h（signed PTM-SEA 热图） | 红 | `panel_h_signed_ptmsea.R` 读 `fig5d_ptmsea_signature_heatmap_matrix.tsv`；无任何 canonical 脚本产出此文件（module_clinical.md GAP #2），PTM-SEA 运行参数/数据库版本不可追溯 |
| i（SC 恶性程序 / KIRC pS6 vs angiogenesis） | 黄 | `plot_kirc_rps6_malignant_meantest.R` 读 `kirc_cell_rps6_prediction.tsv`；上游 `run_kirc_rps6_validation.py` 链路完整（SC11 推理 → 细胞表）；但 GSE242299 h5ad 下载/预处理脚本缺失（GAP #4），malignant 标注逻辑依赖启发式规则（epithelial+tumor_tissue 推断），非显式 InferCNV/CopyKAT 标签 |

---

## Panel a — 泛癌入口图（BH-FDR 153.3k → 91.5k）

**状态：黄（多重检验口径已修正但局部残留 nominal-p）**

**链路：**
`23_rebuild_fig5_support_from_full_tcga.py`（产出 `full_tcga_scp682_main_architecture_effect_matrix.tsv.gz` 以及三列 BH-q：`cox_q_full_bh` / `add_site_to_parent_mrna_lrt_q_bh` / `gain_q_bh`）→ `panel_a_cancer_entry_data.py`（读入架构矩阵，原地重定义三个 flag 为 BH-q < 0.05）→ `panel_a_cancer_entry_plot.R`（渲染）

**代码印证：**
`panel_a_cancer_entry_data.py` 第 76–79 行明确注释"BH-FDR 重定义"并调用 `cox_q_full_bh` / `add_site_to_parent_mrna_lrt_q_bh` / `gain_q_bh`；audit json 字段 `"multiple_testing": "BH-FDR q<0.05"` 可作为审计留存。

**疑点/缺口：**
1. **模块计数行（r08–r10）未经 BH 重定义**：读入的 `modules` 来自 `full_tcga_scp682_main_residual_module_nmf_input.tsv`，而该 NMF 输入本身由 `23_rebuild` 中 nominal-p `clinical_significant` flag 筛选的 `module_selected_for_fig5`（第 769 行）生成，阈值 `module_threshold = 0.10` 是硬编码 NMF 归一化比例而非 BH-q。因此柱图中 r08–r10 的计数基于 nominal-p 位点集，与 r02–r07（BH-q）口径不一致。
2. **`fig5a_tcga_cancer_effect_classification.tsv`（旧表）作为癌种排序和筛选骨架**：`panel_a_cancer_entry_data.py` 第 54–85 行读取此表并从中取 `cancers` 列表 + `classification_score` 排序；但该文件不在 organized_code 任何 canonical 脚本的输出路径中，生成脚本缺失。
3. **parent_mrna_clinical_genes 行（r01）保持 nominal p < 0.05**：audit json 中有明确备注，但图表中此行与其他 BH-q 行并排展示，需在图注中特别说明口径差异。
4. **`panel_a.R`（panels/panel_a.R）与 `panel_a_cancer_entry_plot.R` 并存**：前者渲染 KIRC raincloud（6 子图），后者渲染 12×28 入口矩阵；模块归属链路中命名重叠，可能导致调用方混淆。

---

## Panel b — 通路方向模块热图（4 癌种 pathway_module × risk/protective）

**状态：黄（分析链完整，但统计口径为 nominal-p 过滤后的子集）**

**链路：**
`23_rebuild_fig5_support_from_full_tcga.py`（`build_fig5b_module_tables` 函数，第 866 行，用 `clinical_significant & (graph_residual_significant | parent_mrna_independent)` 过滤，均为 nominal-p flag）→ 产出 `full_tcga_scp682_main_fig5b_pathway_module_cox_heatmap_matrix.tsv` → `build_panel_b_module_cancer.py`（读后计算 24 module × 28 癌种比例矩阵）→ `panel_b_module_cancer.R`

**代码印证：**
`23_rebuild` 第 866–943 行 `build_fig5b_module_tables` 函数中的筛选条件明确使用 `clinical_significant`（nominal p < 0.05）而非 `cox_q_full_bh`；四癌种硬编码为 `FOUR_CANCERS = ["TCGA-LGG", "TCGA-KIRC", "TCGA-SARC", "TCGA-LIHC"]`（第 46 行）。

**疑点/缺口：**
1. **nominal-p 口径风险**：`build_fig5b_module_tables` 没有重新读取 BH-q 列做过滤，Fig5b 的位点集基于 nominal p < 0.05，在 ~153k 位点规模下存在大量假阳性。与 Fig5a 的 BH-q 重定义不一致。
2. **四癌种选择理由无代码依据**：LGG / KIRC / SARC / LIHC 硬编码，无驱动该选择的统计脚本（如挑选 top clinical-fraction 或 top pathway-score 癌种的脚本）。
3. **`build_pathway_module_direction.py` 产出的 24 module 是否与 `build_panel_b_module_cancer.py` 中的 module 列表完全对齐**：两脚本独立运行，未见明确的 join 或 key 对齐验证步骤。

---

## Panel c — 通路方向富集柱图（增殖 RISK / mTOR PROTECTIVE）

**状态：黄（双口径分析脚本存在，但绘图脚本实际读入表未在 canonical 链中验证）**

**链路：**
`build_pathway_direction_enrichment.py`（Fisher 富集，双口径：clinical FDR / beyond parent mRNA FDR）+ `build_pathway_module_direction.py`（24 module risk_frac + 代表基因）→ `enh_plots.R`（渲染）

**代码印证：**
`module_clinical.md` 链路三注释"双口径 (clinical FDR/beyond parent mRNA FDR)"，但实际 `enh_plots.R` 读取的表文件名在本次审查 grep 中未被确认（无法看到 `enh_plots.R` 的 read 语句是对接 FDR 版还是 nominal-p 版产出）。

**疑点/缺口：**
1. **`enh_plots.R` 的输入文件名未在本次审查中被 grep 到**：无法确认 `enh_plots.R` 读取的是 `build_pathway_direction_enrichment.py` 的 BH-FDR 口径产出还是 nominal-p 旧版。
2. **Fig5c 在 module_clinical.md 中归入链路四（画图只列 `enh_plots.R`）**：分析脚本与画图脚本之间的中间表文件名未显式列出，链路断点在分析→画图交界处。
3. **`enh2` 与 `enh1` 混在同一 R 文件**：`enh_plots.R` 描述"渲染 enh2（Fig5i 前身 panel c 通路方向柱状图）及 enh1（恶性细胞 pS6 vs 血管生成程序）"，版本历史标注"前身"意味着该脚本可能是过渡版本，其输出是否等同于最终 Fig5c 需确认。

---

## Panel d — KIRC pS6 KM 曲线 + mRNA 校正 Cox Forest

**状态：绿（链路最完整，统计口径正确）**

**链路：**
`rescreen_measured_cptac_site_specific_survival_20260527.py`（CPTAC 实测磷酸化 Cox 筛查，RNA/总蛋白双 null）→ `run_tcga_predicted_site_survival_concordance_20260527.py`（TCGA 预测磷酸化 Cox 一致性，含 `false_discovery_control(method="bh")`）→ 产出 `20260530_fig5_exact_site_anchor_search_v1` 目录内的 `tcga_kirc_rps6_s235_s236_sample_table.tsv.gz` / `tcga_kirc_rps6_s235_s236_parent_mrna_cox.tsv` → `prepare_fig5_source_data.py`（重命名复制为 `panel_b_tcga_kirc_rps6_survival_samples.tsv` / `panel_b_tcga_kirc_rps6_cox_forest.tsv`）→ `panel_b.R`（渲染 KM + forest）

**代码印证：**
- `run_tcga_predicted_site_survival_concordance_20260527.py` 第 19 行 `from scipy.stats import false_discovery_control`，第 848 行 `false_discovery_control(..., method="bh")`。
- `panel_b.R` 第 31 行读取 `panel_b_tcga_kirc_rps6_survival_samples.tsv`，第 77–78 行读取 `panel_b_tcga_kirc_rps6_cox_forest.tsv`；forest 子图 subtitle 硬编码 `LRT p=8.26e-7`（需与 Cox 脚本实际输出对齐验证）。
- beyond-parent 口径：`cox_bivariate_add_batch`（联合 Cox 双变量 LRT，df=1，scipy.stats.chi2）在 `23_rebuild` 中已实现，与论文声明的"联合 Cox 增量 LRT"一致。

**疑点/缺口：**
1. **forest 图的 LRT p 硬编码 `8.26e-7`**：该数字直接写在 `panel_b.R` subtitle 而非从数据表动态读取，若重跑 TCGA 样本集变化时不会自动更新，是数字与代码脱钩的脆弱点。
2. **`20260530_fig5_exact_site_anchor_search_v1` 目录的生成脚本不在 organized_code**：`prepare_fig5_source_data.py` 读取该目录，但产出该目录的分析脚本（与 `run_tcga_predicted_site_survival_concordance_20260527.py` 有别的一个更下游脚本）未被识别为 canonical 条目，路径 `02_results/model_validation/20260530_fig5_exact_site_anchor_search_v1` 不等于 `20260527` 脚本的默认输出。

---

## Panel e — site-over-parent 残差瀑布图

**状态：绿（画图链路完整；但上游分析脚本缺失）**

**链路：**
（缺失）→ `02_results/model_validation/20260530_fig5_rps6_ps6_panel_v1/tables/tcga_kirc_rps6_ps6_site_over_parent_waterfall_data.tsv` → `prepare_fig5_source_data.py`（重命名复制）→ `panel_e.R`

**代码印证：**
- `prepare_fig5_source_data.py` 第 60–62 行明确从 `RES / "tables/tcga_kirc_rps6_ps6_site_over_parent_waterfall_data.tsv"` 复制到 `panel_e_tcga_kirc_site_over_parent_waterfall.tsv`。
- `panel_e.R` 第 5 行读取该文件，计算每患者 `site_over_parent_residual`（预测 pS6 对 RPS6 mRNA 残差）；Fisher 检验在 R 中现场计算（四分位极端组对比），无预计算 p 值硬编码。

**疑点/缺口：**
1. **`tcga_kirc_rps6_ps6_site_over_parent_waterfall_data.tsv` 生成脚本缺失**：`20260530_fig5_rps6_ps6_panel_v1` 目录产出脚本不在 organized_code 任何 canonical 条目中，该残差是如何计算的（线性模型残差？分位数回归？）、每患者汇聚逻辑（中位数还是均值？）无法核实。
2. **`panel_e.R` 的 Fisher 检验**：使用顶/底四分位患者（各 n/4 人）的事件数 2×2 表，口径合理，但四分位分割是 R 内现场计算，与上游是否有一致的临界值对齐未见验证。

---

## Panel f — DepMap ccRCC mTORi AUC（Spearman 相关）

**状态：红（上游数据准备脚本完全缺失）**

**链路：**
（完全缺失）→ `panel_f_depmap_rcc_mtor_drug_correlations.tsv` → `panel_f.R`

**代码印证：**
- `panel_f.R` 第 5 行直接读取 `panel_f_depmap_rcc_mtor_drug_correlations.tsv`；`prepare_fig5_source_data.py` 第 64–65 行将同名表从 `RES/tables/panel_c_plotted_drug_correlations.tsv` 复制（**注意：源文件名为 `panel_c_plotted_drug_correlations.tsv`，与目标 `panel_f_` 命名不一致**，存在重命名歧义）。
- 无任何 canonical 脚本涵盖 DepMap 数据下载（PortalDB / PRISM AUC）、ccRCC 细胞系筛选、predicted pS6 提取与 AUC Spearman 相关计算。

**疑点/缺口：**
1. **数据准备脚本全链缺失**：DepMap PRISM AUC 版本、ccRCC 细胞系定义（哪些算 ccRCC / RCC？）、mTORi 药物列表筛选方式均不可追溯。
2. **源文件名 `panel_c_plotted_drug_correlations.tsv` 与目标 `panel_f_depmap_rcc_mtor_drug_correlations.tsv`**：prepare 脚本中源路径以 `panel_c_` 前缀命名（历史上此 panel 曾是 panel c），表明该文件是版本迁移遗产，需确认内容与最终 Fig5f 完全对应。
3. **`panel_f.R` subtitle 称 "AUC lower value means higher sensitivity"**：AUC 方向定义（是 area-under-dose-response 还是 viability-AUC？）无分析脚本可确认，仅依赖 TSV 内数值方向。

---

## Panel g — 全位点模块 × 癌种方向热图

**状态：红（三张源数据表的生成脚本缺失；文件名命名体系错位）**

**链路：**
（缺失）→ `fig5b_all_site_module_risk_minus_protective_fraction.tsv` + `fig5b_all_site_module_significance_density.tsv` + `fig5b_all_site_module_summary.tsv` + `fig5b_all_site_module_cancer_order.tsv` → `panel_g_all_site_modules.R`

**代码印证：**
- `panel_g_all_site_modules.R` 第 4–8 行明确读取上述四张文件。
- module_clinical.md GAP #3 说"可能与 `build_panel_b_module_cancer.py` 产出有重叠但未明确列出"——经 grep 确认：`build_panel_b_module_cancer.py` 输出的是 `full_tcga_scp682_main_fig5b_pathway_module_*.tsv`，而 R 脚本读的是 `fig5b_all_site_module_*.tsv`，**文件名不同，不能确认为同一产物**。

**疑点/缺口：**
1. **四张 `fig5b_all_site_module_*.tsv` 的生成脚本完全缺失**：链路断点在分析→画图之间，无法核实模块定义、癌种排序依据、risk-minus-protective fraction 的计算方式。
2. **`fig5b_` 前缀而 R 文件是 `panel_g`**：表明文件是在 panel 序号重新编排之前生成的，版本冻结与最终图序是否对应需要人工确认。
3. **模块粒度**：不清楚此处的"位点模块"是 ICA 组件（`panel_i_full_phospho_specific_modules_data.py` 的产物）、NMF 成分、还是 `build_panel_b_module_cancer.py` 的 pathway_module，无脚本可核实。

---

## Panel h — signed PTM-SEA 通路富集热图

**状态：红（上游 PTM-SEA 运行脚本及数据库配置完全缺失）**

**链路：**
（缺失）→ `fig5d_ptmsea_signature_heatmap_matrix.tsv` + `fig5d_ptmsea_signature_heatmap_rows.tsv` → `panel_h_signed_ptmsea.R`

**代码印证：**
- `panel_h_signed_ptmsea.R` 第 4–5 行明确读取两张 TSV。
- module_clinical.md GAP #2 确认脚本缺失，且文件名前缀 `fig5d_` （历史 panel d）与当前 panel h 错位。
- `panel_i_full_phospho_specific_modules_data.py` 可通过 FastICA + PTMSigDB 富集产出类似数据，但其默认输出目录为 `20260531_panel_i_phospho_specific_modules_full_v1/tables`，与 R 脚本读取的文件名不匹配。

**疑点/缺口：**
1. **PTMSigDB 版本不可追溯**：脚本中有 `DEFAULT_PTMSIGDB_GMT` 路径指向 Linux 服务器 `ptm.sig.db.all.flanking.human.v2.0.0.gmt`（v2.0.0），但该路径是 `panel_i` 脚本的配置，不一定等同于 `fig5d_ptmsea` 文件的产出条件。
2. **"signed" PTM-SEA 的具体打分函数**：标准 PTM-SEA 不是 signed，产出 signed ES 需要特定参数配置，但实际运行参数无脚本可查。
3. **`fig5d_` 文件名残留**：若 panel 序号在最终定稿时重新排过，需确认 TSV 内容对应的是当前 panel h 所展示的图，而非历史版本的 panel d。

---

## Panel i — SC 恶性细胞 pS6 + pS6 vs 血管生成程序

**状态：黄（SC 推理链路完整；缺 GSE242299 数据准备脚本和显式恶性标注）**

**链路：**
（缺失：GSE242299 h5ad 下载/预处理）→ `run_kirc_rps6_validation.py`（SC11 推理 + pathway score + 恶性推断）→ 产出 `kirc_cell_rps6_prediction.tsv` / `kirc_rps6_tumor_healthy_mean_tests.tsv` → `plot_kirc_rps6_malignant_meantest.R`（三子图：细胞级分布 / 患者配对 / 效应量 forest）

**代码印证：**
- `run_kirc_rps6_validation.py` 第 566 行调用 SC11 checkpoint 产出 `predicted_RPS6_pS235_S236`；第 280–282 行推断 malignant 标签（epithelial+tumor_tissue → `malignant_inferred`）。
- `plot_kirc_rps6_malignant_meantest.R` 读取预计算 `kirc_rps6_tumor_healthy_mean_tests.tsv` 中的 Welch t 检验 p 值，患者配对 t 检验在 R 内现场重算并打印到 log（双重确认）。

**疑点/缺口：**
1. **GSE242299 h5ad 下载/预处理脚本缺失**：启动器 `run_kirc_rps6_validation_full.ps1` 只是 PowerShell wrapper，真正的 GEO 下载和 Cell Ranger / Seurat 预处理代码不在 organized_code 中。数据质控标准（过滤 nCounts/nGenes 阈值、doublet 去除）不可核实。
2. **恶性标注依赖启发式规则**：`run_kirc_rps6_validation.py` 中 malignant_status 通过 cell_type 含 "epithelial/malignant" 且 tissue 含 "tumour/kirc" 推断（第 288–292 行），而非基于 InferCNV/CopyKAT 的 CNV 证据。若 GSE242299 h5ad 的 obs 字段中原本就有 malignant 注释（`malignant_col` 非空），则优先使用原注释，但该路径的实际结果未见验证日志。
3. **患者数硬编码 `n=8`**：`plot_kirc_rps6_malignant_meantest.R` 第 116 行 `lab = paste0("paired ", pf(p_paired_all), "  (n=8)")`，n=8 硬编码而非动态统计，若数据集变化则标注会静默错误。
4. **pS6 vs angiogenesis 程序**（module_clinical.md 称 `enh1`）：在 `enh_plots.R` 中渲染，但与 SC 程序的关系（是单细胞层面 angiogenesis score 还是 bulk 预测值关联？）未在链路文档中明确，且 `enh_plots.R` 的输入表来源如 panel c 所述未得到验证。

---

## 跨 panel 共性问题

1. **`20260530_fig5_exact_site_anchor_search_v1` 目录产出脚本不在 canonical 集**：panels b（KM/forest 数据）、c（CPTAC 实测散点）、e（瀑布图数据）均依赖此目录，但产出该目录的分析脚本未被识别并收录，是整个 pS6 beyond-parent 核心 panels 的共性缺口。
2. **文件名版本漂移**：`fig5b_`、`fig5d_`、`panel_c_plotted_` 等历史前缀残留在当前画图输入文件名中，表明 panel 序号在定稿时经历至少一次重排，但文件内容是否同步更新无脚本可证。
3. **`prepare_fig5_source_data.py` 充当"胶水层"**：该脚本是 panels b/c/e/f 的最终数据复制入口，但它只做 copy/rename，不做任何验证或转换；若上游文件名或路径变化，该脚本会静默通过，无断言保护。
4. **verify 阶段未跑**（REPRODUCE.md §7 明确）：所有链路均为多智能体综合梳理，未经端到端执行验证。

---

## 建议优先级

| 优先级 | 行动 |
|---|---|
| P0 | 补齐 `20260530_fig5_exact_site_anchor_search_v1` 目录的产出脚本（影响 panels d/e 的数据可复现性） |
| P0 | 补齐 DepMap 数据提取脚本（panel f 完全无法追溯） |
| P0 | 补齐 `fig5b_all_site_module_*.tsv` 和 `fig5d_ptmsea_*.tsv` 的生成脚本（panels g/h 完全无法追溯） |
| P1 | 将 `build_fig5b_module_tables` 中的 flag 过滤改为 BH-q 口径，或在图注中明确 panel b 使用 nominal-p 子集 |
| P1 | 去除 panel d（`panel_b.R`）中的硬编码 `LRT p=8.26e-7`，改为从 `panel_b_tcga_kirc_rps6_cox_forest.tsv` 动态读取 |
| P2 | 补齐 GSE242299 下载/预处理脚本，或在 REPRODUCE.md 中注明数据获取方式（GEO 直接下载 + 已预处理 h5ad） |
| P2 | 统一文件名命名体系（清除 `fig5b_`/`fig5d_`/`panel_c_` 历史前缀歧义） |
