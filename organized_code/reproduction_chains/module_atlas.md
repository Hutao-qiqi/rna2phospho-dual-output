# Atlas 模块复现链 — Fig2e / Fig2f / Fig2g / ED7a-c

生成日期：2026-06-16

---

## 范围说明

Atlas = TCGA 泛癌 signed-split NMF k=30 磷酸化图谱，对应 Fig2e（30模块×32癌种热图）、Fig2f（生存Cox森林图）、Fig2g（Hallmark ORA条形图）及 ED7a（k稳健性）/ ED7b（CPTAC投影验证）/ ED7c（33项目含LAML）。

上游依赖 bulk SCP682-22/PORTABLE v4 canonical 权重（训练链在 bulk 模块中，本模块仅引用其推理输出）。

---

## 复现链

### 步骤 0：TCGA 全样本 SCP682 推理（数据准备，由 bulk 模块提供）

**缺口（GAP-0）**：records 中未找到独立的"对全 TCGA 11,370 例转录组批量推理、生成 `tcga_full_scp682_predicted_phosphosite.parquet` 和 `X_nmf.npy`"的专属 atlas 分析脚本。

- `run_nmf.py` 的 inputs 字段声明其读取 `results/X_nmf.npy (10023样本x37184特征)`，说明这个矩阵在 NMF 之前已存在。
- `04_figures/20260528_fig5_v2/code/22_repredict_tcga_full_scp682_from_raw.py`（module=clinical, canonical=support, Fig5）可重产 TCGA 全样本预测 parquet，是已知的 TCGA 全样本推理路径，但挂在 clinical 模块下，并非 atlas 专属。
- `SCP682_PORTABLE/v4_engine/code/predict_scp682_v4_0_public_bulk_20260508.py`（module=bulk, canonical=canonical, Fig2c）可对任意 bulk RNA 队列推理，可作为 TCGA atlas 推理的执行器，但 records 目标为 Fig2c（外部验证），无 TCGA atlas 专属入口脚本。
- **结论**：TCGA atlas 推理脚本（生成 X_nmf.npy/parquet 的环节）**在 atlas 模块 records 中存在缺口**，需手动核查 `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/` 目录根部是否有 `prepare_*.py` 或 `build_*.py`，或由 paper_final 版中未记录的脚本产出。

---

### 步骤 1：NMF 运算（分析）

| rel | category | machines | canonical | produces | purpose |
|---|---|---|---|---|---|
| `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/run_nmf.py` | 分析 | L | canonical | W_k30/20/40.npy, H_k30.npy, nmf_summary.json | 对 10023 样本×37184 signed-split 特征矩阵运行 NMF k=30（主）+ k=20/40（稳健性） |

**版本说明**：`paper_final/fig2/fig_tcga_pancancer_atlas/scripts/run_nmf.py` 磁盘上存在且内容与 02_results 版相同（records 未单独登记），以 02_results 版为 canonical records 代表，paper_final 版为最终发布拷贝。

---

### 步骤 2：模块注释与活性矩阵（分析）

| rel | category | machines | canonical | produces | purpose |
|---|---|---|---|---|---|
| `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/analyze_modules.py` | 分析 | L | canonical | `module_by_cancer_median_k30.tsv`, `module_pathway_enrichment_k30.tsv`, `module_summary_k30.tsv`, `sample_module_activity_k30.tsv` | 计算 cancer 中位活性矩阵、Hallmark ORA、模块汇总表；Fig2e/f/g 核心中间产物 |

---

### 步骤 3：CPTAC 投影验证（分析）

| rel | category | machines | canonical | produces | purpose |
|---|---|---|---|---|---|
| `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/project_to_cptac.py` | 分析 | L | canonical | `cptac_module_by_cancer_k30.tsv`, `cptac_validation_k30.tsv` | masked NNLS 将 TCGA-NMF H 矩阵投影至 CPTAC 实测磷酸组，验证模块癌种偏好性（ED7b 数据） |

---

### 步骤 4：ED 稳健性 / 33-项目扩展（分析 + 画图）

| rel | category | machines | canonical | produces | purpose |
|---|---|---|---|---|---|
| `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/build_ED7c_33project.py` | 分析 | L | canonical | `module_by_cancer_median_k30_33proj.tsv`, `module_summary_k30_33proj.tsv` | 扩展至 33 项目（含 LAML）重跑 NMF k30，验证 LAML 血液肿瘤模块独立性 |
| `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/build_ED7a_stability.R` | 画图 | L | canonical | `figures/ED7a_stability_k20.pdf`, `figures/ED7a_stability_k40.pdf` | 渲染 ED7a：k=20/40 模块×癌种热图展示 NMF 模块数稳健性 |
| `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/build_ED7b_cptac.py` | 画图 | L | canonical | `figures/ED7b_cptac_projection.pdf` | 渲染 ED7b：NMF 模块在 CPTAC 预测 vs 实测磷酸化中 top-site 富集对比 |
| `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/build_ED7c_heatmap.R` | 画图 | L | canonical | `figures/ED7c_33project_k30.pdf` | 渲染 ED7c：33项目含 LAML 的模块×癌种热图 |

---

### 步骤 5：Fig2e 热图渲染（画图）

两套并行 canonical 脚本：`02_results` 版（工作开发版）和 `paper_final` 版（最终发布版）。Records 中 **两套均标记为 canonical**。以 `paper_final` 版为最终发表脚本，`02_results` 版为其前身（无需弃旧——两者共存作为开发→发布的版本层级，内容基本等价）。

**Fig2e 主图（30模块×32癌种中位活性热图）：**

| rel | category | machines | canonical | produces |
|---|---|---|---|---|
| `paper_final/fig2/fig_tcga_pancancer_atlas/scripts/build_figB_module_cancer.R` | 画图 | L | canonical | figB_module_cancer PDF（带 Hallmark 标注对角块状结构） |
| `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/build_figB_module_cancer.R` | 画图 | L | canonical | figures/figB_module_cancer_k30.pdf |

**Fig2e 样本×模块热图（全注释版 + 简版）：**

| rel | category | machines | canonical | produces |
|---|---|---|---|---|
| `paper_final/fig2/fig_tcga_pancancer_atlas/scripts/build_figA_annotated.R` | 画图 | L | canonical | figA_annotated PDF（含临床注释条） |
| `paper_final/fig2/fig_tcga_pancancer_atlas/scripts/build_figA_sample_overview.R` | 画图 | L | canonical | figA_sample_overview PDF（简洁版） |
| `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/build_figA_annotated.R` | 画图 | L | canonical | figures/figA_annotated_k30.pdf |
| `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/build_figA_sample_overview.R` | 画图 | L | canonical | figures/figA_sample_overview_k30.pdf |

**早期版本（已弃用）：**
- `SCP682_MAIN/attention_export/_build_nmf_heatmap.R`：records 标 canonical，但使用 `nmf_factor_by_cancer.tsv`（旧 attention_export 路径，非 20260612 NMF 产物），已被 paper_final 版取代（版本漂移）。**纳入 canonical 集但标注为已被 paper_final 版本替代**。

---

### 步骤 6：Fig2f 生存 Cox 森林图（画图）

| rel | category | machines | canonical | produces |
|---|---|---|---|---|
| `paper_final/fig2/fig_tcga_pancancer_atlas/scripts/build_figD_survival.R` | 画图 | L | canonical | figD_survival PDF（30模块 OS HR 森林图，cancer 分层 Cox + BH-FDR，16/30 显著） |
| `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/build_figD_survival.R` | 画图 | L | canonical | figures/figD_survival_k30.pdf, module_survival_k30.tsv |

**注**：`build_figD_survival.R` 同时执行 Cox 统计运算并产出 module_survival_k30.tsv，兼具"分析"性质；records 分类为"画图"（含统计计算的一体化渲染脚本）。

---

### 步骤 7：Fig2g 通路 ORA 条形图（画图）

| rel | category | machines | canonical | produces |
|---|---|---|---|---|
| `paper_final/fig2/fig_tcga_pancancer_atlas/scripts/build_figC_pathway_bars.R` | 画图 | L | canonical | figC_pathway_bars PDF（9个代表性模块 Hallmark ORA bar，3×3布局） |
| `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/build_figC_pathway_bars.R` | 画图 | L | canonical | figures/figC_pathway_bars_k30.pdf |

---

## 缺口列表

| 编号 | 缺口描述 | 影响 panel |
|---|---|---|
| GAP-0 | 未找到 atlas 模块专属的"TCGA 全样本 SCP682 推理 → X_nmf.npy 构建"脚本；`22_repredict_tcga_full_scp682_from_raw.py` 挂在 clinical 模块下，`predict_scp682_v4_0_public_bulk_20260508.py` 挂在 bulk 模块下 | Fig2e/f/g 全部 |
| GAP-1 | `paper_final/fig2/fig_tcga_pancancer_atlas/scripts/run_nmf.py`, `analyze_modules.py`, `project_to_cptac.py` 磁盘存在但未在 records c*.tsv 中登记，当前以 02_results 版 records 作代理 | Fig2e/f/g |
| GAP-2 | `build_figD_survival.R` 内嵌 lifelines Cox 统计计算，无独立 Python Cox 分析脚本（all-in-one R 实现）；如需跨语言复现需关注 | Fig2f |
| GAP-3 | ED7b 的 `build_ED7b_cptac.py` 归类为"画图"但实际包含 masked NNLS 投影计算——与 `project_to_cptac.py` 功能有重叠，records 未说明二者关系 | ED7b |
