# Gap Resolution 核验报告

生成日期：2026-06-17
来源：module_*.md 缺口段 + verify_Fig4.md + verify_Fig5.md + _codeorg_staging/records/_light.tsv (28 条 needs_review)

---

## 核验方法

对每一待确认项，在以下路径全量搜索关键词：
- `E:\data\gongke\TCGA-TCPA\03_code`、`remote_scripts`、`paper_final`、`paper_materials_*`、`02_results`、`04_figures`
- `E:\data\gongke\TCGA-TCPA\_codeorg_staging\ubuntu`
- `E:\data\gongke\TCGA-TCPA\_codeorg_staging\winwsl`

---

## 一、重点待确认项（7项）

### GAP-1 TCGA 全量 RNA-seq 下载脚本

**状态：FOUND**

- 文件：`_codeorg_staging/ubuntu/src/R/get_tcga_rnaseq.R`（同路径亦存于 `99_archive/...`）
- 该脚本使用 `TCGAbiolinks::GDCquery` + `GDCdownload`，支持 `--project` 参数批量下载 TCGA 各项目 STAR counts RNA-seq；包含 5 次重试和临床信息导出逻辑。
- _light.tsv 标为 `needs_review`（`src/R/get_tcga_rnaseq.R`）。
- **建议**：补入复现集（category：`support`），与 `04_figures/20260528_fig5_v2/code/22_repredict_tcga_full_scp682_from_raw.py` 形成完整 TCGA 数据→推理链；复现时需按 32 癌种逐项目调用。

---

### GAP-2 graph_statistics 边数（420,102 / 21,925）导出脚本

**状态：MISSING**

- 搜索 `graph_statistics`、`420102`、`420.?102`、`21925`：无专用导出脚本命中。
- `process_copheemap_prior_20260428.py` 内部计数位点图边（420,102），但无 `print_graph_stats()` 或专用 CSV 输出；`make_fig1_rebuild.py` 和 `panel_a.R` 中为硬编码数字。
- `remote_scripts/summarize_copheemap_overlap.py`（已入册 `bulk/support`）统计 CoPheeMap/KSA/n2v 重叠覆盖率，但不直接输出 graph_statistics.tsv。
- **推测来源**：训练脚本 `train_scp682_exact_scnet_gnn_v1.py` 建图时 stdout 打印的边数，或 `build_global_phosphoprotein_heterograph_v10.py`（PPKO 侧，节点/边统计无 bulk 样本图信息）。
- **建议**：在 `process_copheemap_prior_20260428.py` 末尾添加 `print_graph_statistics()` 或写出 `graph_statistics.tsv`，或在复现说明中标注数字来源于 training log 的 stdout。无需新增脚本，标注即可。

---

### GAP-3 外部队列 DeepGxP 推理脚本（Fig2c 外部 benchmark 中的 DeepGxP baseline）

**状态：FOUND（internal CPTAC 推理有脚本；external 队列有合并脚本）**

- **内部 CPTAC 5-fold**：`remote_scripts/run_scp682_deep_learning_baselines.py`（已入册 `organized_code/2_analysis/bulk/run_deepgxp_cptac_5fold_oof_20260526.py`）；包含 `DeepGxPBulkTorch` 类实现和 `predict_deepgxp_external()` 函数，可在外部队列上重运行。
- **外部队列 DeepGxP baseline 对比**：`remote_scripts/run_scp682_external_deep_methods.py`（已入册 `organized_code/2_analysis/bulk/`）；该脚本对外部队列同步运行 DeepGxP 推理（读 `selected_genes_by_fold.tsv` + 预训练 fold 权重），写出 `per_site_spearman_external_deep_methods.tsv`。
- **汇总**：`make_external_9model_benchmark.py`（已入册 `organized_code/2_analysis/bulk/`）。
- **建议**：两个脚本均已入复现集，无需追加。外部 DeepGxP 推理需先运行内部 CPTAC 5-fold 产出 `deepgxp_fold*.pt` 权重。

---

### GAP-4 per_site_spearman_with_deep_learning.tsv 合并链

**状态：FOUND（TSV 已存在；生成链分散但可追溯）**

- 文件实体：`paper_materials_SCP682/01_key_results/per_site_spearman_with_deep_learning.tsv` 已存在于本地。
- `panel_b.R` 直接读取该文件（第 16 行）。
- 生成文档：`paper_materials_SCP682/01_key_results/deep_learning_baseline_addendum.md` 明确说明该表是"原 7 方法内部基线长表加上 VAE 和 DeepGxP_5fold"。
- 生成步骤（分散）：`run_scp682_deep_learning_baselines.py`（产出 DeepGxP + MLP + VAE 折叠 TSV）→ 手动或脚本 merge 到 7 方法长表。merge 脚本未单独入册，但 `deep_learning_baseline_addendum.md` 提供了数值锚点（CPTAC_all median：SCP682 0.5474、DeepGxP_5fold 0.0517、MLP 0.2846、VAE 0.2829）可用于验证合并正确性。
- **建议**：无需补充脚本。在 REPRODUCE.md 中加注"合并步骤参见 deep_learning_baseline_addendum.md"；TSV 本体已作为 source data 冻结。

---

### GAP-5a Fig4：global_graph_v9_p100_metrics.tsv 生成脚本

**状态：FOUND（生成脚本存在，但为旧版 V9 模型脚本）**

- 文件名包含 `global_graph_v9_p100`：`_codeorg_staging/winwsl/03_code/single_cell/modeling/validate_scp682_global_graph_v9_p100.py` 及 `99_archive/` 下对应脚本。
- 该脚本明确加载 `scp682_ppko_global_graph_manifold_v9_best.pt`（V9 模型），输出写入 `20260519_scp682_ppko_1_global_graph_manifold_v9_p100_validation` 目录。
- 这意味着 `02_results/raw_external/v10b_p100_validation/global_graph_v9_p100_metrics.tsv` 是 **V9 模型在 P100 上的验证结果**，而非 V10B 的。与目录名 `v10b_p100_validation` 命名不一致，存在版本混淆（verify_Fig4.md 疑点 1 属实）。
- **建议**：该文件是 V9 遗留产物。若 Fig4b-f 实际使用 V9 结果（rerun_missing_items.py 第 549 行注释已确认 P100 指标 0.602/0.888/0.942 来自 V10 非 V10B），需在论文/补充说明中区分；不应补入复现集（属 legacy）。正确的 V10B P100 metrics 应从 `export_v10b_p100_sitelevel_all125.py` 产出。

### GAP-5b Fig4：TCGA AUC bootstrap/permutation 脚本

**状态：FOUND**

- 文件：`_server_ppko_tcga_stats.py`（本地根目录）
- 该文件实现 2000 次 bootstrap CI 和 2000 次 permutation p 值，写出 `model_score_auc_ci_permutation.tsv`（第 66 行）。
- 已在 `_codeorg_staging/inventory_winwsl.txt` 登记，`_codeorg_staging/records/c01.tsv` 有记录。
- **但未入复现集**（未出现在 `organized_code/` 任何目录）。
- **建议**：应补入 `organized_code/2_analysis/ppko/` 或等价路径，category=`support`，可作为 Fig4g 链路中 "AUC bootstrap" 步骤的独立脚本。

---

### GAP-6 Fig5 的 `20260530_fig5_exact_site_anchor_search_v1` 产出脚本

**状态：MISSING（结果目录存在但生成脚本不在复现集）**

- 结果目录 `02_results/model_validation/20260530_fig5_exact_site_anchor_search_v1/tables/` 存在，包含 6 个 TSV/JSON（`tcga_kirc_rps6_s235_s236_sample_table.tsv.gz` 等）。
- `run_tcga_predicted_site_survival_concordance_20260527.py` 默认输出到 `20260527_cptac_measured_tcga_predicted_site_survival_concordance_v1`，非 `20260530_fig5_exact_site_anchor_search_v1`（日期和目录名均不同）。
- 搜索 `exact_site_anchor_search`、`20260530_fig5_exact`：仅命中 plots/prepare 等下游脚本，未找到真正写出该目录的脚本。
- **推测来源**：可能是 `run_tcga_predicted_site_survival_concordance_20260527.py` 加 `--out-dir 02_results/model_validation/20260530_fig5_exact_site_anchor_search_v1` 参数运行（文件名 `s235_s236` 与该脚本的 KIRC RPS6 S235/S236 分析一致），但无 launch 脚本记录。
- **建议**：补充一个 `launch_fig5_exact_site_anchor_search_v1.sh`（或在 `run_tcga_predicted_site_survival_concordance_20260527.py` 的 REPRODUCE.md 注释中添加 `--out-dir` 参数），标为 `support`。这是 panels d/e/c 的数据来源，P0 优先级。

---

### GAP-7 DepMap 数据提取脚本（Fig5f panel_f_depmap_rcc_mtor_drug_correlations.tsv）

**状态：MISSING（生成脚本完全缺失）**

- TSV 文件存在于：`paper_final/fig5/source_data/tables/panel_f_depmap_rcc_mtor_drug_correlations.tsv`（由 `prepare_fig5_source_data.py` 从 `panel_c_plotted_drug_correlations.tsv` 复制而来）。
- 搜索 `depmap.*rcc`、`panel_c_plotted`、`prism.*auc`：仅命中 `plot_rps6_ps6_fig5_anchor_20260530.py`（读取 TSV 做图）和 `prepare_fig5_source_data.py`（复制）；无任何 canonical/support 脚本涵盖 DepMap PRISM AUC 下载、ccRCC 细胞系定义、predicted pS6 vs AUC Spearman 计算。
- `03_code/biomarker_analysis/paad/run_pdac_fam20c_axis_depmap_prism.py` 涉及 DepMap 但针对 PAAD，非 KIRC/ccRCC。
- **推测来源**：手动从 DepMap portal（depmap.org）下载 PRISM AUC 21Q2 或 23Q4，筛选 ccRCC/RCC 细胞系，与 SCP682 预测 pS6 做 Spearman 相关；脚本不在本地任何已知路径。
- **建议**：P0 缺口。需补充一个 `build_depmap_ccRCC_mtor_pS6_correlations.py` 脚本（或附 DepMap 版本号 + 下载链接 + 细胞系列表），并入 `organized_code/2_analysis/clinical/support`。

---

### GAP-8 Fig5g（fig5b_all_site_module_*.tsv）和 Fig5h（fig5d_ptmsea_*.tsv）生成脚本

**状态：TSV 文件存在，生成脚本 MISSING**

**Fig5g（`fig5b_all_site_module_*.tsv` 四张表）**：
- TSV 存在于 `paper_final/fig5/source_data/tables/` 和 `05_manuscript/source_data_SI/Fig5/`。
- 搜索 `fig5b_all_site_module`、`all_site_module_risk` in `.py`：无命中。
- `build_panel_b_module_cancer.py` 输出 `full_tcga_scp682_main_fig5b_pathway_module_*.tsv`（不同文件名前缀），不是 `fig5b_all_site_module_*.tsv`；两者不能对应。
- **推测**：`panel_i_full_phospho_specific_modules_data.py` 的 FastICA 产物（`20260531_panel_i_phospho_specific_modules_full_v1/tables/`）可能包含这些数据，但默认路径与 `fig5b_all_site_module_*.tsv` 文件名不匹配，需手动确认。

**Fig5h（`fig5d_ptmsea_*.tsv` 两张表）**：
- TSV 存在于 `paper_final/fig5/source_data/tables/` 和 `05_manuscript/source_data_SI/Fig5/`。
- `panel_i_full_phospho_specific_modules_data.py` 使用 `DEFAULT_PTMSIGDB_GMT`（PTMSigDB v2.0.0 flanking human），可通过 FastICA 模块富集产出 PTM-SEA 矩阵；但默认输出目录为 Linux 服务器路径 `/data/lsy/Infinite_Stream/02_results/fig5/20260531_panel_i.../tables/`，文件名为该脚本内部命名，非 `fig5d_ptmsea_signature_heatmap_matrix.tsv`。
- **推测**：`panel_i_full_phospho_specific_modules_data.py` 是最可能的生成源，但需在 Linux 服务器上运行并后处理重命名；或存在一个未收录的临时脚本完成了这步。
- **建议**：P0 缺口（verify_Fig5.md red panel）。补充两个 launch 脚本或在 `panel_i_full_phospho_specific_modules_data.py` 的 docstring 中明确注明 `fig5b_` / `fig5d_` 文件名映射关系，并入 `support` 类。

---

### GAP-9 GSE242299 h5ad 下载/预处理脚本

**状态：MISSING（GEO 直接下载，无预处理脚本）**

- `run_kirc_rps6_validation_full.ps1`（已入册）第 22 行明确：h5ad 文件从 `D:\data\lsy\GSE242299_all_cells_50236_33538.h5ad.gz` 直接读取，说明文件是预先手动下载的。
- 搜索 `GSE242299`、`kirc.*h5ad.*download`：无任何 Python/R 下载脚本命中；仅在 `.tex`/`.bib` 等文献引用文件中出现。
- **推测来源**：从 GEO 网站手动下载 `GSE242299_all_cells_50236_33538.h5ad.gz`，直接用于推理（已预处理，h5ad 包含 obs 注释），无需 Cell Ranger 等原始重处理。
- **建议**：在 REPRODUCE.md 中补注"GSE242299 h5ad 预处理版（50,236 细胞×33,538 基因）直接从 NCBI GEO GSE242299 下载 `.h5ad.gz` 文件；无需额外预处理"。无需新增脚本，标注下载 accession 即可（P2 优先级）。

---

## 二、needs_review 条目核查（_light.tsv 中 28 条）

| rel | 实体状态 | 与复现集的关系 | 建议 |
|-----|----------|---------------|------|
| `src/R/get_tcga_rnaseq.R` | FOUND（见 GAP-1） | needs_review → 应补入 support | 补入复现集 |
| `src/R/get_tcpa_l4.R` | FOUND（`_codeorg_staging/ubuntu/src/R/get_tcpa_l4.R`） | TCPA L4 数据获取，服务于 TCPA 32 项目 RNA/RPPA 配对 | 补入 support |
| `src/R/preprocess_rna_tpm.R` | FOUND（同目录） | RNA TPM 预处理 | 补入 support |
| `src/R/preprocess_rna_tpm_all.R` | FOUND（同目录） | 多癌种批量 RNA TPM 预处理 | 补入 support |
| `src/R/compute_gsva.R` | FOUND（`_codeorg_staging/ubuntu/compute_gsva.R`） | 已在 module_framework.md 列为 shared/support，未入 organized_code | 补入 support |
| `src/R/enrichment_top_features.R` | FOUND（同目录） | 顶部特征富集分析 | 确认是否被 canonical 脚本调用后决定 |
| `src/py/etl/match_samples.py` | FOUND（`_codeorg_staging/ubuntu/src/py/etl/match_samples.py`） | 样本匹配 ETL，为训练数据对齐共用工具 | 补入 support |
| `src/py/etl/merge_barcodes.py` | FOUND（同目录） | barcode 合并 | 补入 support |
| `src/py/etl/merge_parquet_matrices.py` | FOUND（同目录） | Parquet 矩阵合并 | 补入 support |
| `src/py/etl/merge_tcpa_l4.py` | FOUND（同目录） | TCPA L4 数据合并 | 补入 support |
| `src/py/eval/protein_mrna_correlation.py` | FOUND（同目录） | 蛋白-mRNA 相关性计算 | 确认是否服务 canonical 评估后决定 |
| `src/py/eval/rank_export.py` | FOUND（同目录） | rank 导出 | 同上 |
| `src/py/fe/feature_builder.py` | FOUND（同目录） | 特征构建器 | 确认后补 support |
| `src/py/fe/impute_rppa.py` | FOUND（同目录） | RPPA 缺失值填补 | 确认后补 support |
| `src/py/models/graph_prior_utils.py` | FOUND（同目录） | 图先验工具函数 | 确认是否被 canonical 模型导入后补入 |
| `src/py/models/predict_unmeasured.py` | FOUND（同目录） | 预测未测量位点 | 确认是否为 canonical 推理路径的一部分 |
| `src/py/models/train_all.py` | FOUND（同目录） | 全位点训练入口 | 确认与 exact_scnet_gnn 的关系；若为 legacy 则排除 |
| `src/py/models/update_model.py` | FOUND（同目录） | 模型更新 | 同上 |
| `src/py/models/export_tcpa_full_csv.py` | FOUND（同目录） | TCPA 完整 CSV 导出 | 数据发布用；补入 support |
| `src/py/drift/detect_drift.py` | FOUND（同目录） | 数据漂移检测 | 非 canonical 训练必需，建议 support/optional |
| `scripts/fill_tcpas_missing.py` | FOUND（`_codeorg_staging/ubuntu/scripts/fill_tcpas_missing.py`） | TCPA 缺失值填补 | 确认后补 support |
| `scripts/external_validation/build_pairing_manifest.py` | FOUND（同目录） | 外验配对清单 | 服务于 Fig2c 外验，补 support |
| `scripts/external_validation/compute_egfr_deltas_from_manifest.py` | FOUND | 计算 EGFR delta | 补 support |
| `scripts/external_validation/compute_predicted_deltas_from_manifest.py` | FOUND | 计算预测 delta | 补 support |
| `scripts/external_validation/map_ensembl_to_symbol.R` | FOUND | Ensembl→symbol | 补 support |
| `scripts/external_validation/parse_geo_series_matrix.py` | FOUND | GEO series matrix 解析 | 补 support |
| `scripts/external_validation/stats_test_external_deltas.py` | FOUND | 外验统计检验 | 补 support |
| `scripts/external_validation/summarize_external_predicted_deltas.py` | FOUND | 外验汇总 | 补 support |

所有 28 条 needs_review 条目均在 `_codeorg_staging/ubuntu/` 下找到实体文件，无真正 missing 项；问题在于未决定是否纳入 organized_code 复现集。

---

## 三、统计汇总

| 类别 | 计数 |
|------|------|
| **Found（主重点 9 项中）** | 6 |
| **Missing（主重点 9 项中）** | 3 |
| **needs_review 全部 found** | 28/28 |

**Found 的 6 项**（有脚本/文件实体）：
1. GAP-1：`src/R/get_tcga_rnaseq.R`（TCGA RNA 下载）
2. GAP-3：`run_scp682_external_deep_methods.py` + `run_deepgxp_cptac_half_retrain_20260511.py`（DeepGxP 推理）
3. GAP-4：`per_site_spearman_with_deep_learning.tsv` 已冻结 + `deep_learning_baseline_addendum.md` 说明链路
4. GAP-5b：`_server_ppko_tcga_stats.py`（AUC bootstrap/permutation）— 已存在但**未入复现集**
5. GAP-5a：`validate_scp682_global_graph_v9_p100.py`（V9 P100 metrics 生成脚本）— 是 V9 非 V10B，legacy

**Missing 的 3 项**（无生成脚本）：
1. **GAP-2**：graph_statistics 边数导出（数字硬编码，无专用脚本）
2. **GAP-6**：`20260530_fig5_exact_site_anchor_search_v1` 目录的生成脚本（结果存在，脚本缺失）
3. **GAP-7**：DepMap ccRCC mTORi 提取脚本（完全缺失）
4. **GAP-8**：`fig5b_all_site_module_*.tsv` 和 `fig5d_ptmsea_*.tsv` 生成脚本（TSV 存在，脚本缺失）
5. **GAP-9**：GSE242299 下载/预处理脚本（GEO 手动下载，无脚本）

注：GAP-2/GAP-9 属轻度缺口（边数可从日志读取；h5ad 直接下载无需预处理），GAP-6/GAP-7/GAP-8 是真正的复现链断点（P0 级别）。

---

## 四、应补入复现集的脚本 rel 列表

以下脚本目前**在 _codeorg_staging 或本地存在但未进 organized_code**，建议追加：

| 优先级 | rel（相对于根目录）| 建议 category | 建议放置路径 |
|--------|-------------------|--------------|------------|
| **P0** | `_server_ppko_tcga_stats.py` | ppko/分析/support | `organized_code/2_analysis/ppko/` |
| P1 | `_codeorg_staging/ubuntu/src/R/get_tcga_rnaseq.R` | data/分析/support | `organized_code/2_analysis/data/` |
| P1 | `_codeorg_staging/ubuntu/src/R/get_tcpa_l4.R` | data/分析/support | 同上 |
| P1 | `_codeorg_staging/ubuntu/src/R/preprocess_rna_tpm.R` | data/分析/support | 同上 |
| P1 | `_codeorg_staging/ubuntu/src/R/preprocess_rna_tpm_all.R` | data/分析/support | 同上 |
| P1 | `_codeorg_staging/ubuntu/compute_gsva.R` | shared/分析/support | `organized_code/2_analysis/shared/` |
| P1 | `_codeorg_staging/ubuntu/scripts/external_validation/` (7 个脚本) | bulk/分析/support | `organized_code/2_analysis/bulk/` |
| P2 | `_codeorg_staging/ubuntu/src/py/etl/` (4 个 ETL 脚本) | data/分析/support | `organized_code/2_analysis/data/` |
| P2 | `_codeorg_staging/ubuntu/scripts/fill_tcpas_missing.py` | data/分析/support | 同上 |

---

## 五、需新建脚本（无法从现有脚本追溯）

| 缺口 | 建议 | 优先级 |
|------|------|--------|
| GAP-6：`20260530_fig5_exact_site_anchor_search_v1` 生成脚本 | 补 `launch_fig5_exact_site_anchor_search_v1.sh`（或在 `run_tcga_predicted_site_survival_concordance_20260527.py` REPRODUCE.md 中注明 `--out-dir` 参数） | P0 |
| GAP-7：DepMap ccRCC mTORi 提取脚本 | 补 `build_depmap_ccRCC_mtor_pS6_correlations.py`（DepMap PRISM 版本 + ccRCC 细胞系列表 + Spearman 计算） | P0 |
| GAP-8a：`fig5b_all_site_module_*.tsv` 生成链 | 确认 `panel_i_full_phospho_specific_modules_data.py` 是否产出这些文件（运行目录对齐） | P0 |
| GAP-8b：`fig5d_ptmsea_*.tsv` 生成链 | 确认 `panel_i_full_phospho_specific_modules_data.py` 中 PTM-SEA 产物的文件名映射 | P0 |
| GAP-2：graph_statistics 边数 | 在 `process_copheemap_prior_20260428.py` 末尾添加统计输出（或注明来自 training log） | P2 |
| GAP-9：GSE242299 下载 | 在 REPRODUCE.md 添注 GEO accession 和 `.h5ad.gz` 文件名，无需新脚本 | P2 |
