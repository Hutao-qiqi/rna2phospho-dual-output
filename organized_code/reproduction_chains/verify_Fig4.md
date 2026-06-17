# Fig4 PPKO 复现链对抗式核验报告

生成日期：2026-06-16 | 核验员：对抗式只读审计（Claude Code）

核验方法：
1. 读 REPRODUCE.md + module_ppko.md 获取链路声明；
2. 逐类目录 glob 比对实际入库脚本；
3. 对每个 canonical 脚本 grep 实际读写文件名，比对链路声明；
4. 跨脚本追溯输入输出衔接。

---

## 总览

| Panel | 链完整性 | V10B 身份 | 文件路径印证 | 综合状态 |
|-------|----------|-----------|-------------|----------|
| Fig4a 架构示意 | 链头无外部输入（正常）| 脚本存在 | matplotlib 版存在；正式 SVG 缺 | 部分 |
| Fig4b P100 整体 bar | 中间有断点 | 输入 TSV 文件名含 "v9" 不含 "v10b" | 输入文件名不一致，疑版本混淆 | 缺口 |
| Fig4c 雨云图 | 依赖 make_fig4_locked_p100_tables.py 产出 | 同上 | 同上 | 缺口 |
| Fig4d 药物类别哑铃 | 同 b/c | 同上 | 同上 | 缺口 |
| Fig4e per-drug 热图 | 同 b/c | 同上 | 同上 | 缺口 |
| Fig4f true vs zero 散点 | 同 b/c | 同上 | 同上 | 缺口 |
| Fig4g TCGA ROC+boxplot (AUC=0.72) | TCGA 分析→画图链有断 | 患者推理脚本加载 V10（非 V10B）权重 + 预训练脚本 | 重大版本身份疑点 | 疑点 |
| Fig4h 对照 AUC 棒棒糖 | 依赖 tcga_tcpa_general_score_controls_v1.py | 同 g | 同 g | 疑点 |
| ED baseline bar (M2) | evaluate_ppko_p100_published_baselines.py→40_ed_baselines.R | baseline 评估脚本使用 V10B 正确权重路径 | 链相对完整 | 基本完整 |

---

## 逐 Panel 详细核验

### Fig4a — 架构示意图

**状态：部分**

链路：无分析步，直接画图。

脚本：`3_plotting/ppko/make_scp682_ppko_methods_schematic.py`（matplotlib 版），`3_plotting/ppko/11_panel_a_placeholder.R`（ggplot 占位）。

证据：
- `make_scp682_ppko_methods_schematic.py` 存在，无外部文件依赖，可独立运行。
- `11_panel_a_placeholder.R` 明确注释"正式版由 BioRender SVG 替换"。

**缺口：**
1. 正式 BioRender SVG 源文件路径未在任何脚本/记录中出现，与论文最终图的链路断裂。
2. matplotlib 版与 BioRender 版的关系未记录（是蓝本还是被替代？）。
3. 图中标注的图节点/边统计数（8192 site / 8751 protein）无专用导出脚本，需从 `build_global_phosphoprotein_heterograph_v10.py` 产出的 TSV 直接读数，且 `build_global_phosphoprotein_heterograph_v10.py` 的 `OUT_DIR` 指向 `...\global_phosphoprotein_heterograph_v10`（无 `_measured_string700_top50` 后缀），与训练脚本的 `DEFAULT_GRAPH`（`...\global_phosphoprotein_heterograph_v10_measured_string700_top50`）路径不完全对应。

**建议：** 补充 BioRender SVG 文件的原始路径记录；确认图统计数的导出来源。

---

### Fig4b/c/d/e/f — P100 验证系列（n=125，位点级指标）

**状态：有重大缺口（输入文件名含 "v9"，疑版本混淆）**

声称链路：
`export_v10b_p100_sitelevel_all125.py`（V10B 推理）→`make_fig4_locked_p100_tables.py`（锁定源表）→`make_fig4_locked_p100_direction_stats.py`（方向统计）→画图脚本（03~07_panel_*.R）。

实际代码发现：

**疑点 1（最重要）：输入文件名含 "global_graph_v9"**
`make_fig4_locked_p100_tables.py` 第 69-70 行：
```python
metrics = pd.read_csv(RAW / "global_graph_v9_p100_metrics.tsv", sep="\t")
drug_summary = pd.read_csv(RAW / "global_graph_v9_p100_drug_summary.tsv", sep="\t")
```
其中 `RAW = E:\data\gongke\TCGA-TCPA\02_results\raw_external\v10b_p100_validation`。
- 文件名前缀为 `global_graph_v9`，表明该文件由 `signed_phospho_regulatory_prior_v9` 图先验（V9 图）生成，而非 V10 异质图（`global_phosphoprotein_heterograph_v10_measured_string700_top50`）。
- 但目录名为 `v10b_p100_validation`。
- 这意味着：要么该 TSV 是用 V9 图先验+V10B 架构的混合配置生成，要么该 TSV 实际对应旧版本结果但被挪入 v10b 目录。无法从现有脚本链中判断该 TSV 是否真正来自 V10B strong300。

**疑点 2：RAW 目录中的文件无对应生成脚本**
`02_results/raw_external/v10b_p100_validation/global_graph_v9_p100_metrics.tsv` 的来源脚本在 organized_code 中**不存在**。
- `validate_v10b_p100_all_drugs.py` 的输出写到 `{PACKAGE_ROOT}/validation_outputs/p100_all_drugs/tables/v10b_p100_all_drug_metrics.tsv`，与 RAW 目录路径不匹配。
- `export_v10b_p100_sitelevel_all125.py` 的输出写到 `02_results/single_cell/20260531_scp682_ppko_v10b_p100_sitelevel_all125/tables/`，也与 RAW 不匹配。
- 结论：`global_graph_v9_p100_metrics.tsv` 的生成脚本不在复现集中，链路有断点。

**疑点 3：01_config.R 硬编码锁定路径**
`01_config.R` 第 11 行：
```r
SRC_DIR <- "E:/data/gongke/TCGA-TCPA/02_results/figure_sources/20260528_fig4_locked_p100_v10b_cosine_direction"
```
这是本地绝对路径，依赖锁定目录下已存在的 TSV 文件，而这些 TSV 由 `make_fig4_locked_p100_tables.py` 生成。后者又依赖上述来源不明的 `global_graph_v9_p100_metrics.tsv`。

**疑点 4：make_fig4_locked_p100_direction_stats.py 部分列名不一致**
第 80 行从 `panel_f_p100_true_vs_zero_paired.tsv` 读取 `all_direction_true` 列，但 `make_fig4_locked_p100_tables.py` 输出的 paired 表列名为 `all_cosine_true`/`all_direction_true`（通过 `all_direction` 合并后缀 `_true`），需核实合并后字段是否确实存在。

**建议：**
1. 追溯 `global_graph_v9_p100_metrics.tsv` 的实际来源脚本，确认是否确为 V10B 推理结果。
2. 若该文件由 `evaluate_ppko_p100_published_baselines.py` 的 side-output 产生，需在链路中明确标注。
3. 重命名该 TSV 去除 `v9` 前缀，或在复现链文档中解释命名的由来（"v9 图先验" vs "v10b 模型"）。

---

### Fig4g — TCGA-TCPA 患者响应 ROC+boxplot（AUC=0.72）

**状态：有重大身份疑点（V10B 权重路径指向旧文件名）**

链路：`tcga_tcpa_ppko_patient_response_v1.py`（患者推理）→`make_fig4_tcga_validation_plot_tables.py`（建图源表）→`08_panel_g.R`（画图）。

**疑点 1（最重要）：checkpoint 文件名为 V10 而非 V10B**
`tcga_tcpa_ppko_patient_response_v1.py` 第 18 行：
```python
"v10b_300": ROOT / "02_results" / "single_cell" / "20260520_scp682_ppko_1_attention_prior_v10b_strong_contrast_300" / "models" / "scp682_ppko_attention_prior_v10_best.pt"
```
- 权重文件名为 `scp682_ppko_attention_prior_v10_best.pt`（V10 命名），不是文档声称的 `scp682_ppko_v10b_strong300_best.pt`（V10B 命名）。
- 目录名含 `v10b_strong_contrast_300`，表明这是用于 V10B strong300 训练的运行目录，但保存的 checkpoint 仍用 V10 命名方案（见训练脚本：`pretrain_v10b_strong300.py` 第 324 行输出 `scp682_ppko_attention_prior_v10_best.pt`）。
- 结论：此处没有版本混淆问题——训练脚本确实输出 `_v10_best.pt` 而非 `_v10b_strong300_best.pt`，checkpoint 文件名与架构名（`AttentionPriorManifoldV10`）一致。但这造成文档声称的文件名（`scp682_ppko_v10b_strong300_best.pt`）与脚本实际读取的文件名（`scp682_ppko_attention_prior_v10_best.pt`）不一致，复现时将找不到文档声称的路径。

**疑点 2：预训练脚本加载路径指向 V10（非 V10B 的独立 pretrain 脚本）**
`tcga_tcpa_ppko_patient_response_v1.py` 第 9 行：
```python
PRETRAIN = ROOT / "02_results" / "single_cell" / "20260520_SCP682_PPKO_release_v1" / "scripts" / "pretrain_scp682_ppko_attention_prior_v10.py"
```
- 该脚本从 `SCP682_PPKO_release_v1` 的 V10 脚本导入 `AttentionPriorManifoldV10`，而非从 `SCP682_PPKO_V10B_transferable` 的 `pretrain_v10b_strong300.py` 导入。
- 两者虽然都定义 `AttentionPriorManifoldV10`（代码等价），但代码 provenance 不同，无法从这个脚本追溯到 canonical `pretrain_v10b_strong300.py`。
- 对比：`export_v10b_p100_sitelevel_all125.py`（P100 推理）正确地 `from pretrain_v10b_strong300 import ...`，而患者响应脚本使用 release_v1 的旧路径。

**疑点 3：TCGA 分析结果写入目录与画图读取目录不一致**
- `tcga_tcpa_ppko_patient_response_v1.py` 写到：`02_results/clinical_validation/20260531_tcga_tcpa_ppko_patient_response_v1_targeted_expanded/tables/`
- `tcga_tcpa_general_score_controls_v1.py` 从 `20260527_tcga_tcpa_ppko_patient_response_v1` 读 PPKO 结果（第 8 行），写到 `20260527_...` 目录。
- `make_fig4_tcga_validation_plot_tables.py` 从 `FIG / "tcga_validation"` 读（`FIG` = `20260528_fig4_locked_p100_v10b_cosine_direction`），即 `figure_sources/.../tcga_validation/tables/v10b_300_patient_predictions.tsv`。
- 这三个目录路径均不同：`20260531_...`、`20260527_...`、`20260528_.../tcga_validation/`。复现时需手动将输出拷贝到指定位置，链路实际上是**断开**的（缺中间拷贝脚本）。

**疑点 4：model_score_auc_ci_permutation.tsv 的生成脚本缺失**
`make_fig4_tcga_validation_plot_tables.py` 第 75 行读 `model_auc = pd.read_csv(TCGA / "tables" / "model_score_auc_ci_permutation.tsv", ...)`，该文件的生成脚本在 organized_code 中**不存在**（AUC bootstrap + permutation 计算脚本未入册）。

**建议：**
1. 统一 checkpoint 文件名文档：将 REPRODUCE.md / module_ppko.md 的 `scp682_ppko_v10b_strong300_best.pt` 更正为 `scp682_ppko_attention_prior_v10_best.pt`（在 v10b strong_contrast_300 运行目录下）。
2. 将 `tcga_tcpa_ppko_patient_response_v1.py` 的 pretrain 导入路径指向 `pretrain_v10b_strong300.py`，与 `export_v10b_p100_sitelevel_all125.py` 保持一致。
3. 补充目录间文件拷贝步骤，或统一写出路径到单一目录。
4. 将 AUC bootstrap 计算脚本（生成 `model_score_auc_ci_permutation.tsv`）纳入复现集。

---

### Fig4h — 对照 AUC 棒棒糖图

**状态：疑点（依赖链继承 Fig4g 的断点）**

链路：`tcga_tcpa_general_score_controls_v1.py`（生成对照 AUC）→`make_fig4_tcga_validation_plot_tables.py`（图源表）→`09_panel_h.R`（画图）。

**疑点 1：读取路径与 g 相同，继承断点**
`tcga_tcpa_general_score_controls_v1.py` 第 8-11 行从 `20260527_tcga_tcpa_ppko_patient_response_v1` 读 PPKO 预测结果（与 g panel 路径不统一，见 g panel 疑点 3）。

**疑点 2："V10B" 层推理读的是哪个版本**
`tcga_tcpa_general_score_controls_v1.py` 第 100 行：
```python
base = pred[pred["model_name"].eq("v10b_300")].copy()
```
这个 `pred` 来自 `20260527_tcga_tcpa_ppko_patient_response_v1/tables/all_model_patient_predictions.tsv`，而该文件由 `tcga_tcpa_ppko_patient_response_v1__W.py`（W机版，输出到 `20260527_...` 目录）生成。`__W.py` 的 `v10b_300` checkpoint 同样指向 `scp682_ppko_attention_prior_v10_best.pt`（V10 命名，见上）。

**疑点 3：random marker control 脚本未明确使用 V10B 推理**
`tcga_tcpa_general_score_controls_v1.py` 的 random k marker AUC 是基于 TCPA RPPA 值的简单统计，不涉及模型推理，不受版本影响。这一部分是干净的。

**建议：** 解决 Fig4g 的路径/版本疑点后，h panel 链路会同步修正。

---

### ED Fig M2 — 已发表基线对比 bar

**状态：基本完整（但链头 P100 基线输入路径未入册）**

链路：`evaluate_ppko_p100_published_baselines.py`（V10B vs 基线）→`40_ed_baselines.R`（画图）。

证据（正确）：
- `evaluate_ppko_p100_published_baselines.py` 第 282 行：`model_path = package_root / "models" / "scp682_ppko_v10b_strong300_best.pt"`——使用正确的 transferable 包中的文件名。
- `evaluate_ppko_p100_published_baselines.py` 第 276 行：`from pretrain_v10b_strong300 import AttentionPriorManifoldV10, ...`——正确导入 V10B 脚本。
- `40_ed_baselines.R` 第 6 行读 `p100_v10b_published_baseline_comparison_summary.tsv`，与 `evaluate_ppko_p100_published_baselines.py` 第 470 行输出文件名一致。

**缺口：**
1. `evaluate_ppko_p100_published_baselines.py` 需要 `lincs_p100_comparison_delta_v7` 和 `decryptm_comparison_delta_v8`，这两个输入的构建步骤在 `2_analysis/ppko/` 中有入口脚本（`build_lincs_p100_comparison_delta_v7.py`、`build_decryptm_comparison_delta_v8.py`），但这些脚本本身的上游（原始数据下载→标准化）路径依赖 `D:\data\lsy\...` 硬编码路径，链头不可移植。
2. 外部已发表基线方法（KSEA、热扩散、RWR、Ridge、Retrieval）的实现代码未在 organized_code 中出现，在 `evaluate_ppko_p100_published_baselines.py` 内应有内置实现，需确认是否完整。

**建议：** 核查 `evaluate_ppko_p100_published_baselines.py` 是否内置了全部基线方法实现，或是否存在外部依赖脚本未入册。

---

## 汇总：关键疑点/缺口（全 Fig4）

1. **checkpoint 文件名文档不一致**：`module_ppko.md` 和 `REPRODUCE.md` 声称 canonical checkpoint 为 `scp682_ppko_v10b_strong300_best.pt`，但实际训练脚本输出为 `scp682_ppko_attention_prior_v10_best.pt`（V10 命名方案）。`validate_v10b_p100_all_drugs.py` 和 `export_v10b_pxd063604_sitelevel.py` 的默认 `MODEL` 路径用 V10B 命名方案（`scp682_ppko_v10b_strong300_best.pt`），与训练脚本输出路径不一致，一次性复现将在加载 checkpoint 时失败。

2. **P100 panel b-f 的输入 TSV 文件名含 "v9"**：`make_fig4_locked_p100_tables.py` 读取 `global_graph_v9_p100_metrics.tsv`（目录 `raw_external/v10b_p100_validation`），该文件的生成脚本不在复现集，无法判断其是否真正来自 V10B strong300 推理，存在结果被旧版图先验（V9）数据污染的风险。

3. **TCGA 患者响应脚本加载旧版 pretrain 路径**：`tcga_tcpa_ppko_patient_response_v1.py` 从 `SCP682_PPKO_release_v1` 的 V10 脚本导入架构（非 `pretrain_v10b_strong300.py`），同时 `v10b_300` checkpoint 指向 V10 文件名。两个平行版本（`tcga_tcpa_ppko_patient_response_v1.py` 和 `__W.py`）均如此。

4. **TCGA 分析输出目录与画图读取目录不连续**：三个脚本写出/读入分别指向 `20260527_...`、`20260531_...`、`20260528_.../tcga_validation/` 三个目录，缺乏中间拷贝/汇整脚本，端到端复现需手动操作。

5. **AUC bootstrap+permutation 计算脚本缺失**：`make_fig4_tcga_validation_plot_tables.py` 读取 `model_score_auc_ci_permutation.tsv`（含置信区间和置换 p 值），其生成脚本未在 organized_code 中出现，是 AUC=0.72 这一核心结果的关键缺口。

6. **`rerun_missing_items.py` 明确承认 P100 指定口径来自 V10 而非 V10B**：该文件第 549 行注释 "P100 指定口径 0.602/0.888/0.942 已定位到 SCP682-PPKO release v1 的 V10 结果，不是 V10B strong300 结果"。若论文 Fig4b/c 引用的是这套数字，则图中实际对应 V10 而非 V10B，需在论文中或补充中注明。
