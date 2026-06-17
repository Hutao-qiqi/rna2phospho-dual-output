# Fig2 复现链对抗式核验报告

生成日期：2026-06-17
核验员：对抗式只读核查（Grep + Read 抽查实际脚本输入输出）
范围：Fig2 a-g（bulk RNA→phospho 架构 / CPTAC OOF benchmark / 外部队列 / 注意力可解释性 / NMF 图谱 + Cox + 通路）

---

## 快速状态表

| Panel | 描述 | 状态 | 最关键缺口/疑点 |
|---|---|---|---|
| **Fig2a** | 架构示意图 | **risk** | 边数硬编码、ρ 来源文件不在链中 |
| **Fig2b** | CPTAC OOF 跨组织 benchmark（9方法） | **gap** | TSV 产出脚本路径全为 Ubuntu、无本地执行入口 |
| **Fig2c** | 4 外部队列 benchmark | **gap** | 实际仅 3 队列与正文"4队列"矛盾；deep_methods TSV 生成链断裂 |
| **Fig2d** | 位点图注意力可解释性 | **gap** | analyze 产出文件名与 panel_d.R 读取文件名不匹配；两个 canonical panel_d 脚本功能不同 |
| **Fig2e** | 泛癌 NMF 热图（10,023例） | **gap** | X_nmf.npy 前置推理链断裂；paper_final 版脚本未注册 |
| **Fig2f** | 模块生存 Cox 森林图 | **risk** | 脚本 title 写 n=9,102 与宣称 10,023 不一致 |
| **Fig2g** | Hallmark ORA 条形图 | **complete** | 链路完整，无额外缺口 |

---

## Fig2a — 架构示意图

**状态：risk**

**链路追溯**

| 步骤 | 脚本 | 实际文件 |
|---|---|---|
| 图先验构建 | `2_analysis/bulk/process_copheemap_prior_20260428.py` | 未找到产出 graph_statistics.tsv 的专用脚本 |
| α-scan 确认 | `2_analysis/bulk/run_scp682_shrinkage_sensitivity_grid.py` | 存在 |
| panel 渲染 | `3_plotting/bulk/panel_a.R` | 存在 |

**Grep 实证（panel_a.R 第 101-102 行）**

```
label = "18,592 sites · 420,102 edges"   # 硬编码，非读文件
label = "1,431 samples · 21,925 edges"   # 硬编码，非读文件
```

注释中写明数字来源 `graph_statistics.tsv`，但脚本实际未 `read.delim` 任何 tsv，数字全部 `annotate("text", label=…)` 硬编码在 panel_a.R 内。

**ρ=0.5474** 注释写来自 `headline_metrics.tsv`（`paper_materials_SCP682/01_key_results/`），但无从本地 organized_code 追溯到 headline_metrics.tsv 的产出脚本。

**缺口/疑点**

1. 420,102 / 21,925 / 18,592 三个数字均硬编码，graph_statistics.tsv 专用导出脚本不在链中（REPRODUCE.md 已承认）。
2. `ρ=0.5474` 来源文件 `headline_metrics.tsv` 不在 organized_code 中，无法追溯产出路径。
3. B_phi 训练入口（`train_cptac_total_proteome_film_vae_z_direct_residual_v2_20260429.py`）命名含 `film_vae`，但 REPRODUCE.md 把 `film_vae` 系列列为 legacy 排除模式——此脚本以 "support" 身份保留，需确认其在 canonical 链中的角色是"组件训练"还是"legacy 残留"。

**修正建议**

- 把 graph_statistics.tsv 的产出脚本（或 process_copheemap_prior 的计数输出段）补入链，或在代码注释中明确数字来源行号。
- 检查 headline_metrics.tsv 是否已提交 source data，否则 ρ 无可重现路径。

---

## Fig2b — CPTAC OOF 跨组织 benchmark（9 方法）

**状态：gap**

**链路追溯**

```
数据准备（CPTAC PDC + GDC 下载 + 标准化） → canonical 主训练 e160
  → build_scp682_bulk_main_panels.py（产出 per_site_spearman_with_deep_learning.tsv）
  → panel_b.R（读该 TSV + 现场做 Wilcoxon）
```

**Grep 实证**

panel_b.R 第 14-16 行：
```r
.PANEL_B_ROOT <- "E:/data/gongke/TCGA-TCPA/paper_materials_SCP682"
.PANEL_B_TSV  <- file.path(.PANEL_B_ROOT, "01_key_results",
                           "per_site_spearman_with_deep_learning.tsv")
```

build_scp682_bulk_main_panels.py 第 19-21 行：
```python
MAIN_RELEASE = Path("/data/lsy/Infinite_Stream/SCP682/frozen_release/...")
RNA_PATH = Path("/data/lsy/Infinite_Stream/01_data/...")
OUT_DIR = Path("/data/lsy/Infinite_Stream/SCP682-main/results/...")
```

**缺口/疑点**

1. `build_scp682_bulk_main_panels.py`（产出 TSV 的唯一 canonical 脚本）路径全为 Ubuntu 绝对路径，本地机器无对应数据目录，不能直接在本地机器执行。`per_site_spearman_with_deep_learning.tsv` 已存在于 `paper_materials_SCP682/01_key_results/` 的本地拷贝中，但无法原地重现。
2. panel_b.R 当前显示 9 方法（包含 `DeepGxP_5fold` 和 `VAE`），注释写"v8 (2026-05-26) 加 DeepGxP_5fold 和 VAE"，而 module_bulk.md 链表（步骤 12-16）的基线部分描述比 panel 实际方法少——链文档可能未同步 v8 更新。
3. 配对 Wilcoxon 统计计算混在 panel_b.R（画图脚本）里，无独立统计分析脚本，p 值无独立可重现路径。

**修正建议**

- 记录 Ubuntu 机器执行 `build_scp682_bulk_main_panels.py` 的精确命令与输出哈希，以证明本地 TSV 来源可信。
- 把 Wilcoxon 计算提到独立 Python/R 分析脚本，或在链文档中注明统计计算内嵌于画图脚本。

---

## Fig2c — 外部多队列冻结 benchmark

**状态：gap**

**链路追溯**

```
外部数据下载 → 冻结推理 predict_scp682_general_graph_external.py
  → make_external_9model_benchmark.py
      ├─ per_site_spearman_external.tsv（非深度方法，来源已有脚本）
      └─ per_site_spearman_external_deep_methods.tsv（【来源链断裂】）
  → per_site_spearman_external_9models.tsv
  → panel_c.R
```

**Grep 实证**

make_external_9model_benchmark.py 第 16-17 行：
```python
BASE_EXTERNAL = KEY / "per_site_spearman_external.tsv"
DEEP_EXTERNAL = KEY / "per_site_spearman_external_deep_methods.tsv"  # 产出脚本不在链中
```

panel_c.R 第 51-57 行（队列定义）：
```r
.PANEL_C_DATASET_ORDER <- c("fu_icca", "tu_sclc", "chcc_hbv_fpkm")
# 注释："v9 去掉 CHCC-HBV RSEM：与 RSEM 同批 HCC 病人，非独立队列"
```

make_external_9model_benchmark.py 第 41 行：
```python
DATASET_ORDER = ["fu_icca", "tu_sclc", "chcc_hbv_fpkm", "chcc_hbv_rsem"]  # 4个
```

**缺口/疑点**

1. **队列数量矛盾**：module_bulk.md 描述"4个外部队列"，REPRODUCE.md 也写"4个外部队列"，但 panel_c.R（v9 定稿画图脚本）实际只渲染 3 个队列（去掉了 CHCC-HBV RSEM），文档未同步更新。
2. `per_site_spearman_external_deep_methods.tsv`（包含 DeepGxP_5fold 和 VAE 在外部队列的结果）的生成脚本未在 organized_code 中注册，是已知的 REPRODUCE.md 第 7 节 Gap 4。
3. `run_deepgxp_cptac_half_retrain_20260511.py` 覆盖 CPTAC 内部，`run_deepgxp_tcpa_reproduction_20260507.py` 覆盖 TCPA RPPA，均不产出外部队列 DeepGxP benchmark TSV。

**修正建议**

- 修正 module_bulk.md 的"4个外部队列"为"3个（Fig2c 正式面板）+ 1个补充（CHCC-HBV RSEM）"。
- 补充 `per_site_spearman_external_deep_methods.tsv` 的产出脚本记录，或明确该文件由哪个 run_deepgxp_* 脚本产出。

---

## Fig2d — 位点图注意力可解释性

**状态：gap**

**链路追溯**

```
canonical e160 checkpoint → export_scp682_site_attention.py
  → scp682_e160_site_attention.tsv（注意力边权重）
  → analyze_site_attention.py
      → site_attention_functional_enrichment.tsv  ← ❌ 名称不匹配
      → site_attention_by_label.tsv               ← ❌ 名称不匹配
  → panel_d.R（读 attn_bar_data.tsv）             ← ❌ 找不到生成 attn_bar_data.tsv 的脚本
```

**Grep 实证**

analyze_site_attention.py 第 100-102 行（产出文件）：
```python
enrich.to_csv(OUT / "site_attention_functional_enrichment.tsv", ...)
pd.DataFrame(summ).to_csv(OUT / "site_attention_by_label.tsv", ...)
```

panel_d.R 第 5-6 行（读取文件）：
```r
ROOT <- "E:/data/gongke/TCGA-TCPA/SCP682_MAIN/attention_export"
d <- read.delim(file.path(ROOT, "attn_bar_data.tsv"), ...)
```

两个文件名完全不匹配，`attn_bar_data.tsv` 没有任何脚本产出它（organized_code 内 `grep attn_bar_data` 只在两个 R 画图脚本中出现，无 Python 产出脚本）。

**附加疑点**

- 有两个均标 canonical 的 panel_d 脚本（`3_plotting/bulk/panel_d.R` = 注意力 bar 图；`3_plotting/bulk/panel_d__L.R` = 消融 lollipop/bar/forest）。两者都指向 Fig2d 但功能完全不同，实际发表的 Fig2d 内容不明确。
- panel_d.R 内嵌硬编码数值（`enrich = c(4.73, 1.79)`），与 `_build_fig2d_interp.R` 标注（`4.7× / 1.8×`）略有差异，均非从 TSV 动态计算。
- panel_d.R 和 `_build_fig2d_interp.R` 都用 `setwd("E:/...SCP682_MAIN/attention_export")` 或绝对路径，指向 `SCP682_MAIN`（非 organized_code），数据源不在代码包内。

**修正建议**

1. 确认 `attn_bar_data.tsv` 的产出脚本（很可能存在于 `SCP682_MAIN/attention_export/` 下但未被收录），将其补入 organized_code。
2. 明确两个 panel_d 脚本中哪个对应实际发表图，弃用另一个或备注关系。
3. 将硬编码 4.73/1.79 替换为从 `site_attention_functional_enrichment.tsv` 动态读取，或在链文档中明注"数值来自 attn_bar_data.tsv，与 functional_enrichment.tsv 同源"。

---

## Fig2e — 泛癌 NMF 磷酸化图谱（30模块 × 32癌种）

**状态：gap**

**链路追溯**

```
[缺失] TCGA RNA 下载 → [缺失] TCGA 全量 SCP682 推理 → X_nmf.npy
  → run_nmf.py → W_k30.npy / H_k30.npy
  → analyze_modules.py → module_by_cancer_median_k30.tsv
  → build_figB_module_cancer.R → PDF
```

**Grep 实证**

run_nmf.py 第 3 行注释 + 第 9 行代码：
```python
# Input X_nmf.npy = [Z+, Z-] (10023 primary tumors x 37184 signed features).
X = np.load(OUT + "/X_nmf.npy")
```

在全 organized_code 中 grep `X_nmf`，仅此一处读取，无任何脚本写入 `X_nmf.npy`。

`22_repredict_tcga_full_scp682_from_raw.py` 使用 Ubuntu 路径输出到 `20260529_tcga_full_scp682_main_reprediction_v1`，而 run_nmf.py 读的是 `20260612_tcga_pancancer_nmf_v1/results/X_nmf.npy`——输出目录不匹配，推理脚本与 NMF 输入矩阵之间无文档化的连接步骤。

**缺口/疑点**

1. 推理输出目录（`20260529_tcga_full_...`）与 NMF 输入目录（`20260612_...`）不一致，中间存在数据搬移或格式转换步骤，无脚本记录。
2. `sample_meta.tsv` 和 `site_names.tsv`（analyze_modules.py 的关键输入）无任何产出脚本（organize_code 内 grep `sample_meta.tsv` 仅见读取）。
3. paper_final 版 `fig_tcga_pancancer_atlas/scripts/run_nmf.py`、`analyze_modules.py`、`project_to_cptac.py` 未在 `_copy_index.tsv` 中注册（module_atlas.md GAP-1 已知），当前以 02_results 版作代理，但两者实际内容是否完全一致无脚本级验证。
4. `_build_nmf_heatmap.R`（`3_plotting/atlas/`，来自 `SCP682_MAIN/attention_export/`）读取 `nmf_factor_by_cancer.tsv`（旧 attention_export 路径），与 20260612 NMF 路径不同，但仍标 canonical，可能为过期版本。

**修正建议**

- 补充"从 22_repredict 输出 parquet → 构建 signed-split [Z+, Z-] 矩阵 → 存 X_nmf.npy 及 sample_meta/site_names"的脚本。
- 明确 `_build_nmf_heatmap.R` 的废弃状态，或更新其读取路径至 20260612 结果目录。

---

## Fig2f — NMF 模块生存 Cox 森林图

**状态：risk**

**链路追溯**

```
run_nmf.py → W_k30.npy（同 Fig2e，共享上游缺口）
  → analyze_modules.py → sample_module_activity_k30.tsv + module_summary_k30.tsv
  → build_figD_survival.R → 内嵌 Cox 计算 + 输出 PDF
```

**Grep 实证**

build_figD_survival.R 第 39 行：
```r
title = "Phospho-module activity vs overall survival (n = 9,102)",
```

但论文/REPRODUCE.md 声称 "10,023 TCGA 原发瘤"。

build_figD_survival.R 第 9-11 行的过滤逻辑：
```r
df <- merge(sma, meta[,c("sample_id","survival_time","survival_event")], by="sample_id")
df <- df[!is.na(df$survival_time) & df$survival_time>0 & !is.na(df$survival_event), ]
cat("samples with usable OS:", nrow(df), " events:", sum(df$survival_event), "\n")
```

说明 9,102 是有完整 OS 数据的子集（10,023 全量 → 过滤缺失 OS → 9,102），属于设计合理的样本筛选，但论文摘要/图注若写"10,023"则与图内 title 的"9,102"矛盾，需统一措辞。

**缺口/疑点**

1. 论文声称"10,023 原发瘤"，图 title 显示 `n = 9,102`——两者的分母在摘要/图注和图内不一致，需核对论文正文 Fig2f 图注写的哪个数。
2. Cox 统计和画图在同一 R 脚本中完成（all-in-one），无独立统计输出文件可供审查。
3. Fig2e 的上游缺口（X_nmf.npy 前置链断裂）同样影响 Fig2f，此处继承。

**修正建议**

- 确认论文正文 Fig2f 图注样本量措辞，与脚本 `n=9,102` 保持一致，或在正文明确"10,023 例中 9,102 例有完整 OS 数据"。

---

## Fig2g — 代表性模块 Hallmark ORA 条形图

**状态：complete**

**链路追溯**

```
run_nmf.py → analyze_modules.py（产出 module_pathway_enrichment_k30.tsv + module_summary_k30.tsv）
  → build_figC_pathway_bars.R → PDF
```

所有三步脚本均在 `organized_code/2_analysis/atlas/` 和 `3_plotting/atlas/` 中，路径一致（`E:/data/gongke/TCGA-TCPA/02_results/model_validation/20260612_tcga_pancancer_nmf_v1`），无文件名不匹配、无路径跳机。

**唯一继承风险**：上游 run_nmf.py 依赖 X_nmf.npy（同 Fig2e 缺口），若 X_nmf.npy 无法重建，则此链虽结构完整但不可执行。

---

## 综合疑点（跨 panel）

| # | 疑点 | 影响范围 |
|---|---|---|
| G1 | X_nmf.npy 前置链断裂（推理脚本输出目录 ≠ NMF 输入目录，缺转换步骤） | Fig2e/f/g |
| G2 | `attn_bar_data.tsv` 无任何脚本产出（analyze_site_attention.py 输出的是不同文件名） | Fig2d |
| G3 | 外部队列数量矛盾（文档写 4 个，定稿画图脚本只渲染 3 个） | Fig2c |
| G4 | `build_scp682_bulk_main_panels.py` 路径全为 Ubuntu，本地无法复现 TSV | Fig2b |
| G5 | 图 title n=9,102 与文档声称 10,023 未在正文统一 | Fig2f |

---

## 核验方法说明

- 使用 Read 直接读取 organized_code/ 下的实际脚本，核查输入 `read.delim` / `np.load` / `pd.read_csv` 路径。
- 使用 Grep 全量搜索 `X_nmf`、`attn_bar_data`、`per_site_spearman_*` 等关键文件名，确认产出方。
- 检查 `_copy_index.tsv` 的 canonical 字段，确认脚本身份（无 film_vae/SCP682-7~28/stacking/72-token 混入）。
- 不跑代码，只做静态链路追溯。
