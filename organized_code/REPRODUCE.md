# SCP682 论文复现地图（REPRODUCE.md）

生成日期：2026-06-16　｜　模型：SCP682（投 Nature Methods）　｜　范围：Fig 1–5 + ED/Supp

本目录 `organized_code/` 是**跨三台工作区精选出的 canonical 复现代码集**，按 **训练 / 分析 / 画图** 三类整理，并附每张主图的复现链。

> ⚠ **零改动承诺**：本目录内全部为**复制件**。本地 `E:\`、Ubuntu `/data/lsy/Infinite_Stream`、Windows-WSL `/mnt/d/data/lsy/vm_lsy_parent/lsy` 三处的原始文件**一律未被改动或删除**。

---

## 1. 目录结构

```
organized_code/
├─ 1_training/      模型训练：拟合并产出权重的脚本、模型 engine/架构、训练启动包装、训练输入构建
│   ├─ bulk/  sc/  ppko/
├─ 2_analysis/      数据分析：下载/预处理、推理预测、外部验证、基线对照、NMF图谱、生存Cox、统计、P100、注意力分析、图源数据表
│   ├─ bulk/  sc/  ppko/  atlas/  clinical/  shared/  data/
├─ 3_plotting/      画图：渲染图/panel 的脚本（panels/*.R、build_fig*、make_fig*）
│   ├─ bulk/  sc/  ppko/  atlas/  clinical/  shared/
├─ reproduction_chains/   每模块「数据准备→训练→分析→画图」复现链（module_*.md）+ 复制清单
├─ _copy_index.tsv        全部复制文件的索引：category / module / 包内路径 / 原始相对路径 / 来源机器 / canonical / 用途
└─ REPRODUCE.md           本文件
```

文件计数：**训练 50 ｜ 分析 250 ｜ 画图 115 ＝ 共 415 个脚本**（去重后）。

**来源机器标记**（`_copy_index.tsv` 的 machines 列）：`L`=本地 `E:\data\gongke\TCGA-TCPA`；`U`=Ubuntu `/data/lsy/Infinite_Stream`；`W`=Windows-WSL `/mnt/d/data/lsy/vm_lsy_parent/lsy`。多字母=该文件内容在多机一致。

---

## 2. 三机整合与 canonical 甄选方法

1. **建底账**：三机代码文件全量 sha256 清点（排除环境/第三方/`99_archive`/临时）→ 去重得 **1,387 个唯一文件**。
2. **多智能体分类**：32 个 agent 逐文件略读，判定 模块 / 训练·分析·画图 / `canonical·legacy·support·exploratory·thirdparty`，覆盖 1,230 个 deep 文件。
3. **精选复现集**：取 `canonical` + `support` = **415 个**，复制入本目录；`legacy`(471)/`exploratory`(310)/`thirdparty`(30) 一律排除。
4. **复现链综合**：按模块构建 Fig→脚本链（见 `reproduction_chains/`）。

---

## 3. Canonical 模型身份（每模块唯一权威版本）

| 模块 | 论文图 | Canonical 模型 | 关键训练脚本 | checkpoint |
|---|---|---|---|---|
| **bulk** | Fig2 a–d | SCP682-22 / PORTABLE，v4 exact_scnet 双轴GNN，`Ŷ=B_φ+0.3·Δ` | `remote_scripts/train_scp682_exact_scnet_gnn_v1.py`；论文包版 `paper_materials_SCP682/03_code/training/train_scp682_general_graph_residual.py` | `scp682_main_v4_exact_scnet_gnn_best.pt` |
| └ engine/推理 | — | 部署引擎 | `portable_src/scp682_v4_engine.py`（baseline）、`portable_src/scp682_graph_runtime.py`（图残差runtime）、`portable_src/predict_scp682.py` | `scp682_graph_runtime_state.pt` |
| **atlas** | Fig2 e–g | 泛癌 signed-split NMF k=30（10,023 原发瘤） | `02_results/model_validation/20260612_tcga_pancancer_nmf_v1/scripts/run_nmf.py` | （NMF W/H 矩阵） |
| **sc** | Fig3 | SCP682-SC11，scFoundation + pathway attention + expanded ScNET site-GNN | `paper_materials_SCP682_SC11/03_code/architecture/train_scp682_sc11_expanded_scnet_site_gnn.py` | sc11 best.pt |
| **ppko** | Fig4 | SCP682-PPKO V10B strong300 | `paper_materials_SCP682_PPKO/03_code/training/pretrain_v10b_strong300.py` | `scp682_ppko_v10b_strong300_best.pt` |
| **clinical** | Fig5 | 复用 bulk 预测谱 → KIRC/RPS6 beyond-parent-mRNA 预后（BH-FDR / 联合Cox LRT） | 见 `reproduction_chains/module_clinical.md` | — |

---

## 4. 各主图复现链（详见 reproduction_chains/）

| 论文图 | 内容 | 复现链文档 |
|---|---|---|
| **Fig2 a–g** | bulk RNA→phospho 架构 / 跨组织benchmark(对8基线) / 4外部队列 / 注意力可解释性 / 泛癌NMF图谱+生存+通路 | [`module_bulk.md`](reproduction_chains/module_bulk.md) ＋ [`module_atlas.md`](reproduction_chains/module_atlas.md) |
| **Fig3** | 单细胞 SCP682-SC11：跨通路/平台/物种、内部图消融 | [`module_sc.md`](reproduction_chains/module_sc.md) |
| **Fig4** | PPKO 药物诱导磷酸化扰动、P100验证(n=125)、患者级TCGA-TCPA响应 | [`module_ppko.md`](reproduction_chains/module_ppko.md) |
| **Fig5** | 临床：KIRC deep-dive、pS6/RPS6 beyond-parent 预后、SC恶性程序 | [`module_clinical.md`](reproduction_chains/module_clinical.md) |
| **Fig1** | 三模型统一框架示意图（无独立复现代码；共享引擎/图先验见 bulk 与各模块 shared/data） | — |

每份 `module_*.md` 对该模块每个 panel 给出有序链「数据准备→训练→分析→画图」、每步脚本的 rel + 机器 + category，并在末尾列出**版本漂移说明**与**Gap 清单**。

---

## 5. 三分类定义

- **训练（1_training）**：拟合并产出模型权重的脚本（`train_*`/`pretrain_*`/`retrain`）、模型 engine/架构代码、训练启动包装、构建训练输入的数据准备。
- **分析（2_analysis）**：数据下载/预处理、推理预测（`predict`/`infer`）、外部验证、基线对照、NMF泛癌图谱、生存/Cox、统计、P100验证、注意力分析、图源数据表生成。
- **画图（3_plotting）**：真正渲染图/panel 的脚本（`paper_final/*/scripts/panels/*.R`、`build_fig*`、`make_fig*`、`plot_*`）。

> 同一复现链会横跨三类（如 Fig2b：下载预处理→训练→benchmark分析→`panel_b.R` 画图）。`reproduction_chains/module_*.md` 保留完整链路顺序；本三类目录是按**功能**物理分开，便于查阅与打包。

---

## 6. 已排除的 legacy（不进复现集）

由 Foundation 阶段跨三机核验，下列命名族为被取代的旧版/探索版，已排除：

`film_vae` / `rppa_film_vae` / `cvae` 残差系列；`72-pathway-token`（旧架构）；`SCP682-7…21` 与 `SCP682-23…28`（dev 编号变体：pathway_dropout / denovo_teacher / error_aware / attnres 等）；`SCP682-30/31/32`（exact_scnet 的前身）；单细胞 `sc1/sc2/sc3` 及 `sc4…sc10`（SC11 前身）；`scko/scko1/scko3`；PPKO 的 `v5…v22` 与 `v10c`（v10b 之外）；`dann/coral` locked 基线；`_archive` / `OBSOLETE` / `experimental_model_archive`；`remote_scripts/_paper_extract_sources`（中转副本）。

---

## 7. 复现状态与已知缺口（诚实披露）

**已完成**：①三机底账+去重 ②全量分类（1,230 deep 文件）③精选 415 canonical 复制入库 ④Fig2–5 五个模块复现链（bulk/atlas/sc/ppko/clinical）。

**对抗式验证结论**（`reproduction_chains/verify_Fig2..5.md` + `gap_resolution.md`，逐脚本 grep 印证；已全部完成）：
- **Fig2**：panel g 链最完整。缺口：(d) `analyze_site_attention.py` 产 `site_attention_functional_enrichment.tsv` 但 `panel_d.R` 读 `attn_bar_data.tsv`，名不符、缺转换脚本；(e–g) `run_nmf.py` 读 `20260612_*/X_nmf.npy`，而全 TCGA 推理 `22_repredict_tcga_full_*` 输出在 `20260529_*`，中间 signed-split 矩阵构建步骤无脚本；(c) 文档写"4 外部队列"但定稿 `panel_c.R` 只渲染 3 个；(f) `n=9,102`(有OS子集) vs 文档 10,023 需统一措辞；(b) 唯一产表脚本全为 Ubuntu 绝对路径。
- **Fig3**：外部推理脚本多指向旧 checkpoint 路径 `20260522_`（非 5 折正式版 `20260529_`），需确认论文 final 用哪个；Blair、Vivo-seq Th17 预处理脚本缺失；m4 图消融(Fig3g)用 SC7 warm-start 而 full model 无，配对基线不齐。
- **Fig4**：PPKO 已统一到 `SCP682_PPKO_V10B_transferable` 冻结包；训练脚本会输出正式权重 `scp682_ppko_v10b_strong300_best.pt`，并保留旧名别名；P100 全药物验证脚本和 TCGA 患者响应脚本均从冻结包导入 `pretrain_v10b_strong300.py` 与正式权重；TCGA AUC bootstrap 脚本=`_server_ppko_tcga_stats.py`（已补入 `2_analysis/ppko/`）。
- **Fig5**：panel d(KIRC KM+Cox，BH-FDR+联合Cox LRT) 口径正确、链最完整；panel f(DepMap)/g(全位点模块)/h(PTM-SEA) 上游数据准备脚本**缺失(需新建)**；`20260530_fig5_exact_site_anchor_search_v1` 产出脚本未入册；panel b 仍用 nominal-p(非 BH-q)；文件名残留 `fig5b_`/`fig5d_`/`panel_c_` 历史前缀。
- **gap_resolution**：28 项 needs_review 在 Ubuntu 暂存下均找到实体；**已补 15 个 support 脚本**入 `2_analysis/{ppko,shared,data}/`（AUC bootstrap、`compute_gsva.R`、TCGA/TCPA 下载与 RNA 预处理 R、外部验证 delta 计算等）。

**仍待办（需作者新建/确认，非本次整理可补）**：Fig5 f/g/h 与 anchor-search 的少数生成脚本确不在任何机器（需新建或从 DepMap/GDC 手动产生）；Fig3 的 `20260522_` vs `20260529_` checkpoint 身份待作者确认。

**复现入口建议**：先读对应 `reproduction_chains/module_*.md` 看链路顺序 → 在 `1_training/`→`2_analysis/`→`3_plotting/` 对应模块子目录取脚本。原始数据与权重不在本代码包内，路径见各脚本头部与 `SCP682_CURRENT.json` / `SCP682_PPKO_CURRENT.json`。
