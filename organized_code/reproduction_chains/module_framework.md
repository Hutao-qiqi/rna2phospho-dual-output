# SCP682 框架层与共享基础设施复现链（module_framework.md）

生成日期：2026-06-16　｜　对应论文图：Fig1（示意）+ 各模块共享层

---

## 关于 Fig1

Fig1 为三模型统一框架示意图，**无独立绘图代码**。有两处绘图脚本覆盖该图内容：

| 脚本 rel | 机器 | primary_category | 说明 |
|---|---|---|---|
| `04_figures/fig1/make_fig1_rebuild.py` | L | shared / 画图 / canonical | matplotlib 绘制 bulk/PPKO/SC 三模型总览示意图（SVG/PDF），读入 graph_statistics（边数等数字标注） |
| `paper_materials_SCP682_SC11/03_code/visualization/generate_fig1a_scp682_sc_architecture.py` | L | sc / 画图 / canonical | 程序生成 SC11 架构示意图（scFoundation→pathway→site→ScNET GNN 流程，116.675mm×58.535mm） |

两者均为说明性示意，无需训练产物即可独立运行。运行 `make_fig1_rebuild.py` 需要事先从各模块分析结果中读入边数统计（硬编码或从 `graph_statistics` 目录读取），该统计数字散落于各模块 `launch_*` 脚本输出中，**无单一导出脚本**（见 Gap 节）。

---

## (a) 模型引擎 / runtime

三个 portable 引擎文件组成 bulk 推理核心，跨 bulk / SC / clinical 三模块均需调用。

| 脚本 rel | 机器 | primary_category | 用途 | 服务于 |
|---|---|---|---|---|
| `portable_src/scp682_v4_engine.py` | LUW | bulk / 训练 / canonical | SCP682 v4 baseline 引擎封装：加载 CPTAC 多任务冻结 baseline 并推理，被 `predict_scp682.py` 调用，输出 v4 baseline 磷酸化矩阵 | Fig2a, Fig2b |
| `portable_src/scp682_graph_runtime.py` | LUW | bulk / 训练 / canonical | 图残差 runtime 引擎：`SCP682GraphDecoder`（`baseline + 0.3 * delta`）+ `SCP682GraphRuntime` 推理核心，加载 `scp682_graph_runtime_state.pt` | Fig2a, Fig2b, Fig2c |
| `portable_src/predict_scp682.py` | LUW | bulk / 分析 / canonical | bulk canonical 可迁移预测入口：读取 RNA 矩阵 → v4 baseline → 图残差 → 输出 per-site 预测值；外部队列、泛癌图谱、临床预测均以此为入口 | Fig2b, Fig2c, Fig2e（Fig5 通过 bulk 预测谱复用） |
| `portable_src/scripts/export_graph_runtime_state.py` | LU | bulk / 分析 / support | 从 `scp682_main_v4_exact_scnet_gnn_best.pt` 提取 row/col embedding + decoder 参数，打包为 `scp682_graph_runtime_state.pt`（portable 推理包的前置步骤） | Fig2a |
| `SCP682_PORTABLE/scripts/export_graph_runtime_state.py` | LW | bulk / 分析 / canonical | 同上的论文包 curated 版，路径为 `SCP682_PORTABLE/`；两者功能等同，以此版为权威 | Fig2a, Fig2b, Fig2c |

**注意**：`scp682_graph_runtime_state.pt` 是从 canonical 训练权重 `scp682_main_v4_exact_scnet_gnn_best.pt`（Windows-WSL 机器产出）导出后的精简推理包，portable 三件套（`scp682_v4_engine.py` / `scp682_graph_runtime.py` / `predict_scp682.py`）在 L/U/W 三机均有一致副本，是跨机部署的最小推理单元。

---

## (b) 图先验构建

SCP682 三模块各自使用独立图先验，但共享 CoPheeMap / CoPheeKSA / KSTAR 三个原始数据库来源。

### bulk 模块：位点图（site graph）先验

| 脚本 rel | 机器 | primary_category | 用途 | 服务于 |
|---|---|---|---|---|
| `03_code/model_validation/priors/process_copheemap_prior_20260428.py` | LU | bulk / 分析 / support | 处理 CoPheeMap / CoPheeKSA 原始数据，构建 bulk 磷酸化位点图先验（420k+ 边，site_map.parquet + edge 表格） | Fig2a, Fig2b |
| `remote_scripts/summarize_copheemap_overlap.py` | L | bulk / 分析 / support | 统计 CoPheeMap / KSA / n2v 与模型位点的序列重叠，用于图先验覆盖审计（支撑 Fig2a 图边统计数字） | Fig2a, Fig2d |
| `remote_scripts/audit_original_copheemap_local.py` | L | shared / 分析 / support | 本地审计 CoPheeMap / CoPheeKSA / n2v 原始文件与 SCP682 靶位点重叠覆盖率（图先验质控） | Fig2a, Fig3a |
| `remote_scripts/audit_original_gnn_tables.py` | LU | shared / 分析 / support | Ubuntu 路径版：审计 CoPheeMap / n2v / KSA 与 SCP682 靶点重叠（Ubuntu 路径） | Fig2a, Fig3a |
| `remote_scripts/launch_scp682_m4_knowledge_graph_controls_e160_windows.ps1` | LW | bulk / 训练 / support | 串行启动 4 组知识图来源消融（rewired / no_copheemap / no_copheeksa / no_kstar），产出 Fig2d 边源消融汇总 | Fig2d |

**说明**：bulk 训练器（`train_scp682_general_graph_residual.py` / `train_scp682_exact_scnet_gnn_v1.py`）通过 `--prior-dir` 参数读入上述先验；论文包版 `paper_materials_SCP682/03_code/priors/` 目录下亦有独立的 `process_copheemap_prior_*.py` curated 版本，与本地路径内容一致。

### PPKO 模块：有符号磷酸化调控图先验

PPKO 不复用 bulk 图先验，而是针对扰动预测场景另建有符号先验图，层级如下：

| 脚本 rel | 机器 | primary_category | 用途 | 服务于 |
|---|---|---|---|---|
| `remote_scripts/build_kstar_edges_for_scp682_ppko_v5.py` | LW | ppko / 分析 / support | 将 KSTAR 激酶底物预测网络边映射到 PPKO 训练靶点索引（v5 版，被 v9/v10 引用） | Fig4a |
| `remote_scripts/build_copheemap_prior_for_scp682_ppko_v6.py` | LW | ppko / 分析 / support | 为 PPKO V10B 构建基于 CoPheeMap 的磷酸化位点图先验（v6 版，kmer 序列匹配） | Fig4a |
| `remote_scripts/build_signed_phospho_regulatory_prior_v9.py` `paper_materials_SCP682_PPKO/03_code/preprocessing/build_signed_phospho_regulatory_prior_v9.py` | LW / L | ppko / 训练 / support+canonical | 构建 PPKO V10B 所需有符号磷酸化调控先验图 v9（CoPheeMap + KSTAR + PhosphoSitePlus），是 `build_global_phosphoprotein_heterograph_v10.py` 的上游 | Fig4a |
| `remote_scripts/build_global_phosphoprotein_heterograph_v10.py` `paper_materials_SCP682_PPKO/03_code/preprocessing/build_global_phosphoprotein_heterograph_v10.py` | LW / L | ppko / 分析+训练 / support+canonical | 构建 PPKO V10B 训练所需全局磷酸蛋白异质图（8192 site + 8751 protein 节点，STRING700 top50 PPI）；是 `pretrain_v10b_strong300.py` 的必需前置 | Fig4a |

**图先验层级**（PPKO 依赖顺序）：
```
KSTAR edges (v5)
CoPheeMap prior (v6)      ──►  signed_phospho_regulatory_prior_v9
PhosphoSitePlus              ──►  global_phosphoprotein_heterograph_v10
                                 ──►  pretrain_v10b_strong300.py (PPKO canonical 训练)
```

### SC 模块：bulk→SC transfer 先验

SC11 不从头构建图先验，而是从 bulk canonical 模型抽取 teacher prior：

| 脚本 rel | 机器 | primary_category | 用途 | 服务于 |
|---|---|---|---|---|
| `paper_materials_SCP682_SC11/03_code/preprocessing/export_scp682_main_sc_transfer_prior.py` | LW | sc / 训练 / canonical | 从 bulk SCP682-22（`SCP682_PORTABLE` runtime state + `exact_scnet_gnn` checkpoint + observed/baseline phosphosite parquet）抽取 teacher transfer prior（site attention + bulk Spearman），产出 `scp682_main_sc_transfer_prior_v1`；是 SC11 训练的必需数据准备前置 | Fig3a, Fig3g |

**说明**：SC11 同时引用 bulk 的 site graph 先验边（CoPheeMap / CoPheeKSA / KSTAR）参与图消融（Fig3g），但通过 `export_scp682_main_sc_transfer_prior.py` 的 attention 权重转移完成，而非直接重建图；SC 图消融启动脚本 `remote_scripts/run_scp682_sc11_m4_graph_controls.bat` 以 `rewired_all / no_copheemap / no_copheeksa / no_kstar` 四模式复现 Fig3g 消融结果。

---

## (c) 跨模块通用数据准备与工具

### 原始数据下载（module = data，canonical/support）

| 脚本 rel | 机器 | primary_category | 用途 | 服务于 |
|---|---|---|---|---|
| `03_code/model_validation/download_pdc_phosphoproteome_quant_matrices.py` | LU | data / 分析 / support | 从 PDC GraphQL API 下载 CPTAC 磷酸化蛋白组 quantDataMatrix（bulk 训练主数据） | Fig2b |
| `03_code/model_validation/download_pdc_phosphoproteome_report_files.py` | LU | data / 分析 / support | 从 PDC 下载 CPTAC 磷酸化蛋白组报告文件（bulk 训练数据准备） | Fig2b |
| `03_code/model_validation/download_pdc_proteome_report_files.py` | LU | data / 分析 / support | 从 PDC 下载 CPTAC 总蛋白报告文件（伴随蛋白质组，bulk 训练辅助） | Fig2b |
| `03_code/model_validation/download_cptac_gdc_open_star_counts.py` | LU | data / 分析 / support | 下载 CPTAC 各研究对应的 GDC 开放 STAR count RNA 文件（bulk 训练 RNA 矩阵） | Fig2b |
| `03_code/model_validation/build_tcpa_32_project_rna_rppa_contract_20260501.py` | LU | data / 分析 / support | 为 TCGA-TCPA 32 项目下载缺失 RNA 并构建 RNA/RPPA 配对锁定矩阵 | — |
| `03_code/external_validation/proteogenomics/download_requested_pdc_contexts_20260429.py` | LU | data / 分析 / support | 从 PDC/GDC 下载扩展 CPTAC 磷蛋白组/蛋白质组/RNA 原始数据（训练数据准备） | — |
| `remote_scripts/run_signal_seq_fastq_download.cmd` | LW | data / 分析 / support | 下载 SIGNAL-seq GSE256405 原始 FASTQ（SC 外部验证队列原始数据） | Fig3b, Fig3c, Fig3f |
| `remote_scripts/run_signal_seq_processed_h5ad_download.cmd` | LW | data / 分析 / support | 下载 SIGNAL-seq GSE256405 预处理 h5ad（SC 外部验证预处理版） | Fig3b, Fig3c, Fig3f |

### 共享分析工具（module = shared，canonical/support）

| 脚本 rel | 机器 | primary_category | 用途 | 服务于 |
|---|---|---|---|---|
| `_codeorg_staging/ubuntu/compute_gsva.R` | U | shared / 分析 / support | 通用 GSVA 计算脚本：Ensembl→symbol 映射后计算通路得分（多模块共用辅助工具） | Fig2e, Fig5（通路富集面板） |
| `_codeorg_staging/ubuntu/03_code/pathway_prior/build_modeling_prior_v1.py` | U | shared / 分析 / support | 从 PhosphoNetworks + OmniPath + KinHub 构建 SCP682 共享激酶底物先验 v1（graph prior 数据准备上游） | — |
| `_codeorg_staging/ubuntu/03_code/pathway_prior/audit_open_kinase_prior.py` | U | shared / 分析 / support | 审计 PhosphoNetworks + OmniPath 激酶底物先验的覆盖率和质量（先验质控） | — |
| `_codeorg_staging/ubuntu/03_code/pathway_prior/augment_kinhub_audit.py` | U | shared / 分析 / support | 用 KinHub 人类激酶列表补充审计激酶覆盖率（先验构建辅助工具） | — |
| `SCP682_MAIN/scripts/check_remote_torch_env.py` | LW | shared / 分析 / support | 检查远程 PyTorch / CUDA / torch_geometric 环境版本，训练启动前环境诊断工具 | — |
| `paper_final/fig5/scripts/panels/panel_fig2_style.R` | L | shared / 画图 / support | 从 Fig2 复制视觉语法到 Fig5 的共享样式常量（方法颜色 / 热图色板 / 主题函数） | Fig5a, Fig5d-i |

---

## 各图先验与模块对应关系汇总

| 先验类型 | 原始来源 | 服务模块 | 服务图 |
|---|---|---|---|
| bulk 位点图先验（420k+ 位点边） | CoPheeMap v1 + CoPheeKSA | bulk、SC（teacher prior 中） | Fig2a/b/c/d、Fig3a/g |
| bulk 样本图先验（21,925 样本边） | SCP682 exact_scnet site/sample 双轴 | bulk | Fig2a/d（样本轴消融） |
| PPKO 有符号调控先验 v9 | CoPheeMap v6 + KSTAR v5 + PhosphoSitePlus | ppko | Fig4a |
| PPKO 异质图 v10（site+protein节点） | signed_prior_v9 + STRING v12 + UniProt | ppko | Fig4a |
| SC teacher transfer prior v1 | bulk `scp682_graph_runtime_state.pt` + exact_scnet attention | sc | Fig3a/g |
| SC site graph（消融用） | CoPheeMap + CoPheeKSA + KSTAR（复用 bulk 先验边） | sc | Fig3g（消融比较） |

---

## Gap 清单（已知缺口）

1. **图边数字散落**：bulk 图的「420,102 位点边」与「21,925 样本边」等统计数字未有单一导出脚本，数字硬编码或散落于 `launch_*` 脚本输出，`make_fig1_rebuild.py` 读入的 `graph_statistics` 目录来源未在记录中追踪到。
2. **bulk 样本图构建脚本缺失**：records 中搜索 `sample_graph` 无 canonical 脚本命中；样本图边表的具体构建路径不在 `process_copheemap_prior_*` 脚本中，推测在 `train_scp682_exact_scnet_gnn_v1.py` 内部一次性生成，**无独立导出脚本**。
3. **`build_modeling_prior_v1.py` 未被下游引用**：该 Ubuntu 上的激酶底物先验 v1 与 bulk 使用的 CoPheeMap/CoPheeKSA 关系不明，records 中无 canonical 脚本显式 `--prior` 指向该文件，疑为早期探索性上游步骤。
4. **`_codeorg_staging/ubuntu/compute_gsva.R` 未被 Fig2e 显式引用**：该脚本为 shared / support，但 atlas 复现链（`module_atlas.md`）未将其列为必需前置；其在 Fig5 通路富集中的角色待 `module_clinical.md` 核查确认。
5. **CoPheeMap 第三方包笔记本（Ubuntu）**：`CoPheeKSA_XGBoost.ipynb` 与 `PSSM.ipynb` 标记为 `thirdparty`，系 CoPheeMap 原始包随附的训练笔记本，不进复现集，仅供溯源参考。
