# organized_code — SCP682 论文复现代码包

把 SCP682 论文（Fig 1–5）的 **canonical 复现代码**从三台工作区（本地 / Ubuntu / Windows-WSL）跨机去重、精选、并按 **训练 / 分析 / 画图** 三类整理。

当前 bulk 主模型入口以 `2_analysis/bulk/predict_scp682.py` 为准。该入口先用冻结的磷酸化状态估计器 `S_phi` 从 bulk RNA 和样本上下文生成全位点初始状态，再用图约束残差算子输出 `0.3 * Delta` 修正项，最终得到样本级 phosphosite 丰度预测。历史文件名和历史编号只代表内部组件来源或旧实验记录，不再作为论文主模型名称使用。

- **怎么用**：先看 [`REPRODUCE.md`](REPRODUCE.md)（总复现地图）→ 再看 [`reproduction_chains/`](reproduction_chains/) 里对应模块的 `module_*.md`（每张图的「数据准备→训练→分析→画图」有序链）→ 到 `1_training/ 2_analysis/ 3_plotting/` 下对应模块子目录取脚本。
- **主模型入口审计**：见 [`CURRENT_SCP682_MAIN_AUDIT.md`](CURRENT_SCP682_MAIN_AUDIT.md)，其中列出已替换的旧预测入口、当前 portable 运行时和保留历史组件的原因。
- **索引**：[`_copy_index.tsv`](_copy_index.tsv) 列出每个文件的 类别 / 模块 / 包内路径 / 原始相对路径 / 来源机器 / canonical 标记 / 用途。

```
1_training/   50   训练（含模型 engine/架构、训练输入构建）
2_analysis/  250   分析（下载预处理、推理、外部验证、基线、NMF、生存、统计、图源数据）
3_plotting/ 115   画图（panels/*.R、build_fig*、make_fig*）
共 415 个脚本（去重）
```

> ⚠ 全部为**复制件**。三台机器上的原始文件未被改动或删除。原始数据/权重不在本包内，路径见各脚本头部及根目录 `SCP682_CURRENT.json` / `SCP682_PPKO_CURRENT.json`。若需要可复制预测包，请使用根目录 `SCP682_PORTABLE/`。

构建方式：三机 sha256 全量清点去重（1,387 唯一文件）→ 多智能体逐文件分类（canonical/legacy/support/...）→ 取 canonical+support 复制入库。详见 `REPRODUCE.md` 第 2、7 节（含已知缺口与未跑完的 verify 阶段说明）。
