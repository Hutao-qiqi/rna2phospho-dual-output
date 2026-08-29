# SCP682-SC15 Dual-Input Ensemble

## 正式模型

正式模型采用双输入直接预测集成：

- 每折训练集确定的 4,000 个高变 RNA 基因；
- 冻结的 scFoundation 细胞表示；
- 高容量残差 RNA 编码塔；
- 非线性融合层与磷酸化输出头；
- 条件感知训练批次；
- Huber、相关性、排序、伪批量、条件内变化和跨板对齐损失；
- 每个外层折训练三个独立模型；
- 根据验证集选择均值集成或秩集成。

每个外层折独立完成模型选择。69001、69003 和 69004 折采用秩集成，69002 和 69005 折采用算术均值集成。

## 性能

| 评价方案 | 总体 Spearman | 条件内 Spearman | 伪批量 Spearman |
|---|---:|---:|---:|
| 留一板五折均值 | 0.4658 | 0.3594 | 0.6479 |
| 分层随机 81/9/10 三种子均值 | 0.7180 | 0.7034 | 0.7942 |

同预算比较使用三模型算术均值集成：

| 模型 | 总体 Spearman | 条件内 Spearman | 伪批量 Spearman |
|---|---:|---:|---:|
| cTP 风格三模型集成 | 0.4345 | 0.3333 | 0.6485 |
| sciPENN 风格三模型集成 | 0.4499 | 0.3526 | 0.6486 |
| SCP682-SC15 三模型集成 | **0.4652** | **0.3620** | **0.6494** |

折×位点配对统计显示，SC15 相对两种同预算对照在总体和条件内相关上均达到统计显著。伪批量差异的置信区间跨零，按数值接近报告。

## 主要文件

- `FINAL_MODEL_LOCK.json`：模型名称、性能、折内选择和推理入口。
- `model_hashes.tsv`：15 个冻结权重的文件大小和 SHA-256。
- `hvg4000_ordered.txt`：五个训练折的 4,000 基因顺序。
- `input_normalization.json`：RNA、scFoundation 表示和输出变换规则。
- `readout_schema.tsv`：20 个磷酸化读数定义。
- `ensemble_config.json`：五折、三成员及集成方式。
- `external_validation_protocol.md`：冻结外部验证协议。
- `final_performance_summary.tsv`：主要性能对照表。
- `model_selection.tsv`：结构筛选记录。

## 获取权重

15 个权重按训练折拆分为五个 GitHub Release 附件：

[`scp682-sc15-dual-input-ensemble-v1`](https://github.com/Hutao-qiqi/rna2phospho-dual-output/releases/tag/scp682-sc15-dual-input-ensemble-v1)

下载五个 `SCP682-SC15_models_split_*.zip` 后，在本目录执行：

```powershell
Get-ChildItem SCP682-SC15_models_split_*.zip | ForEach-Object {
  Expand-Archive -LiteralPath $_.FullName -DestinationPath . -Force
}
```

文件校验值见 `release_asset_hashes.tsv` 和 `model_hashes.tsv`。

## 推理

输入为细胞顺序一致的 RNA `h5ad` 和冻结 scFoundation 表示：

```powershell
python code/predict_scp682_sc15_locked.py `
  --release-dir . `
  --rna-h5ad <RNA.h5ad> `
  --scfoundation-embeddings <embeddings.npy> `
  --output-dir <output_dir> `
  --device cuda:0
```

输出包含 20 个读数的最终预测矩阵以及五折预测。外部评价按 `external_validation_protocol.md` 执行。

## 服务器模型目录

`D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260829_scp682_sc15_direct_prior_residual_v1`

各折模型保存在 `A2_seed69001` 至 `A2_seed69005` 以及 `direct_ensemble/split_*`。
