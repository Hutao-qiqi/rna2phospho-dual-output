# SCP682-SC15 外部验证协议

## 固定输入

- 五个训练折，每折三个模型成员；
- 每折固定的 4,000 个高变基因及顺序；
- 冻结的 scFoundation 表示；
- 20 个磷酸化读数及抗体映射；
- 每折 RNA 与输出标准化参数；
- 折内集成方式和五折平均方式。

## 零样本外部验证

外部数据只执行锁定的 RNA 归一化、缺失基因均值填补、scFoundation 表示提取和模型推理。外部磷酸化标签不参与模型选择、参数更新、输出校准、位点筛选或集成权重调整。

逐队列报告以下指标：

- 可映射且具有非零观测方差的读数数量；
- 每个读数的细胞级 Spearman 相关；
- 条件内 Spearman 相关；
- 条件级伪批量 Spearman 相关；
- 逐读数结果和置信区间。

抗体或位点定义不一致的读数保留在映射表中并标记，不并入同位点评价。任何使用外部标签的校准单独归类为少样本适配实验。

## 推理命令

```powershell
python code/predict_scp682_sc15_locked.py `
  --release-dir . `
  --rna-h5ad <RNA.h5ad> `
  --scfoundation-embeddings <embeddings.npy> `
  --output-dir <output_dir> `
  --device cuda:0
```
