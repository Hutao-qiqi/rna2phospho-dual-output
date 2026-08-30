# SCP682-ChemState-MODZ

SCP682-ChemState-MODZ 是冻结的化学药物磷酸化扰动预测模型。模型根据具体药物结构、匹配基线磷酸化状态、细胞系、剂量和处理时间，预测 45 个 P100 磷酸化分析物的 MODZ 共识变化量。

## 冻结模型

| 模型 | 训练范围 | 条件数 | 全位点余弦 | 全位点斯皮尔曼 |
|---|---:|---:|---:|---:|
| `SCP682-ChemState-MODZ-KI` | 52 种激酶相关抑制剂 | 370 | 0.5919 ± 0.0054 | 0.5671 ± 0.0056 |
| `SCP682-ChemState-MODZ-All` | 118 种结构可解析药物 | 849 | 0.6220 ± 0.0038 | 0.5925 ± 0.0062 |

两套模型统一采用 Q1 标签：按重复样本整谱相关性加权形成 MODZ 条件共识。

## 文件

```text
models/
  scp682_chemstate_modz_ki.pt
  scp682_chemstate_modz_all.pt
scripts/
  chemstate_modz_model.py
  build_p100_modz_labels.py
  train_chemstate_modz.py
  infer_chemstate_modz.py
  test_frozen_models.py
tables/
  site_table.tsv
MODEL_CARD.md
USAGE.md
FINAL_MODEL_LOCK.json
SHA256SUMS.tsv
```

## 核心输入

- 512 位 Morgan 药物指纹，半径为 2
- 45 维匹配基线磷酸化状态
- 45 维基线观测掩码
- 细胞系编号
- 摩尔剂量的常用对数
- 处理小时数加一后的自然对数

输出为与 `tables/site_table.tsv` 顺序一致的 45 维磷酸化变化向量。
