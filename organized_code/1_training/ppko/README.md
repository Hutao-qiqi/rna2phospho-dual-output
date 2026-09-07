# SCP682-PPKO

PPKO包含靶点干预和具体药物预测两种模型，分别使用独立权重。两者均预测基线之后的磷酸化变化。

| 模型 | 输入 | 输出 | 模型包 |
|---|---|---|---|
| V10B靶点干预 | 8192维基线、观测掩码、靶点及作用方向、完整图先验 | 8192个原生磷酸肽的变化 | [V10B](SCP682_PPKO_V10B_transferable/USAGE.md) |
| ChemState-MODZ激酶抑制剂 | 化合物结构、45维基线及掩码、细胞背景、剂量、时间 | 45个P100分析物的变化；训练覆盖52药 | [ChemState-MODZ](SCP682_ChemState_MODZ_final/USAGE.md) |
| ChemState-MODZ全药物 | 同上 | 45个P100分析物的变化；训练覆盖118药 | [ChemState-MODZ](SCP682_ChemState_MODZ_final/USAGE.md) |

## 权重与代码

三份权重均直接包含在对应包的`models/`目录。`PPKO_MODELS.json`列出架构、训练和推理入口。

- V10B架构与训练：`SCP682_PPKO_V10B_transferable/scripts/pretrain_v10b_strong300.py`。
- ChemState-MODZ架构：`SCP682_ChemState_MODZ_final/scripts/chemstate_modz_model.py`。
- ChemState-MODZ训练：`SCP682_ChemState_MODZ_final/scripts/train_chemstate_modz.py`。
- MODZ标签构建：`SCP682_ChemState_MODZ_final/scripts/build_p100_modz_labels.py`。

## 运行

从本目录运行三个检查点的结构与确定性检查：

```bash
python test_ppko_release.py
```

具体训练、推理输入和命令见两包的`USAGE.md`。V10B的完整推理还需要原始图先验与位点词表；数据路径通过命令行传入。ChemState的细胞类别顺序保存在检查点`cells`中，分析物顺序见其`tables/site_table.tsv`。

两类输出的分析物词表和训练尺度分别保留。选择模型后按对应输入准备数据，输出不自动拼接、相加或互相替代。
