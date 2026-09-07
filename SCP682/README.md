# SCP682 M2.2

本目录包含核心推理权重、架构实现、RNA→总蛋白训练代码、母蛋白分解、128维残差基底训练、坐标预测、PT7及患者谱堆叠源代码。

## 推理

```bash
pip install -r SCP682/requirements.txt
python SCP682/predict.py --rna rna.parquet --predicted-protein protein_prediction.parquet --runtime-bundle SCP682/runtime/scp682_m22_runtime.npz --output prediction.npz
```

两个输入矩阵按患者行、分子列组织，使用相同患者编号。RNA包含19,938个基因，预测总蛋白包含11,311个蛋白。输入采用训练参考分位数坐标；输出包含18,592个位点，位点与输入词表保存在权重文件。

`predict.py`实现基础M2.2。PT7及患者谱堆叠有单独训练入口，需要目标队列校准标签。`parameters/`提供当前目标域选定的位点参数；这些表不能单独替代目标域完整适配器。

## 权重

`runtime/scp682_m22_runtime.npz`包含输入中心与尺度、投影矩阵、坐标岭回归、8个坐标修正、位点基底、位点均值、母蛋白映射与系数、全局背景。

此文件接收已预测总蛋白。RNA→总蛋白回归器和目标域完整适配器需依据对应训练代码与数据生成。

模型定义见`MODEL.json`，数学结构见`ARCHITECTURE.md`，训练数据要求见`TRAINING.md`。
