# SCP682

## 统一单细胞模型

[U1架构、训练和推理代码](organized_code/1_training/sc/scp682_u1/)

[U1模型发布页](https://github.com/Hutao-qiqi/rna2phospho-dual-output/releases/tag/scp682-u1-v1)

U1在GSE300551、icCITE和QuRIE上联合训练，共享RNA双输入编码器与位点条件输出，覆盖47个有监督读出。

## 整体组织模型

[M2.2代码与核心权重](SCP682/README.md) · [架构](SCP682/ARCHITECTURE.md) · [训练说明](SCP682/TRAINING.md)

M2.2使用256维RNA表示与128维RNA预测蛋白表示，预测128维PTM坐标，经选择性坐标修正和位点基底重构后，结合背景与母蛋白直接项输出18,592个位点。目标域适配采用PT7，患者谱采用逐位点收缩堆叠。

当前核心推理接收参考分位数RNA和RNA预测总蛋白；RNA→总蛋白回归器权重与目标域完整适配器需依据对应代码和数据生成。

## 扰动与分析

[扰动训练](organized_code/1_training/ppko/) · [数据分析与制图](organized_code/)
