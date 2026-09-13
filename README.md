# SCP2000

## 统一单细胞模型

**正式模型名称：SCP2000。** U1仅用于精确标识单细胞检查点，不进入公开模型名。SC11、SC15和F_ALL属于历史开发模型及消融。锁定元数据见[`SCP2000_CURRENT.json`](SCP2000_CURRENT.json)。

[U1架构、训练和推理代码](organized_code/1_training/sc/scp682_u1/)

[模型发布页](https://github.com/Hutao-qiqi/rna2phospho-dual-output/releases/tag/scp2000-v1)

U1在GSE300551、icCITE和QuRIE共177,078个细胞上联合训练，共享RNA双输入编码器、位点条件输出和轻量域校准；包含52个输出坐标，其中47个获得实测监督。

## 整体组织模型

[模型代码与核心权重](SCP2000/README.md) · [架构](SCP2000/ARCHITECTURE.md) · [训练说明](SCP2000/TRAINING.md)

SCP2000使用256维RNA表示与128维RNA预测蛋白表示，预测128维PTM坐标，经选择性坐标修正和位点基底重构后，结合背景与母蛋白直接项输出18,592个位点。

当前核心推理接收参考分位数RNA和RNA预测总蛋白；RNA→总蛋白回归器权重与目标域完整适配器需依据对应代码和数据生成。

## 扰动与分析

[扰动训练](organized_code/1_training/ppko/) · [数据分析与制图](organized_code/)
