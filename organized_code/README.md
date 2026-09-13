# SCP2000分析与制图代码

- 整体组织模型：[架构、核心权重与训练代码](../SCP2000/README.md)。
- 统一单细胞实现：[SCP2000-sc U1 模型代码](1_training/sc/scp2000_sc_u1_20260914/)；U1仅为内部检查点标识，公开模型名为`SCP2000-sc`，权重见`scp2000-sc-u1-v1` GitHub Release。
- 历史单细胞模型：`sc15_dual_input_ensemble_20260829/`等目录仅用于开发过程和消融复现。
- 扰动训练：`1_training/ppko/`。
- 数据分析：`2_analysis/`。
- 制图：`3_plotting/`。
- 模块复现说明：`reproduction_chains/`。
