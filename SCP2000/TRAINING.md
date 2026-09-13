# 训练与复现

代码保持项目相对目录。下表各路径相对本目录，输入数据按原训练任务的患者与分子编号对齐。

| 阶段 | 入口 | 输入与产物 |
|---|---|---|
| RNA预测总蛋白 | `03_code/bulk/protein_anchor/train_random70_multibranch_protein_head.py` | 参考分位数HDF5、蛋白图先验；输出折外与开发预测 |
| 母蛋白分解 | `03_code/model_validation/scp682_v2_m1_parent_ptm_decomposition/prepare_m1_decomposition.py` | 磷酸化、预测蛋白、研究和划分；输出背景、beta及残差 |
| 128维基底 | `03_code/model_validation/scp682_v2_m2_residual_latent/diagnose_residual_latent.py` | 母蛋白分解后的残差；输出冻结位点基底 |
| 坐标预测 | `03_code/model_validation/scp682_v2_m2_residual_latent/train_deployable_coordinate_predictor.py` | RNA、预测蛋白和残差坐标 |
| M2.2选择性坐标修正 | `03_code/model_validation/scp682_v2_m22_study_centered_evaluation/01_rebuild_primary_predictions.py` | 基底、预测蛋白、训练/开发划分；输出折外及770例预测 |
| 核心权重导出 | `03_code/model_validation/scp682_main_m22_migration/build_m22_runtime_release.py` | 完成的训练资产；导出运行NPZ |
| PT7 | `03_code/model_validation/scp682_v2_m22_study_centered_evaluation/02_rerun_external_pt7_centered.py` | 247例校准与106例测试输入、冻结模型资产 |
| 患者谱堆叠 | `03_code/model_validation/scp682_v2_m22_phosphosct/run_output_calibration_shrinkage.py` | M2.2与S1目标域预测；输出收缩后的患者谱 |

训练患者1796例，开发770例，训练五折种子20260823。蛋白头训练种子20260810，分支包括512维全局投影、64基因局部先验和128基因监督选择。具体参数保留在各入口参数及函数实现中。

需要准备的训练资产：

- 参考分位数RNA与真实蛋白HDF5，含患者编号、分子词表、原始测量掩码和训练/开发标记。
- 蛋白图先验及RNA→蛋白代码所需先验。
- 原坐标磷酸化矩阵、预测蛋白矩阵、研究编号和位点—母蛋白映射。
- 已固定的内部五折、外部247/106划分及校准五折编号。

这些患者级训练数据由数据持有人提供。参数化入口可运行`--help`查看路径参数；固定路径入口以本目录为项目根目录，读取源代码中声明的相对路径。不得将测试患者用于背景、投影或适配器训练。

本次发布包含基础M2.2冻结权重。蛋白头原训练程序保存预测与先验，未保存可直接部署到新患者的全套回归权重；重建RNA端到端部署时需在原训练输入上拟合并导出蛋白回归器。目标域适配器同样需根据校准输入生成。
