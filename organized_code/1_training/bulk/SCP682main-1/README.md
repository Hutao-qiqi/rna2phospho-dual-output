# SCP682main-1

最新轴向动态通路图模型见：

- `docs/AXIAL_DYNAMIC_HETEROGENEOUS_HYPERGRAPH.md`
- `docs/FINAL_ARCHITECTURE_AUDIT_20260720.md`
- `code/prepare_total_protein_anchor.py`
- `code/train_axial_dynamic_hypergraph.py`

实验性磷酸化分支。该分支修复磷酸化目标、母蛋白基线和残差的样本内中位数处理不一致，并实现总蛋白锚定的轴向动态通路图异质超图残差模型。

正式主模型目录 `SCP682-main` 不受本分支影响。

## 当前补丁

训练、验证和外部评价的官方目标均为每个样本在其可观测磷酸化位点上的中位数归零值。基线使用可用基线位点的样本中位数归零值。残差读出在可用位点上投影为零中位数。

\[
\widetilde Y_{is}=Y_{is}-\operatorname{median}_{s\in\Omega_i}Y_{is}
\]

\[
\widetilde B_{is}=B_{is}-\operatorname{median}_{s\in\Omega_i^{B}}B_{is}
\]

\[
\widehat Y_{is}=\widetilde B_{is}+\Pi_0\left(\widehat\Delta_{is}\right)
\]

其中 \(\Pi_0\) 表示按样本、在可用位点上投影到零中位数空间。

原始未中心化数据仅保留为诊断输出，不用于本分支的正式模型选择或指标报告。

## 通路特异样本图

训练折 RNA 生成一张候选近邻图。每条通路在候选邻居中独立学习边权，保留样本自身状态的残差连接。每个位点根据母基因所属通路进行稀疏读取，再由位点解码器生成中心化残差。

外部样本只连接检查点中的训练参考样本。候选边、通路边权和位点输出均不读取外部样本的真实磷酸化标签。完整约束见 `docs/PATHWAY_SAMPLE_GRAPH_CONTRACT.md`。

## 当前主候选

主候选代码：

- `code/axial_dynamic_hypergraph.py`：模型核心；
- `code/axial_dynamic_data.py`：划分、折外总蛋白来源、通路和激酶先验及母蛋白校准；
- `code/train_axial_dynamic_hypergraph.py`：固定 916/229/286 划分训练入口；
- `code/predict_axial_dynamic_hypergraph.py`：训练参考库模式的外部推理；
- `config/axial_dynamic_hypergraph_a800.json`：主配置；
- `docs/AXIAL_DYNAMIC_HETEROGENEOUS_HYPERGRAPH.md`：完整架构和输入契约。

总蛋白模型不属于该训练图。训练入口读取预先生成的总蛋白预测：916 例必须是折外预测，229 例必须来自只用 916 例拟合的模型，286 例禁止出现在总蛋白预测文件中。

输出目录固定包含 `models/`、`predictions/`、`tables/`、`logs/` 和 `reports/`。逐位点斯皮尔曼中位数用于检查点选择，均方误差、皮尔森和方差恢复比同步保存。
