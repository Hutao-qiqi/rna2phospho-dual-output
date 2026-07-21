# 总蛋白锚定的轴向动态通路图异质超图磷酸化残差模型

## 任务边界

模型输入为 RNA 和已经完成交叉拟合的总蛋白预测。磷酸化训练不包含总蛋白预测器，总蛋白张量进入模型后立即停止梯度。

916 例训练样本使用内部五折产生的折外总蛋白预测。229 例选择验证样本使用只在 916 例上完成拟合和选择的总蛋白模型预测。286 例封存样本不读取磷酸化标签，也不进入当前开发训练。

## 母蛋白锚点

每个位点在 916 例训练样本内拟合：

\[
B_{is}=a_s+\beta_s\widehat P_{i,\operatorname{parent}(s)}.
\]

斜率采用带岭惩罚的一元回归。母蛋白无法映射或有效样本不足时，回退为训练位点均值。校准参数不会使用 229 例或 286 例磷酸化标签。

真实磷酸化先按研究、按位点使用训练折均值和标准差变换，再执行样本内中位数中心化。验证集沿用训练参数；没有训练参数的研究—位点组合保持不可用，不允许池化回退。

\[
\widetilde Y_{is}=Y_{is}-\operatorname{median}_{s\in\Omega_i}Y_{is},
\]

\[
\widetilde B_{is}=B_{is}-\operatorname{median}_{s\in\Omega_i^B}B_{is}.
\]

残差监督为 \(\widetilde Y-\widetilde B\)。

## 通路状态

每条通路分别汇总 RNA 秩值和预测总蛋白：

\[
z_{i,p}=f_p\left(\operatorname{rank}(RNA_{i,G_p}),\widehat P_{i,G_p}\right).
\]

RNA 和蛋白成员均包含分子身份嵌入、连续值投影、均值和标准差。缺失成员通过掩码排除。

## 两个交替轴

每层包含一次通路轴更新和一次样本轴更新。

通路轴只允许有共享基因关系的通路相互读取，全局通路连接全部通路：

\[
z'_{i,p}=\operatorname{PathwayAttention}_p(z_{i,1:P}).
\]

通用 RNA 候选图只定义检索范围：

\[
\mathcal N_0(i)=\operatorname{kNN}(\operatorname{rank}(RNA_i)).
\]

每条通路、每个样本和每个注意力头在候选邻居中重新计算边权：

\[
\alpha^{(p,h)}_{ij}=\operatorname{softmax}_{j\in\mathcal N_0(i)}
\left(q^{(p,h)}_i k^{(p,h)}_j+g_h(z_{i,p},z_{j,p},s_{ij})+\log \pi_{ij}\right).
\]

传播矩阵使用共享主变换和通路低秩适配器。两层交替更新后，同一对患者可以在不同通路上得到不同连接权重。

## 异质超图位点查询

位点查询包含母蛋白基线、母蛋白预测值、训练锚点绝对皮尔森质量、训练配对覆盖度和通路特异训练残差记忆。残差记忆按位点块聚合并遵守参考缺失掩码；真实参考残差不参与查询编码和边权计算。

位点查询还包含四类状态：

- 位点身份；
- 母蛋白身份、预测丰度和校准基线；
- 已知激酶集合；
- 与母基因相关的通路状态。

位点只读取先验映射的通路。母蛋白、激酶和通路形成三类超边消息，位点级门控决定三类消息的权重：

\[
q_{is}=E_s+g^{parent}_{is}h^{parent}_{is}
+g^{kinase}_{is}h^{kinase}_{s}
+g^{pathway}_{is}h^{pathway}_{is}.
\]

门控在三类消息上归一化，不包含能够整体抬高全部位点的样本级输出门。

## 输出

原始残差在固定输出位点词表上投影到零中位数空间：

\[
\Delta^*_{is}=\Delta_{is}-\operatorname{median}_{s}\Delta_{is}.
\]

母蛋白基线与神经修正相加后再次执行固定词表行中位数投影，最终预测为：

\[
\widehat Y_{is}=\widetilde B_{is}+\lambda_s\Delta^*_{is}.
\]

\(\lambda_s\) 为位点级收缩参数，并使用向零收缩的正则。输出层保持线性。

## 损失和选模

每个训练样本先在其可观测位点上消除一个残差截距，用来协调“观测位点中位化标签”和“固定词表中位化输出”的中心域差异。随后计算逐位点等权均方误差。附加项为逐位点皮尔森、逐位点方差恢复和位点收缩。训练目标不含额外残差均方误差。默认批量为 128，可由命令行修改。

研究内位点标准化默认要求训练折内至少 8 个观测值，并对极小尺度设置训练折内下限。验证值不截断。

229 例逐位点斯皮尔曼中位数用于保存检查点。逐位点皮尔森、均方误差、预测与实测标准差比同时写入表格。

A800 默认使用 32 位浮点训练。混合精度保留为显式参数；本轮真实输入启动检查和数值修复均以 32 位完成，正式训练首轮采用该模式。混合精度需通过完整数值回归后再启用。

## 外部推理

检查点保存 916 例训练参考通路状态和 RNA 检索特征。外部样本只连接训练参考库，外部样本之间没有边。输入只包含：

- 与检查点一致的 RNA 基因；
- 与检查点一致的总蛋白预测；
- 检查点保存的训练参考状态。

真实总蛋白和真实磷酸化标签均不进入外部图输入。

## 保护条件

训练入口遇到以下情况会停止：

- 固定划分数量不等于 916/229/286；
- 916 例总蛋白预测未标记为折外；
- 229 例总蛋白预测未标记为只使用 916 例拟合；
- 总蛋白预测声明使用磷酸化标签；
- 286 例出现在总蛋白预测矩阵；
- RNA、蛋白、位点或先验成员缺失；
- 磷酸化矩阵无法按开发样本过滤读取。

划分清单为制表符分隔文件，包含 `sample_id` 和 `role`。`role` 只允许 `selection_train`、`selection_validation` 和 `sealed_test`。

总蛋白来源清单包含：

| 字段 | 取值 |
|---|---|
| `sample_id` | 与总蛋白预测矩阵行名一致 |
| `prediction_role` | 916 例为 `cross_fitted`，229 例为 `selection_train_only` |
| `phosphosite_labels_used` | 全部为 `false` |
| `source_model_id` | 逐样本预测器标识 |
| `source_fold` | 逐样本来源折标识 |
| `sample_in_source_training` | 全部为 `false` |

总蛋白预测矩阵只包含 916+229 例，列顺序为总蛋白基因词表。训练入口保存全部输入文件的 SHA-256 散列值。

旧锚点包若只保存 `source_archive`、`source_row_index` 和折外角色，可以继续用于实验，但检查点会标记为 `audited_archive_and_role_only`，不能声明逐样本模型与折证据完整。新锚点包必须保存上表三项增强字段。

## 训练入口

```bash
python code/train_axial_dynamic_hypergraph.py \
  --rna rna_log2_tpm_paired.parquet \
  --protein-prediction total_protein_development_predictions.parquet \
  --protein-provenance total_protein_prediction_provenance.tsv \
  --phosphosite observed_phosphosite.parquet \
  --phosphosite-manifest phosphosite_target_manifest.tsv \
  --split-manifest locked_916_229_286.tsv \
  --sample-metadata sample_manifest.tsv \
  --sample-id-column aliquot_id \
  --study-column study_id \
  --case-id-column case_submitter_id \
  --hallmark-gmt h.all.v2025.1.Hs.symbols.gmt \
  --canonical-gmt c2.cp.v2025.1.Hs.symbols.gmt \
  --kinase-prior kstar_edges.tsv \
  --kinase-prior copheeksa_edges.tsv \
  --output-dir 02_results/bulk/20260720_axial_dynamic_hypergraph \
  --device cuda:0
```

## 必做消融

- 仅母蛋白基线；
- 去除样本轴；
- 通用样本图替代通路动态边权；
- 去除通路轴；
- 去除激酶超边；
- 去除母蛋白超边动态值；
- 固定 \(\lambda_s=1\)；
- 动态图同度随机重连。
