# SCP682main-1

实验性磷酸化分支。该分支从冻结 SCP682 图残差模型复制代码，修复磷酸化目标、基线和残差的样本内中位数处理不一致。

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
