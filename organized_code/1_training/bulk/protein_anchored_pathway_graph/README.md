# 蛋白锚定的通路特异样本图残差模型

该目录保存 SCP682 bulk 的独立实验架构。当前发布模型、便携推理包和论文主结果均未被替换。

## 计算链

模型把 RNA 到磷酸化位点的映射分成五段：

1. RNA 编码器预测全部总蛋白丰度。
2. 每个位点使用所属蛋白预测值拟合折内线性校准，得到蛋白锚定状态：

   `B(i,s) = intercept(s) + beta(s) * protein_hat(i,parent(s))`

3. 磷酸化目标扣除 `B`，再对每个样本的残差做中位数中心化，移除样本整体磷酸化平移。
4. RNA 秩值和预测总蛋白按 MSigDB 通路汇总。每条通路单独构建样本近邻图，图边只使用 RNA 与预测总蛋白。
5. 位点查询由所属蛋白、位点身份和已知激酶组成，只读取相关通路状态。最终输出为：

   `phosphosite_hat = B + lambda_site * centered_pathway_graph_residual`

`lambda_site` 是位点级收缩参数。模型不使用样本级或队列级缩放门。

## 训练约束

- 总蛋白头只接受真实总蛋白标签的梯度。磷酸化损失不能更新总蛋白头。
- 所有归一化、蛋白到位点校准和训练样本图均在训练折内计算。
- 严格归纳评估时，测试样本只能连接训练参考节点，测试样本之间不传递信息。
- 外部批量推理可在目标队列内部重建通路样本图，输入仍限于 RNA 和模型预测总蛋白。
- 单样本推理使用 checkpoint 内保存的训练参考通路状态。
- 真实磷酸化值不参与样本图构建和外部适配。

## 文件

- `model.py`：统一模型类与计算公式。
- `data.py`：路径发现、通路构建、折内校准、通路近邻图和评价函数。
- `train.py`：五折训练入口。
- `predict.py`：外部队列或单样本推理入口。
- `summarize.py`：合并五折严格归纳预测。
- `run_two_gpu_5fold.ps1`：两张显卡按折并行训练。
- `smoke_test.py`：形状、残差中心化和梯度隔离检查。
- `audit_inputs.py`：只读检查样本、位点、父蛋白、激酶和通路映射。

## 最小检查

```powershell
python smoke_test.py
```

服务器真实输入审计：

```powershell
python audit_inputs.py --project-root D:\data\lsy\vm_lsy_parent\lsy
```

## 两张显卡训练

```powershell
powershell -ExecutionPolicy Bypass -File run_two_gpu_5fold.ps1 `
  -ProjectRoot D:\data\lsy\vm_lsy_parent\lsy `
  -OutputDir D:\data\lsy\scp682_protein_anchored_pathway_graph_v1
```

## 外部队列推理

队列内无标签建图：

```powershell
python predict.py `
  --checkpoint-dir D:\data\lsy\scp682_protein_anchored_pathway_graph_v1 `
  --rna external_rna_log2_tpm.parquet `
  --output-dir external_prediction `
  --sample-graph-mode cohort
```

单样本或严格参考库模式：

```powershell
python predict.py `
  --checkpoint-dir D:\data\lsy\scp682_protein_anchored_pathway_graph_v1 `
  --rna one_sample_log2_tpm.parquet `
  --output-dir one_sample_prediction `
  --sample-graph-mode reference
```

## 投稿前必跑消融

- 蛋白锚定状态单独输出。
- 蛋白锚定状态加中心化残差。
- 全局样本图与通路特异样本图对比。
- 通路图同度随机重连。
- 去除激酶先验。
- 外部队列的队列内图与训练参考库图对比。
