# SCP682：统一单细胞磷酸化预测

本发布使用U1配置：GSE300551、icCITE及QuRIE联合训练一个共享模型。输入为4,000个固定基因的表达和3,072维冻结scFoundation表示，经双输入编码、位点条件输出与轻量域校准，预测磷酸化读出。

## 获取模型

[下载完整模型包](https://github.com/Hutao-qiqi/rna2phospho-dual-output/releases/tag/scp682-u1-v1)。解压后包含`SCP682_U1.pt`、预处理参数、读出定义、划分清单和可运行代码。

模型有52个输出坐标，其中47个在联合训练中有真实监督。推理默认只导出有监督的读出；坐标定义和监督标志见`readouts.tsv`。抗体定义尚未确定残基的项目保留原始读出名称。

## 架构

```text
4,000-gene RNA -> 2048 -> 1024 -> 512 residual MLP
scFoundation 3072 -> 512 residual MLP
concatenate -> 512 shared cell representation
shared readout head + 32D site-conditioned query
lightweight domain conditioning and readout calibration
```

完整模块位于`code/model.py`。位点查询、共享编码器及主输出参数在三个训练队列间共享。模型记录辅助检测层参数，U1训练目标使用连续加权回归。

## 推理

先在适合当前显卡的PyTorch环境安装`requirements.txt`。scFoundation主模型及原始数据不包含在本包中；输入其已经生成的3,072维细胞表示，保持细胞顺序一致。

```bash
python code/predict.py \
  --release-dir . \
  --rna-h5ad RNA_counts.h5ad \
  --embeddings scfoundation_embeddings.npy \
  --embedding-cell-ids embedding_cell_ids.tsv \
  --output-dir predictions \
  --domain 0 \
  --device cuda:0
```

`RNA_counts.h5ad`的X为未变换非负计数，基因名和细胞名唯一。`embedding_cell_ids.tsv`为无表头单列细胞编号，须与RNA及表示矩阵逐行一致。

基因按原训练名称精确对应，区分大小写和物种命名空间。人类基因采用原符号；小鼠基因在本项目使用`__`前缀并保留原大小写，例如`__Stat3`。混合物种输入应沿用这一规则。缺失基因先补零计数，再应用固定训练均值与标准差；每细胞一万计数归一化在完整RNA基因矩阵上完成。

域编号：0为未知平台，1为GSE300551，2为icCITE，3为QuRIE。未知平台输出统一标准化磷酸化分数；已知域输出相应训练域原生尺度，并仅导出该域的监督读出。未知平台分数不解释为抗体绝对计数。`gene_coverage.tsv`给出输入基因覆盖。

## 训练复现

`code/train.py`包含RNA准备、固定基因选择、掩码处理、损失、优化及检查点选择。`splits.tsv.gz`给出177,078个细胞的原始划分；GSE使用留出板，icCITE与QuRIE使用原分层细胞划分。三个队列未统一采用相同90/10比例。

```text
DATA_ROOT/data/model_input/
  embeddings.npy
  targets.npy
  target_mask.npy
  cell_metadata.tsv
  phospho_target_table.tsv
DATA_ROOT/data/h5ad/
  gse300551_iccite_plex_kinase_2025.h5ad
  iccite_seq_tcell_2025.h5ad
  qurie_seq_bjab_2021.h5ad
```

表示、标签、掩码与元数据必须共用细胞顺序。标签矩阵列号与`phospho_target_table.tsv`的`target_index`一致；每个RNA文件使用`cell_id`作为行名。原始测量与训练数据由相应公开队列提供，不包含在发布包中。

```bash
python code/train.py prepare --root DATA_ROOT
python code/train.py train --root DATA_ROOT --variant U1 --device cuda:0
```

训练种子682，AdamW学习率0.0001、权重衰减0.0001；每步从三个队列各抽64个细胞，每轮300步，最多40轮。以三个队列验证位点Spearman中位数的平均值选择检查点。本次发布选中第5轮；训练历史与真实划分数量见`training_history.tsv`及`training_config.json`。

损失为各位点掩码加权Huber。低值阈值及零值比例只根据优化细胞计算，域随机置零比例为20%。原始训练未使用外部后训练适配权重。

## 文件与验证

- `SCP682_U1.pt`：单个统一模型权重。
- `preprocessing.npz`：固定基因顺序、RNA训练均值及标准差。
- `domain_normalization.npz`：三个域的标签尺度、阈值及监督掩码。
- `readouts.tsv`、`readout_definitions.tsv`：输出坐标及原始抗体映射。
- `splits.tsv.gz`：复现用固定细胞划分。
- `code/model.py`、`code/train.py`、`code/predict.py`：架构、训练、推理。
- `verification.json`：公开架构与训练架构的输出对应，以及真实细胞推理检查。
- `verification_inputs.npz`：合成测试输入及参考输出，不含真实标签。

公开架构严格载入原权重，合成输入输出误差为0；真实Vivo细胞推理误差约2.5e-7，训练反向传播测试通过。验证环境记录于`verification.json`。
