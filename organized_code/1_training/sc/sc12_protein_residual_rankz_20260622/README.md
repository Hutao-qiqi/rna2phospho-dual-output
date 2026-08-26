# SCP682-SC12 protein-residual rank-z

这是当前最新的单细胞磷酸化预测模型包。模型把单细胞 RNA 表示、预测总蛋白信息、通路注意力和位点图残差放在同一套训练流程里，用于从单细胞 RNA 预测细胞级磷酸化读数。

## 模型公式

```text
scRNA
→ frozen scFoundation RNA state
→ pathway-attention residual state
→ scTranslator predicted parent-protein context
→ site graph residual refinement
→ phospho prediction
```

训练时显式拟合父蛋白项：

```text
phospho_pred =
  frozen_beta_site * predicted_parent_protein
  + frozen_bias_site
  + phospho_residual(RNA_state, pathway_attention, site_graph, protein_context)
```

`rank-z` 版对每个队列的 scTranslator 预测蛋白矩阵做队列内秩正态化/标准化，解决原始 scTranslator 输出跨队列尺度漂移的问题。

## 训练数据

统一输入目录：

```text
D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_masked_multisite_v1
```

训练和内部验证来自：

```text
icCITE
QuRIE control
QuRIE ibrutinib
```

全量缓存训练覆盖 212,820 cells，监督磷酸化读数 56 个。外部验证覆盖 GSE300551、Blair、SIGNAL-seq HeLa、SIGNAL-seq PDO/CAF、Vivo-seq Th17。

## 蛋白预测缓存

原始 scTranslator 全蛋白缓存：

```text
D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\protein_prediction_cache\scTranslator_sc12_all_protein_cache_merged_v1
```

rank-z 缓存：

```text
D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\protein_prediction_cache\scTranslator_sc12_all_protein_cache_merged_rankz_v1
```

缓存目录需要包含：

```text
protein_predicted.npy
protein_features.tsv
cells.tsv
manifest.json
```

## 运行入口

生成全蛋白查询：

```bat
D:\Tools\anaconda3\python.exe code\build_sctranslator_protein_coding_query.py
```

运行 scTranslator 推理：

```bat
D:\Tools\anaconda3\python.exe code\run_sctranslator_sc12_all_protein_inference.py ^
  --repo-dir D:\data\lsy\models\scTranslator ^
  --checkpoint D:\data\lsy\models\scTranslator\checkpoint\scTranslator_2M.pt ^
  --rna-h5ad <input.h5ad> ^
  --protein-list <protein_query.txt> ^
  --output-dir <cache_output_dir> ^
  --chunk-size 128 ^
  --batch-size 64 ^
  --protein-shard-size 1000 ^
  --resume
```

合并缓存：

```bat
scripts\run_scp682_sc12_merge_all_protein_cache_only.bat
```

生成 rank-z 缓存：

```bat
D:\Tools\anaconda3\python.exe code\make_sctranslator_sc12_datasetwise_rankz_cache.py
```

正式训练：

```bat
scripts\run_scp682_sc12_protein_residual_rankz_logged.bat
```

## 模型权重

```text
models\scp682_sc12_final.pt
```

该权重来自：

```text
D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260622_scp682_sc12_protein_residual_rankz_v1\models\scp682_sc12_final.pt
```

## 关键结果

结果目录：

```text
D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260622_scp682_sc12_protein_residual_rankz_v1
```

主要外部验证中位 Spearman：

| cohort | median Spearman |
|---|---:|
| GSE300551 | 0.309 |
| Blair RPS6 | 0.376 |
| SIGNAL-seq HeLa | 0.573 |
| SIGNAL-seq PDO/CAF | 0.118 |
| Vivo-seq Th17 | 0.210 |

关键表：

```text
tables\scp682_sc12_reconstruction_performance.tsv
tables\scp682_sc12_component_gain_summary.tsv
tables\scp682_sc12_component_performance.tsv
tables\scp682_sc12_parent_protein_mapping.tsv
tables\scp682_sc12_site_graph_edges.tsv
tables\scp682_sc12_target_pathway_prior.tsv
reports\scp682_sc12_summary.json
reports\scp682_sc12_protein_cache.json
```

文件清单见：

```text
MANIFEST.csv
```
