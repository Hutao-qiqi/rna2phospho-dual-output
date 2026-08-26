# SCP682-SC12 protein-residual 训练说明

## 模型目的

SCP682-SC12 将单细胞磷酸化预测拆成两层：

```text
scRNA
→ scTranslator 或 scProTrans
→ predicted total protein abundance
→ SCP682-SC residual head
→ phospho_pred = beta_site * predicted_parent_protein + phospho_residual(RNA_state, pathway_attention, site_graph, protein_context)
```

`scTranslator` 或 `scProTrans` 提供总蛋白丰度底座。SCP682-SC12 显式拟合每个位点的父蛋白线性项，并把同一个预测父蛋白作为 `protein_context` 投入残差头。通路注意力和位点图模块学习总蛋白丰度之外的磷酸化残差。

## 蛋白预测缓存格式

训练脚本读取一个缓存目录，目录内至少需要：

```text
protein_predicted.npy
protein_features.tsv
cells.tsv
manifest.json
```

其中 `protein_predicted.npy` 是细胞 × 蛋白矩阵，`protein_features.tsv` 里要有可映射到 HGNC gene symbol 的列，例如 `canonical_gene_symbol`、`protein_symbol`、`gene` 或 `protein_id`。`cells.tsv` 用于把缓存细胞顺序对齐到 SC 模型输入里的 `cell_metadata.tsv`。

## 准备 SC12 全蛋白查询清单

```bat
D:\Tools\anaconda3\python.exe D:\data\lsy\vm_lsy_parent\lsy\03_code\single_cell\modeling\build_sctranslator_protein_coding_query.py
```

关键输出：

```text
D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\protein_prediction_cache\scp682_sc12_all_predictable_proteins_v1\scTranslator_hgnc_protein_coding_query.txt
```

当前全蛋白查询为 18,870 个 HGNC protein-coding genes 与 scTranslator decoder 字典交集。由于 scTranslator checkpoint 解码长度固定为 1000，推理脚本会按 19 个蛋白 shard 写出同一个 `protein_predicted.npy`。

## scTranslator 全细胞全蛋白推理

先放置权重：

```text
D:\data\lsy\models\scTranslator\checkpoint\scTranslator_2M.pt
```

示例：

```bat
D:\Tools\anaconda3\python.exe D:\data\lsy\vm_lsy_parent\lsy\03_code\single_cell\modeling\run_sctranslator_sc12_all_protein_inference.py ^
  --repo-dir D:\data\lsy\models\scTranslator ^
  --checkpoint D:\data\lsy\models\scTranslator\checkpoint\scTranslator_2M.pt ^
  --rna-h5ad D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\foundation_model_h5ad_inputs_v1\iccite_seq_tcell_2025.h5ad ^
  --protein-list D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\protein_prediction_cache\scp682_sc12_all_predictable_proteins_v1\scTranslator_hgnc_protein_coding_query.txt ^
  --output-dir D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\protein_prediction_cache\scTranslator_sc12_all_protein_cache_v1\iccite_seq_tcell_2025 ^
  --chunk-size 128 ^
  --batch-size 64 ^
  --protein-shard-size 1000 ^
  --resume
```

## 合并 7 个队列的全蛋白缓存

缓存完成后只运行合并入口，不启动训练：

```bat
D:\data\lsy\vm_lsy_parent\lsy\03_code\single_cell\modeling\run_scp682_sc12_merge_all_protein_cache_only.bat
```

合并输出目录：

```text
D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\protein_prediction_cache\scTranslator_sc12_all_protein_cache_merged_v1
```

## 烟测

```bat
D:\data\lsy\vm_lsy_parent\lsy\03_code\single_cell\modeling\run_scp682_sc12_protein_residual_smoke.bat
```

## 正式训练

```bat
D:\data\lsy\vm_lsy_parent\lsy\03_code\single_cell\modeling\run_scp682_sc12_protein_residual_formal.bat
```

## 关键输出

```text
tables\scp682_sc12_parent_protein_mapping.tsv
tables\scp682_sc12_component_performance.tsv
tables\scp682_sc12_component_gain_summary.tsv
tables\scp682_sc12_pr_gain_summary.tsv
tables\scp682_sc12_vs_sc11_external_per_target.tsv
tables\scp682_sc12_vs_sc11_external_summary.tsv
models\scp682_sc12_final.pt
reports\scp682_sc12_summary.json
```

`scp682_sc12_component_gain_summary.tsv` 用来报告：

```text
protein_only
phospho_residual_only
protein_plus_residual
residual_gain_over_protein_only
protein_component_gain_over_residual_only
graph_gain
pathway_gain
```

SC12 与 SC11 的外部验证对比必须按 `cohort_id + target_id` 交集配对，避免不同位点集合混比。训练结束后使用：

```bat
D:\Tools\anaconda3\python.exe D:\data\lsy\vm_lsy_parent\lsy\03_code\single_cell\modeling\summarize_scp682_sc12_vs_sc11.py ^
  --sc11-performance D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260523_scp682_sc11_expanded_scnet_site_gnn_v1\tables\scp682_sc11_reconstruction_performance.tsv ^
  --sc12-performance D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260621_scp682_sc12_protein_residual_v1\tables\scp682_sc12_reconstruction_performance.tsv ^
  --output-dir D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\20260621_scp682_sc12_protein_residual_v1\tables
```
