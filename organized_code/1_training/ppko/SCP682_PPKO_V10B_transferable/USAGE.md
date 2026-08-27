# SCP682-PPKO V10B strong300 使用方法

本目录是当前冻结的可迁移主模型包。权重、训练脚本、验证脚本和 P100 结果表均以相对路径保存在包内，目录可以整体复制到服务器或其他工作区。

## 文件结构

- `models/scp682_ppko_v10b_strong300_best.pt`：冻结主权重。
- `scripts/pretrain_v10b_strong300.py`：训练脚本，用于复现实验或重新训练。
- `scripts/validate_v10b_p100_all_drugs.py`：P100 全药物验证脚本，已改为默认读取本包权重。
- `results/v10b_strong300_p100_metrics.tsv`：P100 全药物逐比较项原始结果。
- `results/v10b_p100_all_drug_mode_summary_reannotated.tsv`：修正靶点注释后的主汇总表。
- `results/v10b_p100_all_drug_drug_summary_reannotated.tsv`：修正靶点注释后的逐药物表。
- `results/p100_published_baselines/`：P100 全药物 published-method baseline 对照，包含图扩散、RWR、KSEA forward、ridge 和 nearest-neighbor retrieval。
- `notes/README_MODEL.md`：模型选择理由、使用边界和权重校验值。

## 权重校验

```powershell
Get-FileHash .\models\scp682_ppko_v10b_strong300_best.pt -Algorithm SHA256
```

期望值：

```text
A7498D0056DD83095E0AB8135DAC3C2D60A70ABDB2251184D279F7D3731A11FD
```

## P100 全药物验证

验证脚本需要读取原始 decryptM 训练输入、P100 输入、共享位点映射表和全局磷蛋白图。若这些数据位于服务器标准根目录 `D:\data\lsy\vm_lsy_parent\lsy`，可以直接运行：

```powershell
cd D:\data\lsy\vm_lsy_parent\lsy\SCP682_PPKO_V10B_transferable
python .\scripts\validate_v10b_p100_all_drugs.py --device cuda:0
```

若数据根目录不同，设置 `SCP682_DATA_ROOT` 或显式传参：

```powershell
$env:SCP682_DATA_ROOT="D:\data\lsy\vm_lsy_parent\lsy"
python .\scripts\validate_v10b_p100_all_drugs.py `
  --model .\models\scp682_ppko_v10b_strong300_best.pt `
  --graph-dir "$env:SCP682_DATA_ROOT\01_data\pathway_prior\intermediate\global_phosphoprotein_heterograph_v10_measured_string700_top50" `
  --train-dir "$env:SCP682_DATA_ROOT\01_data\single_cell\intermediate\phospho_perturb\decryptm_comparison_delta_v8" `
  --p100-dir "$env:SCP682_DATA_ROOT\01_data\single_cell\intermediate\phospho_perturb\lincs_p100_comparison_delta_v7" `
  --map-dir "$env:SCP682_DATA_ROOT\02_results\single_cell\20260519_scp682_ppko_1_decryptm_network_v8_full_p100_shared_site_validation\tables" `
  --output-dir .\validation_outputs\p100_all_drugs `
  --device cuda:0
```

输出会写入：

```text
validation_outputs/p100_all_drugs/tables/v10b_p100_all_drug_metrics.tsv
validation_outputs/p100_all_drugs/tables/v10b_p100_all_drug_mode_summary.tsv
validation_outputs/p100_all_drugs/tables/v10b_p100_all_drug_drug_summary.tsv
validation_outputs/p100_all_drugs/reports/summary.json
```

## P100 published-method baseline 对照

已固定在：

```text
results/p100_published_baselines/
```

该目录按唯一 P100 共享位点去重评价，包含：

```text
tables/p100_v10b_published_baseline_comparison_metrics.tsv
tables/p100_v10b_published_baseline_comparison_summary.tsv
tables/p100_v10b_published_baseline_paired_wilcoxon_vs_ppko.tsv
tables/baseline_method_descriptions.tsv
tables/ridge_alpha_cv.tsv
```

主结果以 selection-free 和 model-selected 指标为头条：`ppko_v10b` 的 response AUROC 为 0.713，all cosine 为 0.560，all direction 为 0.696，predicted top20 cosine 为 0.704，predicted top20 direction 为 0.861。均高于图扩散、RWR、KSEA forward、ridge 和 nearest-neighbor retrieval baseline。`responsive top20` 使用真实响应位点分组，只放补充表，不作为 baseline 主结论。

## 重新训练

复现实验时使用：

```powershell
cd D:\data\lsy\vm_lsy_parent\lsy\SCP682_PPKO_V10B_transferable
python .\scripts\pretrain_v10b_strong300.py `
  --input-dir D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\phospho_perturb\decryptm_comparison_delta_v8 `
  --graph-dir D:\data\lsy\vm_lsy_parent\lsy\01_data\pathway_prior\intermediate\global_phosphoprotein_heterograph_v10_measured_string700_top50 `
  --output-dir D:\data\lsy\vm_lsy_parent\lsy\02_results\single_cell\retrain_scp682_ppko_v10b_strong300 `
  --device cuda:0
```

训练脚本的固定主要超参为：`epochs=620`、`batch_size=16`、`lr=8e-4`、`hidden=192`、`latent=96`、`seed=20260519`。

## 使用边界

该模型用于 P100 全药物共享磷酸化位点响应预测，并覆盖 V20D 词表外靶点。论文中可表述为“当前通用药物磷酸化扰动算子候选模型”。不要把它表述为已经完成药物结构驱动的新化合物泛化，也不要用乱置靶点对照主张严格靶点机制特异性。
