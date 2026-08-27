# SCP682-PPKO V10B strong300 可迁移包

本目录固定为当前投稿主模型包。主模型为 V10B strong300，保留理由是它在 P100 全 125 个药物比较项上可输出训练外靶点结果，适合作为通用药物磷酸化响应预测模型的当前版本。

## 固定文件

- 权重：`models/scp682_ppko_v10b_strong300_best.pt`
- 训练脚本：`scripts/pretrain_v10b_strong300.py`
- 全药物验证脚本：`scripts/validate_v10b_p100_all_drugs.py`
- 原始 P100 全药物逐项结果：`results/v10b_strong300_p100_metrics.tsv`
- 修正靶点注释后的 P100 全药物汇总：`results/v10b_p100_all_drug_mode_summary_reannotated.tsv`
- 修正靶点注释后的逐药物汇总：`results/v10b_p100_all_drug_drug_summary_reannotated.tsv`
- Published-method baseline 对照：`results/p100_published_baselines/`

## 主结果口径

P100 全 125 个药物比较项，修正 curcumin 靶点注释为 `CREBBP;EP300` 后，真实靶点输入结果为：

- 全体共享位点余弦：0.543
- 全体共享位点方向一致性：0.682
- 响应位点余弦：0.709
- 响应位点方向一致性：0.857
- 强响应位点找回率：0.439
- 响应位点识别曲线下面积：0.697

旧版 V10B strong300 全 125 个比较项汇总约为：

- 全体共享位点余弦：0.578
- 全体共享位点等级相关：0.575
- 全体共享位点方向一致性：0.712
- 响应位点余弦：0.728
- 响应位点方向一致性：0.864
- 强响应位点找回率：0.458
- 响应位点识别曲线下面积：0.714

## Published-method baseline 对照

新增 P100 全 125 个比较项 baseline 对照，按唯一 P100 共享位点去重评价。对照包括：全零、KSEA forward、同图 heat diffusion、同图 RWR、signed target ridge、target retrieval、target+baseline retrieval。

主结果以 selection-free 和 model-selected 指标为头条：

- `ppko_v10b` response AUROC：0.713
- `ppko_v10b` all cosine：0.560
- `ppko_v10b` all direction：0.696
- `ppko_v10b` predicted top20 cosine：0.704
- `ppko_v10b` predicted top20 direction：0.861

`responsive top20` 使用真实响应位点分组，只作为补充指标。最接近的朴素图传播 baseline 仍明显较低：`rwr_signed_graph` response AUROC 0.517、all cosine 0.032、all direction 0.492、predicted top20 cosine 0.049、predicted top20 direction 0.526。配对 Wilcoxon 检验见 `results/p100_published_baselines/tables/p100_v10b_published_baseline_paired_wilcoxon_vs_ppko.tsv`。

方向指标只在有非零预测符号的位点上计算，并同时报告 `signed_fraction_all`。因此 KSEA forward 的方向约 0.55 不再被零预测压低，但其 `signed_fraction_all` 只有 0.045，覆盖率很低。

## 使用边界

可以主张：模型在 P100 全药物共享磷酸化位点上预测药物处理后的磷酸化响应，并覆盖训练外药物靶点。

不要主张：模型已经严格证明靶点机制特异性，或已经完成由药物结构直接驱动的新化合物泛化。

## 权重校验

- 大小：15680895 字节
- SHA256：`A7498D0056DD83095E0AB8135DAC3C2D60A70ABDB2251184D279F7D3731A11FD`
