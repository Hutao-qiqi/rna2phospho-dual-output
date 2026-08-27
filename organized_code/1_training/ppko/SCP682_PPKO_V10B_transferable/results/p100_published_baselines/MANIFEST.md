# PPKO V10B P100 published-method baseline comparison

目的：补齐 SCP682-PPKO 在 P100 全 125 个药物比较项上的 published-method baseline。评价按唯一 P100 共享位点去重；同一 P100 位点映射到多个 decryptM 位点时先平均预测值，再计算指标。

## 头条指标口径

主文 baseline 对照使用 selection-free 或 model-selected 指标：

- `response_auroc_abs_pred`：用预测绝对值识别真实 responsive top20 位点，属于 selection-free 排序指标。
- `all_cosine` / `all_direction`：所有共享位点。
- `predicted20_cosine` / `predicted20_direction`：模型自己预测的 top20 位点。
- `top20_recall`：模型预测 top20 与真实 responsive top20 的重叠。

`responsive20_cosine` 和 `responsive20_direction` 使用真实 responsive top20 位点，是 oracle 分组，只放补充表，不作为 baseline 主结论。

## 方向指标修正

方向准确率只在 `abs(predicted_delta) > 1e-8` 且 `abs(observed_delta) > 1e-8` 的位点上计算。全零或近零预测不再被计为方向错误；同时报告 `signed_fraction_all`、`signed_fraction_responsive20`、`signed_fraction_predicted20`，用于说明该方法实际给出符号预测的覆盖率。

这修正了早先 `direct_kinase_substrate` responsive top20 direction 约 0.027 的反常值。该值来自把大量零预测计为方向错误，不适合作为方法对照头条。修正后，KSEA forward 的方向约 0.55，但 `signed_fraction_all` 只有 0.045，说明覆盖率很低；图扩散和 RWR 的方向约 0.49，接近随机。

## Baseline

- `zero_vector`：全零预测。
- `direct_kinase_substrate`：signed target seed 直接投影到 regulator-to-site 边，作为 KSEA forward 对照。
- `graph_heat_diffusion`：同一异质图上的两步 heat diffusion，无学习参数。
- `rwr_signed_graph`：同一 signed protein graph 上的 random-walk-with-restart，无学习参数。
- `ridge_signed_target`：signed target vector + intercept 到 phosphosite delta 的 ridge 回归，alpha 在 decryptM 内部交叉验证选择。
- `retrieval_target`：按 target protein-context 相似度从 decryptM 训练集中检索最近 comparison。
- `retrieval_target_baseline`：按 target protein-context 相似度和 baseline phosphosite 相似度的均值检索最近 comparison。

## 主结果

| 模型 | AUROC | all cosine | all direction | predicted top20 cosine | predicted top20 direction | top20 recall | signed fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ppko_v10b` | 0.713 | 0.560 | 0.696 | 0.704 | 0.861 | 0.458 | 1.000 |
| `graph_heat_diffusion` | 0.517 | 0.024 | 0.489 | 0.037 | 0.548 | 0.230 | 0.861 |
| `rwr_signed_graph` | 0.517 | 0.032 | 0.492 | 0.049 | 0.526 | 0.222 | 0.860 |
| `direct_kinase_substrate` | 0.503 | 0.033 | 0.548 | 0.016 | 0.531 | 0.216 | 0.045 |
| `retrieval_target_baseline` | 0.499 | -0.028 | 0.474 | -0.094 | 0.457 | 0.209 | 0.804 |
| `ridge_signed_target` | 0.471 | -0.031 | 0.476 | -0.099 | 0.457 | 0.206 | 0.971 |
| `retrieval_target` | 0.460 | -0.022 | 0.492 | -0.038 | 0.477 | 0.161 | 0.765 |
| `zero_vector` | 0.500 | NA | NA | NA | NA | 0.215 | 0.000 |

PPKO 对所有非空 baseline 在 AUROC、all cosine、predicted top20 direction、top20 recall 等主指标上均为配对 Wilcoxon 单侧检验显著；详见 `tables/p100_v10b_published_baseline_paired_wilcoxon_vs_ppko.tsv`。

## 输出表

- `tables/baseline_method_descriptions.tsv`：baseline 定义。
- `tables/ridge_alpha_cv.tsv`：ridge alpha 的 decryptM 内部选择结果。
- `tables/p100_v10b_published_baseline_comparison_metrics.tsv`：逐 comparison × 模型指标。
- `tables/p100_v10b_published_baseline_comparison_summary.tsv`：模型级均值和 95% CI。
- `tables/p100_v10b_published_baseline_paired_wilcoxon_vs_ppko.tsv`：PPKO 对每个 baseline 的配对 Wilcoxon 检验。
- `reports/published_baseline_comparison_report.json`：机器可读汇总。

## 未纳入项

CellOracle 类转录调控扰动框架需要基因表达输入、调控网络和转录输出层，不能直接作为 phosphosite delta 输出 baseline。若主文需要，可在 Discussion 写为非直接可比方法，不放入 P100 主对照。
