# Fig4 PPKO 复现链核验摘要

更新日期：2026-06-18

## 当前结论

Fig4 的 PPKO 代码入口已统一到 `SCP682_PPKO_V10B_transferable` 冻结包。

| 项目 | 当前状态 |
|---|---|
| 训练脚本 | `1_training/ppko/pretrain_v10b_strong300.py`，默认 620 epoch |
| 冻结权重名 | `models/scp682_ppko_v10b_strong300_best.pt` |
| 兼容旧权重名 | 训练脚本同时写出 `scp682_ppko_attention_prior_v10_best.pt` |
| P100 全药物验证 | `2_analysis/ppko/validate_v10b_p100_all_drugs.py` |
| TCGA-TCPA 患者响应 | `2_analysis/ppko/tcga_tcpa_ppko_patient_response_v1.py`，从冻结包导入模型 |
| 已发表基线对照 | `2_analysis/ppko/evaluate_ppko_p100_published_baselines.py` |
| release-v1 旧补表脚本 | 已归档到 `organized_code/legacy/ppko/rerun_missing_items_release_v10.py` |

## 已修正的旧问题

- 患者响应脚本不再从 `20260520_SCP682_PPKO_release_v1` 导入旧训练脚本。
- `v10b_300` 不再指向 `scp682_ppko_attention_prior_v10_best.pt` 作为主权重。
- P100 验证脚本已替换为冻结包版本。
- `rerun_missing_items.py` 已改为当前 V10B 补跑调度入口。

## 仍需人工确认

- Fig4a 正式图如果使用 BioRender，需要记录源文件路径。
- 部分画图脚本仍使用本地绝对路径读取锁定源表，复现时需保证源表目录存在。
- P100 锁定源表中的历史文件名前缀 `global_graph_v9` 是结果文件命名遗留，不应解释为模型版本。
