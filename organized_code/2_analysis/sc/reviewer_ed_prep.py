# reviewer_ed_prep.py — 为 reviewer 扩展图 fig22 / fig23 / fig25 生成整洁中间表。
#
# 配对/聚合在 Python 里做（pandas + scipy），避免 R 端数据整形出错；
# 产出的 tidy TSV 写回 fig3 源文件夹，R 脚本只负责读表 + 渲染。
#
# 产物（写到 04_figure_source_data/fig3/）：
#   fig22_component_ablation_paired_per_target.tsv   每 (variant,cohort,target) 的配对 Δ
#   fig22_component_ablation_paired_summary.tsv       每 (variant,cohort) 的中位 Δ + n + Wilcoxon p
#   fig23_gnn_vs_mlp_per_target.tsv                    每 (dataset,target) 跨 5 折中位
#   fig25_calibration_curves_z.tsv                    每 (cohort,target,bin) 的 z 化 pred/obs
#   fig25_calibration_cohort_summary.tsv              每 cohort 的中位 bin-Spearman + n
#
# 仅做诚实聚合：scFoundation 变体在不同 target 子集上跑过 → 必须在「共享 target」上配对，
# 否则逐队列中位 Δ 会被子集差异污染（已在 Results 标注的混淆点）。

import os
import numpy as np
import pandas as pd
from scipy import stats

RV = ("E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11/"
      "04_figure_source_data/reviewer_requested_tables_v2")
DT = ("E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11/"
      "02_data_tables")
OUT = ("E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11/"
       "04_figure_source_data/fig3")


def w(df, name):
    p = os.path.join(OUT, name)
    df.to_csv(p, sep="\t", index=False)
    print(f"  wrote {name}  ({df.shape[0]} rows x {df.shape[1]} cols)")


# ------------------------------------------------------------------ fig22
# 组件消融：每个 non-full variant 与 full 在「共享且两侧都有限」的 target 上配对求 Δ。
print("[fig22] component ablation paired Δ on shared readouts")
ca = pd.read_csv(os.path.join(RV, "component_ablation_reviewer_per_target.tsv"),
                 sep="\t")
ca["spearman"] = pd.to_numeric(ca["spearman"], errors="coerce")

full = ca[ca["component_variant"] == "full_SCP682_SC"].copy()
full_map = {(r.cohort_id, r.target_id): r.spearman for r in full.itertuples()}

VAR_NICE = {
    "pathway_attention_removed":               "- pathway attn",
    "expanded_site_graph_removed_matched":     "- expanded graph",
    "scFoundation_removed_raw_expression_ridge": "- scFoundation",
}
VAR_ORDER = ["- pathway attn", "- expanded graph", "- scFoundation"]

rows = []
for var_raw, nice in VAR_NICE.items():
    sub = ca[ca["component_variant"] == var_raw]
    for r in sub.itertuples():
        f = full_map.get((r.cohort_id, r.target_id), np.nan)
        v = r.spearman
        if np.isfinite(f) and np.isfinite(v):
            rows.append(dict(variant=nice, variant_raw=var_raw,
                             cohort_id=r.cohort_id, cohort_name=r.cohort_name,
                             target_id=r.target_id,
                             full_spearman=f, variant_spearman=v,
                             delta=v - f))
per = pd.DataFrame(rows)
per["variant"] = pd.Categorical(per["variant"], VAR_ORDER, ordered=True)
per = per.sort_values(["variant", "cohort_name", "delta"])
w(per, "fig22_component_ablation_paired_per_target.tsv")

srows = []
for (nice, coh), g in per.groupby(["variant", "cohort_name"], observed=True):
    d = g["delta"].values
    n = len(d)
    p = np.nan
    if n >= 3:
        # 单侧：variant 是否劣于 full（delta < 0）
        try:
            p = stats.wilcoxon(d, alternative="less").pvalue
        except ValueError:
            p = np.nan
    srows.append(dict(variant=nice, cohort_name=coh, n_shared=n,
                      median_delta=float(np.median(d)),
                      mean_delta=float(np.mean(d)),
                      median_full=float(np.median(g["full_spearman"])),
                      median_variant=float(np.median(g["variant_spearman"])),
                      wilcoxon_p_less=p))
summ = pd.DataFrame(srows)
summ["variant"] = pd.Categorical(summ["variant"], VAR_ORDER, ordered=True)
summ = summ.sort_values(["variant", "cohort_name"])
w(summ, "fig22_component_ablation_paired_summary.tsv")
print(summ.to_string(index=False))


# ------------------------------------------------------------------ fig23
# GNN vs site-aware MLP：每 (dataset,target) 跨 5 折取中位，得到每 readout 一个配对点。
print("\n[fig23] GNN vs site-aware MLP per-target medians")
gm = pd.read_csv(os.path.join(RV, "gnn_vs_site_aware_mlp_paired_per_fold_target.tsv"),
                 sep="\t")
for c in ["scp682_sc_spearman", "site_aware_mlp_spearman",
          "delta_scp682_sc_minus_mlp"]:
    gm[c] = pd.to_numeric(gm[c], errors="coerce")
agg = (gm.groupby(["test_dataset", "target_id"])
         .agg(scp682_sc_median=("scp682_sc_spearman", "median"),
              mlp_median=("site_aware_mlp_spearman", "median"),
              delta_median=("delta_scp682_sc_minus_mlp", "median"),
              n_folds=("fold", "nunique"))
         .reset_index())
agg = agg.sort_values(["test_dataset", "delta_median"])
w(agg, "fig23_gnn_vs_mlp_per_target.tsv")
print(f"  datasets: {sorted(agg['test_dataset'].unique())}")
print(f"  n targets: {agg.groupby('test_dataset').size().to_dict()}")


# ------------------------------------------------------------------ fig25
# 校准：每 (cohort,target) 把 predicted / observed 的 bin 均值各自 z 化（跨抗体尺度可比）。
print("\n[fig25] calibration reliability (z-scored bins)")
cb = pd.read_csv(os.path.join(RV, "calibration_bins_by_cohort_target.tsv"), sep="\t")
for c in ["predicted_mean", "observed_mean", "prediction_bin"]:
    cb[c] = pd.to_numeric(cb[c], errors="coerce")


def zsc(x):
    x = np.asarray(x, float)
    s = x.std(ddof=0)
    return (x - x.mean()) / s if s > 0 else x * np.nan


zrows = []
for (coh, tgt), g in cb.groupby(["cohort_id", "target_id"]):
    g = g.sort_values("prediction_bin")
    if g.shape[0] < 4:
        continue
    pz = zsc(g["predicted_mean"].values)
    oz = zsc(g["observed_mean"].values)
    if not (np.all(np.isfinite(pz)) and np.all(np.isfinite(oz))):
        continue
    for b, a, o in zip(g["prediction_bin"].values, pz, oz):
        zrows.append(dict(cohort_id=coh, target_id=tgt,
                          prediction_bin=int(b), pred_z=a, obs_z=o))
zc = pd.DataFrame(zrows)
w(zc, "fig25_calibration_curves_z.tsv")

cs = pd.read_csv(os.path.join(RV, "calibration_summary_by_cohort_target.tsv"), sep="\t")
cs["spearman_bin_mean"] = pd.to_numeric(cs["spearman_bin_mean"], errors="coerce")
csum = (cs.groupby("cohort_id")
          .agg(n_targets=("target_id", "nunique"),
               median_bin_spearman=("spearman_bin_mean", "median"),
               mean_bin_spearman=("spearman_bin_mean", "mean"))
          .reset_index())
w(csum, "fig25_calibration_cohort_summary.tsv")
print(csum.to_string(index=False))

print("\nDONE.")
