# external_funnel_prep.py — 锁定外部验证「分母」并拆分 精度 vs 覆盖。
#
# 解决两个会被 NM reviewer 一击致命的问题：
#  (1) 不同方法在不同位点集上比 median（covereage 与 accuracy 混淆）；
#  (2) median 分母未锁定 → 跨版本漂移（HeLa 0.402↔0.498）。
#
# 口径（对每个外部队列，统一 a priori 过滤）：
#   56 vocab → mappable → external-finite(非零方差, SC 可算 Spearman)
#        → both_eval_cognate   : SC 与 cognate mRNA 都有限（可做同位点精度 head-to-head）
#        → trained_supervised  : 该位点在训练集(iCCITE/QuRIE)有同名监督 → 需训练的基线
#                                （foundation-embedding 线性回归 / site-aware MLP）才可能拟合
#        → transfer_only       : SC 可评但训练集无同名监督 → 需训练的基线 n_predictable=0，
#                                SC 经 site-graph/target-prior 仍可预测（覆盖能力点）
#
# 所有数值取自权威表（20260522 expanded_scnet 正式模型）。不用任何旧版/口述数。

import os
import numpy as np
import pandas as pd

RV = ("E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11/"
      "04_figure_source_data/reviewer_requested_tables_v2")
FM = ("E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11/"
      "04_figure_source_data/foundation_model_linear_regression_benchmark_v1_plus_remaining")
OUT = "E:/data/gongke/TCGA-TCPA/paper_final/fig3/supp_tables"
os.makedirs(OUT, exist_ok=True)


def w(df, name):
    df.to_csv(os.path.join(OUT, name), sep="\t", index=False)
    print(f"  wrote {name}  ({df.shape[0]}x{df.shape[1]})")


# ---- 权威 SCP682-SC 外部逐位点 ----
full = pd.read_csv(os.path.join(RV, "benchmark_table_reviewer_full_per_target.tsv"), sep="\t")
sc = full[(full.method_name == "SCP682-SC") &
          (full.evaluation_scope == "external_validation")].copy()
sc["spearman"] = pd.to_numeric(sc["spearman"], errors="coerce")
sc_map = {(r.cohort_id, r.target_id): r.spearman for r in sc.itertuples()}

# ---- cognate mRNA 外部逐位点 ----
cog = pd.read_csv(os.path.join(RV, "cognate_mRNA_per_target.tsv"), sep="\t")
cog["spearman"] = pd.to_numeric(cog["spearman"], errors="coerce")
cog_ok = cog[cog.spearman_status == "ok"]
cog_map = {(r.dataset_id, r.target_id): r.spearman for r in cog_ok.itertuples()}

# ---- 训练集同名监督的 38 读数（fivefold "all"）----
ff = pd.read_csv(os.path.join(RV, "fivefold_stability_by_readout.tsv"), sep="\t")
trained_vocab = set(ff[ff.test_dataset == "all"].target_id)
print(f"[trained-supervised vocab] n={len(trained_vocab)}")

COHORTS = [
    ("signal_seq_gse256403_hela_2024", "SIGNAL-seq HeLa"),
    ("gse300551_iccite_plex_kinase_2025", "GSE300551"),
    ("phospho_seq_blair_2025_phospho_multi", "Blair"),
    ("vivo_seq_th17_2025", "Vivo-seq Th17"),
    ("signal_seq_gse256404_pdo_caf_2024", "SIGNAL-seq PDO/CAF"),
]

funnel, persite = [], []
for cid, cname in COHORTS:
    sc_sites = sc[(sc.cohort_id == cid)]
    finite = sc_sites[np.isfinite(sc_sites.spearman)]
    finite_ids = list(finite.target_id)
    both, trained, transfer = [], [], []
    for tid in finite_ids:
        scv = sc_map[(cid, tid)]
        cgv = cog_map.get((cid, tid), np.nan)
        is_trained = tid in trained_vocab
        cls = "trained_supervised" if is_trained else "transfer_only"
        (trained if is_trained else transfer).append(tid)
        if np.isfinite(cgv):
            both.append(tid)
        persite.append(dict(cohort_id=cid, cohort_name=cname, target_id=tid,
                            scp682_sc=scv, cognate_mRNA=(cgv if np.isfinite(cgv) else np.nan),
                            delta_sc_minus_cognate=(scv - cgv if np.isfinite(cgv) else np.nan),
                            same_name_trained_supervised=is_trained, site_class=cls))
    med = float(np.median(finite.spearman)) if len(finite_ids) else np.nan
    funnel.append(dict(
        cohort_id=cid, cohort_name=cname,
        n_sc_evaluable_finite=len(finite_ids),
        sc_median_locked=med,
        n_both_eval_cognate=len(both),
        n_trained_supervised=len(trained),
        n_transfer_only=len(transfer),
        sc_evaluable_sites=";".join(finite_ids),
        transfer_only_sites=";".join(transfer)))

fdf = pd.DataFrame(funnel)
pdf = pd.DataFrame(persite)
w(fdf, "supp_external_funnel_by_cohort.tsv")
w(pdf, "supp_per_site_sc_vs_cognate.tsv")
print("\n[funnel]")
print(fdf[["cohort_name", "n_sc_evaluable_finite", "sc_median_locked",
           "n_both_eval_cognate", "n_trained_supervised", "n_transfer_only"]].to_string(index=False))

# ---- GSE300551 去重(每基因一克隆) vs 全克隆 两个锁定 median ----
gse = sc[(sc.cohort_id == "gse300551_iccite_plex_kinase_2025") & np.isfinite(sc.spearman)]
drop_clones = {"MAPK14_pSitePending_D3F9", "RELA_pSitePending_93H1"}
gse20 = float(np.median(gse.spearman))
gse18 = float(np.median(gse[~gse.target_id.isin(drop_clones)].spearman))
print(f"\n[GSE300551 lock] all-clones n={len(gse)} median={gse20:.4f} | "
      f"dedup(one clone/gene) n={len(gse)-2} median={gse18:.4f}")

# ---- HeLa / Vivo 逐位点 head-to-head（写出，正文/补充直接引用）----
for cname in ["SIGNAL-seq HeLa", "Vivo-seq Th17"]:
    sub = pdf[pdf.cohort_name == cname].sort_values("scp682_sc", ascending=False)
    print(f"\n[{cname}] per-site SC vs cognate")
    print(sub[["target_id", "scp682_sc", "cognate_mRNA",
               "delta_sc_minus_cognate", "same_name_trained_supervised"]].to_string(index=False))

# ---- 6 大模型 + 线性回归 透视（GSE 为主，其余 n=1 标注）----
fl = pd.read_csv(os.path.join(FM, "single_cell_foundation_backbone_linear_regression_summary_current.tsv"),
                 sep="\t")
fl["median_spearman"] = pd.to_numeric(fl["median_spearman"], errors="coerce")
fl["n_targets_finite"] = pd.to_numeric(fl["n_targets_finite"], errors="coerce")
cmap = {"gse300551_iccite_plex_kinase_2025": "GSE300551",
        "phospho_seq_blair_2025_phospho_multi": "Blair",
        "signal_seq_gse256403_hela_2024": "HeLa",
        "vivo_seq_th17_2025": "Vivo"}
fl["coh"] = fl.test_dataset.map(cmap)
piv = fl.pivot_table(index="model_name", columns="coh", values="median_spearman", aggfunc="first")
npiv = fl.pivot_table(index="model_name", columns="coh", values="n_targets_finite", aggfunc="first")
piv = piv[["GSE300551", "Blair", "HeLa", "Vivo"]]
print("\n[foundation backbone + linear regression] median Spearman (n in parens)")
for m in piv.index:
    cells = []
    for c in ["GSE300551", "Blair", "HeLa", "Vivo"]:
        v = piv.loc[m, c]; n = npiv.loc[m, c]
        cells.append(f"{c} {v:.3f} (n={int(n)})" if np.isfinite(v) else f"{c} NA")
    print(f"  {m:20s} " + " | ".join(cells))

# 写出透视表（median 与 n 并排）
rows = []
for m in piv.index:
    row = {"model": m}
    for c in ["GSE300551", "Blair", "HeLa", "Vivo"]:
        row[f"{c}_median"] = piv.loc[m, c]
        row[f"{c}_n"] = npiv.loc[m, c]
    rows.append(row)
w(pd.DataFrame(rows), "supp_foundation_backbone_linear_regression_pivot.tsv")

print("\nDONE.")
