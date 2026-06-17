# winscatter_prep.py — GSE300551 matched 读数集上，SCP682-SC vs 每个 baseline 的逐读数 Spearman 长表。
# 复用 benchmark_leaderboard_prep 的匹配集（SCP682-SC ∩ 6 backbone），但导出 per-readout 用于 win-scatter grid。
# 展示的 baseline 与柱状 b 一致：Cognate mRNA + 5 backbone（scFoundation 作为 head ablation，不在主擂台展示）。

import os, numpy as np, pandas as pd
from scipy import stats

RV = ("E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11/"
      "04_figure_source_data/reviewer_requested_tables_v2")
FM = ("E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11/"
      "04_figure_source_data/foundation_model_linear_regression_benchmark_v1_plus_remaining")
OUT = "E:/data/gongke/TCGA-TCPA/paper_final/fig3/main_figure_biology_v1/source_data"
GSE = "gse300551_iccite_plex_kinase_2025"

# --- SCP682-SC 逐读数 ---
full = pd.read_csv(os.path.join(RV, "benchmark_table_reviewer_full_per_target.tsv"), sep="\t")
sc = full[(full.method_name == "SCP682-SC") & (full.cohort_id == GSE)].copy()
sc["spearman"] = pd.to_numeric(sc["spearman"], errors="coerce")
sc = sc.dropna(subset=["spearman"]).drop_duplicates("target_id").set_index("target_id")["spearman"]

# --- cognate mRNA 逐读数 ---
cog = pd.read_csv(os.path.join(RV, "cognate_mRNA_per_target.tsv"), sep="\t")
cog = cog[(cog.dataset_id == GSE) & (cog.spearman_status == "ok")].copy()
cog["spearman"] = pd.to_numeric(cog["spearman"], errors="coerce")
cog = cog.dropna(subset=["spearman"]).drop_duplicates("target_id").set_index("target_id")["spearman"]

# --- 6 backbone + LR 逐读数 ---
fl = pd.read_csv(os.path.join(FM, "single_cell_foundation_backbone_linear_regression_per_target_current.tsv"),
                 sep="\t")
fl = fl[fl.test_dataset == GSE].copy()
fl["spearman"] = pd.to_numeric(fl["spearman"], errors="coerce")
fl = fl.dropna(subset=["spearman"])
bb_disp = {"Geneformer_pathway": "Geneformer (LR)", "scFoundation": "scFoundation (LR)",
           "scGPT": "scGPT (LR)", "UCE_4layer": "UCE (LR)",
           "tGPT_top64": "tGPT (LR)", "scimilarity_v1_1": "SCimilarity (LR)"}
bb = {d: fl[fl.model_name == m].drop_duplicates("target_id").set_index("target_id")["spearman"]
      for m, d in bb_disp.items()}

# --- matched 集 = SCP682-SC ∩ 全部 6 backbone（与 leaderboard 同口径）---
common = set(sc.index)
for s in bb.values():
    common &= set(s.index)
common = sorted(common)
print(f"matched readouts: n={len(common)}")

# --- 展示的 baseline（去掉 scFoundation，与柱状 b 一致）---
shown = {"Cognate mRNA": cog, "tGPT (LR)": bb["tGPT (LR)"], "scGPT (LR)": bb["scGPT (LR)"],
         "UCE (LR)": bb["UCE (LR)"], "Geneformer (LR)": bb["Geneformer (LR)"],
         "SCimilarity (LR)": bb["SCimilarity (LR)"]}

# --- 读出显示名 / 通路（从 external_by_readout 的 GSE 行借）---
ext = pd.read_csv(os.path.join(OUT, "fig3_biology_external_by_readout.tsv"), sep="\t")
extg = ext[ext.cohort == "GSE300551"].drop_duplicates("target_id").set_index("target_id")
disp_map = extg["target_display"].to_dict() if "target_display" in extg else {}
fam_map = extg["family_display"].to_dict() if "family_display" in extg else {}

rows = []
for bname, bs in shown.items():
    bs2 = bs.reindex(common)
    for t in common:
        b = bs2.get(t, np.nan)
        if not np.isfinite(b):
            continue
        rows.append(dict(target_id=t, target_display=disp_map.get(t, t),
                         family_display=fam_map.get(t, "other"),
                         baseline=bname, base_rho=float(b), scp_rho=float(sc[t])))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "fig3_benchmark_gse300551_per_readout.tsv"), sep="\t", index=False)

# --- 每 baseline 胜场 + 配对 Wilcoxon（与柱状一致口径）---
print("\nbaseline                 n   wins  medianΔ   P(SCP>base)")
summ = []
for bname, bs in shown.items():
    sub = df[df.baseline == bname]
    n = len(sub); wins = int((sub.scp_rho > sub.base_rho).sum())
    dmed = float(np.median(sub.scp_rho - sub.base_rho))
    m = sub.scp_rho.values != sub.base_rho.values
    p = stats.wilcoxon(sub.scp_rho.values[m], sub.base_rho.values[m],
                       alternative="greater").pvalue if m.sum() >= 5 else np.nan
    summ.append(dict(baseline=bname, n=n, wins=wins, median_delta=dmed, p_value=p))
    print(f"{bname:22s} {n:3d}  {wins:3d}/{n:<3d} {dmed:+.3f}   {p:.2e}" if np.isfinite(p)
          else f"{bname:22s} {n:3d}  {wins:3d}/{n:<3d} {dmed:+.3f}   n/a")
pd.DataFrame(summ).to_csv(os.path.join(OUT, "fig3_benchmark_gse300551_winsummary.tsv"), sep="\t", index=False)
print("\nwrote fig3_benchmark_gse300551_per_readout.tsv + winsummary.tsv")
