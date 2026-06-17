# benchmark_leaderboard_prep.py — GSE300551 公平擂台（matched 读数集）。
# 各方法在「SCP682-SC 与 6 个大模型 backbone 都有限」的共同读数集上算逐读数 Spearman，
# 得 median/IQR + 配对 Wilcoxon(SCP682-SC > baseline)。cognate mRNA 限到同一读数集。
# raw-expr / per-site ridge 按既定决定不进主擂台（补充里已披露）。

import os, numpy as np, pandas as pd
from scipy import stats

RV = ("E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11/"
      "04_figure_source_data/reviewer_requested_tables_v2")
FM = ("E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11/"
      "04_figure_source_data/foundation_model_linear_regression_benchmark_v1_plus_remaining")
OUT = "E:/data/gongke/TCGA-TCPA/paper_final/fig3/main_figure_biology_v1/source_data"
GSE = "gse300551_iccite_plex_kinase_2025"

# --- SCP682-SC 逐读数（GSE）---
full = pd.read_csv(os.path.join(RV, "benchmark_table_reviewer_full_per_target.tsv"), sep="\t")
sc = full[(full.method_name == "SCP682-SC") & (full.cohort_id == GSE)].copy()
sc["spearman"] = pd.to_numeric(sc["spearman"], errors="coerce")
sc = sc.dropna(subset=["spearman"])[["target_id", "spearman"]].drop_duplicates("target_id")

# --- cognate mRNA 逐读数（GSE）---
cog = pd.read_csv(os.path.join(RV, "cognate_mRNA_per_target.tsv"), sep="\t")
cog = cog[(cog.dataset_id == GSE) & (cog.spearman_status == "ok")].copy()
cog["spearman"] = pd.to_numeric(cog["spearman"], errors="coerce")
cog = cog.dropna(subset=["spearman"])[["target_id", "spearman"]].drop_duplicates("target_id")

# --- 6 个 backbone + 线性回归 逐读数（GSE）---
fl = pd.read_csv(os.path.join(FM, "single_cell_foundation_backbone_linear_regression_per_target_current.tsv"),
                 sep="\t")
fl = fl[fl.test_dataset == GSE].copy()
fl["spearman"] = pd.to_numeric(fl["spearman"], errors="coerce")
fl = fl.dropna(subset=["spearman"])
bb_disp = {"Geneformer_pathway": "Geneformer (LR)", "scFoundation": "scFoundation (LR)",
           "scGPT": "scGPT (LR)", "UCE_4layer": "UCE (LR)",
           "tGPT_top64": "tGPT (LR)", "scimilarity_v1_1": "SCimilarity (LR)"}
bb = {d: fl[fl.model_name == m][["target_id", "spearman"]].drop_duplicates("target_id")
            .set_index("target_id")["spearman"]
      for m, d in bb_disp.items()}

# --- matched 读数集 = SCP682-SC ∩ 全部 6 backbone ---
sc_s = sc.set_index("target_id")["spearman"]
common = set(sc_s.index)
for s in bb.values():
    common &= set(s.index)
common = sorted(common)
print(f"matched readouts (SCP682-SC ∩ 6 backbones) on GSE300551: n={len(common)}")
print(common)

# --- 组装方法 × 读数 矩阵（同一 matched 集）---
methods = {"SCP682-SC": sc_s}
methods["Cognate mRNA"] = cog.set_index("target_id")["spearman"]
for d, s in bb.items():
    methods[d] = s

rows = []
scp_vec = sc_s.reindex(common).values
for name, s in methods.items():
    v = s.reindex(common).values.astype(float)
    ok = np.isfinite(v)
    vv = v[ok]
    n = len(vv)
    if name == "SCP682-SC":
        p = np.nan
    else:
        # 配对（仅两者都有限的读数）
        m = np.isfinite(v) & np.isfinite(scp_vec)
        if m.sum() >= 5 and np.any(scp_vec[m] != v[m]):
            p = stats.wilcoxon(scp_vec[m], v[m], alternative="greater").pvalue
        else:
            p = np.nan
    rows.append(dict(method=name, n=n,
                     median=float(np.median(vv)), q25=float(np.percentile(vv, 25)),
                     q75=float(np.percentile(vv, 75)),
                     p_value=p))

df = pd.DataFrame(rows).sort_values("median").reset_index(drop=True)
def sig(p):
    if not np.isfinite(p): return ""
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
df["signif"] = df.p_value.apply(sig)
os.makedirs(OUT, exist_ok=True)
df.to_csv(os.path.join(OUT, "fig3_benchmark_gse300551_leaderboard.tsv"), sep="\t", index=False)
print()
print(df.to_string(index=False))
