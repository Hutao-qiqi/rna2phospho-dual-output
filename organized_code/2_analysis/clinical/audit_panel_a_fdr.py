# 诊断3: 精确反推 flag 组合公式 + 全 category 的 nominal-p vs BH-q 计数对照
import pandas as pd, numpy as np
from pathlib import Path
SRC = Path(r"E:/data/gongke/TCGA-TCPA/04_figures/20260528_fig5_v2/tables/full_tcga_scp682_main_20260529")
ev = pd.read_csv(SRC/"full_tcga_scp682_main_architecture_effect_matrix.tsv.gz", sep="\t")
ev = ev[ev["analysis_evaluable"].fillna(False)].copy()
cs  = ev["clinical_significant"].fillna(False)
pmi = ev["parent_mrna_independent"].fillna(False)
grs = ev["graph_residual_significant"].fillna(False)
def lt(c, thr=0.05): return (pd.to_numeric(ev[c],errors="coerce")<thr).fillna(False)

print("=== reverse-engineer parent_mrna_independent (TRUE=%d) ===" % pmi.sum())
for k,v in {
 "cs & site_p_adj<.05": cs & lt("site_p_adjusted_for_parent_mrna"),
 "cs & add_lrt_p<.05": cs & lt("add_site_to_parent_mrna_lrt_p"),
 "cs & site_p_adj & add_lrt_p": cs & lt("site_p_adjusted_for_parent_mrna") & lt("add_site_to_parent_mrna_lrt_p"),
 "site_p_adj & add_lrt_p": lt("site_p_adjusted_for_parent_mrna") & lt("add_site_to_parent_mrna_lrt_p"),
}.items():
    print(f"  PMI == [{k}] agree={(pmi==v).mean():.4f}")

print("=== reverse-engineer graph_residual_significant (TRUE=%d) ===" % grs.sum())
for k,v in {
 "cs & graph_delta_p<.05": cs & lt("graph_delta_p"),
 "cs & graph_delta_p_adj<.05": cs & lt("graph_delta_p_adjusted_for_baseline"),
 "cs & gain_p<.05": cs & lt("gain_p"),
 "cs & graph_delta_p & graph_delta_p_adj": cs & lt("graph_delta_p") & lt("graph_delta_p_adjusted_for_baseline"),
}.items():
    print(f"  GRS == [{k}] agree={(grs==v).mean():.4f}")

# 全 category: nominal-p (现状) vs BH-q 计数
def cnt(mask, beta_col=None):
    if beta_col is None: return int(mask.sum())
    b = pd.to_numeric(ev.loc[mask, beta_col], errors="coerce")
    return int((b>0).sum()), int((b<0).sum()), int(mask.sum())

print("\n=== category counts: nominal-p (CURRENT)  vs  BH-q ===")
print("clinical sites:           p=%d   q=%d" % (cs.sum(), lt("cox_q_full_bh").sum()))
# beyond mRNA: 用反推到的最佳组合, p 版与 q 版
pmi_q = cs & lt("add_site_to_parent_mrna_lrt_q_bh")  # q 版 (若组合=cs&add_lrt)
r,pr,t = cnt(pmi, "site_beta_adjusted_for_parent_mrna"); print("beyond parent mRNA (current flag): risk=%d prot=%d total=%d" % (r,pr,t))
r,pr,t = cnt(pmi_q,"site_beta_adjusted_for_parent_mrna"); print("beyond parent mRNA (cs & add_lrt_q_bh<.05): risk=%d prot=%d total=%d" % (r,pr,t))
# graph residual p vs q
grs_q = cs & lt("gain_q_bh")
print("graph residual (current flag): %d" % grs.sum())
print("graph residual (cs & gain_q_bh<.05): %d" % grs_q.sum())
