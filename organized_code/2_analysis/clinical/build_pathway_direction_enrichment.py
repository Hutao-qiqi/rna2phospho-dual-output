# 点2: pathway_family x 方向(risk/protective) 富集 —— 给 M1-M7 抽象模块换成生物学通路论断
# FDR clinical-sig 与 beyond-parent-mRNA 两个口径; 泛癌占比 + 每家族 vs 全局 Fisher
import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import fisher_exact
SRC = Path(r"E:/data/gongke/TCGA-TCPA/04_figures/20260528_fig5_v2/tables/full_tcga_scp682_main_20260529")
OUT = Path(r"E:/data/gongke/TCGA-TCPA/paper_final/fig5/source_data/tables")

arch = pd.read_csv(SRC/"full_tcga_scp682_main_architecture_effect_matrix.tsv.gz", sep="\t")
ev = arch[arch["analysis_evaluable"].fillna(False)].copy()
q = lambda c: pd.to_numeric(ev[c], errors="coerce") < 0.05
ev["clin_q"] = q("cox_q_full_bh")
ev["pmi_q"]  = ev["clin_q"] & q("add_site_to_parent_mrna_lrt_q_bh")
ev["dir"] = np.where(pd.to_numeric(ev["cox_beta_full"],errors="coerce") > 0, "risk", "protective")

def enrich(mask, label):
    s = ev[mask]
    g = s.groupby("pathway_family")["dir"].value_counts().unstack(fill_value=0)
    for col in ["risk","protective"]:
        if col not in g: g[col]=0
    g["n"] = g["risk"] + g["protective"]
    g["risk_frac"] = g["risk"] / g["n"]
    glob_risk = s["dir"].eq("risk").sum(); glob_prot = s["dir"].eq("protective").sum()
    # 每家族 risk 富集 vs 其余 (Fisher one-sided greater)
    p_enrich=[]
    for fam,row in g.iterrows():
        a=row["risk"]; b=row["protective"]; c=glob_risk-a; d=glob_prot-b
        _,p = fisher_exact([[a,b],[c,d]], alternative="two-sided")
        p_enrich.append(p)
    g["fisher_p_vs_rest"]=p_enrich
    g["scope"]=label
    g=g.sort_values("risk_frac", ascending=False)
    print(f"\n===== {label}  (global risk%={100*glob_risk/(glob_risk+glob_prot):.1f}) =====")
    print(g[["risk","protective","n","risk_frac","fisher_p_vs_rest"]].round(3).to_string())
    return g.reset_index()

e1 = enrich(ev["clin_q"], "clinical_FDR")
e2 = enrich(ev["pmi_q"],  "beyond_parent_mRNA_FDR")
out = pd.concat([e1,e2], ignore_index=True)
out.to_csv(OUT/"fig5b_pathway_family_direction_enrichment.tsv", sep="\t", index=False)
print("\nwrote", OUT/"fig5b_pathway_family_direction_enrichment.tsv")
