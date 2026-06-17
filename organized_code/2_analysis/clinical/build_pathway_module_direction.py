# panel c (新): 24 生物学 pathway_module 的 FDR 方向富集 + 代表基因
# 用 module 级(细分, 解决 family 混杂); risk_frac=HR>1 占比; 接 representative_genes
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
ev["dir"] = np.where(pd.to_numeric(ev["cox_beta_full"],errors="coerce")>0, "risk", "protective")

def frac(mask):
    s = ev[mask]
    g = s.groupby("pathway_module")["dir"].value_counts().unstack(fill_value=0)
    for c in ["risk","protective"]:
        if c not in g: g[c]=0
    g["n"]=g["risk"]+g["protective"]; g["risk_frac"]=g["risk"]/g["n"]
    return g

clin = frac(ev["clin_q"]).add_suffix("_clin")
bm   = frac(ev["pmi_q"]).add_suffix("_bm")
out = clin.join(bm, how="left").reset_index()

# 代表基因 + family
mods = pd.read_csv(SRC/"full_tcga_scp682_main_fig5b_pathway_module_modules.tsv", sep="\t")
mods["rep3"] = mods["representative_genes"].fillna("").apply(lambda s: ", ".join(s.split(";")[:3]))
out = out.merge(mods[["pathway_module","rep3","module_direction","dominant_cancer"]], on="pathway_module", how="left")
out["pathway_family"] = out["pathway_module"].str.split(" / ").str[0]

# Fisher: risk 富集 vs 其余 (clinical FDR)
sig = ev[ev["clin_q"]]; gr=sig["dir"].eq("risk").sum(); gp=sig["dir"].eq("protective").sum()
def fp(r,p):
    _,pv=fisher_exact([[r,p],[gr-r,gp-p]],alternative="two-sided"); return pv
out["fisher_p_clin"]=[fp(r,p) for r,p in zip(out["risk_clin"].fillna(0).astype(int), out["protective_clin"].fillna(0).astype(int))]

out = out[out["pathway_module"]!="Other context-linked / other"]
out = out.sort_values("risk_frac_clin", ascending=False)
out.to_csv(OUT/"fig5c_pathway_module_direction.tsv", sep="\t", index=False)

print(f"global clinical risk_frac = {gr/(gr+gp):.3f}")
print(out[["pathway_module","risk_clin","protective_clin","n_clin","risk_frac_clin","risk_frac_bm","rep3","fisher_p_clin"]].round(3).to_string(index=False))
print("\nwrote", OUT/"fig5c_pathway_module_direction.tsv", " | n_modules(n>=30):", (out["n_clin"]>=30).sum())
