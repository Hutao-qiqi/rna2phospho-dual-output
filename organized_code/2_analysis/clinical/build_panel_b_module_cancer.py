# 新 panel b: 24 生物学 pathway_module × 28 癌种 risk-minus-protective 热图(从 ARCH 重算, FDR)
# 行序与 c 一致(按泛癌 risk_frac), 列序与 a 一致(cancer_order) -> b/c 行对行对应
import pandas as pd, numpy as np
from pathlib import Path
SRC = Path(r"E:/data/gongke/TCGA-TCPA/04_figures/20260528_fig5_v2/tables/full_tcga_scp682_main_20260529")
OUT = Path(r"E:/data/gongke/TCGA-TCPA/paper_final/fig5/source_data/tables")

arch = pd.read_csv(SRC/"full_tcga_scp682_main_architecture_effect_matrix.tsv.gz", sep="\t")
ev = arch[arch["analysis_evaluable"].fillna(False)].copy()
ev["clin_q"] = pd.to_numeric(ev["cox_q_full_bh"],errors="coerce")<0.05
ev["dir"] = np.where(pd.to_numeric(ev["cox_beta_full"],errors="coerce")>0,"risk","protective")
sig = ev[ev["clin_q"] & (ev["pathway_module"]!="Other context-linked / other")].copy()
sig["cs"] = sig["cancer"].str.replace("TCGA-","",regex=False)

# 癌种顺序 from panel a (28, drop-zero 已在 a 处理)
top = pd.read_csv(OUT/"panel_a_cancer_entry_top_bars.tsv", sep="\t")
cancer_order = top.sort_values("cancer_order")["cancer_short"].tolist()
# module 顺序 from panel c (n>=30), 升序(ggplot bottom->top = risk 在上)
modc = pd.read_csv(OUT/"fig5c_pathway_module_direction.tsv", sep="\t")
modc = modc[modc["n_clin"]>=30].copy()
mod_order = modc.sort_values("risk_frac_clin")["pathway_module"].tolist()

gg = sig.groupby(["pathway_module","cs","dir"]).size().reset_index(name="k")
piv = gg.pivot_table(index=["pathway_module","cs"], columns="dir", values="k", fill_value=0).reset_index()
for c in ["risk","protective"]:
    if c not in piv: piv[c]=0
piv["n"]=piv["risk"]+piv["protective"]
piv["rmp"]=(piv["risk"]-piv["protective"])/piv["n"].replace(0,np.nan)

mat = piv.pivot(index="pathway_module", columns="cs", values="rmp")
nct = piv.pivot(index="pathway_module", columns="cs", values="n")
keep_mods = [m for m in mod_order if m in mat.index]
keep_canc = [c for c in cancer_order if c in mat.columns]
mat = mat.reindex(index=keep_mods, columns=keep_canc)
nct = nct.reindex(index=keep_mods, columns=keep_canc)
mat.to_csv(OUT/"fig5b_module_cancer_rmp.tsv", sep="\t")
nct.to_csv(OUT/"fig5b_module_cancer_n.tsv", sep="\t")

meta = modc[["pathway_module","risk_frac_clin","n_clin","rep3","module_direction"]].copy()
meta["short"] = meta["pathway_module"].str.replace(r"^.*/ ","",regex=True)
meta = meta.set_index("pathway_module").reindex(keep_mods).reset_index()
meta.to_csv(OUT/"fig5b_module_meta.tsv", sep="\t", index=False)

print("modules:", len(keep_mods), " cancers:", len(keep_canc))
print("cancers:", keep_canc)
print("\nrmp matrix (rounded):")
print(mat.round(2).to_string())
