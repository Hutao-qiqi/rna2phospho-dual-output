#!/usr/bin/env python
"""Analyse TCGA pan-cancer NMF modules: module x cancer activity, top sites,
Hallmark ORA, auto-naming. Produces tables consumed by the R figure scripts."""
import numpy as np, pandas as pd, os, sys
from scipy.stats import hypergeom

K = int(sys.argv[1]) if len(sys.argv) > 1 else 30
ROOT = "E:/data/gongke/TCGA-TCPA/02_results/model_validation/20260612_tcga_pancancer_nmf_v1"
OUT = ROOT + "/results"
GMT = "E:/data/gongke/TCGA-TCPA/resources/msigdb/h.all.v2025.1.Hs.symbols.gmt"
TOPN = 150           # top sites per module (pos direction) for enrichment

W = np.load(f"{OUT}/W_k{K}.npy")          # samples x K
H = np.load(f"{OUT}/H_k{K}.npy")          # K x 37184  (first half = Z+ , second = Z-)
meta = pd.read_csv(f"{OUT}/sample_meta.tsv", sep="\t")
sites = pd.read_csv(f"{OUT}/site_names.tsv", sep="\t")["site"].tolist()
nS = len(sites)
mods = [f"M{i+1}" for i in range(K)]
print(f"k={K}  W{W.shape}  H{H.shape}  nsites={nS}", flush=True)

# ---- module activity z-scored across samples ----
Wz = (W - W.mean(0)) / (W.std(0) + 1e-9)
df = pd.DataFrame(Wz, columns=mods); df["cancer"] = meta.cancer.values

# ---- module x cancer median activity ----
mbc = df.groupby("cancer")[mods].median().T          # K x cancers
mbc.to_csv(f"{OUT}/module_by_cancer_median_k{K}.tsv", sep="\t")
pref_cancer = mbc.idxmax(1); pref_z = mbc.max(1)

# ---- per-module top sites (pos = relative up, neg = relative down) ----
Hpos = H[:, :nS]; Hneg = H[:, nS:]
gene_of = np.array([s.split("|")[0] for s in sites])

# ---- Hallmark ORA setup ----
hall = {}
for ln in open(GMT):
    p = ln.rstrip("\n").split("\t")
    hall[p[0].replace("HALLMARK_", "")] = set(p[2:])
bg_genes = set(gene_of)                                # background = genes with >=1 measured site
M_bg = len(bg_genes)
hall = {k: (v & bg_genes) for k, v in hall.items()}

def bh(pvals):
    p = np.asarray(pvals); n = len(p); o = np.argsort(p)
    r = p[o] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1.0); return out

enr_rows, summ_rows = [], []
for mi, m in enumerate(mods):
    top_idx = np.argsort(Hpos[mi])[::-1][:TOPN]
    fg = set(gene_of[top_idx]); n_fg = len(fg)
    names, ks, Ks, ps = [], [], [], []
    for hname, hset in hall.items():
        Kk = len(hset)
        if Kk < 5: continue
        kk = len(fg & hset)
        pv = hypergeom.sf(kk - 1, M_bg, Kk, n_fg)
        names.append(hname); ks.append(kk); Ks.append(Kk); ps.append(pv)
    q = bh(ps)
    e = pd.DataFrame({"module": m, "pathway": names, "k": ks, "K": Ks,
                      "p": ps, "fdr": q})
    e = e.sort_values("p")
    e["neg_log10_p"] = -np.log10(e.p.clip(lower=1e-300))
    e["rank"] = np.arange(1, len(e) + 1)
    enr_rows.append(e)
    topn_names = "; ".join(e.pathway.head(3))
    top_sites = "; ".join([sites[i] for i in top_idx[:8]])
    summ_rows.append({"module": m, "pref_cancer": pref_cancer[m],
                      "pref_z": round(float(pref_z[m]), 2),
                      "top_hallmark": e.pathway.iloc[0],
                      "top_fdr": float(e.fdr.iloc[0]),
                      "top3_hallmark": topn_names, "top_sites": top_sites})

enr = pd.concat(enr_rows, ignore_index=True)
enr.to_csv(f"{OUT}/module_pathway_enrichment_k{K}.tsv", sep="\t", index=False)
summ = pd.DataFrame(summ_rows)
summ = summ.sort_values(["pref_cancer", "pref_z"], ascending=[True, False])
summ.to_csv(f"{OUT}/module_summary_k{K}.tsv", sep="\t", index=False)

# also write per-sample module activity (for fig A / fig D)
pd.DataFrame(Wz, columns=mods).assign(sample_id=meta.sample_id.values,
    cancer=meta.cancer.values).to_csv(f"{OUT}/sample_module_activity_k{K}.tsv",
    sep="\t", index=False)

print("=== module summary (k=%d) ===" % K, flush=True)
print(summ.to_string(index=False), flush=True)
