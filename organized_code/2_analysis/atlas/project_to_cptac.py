#!/usr/bin/env python
"""Project TCGA-learned NMF modules (H) onto CPTAC measured phosphoproteome.
CPTAC mass-spec is ~55% missing, so we use MASKED non-negative least squares:
each sample's module activities are solved using ONLY its observed sites."""
import pandas as pd, numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

ROOT = "E:/data/gongke/TCGA-TCPA/02_results/model_validation/20260612_tcga_pancancer_nmf_v1"
OUT  = ROOT + "/results"; K = 30
OBS  = "E:/data/gongke/TCGA-TCPA/SCP682_MAIN/training_set/observed_phosphosite.parquet"
SC   = "E:/data/gongke/TCGA-TCPA/SCP682_MAIN/attention_export/nmf_sample_cancer.tsv"

sites = pd.read_csv(f"{OUT}/site_names.tsv", sep="\t")["site"].tolist()
nS = len(sites)
obs = pd.read_parquet(OBS); obs.index.name = "sample"
sc  = pd.read_csv(SC, sep="\t").set_index("sample")
obs = obs.loc[obs.index.isin(sc.index)]
cancer = sc.loc[obs.index, "cancer"].values
obs = obs.reindex(columns=sites)

X = obs.values.astype("float64")                       # NaN kept
mu = np.nanmean(X, 0); sd = np.nanstd(X, 0); sd[sd == 0] = 1
Zr = (X - mu) / sd
Zr = np.where(np.isnan(Zr), np.nan, np.clip(Zr, -8, 8))
H = np.load(f"{OUT}/H_k{K}.npy").astype("float64")     # K x 2nS  (pos|neg)
n = X.shape[0]

W = np.zeros((n, K)); cov = np.zeros(n, dtype=int)
for i in range(n):
    m = ~np.isnan(Zr[i]); cov[i] = m.sum()
    idx = np.where(m)[0]
    zi = Zr[i, idx]
    feat = np.concatenate([idx, idx + nS])
    xi = np.concatenate([np.clip(zi, 0, None), np.clip(-zi, 0, None)])
    W[i], _ = nnls(H[:, feat].T, xi)
    if (i + 1) % 300 == 0: print(f"  nnls {i+1}/{n}", flush=True)
print("mean observed sites/sample: %.0f / %d" % (cov.mean(), nS), flush=True)

Wz = (W - W.mean(0)) / (W.std(0) + 1e-9)
mods = [f"M{i+1}" for i in range(K)]
df = pd.DataFrame(Wz, columns=mods); df["cancer"] = cancer
cmbc = df.groupby("cancer")[mods].median().T
mapc = {"LSCC":"LUSC","CCRCC":"KIRC","HNSCC":"HNSC","PDA":"PAAD","ccPRCC":"KIRP"}
cmbc.columns = [mapc.get(c, c) for c in cmbc.columns]
cmbc.to_csv(f"{OUT}/cptac_module_by_cancer_k{K}.tsv", sep="\t")

tmbc = pd.read_csv(f"{OUT}/module_by_cancer_median_k{K}.tsv", sep="\t", index_col=0)
common = [c for c in cmbc.columns if c in tmbc.columns]
summ = pd.read_csv(f"{OUT}/module_summary_k{K}.tsv", sep="\t").set_index("module")

rows = []
for m in mods:
    pref = summ.loc[m, "pref_cancer"]
    if pref in common:
        cvals = cmbc.loc[m, common]
        rows.append({"module": m, "pref_cancer": pref,
                     "tcga_z": round(float(summ.loc[m, "pref_z"]), 2),
                     "cptac_z": round(float(cmbc.loc[m, pref]), 2),
                     "cptac_rank": int(cvals.rank(ascending=False)[pref]),
                     "n_common": len(common), "top_hallmark": summ.loc[m, "top_hallmark"]})
val = pd.DataFrame(rows)
val.to_csv(f"{OUT}/cptac_validation_k{K}.tsv", sep="\t", index=False)
print("common cancers (%d):" % len(common), " ".join(common), flush=True)
print("=== TCGA modules whose pref-cancer is among CPTAC 12 ===", flush=True)
print(val.to_string(index=False), flush=True)
print(f"\nrank-1 reproduction: {(val.cptac_rank==1).sum()}/{len(val)}   "
      f"rank<=3: {(val.cptac_rank<=3).sum()}/{len(val)}", flush=True)
tv = tmbc.loc[mods, common].values.flatten()
cv = cmbc.loc[mods, common].values.flatten()
r, p = spearmanr(tv, cv)
print(f"module x cancer  TCGA-pred vs CPTAC-measured  Spearman r={r:.3f}  p={p:.2e}", flush=True)
