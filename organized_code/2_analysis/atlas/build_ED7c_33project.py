#!/usr/bin/env python
"""ED Fig 7c — 33-project NMF including LAML (acute myeloid leukaemia).
Adds the 151 LAML primary-blood samples to the 10,023 solid primary tumours and
re-runs the same signed-split NMF (k=30); checks whether LAML forms its own
haematopoietic module separate from the 32 solid-tumour modules."""
import pandas as pd, numpy as np, time
from sklearn.decomposition import NMF

BASE = "02_results/model_prediction/20260529_tcga_full_scp682_main_reprediction_v1"
OUT  = "02_results/model_validation/20260612_tcga_pancancer_nmf_v1/results"
t0 = time.time()

mani = pd.read_csv(BASE + "/tables/tcga_scp682_prediction_sample_manifest.tsv", sep="\t")
keep = mani[mani.sample_type.isin(["Primary Tumor",
                                   "Primary Blood Derived Cancer - Peripheral Blood"])]
keep_ids = set(keep.sample_id)
pred = pd.read_parquet(BASE + "/predictions/tcga_full_scp682_predicted_phosphosite.parquet")
if "sample_id" in pred.columns: pred = pred.set_index("sample_id")
pred.index.name = "sample_id"
pred = pred[pred.index.isin(keep_ids)]
meta = mani.set_index("sample_id").loc[pred.index]
print("33-proj samples:", pred.shape, " cancers:", meta.cancer.nunique(),
      " LAML n =", int((meta.cancer == "LAML").sum()), flush=True)

X = pred.values.astype("float32")
mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
Z = (X - mu) / sd; np.clip(Z, -8, 8, out=Z); del X
Xc = np.ascontiguousarray(np.hstack([np.clip(Z, 0, None), np.clip(-Z, 0, None)]).astype("float32")); del Z
print("NMF input:", Xc.shape, flush=True)
m = NMF(n_components=30, init="nndsvd", solver="cd", random_state=0, max_iter=400)
W = m.fit_transform(Xc); H = m.components_
print("NMF done t=%.0fs" % (time.time() - t0), flush=True)

Wz = (W - W.mean(0)) / (W.std(0) + 1e-9)
mods = [f"M{i+1}" for i in range(30)]
df = pd.DataFrame(Wz, columns=mods); df["cancer"] = meta.cancer.values
mbc = df.groupby("cancer")[mods].median().T
mbc.to_csv(OUT + "/module_by_cancer_median_k30_33proj.tsv", sep="\t")
pref = mbc.idxmax(1); prefz = mbc.max(1)
summ = pd.DataFrame({"module": mods, "pref_cancer": pref.values, "pref_z": prefz.values.round(2)})
summ = summ.sort_values(["pref_cancer", "pref_z"], ascending=[True, False])
summ.to_csv(OUT + "/module_summary_k30_33proj.tsv", sep="\t", index=False)
laml = summ[summ.pref_cancer == "LAML"]
print("LAML-preferred modules:", laml.module.tolist(), " z:", laml.pref_z.tolist(), flush=True)
print("ALL DONE t=%.0fs" % (time.time() - t0), flush=True)
