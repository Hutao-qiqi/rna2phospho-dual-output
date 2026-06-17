#!/usr/bin/env python
"""TCGA pan-cancer signed-split NMF over SCP682 predicted phosphoproteome.
Input X_nmf.npy = [Z+, Z-] (10023 primary tumors x 37184 signed features).
Runs k = 30 (main), 20, 40 (stability). Saves W (sample x module) and H (module x feature)."""
import numpy as np, time, json, os
from sklearn.decomposition import NMF

OUT = "E:/data/gongke/TCGA-TCPA/02_results/model_validation/20260612_tcga_pancancer_nmf_v1/results"
X = np.load(OUT + "/X_nmf.npy")
print(f"loaded X: {X.shape}  {X.nbytes/1e9:.2f}GB", flush=True)

summary = {}
for k in [30, 20, 40]:
    t0 = time.time()
    m = NMF(n_components=k, init="nndsvd", solver="cd",
            random_state=0, max_iter=400, tol=1e-4)
    W = m.fit_transform(X)        # samples x k
    H = m.components_             # k x features
    np.save(f"{OUT}/W_k{k}.npy", W.astype("float32"))
    np.save(f"{OUT}/H_k{k}.npy", H.astype("float32"))
    dt = time.time() - t0
    summary[f"k{k}"] = {"recon_err": float(m.reconstruction_err_),
                         "n_iter": int(m.n_iter_), "sec": round(dt, 1)}
    print(f"k={k} done  err={m.reconstruction_err_:.1f}  iter={m.n_iter_}  t={dt:.0f}s", flush=True)
    json.dump(summary, open(f"{OUT}/nmf_summary.json", "w"), indent=2)

print("ALL DONE", flush=True)
