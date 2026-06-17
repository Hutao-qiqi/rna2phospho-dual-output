#!/usr/bin/env python
"""ED Fig 7b — CPTAC projection control. For each module whose preferred cancer
is among the 12 CPTAC subtypes, the top-50 module sites' enrichment in the
matching cancer is computed in BOTH the CPTAC SCP682-predicted phosphoproteome
and the CPTAC measured phosphoproteome. Predicted reproduces (~0.4), measured
does not (~0), locating the atlas as an RNA-inferred construct."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"

OUT  = "02_results/model_validation/20260612_tcga_pancancer_nmf_v1/results"
OBS  = "SCP682_MAIN/training_set/observed_phosphosite.parquet"
PRED = "SCP682_MAIN/predictions/scp682_main_oof_phosphosite.parquet"
SC   = "SCP682_MAIN/attention_export/nmf_sample_cancer.tsv"

sites = pd.read_csv(OUT + "/site_names.tsv", sep="\t")["site"].tolist(); nS = len(sites)
H = np.load(OUT + "/H_k30.npy"); Hpos = H[:, :nS]
summ = pd.read_csv(OUT + "/module_summary_k30.tsv", sep="\t").set_index("module")

obs = pd.read_parquet(OBS); obs.index.name = "sample"
sc = pd.read_csv(SC, sep="\t").set_index("sample")
obs = obs.loc[obs.index.isin(sc.index)]; cancer = sc.loc[obs.index, "cancer"].values
Zo = (obs - obs.mean()) / obs.std()
pred = pd.read_parquet(PRED)
if "sample_id" in pred.columns: pred = pred.set_index("sample_id")
pred.index.name = "sample"; pred = pred.reindex(index=obs.index, columns=sites)
Zp = (pred - pred.mean()) / pred.std()

m2c = {"LUSC":"LSCC","KIRC":"CCRCC","HNSC":"HNSCC","PAAD":"PDA","KIRP":"ccPRCC"}
cset = set(cancer); rows = []
for i in range(30):
    m = f"M{i+1}"; pt = summ.loc[m, "pref_cancer"]; pc = m2c.get(pt, pt)
    if pc not in cset: continue
    top = np.argsort(Hpos[i])[-50:]
    sn = [sites[j] for j in top if sites[j] in Zo.columns]
    inm = cancer == pc
    so = Zo[sn].mean(axis=1).values; sp = Zp[sn].mean(axis=1).values
    rows.append({"module": m, "cancer": pt,
                 "pred_delta": round(float(np.nanmedian(sp[inm]) - np.nanmedian(sp[~inm])), 3),
                 "meas_delta": round(float(np.nanmedian(so[inm]) - np.nanmedian(so[~inm])), 3)})
r = pd.DataFrame(rows)
r.to_csv(OUT + "/ED7b_cptac_projection.tsv", sep="\t", index=False)
print(r.to_string(index=False))
print("median pred=%.3f  meas=%.3f" % (r.pred_delta.median(), r.meas_delta.median()))

# grouped horizontal bar
y = np.arange(len(r))
fig, ax = plt.subplots(figsize=(120/25.4, 70/25.4))
ax.barh(y - 0.2, r.pred_delta, 0.38, color="#ED8D5A", edgecolor="black", linewidth=0.4, label="CPTAC predicted (SCP682)")
ax.barh(y + 0.2, r.meas_delta, 0.38, color="#9FB6C9", edgecolor="black", linewidth=0.4, label="CPTAC measured (MS)")
ax.axvline(0, color="#888888", linewidth=0.5)
ax.set_yticks(y); ax.set_yticklabels([f"{m}·{c}" for m, c in zip(r.module, r.cancer)], fontsize=6)
ax.set_xlabel("Top-site enrichment in matching cancer (Δ median z)", fontsize=7)
ax.set_title("CPTAC projection: modules reproduce in predicted, not measured phosphoproteome", fontsize=7, fontweight="bold")
ax.tick_params(axis="x", labelsize=6); ax.invert_yaxis()
ax.legend(fontsize=6, loc="lower right", frameon=False, bbox_to_anchor=(1.0, 0.0))
for s in ["top", "right"]: ax.spines[s].set_visible(False)
med_p, med_m = r.pred_delta.median(), r.meas_delta.median()
ax.text(0.50, 0.97, f"median Δ:  predicted {med_p:.2f}   vs   measured {med_m:.2f}",
        transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color="#555555")
plt.tight_layout()
plt.savefig(OUT.replace("/results", "/figures") + "/ED7b_cptac_projection.pdf")
print("wrote ED7b_cptac_projection.pdf")
