from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


BASE = Path("/data/lsy/Infinite_Stream")
PKG = BASE / "SCP682_PORTABLE"
OUT = BASE / "02_results/model_validation/20260520_scp682_reviewer_kinpred_rna_style"
TABLES = OUT / "tables"
LOGS = OUT / "logs"
TABLES.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> tuple[float, int]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(mask.sum())
    if n < 8:
        return np.nan, n
    yt = y_true[mask]
    yp = y_pred[mask]
    if np.nanstd(yt) == 0 or np.nanstd(yp) == 0:
        return np.nan, n
    if method == "spearman":
        return float(spearmanr(yt, yp).statistic), n
    return float(pearsonr(yt, yp).statistic), n


def site_to_kinase_edges(targets: list[str]) -> pd.DataFrame:
    pri = pd.read_csv(BASE / "01_data/pathway_prior/processed/kinase_substrate_prior_for_modeling_v1.tsv", sep="\t")
    pri = pri[pri["has_site"].astype(str).str.lower().eq("true")].copy()
    pri["target"] = pri["substrate_gene"].astype(str) + "|" + pri["substrate_site"].astype(str)
    pri = pri[pri["target"].isin(targets)]
    pri = pri[["kinase_gene", "target", "weight", "source", "edge_level"]].dropna()

    cop = BASE / "01_data/pathway_prior/processed/copheemap_v1/copheeksa_model_phosphosite_kinase_predictions.tsv"
    if cop.exists():
        c = pd.read_csv(cop, sep="\t")
        c = c[c["gene_site_id"].isin(targets)].copy()
        c = c.rename(columns={"kinase": "kinase_gene", "gene_site_id": "target", "copheeksa_score": "weight"})
        c["source"] = "CoPheeKSA"
        c["edge_level"] = "predicted"
        c = c[["kinase_gene", "target", "weight", "source", "edge_level"]]
        pri = pd.concat([pri, c], ignore_index=True)

    pri["weight"] = pd.to_numeric(pri["weight"], errors="coerce").fillna(0.3)
    pri = pri.drop_duplicates(["kinase_gene", "target"])
    counts = pri.groupby("kinase_gene")["target"].nunique()
    return pri[pri["kinase_gene"].isin(counts[counts >= 5].index)].copy()


def aggregate_kinase_activity(mat: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for kinase, e in edges.groupby("kinase_gene"):
        cols = [c for c in e["target"].tolist() if c in mat.columns]
        if len(cols) < 5:
            continue
        w = e.set_index("target").loc[cols, "weight"].astype(float).to_numpy()
        w = np.maximum(w, 0.05)
        x = mat[cols].to_numpy(dtype=float)
        valid = np.isfinite(x)
        wx = np.where(valid, x * w[None, :], 0.0)
        denom = np.where(valid, w[None, :], 0.0).sum(axis=1)
        vals = np.divide(wx.sum(axis=1), denom, out=np.full(x.shape[0], np.nan), where=denom > 0)
        out[kinase] = vals
    return pd.DataFrame(out, index=mat.index)


def per_kinase(y: pd.DataFrame, pred: pd.DataFrame, model: str) -> pd.DataFrame:
    cols = y.columns.intersection(pred.columns)
    rows = []
    for kinase in cols:
        sp, n = safe_corr(y[kinase].to_numpy(dtype=float), pred[kinase].to_numpy(dtype=float), "spearman")
        pr, _ = safe_corr(y[kinase].to_numpy(dtype=float), pred[kinase].to_numpy(dtype=float), "pearson")
        rows.append((model, kinase, n, sp, pr))
    return pd.DataFrame(rows, columns=["model", "kinase", "n", "spearman", "pearson"])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, x in df.groupby("model", sort=False):
        s = x["spearman"].dropna()
        p = x["pearson"].dropna()
        rows.append(
            {
                "model": model,
                "kinases_tested": int(s.shape[0]),
                "median_spearman": float(s.median()),
                "mean_spearman": float(s.mean()),
                "median_pearson": float(p.median()),
                "mean_pearson": float(p.mean()),
                "ge_0_2": int((s >= 0.2).sum()),
                "ge_0_3": int((s >= 0.3).sum()),
                "ge_0_5": int((s >= 0.5).sum()),
            }
        )
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame, ref: str = "SCP682-derived kinase activity") -> pd.DataFrame:
    wide = df.pivot(index="kinase", columns="model", values="spearman")
    rows = []
    for model in wide.columns:
        if model == ref:
            continue
        pair = wide[[ref, model]].dropna()
        diff = pair[ref] - pair[model]
        try:
            p = float(wilcoxon(pair[ref], pair[model], alternative="greater").pvalue)
        except Exception:
            p = np.nan
        rows.append(
            {
                "reference": ref,
                "baseline": model,
                "n_kinases": int(pair.shape[0]),
                "median_delta": float(diff.median()),
                "mean_delta": float(diff.mean()),
                "wilcoxon_greater_p": p,
                "reference_win_kinases": int((diff > 0).sum()),
                "baseline_win_kinases": int((diff < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    y_site = pd.read_parquet(PKG / "training_set/observed_phosphosite.parquet")
    scp_site = pd.read_parquet(PKG / "predictions/scp682_main_oof_phosphosite.parquet")
    sample_manifest = pd.read_csv(PKG / "training_set/sample_manifest.tsv", sep="\t").set_index("index")
    rna = pd.read_parquet(BASE / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2/rna_log2_tpm_paired.parquet")

    samples = y_site.index.intersection(scp_site.index).intersection(rna.index).intersection(sample_manifest.index)
    targets = y_site.columns.intersection(scp_site.columns)
    y_site = y_site.loc[samples, targets]
    scp_site = scp_site.loc[samples, targets]
    rna = rna.loc[samples]
    cancer = sample_manifest.loc[samples, "cancer_label"].astype(str).to_numpy()

    edges = site_to_kinase_edges(list(targets))
    edges.to_csv(TABLES / "kinase_site_edges_used.tsv", sep="\t", index=False)
    y_kin = aggregate_kinase_activity(y_site, edges)
    scp_kin = aggregate_kinase_activity(scp_site, edges).loc[y_kin.index, y_kin.columns]

    gene_var = rna.var(axis=0).sort_values(ascending=False)
    selected_genes = list(gene_var.head(min(5000, len(gene_var))).index)
    x = rna[selected_genes].to_numpy(dtype=np.float32)
    y = y_kin.to_numpy(dtype=np.float32)
    oof = np.full_like(y, np.nan, dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260520)
    for fold, (tr, va) in enumerate(skf.split(np.arange(len(samples)), cancer), start=1):
        scaler = StandardScaler()
        xtr = scaler.fit_transform(x[tr])
        xva = scaler.transform(x[va])
        ytr = y[tr].copy()
        means = np.nanmean(ytr, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        ytr = np.where(np.isfinite(ytr), ytr, means[None, :])
        model = Ridge(alpha=20.0, fit_intercept=True, random_state=fold)
        model.fit(xtr, ytr)
        oof[va] = model.predict(xva).astype(np.float32)
        print(f"fold {fold} train={len(tr)} val={len(va)}", flush=True)

    kinpred = pd.DataFrame(oof, index=y_kin.index, columns=y_kin.columns)
    kinpred.to_parquet(TABLES / "kinpred_rna_style_oof_kinase_activity.parquet")
    rows = pd.concat(
        [
            per_kinase(y_kin, kinpred, "KinPred-RNA-style ridge"),
            per_kinase(y_kin, scp_kin, "SCP682-derived kinase activity"),
        ],
        ignore_index=True,
    )
    rows.to_csv(TABLES / "kinpred_rna_style_per_kinase.tsv", sep="\t", index=False)
    summary = summarize(rows)
    summary.to_csv(TABLES / "kinpred_rna_style_summary.tsv", sep="\t", index=False)
    ps = paired(rows)
    ps.to_csv(TABLES / "kinpred_rna_style_paired_stats.tsv", sep="\t", index=False)
    (LOGS / "metadata.json").write_text(
        pd.Series(
            {
                "samples": len(samples),
                "sites": len(targets),
                "kinases": y_kin.shape[1],
                "selected_rna_genes": len(selected_genes),
            }
        ).to_json(indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(ps.to_string(index=False))
    (OUT / "done.txt").write_text("done\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

