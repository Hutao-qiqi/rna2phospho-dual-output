from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon


BASE = Path("/data/lsy/Infinite_Stream")
PKG = BASE / "SCP682_PORTABLE"
OUT = BASE / "02_results/model_validation/20260520_scp682_reviewer_head_to_head"
TABLES = OUT / "tables"
LOGS = OUT / "logs"
TABLES.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str = "spearman") -> tuple[float, int]:
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


def per_site_table(y: pd.DataFrame, pred: pd.DataFrame, model: str) -> pd.DataFrame:
    samples = y.index.intersection(pred.index)
    targets = y.columns.intersection(pred.columns)
    y2 = y.loc[samples, targets]
    p2 = pred.loc[samples, targets]
    rows = []
    for target in targets:
        yt = y2[target].to_numpy(dtype=float, copy=False)
        yp = p2[target].to_numpy(dtype=float, copy=False)
        sp, n = safe_corr(yt, yp, "spearman")
        pr, _ = safe_corr(yt, yp, "pearson")
        rows.append((model, target, n, sp, pr))
    return pd.DataFrame(rows, columns=["model", "target", "n", "spearman", "pearson"])


def summarize(per_site: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, df in per_site.groupby("model", sort=False):
        s = df["spearman"].dropna()
        p = df["pearson"].dropna()
        rows.append(
            {
                "model": model,
                "targets_tested": int(s.shape[0]),
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


def paired_stats(per_site: pd.DataFrame, reference: str, n_boot: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(68222)
    wide = per_site.pivot(index="target", columns="model", values="spearman")
    rows = []
    ref = wide[reference]
    for model in wide.columns:
        if model == reference:
            continue
        pair = pd.concat([ref, wide[model]], axis=1, keys=[reference, model]).dropna()
        if pair.empty:
            continue
        diff = pair[reference].to_numpy() - pair[model].to_numpy()
        boot = np.empty(n_boot)
        n = diff.shape[0]
        for i in range(n_boot):
            boot[i] = np.median(rng.choice(diff, size=n, replace=True))
        try:
            p = float(wilcoxon(pair[reference], pair[model], zero_method="wilcox", alternative="greater").pvalue)
        except Exception:
            p = np.nan
        rows.append(
            {
                "reference": reference,
                "baseline": model,
                "n_targets": int(n),
                "median_delta": float(np.median(diff)),
                "mean_delta": float(np.mean(diff)),
                "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                "wilcoxon_greater_p": p,
                "reference_win_targets": int((diff > 0).sum()),
                "baseline_win_targets": int((diff < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


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
    keep = counts[counts >= 5].index
    return pri[pri["kinase_gene"].isin(keep)].copy()


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


def main() -> None:
    y = pd.read_parquet(PKG / "training_set/observed_phosphosite.parquet")
    parent = pd.read_parquet(PKG / "training_set/oof_candidate_parent_only_phosphosite.parquet")
    ridge = pd.read_parquet(PKG / "training_set/oof_candidate_ridge_direct_phosphosite.parquet")
    rna = pd.read_parquet(PKG / "training_set/oof_candidate_rna_direct_phosphosite.parquet")
    scp = pd.read_parquet(PKG / "predictions/scp682_main_oof_phosphosite.parquet")

    common_samples = y.index.intersection(scp.index)
    common_targets = y.columns.intersection(scp.columns)
    y = y.loc[common_samples, common_targets]
    preds = {
        "RNA-direct": rna.loc[common_samples, common_targets],
        "Ridge-direct": ridge.loc[common_samples, common_targets],
        "Parent-only": parent.loc[common_samples, common_targets],
        "Ordinary-ML mean": (
            rna.loc[common_samples, common_targets]
            + ridge.loc[common_samples, common_targets]
            + parent.loc[common_samples, common_targets]
        )
        / 3.0,
        "SCP682": scp.loc[common_samples, common_targets],
    }

    per_site = pd.concat([per_site_table(y, pred, model) for model, pred in preds.items()], ignore_index=True)
    per_site.to_csv(TABLES / "scp682_reviewer_head_to_head_per_site.tsv", sep="\t", index=False)
    summary = summarize(per_site)
    summary.to_csv(TABLES / "scp682_reviewer_head_to_head_summary.tsv", sep="\t", index=False)
    paired = paired_stats(per_site, "SCP682")
    paired.to_csv(TABLES / "scp682_reviewer_head_to_head_paired_stats.tsv", sep="\t", index=False)

    edges = site_to_kinase_edges(list(common_targets))
    edges.to_csv(TABLES / "scp682_reviewer_kinase_site_edges_used.tsv", sep="\t", index=False)
    kin_true = aggregate_kinase_activity(y, edges)
    kin_rows = []
    for model, pred in preds.items():
        kin_pred = aggregate_kinase_activity(pred, edges)
        kin_pred = kin_pred.loc[kin_true.index, kin_true.columns.intersection(kin_pred.columns)]
        kt = kin_true.loc[:, kin_pred.columns]
        for kinase in kin_pred.columns:
            sp, n = safe_corr(kt[kinase].to_numpy(dtype=float), kin_pred[kinase].to_numpy(dtype=float), "spearman")
            pr, _ = safe_corr(kt[kinase].to_numpy(dtype=float), kin_pred[kinase].to_numpy(dtype=float), "pearson")
            kin_rows.append((model, kinase, n, sp, pr))
    kin = pd.DataFrame(kin_rows, columns=["model", "kinase", "n", "spearman", "pearson"])
    kin.to_csv(TABLES / "scp682_reviewer_kinase_activity_per_kinase.tsv", sep="\t", index=False)
    kin_summary = summarize(kin.rename(columns={"kinase": "target"}))
    kin_summary.to_csv(TABLES / "scp682_reviewer_kinase_activity_summary.tsv", sep="\t", index=False)
    kin_paired = paired_stats(kin.rename(columns={"kinase": "target"}), "SCP682")
    kin_paired.to_csv(TABLES / "scp682_reviewer_kinase_activity_paired_stats.tsv", sep="\t", index=False)

    meta = {
        "common_samples": int(len(common_samples)),
        "common_targets": int(len(common_targets)),
        "kinase_edges": int(edges.shape[0]),
        "kinases_with_at_least_5_sites": int(edges["kinase_gene"].nunique()),
        "output_dir": str(OUT),
    }
    (LOGS / "run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("\nHEAD_TO_HEAD")
    print(summary.to_string(index=False))
    print("\nPAIRED")
    print(paired.to_string(index=False))
    print("\nKINASE")
    print(kin_summary.to_string(index=False))


if __name__ == "__main__":
    main()

