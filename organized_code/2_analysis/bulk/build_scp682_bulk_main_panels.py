#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MAIN_RELEASE = Path("/data/lsy/Infinite_Stream/SCP682/frozen_release/SCP682_main_exact_scnet_gnn_20260522")
RNA_PATH = Path("/data/lsy/Infinite_Stream/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/rna_log2_tpm_paired.parquet")
PHOSPHO_LOGRATIO_PATH = Path("/data/lsy/Infinite_Stream/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/phosphosite_gene_site_logratio_min20pct_targets.parquet")
OUT_DIR = Path("/data/lsy/Infinite_Stream/SCP682-main/results/20260526_bulk_main_panels")

PALETTE = {
    "lung": "#92B1D9",
    "breast": "#C1D8E9",
    "brain": "#DBDDEF",
    "gynecologic": "#F6C8B6",
    "gi_kidney_hn": "#9A9A9A",
}

COMPLEXES = {
    "CDK-cyclin": ["CDK1", "CDK2", "CDK4", "CDK6", "CCNA2", "CCNB1", "CCNB2", "CCND1", "CCNE1", "RB1", "E2F1", "WEE1", "CDC25A", "CDC25B", "CDC25C"],
    "MAPK": ["MAPK1", "MAPK3", "MAPK8", "MAPK9", "MAPK14", "MAP2K1", "MAP2K2", "RAF1", "BRAF", "KRAS", "NRAS", "DUSP6", "ELK1"],
    "mTORC1": ["MTOR", "RPTOR", "MLST8", "AKT1", "AKT2", "RPS6KB1", "EIF4EBP1", "RPS6", "TSC1", "TSC2"],
    "EGFR-ERBB": ["EGFR", "ERBB2", "ERBB3", "ERBB4", "GRB2", "SHC1", "SOS1", "CBL", "PIK3CA", "PLCG1"],
    "DNA damage": ["ATM", "ATR", "CHEK1", "CHEK2", "TP53", "BRCA1", "BRCA2", "H2AFX", "MDC1", "RAD51"],
    "NF-kB": ["NFKB1", "NFKB2", "RELA", "RELB", "IKBKB", "IKBKG", "CHUK", "NFKBIA", "TNFAIP3"],
    "Hippo-YAP": ["YAP1", "WWTR1", "LATS1", "LATS2", "STK3", "STK4", "TEAD1", "TEAD2", "TEAD3", "TEAD4"],
    "WNT": ["CTNNB1", "GSK3B", "APC", "AXIN1", "AXIN2", "DVL1", "DVL2", "DVL3", "TCF7L2"],
}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ensure_dirs(out_dir):
    for sub in ["figures", "tables", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)


def savefig(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def make_cancer_groups(meta):
    mapping = {
        "LUAD": "lung",
        "LUAD_CONFIRM": "lung",
        "LSCC": "lung",
        "BRCA_TCGA": "breast",
        "BRCA_PROSPECTIVE": "breast",
        "GBM_DISCOVERY": "brain",
        "GBM_CONFIRMATORY": "brain",
        "UCEC": "gynecologic",
        "UCEC_CONFIRM": "gynecologic",
        "OV_PROSPECTIVE": "gynecologic",
        "OV_TCGA": "gynecologic",
        "CCRCC": "kidney",
        "NON_CCRCC": "kidney",
        "PDA": "pancreas_headneck",
        "HNSCC": "pancreas_headneck",
        "STAD": "gi_hepatobiliary",
        "COAD_PROSPECTIVE": "gi_hepatobiliary",
    }
    groups = meta["cancer_label"].astype(str).map(mapping).fillna("gi_kidney_hn")
    groups = groups.replace({
        "kidney": "gi_kidney_hn",
        "pancreas_headneck": "gi_kidney_hn",
        "gi_hepatobiliary": "gi_kidney_hn",
    })
    return groups


def load_release_tables(release):
    train = release / "training_set"
    obs = pd.read_parquet(train / "observed_phosphosite.parquet")
    pred = pd.read_parquet(release / "predictions/scp682_exact_scnet_gnn_oof_phosphosite_best.parquet")
    parent = pd.read_parquet(train / "oof_candidate_parent_only_phosphosite.parquet")
    ridge = pd.read_parquet(train / "oof_candidate_ridge_direct_phosphosite.parquet")
    rna_direct = pd.read_parquet(train / "oof_candidate_rna_direct_phosphosite.parquet")
    sample_meta = pd.read_csv(train / "sample_manifest.tsv", sep="\t")
    if "sample_id" not in sample_meta.columns and "index" in sample_meta.columns:
        sample_meta = sample_meta.rename(columns={"index": "sample_id"})
    sample_meta["sample_id"] = sample_meta["sample_id"].astype(str)
    sample_meta["cancer_group"] = make_cancer_groups(sample_meta)
    site_meta = pd.read_csv(train / "phosphosite_target_manifest.tsv", sep="\t")
    site_meta["scp682_site_id"] = site_meta["scp682_site_id"].astype(str)
    per_site = pd.read_csv(release / "performance/per_site_spearman.tsv", sep="\t")
    per_site["target"] = per_site["target"].astype(str)
    common_samples = obs.index.intersection(pred.index).intersection(parent.index).intersection(ridge.index).intersection(rna_direct.index)
    common_targets = obs.columns.intersection(pred.columns).intersection(parent.columns).intersection(ridge.columns).intersection(rna_direct.columns)
    obs = obs.loc[common_samples, common_targets].astype(np.float32)
    pred = pred.loc[common_samples, common_targets].astype(np.float32)
    parent = parent.loc[common_samples, common_targets].astype(np.float32)
    ridge = ridge.loc[common_samples, common_targets].astype(np.float32)
    rna_direct = rna_direct.loc[common_samples, common_targets].astype(np.float32)
    sample_meta = sample_meta.set_index("sample_id").reindex(common_samples).reset_index()
    if "sample_id" not in sample_meta.columns and "index" in sample_meta.columns:
        sample_meta = sample_meta.rename(columns={"index": "sample_id"})
    site_meta = site_meta[site_meta["scp682_site_id"].isin(common_targets)].copy()
    return obs, pred, parent, ridge, rna_direct, sample_meta, site_meta, per_site


def load_scp682_module(release):
    script = release / "scripts/train_scp682_exact_scnet_gnn_v1.py"
    spec = importlib.util.spec_from_file_location("scp682_exact_scnet_module", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["scp682_exact_scnet_module"] = module
    spec.loader.exec_module(module)
    return module


def checkpoint_forward(release, obs, parent, ridge, rna_direct, out_dir, force=False):
    latent_path = out_dir / "tables/scp682_sample_latent.tsv"
    comp_path = out_dir / "tables/scp682_component_contribution_by_site.tsv"
    if latent_path.exists() and comp_path.exists() and not force:
        return pd.read_csv(latent_path, sep="\t"), pd.read_csv(comp_path, sep="\t")

    import torch

    module = load_scp682_module(release)
    ckpt = torch.load(release / "models/scp682_exact_scnet_gnn_best.pt", map_location="cpu", weights_only=False)
    args = ckpt["args"]
    samples = [str(x) for x in ckpt["samples"]]
    targets = [str(x) for x in ckpt["targets"]]
    obs = obs.loc[samples, targets]
    baseline = ((parent.loc[samples, targets] + ridge.loc[samples, targets] + rna_direct.loc[samples, targets]) / 3.0).astype(np.float32)
    y = obs.to_numpy(np.float32)
    b = baseline.to_numpy(np.float32)
    mask = np.isfinite(y) & np.isfinite(b)
    b = np.nan_to_num(b, nan=0.0).astype(np.float32)
    feature = np.where(mask.T, b.T, 0.0).astype(np.float32)
    feature = ((feature - feature.mean(axis=1, keepdims=True)) / (feature.std(axis=1, keepdims=True) + 1e-5)).astype(np.float32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = module.SCP682ExactScNETResidual(
        n_sites=len(targets),
        n_samples=len(samples),
        shrinkage=args.get("shrinkage", 0.3),
        hidden=args.get("hidden", 160),
        latent=args.get("latent", 64),
        inter_dim=args.get("inter_dim", 192),
        embd_dim=args.get("embd_dim", 64),
        num_layers=args.get("num_layers", 2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    site_prior_np = np.asarray(ckpt["site_prior"], dtype=np.float32)
    sample_edge = ckpt["sample_edge_index"]
    if hasattr(sample_edge, "numpy"):
        sample_edge = sample_edge.numpy()

    with torch.no_grad():
        feature_x = torch.as_tensor(feature, dtype=torch.float32, device=device)
        row_edge_index = torch.as_tensor(ckpt["site_edge_index"], dtype=torch.long, device=device)
        col_edge_index = torch.as_tensor(sample_edge, dtype=torch.long, device=device)
        row_embed, col_embed, _ = model.graph_core(feature_x, col_edge_index, row_edge_index, collect_attention=False)
        sample_latent = model.sample_proj(col_embed).detach().cpu().numpy()

        n_sites = len(targets)
        abs_b = np.zeros(n_sites, dtype=np.float64)
        abs_g = np.zeros(n_sites, dtype=np.float64)
        abs_graph = np.zeros(n_sites, dtype=np.float64)
        abs_resid = np.zeros(n_sites, dtype=np.float64)
        attention_sum = np.zeros(n_sites, dtype=np.float64)
        count = np.zeros(n_sites, dtype=np.float64)
        shrinkage = float(args.get("shrinkage", 0.3))
        site_prior = torch.as_tensor(site_prior_np, dtype=torch.float32, device=device)
        batch_size = int(args.get("batch_size", 4))
        for start in range(0, len(samples), batch_size):
            end = min(len(samples), start + batch_size)
            idx = torch.arange(start, end, dtype=torch.long, device=device)
            baseline_b = torch.as_tensor(b[start:end], dtype=torch.float32, device=device)
            mask_b = torch.as_tensor(mask[start:end], dtype=torch.bool, device=device)
            _, delta_b, graph_b, resid_b, attention_b = model.decode(
                row_embed,
                col_embed,
                baseline_b,
                mask_b,
                site_prior,
                sample_idx=idx,
            )
            m = mask[start:end]
            abs_b += (np.abs(b[start:end]) * m).sum(axis=0)
            abs_g += (np.abs(delta_b.detach().cpu().numpy() * shrinkage) * m).sum(axis=0)
            abs_graph += (np.abs(graph_b.detach().cpu().numpy() * shrinkage) * m).sum(axis=0)
            abs_resid += (np.abs(resid_b.detach().cpu().numpy() * shrinkage) * m).sum(axis=0)
            attention_sum += (attention_b.detach().cpu().numpy() * m).sum(axis=0)
            count += m.sum(axis=0)

    count_safe = np.maximum(count, 1.0)
    comp = pd.DataFrame({
        "target": targets,
        "mean_abs_B_phi": abs_b / count_safe,
        "mean_abs_G_theta": abs_g / count_safe,
        "mean_abs_graph_delta": abs_graph / count_safe,
        "mean_abs_residual_delta": abs_resid / count_safe,
        "mean_attention": attention_sum / count_safe,
        "n_observed": count.astype(int),
    })
    comp["G_theta_fraction"] = comp["mean_abs_G_theta"] / (comp["mean_abs_B_phi"] + comp["mean_abs_G_theta"] + 1e-12)
    comp["graph_delta_fraction_within_G"] = comp["mean_abs_graph_delta"] / (comp["mean_abs_graph_delta"] + comp["mean_abs_residual_delta"] + 1e-12)
    comp.to_csv(comp_path, sep="\t", index=False)

    latent_cols = [f"latent_{i:02d}" for i in range(sample_latent.shape[1])]
    latent = pd.DataFrame(sample_latent, columns=latent_cols)
    latent.insert(0, "sample_id", samples)
    latent.to_csv(latent_path, sep="\t", index=False)
    return latent, comp


def panel_a_sample_umap(latent, sample_meta, out_dir):
    import umap
    from sklearn.preprocessing import StandardScaler

    sample_meta = sample_meta.copy()
    sample_meta["sample_id"] = sample_meta["sample_id"].astype(str)
    df = latent.merge(sample_meta[["sample_id", "cancer_label", "cancer_group"]], on="sample_id", how="left")
    latent_cols = [c for c in df.columns if c.startswith("latent_")]
    x = StandardScaler().fit_transform(df[latent_cols].to_numpy(np.float32))
    emb = umap.UMAP(n_neighbors=30, min_dist=0.22, metric="cosine", random_state=42).fit_transform(x)
    df["UMAP1"] = emb[:, 0]
    df["UMAP2"] = emb[:, 1]
    df.to_csv(out_dir / "tables/A_sample_umap_coordinates.tsv", sep="\t", index=False)

    order = ["lung", "breast", "brain", "gynecologic", "gi_kidney_hn"]
    plt.figure(figsize=(6.2, 5.2))
    for group in order:
        sub = df[df["cancer_group"] == group]
        if sub.empty:
            continue
        plt.scatter(sub["UMAP1"], sub["UMAP2"], s=16, c=PALETTE[group], label=group, linewidths=0, alpha=0.88)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("SCP682 sample latent")
    plt.legend(frameon=False, markerscale=1.8, fontsize=8, loc="best")
    sns.despine()
    savefig(out_dir / "figures/A_sample_latent_umap")


def panel_b_hex(obs, pred, out_dir):
    y = obs.to_numpy(np.float32)
    p = pred.to_numpy(np.float32)
    m = np.isfinite(y) & np.isfinite(p)
    x = y[m]
    z = p[m]
    lo = float(np.nanpercentile(np.concatenate([x, z]), 0.2))
    hi = float(np.nanpercentile(np.concatenate([x, z]), 99.8))
    plt.figure(figsize=(5.8, 5.4))
    hb = plt.hexbin(x, z, gridsize=170, bins="log", mincnt=1, cmap="viridis", extent=[lo, hi, lo, hi])
    plt.plot([lo, hi], [lo, hi], color="#2B2B2B", linewidth=1.1, alpha=0.8)
    cb = plt.colorbar(hb)
    cb.set_label("log10 bin count")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Observed phosphosite target")
    plt.ylabel("SCP682 predicted")
    plt.title("Predicted vs observed")
    plt.gca().set_aspect("equal", adjustable="box")
    sns.despine()
    savefig(out_dir / "figures/B_predicted_vs_observed_hex")
    pd.DataFrame({"n_pairs": [int(m.sum())], "x_min": [lo], "x_max": [hi]}).to_csv(out_dir / "tables/B_hex_summary.tsv", sep="\t", index=False)

    obs_z = obs.copy()
    pred_z = pred.copy()
    obs_z = (obs_z - obs_z.mean(axis=0, skipna=True)) / obs_z.std(axis=0, skipna=True).replace(0, np.nan)
    pred_z = (pred_z - pred_z.mean(axis=0, skipna=True)) / pred_z.std(axis=0, skipna=True).replace(0, np.nan)
    yz = obs_z.to_numpy(np.float32)
    pz = pred_z.to_numpy(np.float32)
    mz = np.isfinite(yz) & np.isfinite(pz)
    xz = yz[mz]
    zz = pz[mz]
    lo_z = float(np.nanpercentile(np.concatenate([xz, zz]), 0.2))
    hi_z = float(np.nanpercentile(np.concatenate([xz, zz]), 99.8))
    lim = max(abs(lo_z), abs(hi_z))
    plt.figure(figsize=(5.8, 5.4))
    hb = plt.hexbin(xz, zz, gridsize=170, bins="log", mincnt=1, cmap="viridis", extent=[-lim, lim, -lim, lim])
    plt.plot([-lim, lim], [-lim, lim], color="#2B2B2B", linewidth=1.1, alpha=0.8)
    cb = plt.colorbar(hb)
    cb.set_label("log10 bin count")
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel("Observed within-site z")
    plt.ylabel("Predicted within-site z")
    plt.title("Predicted vs observed, within-site z")
    plt.gca().set_aspect("equal", adjustable="box")
    sns.despine()
    savefig(out_dir / "figures/B_predicted_vs_observed_within_site_z_hex")


def site_class_tables(release, targets):
    priors = release / "priors"
    targets = pd.Index(targets.astype(str) if hasattr(targets, "astype") else [str(x) for x in targets])
    ksa_sites = set()
    cophee_sites = set()
    kinase_count = pd.Series(0.0, index=targets, dtype=np.float64)

    map_path = priors / "copheemap_site_id_to_model_gene_site.tsv"
    id_map = {}
    if map_path.exists():
        m = pd.read_csv(map_path, sep="\t")
        id_map = dict(zip(m["cophee_site_id"].astype(str), m["gene_site_id"].astype(str)))

    ksa_path = priors / "positive_KSA.csv"
    if ksa_path.exists() and id_map:
        ksa = pd.read_csv(ksa_path)
        site_col = "sites" if "sites" in ksa.columns else ksa.columns[1]
        kinase_col = "kinases" if "kinases" in ksa.columns else ("kinase" if "kinase" in ksa.columns else ksa.columns[-1])
        ksa["target"] = ksa[site_col].astype(str).map(id_map)
        ksa = ksa.dropna(subset=["target"])
        ksa_sites = set(ksa.loc[ksa["target"].isin(targets), "target"].astype(str))
        kc = ksa[ksa["target"].isin(targets)].groupby("target")[kinase_col].nunique()
        kinase_count.loc[kc.index.astype(str)] = np.maximum(kinase_count.loc[kc.index.astype(str)].to_numpy(), kc.to_numpy(dtype=float))

    cophee_path = priors / "Table_S2_CoPheeMap.tsv.zip"
    if cophee_path.exists() and id_map:
        cm = pd.read_csv(cophee_path, sep="\t", usecols=["site1", "site2"])
        a = cm["site1"].astype(str).map(id_map)
        b = cm["site2"].astype(str).map(id_map)
        cophee_sites = set(pd.concat([a, b]).dropna().astype(str))
        cophee_sites = {x for x in cophee_sites if x in set(targets)}

    kstar_path = priors / "kstar_default_network_edges_long.tsv"
    if kstar_path.exists():
        try:
            ks = pd.read_csv(kstar_path, sep="\t", usecols=["kinase", "substrate_gene", "site", "model_site_id"])
        except Exception:
            ks = pd.read_csv(kstar_path, sep="\t", usecols=["kinase", "substrate_gene", "site"])
            ks["model_site_id"] = ks["substrate_gene"].fillna("").astype(str) + "|" + ks["site"].fillna("").astype(str)
        ks["target"] = ks["model_site_id"].astype(str)
        kc = ks[ks["target"].isin(targets)].groupby("target")["kinase"].nunique()
        kinase_count.loc[kc.index.astype(str)] = np.maximum(kinase_count.loc[kc.index.astype(str)].to_numpy(), kc.to_numpy(dtype=float))

    cls = pd.DataFrame({"target": targets})
    cls["site_class"] = "orphan"
    cls.loc[cls["target"].isin(cophee_sites), "site_class"] = "CoPheeMap"
    cls.loc[cls["target"].isin(ksa_sites), "site_class"] = "KSA"
    cls["kinase_count"] = cls["target"].map(kinase_count.to_dict()).fillna(0).astype(int)
    return cls


def build_site_summary(release, obs, site_meta, per_site, out_dir):
    y = obs.to_numpy(np.float32)
    targets = obs.columns.astype(str)
    if PHOSPHO_LOGRATIO_PATH.exists():
        raw = pd.read_parquet(PHOSPHO_LOGRATIO_PATH)
        raw.index = raw.index.astype(str)
        raw.columns = raw.columns.astype(str)
        raw = raw.reindex(index=obs.index, columns=obs.columns)
        mean_abundance = np.nanmean(raw.to_numpy(np.float32), axis=0)
    else:
        mean_abundance = np.nanmean(y, axis=0)
    missingness = np.mean(~np.isfinite(y), axis=0)
    site = pd.DataFrame({
        "target": targets,
        "mean_abundance": mean_abundance,
        "missingness": missingness,
    })
    site = site.merge(per_site[["target", "spearman", "n"]], on="target", how="left")
    site = site.merge(site_meta[["scp682_site_id", "parent_gene", "residue", "position"]], left_on="target", right_on="scp682_site_id", how="left")
    site = site.drop(columns=["scp682_site_id"])
    site["parent_site_count"] = site.groupby("parent_gene")["target"].transform("count")

    if RNA_PATH.exists():
        rna = pd.read_parquet(RNA_PATH)
        rna.index = rna.index.astype(str)
        rna.columns = rna.columns.astype(str)
        common = obs.index.intersection(rna.index)
        rna = rna.loc[common]
        dyn = rna.quantile(0.9, axis=0) - rna.quantile(0.1, axis=0)
        site["rna_dynamic_range"] = site["parent_gene"].astype(str).map(dyn.to_dict())
    else:
        site["rna_dynamic_range"] = np.nan

    cls = site_class_tables(release, targets)
    site = site.merge(cls, on="target", how="left")
    site.to_csv(out_dir / "tables/site_level_features.tsv", sep="\t", index=False)
    return site


def panel_d_abundance_predictability(site, out_dir):
    d = site.dropna(subset=["mean_abundance", "rna_dynamic_range", "spearman"]).copy()
    plt.figure(figsize=(6.1, 5.2))
    hb = plt.hexbin(
        d["mean_abundance"],
        d["rna_dynamic_range"],
        C=d["spearman"],
        reduce_C_function=np.nanmean,
        gridsize=45,
        mincnt=1,
        cmap="viridis",
    )
    cb = plt.colorbar(hb)
    cb.set_label("Mean SCP682 rho")
    plt.xlabel("Mean log2 phosphosite abundance")
    plt.ylabel("Parent RNA dynamic range")
    plt.title("Abundance and predictability")
    sns.despine()
    savefig(out_dir / "figures/D_abundance_rna_dynamic_predictability_hex")


def panel_e_component_violin(comp, site, out_dir):
    d = comp.merge(site[["target", "site_class", "residue"]], on="target", how="left")
    d = d[np.isfinite(d["G_theta_fraction"])].copy()
    d.to_csv(out_dir / "tables/E_component_fraction_with_site_class.tsv", sep="\t", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)
    class_order = ["KSA", "CoPheeMap", "orphan"]
    sns.violinplot(data=d[d["site_class"].isin(class_order)], x="site_class", y="G_theta_fraction", order=class_order, palette=["#92B1D9", "#C1D8E9", "#D4D4D4"], cut=0, inner="quartile", linewidth=0.8, ax=axes[0])
    axes[0].set_xlabel("Site class")
    axes[0].set_ylabel("|G_theta| / (|B_phi| + |G_theta|)")
    res_order = [x for x in ["S", "T", "Y"] if x in set(d["residue"].dropna())]
    sns.violinplot(data=d[d["residue"].isin(res_order)], x="residue", y="G_theta_fraction", order=res_order, palette=["#92B1D9", "#F6C8B6", "#DBDDEF"][: len(res_order)], cut=0, inner="quartile", linewidth=0.8, ax=axes[1])
    axes[1].set_xlabel("Residue")
    axes[1].set_ylabel("")
    for ax in axes:
        ax.set_ylim(0, min(1.0, np.nanpercentile(d["G_theta_fraction"], 99.5) * 1.08))
        sns.despine(ax=ax)
    fig.suptitle("Contribution of graph residual")
    savefig(out_dir / "figures/E_Bphi_Gtheta_contribution_violin")


def panel_f_complex_heatmap(site, out_dir):
    rows = []
    label_rows = []
    selected = []
    max_sites = 12
    for name, genes in COMPLEXES.items():
        sub = site[site["parent_gene"].isin(genes)].dropna(subset=["spearman"]).copy()
        if sub.shape[0] < 4:
            continue
        sub = sub.sort_values("spearman", ascending=False).head(max_sites)
        vals = sub["spearman"].to_list()
        labels = [f"{g}|{str(t).split('|')[-1]}" for g, t in zip(sub["parent_gene"], sub["target"])]
        vals += [np.nan] * (max_sites - len(vals))
        labels += [""] * (max_sites - len(labels))
        rows.append(vals)
        label_rows.append(labels)
        selected.append(name)
    mat = np.asarray(rows, dtype=float)
    pd.DataFrame(mat, index=selected, columns=[f"site_{i+1}" for i in range(max_sites)]).to_csv(out_dir / "tables/F_complex_heatmap_matrix.tsv", sep="\t")
    pd.DataFrame(label_rows, index=selected, columns=[f"site_{i+1}" for i in range(max_sites)]).to_csv(out_dir / "tables/F_complex_heatmap_labels.tsv", sep="\t")

    plt.figure(figsize=(12.2, max(3.8, 0.42 * len(selected) + 1.6)))
    cmap = sns.color_palette("viridis", as_cmap=True)
    ax = sns.heatmap(mat, cmap=cmap, vmin=0, vmax=max(0.75, np.nanpercentile(mat, 95)), linewidths=0.4, linecolor="white", cbar_kws={"label": "Per-site rho"}, mask=~np.isfinite(mat))
    ax.set_yticklabels(selected, rotation=0)
    ax.set_xticklabels([str(i) for i in range(1, max_sites + 1)], rotation=0)
    ax.set_xlabel("Ranked member phosphosite")
    ax.set_ylabel("")
    ax.set_title("Pathway-level phosphosite consistency")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]) and label_rows[i][j]:
                ax.text(j + 0.5, i + 0.5, label_rows[i][j].replace("|", "\n"), ha="center", va="center", fontsize=5.2, color="black")
    savefig(out_dir / "figures/F_complex_consistency_heatmap")


def odds_ratio(a, b, c, d):
    a, b, c, d = [x + 0.5 for x in (a, b, c, d)]
    orv = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo = math.exp(math.log(orv) - 1.96 * se)
    hi = math.exp(math.log(orv) + 1.96 * se)
    return orv, lo, hi


def panel_g_failure_modes(site, out_dir):
    d = site.dropna(subset=["spearman"]).copy()
    d["failure"] = d["spearman"] < 0.1
    degree_proxy = d["kinase_count"].fillna(0)
    q_kinase = max(3, float(np.nanquantile(degree_proxy, 0.75)))
    features = {
        "Low abundance": d["mean_abundance"] <= np.nanquantile(d["mean_abundance"], 0.25),
        "High missingness": d["missingness"] >= np.nanquantile(d["missingness"], 0.75),
        "Singleton site": d["parent_site_count"].fillna(0) <= 1,
        "PTM crosstalk": degree_proxy >= q_kinase,
    }
    rows = []
    fail = d["failure"].to_numpy(bool)
    for name, flag in features.items():
        flag = np.asarray(flag, dtype=bool)
        a = int(np.sum(fail & flag))
        b = int(np.sum((~fail) & flag))
        c = int(np.sum(fail & (~flag)))
        dd = int(np.sum((~fail) & (~flag)))
        orv, lo, hi = odds_ratio(a, b, c, dd)
        rows.append({"feature": name, "odds_ratio": orv, "ci_low": lo, "ci_high": hi, "failure_with_feature": a, "nonfailure_with_feature": b})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "tables/G_failure_mode_odds_ratio.tsv", sep="\t", index=False)

    res = res.sort_values("odds_ratio")
    plt.figure(figsize=(5.6, 3.3))
    y = np.arange(res.shape[0])
    plt.errorbar(res["odds_ratio"], y, xerr=[res["odds_ratio"] - res["ci_low"], res["ci_high"] - res["odds_ratio"]], fmt="o", color="#2B2B2B", ecolor="#92B1D9", elinewidth=2, capsize=3)
    plt.axvline(1.0, color="#8A8A8A", linewidth=0.9, linestyle="--")
    plt.yticks(y, res["feature"])
    plt.xscale("log")
    plt.xlabel("Odds ratio for rho < 0.1")
    plt.title("Failure-mode enrichment")
    sns.despine()
    savefig(out_dir / "figures/G_failure_mode_odds_ratio")


def panel_h_scale_benchmark(release, obs, parent, ridge, rna_direct, out_dir, force=False):
    path = out_dir / "tables/H_runtime_scaling_one_epoch.tsv"
    if path.exists() and not force:
        bench = pd.read_csv(path, sep="\t")
    else:
        import torch
        import torch.nn as nn

        module = load_scp682_module(release)
        ckpt = torch.load(release / "models/scp682_exact_scnet_gnn_best.pt", map_location="cpu", weights_only=False)
        args = ckpt["args"]
        samples = [str(x) for x in ckpt["samples"]]
        targets = pd.Index([str(x) for x in ckpt["targets"]])
        obs = obs.loc[samples, targets]
        baseline = ((parent.loc[samples, targets] + ridge.loc[samples, targets] + rna_direct.loc[samples, targets]) / 3.0).astype(np.float32)
        y = obs.to_numpy(np.float32)
        b = baseline.to_numpy(np.float32)
        mask = np.isfinite(y) & np.isfinite(b)
        y0 = np.nan_to_num(y, nan=0.0).astype(np.float32)
        b0 = np.nan_to_num(b, nan=0.0).astype(np.float32)
        site_var = np.nanvar(y, axis=0)
        order = np.argsort(-site_var)
        sample_edge = ckpt["sample_edge_index"]
        if hasattr(sample_edge, "numpy"):
            sample_edge = sample_edge.numpy()
        full_site_edges = np.asarray(ckpt["site_edge_index"])
        full_prior = np.asarray(ckpt["site_prior"], dtype=np.float32)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x_in = np.nan_to_num(pd.read_parquet(RNA_PATH).reindex(samples).to_numpy(np.float32), nan=0.0)
        x_in = x_in[:, : min(512, x_in.shape[1])]
        x_in = (x_in - x_in.mean(axis=0, keepdims=True)) / (x_in.std(axis=0, keepdims=True) + 1e-6)

        rows = []
        for n_sites in [1000, 3000, 6000, 12000, len(targets)]:
            idx = np.sort(order[:n_sites])
            old_to_new = {old: i for i, old in enumerate(idx.tolist())}
            keep = np.isin(full_site_edges[0], idx) & np.isin(full_site_edges[1], idx)
            edges = full_site_edges[:, keep]
            if edges.shape[1] == 0:
                edges = np.vstack([idx, idx])
            edges = np.vectorize(old_to_new.get)(edges).astype(np.int64)
            self_edges = np.arange(n_sites, dtype=np.int64)
            edges = np.concatenate([edges, np.vstack([self_edges, self_edges])], axis=1)
            bb = b0[:, idx]
            yy = y0[:, idx]
            mm = mask[:, idx]
            feature = np.where(mm.T, bb.T, 0.0).astype(np.float32)
            feature = ((feature - feature.mean(axis=1, keepdims=True)) / (feature.std(axis=1, keepdims=True) + 1e-5)).astype(np.float32)

            def time_scp682():
                torch.cuda.empty_cache() if device.type == "cuda" else None
                model = module.SCP682ExactScNETResidual(
                    n_sites=n_sites,
                    n_samples=len(samples),
                    shrinkage=args.get("shrinkage", 0.3),
                    hidden=args.get("hidden", 64),
                    latent=args.get("latent", 32),
                    inter_dim=args.get("inter_dim", 96),
                    embd_dim=args.get("embd_dim", 32),
                    num_layers=args.get("num_layers", 1),
                ).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
                fx = torch.as_tensor(feature, dtype=torch.float32, device=device)
                row_edge = torch.as_tensor(edges, dtype=torch.long, device=device)
                col_edge = torch.as_tensor(sample_edge, dtype=torch.long, device=device)
                base_t = torch.as_tensor(bb, dtype=torch.float32, device=device)
                y_t = torch.as_tensor(yy, dtype=torch.float32, device=device)
                m_t = torch.as_tensor(mm, dtype=torch.bool, device=device)
                prior_t = torch.as_tensor(full_prior[idx], dtype=torch.float32, device=device)
                start = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                row_embed, col_embed, _ = model.graph_core(fx, col_edge, row_edge, collect_attention=False)
                batch_size = 4
                n_batches = int(math.ceil(len(samples) / batch_size))
                for batch_i, s0 in enumerate(range(0, len(samples), batch_size), start=1):
                    s1 = min(len(samples), s0 + batch_size)
                    si = torch.arange(s0, s1, dtype=torch.long, device=device)
                    pred_b, _, _, _, _ = model.decode(row_embed, col_embed, base_t[s0:s1], m_t[s0:s1], prior_t, sample_idx=si)
                    diff = pred_b[m_t[s0:s1]] - y_t[s0:s1][m_t[s0:s1]]
                    loss = (diff * diff).mean() / n_batches
                    loss.backward(retain_graph=batch_i < n_batches)
                opt.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                return time.perf_counter() - start

            class MLP(nn.Module):
                def __init__(self, n_out):
                    super().__init__()
                    self.net = nn.Sequential(nn.Linear(x_in.shape[1], 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n_out))

                def forward(self, x):
                    return self.net(x)

            class VAE(nn.Module):
                def __init__(self, n_out):
                    super().__init__()
                    self.enc = nn.Sequential(nn.Linear(x_in.shape[1], 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU())
                    self.mu = nn.Linear(128, 32)
                    self.dec = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, n_out))

                def forward(self, x):
                    return self.dec(self.mu(self.enc(x)))

            def time_dense(kind):
                torch.cuda.empty_cache() if device.type == "cuda" else None
                model = MLP(n_sites).to(device) if kind == "MLP" else VAE(n_sites).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
                x_t = torch.as_tensor(x_in, dtype=torch.float32, device=device)
                y_t = torch.as_tensor(yy, dtype=torch.float32, device=device)
                m_t = torch.as_tensor(mm, dtype=torch.bool, device=device)
                start = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                losses = []
                for s0 in range(0, len(samples), 64):
                    pred_b = model(x_t[s0:s0 + 64])
                    diff = pred_b[m_t[s0:s0 + 64]] - y_t[s0:s0 + 64][m_t[s0:s0 + 64]]
                    losses.append((diff * diff).mean())
                loss = torch.stack(losses).mean()
                loss.backward()
                opt.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                return time.perf_counter() - start

            for method, fn in [("SCP682", time_scp682), ("MLP", lambda: time_dense("MLP")), ("VAE", lambda: time_dense("VAE"))]:
                elapsed = fn()
                rows.append({"method": method, "n_phosphosite": int(n_sites), "seconds_per_epoch": float(elapsed)})
                log(f"scale {method} n={n_sites}: {elapsed:.2f}s")
        bench = pd.DataFrame(rows)
        bench.to_csv(path, sep="\t", index=False)

    plt.figure(figsize=(5.8, 3.8))
    for method, color in [("SCP682", "#92B1D9"), ("MLP", "#F6C8B6"), ("VAE", "#9A9A9A")]:
        sub = bench[bench["method"] == method].sort_values("n_phosphosite")
        plt.plot(sub["n_phosphosite"], sub["seconds_per_epoch"], marker="o", linewidth=1.8, label=method, color=color)
    plt.xlabel("Number of phosphosites")
    plt.ylabel("Seconds per epoch")
    plt.yscale("log")
    plt.title("Runtime scaling")
    plt.legend(frameon=False)
    sns.despine()
    savefig(out_dir / "figures/H_runtime_scaling")


def panel_c_manifest(out_dir):
    rows = []
    for fraction in [0.10, 0.25, 0.50, 0.75, 1.00]:
        for method in ["SCP682", "Ridge", "MLP", "VAE"]:
            rows.append({"training_fraction": fraction, "method": method, "status": "not_run_in_current_release"})
    pd.DataFrame(rows).to_csv(out_dir / "tables/C_learning_curve_required_runs.tsv", sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, default=MAIN_RELEASE)
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--force-forward", action="store_true")
    ap.add_argument("--force-scale", action="store_true")
    ap.add_argument("--skip-scale", action="store_true")
    args = ap.parse_args()

    sns.set_theme(style="white", font="DejaVu Sans", rc={"axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})
    ensure_dirs(args.output_dir)
    log("load release tables")
    obs, pred, parent, ridge, rna_direct, sample_meta, site_meta, per_site = load_release_tables(args.release)
    log("checkpoint forward")
    latent, comp = checkpoint_forward(args.release, obs, parent, ridge, rna_direct, args.output_dir, force=args.force_forward)
    log("site summary")
    site = build_site_summary(args.release, obs, site_meta, per_site, args.output_dir)
    log("panel A")
    panel_a_sample_umap(latent, sample_meta, args.output_dir)
    log("panel B")
    panel_b_hex(obs, pred, args.output_dir)
    log("panel D")
    panel_d_abundance_predictability(site, args.output_dir)
    log("panel E")
    panel_e_component_violin(comp, site, args.output_dir)
    log("panel F")
    panel_f_complex_heatmap(site, args.output_dir)
    log("panel G")
    panel_g_failure_modes(site, args.output_dir)
    log("panel C manifest")
    panel_c_manifest(args.output_dir)
    if not args.skip_scale:
        log("panel H")
        panel_h_scale_benchmark(args.release, obs, parent, ridge, rna_direct, args.output_dir, force=args.force_scale)
    summary = {
        "release": str(args.release),
        "output_dir": str(args.output_dir),
        "n_samples": int(obs.shape[0]),
        "n_sites": int(obs.shape[1]),
        "figures": sorted([p.name for p in (args.output_dir / "figures").glob("*.png")]),
    }
    (args.output_dir / "logs/run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
