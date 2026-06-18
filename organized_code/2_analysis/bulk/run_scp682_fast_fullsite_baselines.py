#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")


METHODS_FULL = [
    "mean_pred",
    "parent_mRNA_linear",
    "masked_ridge_linear",
    "masked_elasticnet_linear",
    "PCA_ridge",
    "MLP",
    "SCP682",
]
DATASETS = [
    "CPTAC_all",
    "CPTAC_kidney",
    "CPTAC_pancreas_HN",
    "CPTAC_gynecologic",
    "CPTAC_gi_hepato",
    "CPTAC_lung",
]


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ensure_dirs(out):
    for sub in ["predictions", "tables", "logs", "models"]:
        (out / sub).mkdir(parents=True, exist_ok=True)


def load_inputs(args):
    package = Path(args.package_dir)
    y = pd.read_parquet(package / "training_set" / "observed_phosphosite.parquet")
    y.index = y.index.astype(str)
    y.columns = y.columns.astype(str)
    target_meta = pd.read_csv(package / "training_set" / "phosphosite_target_manifest.tsv", sep="\t")
    target_meta["scp682_site_id"] = target_meta["scp682_site_id"].astype(str)

    meta = pd.read_csv(args.sample_manifest_path, sep="\t")
    if "sample_id" not in meta.columns and "index" in meta.columns:
        meta = meta.rename(columns={"index": "sample_id"})
    meta["sample_id"] = meta["sample_id"].astype(str)
    sample_ids = [s for s in meta["sample_id"] if s in y.index]
    meta = meta.set_index("sample_id").loc[sample_ids].reset_index()
    y = y.loc[sample_ids]
    meta["cancer_group"] = make_cancer_groups(meta)

    rna = pd.read_parquet(args.rna_path)
    rna.index = rna.index.astype(str)
    rna.columns = rna.columns.astype(str)
    rna = rna.loc[sample_ids]
    rna = rna.astype(np.float32)

    target_list = pd.read_csv(args.target_list_path, sep="\t")
    if "spearman" in target_list.columns:
        target_list = target_list[np.isfinite(pd.to_numeric(target_list["spearman"], errors="coerce"))]
    targets = target_list["target"].astype(str).tolist()
    targets = [t for t in targets if t in y.columns]
    if args.max_sites > 0:
        targets = targets[: args.max_sites]
    y = y[targets]
    target_meta = target_meta[target_meta["scp682_site_id"].isin(targets)].copy()

    return rna, y, meta, target_meta, targets, sample_ids


def make_cancer_groups(meta):
    mapping = {
        "CCRCC": "kidney",
        "NON_CCRCC": "kidney",
        "PDA": "pancreas_headneck",
        "HNSCC": "pancreas_headneck",
        "UCEC": "gynecologic",
        "UCEC_CONFIRM": "gynecologic",
        "STAD": "gi_hepatobiliary",
        "LUAD": "lung",
        "LUAD_CONFIRM": "lung",
        "LSCC": "lung",
    }
    return meta["cancer_label"].astype(str).map(mapping).fillna("other")


def dataset_masks(meta):
    g = meta["cancer_group"].astype(str).to_numpy()
    return {
        "CPTAC_all": np.ones(len(meta), dtype=bool),
        "CPTAC_kidney": g == "kidney",
        "CPTAC_pancreas_HN": g == "pancreas_headneck",
        "CPTAC_gynecologic": g == "gynecologic",
        "CPTAC_gi_hepato": g == "gi_hepatobiliary",
        "CPTAC_lung": g == "lung",
    }


def create_or_load_splits(args, out, rna, y, meta):
    path = out / "fold_indices.npz"
    if path.exists():
        dat = np.load(path, allow_pickle=True)
        folds = []
        for k in range(1, args.n_splits + 1):
            folds.append((dat[f"train_idx_fold{k}"], dat[f"test_idx_fold{k}"]))
        return folds
    strat = meta["cancer_group"].astype(str).to_numpy()
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    save = {}
    folds = []
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(meta)), strat), start=1):
        folds.append((tr.astype(np.int64), te.astype(np.int64)))
        save[f"train_idx_fold{k}"] = tr.astype(np.int64)
        save[f"test_idx_fold{k}"] = te.astype(np.int64)
    np.savez(path, **save)
    return folds


def create_or_load_features(args, out, rna, folds):
    feat_path = out / "feature_selection_per_fold.parquet"
    x = rna.to_numpy(dtype=np.float32, copy=True)
    genes = rna.columns.astype(str).to_numpy()
    fold_data = []
    rows = []
    if feat_path.exists():
        fs = pd.read_parquet(feat_path)
        for k, (tr, te) in enumerate(folds, start=1):
            sub = fs[fs["fold"] == k].sort_values("rank")
            sel_genes = sub["gene"].astype(str).tolist()
            idx = np.array([np.where(genes == g)[0][0] for g in sel_genes], dtype=np.int64)
            mu = x[tr][:, idx].mean(axis=0).astype(np.float32)
            sd = x[tr][:, idx].std(axis=0).astype(np.float32)
            sd[sd < 1e-6] = 1.0
            xs = ((x[:, idx] - mu) / sd).astype(np.float32)
            fold_data.append({"fold": k, "train_idx": tr, "test_idx": te, "x": xs, "genes": np.array(sel_genes)})
        return fold_data
    for k, (tr, te) in enumerate(folds, start=1):
        var = x[tr].var(axis=0)
        idx = np.argsort(var)[::-1][: args.n_genes]
        idx = np.sort(idx)
        mu = x[tr][:, idx].mean(axis=0).astype(np.float32)
        sd = x[tr][:, idx].std(axis=0).astype(np.float32)
        sd[sd < 1e-6] = 1.0
        xs = ((x[:, idx] - mu) / sd).astype(np.float32)
        sel_genes = genes[idx]
        fold_data.append({"fold": k, "train_idx": tr, "test_idx": te, "x": xs, "genes": sel_genes})
        for rank, (gene, variance) in enumerate(zip(sel_genes, var[idx]), start=1):
            rows.append({"fold": k, "rank": rank, "gene": gene, "variance_train_fold": float(variance)})
    pd.DataFrame(rows).to_parquet(feat_path, index=False)
    return fold_data


def masked_mse_torch(pred, y, mask):
    diff = pred[mask] - y[mask]
    if diff.numel() == 0:
        return pred.sum() * 0.0
    return (diff * diff).mean()


def build_model(method, n_in, n_out):
    import torch.nn as nn
    if method in {"masked_ridge_linear", "masked_elasticnet_linear", "PCA_ridge"}:
        return nn.Linear(n_in, n_out, bias=True)
    if method == "MLP":
        return nn.Sequential(
            nn.Linear(n_in, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_out),
        )
    raise ValueError(method)


def train_torch_model(method, x_all, y_all, train_idx, test_idx, args, fold, alpha=None, l1_ratio=None):
    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    train_idx = np.asarray(train_idx)
    val_size = 0.05 if method == "masked_elasticnet_linear" else 0.10
    tr_idx, val_idx = train_test_split(train_idx, test_size=val_size, random_state=args.seed + fold, shuffle=True)
    y = y_all.astype(np.float32)
    mask = np.isfinite(y)
    y0 = np.nan_to_num(y, nan=0.0).astype(np.float32)

    x_tr = torch.as_tensor(x_all[tr_idx], dtype=torch.float32)
    y_tr = torch.as_tensor(y0[tr_idx], dtype=torch.float32)
    m_tr = torch.as_tensor(mask[tr_idx], dtype=torch.bool)
    x_val = torch.as_tensor(x_all[val_idx], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(y0[val_idx], dtype=torch.float32, device=device)
    m_val = torch.as_tensor(mask[val_idx], dtype=torch.bool, device=device)

    model = build_model(method, x_all.shape[1], y.shape[1]).to(device)
    weight_decay = 0.0
    lr = args.lr
    max_epochs = args.epochs
    patience = args.patience
    if method == "masked_ridge_linear":
        weight_decay = args.ridge_weight_decay
        max_epochs = args.ridge_epochs
        patience = args.ridge_patience
    elif method == "PCA_ridge":
        weight_decay = args.ridge_weight_decay
        max_epochs = args.pca_epochs
        patience = args.ridge_patience
    elif method == "MLP":
        weight_decay = args.mlp_weight_decay
        lr = args.mlp_lr
        max_epochs = args.mlp_epochs
        patience = args.mlp_patience
    elif method == "masked_elasticnet_linear":
        max_epochs = args.elastic_epochs
        patience = args.elastic_patience
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = np.inf
    bad = 0
    n = len(tr_idx)
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = np.random.permutation(n)
        total = 0.0
        nb = 0
        for start in range(0, n, args.batch_size):
            ii = order[start : start + args.batch_size]
            xb = x_tr[ii].to(device)
            yb = y_tr[ii].to(device)
            mb = m_tr[ii].to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = masked_mse_torch(pred, yb, mb)
            if method == "masked_elasticnet_linear":
                w = model.weight
                l1 = w.abs().mean()
                l2 = (w * w).mean()
                loss = loss + alpha * (l1_ratio * l1 + (1.0 - l1_ratio) * 0.5 * l2)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            nb += 1
        model.eval()
        with torch.no_grad():
            val_loss = float(masked_mse_torch(model(x_val), y_val, m_val).detach().cpu())
        history.append({"fold": fold, "epoch": epoch, "train_loss": total / max(nb, 1), "val_loss": val_loss})
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch % 10 == 0:
            log(f"{method} fold {fold} epoch {epoch} val={val_loss:.5f} best={best_val:.5f}")
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    outs = []
    x_test = torch.as_tensor(x_all[test_idx], dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(test_idx), args.batch_size):
            xb = x_test[start : start + args.batch_size].to(device)
            outs.append(model(xb).detach().cpu().numpy().astype(np.float32))
    return np.vstack(outs), pd.DataFrame(history), best_val


def run_elastic_search(x_all, y_all, train_idx, args, fold):
    combos = [(a, r) for a in args.elastic_alpha_grid for r in args.elastic_l1_grid]
    # Use a fixed subset of the training fold for fast global hyperparameter selection.
    rng = np.random.default_rng(args.seed + 1000 + fold)
    tr = np.asarray(train_idx)
    if len(tr) > args.elastic_search_samples:
        tr = rng.choice(tr, size=args.elastic_search_samples, replace=False)
    best = None
    rows = []
    for alpha, l1_ratio in combos:
        old_epochs = args.elastic_epochs
        old_patience = args.elastic_patience
        args.elastic_epochs = args.elastic_search_epochs
        args.elastic_patience = args.elastic_search_patience
        _, hist, best_val = train_torch_model(
            "masked_elasticnet_linear", x_all, y_all, tr, tr[: min(10, len(tr))],
            args, fold, alpha=alpha, l1_ratio=l1_ratio
        )
        args.elastic_epochs = old_epochs
        args.elastic_patience = old_patience
        rows.append({"fold": fold, "alpha": alpha, "l1_ratio": l1_ratio, "best_val_loss": best_val})
        if best is None or best_val < best[0]:
            best = (best_val, alpha, l1_ratio)
        log(f"elastic search fold {fold}: alpha={alpha} l1={l1_ratio} val={best_val:.5f}")
    return best[1], best[2], pd.DataFrame(rows)


def save_fold_pred(out, method, fold, sample_ids, targets, pred):
    df = pd.DataFrame(pred, index=sample_ids, columns=targets)
    path = out / f"preds_{method}_fold{fold}.parquet"
    df.to_parquet(path)
    return path


def assemble_oof(out, method, sample_ids, targets, folds):
    pred = pd.DataFrame(np.nan, index=sample_ids, columns=targets, dtype=np.float32)
    for k, (_, te) in enumerate(folds, start=1):
        p = out / f"preds_{method}_fold{k}.parquet"
        if not p.exists():
            return None
        part = pd.read_parquet(p)
        pred.iloc[te, :] = part.loc[[sample_ids[i] for i in te], targets].to_numpy(dtype=np.float32)
    oof = out / f"preds_{method}_oof.parquet"
    pred.to_parquet(oof)
    return pred


def split_existing_oof(src, out, method, sample_ids, targets, folds):
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)
    pred = pd.read_parquet(src)
    pred.index = pred.index.astype(str)
    pred.columns = pred.columns.astype(str)
    pred = pred.loc[sample_ids, targets]
    for k, (_, te) in enumerate(folds, start=1):
        ids = [sample_ids[i] for i in te]
        save_fold_pred(out, method, k, ids, targets, pred.loc[ids].to_numpy(dtype=np.float32))
    pred.to_parquet(out / f"preds_{method}_oof.parquet")
    return pred


def run_fullsite(args):
    out = Path(args.output_dir)
    ensure_dirs(out)
    rna, y, meta, target_meta, targets, sample_ids = load_inputs(args)
    folds = create_or_load_splits(args, out, rna, y, meta)
    fold_data = create_or_load_features(args, out, rna, folds)
    with open(out / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, ensure_ascii=False, indent=2)
    y_all = y.to_numpy(dtype=np.float32)

    method_preds = {}
    if args.mean_pred_source and Path(args.mean_pred_source).exists():
        method_preds["mean_pred"] = split_existing_oof(args.mean_pred_source, out, "mean_pred", sample_ids, targets, folds)
    if args.parent_mrna_source and Path(args.parent_mrna_source).exists():
        method_preds["parent_mRNA_linear"] = split_existing_oof(args.parent_mrna_source, out, "parent_mRNA_linear", sample_ids, targets, folds)

    for method in ["masked_ridge_linear", "masked_elasticnet_linear", "PCA_ridge", "MLP"]:
        if (out / f"preds_{method}_oof.parquet").exists() and args.resume:
            log(f"{method}: reuse OOF")
            method_preds[method] = pd.read_parquet(out / f"preds_{method}_oof.parquet")
            continue
        histories = []
        searches = []
        for fd in fold_data:
            k = fd["fold"]
            tr, te = fd["train_idx"], fd["test_idx"]
            x_all = fd["x"]
            if method == "PCA_ridge":
                log(f"{method} fold {k}: fit PCA")
                pca = PCA(n_components=min(args.pca_components, len(tr) - 1, x_all.shape[1]), svd_solver="randomized", random_state=args.seed + k)
                x_train_scores = pca.fit_transform(x_all[tr])
                x_scores = pca.transform(x_all).astype(np.float32)
                np.savez_compressed(out / "models" / f"PCA_ridge_fold{k}_pca.npz",
                                    components=pca.components_.astype(np.float32),
                                    mean=pca.mean_.astype(np.float32),
                                    explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32))
                x_use = x_scores
            else:
                x_use = x_all
            alpha = None
            l1_ratio = None
            if method == "masked_elasticnet_linear":
                alpha, l1_ratio, search_df = run_elastic_search(x_use, y_all, tr, args, k)
                searches.append(search_df)
                log(f"{method} fold {k}: selected alpha={alpha} l1_ratio={l1_ratio}")
            log(f"{method} fold {k}: train")
            pred, hist, best_val = train_torch_model(method, x_use, y_all, tr, te, args, k, alpha=alpha, l1_ratio=l1_ratio)
            hist["method"] = method
            hist["best_val_loss"] = best_val
            if alpha is not None:
                hist["alpha"] = alpha
                hist["l1_ratio"] = l1_ratio
            histories.append(hist)
            ids = [sample_ids[i] for i in te]
            save_fold_pred(out, method, k, ids, targets, pred)
        pd.concat(histories, ignore_index=True).to_csv(out / "logs" / f"{method}_training_history.tsv", sep="\t", index=False)
        if searches:
            pd.concat(searches, ignore_index=True).to_csv(out / "logs" / "masked_elasticnet_linear_search.tsv", sep="\t", index=False)
        method_preds[method] = assemble_oof(out, method, sample_ids, targets, folds)

    if args.scp682_oof_path and Path(args.scp682_oof_path).exists():
        method_preds["SCP682"] = split_existing_oof(args.scp682_oof_path, out, "SCP682", sample_ids, targets, folds)

    rows = []
    for method in METHODS_FULL:
        oof_path = out / f"preds_{method}_oof.parquet"
        if not oof_path.exists():
            continue
        log(f"evaluate {method}")
        pred = pd.read_parquet(oof_path).loc[sample_ids, targets]
        rows.append(evaluate_method(method, pred, y, meta, targets, args.min_eval_samples))
    final = pd.concat(rows, ignore_index=True)
    final.to_csv(out / "per_site_spearman.tsv", sep="\t", index=False)
    write_summary(final, out)
    (out / "done_fullsite.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


def evaluate_method(method, pred_df, y_df, meta, targets, min_samples):
    masks = dataset_masks(meta)
    yv = y_df[targets].to_numpy(dtype=np.float32)
    pv = pred_df[targets].to_numpy(dtype=np.float32)
    rows = []
    for dataset, mask in masks.items():
        idx = np.where(mask)[0]
        for j, target in enumerate(targets):
            yy = yv[idx, j]
            pp = pv[idx, j]
            ok = np.isfinite(yy) & np.isfinite(pp)
            n = int(ok.sum())
            if n < min_samples or np.nanstd(yy[ok]) < 1e-8 or np.nanstd(pp[ok]) < 1e-8:
                rho = np.nan
                pval = np.nan
            else:
                r = spearmanr(pp[ok], yy[ok])
                rho = float(r.statistic)
                pval = float(r.pvalue)
            rows.append({"method": method, "dataset": dataset, "target": target,
                         "n_samples_used": n, "spearman": rho, "rho_p_value": pval})
    return pd.DataFrame(rows)


def write_summary(df, out):
    med = df.groupby(["method", "dataset"])["spearman"].median().unstack("dataset")
    na = df.assign(is_na=df["spearman"].isna()).groupby(["method", "dataset"])["is_na"].mean().unstack("dataset")
    med.to_csv(out / "median_spearman_matrix.tsv", sep="\t")
    na.to_csv(out / "na_ratio_matrix.tsv", sep="\t")
    log("median Spearman matrix")
    print(med.to_string(), flush=True)
    log("NA ratio matrix")
    print(na.to_string(), flush=True)


def build_representative_sites(args, out, y, target_meta, targets):
    rep_path = out / "representative_1k_sites.tsv"
    if rep_path.exists() and args.resume:
        return pd.read_csv(rep_path, sep="\t")
    meta = target_meta.set_index("scp682_site_id").reindex(targets).reset_index()
    coverage = y[targets].notna().mean(axis=0).rename("coverage").reset_index().rename(columns={"index": "target"})
    df = meta.rename(columns={"scp682_site_id": "target"}).merge(coverage, on="target", how="left")
    df["coverage_decile"] = pd.qcut(df["coverage"].rank(method="first"), 10, labels=False, duplicates="drop").astype(int)
    df["residue_type"] = df["residue"].astype(str).str[0].where(df["residue"].astype(str).str[0].isin(["S", "T", "Y"]), "other")

    copheeksa = pd.read_csv(args.copheeksa_path, sep="\t", usecols=["gene_site_id"])
    copheeksa_sites = set(copheeksa["gene_site_id"].astype(str))
    kstar_sites = set()
    kp = Path(args.kstar_path)
    if kp.exists():
        ks = pd.read_csv(kp, sep="\t", usecols=lambda c: c in {"substrate_gene", "site", "model_site_id"})
        if "substrate_gene" in ks.columns and "site" in ks.columns:
            sub = ks.dropna(subset=["substrate_gene", "site"]).copy()
            kstar_sites.update((sub["substrate_gene"].astype(str) + "|" + sub["site"].astype(str)).tolist())
    df["kinase_substrate_tag"] = np.where(df["target"].isin(copheeksa_sites), "in_CoPheeKSA",
                                  np.where(df["target"].isin(kstar_sites), "in_KSTAR", "neither"))
    rng = np.random.default_rng(args.seed)
    pieces = []
    grouped = df.groupby(["coverage_decile", "residue_type", "kinase_substrate_tag"], dropna=False)
    for _, sub in grouped:
        n = max(1, int(math.ceil(args.representative_sites * len(sub) / len(df))))
        n = min(n, len(sub))
        pieces.append(sub.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))
    rep = pd.concat(pieces, ignore_index=True)
    if len(rep) > args.representative_sites:
        rep = rep.sample(n=args.representative_sites, random_state=args.seed)
    rep = rep.sort_values(["coverage_decile", "residue_type", "kinase_substrate_tag", "target"]).reset_index(drop=True)
    rep.to_csv(rep_path, sep="\t", index=False)
    return rep


TREE_CONTEXT = {}


def init_tree_context(x_folds, y, targets, folds, sample_ids):
    TREE_CONTEXT["x_folds"] = x_folds
    TREE_CONTEXT["y"] = y
    TREE_CONTEXT["targets"] = targets
    TREE_CONTEXT["folds"] = folds
    TREE_CONTEXT["sample_ids"] = sample_ids


def fit_tree_site(method, target):
    from xgboost import XGBRegressor
    y = TREE_CONTEXT["y"][target].to_numpy(dtype=np.float32)
    pred = np.full(len(y), np.nan, dtype=np.float32)
    for fd in TREE_CONTEXT["x_folds"]:
        tr = fd["train_idx"]
        te = fd["test_idx"]
        yy = y[tr]
        ok = np.isfinite(yy)
        if ok.sum() < 10 or np.nanstd(yy[ok]) < 1e-8:
            if ok.sum() > 0:
                pred[te] = np.nanmean(yy[ok])
            continue
        if method == "random_forest":
            model = RandomForestRegressor(n_estimators=200, max_depth=4, n_jobs=1, random_state=42)
        elif method == "gbm":
            model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
        elif method == "xgboost":
            model = XGBRegressor(tree_method="hist", device="cuda", n_estimators=500,
                                 max_depth=6, learning_rate=0.05, random_state=42,
                                 objective="reg:squarederror", n_jobs=1)
        else:
            raise ValueError(method)
        try:
            model.fit(fd["x"][tr][ok], yy[ok])
            pred[te] = model.predict(fd["x"][te]).astype(np.float32)
        except Exception:
            pred[te] = np.nanmean(yy[ok])
    return target, pred


def run_tree_mode(args, methods):
    out = Path(args.output_dir)
    ext = Path(args.ext_output_dir)
    ensure_dirs(out)
    ensure_dirs(ext)
    rna, y, meta, target_meta, targets, sample_ids = load_inputs(args)
    folds = create_or_load_splits(args, out, rna, y, meta)
    fold_data = create_or_load_features(args, out, rna, folds)
    rep = build_representative_sites(args, ext, y, target_meta, targets)
    rep_targets = rep["target"].astype(str).tolist()
    init_tree_context(fold_data, y[rep_targets], rep_targets, folds, sample_ids)

    all_eval = []
    for method in methods:
        pred_path = ext / "predictions" / f"{method}_1k_oof.parquet"
        if pred_path.exists() and args.resume:
            pred = pd.read_parquet(pred_path)
        else:
            log(f"{method}: start 1k representative sites")
            if method == "xgboost":
                rows = [fit_tree_site(method, t) for t in rep_targets]
            else:
                with parallel_config(backend="loky", temp_folder=str(ext / "joblib_tmp"), max_nbytes="10M"):
                    rows = Parallel(n_jobs=args.tree_n_jobs, verbose=10)(
                        delayed(fit_tree_site)(method, t) for t in rep_targets
                    )
            pred = pd.DataFrame({t: p for t, p in rows}, index=sample_ids)[rep_targets]
            pred.to_parquet(pred_path)
        ev = evaluate_method(method, pred, y[rep_targets], meta, rep_targets, args.min_eval_samples)
        ev.to_csv(ext / "tables" / f"{method}_1k_per_site.tsv", sep="\t", index=False)
        all_eval.append(ev)
    final = pd.concat(all_eval, ignore_index=True)
    mode_name = "_".join(methods)
    final.to_csv(ext / "tables" / f"tree_models_1k_sites_{mode_name}.tsv", sep="\t", index=False)
    existing = sorted((ext / "tables").glob("tree_models_1k_sites_*.tsv"))
    merged = pd.concat([pd.read_csv(p, sep="\t") for p in existing], ignore_index=True).drop_duplicates(
        subset=["method", "dataset", "target"], keep="last")
    merged.to_csv(ext / "tree_models_1k_sites.tsv", sep="\t", index=False)
    write_summary(merged, ext)
    (ext / f"done_{mode_name}.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fullsite", "trees_cpu", "trees_xgb"], required=True)
    p.add_argument("--package-dir", default="/data/lsy/Infinite_Stream/SCP682_PORTABLE")
    p.add_argument("--rna-path", default="/data/lsy/Infinite_Stream/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/rna_log2_tpm_paired.parquet")
    p.add_argument("--sample-manifest-path", default="/data/lsy/Infinite_Stream/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/sample_manifest.tsv")
    p.add_argument("--target-list-path", default="/data/lsy/Infinite_Stream/SCP682-main/results/20260523_general_graph_residual_e160/tables/per_site_pseudo_external_spearman_best.tsv")
    p.add_argument("--scp682-oof-path", default="/data/lsy/Infinite_Stream/SCP682-main/results/20260523_general_graph_residual_e160/predictions/scp682_general_graph_residual_pseudo_external_phosphosite_best.parquet")
    p.add_argument("--mean-pred-source", default="/data/lsy/Infinite_Stream/SCP682-main/results/20260525_ml_baseline_benchmark_5fold/baseline_predictions/mean_pred_oof.parquet")
    p.add_argument("--parent-mrna-source", default="/data/lsy/Infinite_Stream/SCP682-main/results/20260525_ml_baseline_benchmark_5fold/baseline_predictions/mrna_naive_oof.parquet")
    p.add_argument("--output-dir", default="/data/lsy/Infinite_Stream/SCP682-main/results/fast_fullsite_baselines")
    p.add_argument("--ext-output-dir", default="/data/lsy/Infinite_Stream/SCP682-main/results/fast_fullsite_baselines_extdata")
    p.add_argument("--copheeksa-path", default="/data/lsy/Infinite_Stream/01_data/pathway_prior/processed/copheemap_v1/copheeksa_model_phosphosite_kinase_predictions.tsv")
    p.add_argument("--kstar-path", default="/data/lsy/Infinite_Stream/01_data/pathway_prior/intermediate/kstar_20260516/kstar_default_network_edges_long.tsv")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-genes", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-eval-samples", type=int, default=10)
    p.add_argument("--max-sites", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--ridge-weight-decay", type=float, default=1e-3)
    p.add_argument("--ridge-epochs", type=int, default=200)
    p.add_argument("--ridge-patience", type=int, default=10)
    p.add_argument("--elastic-alpha-grid", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    p.add_argument("--elastic-l1-grid", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--elastic-search-samples", type=int, default=512)
    p.add_argument("--elastic-search-epochs", type=int, default=60)
    p.add_argument("--elastic-search-patience", type=int, default=6)
    p.add_argument("--elastic-epochs", type=int, default=200)
    p.add_argument("--elastic-patience", type=int, default=10)
    p.add_argument("--pca-components", type=int, default=256)
    p.add_argument("--pca-epochs", type=int, default=300)
    p.add_argument("--mlp-lr", type=float, default=5e-4)
    p.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    p.add_argument("--mlp-epochs", type=int, default=500)
    p.add_argument("--mlp-patience", type=int, default=20)
    p.add_argument("--representative-sites", type=int, default=1000)
    p.add_argument("--tree-n-jobs", type=int, default=70)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "fullsite":
        run_fullsite(args)
    elif args.mode == "trees_cpu":
        run_tree_mode(args, ["random_forest", "gbm"])
    elif args.mode == "trees_xgb":
        run_tree_mode(args, ["xgboost"])


if __name__ == "__main__":
    main()

