#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")


SIMPLE_METHODS = {"mean_pred", "mrna_naive", "lasso", "elasticnet", "pls"}
TREE_METHODS = {"random_forest", "gbm"}
GPU_METHODS = {"xgboost", "mlp"}
ALL_METHODS = [
    "mean_pred",
    "mrna_naive",
    "lasso",
    "elasticnet",
    "pls",
    "random_forest",
    "gbm",
    "xgboost",
    "mlp",
]


FOLD_DATA = None
Y_MATRIX = None
TARGETS = None
TARGET_TO_PARENT = None
RNA_FULL = None
RNA_GENES = None
ARGS_GLOBAL = None


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def read_table(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def load_rna_matrix(path, sample_ids):
    rna = pd.read_parquet(path)
    sample_ids = [str(x) for x in sample_ids]
    index_hits = len(set(map(str, rna.index)).intersection(sample_ids))
    col_hits = len(set(map(str, rna.columns)).intersection(sample_ids))
    if index_hits >= max(10, col_hits):
        rna.index = rna.index.astype(str)
        return rna.loc[sample_ids]
    if "gene" in rna.columns and col_hits >= 10:
        tmp = rna.set_index("gene")
        tmp.columns = tmp.columns.astype(str)
        return tmp[sample_ids].T
    if col_hits >= 10:
        rna.columns = rna.columns.astype(str)
        return rna[sample_ids].T
    raise RuntimeError(f"Cannot align RNA matrix to sample ids: {path}")


def make_cancer_groups(meta):
    label = meta["cancer_label"].astype(str)
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
    group = label.map(mapping).fillna("other")
    return group


def make_dataset_masks(meta):
    group = meta["cancer_group"].astype(str).to_numpy()
    return {
        "CPTAC_all": np.ones(len(meta), dtype=bool),
        "CPTAC_kidney": group == "kidney",
        "CPTAC_pancreas_HN": group == "pancreas_headneck",
        "CPTAC_gynecologic": group == "gynecologic",
        "CPTAC_gi_hepato": group == "gi_hepatobiliary",
        "CPTAC_lung": group == "lung",
    }


def prepare_fold_data(x_df, y_df, meta, out_dir, n_genes, seed, n_splits):
    folds = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strat = meta["cancer_group"].astype(str).to_numpy()
    genes = x_df.columns.astype(str).to_numpy()
    x_all = x_df.to_numpy(dtype=np.float32, copy=True)
    manifest_rows = []
    for fold_id, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(meta)), strat), start=1):
        x_train = x_all[tr_idx]
        var = np.nanvar(x_train, axis=0)
        top_idx = np.argsort(var)[::-1][:n_genes]
        top_idx = np.sort(top_idx)
        mu = np.nanmean(x_train[:, top_idx], axis=0).astype(np.float32)
        sd = np.nanstd(x_train[:, top_idx], axis=0).astype(np.float32)
        sd[sd < 1e-6] = 1.0
        x_scaled = ((x_all[:, top_idx] - mu) / sd).astype(np.float32)
        fold_genes = genes[top_idx]
        fold_gene_to_pos = {g: i for i, g in enumerate(fold_genes)}
        folds.append(
            {
                "fold_id": fold_id,
                "train_idx": tr_idx.astype(np.int64),
                "test_idx": te_idx.astype(np.int64),
                "x_scaled": x_scaled,
                "genes": fold_genes,
                "gene_to_pos": fold_gene_to_pos,
            }
        )
        for rank, gene in enumerate(fold_genes, start=1):
            manifest_rows.append({"fold": fold_id, "rank": rank, "gene": gene})
    pd.DataFrame(manifest_rows).to_csv(out_dir / "fold_top5000_genes.tsv", sep="\t", index=False)
    return folds


def parent_feature_for_fold(fold, target):
    parent = TARGET_TO_PARENT.get(target)
    if not parent:
        return None
    gene_to_pos = fold["gene_to_pos"]
    if parent in gene_to_pos:
        return fold["x_scaled"][:, gene_to_pos[parent]].reshape(-1, 1)
    if RNA_FULL is None or parent not in RNA_GENES:
        return None
    col = RNA_GENES[parent]
    raw = RNA_FULL[:, col].astype(np.float32)
    tr = fold["train_idx"]
    mu = np.nanmean(raw[tr])
    sd = np.nanstd(raw[tr])
    if not np.isfinite(sd) or sd < 1e-6:
        sd = 1.0
    return ((raw - mu) / sd).reshape(-1, 1).astype(np.float32)


def make_estimator(method, args):
    if method == "mean_pred":
        return DummyRegressor(strategy="mean")
    if method == "lasso":
        return LassoCV(cv=5, n_jobs=1, max_iter=3000, random_state=args.seed)
    if method == "elasticnet":
        return ElasticNetCV(l1_ratio=0.5, cv=5, n_jobs=1, max_iter=3000, random_state=args.seed)
    if method == "pls":
        return None
    if method == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            n_jobs=args.tree_inner_jobs,
            random_state=args.seed,
        )
    if method == "gbm":
        return GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=args.seed,
        )
    raise ValueError(method)


def fit_one_target(target_pos, method):
    target = TARGETS[target_pos]
    y = Y_MATRIX[:, target_pos]
    pred = np.full(Y_MATRIX.shape[0], np.nan, dtype=np.float32)
    for fold in FOLD_DATA:
        tr = fold["train_idx"]
        te = fold["test_idx"]
        y_tr_all = y[tr]
        ok_tr = np.isfinite(y_tr_all)
        if ok_tr.sum() < 10 or np.nanstd(y_tr_all[ok_tr]) < 1e-8:
            if ok_tr.sum() > 0:
                pred[te] = np.nanmean(y_tr_all[ok_tr])
            continue
        if method == "mrna_naive":
            x_one = parent_feature_for_fold(fold, target)
            if x_one is None:
                pred[te] = np.nanmean(y_tr_all[ok_tr])
                continue
            x_tr = x_one[tr][ok_tr]
            x_te = x_one[te]
            estimator = LinearRegression()
        elif method == "mean_pred":
            x_tr = np.zeros((ok_tr.sum(), 1), dtype=np.float32)
            x_te = np.zeros((len(te), 1), dtype=np.float32)
            estimator = make_estimator(method, ARGS_GLOBAL)
        else:
            x_fold = fold["x_scaled"]
            x_tr = x_fold[tr][ok_tr]
            x_te = x_fold[te]
            if method == "pls":
                n_comp = min(10, x_tr.shape[0] - 1, x_tr.shape[1])
                if n_comp < 1:
                    pred[te] = np.nanmean(y_tr_all[ok_tr])
                    continue
                estimator = PLSRegression(n_components=n_comp)
            else:
                estimator = make_estimator(method, ARGS_GLOBAL)
        try:
            estimator.fit(x_tr, y_tr_all[ok_tr])
            out = estimator.predict(x_te)
            pred[te] = np.asarray(out).reshape(-1).astype(np.float32)
        except Exception:
            pred[te] = np.nanmean(y_tr_all[ok_tr])
    return target, pred


def run_cpu_method(method, args, out_dir, sample_ids):
    pred_path = out_dir / "baseline_predictions" / f"{method}_oof.parquet"
    if pred_path.exists() and args.resume:
        log(f"{method}: reuse {pred_path}")
        return pd.read_parquet(pred_path)
    n_jobs = args.simple_n_jobs if method in SIMPLE_METHODS else args.tree_outer_jobs
    log(f"{method}: start per-target OOF with n_jobs={n_jobs}")
    temp_dir = out_dir / "joblib_tmp" / method
    temp_dir.mkdir(parents=True, exist_ok=True)
    with parallel_config(backend="loky", temp_folder=str(temp_dir), max_nbytes="10M"):
        rows = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(fit_one_target)(j, method) for j in range(len(TARGETS))
        )
    pred = pd.DataFrame({target: values for target, values in rows}, index=sample_ids)
    pred = pred[TARGETS]
    pred.to_parquet(pred_path)
    log(f"{method}: saved {pred_path}")
    return pred


def wait_for_gpu(free_memory_mb, poll_seconds):
    while True:
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            out = subprocess.check_output(cmd, text=True).strip().splitlines()[0]
            used, total = [int(x.strip()) for x in out.split(",")]
            free = total - used
            log(f"GPU memory free={free} MiB total={total} MiB")
            if free >= free_memory_mb:
                return
        except Exception as exc:
            log(f"GPU check failed: {exc}")
            return
        time.sleep(poll_seconds)


def run_xgboost(args, out_dir, sample_ids):
    pred_path = out_dir / "baseline_predictions" / "xgboost_oof.parquet"
    if pred_path.exists() and args.resume:
        log(f"xgboost: reuse {pred_path}")
        return pd.read_parquet(pred_path)
    wait_for_gpu(args.wait_gpu_free_mb, args.gpu_poll_seconds)
    from xgboost import XGBRegressor

    pred_mat = np.full(Y_MATRIX.shape, np.nan, dtype=np.float32)
    log("xgboost: start GPU per-target OOF")
    for j, target in enumerate(TARGETS):
        if j % 100 == 0:
            log(f"xgboost: target {j}/{len(TARGETS)}")
        y = Y_MATRIX[:, j]
        for fold in FOLD_DATA:
            tr = fold["train_idx"]
            te = fold["test_idx"]
            y_tr = y[tr]
            ok = np.isfinite(y_tr)
            if ok.sum() < 10 or np.nanstd(y_tr[ok]) < 1e-8:
                if ok.sum() > 0:
                    pred_mat[te, j] = np.nanmean(y_tr[ok])
                continue
            model = XGBRegressor(
                tree_method="hist",
                device="cuda",
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                random_state=args.seed,
                objective="reg:squarederror",
                n_jobs=1,
            )
            try:
                model.fit(fold["x_scaled"][tr][ok], y_tr[ok], verbose=False)
                pred_mat[te, j] = model.predict(fold["x_scaled"][te]).astype(np.float32)
            except Exception:
                pred_mat[te, j] = np.nanmean(y_tr[ok])
    pred = pd.DataFrame(pred_mat, index=sample_ids, columns=TARGETS)
    pred.to_parquet(pred_path)
    log(f"xgboost: saved {pred_path}")
    return pred


class TorchMLP:
    def __init__(self, args):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.nn = nn
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.args = args

    def build_model(self, n_in, n_out):
        nn = self.nn
        return nn.Sequential(
            nn.Linear(n_in, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_out),
        ).to(self.device)

    def masked_mse(self, pred, y, mask):
        diff = pred[mask] - y[mask]
        if diff.numel() == 0:
            return (pred.sum() * 0.0) + 0.0
        return (diff * diff).mean()

    def fit_fold(self, x_all, y_all, train_idx, test_idx, seed):
        torch = self.torch
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=0.1,
            random_state=seed,
            shuffle=True,
        )
        model = self.build_model(x_all.shape[1], y_all.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x_train = torch.as_tensor(x_all[train_idx], dtype=torch.float32)
        y_train_np = y_all[train_idx].astype(np.float32)
        m_train_np = np.isfinite(y_train_np)
        y_train_np = np.nan_to_num(y_train_np, nan=0.0)
        y_train = torch.as_tensor(y_train_np, dtype=torch.float32)
        m_train = torch.as_tensor(m_train_np, dtype=torch.bool)
        x_val = torch.as_tensor(x_all[val_idx], dtype=torch.float32, device=self.device)
        y_val_np = y_all[val_idx].astype(np.float32)
        m_val = torch.as_tensor(np.isfinite(y_val_np), dtype=torch.bool, device=self.device)
        y_val = torch.as_tensor(np.nan_to_num(y_val_np, nan=0.0), dtype=torch.float32, device=self.device)
        best_state = None
        best_loss = np.inf
        bad = 0
        n = len(train_idx)
        for epoch in range(1, self.args.mlp_max_epochs + 1):
            order = np.random.permutation(n)
            model.train()
            for start in range(0, n, self.args.mlp_batch_size):
                idx = order[start : start + self.args.mlp_batch_size]
                xb = x_train[idx].to(self.device)
                yb = y_train[idx].to(self.device)
                mb = m_train[idx].to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = self.masked_mse(model(xb), yb, mb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(self.masked_mse(model(x_val), y_val, m_val).detach().cpu())
            if val_loss + 1e-6 < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if epoch % 10 == 0:
                log(f"mlp fold seed={seed} epoch={epoch} val_loss={val_loss:.5f} best={best_loss:.5f}")
            if bad >= self.args.mlp_patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        outs = []
        x_test = torch.as_tensor(x_all[test_idx], dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, len(test_idx), self.args.mlp_batch_size):
                xb = x_test[start : start + self.args.mlp_batch_size].to(self.device)
                outs.append(model(xb).detach().cpu().numpy().astype(np.float32))
        return np.vstack(outs)


def run_mlp(args, out_dir, sample_ids):
    pred_path = out_dir / "baseline_predictions" / "mlp_oof.parquet"
    if pred_path.exists() and args.resume:
        log(f"mlp: reuse {pred_path}")
        return pd.read_parquet(pred_path)
    wait_for_gpu(args.wait_gpu_free_mb, args.gpu_poll_seconds)
    trainer = TorchMLP(args)
    pred_mat = np.full(Y_MATRIX.shape, np.nan, dtype=np.float32)
    log("mlp: start PyTorch multi-task OOF")
    for fold in FOLD_DATA:
        log(f"mlp: fold {fold['fold_id']} train")
        out = trainer.fit_fold(
            fold["x_scaled"],
            Y_MATRIX,
            fold["train_idx"],
            fold["test_idx"],
            args.seed + fold["fold_id"],
        )
        pred_mat[fold["test_idx"]] = out
    pred = pd.DataFrame(pred_mat, index=sample_ids, columns=TARGETS)
    pred.to_parquet(pred_path)
    log(f"mlp: saved {pred_path}")
    return pred


def evaluate_predictions(method, pred_df, y_df, meta, out_dir, min_samples):
    per_method_path = out_dir / "tables" / f"{method}_per_site.tsv"
    if per_method_path.exists():
        log(f"{method}: reuse evaluation {per_method_path}")
        return pd.read_csv(per_method_path, sep="\t")
    masks = make_dataset_masks(meta)
    y = y_df[TARGETS].to_numpy(dtype=np.float32)
    p = pred_df[TARGETS].to_numpy(dtype=np.float32)
    rows = []
    for dataset, mask in masks.items():
        idx = np.where(mask)[0]
        for j, target in enumerate(TARGETS):
            yy = y[idx, j]
            pp = p[idx, j]
            ok = np.isfinite(yy) & np.isfinite(pp)
            n = int(ok.sum())
            if n < min_samples or np.nanstd(yy[ok]) < 1e-8 or np.nanstd(pp[ok]) < 1e-8:
                rho = np.nan
                pval = np.nan
            else:
                res = spearmanr(pp[ok], yy[ok])
                rho = float(res.statistic)
                pval = float(res.pvalue)
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "target": target,
                    "n_samples_used": n,
                    "spearman": rho,
                    "rho_p_value": pval,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(per_method_path, sep="\t", index=False)
    log(f"{method}: saved evaluation {per_method_path}")
    return df


def summarize(all_df, out_dir):
    med = (
        all_df.groupby(["method", "dataset"])["spearman"]
        .median()
        .unstack("dataset")
        .reindex(index=[*ALL_METHODS, "SCP682"])
    )
    na = (
        all_df.assign(is_na=all_df["spearman"].isna())
        .groupby(["method", "dataset"])["is_na"]
        .mean()
        .unstack("dataset")
        .reindex(index=[*ALL_METHODS, "SCP682"])
    )
    med.to_csv(out_dir / "tables" / "baseline_benchmark_median_matrix.tsv", sep="\t")
    na.to_csv(out_dir / "tables" / "baseline_benchmark_na_ratio.tsv", sep="\t")
    log("median spearman matrix")
    print(med.to_string(), flush=True)
    log("NA ratio matrix")
    print(na.to_string(), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--package-dir", default="/data/lsy/Infinite_Stream/SCP682-22/frozen_release/SCP682_22_paper_package_20260520")
    p.add_argument("--rna-path", default="/data/lsy/Infinite_Stream/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/rna_log2_tpm_paired.parquet")
    p.add_argument("--sample-manifest-path", default="/data/lsy/Infinite_Stream/01_data/multi_omics/processed/pancancer_multi_task_locked_v2/sample_manifest.tsv")
    p.add_argument("--scp682-oof-path", default="/data/lsy/Infinite_Stream/SCP682-main/results/20260523_general_graph_residual_e160/predictions/scp682_general_graph_residual_pseudo_external_phosphosite_best.parquet")
    p.add_argument("--target-list-path", default="/data/lsy/Infinite_Stream/SCP682-main/results/20260523_general_graph_residual_e160/tables/per_site_pseudo_external_spearman_best.tsv")
    p.add_argument("--output-dir", default="/data/lsy/Infinite_Stream/SCP682-main/results/20260525_ml_baseline_benchmark_5fold")
    p.add_argument("--methods", nargs="+", default=ALL_METHODS)
    p.add_argument("--n-genes", type=int, default=5000)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-eval-samples", type=int, default=10)
    p.add_argument("--simple-n-jobs", type=int, default=140)
    p.add_argument("--tree-outer-jobs", type=int, default=35)
    p.add_argument("--tree-inner-jobs", type=int, default=4)
    p.add_argument("--wait-gpu-free-mb", type=int, default=18000)
    p.add_argument("--gpu-poll-seconds", type=int, default=300)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mlp-max-epochs", type=int, default=200)
    p.add_argument("--mlp-patience", type=int, default=10)
    p.add_argument("--mlp-batch-size", type=int, default=128)
    p.add_argument("--max-sites", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    global FOLD_DATA, Y_MATRIX, TARGETS, TARGET_TO_PARENT, RNA_FULL, RNA_GENES, ARGS_GLOBAL
    args = parse_args()
    ARGS_GLOBAL = args
    out_dir = Path(args.output_dir)
    (out_dir / "baseline_predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "01_key_results").mkdir(parents=True, exist_ok=True)
    package_dir = Path(args.package_dir)
    y_df = pd.read_parquet(package_dir / "training_set" / "observed_phosphosite.parquet")
    target_meta = pd.read_csv(package_dir / "training_set" / "phosphosite_target_manifest.tsv", sep="\t")
    meta = pd.read_csv(args.sample_manifest_path, sep="\t")
    if "sample_id" not in meta.columns and "index" in meta.columns:
        meta = meta.rename(columns={"index": "sample_id"})
    meta["sample_id"] = meta["sample_id"].astype(str)
    y_df.index = y_df.index.astype(str)
    sample_ids = [s for s in meta["sample_id"].tolist() if s in y_df.index]
    meta = meta.set_index("sample_id").loc[sample_ids].reset_index()
    y_df = y_df.loc[sample_ids]
    meta["cancer_group"] = make_cancer_groups(meta)
    x_df = load_rna_matrix(args.rna_path, sample_ids)
    x_df = x_df.loc[sample_ids]
    x_df.columns = x_df.columns.astype(str)
    RNA_FULL = x_df.to_numpy(dtype=np.float32, copy=True)
    RNA_GENES = {g: i for i, g in enumerate(x_df.columns.astype(str))}
    target_meta["scp682_site_id"] = target_meta["scp682_site_id"].astype(str)
    TARGET_TO_PARENT = dict(zip(target_meta["scp682_site_id"], target_meta["parent_gene"].astype(str)))
    candidate_targets = [t for t in y_df.columns.astype(str) if int(y_df[t].notna().sum()) >= args.min_eval_samples]
    if args.target_list_path and Path(args.target_list_path).exists():
        target_list_df = pd.read_csv(args.target_list_path, sep="\t")
        if "spearman" in target_list_df.columns:
            target_list_df = target_list_df[np.isfinite(pd.to_numeric(target_list_df["spearman"], errors="coerce"))]
        target_col = "target" if "target" in target_list_df.columns else target_list_df.columns[0]
        target_list = target_list_df[target_col].astype(str).tolist()
        target_keep = set(target_list)
        candidate_targets = [t for t in candidate_targets if t in target_keep]
    if args.scp682_oof_path and Path(args.scp682_oof_path).exists():
        scp = pd.read_parquet(args.scp682_oof_path)
        scp.index = scp.index.astype(str)
        scp.columns = scp.columns.astype(str)
        candidate_targets = [t for t in candidate_targets if t in scp.columns]
    if args.max_sites and args.max_sites > 0:
        candidate_targets = candidate_targets[: args.max_sites]
    TARGETS = candidate_targets
    y_df = y_df[TARGETS]
    Y_MATRIX = y_df.to_numpy(dtype=np.float32)
    log(f"samples={len(sample_ids)} genes={x_df.shape[1]} targets={len(TARGETS)}")
    log("cancer_group counts")
    print(meta["cancer_group"].value_counts().to_string(), flush=True)
    FOLD_DATA = prepare_fold_data(x_df, y_df, meta, out_dir, args.n_genes, args.seed, args.n_splits)
    run_config = vars(args).copy()
    run_config["n_samples"] = len(sample_ids)
    run_config["n_rna_genes"] = int(x_df.shape[1])
    run_config["n_targets"] = len(TARGETS)
    run_config["cancer_group_counts"] = meta["cancer_group"].value_counts().to_dict()
    run_config["scp682_split_note"] = "SCP682 rows use provided OOF prediction matrix; baseline folds are StratifiedKFold by cancer_group with random_state=42."
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False))
    all_eval = []
    for method in args.methods:
        if method in SIMPLE_METHODS or method in TREE_METHODS:
            pred = run_cpu_method(method, args, out_dir, sample_ids)
        elif method == "xgboost":
            pred = run_xgboost(args, out_dir, sample_ids)
        elif method == "mlp":
            pred = run_mlp(args, out_dir, sample_ids)
        elif method == "SCP682":
            continue
        else:
            raise ValueError(f"Unknown method: {method}")
        all_eval.append(evaluate_predictions(method, pred, y_df, meta, out_dir, args.min_eval_samples))
    if args.scp682_oof_path and Path(args.scp682_oof_path).exists():
        scp = pd.read_parquet(args.scp682_oof_path)
        scp.index = scp.index.astype(str)
        scp.columns = scp.columns.astype(str)
        scp = scp.loc[sample_ids, TARGETS]
        all_eval.append(evaluate_predictions("SCP682", scp, y_df, meta, out_dir, args.min_eval_samples))
    if not all_eval:
        log("No methods evaluated")
        return
    final = pd.concat(all_eval, ignore_index=True)
    final_path = out_dir / "01_key_results" / "baseline_benchmark_per_site.tsv"
    final.to_csv(final_path, sep="\t", index=False)
    table_copy = out_dir / "tables" / "baseline_benchmark_per_site.tsv"
    final.to_csv(table_copy, sep="\t", index=False)
    summarize(final, out_dir)
    size_mb = final_path.stat().st_size / 1024 / 1024
    log(f"done rows={len(final)} file={final_path} size_mb={size_mb:.2f}")
    (out_dir / "done.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


if __name__ == "__main__":
    main()
