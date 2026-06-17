from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

try:
    from scipy import stats
except Exception:
    stats = None


TARGETS = {
    "Gefitinib": "EGFR",
    "Erlotinib": "EGFR",
    "Lapatinib": "EGFR;ERBB2",
    "Dasatinib": "ABL1;SRC;LYN;LCK;HCK;KIT;PDGFRB;YES1",
    "Bosutinib": "ABL1;SRC;LYN;HCK",
    "Imatinib": "ABL1;KIT;PDGFRB",
    "Nilotinib": "ABL1;KIT;PDGFRB",
    "Ponatinib": "ABL1;KIT",
    "Trametinib": "MAP2K1;MAP2K2",
    "Selumetinib": "MAP2K1;MAP2K2",
    "dactolisib": "PIK3CA;PIK3CB;MTOR",
    "Bortezomib": "PSMB5",
    "Carfilzomib": "PSMB5",
    "Vorinostat": "HDAC1;HDAC2;HDAC3;HDAC6",
    "vorinostat": "HDAC1;HDAC2;HDAC3;HDAC6",
    "curcumin": "CREBBP;EP300",
}


def clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return np.nan
    den = np.linalg.norm(a[ok]) * np.linalg.norm(b[ok])
    return float(np.dot(a[ok], b[ok]) / den) if den > 0 else np.nan


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return float(pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman"))


def direction(pred, real, eps=1e-8):
    pred = np.asarray(pred, dtype=float)
    real = np.asarray(real, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(real) & (np.abs(real) > eps) & (np.abs(pred) > eps)
    if ok.sum() == 0:
        return np.nan
    return float((np.sign(pred[ok]) == np.sign(real[ok])).mean())


def top_fraction_mask(values, fraction=0.2):
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    out = np.zeros(len(values), dtype=bool)
    n = int(ok.sum())
    if n == 0:
        return out
    k = max(1, int(math.ceil(n * fraction)))
    idx = np.where(ok)[0]
    out[idx[np.argsort(-np.abs(values[idx]))[:k]]] = True
    return out


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    ok = np.isfinite(scores)
    scores = scores[ok]
    labels = labels[ok]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def eval_pred(pred, real):
    pred = np.asarray(pred, dtype=float)
    real = np.asarray(real, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(real)
    resp = top_fraction_mask(real, 0.2)
    pred20 = top_fraction_mask(pred, 0.2)
    overlap = resp & pred20
    k = max(1, int(math.ceil(ok.sum() * 0.2))) if ok.sum() else 1
    signed = ok & (np.abs(pred) > 1e-8)
    return {
        "n_sites": int(ok.sum()),
        "signed_fraction_all": float(signed.sum() / ok.sum()) if ok.sum() else np.nan,
        "signed_fraction_responsive20": float((signed & resp).sum() / resp.sum()) if resp.sum() else np.nan,
        "signed_fraction_predicted20": float((signed & pred20).sum() / pred20.sum()) if pred20.sum() else np.nan,
        "all_cosine": cosine(pred[ok], real[ok]),
        "all_spearman": spearman(pred[ok], real[ok]),
        "all_direction": direction(pred[ok], real[ok]),
        "responsive20_cosine": cosine(pred[resp], real[resp]),
        "responsive20_spearman": spearman(pred[resp], real[resp]),
        "responsive20_direction": direction(pred[resp], real[resp]),
        "predicted20_cosine": cosine(pred[pred20], real[pred20]),
        "predicted20_spearman": spearman(pred[pred20], real[pred20]),
        "predicted20_direction": direction(pred[pred20], real[pred20]),
        "top20_recall": float(overlap.sum() / k),
        "top20_overlap_direction": direction(pred[overlap], real[overlap]) if overlap.any() else np.nan,
        "response_auroc_abs_pred": auroc(np.abs(pred[ok]), resp[ok]),
        "pred_abs": float(np.nanmean(np.abs(pred[ok]))) if ok.any() else np.nan,
        "real_abs": float(np.nanmean(np.abs(real[ok]))) if ok.any() else np.nan,
    }


def normalize_max_abs(mat):
    arr = np.asarray(mat, dtype=np.float32).copy()
    if arr.ndim == 1:
        mx = float(np.nanmax(np.abs(arr))) if np.any(np.isfinite(arr)) else 0.0
        return np.clip(arr / mx, -1.0, 1.0) if mx > 0 else arr
    for i in range(arr.shape[0]):
        mx = float(np.nanmax(np.abs(arr[i]))) if np.any(np.isfinite(arr[i])) else 0.0
        if mx > 0:
            arr[i] = np.clip(arr[i] / mx, -1.0, 1.0)
    return arr


def parse_args():
    default_root = r"D:\data\lsy\vm_lsy_parent\lsy" if os.name == "nt" else "/mnt/d/data/lsy/vm_lsy_parent/lsy"
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("SCP682_DATA_ROOT", default_root))
    ap.add_argument("--package-root", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def ridge_fit_predict(X_train, Y_train, X_test, alpha):
    XtX = X_train @ X_train.T
    A = XtX + alpha * np.eye(XtX.shape[0], dtype=np.float32)
    coef_dual = np.linalg.solve(A, Y_train)
    return (X_test @ X_train.T) @ coef_dual


def select_ridge_alpha(X, Y, mask):
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    n = len(X)
    folds = np.arange(n) % 5
    rows = []
    for alpha in alphas:
        vals = []
        for fold in range(5):
            train_idx = folds != fold
            val_idx = folds == fold
            pred = ridge_fit_predict(X[train_idx], Y[train_idx], X[val_idx], alpha)
            val_rows = np.where(val_idx)[0]
            for j, row_i in enumerate(val_rows):
                vals.append(cosine(pred[j, mask[row_i]], Y[row_i, mask[row_i]]))
        rows.append({"alpha": alpha, "mean_cv_cosine": float(np.nanmean(vals))})
    table = pd.DataFrame(rows)
    best = float(table.sort_values("mean_cv_cosine", ascending=False).iloc[0]["alpha"])
    return best, table


def vector_cosine_to_matrix(query, matrix):
    q = np.asarray(query, dtype=float)
    mat = np.asarray(matrix, dtype=float)
    qn = np.linalg.norm(q)
    mn = np.linalg.norm(mat, axis=1)
    den = np.maximum(qn * mn, 1e-12)
    return (mat @ q) / den


def baseline_similarity(query_base, query_valid, train_base, train_valid):
    sims = np.full(train_base.shape[0], np.nan, dtype=float)
    for i in range(train_base.shape[0]):
        ok = query_valid & train_valid[i] & np.isfinite(query_base) & np.isfinite(train_base[i])
        sims[i] = cosine(query_base[ok], train_base[i, ok]) if ok.sum() >= 2 else np.nan
    return sims


def summarize(metrics):
    rows = []
    numeric = [c for c in metrics.columns if pd.api.types.is_numeric_dtype(metrics[c])]
    for model, sub in metrics.groupby("model", sort=False):
        rec = {"model": model, "n_comparisons": int(sub["comparison_id"].nunique())}
        for col in numeric:
            if col in {"comparison_index"}:
                continue
            vals = sub[col].dropna().to_numpy(dtype=float)
            rec[f"{col}_mean"] = float(np.nanmean(vals)) if len(vals) else np.nan
            if len(vals) > 1:
                se = float(np.nanstd(vals, ddof=1) / math.sqrt(len(vals)))
                rec[f"{col}_ci95_low"] = rec[f"{col}_mean"] - 1.96 * se
                rec[f"{col}_ci95_high"] = rec[f"{col}_mean"] + 1.96 * se
            else:
                rec[f"{col}_ci95_low"] = np.nan
                rec[f"{col}_ci95_high"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def paired_tests(metrics, reference="ppko_v10b"):
    if stats is None:
        return pd.DataFrame()
    test_metrics = [
        "all_cosine",
        "all_direction",
        "responsive20_cosine",
        "responsive20_direction",
        "predicted20_cosine",
        "predicted20_direction",
        "top20_recall",
        "response_auroc_abs_pred",
    ]
    out = []
    ref = metrics[metrics["model"].eq(reference)].set_index("comparison_id")
    for model in sorted(set(metrics["model"]) - {reference}):
        sub = metrics[metrics["model"].eq(model)].set_index("comparison_id")
        common = ref.index.intersection(sub.index)
        for metric in test_metrics:
            a = ref.loc[common, metric].astype(float)
            b = sub.loc[common, metric].astype(float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 5:
                p = np.nan
                stat = np.nan
            else:
                try:
                    stat, p = stats.wilcoxon(a[ok], b[ok], alternative="greater", zero_method="wilcox")
                except Exception:
                    stat, p = np.nan, np.nan
            out.append(
                {
                    "reference": reference,
                    "model": model,
                    "metric": metric,
                    "n_pairs": int(ok.sum()),
                    "reference_mean": float(np.nanmean(a[ok])) if ok.sum() else np.nan,
                    "baseline_mean": float(np.nanmean(b[ok])) if ok.sum() else np.nan,
                    "mean_difference": float(np.nanmean(a[ok] - b[ok])) if ok.sum() else np.nan,
                    "wilcoxon_greater_p": float(p) if np.isfinite(p) else np.nan,
                    "wilcoxon_statistic": float(stat) if np.isfinite(stat) else np.nan,
                }
            )
    return pd.DataFrame(out)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    package_root = Path(args.package_root) if args.package_root else data_root / "SCP682_PPKO_V10B_transferable"
    scripts_dir = package_root / "scripts"
    sys.path.insert(0, str(scripts_dir))

    from pretrain_v10b_strong300 import AttentionPriorManifoldV10, build_global_signed_inputs, row_normalize

    graph_dir = data_root / "01_data" / "pathway_prior" / "intermediate" / "global_phosphoprotein_heterograph_v10_measured_string700_top50"
    train_dir = data_root / "01_data" / "single_cell" / "intermediate" / "phospho_perturb" / "decryptm_comparison_delta_v8"
    p100_dir = data_root / "01_data" / "single_cell" / "intermediate" / "phospho_perturb" / "lincs_p100_comparison_delta_v7"
    map_dir = data_root / "02_results" / "single_cell" / "20260519_scp682_ppko_1_decryptm_network_v8_full_p100_shared_site_validation" / "tables"
    model_path = package_root / "models" / "scp682_ppko_v10b_strong300_best.pt"
    out_dir = Path(args.output_dir) if args.output_dir else data_root / "02_results" / "single_cell" / "20260602_scp682_ppko_v10b_p100_published_baselines"
    tables = out_dir / "tables"
    reports = out_dir / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    sites = pd.DataFrame(ckpt["sites"])
    proteins = pd.DataFrame(ckpt["proteins"])
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = AttentionPriorManifoldV10(len(sites), len(proteins), hidden=ckpt["args"]["hidden"], latent=ckpt["args"]["latent"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train_delta = np.load(train_dir / "arrays" / "delta_matrix.npy").astype(np.float32)
    train_base = np.load(train_dir / "arrays" / "baseline_matrix.npy").astype(np.float32)
    train_valid = np.load(train_dir / "arrays" / "valid_mask.npy").astype(bool)
    train_comp = pd.read_csv(train_dir / "tables" / "comparison_table.tsv", sep="\t")

    p_delta = np.load(p100_dir / "arrays" / "delta_matrix.npy").astype(np.float32)
    p_base = np.load(p100_dir / "arrays" / "baseline_matrix.npy").astype(np.float32)
    p_valid = np.load(p100_dir / "arrays" / "valid_mask.npy").astype(bool)
    p_comp = pd.read_csv(p100_dir / "tables" / "comparison_table.tsv", sep="\t")
    mapping = pd.read_csv(map_dir / "p100_to_decryptm_shared_site_map.tsv", sep="\t")

    pidx = mapping["p100_target_index"].to_numpy(dtype=int)
    didx = mapping["decryptm_target_index"].to_numpy(dtype=int)
    grouped_mapping = list(mapping.groupby("p100_target_index", sort=False))

    generic = np.zeros(len(sites), dtype=np.float32)
    for site_i in range(len(sites)):
        vals = train_base[train_valid[:, site_i], site_i]
        generic[site_i] = float(np.nanmedian(vals)) if len(vals) else 0.0

    signed_pp = row_normalize(np.load(graph_dir / "arrays" / "signed_protein_protein_matrix.npy").astype(np.float32))
    unsigned_pp = row_normalize(np.load(graph_dir / "arrays" / "unsigned_protein_protein_matrix.npy").astype(np.float32))
    signed_rs = row_normalize(np.load(graph_dir / "arrays" / "signed_regulator_site_matrix.npy").astype(np.float32))

    train_seed, train_pc, train_heat, _ = build_global_signed_inputs(train_comp, graph_dir)
    train_y = np.where(train_valid, train_delta, 0.0).astype(np.float32)
    train_features = np.concatenate([train_seed, np.ones((len(train_seed), 1), dtype=np.float32)], axis=1)
    best_alpha, alpha_table = select_ridge_alpha(train_features, train_y, train_valid)
    alpha_table.to_csv(tables / "ridge_alpha_cv.tsv", sep="\t", index=False)

    methods = [
        {
            "model": "ppko_v10b",
            "category": "full model",
            "description": "Frozen V10B PPKO model using baseline, signed target context, graph prior and learned decoder.",
        },
        {
            "model": "zero_vector",
            "category": "null",
            "description": "Predicts zero phosphosite delta for every site.",
        },
        {
            "model": "direct_kinase_substrate",
            "category": "KSEA forward",
            "description": "Signed target seed projected directly through regulator-to-site edges without protein-protein propagation.",
        },
        {
            "model": "graph_heat_diffusion",
            "category": "graph diffusion",
            "description": "Two-step signed/unsigned protein graph heat diffusion followed by signed regulator-to-site projection; no learned parameters.",
        },
        {
            "model": "rwr_signed_graph",
            "category": "graph diffusion",
            "description": "Random-walk-with-restart on the signed protein graph followed by signed regulator-to-site projection; no learned parameters.",
        },
        {
            "model": "ridge_signed_target",
            "category": "linear regression",
            "description": f"Ridge regression from signed target vector plus intercept to phosphosite delta; alpha selected inside decryptM CV, alpha={best_alpha:g}.",
        },
        {
            "model": "retrieval_target",
            "category": "nearest neighbor",
            "description": "Nearest decryptM comparison by target protein-context cosine; uses the retrieved training delta.",
        },
        {
            "model": "retrieval_target_baseline",
            "category": "nearest neighbor",
            "description": "Nearest decryptM comparison by average of target protein-context cosine and baseline phosphosite cosine; uses the retrieved training delta.",
        },
    ]
    pd.DataFrame(methods).to_csv(tables / "baseline_method_descriptions.tsv", sep="\t", index=False)

    rows = []
    for comp_i, comp in p_comp.iterrows():
        perturbation = clean(comp.get("perturbation", ""))
        target_genes = TARGETS.get(perturbation, clean(comp.get("target_genes", "")))
        if not target_genes:
            continue
        use = p_valid[comp_i, pidx]
        if int(use.sum()) < 5:
            continue
        comparison_id = clean(comp.get("comparison_id", f"p100_delta_{comp_i:05d}"))
        query_comp = pd.DataFrame([{"target_genes": target_genes, "action_type": clean(comp.get("action_type", "inhibition")) or "inhibition"}])
        seed, pc, heat, _ = build_global_signed_inputs(query_comp, graph_dir)
        seed = seed[0]
        pc = pc[0]
        heat = heat[0]

        base_vec = generic.copy()
        valid_vec = np.zeros(len(sites), dtype=bool)
        base_vec[didx[use]] = p_base[comp_i, pidx[use]]
        valid_vec[didx[use]] = True

        with torch.no_grad():
            ppko_pred, _, _, _, _, _ = model(
                torch.as_tensor(base_vec, dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(valid_vec, dtype=torch.bool, device=device).unsqueeze(0),
                torch.as_tensor(pc, dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(heat, dtype=torch.float32, device=device).unsqueeze(0),
            )
        ppko_full = ppko_pred.detach().cpu().numpy()[0]
        zero_full = np.zeros(len(sites), dtype=np.float32)
        direct_full = normalize_max_abs(seed @ signed_rs)
        heat_full = heat

        rwr = seed.copy()
        restart = 0.35
        for _ in range(40):
            rwr = restart * seed + (1.0 - restart) * (rwr @ signed_pp)
        rwr_full = normalize_max_abs(rwr @ signed_rs)

        ridge_pred = ridge_fit_predict(train_features, train_y, np.concatenate([seed, [1.0]], dtype=np.float32)[None, :], best_alpha)[0]

        target_sim = vector_cosine_to_matrix(pc, train_pc)
        best_target = int(np.nanargmax(target_sim))
        retrieval_target_full = train_delta[best_target].copy()

        base_sims = baseline_similarity(base_vec, valid_vec, train_base, train_valid)
        combo = 0.5 * target_sim + 0.5 * np.nan_to_num(base_sims, nan=np.nanmean(base_sims))
        if not np.isfinite(combo).any():
            best_combo = best_target
        else:
            best_combo = int(np.nanargmax(combo))
        retrieval_target_baseline_full = train_delta[best_combo].copy()

        full_preds = {
            "ppko_v10b": ppko_full,
            "zero_vector": zero_full,
            "direct_kinase_substrate": direct_full,
            "graph_heat_diffusion": heat_full,
            "rwr_signed_graph": rwr_full,
            "ridge_signed_target": ridge_pred,
            "retrieval_target": retrieval_target_full,
            "retrieval_target_baseline": retrieval_target_baseline_full,
        }

        real_unique = []
        pred_unique = {name: [] for name in full_preds}
        for p100_target_index, sub in grouped_mapping:
            p100_target_index = int(p100_target_index)
            if not p_valid[comp_i, p100_target_index]:
                continue
            decryptm_indices = sub["decryptm_target_index"].astype(int).to_numpy()
            if len(decryptm_indices) == 0:
                continue
            real_unique.append(float(p_delta[comp_i, p100_target_index]))
            for name, vec in full_preds.items():
                pred_unique[name].append(float(np.nanmean(vec[decryptm_indices])))

        real_unique = np.asarray(real_unique, dtype=float)
        for name, vals in pred_unique.items():
            pred_arr = np.asarray(vals, dtype=float)
            metric = eval_pred(pred_arr, real_unique)
            metric.update(
                {
                    "model": name,
                    "comparison_id": comparison_id,
                    "comparison_index": int(comp_i),
                    "perturbation": perturbation,
                    "target_genes": target_genes,
                    "n_mapping_rows": int(use.sum()),
                    "retrieved_target_comparison": clean(train_comp.iloc[best_target].get("comparison_id", best_target)) if name == "retrieval_target" else "",
                    "retrieved_target_baseline_comparison": clean(train_comp.iloc[best_combo].get("comparison_id", best_combo)) if name == "retrieval_target_baseline" else "",
                }
            )
            rows.append(metric)

    metrics = pd.DataFrame(rows)
    summary = summarize(metrics)
    paired = paired_tests(metrics)
    metrics.to_csv(tables / "p100_v10b_published_baseline_comparison_metrics.tsv", sep="\t", index=False)
    summary.to_csv(tables / "p100_v10b_published_baseline_comparison_summary.tsv", sep="\t", index=False)
    paired.to_csv(tables / "p100_v10b_published_baseline_paired_wilcoxon_vs_ppko.tsv", sep="\t", index=False)

    report = {
        "model": "SCP682-PPKO V10B strong300",
        "data_root": str(data_root),
        "output_dir": str(out_dir),
        "evaluation": "P100 all-drug shared phosphosite validation, unique P100 site deduplication",
        "n_comparisons": int(metrics["comparison_id"].nunique()),
        "n_models": int(metrics["model"].nunique()),
        "ridge_alpha": best_alpha,
        "summary": summary.to_dict(orient="records"),
    }
    (reports / "published_baseline_comparison_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
