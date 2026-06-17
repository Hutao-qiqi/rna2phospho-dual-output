#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold


ROOT = Path("/data/lsy/Infinite_Stream")
TCPA_DIR = ROOT / "01_data/tcga_tcpa/processed/tcpa_32_project_rna_rppa_20260501"
BASE_SCRIPT = ROOT / "03_code/external_validation/phosphoproteomics/deploy_v38_luad_cas_and_evaluate_true_phosphosite_20260502.py"
CURRENT_RELEASE = ROOT / "SCP682/frozen_release/SCP682_main_exact_scnet_gnn_20260522"
TRAIN_SCRIPT = CURRENT_RELEASE / "scripts/train_scp682_exact_scnet_gnn_v1.py"
TRAIN_RNA = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2/rna_log2_tpm_paired.parquet"
OUT = ROOT / "SCP682-main/results/20260526_tcga_tcpa_overlap_scp682"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    return out


def sample_median_center(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    med = df.median(axis=1, skipna=True)
    return df.sub(med, axis=0).astype(np.float32), med


def read_tcga_rna() -> pd.DataFrame:
    raw = pd.read_parquet(TCPA_DIR / "matrices/X_tcpa_32.symbols.parquet")
    sample_cols = [c for c in raw.columns if c != "gene_symbol"]
    raw["gene_symbol"] = raw["gene_symbol"].astype(str)
    raw = raw.loc[raw["gene_symbol"].ne("") & raw["gene_symbol"].ne("nan")]
    raw[sample_cols] = raw[sample_cols].apply(pd.to_numeric, errors="coerce")
    mat = raw.groupby("gene_symbol", sort=False)[sample_cols].median(numeric_only=True)
    x = mat.T
    x.index.name = "sample_id"
    return clean_numeric(x)


def read_tcpa() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(TCPA_DIR / "tables/tcpa_32_antibody_panel.tsv", sep="\t")
    manifest = pd.read_csv(TCPA_DIR / "tables/tcpa_32_sample_manifest.tsv", sep="\t")
    y = pd.read_parquet(TCPA_DIR / "matrices/Y_tcpa_32.rppa.parquet")
    if "rna_sample_id" in y.columns:
        y = y.set_index("rna_sample_id")
    y.index = y.index.astype(str)
    y.columns = y.columns.astype(str)
    y = y.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    return y, panel, manifest


def input_qc(rna: pd.DataFrame, y: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    arr = rna.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    common = sorted(set(rna.index).intersection(y.index))
    return {
        "tcga_rna_shape": [int(rna.shape[0]), int(rna.shape[1])],
        "tcpa_rppa_shape": [int(y.shape[0]), int(y.shape[1])],
        "matched_rna_rppa_samples": int(len(common)),
        "sample_manifest_rows": int(manifest.shape[0]),
        "rna_value_quantiles": {
            str(q): float(np.nanpercentile(finite, q)) for q in [0, 1, 50, 95, 99, 100]
        },
        "rna_finite_fraction": float(np.isfinite(arr).mean()),
    }


def split_gene_sites(antibody: str) -> tuple[str | None, list[str]]:
    text = str(antibody).upper()
    m = re.search(r"P[STY]\d+", text)
    if not m:
        return None, []
    gene_token = text[: m.start()]
    site_part = text[m.start():]
    sites = [f"{aa}{pos}" for aa, pos in re.findall(r"P?([STY])(\d+)", site_part)]
    return gene_token, sites


ALIASES: dict[str, list[str]] = {
    "4EBP1": ["EIF4EBP1"],
    "X4EBP1": ["EIF4EBP1"],
    "S6": ["RPS6"],
    "P70S6K": ["RPS6KB1"],
    "P90RSK": ["RPS6KA1", "RPS6KA3"],
    "MAPK": ["MAPK3", "MAPK1"],
    "P38": ["MAPK14"],
    "P38A": ["MAPK14"],
    "JNK": ["MAPK8", "MAPK9", "MAPK10"],
    "MEK1": ["MAP2K1"],
    "MEK2": ["MAP2K2"],
    "AKT": ["AKT1"],
    "AMPKALPHA": ["PRKAA1"],
    "AMPKA2": ["PRKAA2"],
    "BCATENIN": ["CTNNB1"],
    "BETACATENIN": ["CTNNB1"],
    "HER2": ["ERBB2"],
    "ERALPHA": ["ESR1"],
    "FRS2ALPHA": ["FRS2"],
    "HSP27": ["HSPB1"],
    "NFKBP65": ["RELA"],
    "P27": ["CDKN1B"],
    "PDK1": ["PDPK1"],
    "PRAS40": ["AKT1S1"],
    "TUBERIN": ["TSC2"],
    "YB1": ["YBX1"],
    "CJUN": ["JUN"],
    "CABL": ["ABL1"],
    "SHC": ["SHC1"],
    "SHP2": ["PTPN11"],
    "MYOSINIIA": ["MYH9"],
    "AURORAABC": ["AURKA", "AURKB", "AURKC"],
    "GSK3": ["GSK3B"],
    "GSK3ALPHABETA": ["GSK3A", "GSK3B"],
    "PKCABI": ["PRKCA", "PRKCB"],
    "PKCALPHA": ["PRKCA"],
    "PKCDELTA": ["PRKCD"],
    "PKCPANBETAII": ["PRKCB"],
    "RB": ["RB1"],
    "RPA32": ["RPA2"],
}


SITE_REMAP: dict[tuple[str, tuple[str, ...]], list[tuple[str, tuple[str, ...], str]]] = {
    ("AKT", ("S473",)): [("AKT1", ("S473",), "alias_exact")],
    ("AKT2", ("S474",)): [("AKT2", ("S474",), "alias_exact")],
    ("AKT", ("T308",)): [("AKT1", ("T308",), "alias_exact")],
    ("MAPK", ("T202", "Y204")): [("MAPK3", ("T202", "Y204"), "erk1_exact")],
    ("P38", ("T180", "Y182")): [("MAPK14", ("T180", "Y182"), "p38_exact")],
    ("S6", ("S235", "S236")): [("RPS6", ("S235", "S236"), "alias_exact")],
    ("S6", ("S240", "S244")): [("RPS6", ("S240", "S244"), "alias_exact")],
    ("4EBP1", ("S65",)): [("EIF4EBP1", ("S65",), "alias_exact")],
    ("4EBP1", ("T37", "T46")): [("EIF4EBP1", ("T37", "T46"), "alias_exact")],
    ("4EBP1", ("T70",)): [("EIF4EBP1", ("T70",), "alias_exact")],
    ("PRAS40", ("T246",)): [("AKT1S1", ("T246",), "alias_exact")],
    ("RB", ("S807", "S811")): [("RB1", ("S807", "S811"), "alias_exact")],
    ("HER2", ("Y1248",)): [("ERBB2", ("Y1248",), "alias_exact")],
    ("JNK", ("T183", "Y185")): [("MAPK8", ("T183", "Y185"), "jnk_family"), ("MAPK9", ("T183", "Y185"), "jnk_family")],
    ("GSK3ALPHABETA", ("S21", "S9")): [("GSK3A", ("S21",), "split_pan_antibody"), ("GSK3B", ("S9",), "split_pan_antibody")],
}


def make_site_id(gene: str, sites: tuple[str, ...]) -> str:
    return f"{gene}|{'_'.join(sites)}"


def build_antibody_site_mapping(panel: pd.DataFrame, site_manifest: pd.DataFrame) -> pd.DataFrame:
    site_set = set(site_manifest["scp682_site_id"].astype(str))
    parent_to_sites = {
        g: set(sub["residue_site"].astype(str))
        for g, sub in site_manifest.groupby("parent_gene", sort=False)
    }
    rows = []
    phospho = panel.loc[panel["type"].astype(str).eq("phospho")].copy()
    for _, row in phospho.iterrows():
        antibody = str(row["antibody"])
        gene_token, sites = split_gene_sites(antibody)
        candidates: list[tuple[str, tuple[str, ...], str]] = []
        if gene_token and sites:
            key = (gene_token, tuple(sites))
            candidates.extend(SITE_REMAP.get(key, []))
            for gene in ALIASES.get(gene_token, [gene_token]):
                candidates.append((gene, tuple(sites), "direct_or_alias"))
        seen = set()
        matched = []
        for gene, site_tuple, rule in candidates:
            target = make_site_id(gene, site_tuple)
            if target in site_set and target not in seen:
                matched.append((target, rule))
                seen.add(target)
            if len(site_tuple) > 1:
                for single in site_tuple:
                    target_single = make_site_id(gene, (single,))
                    if target_single in site_set and target_single not in seen:
                        matched.append((target_single, rule + "_single_site"))
                        seen.add(target_single)
        if matched:
            for target, rule in matched:
                target_row = site_manifest.loc[site_manifest["scp682_site_id"].astype(str).eq(target)].iloc[0]
                rows.append({
                    "antibody": antibody,
                    "antibody_type": row["type"],
                    "antibody_n_observed": int(row.get("n_observed", 0)),
                    "parsed_gene_token": gene_token,
                    "parsed_sites": "_".join(sites),
                    "scp682_site_id": target,
                    "parent_gene": str(target_row["parent_gene"]),
                    "residue_site": str(target_row["residue_site"]),
                    "match_rule": rule,
                    "matched": True,
                })
        else:
            rows.append({
                "antibody": antibody,
                "antibody_type": row["type"],
                "antibody_n_observed": int(row.get("n_observed", 0)),
                "parsed_gene_token": gene_token,
                "parsed_sites": "_".join(sites),
                "scp682_site_id": "",
                "parent_gene": "",
                "residue_site": "",
                "match_rule": "no_scp682_site_match",
                "matched": False,
            })
    out = pd.DataFrame(rows)
    out["site_is_available_in_manifest"] = out["scp682_site_id"].astype(str).isin(site_set)
    out["parent_has_any_scp682_site"] = out["parsed_gene_token"].map(
        lambda x: bool(x and any(g in parent_to_sites for g in ALIASES.get(str(x), [str(x)])))
    )
    return out


PROJECT_TO_CANCER = {
    "TCGA-BRCA": ["BRCA"],
    "TCGA-LUAD": ["LUAD"],
    "TCGA-LUSC": ["LSCC", "LUSC", "LUAD"],
    "TCGA-HNSC": ["HNSCC", "HNSC"],
    "TCGA-GBM": ["GBM"],
    "TCGA-LGG": ["GBM"],
    "TCGA-OV": ["OV"],
    "TCGA-UCEC": ["UCEC"],
    "TCGA-KIRC": ["CCRCC", "ccRCC", "KIRC"],
    "TCGA-KIRP": ["CCRCC", "ccRCC", "KIRP"],
    "TCGA-KICH": ["CCRCC", "ccRCC", "KICH"],
    "TCGA-PAAD": ["PDAC", "PAAD"],
    "TCGA-STAD": ["STAD"],
    "TCGA-COAD": ["COAD", "CRC"],
    "TCGA-READ": ["READ", "CRC", "COAD"],
    "TCGA-LIHC": ["HCC", "LIHC"],
}


def resolve_project_context(project: str, levels: list[str], fallback: str) -> str:
    for candidate in PROJECT_TO_CANCER.get(str(project), [str(project).replace("TCGA-", "")]):
        if candidate in levels:
            return candidate
    return fallback if fallback in levels else levels[0]


def patch_context_arrays(base, sample_to_project: dict[str, str], default_cancer: str = "LUAD", default_study: str = "PDC000149") -> None:
    def make_context_arrays(index: pd.Index, ckpt: dict) -> tuple[np.ndarray, np.ndarray]:
        cancer_levels = [str(x) for x in ckpt["cancer_levels"]]
        study_levels = [str(x) for x in ckpt["study_levels"]]
        cancer_map = {c: i for i, c in enumerate(cancer_levels)}
        study_map = {s: i for i, s in enumerate(study_levels)}
        fallback_cancer = default_cancer if default_cancer in cancer_map else cancer_levels[0]
        fallback_study = default_study if default_study in study_map else study_levels[0]
        cancer = []
        study = []
        for sample in index.astype(str):
            key = sample.split("::", 1)[-1]
            project = sample_to_project.get(key, "")
            cancer_name = resolve_project_context(project, cancer_levels, fallback_cancer)
            cancer.append(cancer_map[cancer_name])
            study.append(study_map[fallback_study])
        return np.asarray(cancer, dtype=np.int64), np.asarray(study, dtype=np.int64)

    base.make_context_arrays = make_context_arrays


def export_light_candidates_tcga(base, data: dict, x_ext: pd.DataFrame, v3_total_ext: pd.DataFrame, sample_to_project: dict[str, str], args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    y = data["phospho"]
    total = data["total"]
    x = data["x"]
    feature_idx = {g: i for i, g in enumerate(data["feature_names"])}
    ext_feature_idx = {g: i for i, g in enumerate(x_ext.columns)}
    total_idx = {g: i for i, g in enumerate(data["total_names"])}
    ext_total_idx = {g: i for i, g in enumerate(v3_total_ext.columns)}
    site_genes = [str(s).split("|", 1)[0] for s in data["phospho_names"]]
    out = {name: np.full((len(x_ext), len(data["phospho_names"])), np.nan, dtype=np.float32) for name in ["parent_only", "rna_direct", "ridge_direct"]}
    counts = {name: np.zeros((len(x_ext), len(data["phospho_names"])), dtype=np.float32) for name in out}
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=20260502).split(x, data["cancer_ids"]))
    cancer_series = pd.Series(data["manifest"]["cancer_label"]).astype("category")
    cancer_levels = list(cancer_series.cat.categories.astype(str))
    cancer_onehot = pd.get_dummies(cancer_series).to_numpy(dtype=np.float32)
    fallback_cancer = "LUAD" if "LUAD" in cancer_levels else cancer_levels[0]
    ext_cancer_codes = []
    for sample in x_ext.index.astype(str):
        project = sample_to_project.get(sample.split("::", 1)[-1], "")
        ext_cancer_codes.append(cancer_levels.index(resolve_project_context(project, cancer_levels, fallback_cancer)))
    ext_cancer_onehot = np.zeros((len(x_ext), cancer_onehot.shape[1]), dtype=np.float32)
    ext_cancer_onehot[np.arange(len(x_ext)), np.asarray(ext_cancer_codes, dtype=int)] = 1.0

    for fold, (train_idx, _) in enumerate(folds, start=1):
        fold_pred = {name: np.full_like(out[name], np.nan) for name in out}
        for j, gene in enumerate(site_genes):
            ok_y = np.isfinite(y[train_idx, j])
            if ok_y.sum() < 10:
                continue
            if gene in total_idx and gene in ext_total_idx:
                tj = total_idx[gene]
                ok = ok_y & np.isfinite(total[train_idx, tj])
                if ok.sum() >= 10:
                    xv = total[train_idx, tj][ok]
                    yv = y[train_idx, j][ok]
                    a = np.cov(xv, yv, bias=True)[0, 1] / max(float(np.var(xv)), 1e-6)
                    b = float(yv.mean() - a * xv.mean())
                    fold_pred["parent_only"][:, j] = (a * v3_total_ext.iloc[:, ext_total_idx[gene]].to_numpy(dtype=np.float32) + b).astype(np.float32)
            if gene in feature_idx and gene in ext_feature_idx:
                gi = feature_idx[gene]
                egi = ext_feature_idx[gene]
                xv = x[train_idx, gi]
                ok = ok_y & np.isfinite(xv)
                if ok.sum() >= 10:
                    xx = xv[ok]
                    yy = y[train_idx, j][ok]
                    a = np.cov(xx, yy, bias=True)[0, 1] / max(float(np.var(xx)), 1e-6)
                    b = float(yy.mean() - a * xx.mean())
                    fold_pred["rna_direct"][:, j] = (a * x_ext.iloc[:, egi].to_numpy(dtype=np.float32) + b).astype(np.float32)
                    features_train = [x[:, gi]]
                    features_ext = [x_ext.iloc[:, egi].to_numpy(dtype=np.float32)]
                    if gene in total_idx and gene in ext_total_idx:
                        features_train.append(total[:, total_idx[gene]])
                        features_ext.append(v3_total_ext.iloc[:, ext_total_idx[gene]].to_numpy(dtype=np.float32))
                    features_train.extend([cancer_onehot[:, k] for k in range(cancer_onehot.shape[1])])
                    features_ext.extend([ext_cancer_onehot[:, k] for k in range(ext_cancer_onehot.shape[1])])
                    X = np.vstack(features_train).T.astype(np.float32)
                    Xe = np.vstack(features_ext).T.astype(np.float32)
                    ok = ok_y & np.all(np.isfinite(X[train_idx]), axis=1)
                    if ok.sum() >= 20:
                        col_mean = np.nanmean(X[train_idx][ok], axis=0).astype(np.float32)
                        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
                        Xe = np.where(np.isfinite(Xe), Xe, col_mean[None, :]).astype(np.float32)
                        model = Ridge(alpha=args.ridge_alpha)
                        model.fit(X[train_idx][ok], y[train_idx, j][ok])
                        fold_pred["ridge_direct"][:, j] = model.predict(Xe).astype(np.float32)
        for name in out:
            ok = np.isfinite(fold_pred[name])
            out[name] = np.where(ok, np.nan_to_num(out[name], nan=0.0) + np.where(ok, fold_pred[name], 0.0), out[name])
            counts[name] += ok.astype(np.float32)
        print(f"light candidates fold {fold} done", flush=True)
    frames = {}
    for name, mat in out.items():
        cnt = counts[name]
        mat = np.divide(mat, cnt, out=np.full_like(mat, np.nan), where=cnt > 0)
        frames[name] = pd.DataFrame(mat, index=x_ext.index, columns=data["phospho_names"]).astype(np.float32)
    return frames


def build_baseline(args: argparse.Namespace, out: Path, rna: pd.DataFrame, manifest: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    pred_path = out / "predictions/tcga_baseline_mean_phosphosite.parquet"
    if pred_path.exists() and not args.force_baseline:
        return clean_numeric(pd.read_parquet(pred_path))
    base = import_module(BASE_SCRIPT, "scp682_external_base_for_tcga")
    sample_to_project = dict(zip(manifest["rna_sample_id"].astype(str), manifest["project_id"].astype(str)))
    patch_context_arrays(base, sample_to_project)
    data = base.load_base_data(device)
    x_ext = base.build_external_feature_frame(rna, device)
    x_ext.to_parquet(out / "intermediate/tcga_rna_plus_vae_latent.parquet")
    v3_pred, v3_total = base.predict_v3_family(
        "v3_train_oof",
        base.V3_SCRIPT,
        base.V3_MODEL_DIR,
        "20260502_cptac_parent_residual_kinase_cvae_experimental_v3",
        data,
        x_ext,
        device,
        args,
    )
    _ = v3_pred
    v3_total.to_parquet(out / "predictions/tcga_v3_parent_total_protein.parquet")
    light = export_light_candidates_tcga(base, data, x_ext, v3_total, sample_to_project, args)
    for name, pred in light.items():
        pred.to_parquet(out / f"predictions/tcga_candidate_{name}_phosphosite.parquet")
    common = sorted(set.intersection(*[set(v.columns) for v in light.values()]))
    baseline = sum(light[name][common] for name in ["parent_only", "ridge_direct", "rna_direct"]) / 3.0
    baseline.index.name = "sample_id"
    baseline.astype(np.float32).to_parquet(pred_path)
    return baseline.astype(np.float32)


def topk_softmax_attention(query: np.ndarray, key: np.ndarray, k: int, temperature: float) -> np.ndarray:
    sim = query @ key.T
    k = max(1, min(k, sim.shape[1]))
    top = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    out = np.zeros_like(sim, dtype=np.float32)
    rows = np.arange(sim.shape[0])[:, None]
    vals = sim[rows, top] / max(float(temperature), 1e-6)
    vals = vals - vals.max(axis=1, keepdims=True)
    weights = np.exp(vals).astype(np.float32)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    out[rows, top] = weights
    return out


def build_exact_model(train_mod, ckpt: dict, device: torch.device):
    ck_args = ckpt.get("args", {})
    model = train_mod.SCP682ExactScNETResidual(
        n_sites=len(ckpt["targets"]),
        n_samples=len(ckpt["samples"]),
        shrinkage=float(ck_args.get("shrinkage", 0.3)),
        hidden=int(ck_args.get("hidden", 160)),
        latent=int(ck_args.get("latent", 64)),
        inter_dim=int(ck_args.get("inter_dim", 192)),
        embd_dim=int(ck_args.get("embd_dim", 64)),
        num_layers=int(ck_args.get("num_layers", 2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def l2_sample_block(train_mod, x: np.ndarray) -> np.ndarray:
    if hasattr(train_mod, "l2_sample_block"):
        return train_mod.l2_sample_block(x)
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(axis=1, keepdims=True)
    norm = np.sqrt((x * x).sum(axis=1, keepdims=True)).clip(min=1e-6)
    return x / norm


def align_by_sample_key(df: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
    direct = df.reindex(target_index)
    if direct.notna().any(axis=None):
        return direct
    tmp = df.copy()
    tmp["_sample_key"] = [str(x).split("::", 1)[-1] for x in tmp.index]
    tmp = tmp.drop_duplicates("_sample_key").set_index("_sample_key")
    keys = [str(x).split("::", 1)[-1] for x in target_index]
    out = tmp.reindex(keys)
    out.index = target_index
    return out


def make_rna_context_local(train_rna: pd.DataFrame, train_samples: pd.Index, external_rna: pd.DataFrame, external_samples: pd.Index, n_genes: int) -> tuple[pd.DataFrame, list[str]]:
    train_rna = clean_numeric(train_rna)
    external_rna = clean_numeric(external_rna)
    train_aligned = align_by_sample_key(train_rna, train_samples)
    external_aligned = align_by_sample_key(external_rna, external_samples)
    common = [g for g in train_aligned.columns if g in external_aligned.columns]
    train_block = train_aligned[common].apply(pd.to_numeric, errors="coerce")
    var = train_block.var(axis=0, skipna=True).sort_values(ascending=False)
    genes = list(var.index[: min(n_genes, len(var))])
    mean = train_block[genes].mean(axis=0, skipna=True)
    std = train_block[genes].std(axis=0, skipna=True).replace(0, np.nan).fillna(1.0)
    train_ctx = ((train_block[genes] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-6, 6)
    ext_block = external_aligned[genes].apply(pd.to_numeric, errors="coerce")
    ext_ctx = ((ext_block - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-6, 6)
    ctx = pd.concat([train_ctx, ext_ctx], axis=0)
    return ctx.astype(np.float32), genes


def predict_exact_scnet(args: argparse.Namespace, out: Path, rna: pd.DataFrame, baseline_ext: pd.DataFrame, overlap_targets: list[str], device: torch.device) -> pd.DataFrame:
    subset_path = out / "predictions/scp682_tcga_tcpa_overlap_sample_centered.parquet"
    if subset_path.exists() and not args.force_scp682:
        return clean_numeric(pd.read_parquet(subset_path))
    train_mod = import_module(TRAIN_SCRIPT, "scp682_exact_scnet_release_train")
    ckpt = torch.load(CURRENT_RELEASE / "models/scp682_exact_scnet_gnn_best.pt", map_location="cpu", weights_only=False)
    model = build_exact_model(train_mod, ckpt, device)
    train_dir = CURRENT_RELEASE / "training_set"
    y_df = clean_numeric(pd.read_parquet(train_dir / "observed_phosphosite.parquet"))
    parent = clean_numeric(pd.read_parquet(train_dir / "oof_candidate_parent_only_phosphosite.parquet"))
    ridge = clean_numeric(pd.read_parquet(train_dir / "oof_candidate_ridge_direct_phosphosite.parquet"))
    rna_direct = clean_numeric(pd.read_parquet(train_dir / "oof_candidate_rna_direct_phosphosite.parquet"))
    train_rna = clean_numeric(pd.read_parquet(TRAIN_RNA))

    samples = pd.Index([str(x) for x in ckpt["samples"]])
    targets = [str(x) for x in ckpt["targets"]]
    train_base_df = ((parent.loc[samples, targets] + ridge.loc[samples, targets] + rna_direct.loc[samples, targets]) / 3.0).astype(np.float32)
    y_df = y_df.loc[samples, targets]
    train_mask = np.isfinite(y_df.to_numpy(np.float32)) & np.isfinite(train_base_df.to_numpy(np.float32))
    train_base = np.nan_to_num(train_base_df.to_numpy(np.float32), nan=0.0).astype(np.float32)

    ext_targets = [t for t in targets if t in baseline_ext.columns]
    ext_base_df = baseline_ext.loc[:, ext_targets]
    ext_base_full = pd.DataFrame(np.nan, index=ext_base_df.index, columns=targets, dtype=np.float32)
    ext_base_full.loc[:, ext_targets] = ext_base_df.to_numpy(np.float32)
    ext_mask = np.isfinite(ext_base_full.to_numpy(np.float32))
    ext_base = np.nan_to_num(ext_base_full.to_numpy(np.float32), nan=0.0).astype(np.float32)

    ctx, genes = make_rna_context_local(train_rna, samples, rna, ext_base_full.index, args.rna_context_genes)
    pd.Series(genes, name="gene").to_csv(out / "tables/scp682_rna_context_genes.tsv", sep="\t", index=False)
    train_ctx = ctx.iloc[:len(samples)].to_numpy(np.float32)
    ext_ctx = ctx.iloc[len(samples):].to_numpy(np.float32)
    train_feature = np.concatenate([
        args.knn_baseline_weight * l2_sample_block(train_mod, np.where(train_mask, train_base, 0.0)),
        args.knn_rna_weight * l2_sample_block(train_mod, train_ctx),
    ], axis=1).astype(np.float32)
    ext_feature = np.concatenate([
        args.knn_baseline_weight * l2_sample_block(train_mod, np.where(ext_mask, ext_base, 0.0)),
        args.knn_rna_weight * l2_sample_block(train_mod, ext_ctx),
    ], axis=1).astype(np.float32)
    anchor_weights = topk_softmax_attention(ext_feature, train_feature, args.anchor_k, args.anchor_temperature)
    top = np.argsort(-anchor_weights, axis=1)[:, :5]
    anchor_rows = []
    for i, sample in enumerate(ext_base_full.index.astype(str)):
        for rank, j in enumerate(top[i], 1):
            anchor_rows.append({"tcga_sample": sample, "rank": rank, "train_sample": str(samples[j]), "weight": float(anchor_weights[i, j])})
    pd.DataFrame(anchor_rows).to_csv(out / "tables/scp682_tcga_anchor_top5.tsv", sep="\t", index=False)

    feature_x = np.where(train_mask.T, train_base.T, 0.0).astype(np.float32)
    mu = feature_x.mean(axis=1, keepdims=True)
    sd = feature_x.std(axis=1, keepdims=True) + 1e-5
    feature_x = ((feature_x - mu) / sd).astype(np.float32)

    row_edge = torch.as_tensor(ckpt["site_edge_index"], dtype=torch.long, device=device)
    sample_edge = ckpt["sample_edge_index"]
    if isinstance(sample_edge, torch.Tensor):
        sample_edge = sample_edge.detach().cpu().numpy()
    sample_edge = torch.as_tensor(sample_edge, dtype=torch.long, device=device)
    site_prior = torch.as_tensor(ckpt["site_prior"], dtype=torch.float32, device=device)
    overlap_targets = [t for t in overlap_targets if t in ext_targets]
    overlap_idx = np.asarray([targets.index(t) for t in overlap_targets], dtype=np.int64)

    centered_parts = []
    raw_parts = []
    with torch.no_grad():
        row_embed, col_embed_train, _ = model.graph_core(
            torch.as_tensor(feature_x, dtype=torch.float32, device=device),
            sample_edge,
            row_edge,
            collect_attention=False,
        )
        col_embed_ext = torch.as_tensor(anchor_weights, dtype=torch.float32, device=device) @ col_embed_train
        overlap_idx_t = torch.as_tensor(overlap_idx, dtype=torch.long, device=device)
        for start in range(0, ext_base.shape[0], args.scp682_batch_size):
            end = min(ext_base.shape[0], start + args.scp682_batch_size)
            pred_b, *_ = model.decode(
                row_embed,
                col_embed_ext[start:end],
                torch.as_tensor(ext_base[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(ext_mask[start:end], dtype=torch.bool, device=device),
                site_prior,
            )
            med = pred_b.median(dim=1, keepdim=True).values
            raw_parts.append(pred_b.index_select(1, overlap_idx_t).detach().cpu().numpy().astype(np.float32))
            centered_parts.append((pred_b - med).index_select(1, overlap_idx_t).detach().cpu().numpy().astype(np.float32))
            print(f"scp682 decode {end}/{ext_base.shape[0]}", flush=True)

    raw_df = pd.DataFrame(np.vstack(raw_parts), index=ext_base_full.index, columns=overlap_targets).astype(np.float32)
    centered_df = pd.DataFrame(np.vstack(centered_parts), index=ext_base_full.index, columns=overlap_targets).astype(np.float32)
    raw_df.to_parquet(out / "predictions/scp682_tcga_tcpa_overlap_raw.parquet")
    centered_df.to_parquet(subset_path)
    return centered_df


def evaluate_overlap(pred_centered: pd.DataFrame, pred_raw: pd.DataFrame, y: pd.DataFrame, mapping: pd.DataFrame, out: Path, min_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    matched = mapping.loc[mapping["matched"].astype(bool)].copy()
    rows = []
    for _, row in matched.iterrows():
        antibody = str(row["antibody"])
        target = str(row["scp682_site_id"])
        if antibody not in y.columns or target not in pred_centered.columns:
            continue
        common = sorted(set(y.index.astype(str)).intersection(pred_centered.index.astype(str)))
        yy = y.loc[common, antibody].to_numpy(dtype=float)
        pc = pred_centered.loc[common, target].to_numpy(dtype=float)
        pr = pred_raw.loc[common, target].to_numpy(dtype=float)
        ok_c = np.isfinite(yy) & np.isfinite(pc)
        ok_r = np.isfinite(yy) & np.isfinite(pr)
        rho_c = spearmanr(yy[ok_c], pc[ok_c]).correlation if int(ok_c.sum()) >= min_n else np.nan
        rho_r = spearmanr(yy[ok_r], pr[ok_r]).correlation if int(ok_r.sum()) >= min_n else np.nan
        rows.append({
            "antibody": antibody,
            "scp682_site_id": target,
            "parent_gene": row["parent_gene"],
            "residue_site": row["residue_site"],
            "match_rule": row["match_rule"],
            "n_samples_used": int(ok_c.sum()),
            "spearman_sample_centered": float(rho_c) if np.isfinite(rho_c) else np.nan,
            "spearman_raw": float(rho_r) if np.isfinite(rho_r) else np.nan,
        })
    per = pd.DataFrame(rows).sort_values("spearman_sample_centered", ascending=False, na_position="last")
    vals = pd.to_numeric(per["spearman_sample_centered"], errors="coerce")
    summary = pd.DataFrame([{
        "model": "SCP682_current_exact_scnet_gnn",
        "comparison": "TCGA_RNA_to_TCPA_RPPA_overlap_phospho_antibodies",
        "n_tcga_samples": int(len(pred_centered.index)),
        "n_tcpa_phospho_antibodies": int(mapping.loc[mapping["antibody_type"].eq("phospho"), "antibody"].nunique()),
        "n_matched_antibody_site_pairs": int(per.shape[0]),
        "n_unique_matched_antibodies": int(per["antibody"].nunique()) if not per.empty else 0,
        "n_unique_scp682_sites": int(per["scp682_site_id"].nunique()) if not per.empty else 0,
        "median_spearman_sample_centered": float(vals.median(skipna=True)) if vals.notna().any() else np.nan,
        "mean_spearman_sample_centered": float(vals.mean(skipna=True)) if vals.notna().any() else np.nan,
        "ge_0_1": int((vals >= 0.1).sum()),
        "ge_0_2": int((vals >= 0.2).sum()),
        "ge_0_3": int((vals >= 0.3).sum()),
    }])
    per.to_csv(out / "tables/scp682_tcga_tcpa_overlap_per_antibody_site_spearman.tsv", sep="\t", index=False)
    summary.to_csv(out / "tables/scp682_tcga_tcpa_overlap_summary.tsv", sep="\t", index=False)
    return per, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(OUT))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--scp682-batch-size", type=int, default=8)
    p.add_argument("--bank-k", type=int, default=8)
    p.add_argument("--bank-chunk", type=int, default=512)
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--rna-context-genes", type=int, default=2048)
    p.add_argument("--knn-baseline-weight", type=float, default=0.70)
    p.add_argument("--knn-rna-weight", type=float, default=1.00)
    p.add_argument("--anchor-k", type=int, default=25)
    p.add_argument("--anchor-temperature", type=float, default=0.08)
    p.add_argument("--min-samples", type=int, default=30)
    p.add_argument("--force-baseline", action="store_true")
    p.add_argument("--force-scp682", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    for sub in ["predictions", "tables", "logs", "reports", "intermediate"]:
        ensure_dir(out / sub)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    rna = read_tcga_rna()
    y, panel, manifest = read_tcpa()
    common_samples = sorted(set(rna.index).intersection(y.index))
    rna = rna.loc[common_samples]
    y = y.loc[common_samples]
    manifest = manifest.loc[manifest["rna_sample_id"].astype(str).isin(common_samples)].copy()
    site_manifest = pd.read_csv(CURRENT_RELEASE / "training_set/phosphosite_target_manifest.tsv", sep="\t")
    mapping = build_antibody_site_mapping(panel, site_manifest)
    mapping.to_csv(out / "tables/tcpa_antibody_scp682_site_mapping.tsv", sep="\t", index=False)
    matched_targets = sorted(mapping.loc[mapping["matched"].astype(bool), "scp682_site_id"].astype(str).unique())
    qc = input_qc(rna, y, manifest)
    qc["device"] = str(device)
    qc["n_matched_tcpa_antibody_site_pairs"] = int(mapping["matched"].sum())
    qc["n_unique_matched_scp682_sites"] = int(len(matched_targets))
    (out / "reports/tcga_tcpa_input_and_mapping_qc.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(qc, indent=2, ensure_ascii=False), flush=True)
    if not matched_targets:
        raise SystemExit("No TCPA phospho antibody can be mapped to SCP682 sites.")

    baseline = build_baseline(args, out, rna, manifest, device)
    pred_centered = predict_exact_scnet(args, out, rna, baseline, matched_targets, device)
    pred_raw = clean_numeric(pd.read_parquet(out / "predictions/scp682_tcga_tcpa_overlap_raw.parquet"))
    per, summary = evaluate_overlap(pred_centered, pred_raw, y, mapping, out, args.min_samples)
    run_summary = {
        "output_dir": str(out),
        "device": str(device),
        "files": {
            "mapping": "tables/tcpa_antibody_scp682_site_mapping.tsv",
            "prediction_sample_centered": "predictions/scp682_tcga_tcpa_overlap_sample_centered.parquet",
            "prediction_raw": "predictions/scp682_tcga_tcpa_overlap_raw.parquet",
            "per_site_spearman": "tables/scp682_tcga_tcpa_overlap_per_antibody_site_spearman.tsv",
            "summary": "tables/scp682_tcga_tcpa_overlap_summary.tsv",
        },
        "summary": summary.to_dict(orient="records"),
    }
    (out / "reports/run_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "done.txt").write_text("done\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(per.head(30).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
