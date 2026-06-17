#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold


ROOT = Path("/data/lsy/Infinite_Stream")
BASE_SCRIPT = ROOT / "03_code/external_validation/phosphoproteomics/deploy_v38_luad_cas_and_evaluate_true_phosphosite_20260502.py"
REPO = ROOT / "01_data/public_bulk_phosphoproteome_atlas/raw/cbioportal_datahub/datahub.git"
CACHE = ROOT / "01_data/public_bulk_phosphoproteome_atlas/raw/cbioportal_selected_lfs"
CBIO_META = ROOT / "01_data/public_bulk_phosphoproteome_atlas/metadata/cbioportal_datahub_fixed_v2_supported_candidates_20260508.tsv"
OUT = ROOT / "02_results/public_bulk_phosphoproteome_atlas/20260508_scp682_v4_0_public_bulk_atlas"

MODEL_ID = "SCP682_v4_0"
ROBUST = ["ridge_direct", "parent_only", "rna_direct"]
CVAE = ["v3_train_oof", "v3_1_1_feedback_prior", "v3_1_2_target_attention", "v3_6_ranking_coverage_loss"]

PROJECT_TO_CONTEXT = {
    "TCGA-BRCA": ("BRCA_TCGA", "PDC000174"),
    "TCGA-COAD": ("COAD_PROSPECTIVE", "PDC000117"),
    "TCGA-GBM": ("GBM_DISCOVERY", "PDC000205"),
    "TCGA-HNSC": ("HNSCC", "PDC000222"),
    "TCGA-KIRC": ("CCRCC", "PDC000128"),
    "TCGA-KIRP": ("NON_CCRCC", "PDC000465"),
    "TCGA-LUAD": ("LUAD", "PDC000149"),
    "TCGA-LUSC": ("LSCC", "PDC000232"),
    "TCGA-OV": ("OV_TCGA", "PDC000115"),
    "TCGA-PAAD": ("PDA", "PDC000271"),
    "TCGA-STAD": ("STAD", "PDC000615"),
    "TCGA-UCEC": ("UCEC", "PDC000126"),
}

CONTEXT_TO_STUDY = {
    "BRCA_TCGA": "PDC000174",
    "COAD_PROSPECTIVE": "PDC000117",
    "GBM_DISCOVERY": "PDC000205",
    "HNSCC": "PDC000222",
    "CCRCC": "PDC000128",
    "NON_CCRCC": "PDC000465",
    "LUAD": "PDC000149",
    "LSCC": "PDC000232",
    "OV_TCGA": "PDC000115",
    "PDA": "PDC000271",
    "STAD": "PDC000615",
    "UCEC": "PDC000126",
}

EXCLUDE_CBIO = {
    "difg_glass": "diffuse_glioma_not_HNSCC",
    "difg_glass_2019": "diffuse_glioma_not_HNSCC",
    "nepc_wcm_2016": "neuroendocrine_prostate_not_OV",
}


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def import_base():
    return import_module(BASE_SCRIPT, "scp682_v4_public_base")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.index.name = "sample_id"
    pq.write_table(pa.Table.from_pandas(df.reset_index()), path)


def git_show(path: str) -> str:
    local = CACHE / path.removeprefix("public/")
    if local.exists():
        return local.read_text(errors="replace")
    return subprocess.check_output(["git", "-C", str(REPO), "show", f"HEAD:{path}"], text=True, errors="replace")


def media_url(path: str) -> str:
    return "https://media.githubusercontent.com/media/cBioPortal/datahub/master/" + path


def read_text(path: str) -> str:
    local = CACHE / path.removeprefix("public/")
    if local.exists():
        return local.read_text(errors="replace")
    text = git_show(path)
    if text.startswith("version https://git-lfs.github.com/spec/v1"):
        with urllib.request.urlopen(media_url(path), timeout=240) as fh:
            return fh.read().decode("utf-8", errors="replace")
    return text


def read_cbioportal_table(path: str) -> pd.DataFrame:
    text = read_text(path)
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", low_memory=False)


def expression_to_log2tpm(x: pd.DataFrame, source: str) -> tuple[pd.DataFrame, str]:
    vals = x.to_numpy(dtype=np.float32)
    finite = vals[np.isfinite(vals)]
    source_l = source.lower()
    if "fpkm" in source_l or "rpkm" in source_l:
        nonneg = x.clip(lower=0)
        denom = nonneg.sum(axis=1, skipna=True).replace(0, np.nan)
        tpm = nonneg.div(denom, axis=0) * 1_000_000.0
        return np.log2(tpm + 1.0), "fpkm_or_rpkm_to_tpm_then_log2_tpm_plus_1"
    if "cpm" in source_l or "tpm" in source_l:
        if finite.size and np.nanpercentile(finite, 99) > 80:
            return np.log2(x.clip(lower=0) + 1.0), "log2(x+1)_applied"
        return x, "as_provided_assumed_log2tpm"
    if finite.size and np.nanpercentile(finite, 99) > 80:
        return np.log2(x.clip(lower=0) + 1.0), "auto_log2_x_plus_1"
    return x, "as_provided_assumed_log2tpm"


def read_expression(path: str, study_id: str) -> pd.DataFrame:
    df = read_cbioportal_table(path)
    if df.empty:
        return pd.DataFrame()
    gene_col = "Hugo_Symbol" if "Hugo_Symbol" in df.columns else df.columns[0]
    sample_cols = [c for c in df.columns if c not in {gene_col, "Entrez_Gene_Id", "Cytoband"}]
    mat = df[[gene_col] + sample_cols].copy()
    mat[gene_col] = mat[gene_col].astype(str)
    mat = mat.loc[mat[gene_col].ne("") & mat[gene_col].ne("nan")]
    mat = mat.groupby(gene_col, sort=False)[sample_cols].median(numeric_only=True)
    x = mat.T.apply(pd.to_numeric, errors="coerce")
    x, transform = expression_to_log2tpm(x, path)
    x.index = [f"{study_id}::{s}" for s in x.index.astype(str)]
    x.index.name = "sample_id"
    x.attrs["expression_transform"] = transform
    return x


def load_tcga() -> tuple[pd.DataFrame, pd.DataFrame]:
    x_raw = pd.read_parquet(ROOT / "data/processed/X_all.symbols.parquet")
    x_raw = x_raw.drop_duplicates("gene_symbol", keep="first").set_index("gene_symbol")
    x = x_raw.T.apply(pd.to_numeric, errors="coerce")
    x.index = x.index.astype(str)
    meta = pd.read_csv(ROOT / "metadata/tcga_barcodes_pancan.tsv", sep="\t")
    meta["barcode"] = meta["barcode"].astype(str)
    meta = (
        meta.loc[meta["barcode"].isin(x.index)]
        .drop_duplicates("barcode", keep="first")
        .set_index("barcode")
        .loc[x.index]
        .reset_index()
        .rename(columns={"index": "sample_id"})
    )
    meta["sample_short"] = meta["sample_id"].str[:16]
    meta["cptac_cancer_label"] = meta["project"].map(lambda p: PROJECT_TO_CONTEXT.get(p, (None, None))[0])
    meta["cptac_study_id"] = meta["project"].map(lambda p: PROJECT_TO_CONTEXT.get(p, (None, None))[1])
    keep = meta["cptac_cancer_label"].notna()
    x = x.loc[keep.to_numpy()]
    meta = meta.loc[keep].reset_index(drop=True)
    x.index = meta["sample_id"].astype(str).to_list()
    x.index.name = "sample_id"
    meta["expression_transform"] = "as_provided_tcga_harmonized_assumed_log2tpm"
    return x, meta


def load_cbioportal() -> tuple[pd.DataFrame, pd.DataFrame]:
    cand = pd.read_csv(CBIO_META, sep="\t")
    cand = cand.loc[cand["prelim_context_supported"].astype(bool)].copy()
    cand = cand.loc[~cand["study_id"].isin(EXCLUDE_CBIO)].copy()
    xs = []
    rows = []
    for _, row in cand.iterrows():
        study = str(row["study_id"])
        try:
            x = read_expression(str(row["expression_file"]), study)
        except Exception as exc:
            print("skip", study, repr(exc), flush=True)
            continue
        if x.empty or x.shape[0] < 10:
            continue
        xs.append(x)
        for sid in x.index:
            rows.append({
                "sample_id": sid,
                "study_id": study,
                "source_sample_id": sid.split("::", 1)[1],
                "cptac_cancer_label": row["cptac_context"],
                "cptac_study_id": CONTEXT_TO_STUDY.get(row["cptac_context"]),
                "expression_file": row["expression_file"],
                "expression_transform": x.attrs.get("expression_transform", "unknown"),
                "status": "predicted",
            })
        print("loaded", study, x.shape, x.attrs.get("expression_transform"), flush=True)
    if not xs:
        raise RuntimeError("no cBioPortal matrices loaded")
    x_all = pd.concat(xs, axis=0, join="outer", sort=False)
    manifest = pd.DataFrame(rows).set_index("sample_id").loc[x_all.index].reset_index()
    return x_all, manifest


def patch_dynamic_context(base, manifest: pd.DataFrame) -> None:
    info = manifest.set_index("sample_id")[["cptac_cancer_label", "cptac_study_id"]].to_dict("index")

    def make_context_arrays(index: pd.Index, ckpt: dict) -> tuple[np.ndarray, np.ndarray]:
        cancer_map = {c: i for i, c in enumerate(ckpt["cancer_levels"])}
        study_map = {s: i for i, s in enumerate(ckpt["study_levels"])}
        cancer = []
        study = []
        for sid in index.astype(str):
            row = info[sid]
            cancer.append(cancer_map[row["cptac_cancer_label"]])
            study.append(study_map[row["cptac_study_id"]])
        return np.asarray(cancer, dtype=np.int64), np.asarray(study, dtype=np.int64)

    base.make_context_arrays = make_context_arrays


def export_light_candidates_dynamic(base, data: dict, x_ext: pd.DataFrame, v3_total_ext: pd.DataFrame, manifest: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    y = data["phospho"]
    total = data["total"]
    x = data["x"]
    feature_idx = {g: i for i, g in enumerate(data["feature_names"])}
    ext_feature_idx = {g: i for i, g in enumerate(x_ext.columns)}
    total_idx = {g: i for i, g in enumerate(data["total_names"])}
    ext_total_idx = {g: i for i, g in enumerate(v3_total_ext.columns)}
    site_genes = [str(s).split("|", 1)[0] for s in data["phospho_names"]]
    out = {name: np.full((len(x_ext), len(data["phospho_names"])), np.nan, dtype=np.float32) for name in ROBUST}
    counts = {name: np.zeros((len(x_ext), len(data["phospho_names"])), dtype=np.float32) for name in out}
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=20260502).split(x, data["cancer_ids"]))
    cancer_levels = pd.Series(data["manifest"]["cancer_label"]).astype("category").cat.categories
    cancer_onehot = pd.get_dummies(pd.Series(data["cancer_ids"], dtype="category")).to_numpy(dtype=np.float32)
    ext_cancer_onehot = np.zeros((len(x_ext), cancer_onehot.shape[1]), dtype=np.float32)
    m2 = manifest.set_index("sample_id").loc[x_ext.index]
    for i, ctx in enumerate(m2["cptac_cancer_label"].astype(str)):
        ext_cancer_onehot[i, int(cancer_levels.get_loc(ctx))] = 1.0
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
        frames[name] = pd.DataFrame(mat, index=x_ext.index, columns=data["phospho_names"])
    return frames


def sample_center(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = df.median(axis=1, skipna=True)
    centered = df.sub(med, axis=0).astype(np.float32)
    offsets = pd.DataFrame({"sample_id": df.index.astype(str), "raw_prediction_sample_median": med.to_numpy(dtype=float)})
    return centered, offsets


def predict_one(cohort: str, rna: pd.DataFrame, manifest: pd.DataFrame, args: argparse.Namespace) -> None:
    out = OUT
    for sub in ["predictions", "tables", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    base = import_base()
    patch_dynamic_context(base, manifest)
    base.seed_all(args.seed)
    device = base.torch.device("cuda:0" if base.torch.cuda.is_available() else "cpu")
    data = base.load_base_data(device)
    x_ext = base.build_external_feature_frame(rna, device)
    manifest = manifest.set_index("sample_id").loc[x_ext.index].reset_index()
    manifest.to_csv(out / "tables" / f"{cohort}_prediction_manifest.tsv", sep="\t", index=False)

    candidates: dict[str, pd.DataFrame] = {}
    v3_pred, v3_total = base.predict_v3_family("v3_train_oof", base.V3_SCRIPT, base.V3_MODEL_DIR, "20260502_cptac_parent_residual_kinase_cvae_experimental_v3", data, x_ext, device, args)
    candidates["v3_train_oof"] = v3_pred
    for label, script, model_dir, prefix in [
        ("v3_1_1_feedback_prior", base.V311_SCRIPT, base.V311_MODEL_DIR, "20260502_cptac_parent_residual_kinase_cvae_experimental_v3_1_1_feedback_prior"),
        ("v3_1_2_target_attention", base.V312_SCRIPT, base.V312_MODEL_DIR, "20260502_cptac_parent_residual_kinase_cvae_experimental_v3_1_2_target_attention"),
        ("v3_6_ranking_coverage_loss", base.V36_SCRIPT, base.V36_MODEL_DIR, "20260502_cptac_parent_residual_kinase_cvae_experimental_v3_6_ranking_coverage_loss"),
    ]:
        pred, _ = base.predict_v3_family(label, script, model_dir, prefix, data, x_ext, device, args, v3_external_for_feedback=v3_pred)
        candidates[label] = pred
    candidates.update(export_light_candidates_dynamic(base, data, x_ext, v3_total, manifest, args))
    common = sorted(set.intersection(*[set(candidates[n].columns) for n in ROBUST + CVAE]))
    index = x_ext.index
    robust = sum(candidates[n].reindex(index=index, columns=common) for n in ROBUST) / float(len(ROBUST))
    cvae = sum(candidates[n].reindex(index=index, columns=common) for n in CVAE) / float(len(CVAE))
    raw = (0.8 * robust + 0.2 * cvae).astype(np.float32)
    raw.index.name = "sample_id"
    centered, offsets = sample_center(raw)
    write_parquet(raw, out / "predictions" / f"SCP682_v4_0_{cohort}_phosphosite_raw_before_sample_median_centering.parquet")
    write_parquet(centered, out / "predictions" / f"SCP682_v4_0_{cohort}_phosphosite.parquet")
    offsets.insert(0, "cohort", cohort)
    offsets.to_csv(out / "tables" / f"{cohort}_sample_median_offsets.tsv", sep="\t", index=False)
    summary = {
        "model_id": MODEL_ID,
        "cohort": cohort,
        "n_samples": int(centered.shape[0]),
        "n_phosphosite_targets": int(centered.shape[1]),
        "formula": "0.8 * mean(ridge_direct, parent_only, rna_direct) + 0.2 * mean(v3_train_oof, v3_1_1_feedback_prior, v3_1_2_target_attention, v3_6_ranking_coverage_loss)",
        "postprocessing": "sample_median_centering",
    }
    (out / "logs" / f"{cohort}_prediction_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["tcga", "cbioportal", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bank-k", type=int, default=8)
    parser.add_argument("--bank-chunk", type=int, default=512)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260508)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cohort in {"tcga", "all"}:
        rna, manifest = load_tcga()
        predict_one("tcga_supported", rna, manifest, args)
    if args.cohort in {"cbioportal", "all"}:
        rna, manifest = load_cbioportal()
        predict_one("cbioportal_supported", rna, manifest, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
