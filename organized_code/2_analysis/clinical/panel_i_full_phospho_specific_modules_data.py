#!/usr/bin/env python3
"""Build full TCGA phospho-specific ICA modules for Fig 5 panel i.

The script is intended for the Linux server because it reads the full TCGA
SCP682 prediction matrix and the matched RNA matrix from the portable rerun.
It writes the same small source-data files consumed by
panels/panel_i_representative_sites.R.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, hypergeom, norm
from sklearn.decomposition import FastICA
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests


DEFAULT_RERUN_DIR = Path(
    "/data/lsy/Infinite_Stream/02_results/model_prediction/"
    "20260529_tcga_full_scp682_main_reprediction_v1"
)
DEFAULT_PTMSIGDB_GMT = Path(
    "/data/lsy/Infinite_Stream/references/ptmsigdb_v2_0_0/"
    "ptm.sig.db.all.flanking.human.v2.0.0.gmt"
)
DEFAULT_CLINICAL_XML_DIR = Path("/data/gongke/shared/TCGA/data/clinical")
DEFAULT_OUTPUT_DIR = Path(
    "/data/lsy/Infinite_Stream/02_results/fig5/"
    "20260531_panel_i_phospho_specific_modules_full_v1/tables"
)


def bh_q(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    pvals = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    out = np.full(pvals.shape[0], np.nan, dtype=float)
    ok = np.isfinite(pvals)
    if ok.sum() == 0:
        return out
    out[ok] = multipletests(pvals[ok], method="fdr_bh")[1]
    return out


def split_gene_site(site_ids: list[str]) -> tuple[list[str], list[str]]:
    genes: list[str] = []
    sites: list[str] = []
    for gene_site in site_ids:
        if "|" in gene_site:
            gene, site = gene_site.split("|", 1)
        else:
            gene, site = gene_site, ""
        genes.append(gene)
        sites.append(site)
    return genes, sites


def standardize_columns(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    valid = np.isfinite(sd.ravel()) & (sd.ravel() >= 1e-8)
    sd[:, ~valid] = 1.0
    z = (x - mean) / sd
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z, valid


def prepare_survival(time: np.ndarray, event: np.ndarray) -> dict[str, np.ndarray]:
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.int8)
    ok = np.isfinite(time) & np.isfinite(event) & (time > 0)
    time = time[ok]
    event = event[ok]
    order = np.argsort(-time, kind="mergesort")
    time = time[order]
    event = event[order]
    event_times = np.sort(np.unique(time[event == 1]))[::-1]
    end_idx = np.searchsorted(-time, -event_times, side="right") - 1
    d = np.array([(event[time == t] == 1).sum() for t in event_times], dtype=np.float64)
    event_mask = np.vstack([(time == t) & (event == 1) for t in event_times])
    return {
        "ok": ok,
        "order": order,
        "time": time,
        "event": event,
        "event_times": event_times,
        "end_idx": end_idx,
        "d": d,
        "event_mask": event_mask,
    }


def loglik_univariate(x: np.ndarray, beta: np.ndarray, surv: dict[str, np.ndarray]) -> np.ndarray:
    x = x[surv["order"]]
    beta = np.asarray(beta, dtype=np.float64)
    eta = np.clip(x * beta.reshape(1, -1), -30.0, 30.0)
    w = np.exp(eta)
    cum_w = np.cumsum(w, axis=0)
    s0 = cum_w[surv["end_idx"]]
    event_sum_x = surv["event_mask"].astype(float) @ x
    return beta * event_sum_x.sum(axis=0) - (
        surv["d"][:, None] * np.log(np.maximum(s0, 1e-12))
    ).sum(axis=0)


def cox_univariate_batch(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    max_iter: int = 10,
) -> dict[str, np.ndarray]:
    x, valid = standardize_columns(x)
    surv = prepare_survival(time, event)
    x = x[surv["ok"]]
    n_features = x.shape[1]
    n = x.shape[0]
    events = int(surv["event"].sum())
    if n < 3 or events < 2:
        nan = np.full(n_features, np.nan)
        return {
            "beta": nan,
            "p": nan,
            "wald_chisq": nan,
            "lrt_chisq": nan,
            "loglik": nan,
            "n": np.full(n_features, n),
            "events": np.full(n_features, events),
        }

    x_order = x[surv["order"]]
    event_sum_x = surv["event_mask"].astype(float) @ x_order
    beta = np.zeros(n_features, dtype=np.float64)
    info = np.full(n_features, np.nan, dtype=np.float64)
    for _ in range(max_iter):
        eta = np.clip(x_order * beta.reshape(1, -1), -30.0, 30.0)
        w = np.exp(eta)
        cum_w = np.cumsum(w, axis=0)
        cum_wx = np.cumsum(w * x_order, axis=0)
        cum_wx2 = np.cumsum(w * x_order * x_order, axis=0)
        s0 = cum_w[surv["end_idx"]]
        s1 = cum_wx[surv["end_idx"]]
        s2 = cum_wx2[surv["end_idx"]]
        mean_x = s1 / np.maximum(s0, 1e-12)
        var_x = s2 / np.maximum(s0, 1e-12) - mean_x * mean_x
        score = event_sum_x.sum(axis=0) - (surv["d"][:, None] * mean_x).sum(axis=0)
        info = (surv["d"][:, None] * np.maximum(var_x, 0.0)).sum(axis=0)
        step = np.where(info > 1e-10, score / info, 0.0)
        step = np.clip(step, -1.0, 1.0)
        beta = np.clip(beta + step, -8.0, 8.0)
        if float(np.nanmax(np.abs(step))) < 1e-5:
            break

    ll = loglik_univariate(x, beta, surv)
    ll0 = loglik_univariate(x, np.zeros(n_features), surv)
    wald = np.where(info > 1e-10, beta * beta * info, np.nan)
    lrt = np.maximum(2.0 * (ll - ll0), 0.0)
    p = chi2.sf(wald, df=1)
    for arr in [beta, p, wald, lrt, ll]:
        arr[~valid] = np.nan
    return {
        "beta": beta,
        "p": p,
        "wald_chisq": wald,
        "lrt_chisq": lrt,
        "loglik": ll,
        "n": np.full(n_features, n),
        "events": np.full(n_features, events),
    }


def loglik_bivariate(
    x: np.ndarray,
    z: np.ndarray,
    beta_x: np.ndarray,
    beta_z: np.ndarray,
    surv: dict[str, np.ndarray],
) -> np.ndarray:
    x = x[surv["order"]]
    z = z[surv["order"]]
    eta = np.clip(x * beta_x.reshape(1, -1) + z * beta_z.reshape(1, -1), -30.0, 30.0)
    w = np.exp(eta)
    cum_w = np.cumsum(w, axis=0)
    s0 = cum_w[surv["end_idx"]]
    event_sum_x = surv["event_mask"].astype(float) @ x
    event_sum_z = surv["event_mask"].astype(float) @ z
    return (
        beta_x * event_sum_x.sum(axis=0)
        + beta_z * event_sum_z.sum(axis=0)
        - (surv["d"][:, None] * np.log(np.maximum(s0, 1e-12))).sum(axis=0)
    )


def cox_bivariate_site_parent_batch(
    x_site: np.ndarray,
    z_parent: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    max_iter: int = 10,
) -> dict[str, np.ndarray]:
    x_site, valid_x = standardize_columns(x_site)
    z_parent, valid_z = standardize_columns(z_parent)
    valid = valid_x & valid_z
    surv = prepare_survival(time, event)
    x = x_site[surv["ok"]]
    z = z_parent[surv["ok"]]
    p = x.shape[1]
    if x.shape[0] < 3 or int(surv["event"].sum()) < 2:
        nan = np.full(p, np.nan)
        return {
            "site_beta": nan,
            "site_p": nan,
            "site_wald_chisq": nan,
            "parent_beta": nan,
            "parent_p": nan,
            "joint_loglik": nan,
        }

    x_order = x[surv["order"]]
    z_order = z[surv["order"]]
    event_mat = surv["event_mask"].astype(float)
    event_sum_x = event_mat @ x_order
    event_sum_z = event_mat @ z_order
    beta_x = np.zeros(p, dtype=np.float64)
    beta_z = np.zeros(p, dtype=np.float64)
    ixx = np.full(p, np.nan, dtype=np.float64)
    izz = np.full(p, np.nan, dtype=np.float64)
    ixz = np.full(p, np.nan, dtype=np.float64)

    for _ in range(max_iter):
        eta = np.clip(x_order * beta_x.reshape(1, -1) + z_order * beta_z.reshape(1, -1), -30.0, 30.0)
        w = np.exp(eta)
        cum_w = np.cumsum(w, axis=0)
        cum_wx = np.cumsum(w * x_order, axis=0)
        cum_wz = np.cumsum(w * z_order, axis=0)
        cum_wxx = np.cumsum(w * x_order * x_order, axis=0)
        cum_wzz = np.cumsum(w * z_order * z_order, axis=0)
        cum_wxz = np.cumsum(w * x_order * z_order, axis=0)

        s0 = cum_w[surv["end_idx"]]
        sx = cum_wx[surv["end_idx"]]
        sz = cum_wz[surv["end_idx"]]
        sxx = cum_wxx[surv["end_idx"]]
        szz = cum_wzz[surv["end_idx"]]
        sxz = cum_wxz[surv["end_idx"]]

        mx = sx / np.maximum(s0, 1e-12)
        mz = sz / np.maximum(s0, 1e-12)
        vxx = sxx / np.maximum(s0, 1e-12) - mx * mx
        vzz = szz / np.maximum(s0, 1e-12) - mz * mz
        vxz = sxz / np.maximum(s0, 1e-12) - mx * mz

        score_x = event_sum_x.sum(axis=0) - (surv["d"][:, None] * mx).sum(axis=0)
        score_z = event_sum_z.sum(axis=0) - (surv["d"][:, None] * mz).sum(axis=0)
        ixx = (surv["d"][:, None] * np.maximum(vxx, 0.0)).sum(axis=0)
        izz = (surv["d"][:, None] * np.maximum(vzz, 0.0)).sum(axis=0)
        ixz = (surv["d"][:, None] * vxz).sum(axis=0)

        det = ixx * izz - ixz * ixz
        ok = det > 1e-10
        step_x = np.zeros(p, dtype=np.float64)
        step_z = np.zeros(p, dtype=np.float64)
        step_x[ok] = (score_x[ok] * izz[ok] - score_z[ok] * ixz[ok]) / det[ok]
        step_z[ok] = (ixx[ok] * score_z[ok] - ixz[ok] * score_x[ok]) / det[ok]
        step_x = np.clip(step_x, -1.0, 1.0)
        step_z = np.clip(step_z, -1.0, 1.0)
        beta_x = np.clip(beta_x + step_x, -8.0, 8.0)
        beta_z = np.clip(beta_z + step_z, -8.0, 8.0)
        if float(max(np.nanmax(np.abs(step_x)), np.nanmax(np.abs(step_z)))) < 1e-5:
            break

    det = ixx * izz - ixz * ixz
    site_var = np.where((det > 1e-10) & (izz > 1e-10), izz / det, np.nan)
    parent_var = np.where((det > 1e-10) & (ixx > 1e-10), ixx / det, np.nan)
    site_wald = np.where(site_var > 0, beta_x * beta_x / site_var, np.nan)
    parent_wald = np.where(parent_var > 0, beta_z * beta_z / parent_var, np.nan)
    joint_ll = loglik_bivariate(x, z, beta_x, beta_z, surv)

    for arr in [beta_x, beta_z, site_wald, parent_wald, joint_ll]:
        arr[~valid] = np.nan
    return {
        "site_beta": beta_x,
        "site_p": chi2.sf(site_wald, df=1),
        "site_wald_chisq": site_wald,
        "parent_beta": beta_z,
        "parent_p": chi2.sf(parent_wald, df=1),
        "joint_loglik": joint_ll,
    }


def strip_xml_tag(tag: str) -> str:
    return tag.split("}", 1)[-1].lower()


def clean_missing(text: str | None) -> str | None:
    if text is None:
        return None
    value = str(text).strip()
    if not value:
        return None
    low = value.lower()
    if low in {"not available", "not reported", "not applicable", "unknown", "[not available]", "[not reported]"}:
        return None
    return value


def stage_to_numeric(value: str | None) -> float:
    value = clean_missing(value)
    if value is None:
        return np.nan
    low = value.lower()
    if "stage 0" in low:
        return 0.0
    if "stage i" in low and "stage ii" not in low and "stage iv" not in low:
        return 1.0
    if "stage ii" in low and "stage iii" not in low:
        return 2.0
    if "stage iii" in low:
        return 3.0
    if "stage iv" in low:
        return 4.0
    m = re.search(r"\b([1-4])\b", low)
    return float(m.group(1)) if m else np.nan


def grade_to_numeric(value: str | None) -> float:
    value = clean_missing(value)
    if value is None:
        return np.nan
    low = value.lower()
    if "high" in low:
        return 3.0
    if "low" in low:
        return 1.0
    m = re.search(r"g\s*([1-4])|grade\s*([1-4])|\b([1-4])\b", low)
    if not m:
        return np.nan
    for g in m.groups():
        if g is not None:
            return float(g)
    return np.nan


def build_clinical_covariates(xml_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not xml_dir.exists():
        return pd.DataFrame(columns=["patient_id", "age", "stage_num", "grade_num"])
    age_tags = {"age_at_initial_pathologic_diagnosis", "age_at_diagnosis"}
    stage_tags = {"pathologic_stage", "clinical_stage"}
    grade_tags = {"neoplasm_histologic_grade", "histological_grade", "histologic_grade", "tumor_grade"}
    for path in xml_dir.glob("*/*.xml"):
        try:
            root = ET.parse(path).getroot()
        except Exception:
            continue
        values: dict[str, str] = {}
        patient_id = None
        days_to_birth = None
        for elem in root.iter():
            tag = strip_xml_tag(elem.tag)
            text = clean_missing(elem.text)
            if text is None:
                continue
            if tag == "bcr_patient_barcode" and patient_id is None:
                patient_id = text[:12]
            elif tag == "days_to_birth" and days_to_birth is None:
                days_to_birth = text
            elif tag in age_tags and "age" not in values:
                values["age"] = text
            elif tag in stage_tags and "stage" not in values:
                values["stage"] = text
            elif tag in grade_tags and "grade" not in values:
                values["grade"] = text
        if patient_id is None:
            continue
        age = pd.to_numeric(values.get("age"), errors="coerce")
        if not np.isfinite(age) and days_to_birth is not None:
            age = abs(pd.to_numeric(days_to_birth, errors="coerce")) / 365.25
        rows.append(
            {
                "patient_id": patient_id,
                "age": float(age) if np.isfinite(age) else np.nan,
                "stage": values.get("stage"),
                "stage_num": stage_to_numeric(values.get("stage")),
                "grade": values.get("grade"),
                "grade_num": grade_to_numeric(values.get("grade")),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["patient_id", "age", "stage_num", "grade_num"])
    out = pd.DataFrame(rows)
    return out.sort_values("patient_id").drop_duplicates("patient_id")


def clean_signature_label(signature: str) -> tuple[str, str]:
    if signature.startswith("KINASE-"):
        cls = "Kinase"
        label = signature.replace("KINASE-", "")
    elif signature.startswith("PERT-"):
        cls = "Perturbation"
        label = signature.replace("PERT-", "")
    elif signature.startswith("PATH-"):
        cls = "Pathway"
        label = signature.replace("PATH-", "")
    elif signature.startswith("DISEASE-"):
        cls = "Disease"
        label = signature.replace("DISEASE-", "")
    else:
        cls = "Other"
        label = signature
    label = label.replace("_", " ")
    return label[:46], cls


def parse_ptmsigdb(path: Path) -> tuple[dict[str, set[str]], dict[str, dict[str, int]], pd.DataFrame]:
    sets: dict[str, set[str]] = {}
    signs: dict[str, dict[str, int]] = {}
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            signature = fields[0]
            desc_parts = fields[1].split("|")
            sites = []
            for item in desc_parts[1:]:
                token = item.split(":", 1)[0]
                if "_" not in token:
                    continue
                gene, site = token.split("_", 1)
                sites.append(f"{gene}|{site}")
            peptide_dirs = []
            for peptide in fields[2:]:
                if peptide.endswith(";u"):
                    peptide_dirs.append(1)
                elif peptide.endswith(";d"):
                    peptide_dirs.append(-1)
                else:
                    peptide_dirs.append(0)
            site_set = set(sites)
            sets[signature] = site_set
            sign_map: dict[str, int] = {}
            for site, direction in zip(sites, peptide_dirs):
                if direction == 0:
                    continue
                if site not in sign_map:
                    sign_map[site] = direction
                elif sign_map[site] != direction:
                    sign_map[site] = 0
            signs[signature] = {k: v for k, v in sign_map.items() if v != 0}
            label, cls = clean_signature_label(signature)
            rows.append(
                {
                    "signature": signature,
                    "signature_label": label,
                    "signature_class": cls,
                    "n_sites": len(site_set),
                    "n_signed_sites": len(signs[signature]),
                }
            )
    return sets, signs, pd.DataFrame(rows)


def module_ora(
    module_id: str,
    module_sites: set[str],
    universe: set[str],
    signature_sets: dict[str, set[str]],
    signature_meta: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    n_universe = len(universe)
    n_module = len(module_sites)
    for signature, raw_sites in signature_sets.items():
        sig_sites = raw_sites & universe
        if len(sig_sites) < 5:
            continue
        overlap = len(module_sites & sig_sites)
        if overlap < 3:
            continue
        p = hypergeom.sf(overlap - 1, n_universe, len(sig_sites), n_module)
        rows.append(
            {
                "module_id": module_id,
                "signature": signature,
                "overlap_sites": overlap,
                "module_sites_in_universe": n_module,
                "signature_sites_in_universe": len(sig_sites),
                "p_value": p,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = bh_q(out["p_value"])
    out = out.merge(signature_meta, on="signature", how="left")
    out["odds_like"] = (
        (out["overlap_sites"] / out["module_sites_in_universe"])
        / (out["signature_sites_in_universe"] / n_universe)
    )
    return out.sort_values(["q_value", "p_value", "overlap_sites"], ascending=[True, True, False])


def robust_scale_columns(x: pd.DataFrame) -> pd.DataFrame:
    y = x.copy()
    for col in y.columns:
        vals = y[col].astype(float)
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0:
            scale = np.nanstd(vals)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        y[col] = (vals - med) / scale
    return y.fillna(0.0)


def load_full_inputs(rerun_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = pd.read_parquet(rerun_dir / "predictions" / "tcga_full_scp682_predicted_phosphosite.parquet")
    rna = pd.read_parquet(rerun_dir / "intermediate" / "tcga_gdc_star_tpm_model_genes_full.parquet")
    rna = np.log2(rna.astype("float32") + 1.0)
    manifest = pd.read_csv(rerun_dir / "tables" / "tcga_scp682_prediction_sample_manifest.tsv", sep="\t")
    manifest = manifest.loc[
        manifest["has_os_survival"].astype(bool)
        & pd.to_numeric(manifest["survival_time"], errors="coerce").notna()
        & pd.to_numeric(manifest["survival_event"], errors="coerce").notna()
    ].copy()
    manifest["patient_id"] = manifest["tcga_patient_id"].astype(str).str.slice(0, 12)
    manifest["cancer"] = manifest["cancer"].astype(str)
    return pred, rna, manifest


def build_site_level_tables(
    pred: pd.DataFrame,
    rna: pd.DataFrame,
    manifest: pd.DataFrame,
    min_n: int,
    min_events: int,
    chunk_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    site_cols = [c for c in pred.columns if "|" in str(c)]
    genes, sites = split_gene_site(site_cols)
    gene_by_site = pd.Series(genes, index=site_cols)
    site_by_site = pd.Series(sites, index=site_cols)
    gene_set = sorted(set(genes) & set(rna.columns))
    cancers = sorted(manifest["cancer"].dropna().unique())
    rows: list[pd.DataFrame] = []
    z_cols: dict[str, pd.Series] = {}
    audit_rows = []

    for cancer in cancers:
        m = manifest.loc[manifest["cancer"].eq(cancer)].copy()
        m = m.sort_values(["patient_id", "sample_id"]).drop_duplicates("patient_id")
        m = m.loc[m["sample_id"].isin(pred.index) & m["sample_id"].isin(rna.index)].copy()
        n = m.shape[0]
        events = int(pd.to_numeric(m["survival_event"], errors="coerce").fillna(0).sum())
        audit_rows.append({"cancer": cancer, "n": n, "events": events, "tested": n >= min_n and events >= min_events})
        z_cols[cancer] = pd.Series(np.nan, index=site_cols, dtype=float)
        if n < min_n or events < min_events:
            continue

        sample_ids = m["sample_id"].tolist()
        time = pd.to_numeric(m["survival_time"], errors="coerce").to_numpy(dtype=float)
        event = pd.to_numeric(m["survival_event"], errors="coerce").fillna(0).to_numpy(dtype=int)
        pred_c = pred.loc[sample_ids, site_cols]
        rna_c = rna.loc[sample_ids, gene_set]

        mrna_res = cox_univariate_batch(rna_c.to_numpy(dtype=np.float64), time, event)
        gene_stats = pd.DataFrame(
            {
                "gene": gene_set,
                "parent_mrna_beta": mrna_res["beta"],
                "parent_mrna_p": mrna_res["p"],
                "parent_mrna_wald_chisq": mrna_res["wald_chisq"],
                "parent_mrna_lrt_chisq": mrna_res["lrt_chisq"],
                "parent_mrna_loglik": mrna_res["loglik"],
            }
        ).set_index("gene")

        cancer_parts: list[pd.DataFrame] = []
        cancer_z = pd.Series(np.nan, index=site_cols, dtype=float)
        for start in range(0, len(site_cols), chunk_size):
            cols = site_cols[start : start + chunk_size]
            chunk_genes = gene_by_site.loc[cols].tolist()
            valid_cols = [c for c, g in zip(cols, chunk_genes, strict=False) if g in gene_stats.index]
            if not valid_cols:
                continue
            chunk_genes = gene_by_site.loc[valid_cols].tolist()
            x = pred_c[valid_cols].to_numpy(dtype=np.float64)
            z = np.column_stack([rna_c[g].to_numpy(dtype=np.float64) for g in chunk_genes])
            site_res = cox_univariate_batch(x, time, event)
            joint_res = cox_bivariate_site_parent_batch(x, z, time, event)
            parent_ll = gene_stats.loc[chunk_genes, "parent_mrna_loglik"].to_numpy(dtype=float)
            site_ll = site_res["loglik"]
            joint_ll = joint_res["joint_loglik"]
            add_site = np.maximum(2.0 * (joint_ll - parent_ll), 0.0)
            add_mrna = np.maximum(2.0 * (joint_ll - site_ll), 0.0)
            parent_wald = gene_stats.loc[chunk_genes, "parent_mrna_wald_chisq"].to_numpy(dtype=float)
            parent_lrt = gene_stats.loc[chunk_genes, "parent_mrna_lrt_chisq"].to_numpy(dtype=float)
            parent_beta = gene_stats.loc[chunk_genes, "parent_mrna_beta"].to_numpy(dtype=float)
            parent_p = gene_stats.loc[chunk_genes, "parent_mrna_p"].to_numpy(dtype=float)
            cancer_z.loc[valid_cols] = np.sign(site_res["beta"]) * np.sqrt(site_res["wald_chisq"])
            cancer_parts.append(
                pd.DataFrame(
                    {
                        "cancer": cancer,
                        "gene_site": valid_cols,
                        "gene": chunk_genes,
                        "site": site_by_site.loc[valid_cols].tolist(),
                        "n_mrna_overlap": site_res["n"],
                        "events_mrna_overlap": site_res["events"],
                        "parent_mrna_beta": parent_beta,
                        "parent_mrna_p": parent_p,
                        "parent_mrna_wald_chisq": parent_wald,
                        "parent_mrna_lrt_chisq": parent_lrt,
                        "predicted_site_beta_same_mrna_samples": site_res["beta"],
                        "predicted_site_p_same_mrna_samples": site_res["p"],
                        "predicted_site_wald_chisq_same_mrna_samples": site_res["wald_chisq"],
                        "predicted_site_lrt_chisq_same_mrna_samples": site_res["lrt_chisq"],
                        "predicted_site_minus_parent_mrna_wald_chisq": site_res["wald_chisq"] - parent_wald,
                        "predicted_site_minus_parent_mrna_lrt_chisq": site_res["lrt_chisq"] - parent_lrt,
                        "site_beta_adjusted_for_parent_mrna": joint_res["site_beta"],
                        "site_p_adjusted_for_parent_mrna": joint_res["site_p"],
                        "site_wald_chisq_adjusted_for_parent_mrna": joint_res["site_wald_chisq"],
                        "parent_beta_in_joint_model": joint_res["parent_beta"],
                        "parent_p_in_joint_model": joint_res["parent_p"],
                        "add_site_to_parent_mrna_lrt_chisq": add_site,
                        "add_site_to_parent_mrna_lrt_p": chi2.sf(add_site, df=1),
                        "add_parent_mrna_to_site_lrt_chisq": add_mrna,
                        "add_parent_mrna_to_site_lrt_p": chi2.sf(add_mrna, df=1),
                        "site_parent_direction_concordance": np.sign(site_res["beta"]) == np.sign(parent_beta),
                    }
                )
            )
        rows.append(pd.concat(cancer_parts, ignore_index=True))
        z_cols[cancer] = cancer_z

    lrt = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not lrt.empty:
        lrt["add_site_to_parent_mrna_lrt_q_bh"] = bh_q(lrt["add_site_to_parent_mrna_lrt_p"])
        lrt["site_p_adjusted_for_parent_mrna_q_bh"] = bh_q(lrt["site_p_adjusted_for_parent_mrna"])
        lrt["site_over_mrna_direction"] = np.where(lrt["site_beta_adjusted_for_parent_mrna"] >= 0, "risk", "protective")
    z_mat = pd.DataFrame(z_cols)
    z_mat.index.name = "gene_site"
    return lrt, z_mat, pd.DataFrame(audit_rows)


def fit_modules(
    z_all: pd.DataFrame,
    lrt: pd.DataFrame,
    ptmsigdb_gmt: Path,
    q_threshold: float,
    n_components: int,
    top_fraction: float,
    min_sites: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keep = (
        (lrt["add_site_to_parent_mrna_lrt_q_bh"] < q_threshold)
        & (lrt["site_p_adjusted_for_parent_mrna_q_bh"] < q_threshold)
    )
    filtered = lrt.loc[keep].copy()
    pass_sites = sorted(set(filtered["gene_site"]) & set(z_all.index))
    z_filtered = z_all.loc[pass_sites].copy()
    site_pass = (
        filtered.groupby("gene_site", as_index=False)
        .agg(
            n_pass_cancers=("cancer", "nunique"),
            best_site_over_parent_q=("add_site_to_parent_mrna_lrt_q_bh", "min"),
            best_adjusted_site_q=("site_p_adjusted_for_parent_mrna_q_bh", "min"),
        )
        .set_index("gene_site")
    )
    signature_sets, signature_signs, signature_meta = parse_ptmsigdb(ptmsigdb_gmt)
    x = robust_scale_columns(z_filtered)
    n_components = min(n_components, x.shape[1] - 1, x.shape[0] - 1)
    ica = FastICA(n_components=n_components, whiten="unit-variance", random_state=20260531, max_iter=2000)
    site_scores = pd.DataFrame(
        ica.fit_transform(x.values),
        index=x.index,
        columns=[f"M{i:02d}" for i in range(1, n_components + 1)],
    )
    universe = set(z_all.index)
    module_mats = []
    module_meta = []
    module_sites_rows = []
    ora_rows = []
    top_n = max(min_sites, int(round(z_filtered.shape[0] * top_fraction)))
    top_n = min(top_n, 900)

    for module_id in site_scores.columns:
        weights = site_scores[module_id].copy()
        selected = weights.abs().sort_values(ascending=False).head(top_n).index.tolist()
        selected_weights = weights.loc[selected].copy()
        module_sites = set(selected)
        ora = module_ora(module_id, module_sites, universe, signature_sets, signature_meta)
        if not ora.empty:
            preferred = ora[(ora["signature_class"] != "Disease") & (ora["overlap_sites"] >= 5)]
            if preferred.empty:
                preferred = ora[ora["overlap_sites"] >= 5]
            if preferred.empty:
                top = ora.iloc[0]
                label = "weakly annotated module"
                module_class = "Other"
                top_signature = str(top["signature"])
                top_q = float(top["q_value"])
            else:
                top = preferred.iloc[0]
                label = str(top["signature_label"])
                module_class = str(top["signature_class"])
                top_signature = str(top["signature"])
                top_q = float(top["q_value"])
            ora_rows.append(ora)
        else:
            label = "phospho-specific module"
            module_class = "Other"
            top_signature = ""
            top_q = np.nan

        orientation_score = 0.0
        orientation_sites = 0
        if top_signature in signature_signs:
            sign_map = signature_signs[top_signature]
            for site, weight in selected_weights.items():
                if site in sign_map:
                    orientation_score += float(weight) * int(sign_map[site])
                    orientation_sites += 1
        if orientation_sites > 0 and orientation_score < 0:
            selected_weights = -selected_weights
            weights = -weights
            orientation_score = -orientation_score

        raw = z_all.loc[selected, :]
        denom = np.sum(np.abs(selected_weights.values))
        if denom <= 0:
            denom = 1.0
        module_effect = raw.mul(selected_weights, axis=0).sum(axis=0) / denom
        if orientation_sites == 0:
            strongest = module_effect.iloc[np.nanargmax(np.abs(module_effect.values))]
            if strongest < 0:
                selected_weights = -selected_weights
                weights = -weights
                module_effect = -module_effect

        row_label = f"{module_id} {label}"
        module_mats.append(pd.DataFrame([module_effect.values], index=[row_label], columns=z_all.columns))
        pass_counts = site_pass.reindex(selected)["n_pass_cancers"].fillna(0)
        module_meta.append(
            {
                "row_label": row_label,
                "module_id": module_id,
                "signature_label": label,
                "signature_class": module_class,
                "top_signature": top_signature,
                "top_signature_q": top_q,
                "n_module_sites": len(selected),
                "median_pass_cancers": float(np.nanmedian(pass_counts.values)),
                "orientation_score": float(orientation_score),
                "orientation_sites": int(orientation_sites),
                "max_abs_effect": float(np.nanmax(np.abs(module_effect.values))),
                "mean_abs_effect": float(np.nanmean(np.abs(module_effect.values))),
            }
        )
        for site, weight in selected_weights.items():
            module_sites_rows.append(
                {
                    "module_id": module_id,
                    "row_label": row_label,
                    "gene_site": site,
                    "site_weight": float(weight),
                    "abs_weight_rank": int(selected_weights.abs().rank(ascending=False, method="first").loc[site]),
                    "n_pass_cancers": float(site_pass.reindex([site])["n_pass_cancers"].fillna(0).iloc[0]),
                }
            )

    mat = pd.concat(module_mats, axis=0)
    meta = pd.DataFrame(module_meta)
    site_table = pd.DataFrame(module_sites_rows)
    ora_table = pd.concat(ora_rows, ignore_index=True) if ora_rows else pd.DataFrame()
    class_rank = {"Kinase": 1, "Pathway": 2, "Perturbation": 3, "Other": 4, "Disease": 5}
    meta["_class_rank"] = meta["signature_class"].map(class_rank).fillna(9)
    meta = meta.sort_values(["_class_rank", "max_abs_effect"], ascending=[True, False]).drop(columns=["_class_rank"])
    mat = mat.reindex(meta["row_label"])
    return mat, meta, site_table, ora_table


def fit_phreg(time: np.ndarray, event: np.ndarray, x: pd.DataFrame) -> tuple[float, float, float, float]:
    dat = x.copy()
    dat["time"] = time
    dat["event"] = event
    dat = dat.replace([np.inf, -np.inf], np.nan).dropna()
    if dat.shape[0] < 30 or dat["event"].sum() < 4:
        return np.nan, np.nan, np.nan, np.nan
    y = dat.pop("time").to_numpy(dtype=float)
    status = dat.pop("event").to_numpy(dtype=int)
    exog = dat.to_numpy(dtype=float)
    try:
        fit = PHReg(y, exog, status=status).fit(disp=0)
        beta = float(fit.params[0])
        se = float(fit.bse[0])
        z = beta / se if se > 0 else np.nan
        p = 2.0 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
        return beta, se, z, p
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def build_module_sample_cox(
    pred: pd.DataFrame,
    manifest: pd.DataFrame,
    clinical: pd.DataFrame,
    site_table: pd.DataFrame,
    min_n: int,
    min_events: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_cols = set(pred.columns)
    module_rows = []
    score_frames = []
    for cancer in sorted(manifest["cancer"].dropna().unique()):
        m = manifest.loc[manifest["cancer"].eq(cancer)].copy()
        m = m.sort_values(["patient_id", "sample_id"]).drop_duplicates("patient_id")
        m = m.loc[m["sample_id"].isin(pred.index)].copy()
        n = m.shape[0]
        events = int(pd.to_numeric(m["survival_event"], errors="coerce").fillna(0).sum())
        if n < min_n or events < min_events:
            continue
        cov = m[["sample_id", "patient_id", "survival_time", "survival_event"]].merge(clinical, on="patient_id", how="left")
        pred_c = pred.loc[cov["sample_id"].tolist()]
        time = pd.to_numeric(cov["survival_time"], errors="coerce").to_numpy(dtype=float)
        event = pd.to_numeric(cov["survival_event"], errors="coerce").fillna(0).to_numpy(dtype=int)
        for row_label, g in site_table.groupby("row_label", sort=False):
            g = g.loc[g["gene_site"].isin(pred_cols)].copy()
            if g.empty:
                continue
            sites = g["gene_site"].tolist()
            weights = g["site_weight"].to_numpy(dtype=float)
            vals = pred_c[sites].to_numpy(dtype=np.float64)
            vals, valid = standardize_columns(vals)
            if valid.sum() < 5:
                continue
            weights = weights[valid]
            vals = vals[:, valid]
            denom = float(np.sum(np.abs(weights)))
            if denom <= 0:
                denom = 1.0
            score = vals @ weights / denom
            score = (score - np.nanmean(score)) / (np.nanstd(score) if np.nanstd(score) > 0 else 1.0)
            covariates = pd.DataFrame({"module_score": score})
            cov_used = []
            for col in ["age", "stage_num", "grade_num"]:
                v = pd.to_numeric(cov[col], errors="coerce") if col in cov.columns else pd.Series(np.nan, index=cov.index)
                ok_fraction = float(v.notna().mean())
                if ok_fraction >= 0.55 and v.nunique(dropna=True) >= 2:
                    vv = (v - v.mean()) / (v.std() if v.std() and np.isfinite(v.std()) else 1.0)
                    covariates[col] = vv
                    cov_used.append(col)
            beta, se, z, p = fit_phreg(time, event, covariates)
            beta0, se0, z0, p0 = fit_phreg(time, event, pd.DataFrame({"module_score": score}))
            module_rows.append(
                {
                    "cancer": cancer,
                    "row_label": row_label,
                    "n": int(n),
                    "events": int(events),
                    "module_beta": beta,
                    "module_se": se,
                    "module_z": z,
                    "module_p": p,
                    "module_unadjusted_beta": beta0,
                    "module_unadjusted_se": se0,
                    "module_unadjusted_z": z0,
                    "module_unadjusted_p": p0,
                    "covariates_used": ",".join(cov_used),
                    "n_covariates": len(cov_used),
                }
            )
            score_frames.append(
                pd.DataFrame(
                    {
                        "cancer": cancer,
                        "sample_id": cov["sample_id"].to_numpy(),
                        "patient_id": cov["patient_id"].to_numpy(),
                        "row_label": row_label,
                        "module_score": score,
                        "survival_time": time,
                        "survival_event": event,
                    }
                )
            )
    cox = pd.DataFrame(module_rows)
    if not cox.empty:
        cox["module_q"] = bh_q(cox["module_p"])
        cox = add_cross_cancer_specificity(cox)
    scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    return cox, scores


def add_cross_cancer_specificity(cox: pd.DataFrame) -> pd.DataFrame:
    out = cox.copy()
    out["specificity_z"] = np.nan
    out["specificity_p"] = np.nan
    for row_label, g in out.groupby("row_label"):
        for idx, row in g.iterrows():
            others = g.loc[g.index != idx].copy()
            others = others.loc[
                np.isfinite(others["module_beta"])
                & np.isfinite(others["module_se"])
                & (others["module_se"] > 0)
            ]
            if (
                not np.isfinite(row["module_beta"])
                or not np.isfinite(row["module_se"])
                or row["module_se"] <= 0
                or others.empty
            ):
                continue
            w = 1.0 / np.square(others["module_se"].to_numpy(dtype=float))
            beta_other = float(np.sum(w * others["module_beta"].to_numpy(dtype=float)) / np.sum(w))
            se_other = float(math.sqrt(1.0 / np.sum(w)))
            se_diff = math.sqrt(float(row["module_se"]) ** 2 + se_other**2)
            z = (float(row["module_beta"]) - beta_other) / se_diff
            out.loc[idx, "specificity_z"] = z
            out.loc[idx, "specificity_p"] = 2.0 * norm.sf(abs(z))
    out["specificity_q"] = bh_q(out["specificity_p"])
    return out


def sample_cox_matrix(cox: pd.DataFrame, meta: pd.DataFrame, cancers: list[str]) -> pd.DataFrame:
    mat = cox.pivot(index="row_label", columns="cancer", values="module_z")
    mat = mat.reindex(index=meta["row_label"], columns=cancers)
    return mat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-dir", type=Path, default=DEFAULT_RERUN_DIR)
    parser.add_argument("--ptmsigdb-gmt", type=Path, default=DEFAULT_PTMSIGDB_GMT)
    parser.add_argument("--clinical-xml-dir", type=Path, default=DEFAULT_CLINICAL_XML_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--q-threshold", type=float, default=0.05)
    parser.add_argument("--n-components", type=int, default=14)
    parser.add_argument("--top-fraction", type=float, default=0.04)
    parser.add_argument("--min-sites", type=int, default=180)
    parser.add_argument("--min-n", type=int, default=40)
    parser.add_argument("--min-events", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--reuse-site-tables", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred, rna, manifest = load_full_inputs(args.rerun_dir)

    lrt_path = args.output_dir / "panel_i_full_site_over_parent_lrt.tsv.gz"
    z_path = args.output_dir / "panel_i_full_site_signed_cox_z_matrix.tsv"
    audit_path = args.output_dir / "panel_i_full_site_cox_cancer_audit.tsv"
    if args.reuse_site_tables and lrt_path.exists() and z_path.exists():
        lrt = pd.read_csv(lrt_path, sep="\t")
        z_all = pd.read_csv(z_path, sep="\t").set_index("gene_site")
        audit = pd.read_csv(audit_path, sep="\t") if audit_path.exists() else pd.DataFrame()
    else:
        lrt, z_all, audit = build_site_level_tables(
            pred=pred,
            rna=rna,
            manifest=manifest,
            min_n=args.min_n,
            min_events=args.min_events,
            chunk_size=args.chunk_size,
        )
        lrt.to_csv(lrt_path, sep="\t", index=False, compression="gzip")
        z_all.reset_index().to_csv(z_path, sep="\t", index=False)
        audit.to_csv(audit_path, sep="\t", index=False)

    mat_summary, meta, site_table, ora = fit_modules(
        z_all=z_all,
        lrt=lrt,
        ptmsigdb_gmt=args.ptmsigdb_gmt,
        q_threshold=args.q_threshold,
        n_components=args.n_components,
        top_fraction=args.top_fraction,
        min_sites=args.min_sites,
    )
    clinical = build_clinical_covariates(args.clinical_xml_dir)
    module_cox, module_scores = build_module_sample_cox(
        pred=pred,
        manifest=manifest,
        clinical=clinical,
        site_table=site_table,
        min_n=args.min_n,
        min_events=args.min_events,
    )
    cancers = list(z_all.columns)
    mat = sample_cox_matrix(module_cox, meta, cancers)
    if mat.dropna(how="all").shape[0] == 0:
        mat = mat_summary

    mat.reset_index(names="row_label").to_csv(
        args.output_dir / "panel_i_phospho_specific_module_effect_matrix.tsv",
        sep="\t",
        index=False,
    )
    mat_summary.reset_index(names="row_label").to_csv(
        args.output_dir / "panel_i_phospho_specific_module_site_rank_summary_matrix.tsv",
        sep="\t",
        index=False,
    )
    meta.to_csv(args.output_dir / "panel_i_phospho_specific_module_rows.tsv", sep="\t", index=False)
    site_table.to_csv(args.output_dir / "panel_i_phospho_specific_module_sites.tsv", sep="\t", index=False)
    ora.to_csv(args.output_dir / "panel_i_phospho_specific_module_ptmsigdb_ora.tsv", sep="\t", index=False)
    clinical.to_csv(args.output_dir / "panel_i_tcga_clinical_stage_grade_covariates.tsv", sep="\t", index=False)
    module_cox.to_csv(args.output_dir / "panel_i_phospho_specific_module_sample_cox.tsv", sep="\t", index=False)
    module_scores.to_csv(args.output_dir / "panel_i_phospho_specific_module_sample_scores.tsv.gz", sep="\t", index=False, compression="gzip")

    keep = (
        (lrt["add_site_to_parent_mrna_lrt_q_bh"] < args.q_threshold)
        & (lrt["site_p_adjusted_for_parent_mrna_q_bh"] < args.q_threshold)
    )
    summary = {
        "rerun_dir": str(args.rerun_dir),
        "q_threshold": args.q_threshold,
        "all_ranked_sites": int(z_all.shape[0]),
        "phospho_specific_sites": int(lrt.loc[keep, "gene_site"].nunique()),
        "site_lrt_rows": int(lrt.shape[0]),
        "cancers_in_z_matrix": cancers,
        "n_cancers_in_z_matrix": int(len(cancers)),
        "n_modules": int(mat.shape[0]),
        "module_site_top_fraction": args.top_fraction,
        "module_site_count_per_module": int(site_table.groupby("module_id")["gene_site"].nunique().median()),
        "module_effect_matrix": "clinical covariate adjusted module Cox z when covariates are available",
        "method": "parent-mRNA LRT filter, FastICA site modules, PTMsigDB naming, signed-member orientation, sample-level Cox",
        "min_n": args.min_n,
        "min_events": args.min_events,
        "cancer_audit": audit.to_dict(orient="records") if not audit.empty else [],
    }
    with open(args.output_dir / "panel_i_phospho_specific_module_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
