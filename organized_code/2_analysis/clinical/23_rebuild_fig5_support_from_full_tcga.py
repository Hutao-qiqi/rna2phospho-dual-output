#!/usr/bin/env python3
"""Rebuild Fig. 5 support tables from the full TCGA SCP682 main prediction.

This script is intended to run on the Linux server that contains
SCP682_PORTABLE outputs. It does not train or modify SCP682. It reads:

  - full SCP682 phosphosite prediction
  - V4 baseline prediction
  - exact ScNET graph residual output
  - model-gene TPM matrix used for prediction
  - TCGA sample manifest and OS overlap table

and writes Fig. 5 source tables with explicit full-prediction, baseline,
graph-residual, and parent-mRNA clinical effect decompositions.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

try:
    from sklearn.decomposition import NMF
except Exception:  # pragma: no cover
    NMF = None


DEFAULT_ROOT = Path("/data/lsy/Infinite_Stream")
DEFAULT_REPRED = DEFAULT_ROOT / "02_results/model_prediction/20260529_tcga_full_scp682_main_reprediction_v1"
DEFAULT_OUT = DEFAULT_ROOT / "02_results/model_prediction/20260529_fig5_full_tcga_support_from_scp682_main_v1"

MIN_N_ANALYSIS = 50
MIN_EVENTS_ANALYSIS = 10
MIN_N_MANUSCRIPT = 80
MIN_EVENTS_MANUSCRIPT = 20
CHUNK_SIZE = 1400

FOUR_CANCERS = ["TCGA-LGG", "TCGA-KIRC", "TCGA-SARC", "TCGA-LIHC"]


PROGRAMS = [
    "stress_response",
    "cell_cycle_checkpoint",
    "chromatin_state",
    "metabolic_state",
    "immune_state",
    "translation_stress",
    "signal_transduction",
    "cytoskeleton_adhesion",
    "other",
]


def setup_logger(outdir: Path) -> logging.Logger:
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fig5_full_tcga_support")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    fh = logging.FileHandler(outdir / "logs/run.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def ensure_dirs(outdir: Path) -> None:
    for sub in ["tables", "logs", "reports"]:
        (outdir / sub).mkdir(parents=True, exist_ok=True)


def split_gene_site(site_ids: list[str]) -> tuple[list[str], list[str]]:
    genes: list[str] = []
    sites: list[str] = []
    for gene_site in site_ids:
        if "|" in str(gene_site):
            gene, site = str(gene_site).split("|", 1)
        else:
            gene, site = str(gene_site), ""
        genes.append(gene)
        sites.append(site)
    return genes, sites


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    vals = np.clip(p[ok], 0.0, 1.0)
    order = np.argsort(vals)
    ranked = vals[order]
    n = ranked.size
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(vals)
    out[order] = np.clip(adj, 0.0, 1.0)
    q[ok] = out
    return q


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
    event_mask = np.vstack([(time == t) & (event == 1) for t in event_times]) if event_times.size else np.zeros((0, time.size), dtype=bool)
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


def loglik_univariate_z(x_z: np.ndarray, beta: np.ndarray, surv: dict[str, np.ndarray]) -> np.ndarray:
    x = x_z[surv["order"]]
    beta = np.asarray(beta, dtype=np.float64)
    eta = np.clip(x * beta.reshape(1, -1), -30.0, 30.0)
    w = np.exp(eta)
    cum_w = np.cumsum(w, axis=0)
    s0 = cum_w[surv["end_idx"]]
    event_sum_x = surv["event_mask"].astype(float) @ x
    return beta * event_sum_x.sum(axis=0) - (surv["d"][:, None] * np.log(np.maximum(s0, 1e-12))).sum(axis=0)


def loglik_bivariate_z(x_z: np.ndarray, z_z: np.ndarray, beta_x: np.ndarray, beta_z: np.ndarray, surv: dict[str, np.ndarray]) -> np.ndarray:
    x = x_z[surv["order"]]
    z = z_z[surv["order"]]
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


def cox_univariate_batch(x: np.ndarray, time: np.ndarray, event: np.ndarray, max_iter: int = 10) -> dict[str, np.ndarray]:
    x_z, valid = standardize_columns(x)
    surv = prepare_survival(time, event)
    x_z = x_z[surv["ok"]]
    n_features = x_z.shape[1]
    n = x_z.shape[0]
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
            "valid": valid,
        }

    x_ordered = x_z[surv["order"]]
    end_idx = surv["end_idx"]
    d = surv["d"]
    event_sum_x = surv["event_mask"].astype(float) @ x_ordered
    beta = np.zeros(n_features, dtype=np.float64)
    info = np.full(n_features, np.nan, dtype=np.float64)
    for _ in range(max_iter):
        eta = np.clip(x_ordered * beta.reshape(1, -1), -30.0, 30.0)
        w = np.exp(eta)
        cum_w = np.cumsum(w, axis=0)
        cum_wx = np.cumsum(w * x_ordered, axis=0)
        cum_wx2 = np.cumsum(w * x_ordered * x_ordered, axis=0)
        s0 = cum_w[end_idx]
        s1 = cum_wx[end_idx]
        s2 = cum_wx2[end_idx]
        mean_x = s1 / np.maximum(s0, 1e-12)
        var_x = s2 / np.maximum(s0, 1e-12) - mean_x * mean_x
        score = event_sum_x.sum(axis=0) - (d[:, None] * mean_x).sum(axis=0)
        info = (d[:, None] * np.maximum(var_x, 0.0)).sum(axis=0)
        step = np.divide(score, info, out=np.zeros_like(score), where=info > 1e-10)
        step = np.clip(step, -1.0, 1.0)
        beta = np.clip(beta + step, -8.0, 8.0)
        if float(np.nanmax(np.abs(step))) < 1e-5:
            break

    wald = np.where(info > 1e-10, beta * beta * info, np.nan)
    p = chi2.sf(wald, df=1)
    loglik = loglik_univariate_z(x_z, beta, surv)
    p[~valid] = np.nan
    wald[~valid] = np.nan
    beta[~valid] = np.nan
    loglik[~valid] = np.nan
    return {
        "beta": beta,
        "p": p,
        "wald_chisq": wald,
        "lrt_chisq": 2.0 * loglik,
        "loglik": loglik,
        "n": np.full(n_features, n),
        "events": np.full(n_features, events),
        "valid": valid,
    }


def cox_bivariate_add_batch(
    base: np.ndarray,
    add: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    base_loglik: np.ndarray,
    max_iter: int = 10,
) -> dict[str, np.ndarray]:
    base_z, valid_base = standardize_columns(base)
    add_z, valid_add = standardize_columns(add)
    valid = valid_base & valid_add
    surv = prepare_survival(time, event)
    base_z = base_z[surv["ok"]]
    add_z = add_z[surv["ok"]]
    n_features = base_z.shape[1]
    n = base_z.shape[0]
    events = int(surv["event"].sum())
    if n < 3 or events < 2:
        nan = np.full(n_features, np.nan)
        return {
            "beta_base": nan,
            "beta_add": nan,
            "add_wald_chisq": nan,
            "add_p": nan,
            "joint_loglik": nan,
            "lrt_chisq": nan,
            "lrt_p": nan,
        }

    x = base_z[surv["order"]]
    z = add_z[surv["order"]]
    end_idx = surv["end_idx"]
    d = surv["d"]
    event_mask = surv["event_mask"].astype(float)
    event_sum_x = event_mask @ x
    event_sum_z = event_mask @ z
    beta_x = np.zeros(n_features, dtype=np.float64)
    beta_z = np.zeros(n_features, dtype=np.float64)
    i_xx = np.full(n_features, np.nan, dtype=np.float64)
    i_zz = np.full(n_features, np.nan, dtype=np.float64)
    i_xz = np.full(n_features, np.nan, dtype=np.float64)
    for _ in range(max_iter):
        eta = np.clip(x * beta_x.reshape(1, -1) + z * beta_z.reshape(1, -1), -30.0, 30.0)
        w = np.exp(eta)
        wx = w * x
        wz = w * z
        cum_w = np.cumsum(w, axis=0)
        cum_wx = np.cumsum(wx, axis=0)
        cum_wz = np.cumsum(wz, axis=0)
        cum_wxx = np.cumsum(wx * x, axis=0)
        cum_wzz = np.cumsum(wz * z, axis=0)
        cum_wxz = np.cumsum(wx * z, axis=0)
        s0 = cum_w[end_idx]
        s1x = cum_wx[end_idx]
        s1z = cum_wz[end_idx]
        s2xx = cum_wxx[end_idx]
        s2zz = cum_wzz[end_idx]
        s2xz = cum_wxz[end_idx]
        mx = s1x / np.maximum(s0, 1e-12)
        mz = s1z / np.maximum(s0, 1e-12)
        i_xx = (d[:, None] * np.maximum(s2xx / np.maximum(s0, 1e-12) - mx * mx, 0.0)).sum(axis=0)
        i_zz = (d[:, None] * np.maximum(s2zz / np.maximum(s0, 1e-12) - mz * mz, 0.0)).sum(axis=0)
        i_xz = (d[:, None] * (s2xz / np.maximum(s0, 1e-12) - mx * mz)).sum(axis=0)
        score_x = event_sum_x.sum(axis=0) - (d[:, None] * mx).sum(axis=0)
        score_z = event_sum_z.sum(axis=0) - (d[:, None] * mz).sum(axis=0)
        det = i_xx * i_zz - i_xz * i_xz
        step_x = np.divide(i_zz * score_x - i_xz * score_z, det, out=np.zeros_like(score_x), where=det > 1e-10)
        step_z = np.divide(-i_xz * score_x + i_xx * score_z, det, out=np.zeros_like(score_z), where=det > 1e-10)
        step_x = np.clip(step_x, -1.0, 1.0)
        step_z = np.clip(step_z, -1.0, 1.0)
        beta_x = np.clip(beta_x + step_x, -8.0, 8.0)
        beta_z = np.clip(beta_z + step_z, -8.0, 8.0)
        if float(max(np.nanmax(np.abs(step_x)), np.nanmax(np.abs(step_z)))) < 1e-5:
            break

    joint_loglik = loglik_bivariate_z(base_z, add_z, beta_x, beta_z, surv)
    lrt = 2.0 * (joint_loglik - np.asarray(base_loglik, dtype=float))
    lrt = np.where(lrt >= 0, lrt, 0.0)
    lrt_p = chi2.sf(lrt, df=1)
    add_var = np.divide(i_xx, i_xx * i_zz - i_xz * i_xz, out=np.full_like(i_xx, np.nan), where=(i_xx * i_zz - i_xz * i_xz) > 1e-10)
    add_wald = np.divide(beta_z * beta_z, add_var, out=np.full_like(beta_z, np.nan), where=add_var > 1e-12)
    add_p = chi2.sf(add_wald, df=1)
    for arr in [beta_x, beta_z, joint_loglik, lrt, lrt_p, add_wald, add_p]:
        arr[~valid] = np.nan
    return {
        "beta_base": beta_x,
        "beta_add": beta_z,
        "add_wald_chisq": add_wald,
        "add_p": add_p,
        "joint_loglik": joint_loglik,
        "lrt_chisq": lrt,
        "lrt_p": lrt_p,
    }


def infer_rna_program(gene: str) -> str:
    g = str(gene).upper()
    if g.startswith(("HSP", "DNAJ")) or g in {"JUN", "JUNB", "JUND", "FOS", "DDIT3", "ATF4", "ATF6", "NFKB1", "HSPB1"}:
        return "stress_response"
    if g.startswith(("WEE", "CDK", "CCN", "CDC", "MCM", "CHEK", "PLK", "AURK", "BUB", "MAD", "E2F", "TOP2")):
        return "cell_cycle_checkpoint"
    if g.startswith(("GATAD", "HDAC", "CHD", "KDM", "SETD", "SMAR", "BRD", "DNMT", "HIST", "H1", "H2", "H3", "H4", "ARID")):
        return "chromatin_state"
    if g.startswith(("LTV", "RPL", "RPS", "EIF", "NOP", "RRP", "NPM", "DDX", "MDN", "GAR", "LARP", "PWP", "NOL")):
        return "translation_stress"
    if g.startswith(("HLA", "CXCL", "CCL", "STAT", "IRF", "CD", "LILR", "FCGR")) or g in {"RELA", "RELB", "NFKB1", "NFKB2"}:
        return "immune_state"
    if g.startswith(("UCK", "PKM", "LDH", "ACAC", "G6PD", "IDH", "SDH", "PFK", "SLC2")) or g in {"MTOR", "RICTOR", "RPTOR", "AKT1", "AKT2"}:
        return "metabolic_state"
    if g.startswith(("PLCL", "ARHGEF", "RHO", "RAS", "MAPK", "RAF", "ERBB", "EGFR", "SRC", "PIK", "PRK", "FYN", "YES", "LCK")):
        return "signal_transduction"
    if g.startswith(("ACT", "MYH", "FLN", "VCL", "TJP", "MAP1", "MARCKS", "SPT", "TNS", "PXN", "CTN", "CLDN")):
        return "cytoskeleton_adhesion"
    return "other"


def pathway_family(gene: str, program: str) -> str:
    g = str(gene).upper()
    if program == "stress_response" or g.startswith(("HSP", "DNAJ")):
        return "Stress response"
    if program == "cell_cycle_checkpoint" or g.startswith(("CDK", "CCN", "WEE", "CDC", "MCM", "AURK", "PLK")):
        return "Cell cycle"
    if program == "translation_stress" or g.startswith(("RPL", "RPS", "EIF", "LTV", "MDN", "NOL")):
        return "Ribosome / translation"
    if program == "chromatin_state" or g.startswith(("GATAD", "HDAC", "CHD", "SMAR", "BRD", "KDM")):
        return "Chromatin"
    if program == "immune_state" or g.startswith(("HLA", "CXCL", "CCL", "CD", "STAT", "IRF")):
        return "Immune signaling"
    if program == "metabolic_state" or g.startswith(("UCK", "PKM", "LDH", "SLC", "G6PD", "MTOR", "RICTOR")):
        return "Metabolism / mTOR"
    if program == "cytoskeleton_adhesion" or g.startswith(("ACT", "MYH", "FLN", "VCL", "TJP", "PXN", "SPT")):
        return "Cytoskeleton / adhesion"
    if program == "signal_transduction" or g.startswith(("ARHGEF", "RHO", "RAS", "MAPK", "RAF", "ERBB", "EGFR", "SRC", "PRK", "FYN")):
        return "Signal transduction"
    return "Other context-linked"


def pathway_submodule(gene: str, family: str) -> str:
    g = str(gene).upper()
    if family == "Stress response":
        if g.startswith("HSPB1"):
            return "HSPB1 module"
        if g.startswith("HSP"):
            return "heat-shock proteins"
        if g.startswith("DNAJ"):
            return "chaperone cofactor"
        return "stress transcription"
    if family == "Cell cycle":
        if g.startswith(("WEE", "CHEK")):
            return "checkpoint kinases"
        if g.startswith(("CDK", "CCN")):
            return "CDK-cyclin axis"
        return "replication and mitosis"
    if family == "Ribosome / translation":
        if g.startswith(("LTV", "MDN", "NOL", "NOP")):
            return "ribosome biogenesis"
        return "translation machinery"
    if family == "Chromatin":
        if g.startswith("GATAD"):
            return "NuRD-related"
        return "chromatin remodeling"
    if family == "Metabolism / mTOR":
        if g in {"RICTOR", "MTOR", "RPTOR", "AKT1", "AKT2"}:
            return "mTOR-AKT"
        if g.startswith("UCK"):
            return "pyrimidine metabolism"
        return "metabolic enzymes"
    if family == "Signal transduction":
        if g.startswith(("SRC", "FYN", "YES", "LCK")):
            return "SRC-family kinases"
        if g.startswith(("ARHGEF", "RHO")):
            return "Rho cytoskeleton signaling"
        if g.startswith(("ERBB", "EGFR")):
            return "ERBB-EGFR"
        return "kinase and second messenger"
    if family == "Cytoskeleton / adhesion":
        if g.startswith(("TJP", "CLDN")):
            return "junction and polarity"
        if g.startswith(("FLN", "ACT", "MYH", "SPT")):
            return "actin membrane scaffold"
        return "adhesion signaling"
    if family == "Immune signaling":
        if g.startswith("HLA"):
            return "antigen presentation"
        if g.startswith(("CXCL", "CCL")):
            return "chemokine signaling"
        return "immune receptor signaling"
    return infer_rna_program(gene).replace("_", " ")


def module_label(gene: str) -> tuple[str, str, str]:
    program = infer_rna_program(gene)
    family = pathway_family(gene, program)
    sub = pathway_submodule(gene, family)
    return program, family, f"{family} / {sub}"


def top_mean_abs(values: pd.Series, k: int = 3) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().abs().sort_values(ascending=False)
    if vals.empty:
        return 0.0
    return float(vals.head(k).mean())


def robust_signed_module_score(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if vals.size == 0:
        return 0.0
    pos = vals[vals > 0]
    neg = -vals[vals < 0]
    pos_score = float(np.mean(np.sort(pos)[-min(30, pos.size) :])) if pos.size else 0.0
    neg_score = float(np.mean(np.sort(neg)[-min(30, neg.size) :])) if neg.size else 0.0
    return pos_score if pos_score >= neg_score else -neg_score


def read_inputs(args: argparse.Namespace, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Reading full prediction: %s", args.full_prediction)
    full = pd.read_parquet(args.full_prediction).astype("float32")
    logger.info("Reading V4 baseline: %s", args.v4_baseline)
    baseline = pd.read_parquet(args.v4_baseline).astype("float32")
    logger.info("Reading graph residual: %s", args.graph_delta)
    graph_raw = pd.read_parquet(args.graph_delta).astype("float32")
    logger.info("Reading RNA matrix: %s", args.rna_matrix)
    rna = pd.read_parquet(args.rna_matrix).astype("float32")
    logger.info("Reading sample manifest: %s", args.sample_manifest)
    manifest = pd.read_csv(args.sample_manifest, sep="\t")
    return full, baseline, graph_raw, rna, manifest


def align_samples(
    full: pd.DataFrame,
    baseline: pd.DataFrame,
    graph_raw: pd.DataFrame,
    rna: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = manifest.copy()
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    manifest["has_os_survival"] = manifest["has_os_survival"].astype(bool)
    manifest["survival_time"] = pd.to_numeric(manifest["survival_time"], errors="coerce")
    manifest["survival_event"] = pd.to_numeric(manifest["survival_event"], errors="coerce")
    manifest = manifest.loc[manifest["has_os_survival"] & manifest["survival_time"].notna() & manifest["survival_event"].notna()].copy()
    manifest = manifest.sort_values(["tcga_sample_type_id", "sample_id"]).drop_duplicates("tcga_sample_type_id", keep="first")
    common = [s for s in manifest["sample_id"].tolist() if s in full.index and s in baseline.index and s in graph_raw.index and s in rna.index]
    manifest = manifest.set_index("sample_id").loc[common].reset_index()
    return full.loc[common], baseline.loc[common], graph_raw.loc[common], rna.loc[common], manifest


def compute_effect_tables(
    full: pd.DataFrame,
    baseline: pd.DataFrame,
    graph_raw: pd.DataFrame,
    rna: pd.DataFrame,
    manifest: pd.DataFrame,
    outdir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    site_cols = [c for c in full.columns if "|" in str(c)]
    site_cols = [c for c in site_cols if c in baseline.columns and c in graph_raw.columns]
    genes, sites = split_gene_site(site_cols)
    gene_to_sites = pd.DataFrame({"gene_site": site_cols, "gene": genes, "site": sites})

    rows: list[pd.DataFrame] = []
    t0 = time.time()
    cancers = sorted(manifest["project_id"].dropna().unique())
    for ci, cancer in enumerate(cancers, start=1):
        sample_ids = manifest.loc[manifest["project_id"].eq(cancer), "sample_id"].tolist()
        sub_meta = manifest.loc[manifest["sample_id"].isin(sample_ids)].copy()
        n = len(sample_ids)
        events = int(pd.to_numeric(sub_meta["survival_event"], errors="coerce").fillna(0).sum())
        logger.info("Cox %s (%d/%d): n=%d events=%d", cancer, ci, len(cancers), n, events)
        if n < 30 or events < 5:
            continue
        time_arr = sub_meta["survival_time"].to_numpy(float)
        event_arr = sub_meta["survival_event"].to_numpy(float)
        full_sub = full.loc[sample_ids, site_cols]
        base_sub = baseline.loc[sample_ids, site_cols]
        graph_scaled_sub = (full_sub - base_sub).astype("float32")
        rna_sub = rna.loc[sample_ids]

        cancer_rows: list[pd.DataFrame] = []
        for start in range(0, len(site_cols), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(site_cols))
            chunk_sites = site_cols[start:end]
            chunk_genes = genes[start:end]
            x_full = full_sub.iloc[:, start:end].to_numpy(dtype=np.float64)
            x_base = base_sub.iloc[:, start:end].to_numpy(dtype=np.float64)
            x_graph = graph_scaled_sub.iloc[:, start:end].to_numpy(dtype=np.float64)

            present_gene = [g if g in rna_sub.columns else None for g in chunk_genes]
            parent = np.zeros((n, len(chunk_sites)), dtype=np.float64)
            parent_valid = np.zeros(len(chunk_sites), dtype=bool)
            for j, g in enumerate(present_gene):
                if g is not None:
                    parent[:, j] = rna_sub[g].to_numpy(dtype=np.float64)
                    parent_valid[j] = True
                else:
                    parent[:, j] = np.nan

            res_full = cox_univariate_batch(x_full, time_arr, event_arr)
            res_base = cox_univariate_batch(x_base, time_arr, event_arr)
            res_graph = cox_univariate_batch(x_graph, time_arr, event_arr)
            res_parent = cox_univariate_batch(parent, time_arr, event_arr)
            graph_add = cox_bivariate_add_batch(x_base, x_graph, time_arr, event_arr, res_base["loglik"])
            site_add = cox_bivariate_add_batch(parent, x_full, time_arr, event_arr, res_parent["loglik"])

            frame = pd.DataFrame(
                {
                    "cancer": cancer,
                    "gene_site": chunk_sites,
                    "gene": chunk_genes,
                    "site": sites[start:end],
                    "n": n,
                    "events": events,
                    "cox_beta_full": res_full["beta"],
                    "cox_p_full": res_full["p"],
                    "full_chisq": res_full["wald_chisq"],
                    "full_lrt_chisq": res_full["lrt_chisq"],
                    "cox_beta_baseline": res_base["beta"],
                    "cox_p_baseline": res_base["p"],
                    "baseline_chisq": res_base["wald_chisq"],
                    "baseline_lrt_chisq": res_base["lrt_chisq"],
                    "graph_delta_beta": res_graph["beta"],
                    "graph_delta_p": res_graph["p"],
                    "graph_delta_chisq": res_graph["wald_chisq"],
                    "graph_delta_lrt_chisq": res_graph["lrt_chisq"],
                    "graph_delta_beta_adjusted_for_baseline": graph_add["beta_add"],
                    "graph_delta_p_adjusted_for_baseline": graph_add["add_p"],
                    "graph_delta_wald_chisq_adjusted_for_baseline": graph_add["add_wald_chisq"],
                    "likelihood_gain": graph_add["lrt_chisq"],
                    "gain_p": graph_add["lrt_p"],
                    "parent_mrna_present": parent_valid,
                    "parent_mrna_beta": res_parent["beta"],
                    "parent_mrna_p": res_parent["p"],
                    "parent_mrna_wald_chisq": res_parent["wald_chisq"],
                    "parent_mrna_lrt_chisq": res_parent["lrt_chisq"],
                    "site_beta_adjusted_for_parent_mrna": site_add["beta_add"],
                    "site_p_adjusted_for_parent_mrna": site_add["add_p"],
                    "site_wald_chisq_adjusted_for_parent_mrna": site_add["add_wald_chisq"],
                    "add_site_to_parent_mrna_lrt_chisq": site_add["lrt_chisq"],
                    "add_site_to_parent_mrna_lrt_p": site_add["lrt_p"],
                    "parent_beta_in_joint_model": site_add["beta_base"],
                    "predicted_site_minus_parent_mrna_wald_chisq": res_full["wald_chisq"] - res_parent["wald_chisq"],
                    "predicted_site_minus_parent_mrna_lrt_chisq": res_full["lrt_chisq"] - res_parent["lrt_chisq"],
                }
            )
            cancer_rows.append(frame)
        if cancer_rows:
            rows.append(pd.concat(cancer_rows, ignore_index=True))
        logger.info("Cox %s done in %.1fs", cancer, time.time() - t0)

    if not rows:
        raise RuntimeError("No Cox rows were generated.")
    out = pd.concat(rows, ignore_index=True)
    out["direction"] = np.where(pd.to_numeric(out["cox_beta_full"], errors="coerce") >= 0, "risk", "protective")
    for col in ["full", "baseline", "graph_delta"]:
        beta_col = "cox_beta_full" if col == "full" else f"{col}_beta" if col == "graph_delta" else "cox_beta_baseline"
        chisq_col = "full_chisq" if col == "full" else f"{col}_chisq" if col == "graph_delta" else "baseline_chisq"
        out[f"{col}_signed_z"] = np.sign(pd.to_numeric(out[beta_col], errors="coerce")) * np.sqrt(
            np.clip(pd.to_numeric(out[chisq_col], errors="coerce"), 0, np.inf)
        )
    out["analysis_evaluable"] = (out["n"] >= MIN_N_ANALYSIS) & (out["events"] >= MIN_EVENTS_ANALYSIS)
    out["manuscript_evaluable"] = (out["n"] >= MIN_N_MANUSCRIPT) & (out["events"] >= MIN_EVENTS_MANUSCRIPT)
    out["clinical_significant"] = out["analysis_evaluable"] & (pd.to_numeric(out["cox_p_full"], errors="coerce") < 0.05)
    out["graph_residual_significant"] = out["clinical_significant"] & (pd.to_numeric(out["gain_p"], errors="coerce") < 0.05)
    out["parent_mrna_independent"] = (
        out["clinical_significant"]
        & out["parent_mrna_present"].astype(bool)
        & (pd.to_numeric(out["add_site_to_parent_mrna_lrt_p"], errors="coerce") < 0.05)
    )
    out["site_over_mrna_direction"] = np.where(
        pd.to_numeric(out["predicted_site_minus_parent_mrna_wald_chisq"], errors="coerce") > 0,
        "site_greater",
        "parent_greater_or_equal",
    )
    out["gain_q_bh"] = np.nan
    out["add_site_to_parent_mrna_lrt_q_bh"] = np.nan
    out["cox_q_full_bh"] = np.nan
    for cancer, idx in out.groupby("cancer").groups.items():
        ii = np.asarray(list(idx))
        out.loc[ii, "gain_q_bh"] = bh_qvalues(pd.to_numeric(out.loc[ii, "gain_p"], errors="coerce").to_numpy())
        out.loc[ii, "add_site_to_parent_mrna_lrt_q_bh"] = bh_qvalues(
            pd.to_numeric(out.loc[ii, "add_site_to_parent_mrna_lrt_p"], errors="coerce").to_numpy()
        )
        out.loc[ii, "cox_q_full_bh"] = bh_qvalues(pd.to_numeric(out.loc[ii, "cox_p_full"], errors="coerce").to_numpy())

    prog_family_module = [module_label(g) for g in out["gene"]]
    out["rna_background_program"] = [x[0] for x in prog_family_module]
    out["pathway_family"] = [x[1] for x in prog_family_module]
    out["pathway_module"] = [x[2] for x in prog_family_module]

    table_dir = outdir / "tables"
    out.to_csv(table_dir / "full_tcga_scp682_main_graph_residual_clinical_gain.tsv.gz", sep="\t", index=False, compression="gzip")
    parent_cols = [
        "cancer",
        "gene_site",
        "gene",
        "site",
        "n",
        "events",
        "parent_mrna_present",
        "parent_mrna_beta",
        "parent_mrna_p",
        "parent_mrna_wald_chisq",
        "parent_mrna_lrt_chisq",
        "cox_beta_full",
        "cox_p_full",
        "full_chisq",
        "full_lrt_chisq",
        "predicted_site_minus_parent_mrna_wald_chisq",
        "predicted_site_minus_parent_mrna_lrt_chisq",
        "site_beta_adjusted_for_parent_mrna",
        "site_p_adjusted_for_parent_mrna",
        "site_wald_chisq_adjusted_for_parent_mrna",
        "add_site_to_parent_mrna_lrt_chisq",
        "add_site_to_parent_mrna_lrt_p",
        "add_site_to_parent_mrna_lrt_q_bh",
        "site_over_mrna_direction",
        "analysis_evaluable",
        "manuscript_evaluable",
    ]
    out[parent_cols].to_csv(
        table_dir / "full_tcga_scp682_main_predicted_phosphosite_vs_parent_mrna_clinical_gain.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    out.to_csv(table_dir / "full_tcga_scp682_main_architecture_effect_matrix.tsv.gz", sep="\t", index=False, compression="gzip")
    logger.info("Effect rows: %d; elapsed %.1fs", out.shape[0], time.time() - t0)
    return out


def build_project_summary(arch: pd.DataFrame, coverage_path: Path, outdir: Path) -> pd.DataFrame:
    arch = arch.copy()
    arch["full_q_sig"] = arch["analysis_evaluable"] & (pd.to_numeric(arch["cox_q_full_bh"], errors="coerce") < 0.05)
    arch["graph_gain_q_sig"] = (
        arch["analysis_evaluable"]
        & (pd.to_numeric(arch["cox_p_full"], errors="coerce") < 0.05)
        & (pd.to_numeric(arch["gain_q_bh"], errors="coerce") < 0.05)
    )
    arch["mrna_gain_q_sig"] = (
        arch["analysis_evaluable"]
        & (pd.to_numeric(arch["cox_p_full"], errors="coerce") < 0.05)
        & (pd.to_numeric(arch["add_site_to_parent_mrna_lrt_q_bh"], errors="coerce") < 0.05)
    )
    cov = pd.read_csv(coverage_path, sep="\t") if coverage_path.exists() else pd.DataFrame()
    summary = (
        arch.groupby("cancer")
        .agg(
            cox_rows=("gene_site", "size"),
            n=("n", "max"),
            events=("events", "max"),
            clinical_sites=("clinical_significant", "sum"),
            graph_residual_sites=("graph_residual_significant", "sum"),
            parent_mrna_independent_sites=("parent_mrna_independent", "sum"),
            clinical_sites_q=("full_q_sig", "sum"),
            graph_residual_sites_q=("graph_gain_q_sig", "sum"),
            parent_mrna_independent_sites_q=("mrna_gain_q_sig", "sum"),
            manuscript_evaluable=("manuscript_evaluable", "max"),
            median_full_chisq=("full_chisq", "median"),
            max_full_chisq=("full_chisq", "max"),
            max_graph_gain=("likelihood_gain", "max"),
            max_site_over_mrna_gain=("add_site_to_parent_mrna_lrt_chisq", "max"),
        )
        .reset_index()
        .rename(columns={"cancer": "project_id"})
    )
    summary["clinical_site_fraction"] = summary["clinical_sites"] / summary["cox_rows"].clip(lower=1)
    summary["graph_residual_site_fraction"] = summary["graph_residual_sites"] / summary["cox_rows"].clip(lower=1)
    summary["parent_mrna_independent_site_fraction"] = summary["parent_mrna_independent_sites"] / summary["cox_rows"].clip(lower=1)
    summary["clinical_site_q_fraction"] = summary["clinical_sites_q"] / summary["cox_rows"].clip(lower=1)
    summary["graph_residual_site_q_fraction"] = summary["graph_residual_sites_q"] / summary["cox_rows"].clip(lower=1)
    summary["parent_mrna_independent_site_q_fraction"] = summary["parent_mrna_independent_sites_q"] / summary["cox_rows"].clip(lower=1)
    if not cov.empty:
        summary = cov.merge(summary, on="project_id", how="left")
    summary.to_csv(outdir / "tables/full_tcga_scp682_main_project_effect_summary.tsv", sep="\t", index=False)
    q_summary = summary[
        [
            "project_id",
            "cox_rows",
            "clinical_sites",
            "clinical_sites_q",
            "graph_residual_sites",
            "graph_residual_sites_q",
            "parent_mrna_independent_sites",
            "parent_mrna_independent_sites_q",
            "clinical_site_fraction",
            "clinical_site_q_fraction",
            "graph_residual_site_q_fraction",
            "parent_mrna_independent_site_q_fraction",
        ]
    ].copy()
    q_summary.to_csv(outdir / "tables/full_tcga_scp682_main_project_effect_summary_with_q.tsv", sep="\t", index=False)
    return summary


def build_modules_and_nmf(arch: pd.DataFrame, outdir: Path, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    use = arch.loc[arch["analysis_evaluable"]].copy()
    use["abs_full_signed_z"] = pd.to_numeric(use["full_signed_z"], errors="coerce").abs()
    use["risk_score"] = np.where(use["full_signed_z"] > 0, use["abs_full_signed_z"], 0.0)
    use["protective_score"] = np.where(use["full_signed_z"] < 0, use["abs_full_signed_z"], 0.0)
    rep_idx = use.groupby(["cancer", "gene"])["abs_full_signed_z"].idxmax()
    reps = use.loc[rep_idx, ["cancer", "gene", "gene_site", "direction", "rna_background_program", "pathway_family", "pathway_module"]].rename(
        columns={"gene_site": "representative_site", "direction": "module_direction"}
    )
    modules = (
        use.groupby(["cancer", "gene"])
        .agg(
            n_sites_total=("gene_site", "nunique"),
            n_clinical_sites=("clinical_significant", "sum"),
            n_graph_residual_sites=("graph_residual_significant", "sum"),
            n_site_over_mrna_sites=("parent_mrna_independent", "sum"),
            max_full_chisq=("full_chisq", "max"),
            max_likelihood_gain=("likelihood_gain", "max"),
            max_site_over_mrna_lrt_gain=("add_site_to_parent_mrna_lrt_chisq", "max"),
            mean_top3_site_over_mrna_lrt_gain=("add_site_to_parent_mrna_lrt_chisq", top_mean_abs),
            max_parent_mrna_chisq=("parent_mrna_wald_chisq", "max"),
            max_site_minus_mrna_wald_delta=("predicted_site_minus_parent_mrna_wald_chisq", "max"),
            mean_risk_score=("risk_score", "mean"),
            mean_protective_score=("protective_score", "mean"),
            max_risk_score=("risk_score", "max"),
            max_protective_score=("protective_score", "max"),
            dominant_signed_cox_score=("full_signed_z", robust_signed_module_score),
        )
        .reset_index()
    )
    modules = modules.merge(reps, on=["cancer", "gene"], how="left")
    modules["module_key"] = modules["cancer"] + "|" + modules["gene"]
    modules["module_selected_for_fig5"] = (
        (modules["n_clinical_sites"] > 0) & ((modules["n_graph_residual_sites"] > 0) | (modules["n_site_over_mrna_sites"] > 0))
    )
    modules.to_csv(outdir / "tables/full_tcga_scp682_main_residual_module_matrix.tsv", sep="\t", index=False)

    selected = modules.loc[modules["module_selected_for_fig5"]].copy()
    selected = selected.sort_values(["max_full_chisq", "max_likelihood_gain"], ascending=False).head(40000).reset_index(drop=True)
    for prog in PROGRAMS:
        selected[f"program_{prog}"] = (selected["rna_background_program"] == prog).astype(float)
    feature_cols = [
        "n_clinical_sites",
        "n_graph_residual_sites",
        "n_site_over_mrna_sites",
        "max_full_chisq",
        "max_likelihood_gain",
        "max_site_over_mrna_lrt_gain",
        "mean_top3_site_over_mrna_lrt_gain",
        "max_parent_mrna_chisq",
        "max_site_minus_mrna_wald_delta",
        "mean_risk_score",
        "mean_protective_score",
        "max_risk_score",
        "max_protective_score",
    ] + [f"program_{p}" for p in PROGRAMS]
    X = selected[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    cap = X.quantile(0.995).replace(0, 1.0)
    Xn = (X / cap).clip(0, 1)
    nmf_input = pd.concat([selected.drop(columns=[c for c in feature_cols if c in selected.columns]), Xn.add_prefix("nmf_")], axis=1)
    nmf_input.to_csv(outdir / "tables/full_tcga_scp682_main_residual_module_nmf_input.tsv", sep="\t", index=False)

    if NMF is None or nmf_input.empty:
        weights = pd.DataFrame()
        components = pd.DataFrame()
    else:
        n_components = min(6, max(2, int(math.sqrt(max(2, Xn.shape[0] // 450)))))
        model = NMF(n_components=n_components, init="nndsvda", random_state=682, max_iter=800)
        W = model.fit_transform(Xn.to_numpy(dtype=float))
        H = model.components_
        comp_names = [f"NMF{i+1}" for i in range(n_components)]
        weights = pd.concat(
            [
                selected[["cancer", "gene", "module_key", "representative_site", "pathway_module", "rna_background_program"]].reset_index(drop=True),
                pd.DataFrame(W, columns=comp_names),
            ],
            axis=1,
        )
        weights["dominant_component"] = weights[comp_names].idxmax(axis=1)
        weights["dominant_component_weight"] = weights[comp_names].max(axis=1)
        comp_rows = []
        nmf_feature_cols = Xn.columns.tolist()
        for i, comp in enumerate(H):
            order = np.argsort(-comp)
            top_features = [nmf_feature_cols[j] for j in order[:10]]
            top_idx = np.argsort(-W[:, i])[:10]
            top_modules = [f"{selected.loc[j, 'cancer']} {selected.loc[j, 'gene']}" for j in top_idx]
            label = "; ".join([x.replace("nmf_", "") for x in top_features[:4]])
            for rank, j in enumerate(order[:18], start=1):
                comp_rows.append(
                    {
                        "component": comp_names[i],
                        "component_label": label,
                        "feature_rank": rank,
                        "feature": nmf_feature_cols[j],
                        "feature_weight": float(comp[j]),
                        "top_modules": ";".join(top_modules),
                        "top_features": ";".join(top_features),
                    }
                )
        components = pd.DataFrame(comp_rows)
        weights.to_csv(outdir / "tables/full_tcga_scp682_main_residual_module_nmf_weights.tsv", sep="\t", index=False)
        components.to_csv(outdir / "tables/full_tcga_scp682_main_residual_module_nmf_components.tsv", sep="\t", index=False)
    logger.info("Module rows=%d; NMF input rows=%d", modules.shape[0], selected.shape[0])
    return modules, weights, components


def build_candidate_shortlist(modules: pd.DataFrame, weights: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    mod = modules.copy()
    mod["candidate_score"] = (
        np.log1p(pd.to_numeric(mod["max_full_chisq"], errors="coerce").fillna(0))
        + np.log1p(pd.to_numeric(mod["max_likelihood_gain"], errors="coerce").fillna(0))
        + np.log1p(pd.to_numeric(mod["max_site_over_mrna_lrt_gain"], errors="coerce").fillna(0))
        + 0.4 * np.log1p(pd.to_numeric(mod["n_clinical_sites"], errors="coerce").fillna(0))
        + 0.7 * np.log1p(pd.to_numeric(mod["n_graph_residual_sites"], errors="coerce").fillna(0))
        + 0.7 * np.log1p(pd.to_numeric(mod["n_site_over_mrna_sites"], errors="coerce").fillna(0))
    )
    if not weights.empty:
        mod = mod.merge(weights[["module_key", "dominant_component", "dominant_component_weight"]], on="module_key", how="left")
    shortlist = (
        mod.loc[mod["module_selected_for_fig5"]]
        .sort_values("candidate_score", ascending=False)
        .head(200)
        .reset_index(drop=True)
    )
    shortlist.insert(0, "rank", np.arange(1, shortlist.shape[0] + 1))
    shortlist.to_csv(outdir / "tables/full_tcga_scp682_main_candidate_biology_shortlist.tsv", sep="\t", index=False)
    return shortlist


def build_fig5b_module_tables(arch: pd.DataFrame, outdir: Path) -> None:
    df = arch.loc[
        arch["cancer"].isin(FOUR_CANCERS)
        & arch["clinical_significant"]
        & (arch["graph_residual_significant"] | arch["parent_mrna_independent"])
    ].copy()
    df["signed_cox_score"] = pd.to_numeric(df["full_signed_z"], errors="coerce")
    df["risk_site"] = df["signed_cox_score"] > 0
    df["protective_site"] = df["signed_cox_score"] < 0
    df["cancer_short"] = df["cancer"].str.replace("TCGA-", "", regex=False)
    grouped_rows = []
    for (module, cancer), sub in df.groupby(["pathway_module", "cancer_short"], sort=False):
        score = robust_signed_module_score(sub["signed_cox_score"])
        top = sub.assign(abs_score=sub["signed_cox_score"].abs()).sort_values("abs_score", ascending=False).head(12)
        grouped_rows.append(
            {
                "pathway_module": module,
                "cancer": cancer,
                "module_signed_cox_score": score,
                "n_sites": int(sub["gene_site"].nunique()),
                "n_risk_sites": int(sub["risk_site"].sum()),
                "n_protective_sites": int(sub["protective_site"].sum()),
                "representative_sites": ";".join(top["gene_site"].astype(str).str.replace("|", " ", regex=False).tolist()),
                "representative_genes": ";".join(top["gene"].astype(str).drop_duplicates().head(12).tolist()),
            }
        )
    module_long = pd.DataFrame(grouped_rows)
    if module_long.empty:
        matrix = pd.DataFrame(columns=[c.replace("TCGA-", "") for c in FOUR_CANCERS])
        counts = matrix.copy()
        meta = pd.DataFrame()
    else:
        keep = (
            module_long.groupby("pathway_module")
            .agg(total_sites=("n_sites", "sum"), max_sites=("n_sites", "max"), max_score=("module_signed_cox_score", lambda x: float(np.nanmax(np.abs(x)))))
            .query("total_sites >= 20 or max_sites >= 8 or max_score >= 5.0")
            .index
        )
        module_long = module_long.loc[module_long["pathway_module"].isin(keep)].copy()
        cancers_short = [c.replace("TCGA-", "") for c in FOUR_CANCERS]
        matrix = module_long.pivot_table(index="pathway_module", columns="cancer", values="module_signed_cox_score", aggfunc="first", fill_value=0.0)
        counts = module_long.pivot_table(index="pathway_module", columns="cancer", values="n_sites", aggfunc="sum", fill_value=0)
        for cancer in cancers_short:
            if cancer not in matrix.columns:
                matrix[cancer] = 0.0
            if cancer not in counts.columns:
                counts[cancer] = 0
        matrix = matrix[cancers_short]
        counts = counts[cancers_short]
        meta_rows = []
        for module, sub in module_long.groupby("pathway_module", sort=False):
            dominant = sub.assign(abs_score=sub["module_signed_cox_score"].abs()).sort_values("abs_score", ascending=False).iloc[0]
            total_risk = int(sub["n_risk_sites"].sum())
            total_protect = int(sub["n_protective_sites"].sum())
            meta_rows.append(
                {
                    "pathway_module": module,
                    "dominant_cancer": dominant["cancer"],
                    "dominant_signed_cox_score": dominant["module_signed_cox_score"],
                    "total_sites": int(sub["n_sites"].sum()),
                    "total_risk_sites": total_risk,
                    "total_protective_sites": total_protect,
                    "module_direction": "risk" if total_risk >= total_protect else "protective",
                    "representative_sites": dominant["representative_sites"],
                    "representative_genes": dominant["representative_genes"],
                }
            )
        meta = pd.DataFrame(meta_rows).set_index("pathway_module")
        order = meta.sort_values(["module_direction", "dominant_signed_cox_score"], ascending=[True, False]).index
        matrix = matrix.reindex(order)
        counts = counts.reindex(order)

    table_dir = outdir / "tables"
    df.to_csv(table_dir / "full_tcga_scp682_main_fig5b_pathway_module_site_events.tsv.gz", sep="\t", index=False, compression="gzip")
    module_long.to_csv(table_dir / "full_tcga_scp682_main_fig5b_pathway_module_long.tsv", sep="\t", index=False)
    matrix.to_csv(table_dir / "full_tcga_scp682_main_fig5b_pathway_module_cox_heatmap_matrix.tsv", sep="\t")
    counts.to_csv(table_dir / "full_tcga_scp682_main_fig5b_pathway_module_site_counts.tsv", sep="\t")
    meta.to_csv(table_dir / "full_tcga_scp682_main_fig5b_pathway_module_modules.tsv", sep="\t")


def write_report(
    args: argparse.Namespace,
    outdir: Path,
    arch: pd.DataFrame,
    project_summary: pd.DataFrame,
    modules: pd.DataFrame,
    nmf_input_rows: int,
    shortlist: pd.DataFrame,
    elapsed: float,
) -> None:
    manifest = {
        "input_full_prediction": str(args.full_prediction),
        "input_v4_baseline": str(args.v4_baseline),
        "input_graph_delta": str(args.graph_delta),
        "input_rna_matrix": str(args.rna_matrix),
        "input_sample_manifest": str(args.sample_manifest),
        "output_dir": str(outdir),
        "rows_architecture_effect_matrix": int(arch.shape[0]),
        "cancers_with_cox_rows": int(arch["cancer"].nunique()),
        "unique_sites": int(arch["gene_site"].nunique()),
        "analysis_evaluable_rows": int(arch["analysis_evaluable"].sum()),
        "manuscript_evaluable_rows": int(arch["manuscript_evaluable"].sum()),
        "clinical_significant_rows": int(arch["clinical_significant"].sum()),
        "graph_residual_significant_rows": int(arch["graph_residual_significant"].sum()),
        "parent_mrna_independent_rows": int(arch["parent_mrna_independent"].sum()),
        "project_summary": str(outdir / "tables/full_tcga_scp682_main_project_effect_summary.tsv"),
        "modules": str(outdir / "tables/full_tcga_scp682_main_residual_module_matrix.tsv"),
        "nmf_input_rows": int(nmf_input_rows),
        "shortlist_rows": int(shortlist.shape[0]),
        "elapsed_seconds": elapsed,
    }
    (outdir / "reports/full_tcga_scp682_main_fig5_support_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    brief = [
        "# Fig 5 full-TCGA SCP682 support rebuild",
        "",
        f"Architecture-effect rows: {arch.shape[0]:,}",
        f"Cancers with Cox rows: {arch['cancer'].nunique()}",
        f"Unique phosphosites: {arch['gene_site'].nunique():,}",
        f"Analysis-evaluable rows: {int(arch['analysis_evaluable'].sum()):,}",
        f"Clinical significant rows: {int(arch['clinical_significant'].sum()):,}",
        f"Graph-residual significant rows: {int(arch['graph_residual_significant'].sum()):,}",
        f"Parent-mRNA independent rows: {int(arch['parent_mrna_independent'].sum()):,}",
        f"Module rows: {modules.shape[0]:,}",
        f"NMF input rows: {nmf_input_rows:,}",
        f"Elapsed seconds: {elapsed:.1f}",
        "",
        "The graph-residual gain is the likelihood-ratio gain from adding the scaled graph residual",
        "(full prediction minus V4 baseline) to the V4 baseline in a Cox model.",
        "The parent-mRNA gain is the likelihood-ratio gain from adding the predicted phosphosite",
        "to its parent-gene mRNA in a Cox model.",
    ]
    (outdir / "reports/full_tcga_scp682_main_fig5_support_summary.md").write_text("\n".join(brief), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repred-dir", type=Path, default=DEFAULT_REPRED)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--full-prediction", type=Path, default=None)
    ap.add_argument("--v4-baseline", type=Path, default=None)
    ap.add_argument("--graph-delta", type=Path, default=None)
    ap.add_argument("--rna-matrix", type=Path, default=None)
    ap.add_argument("--sample-manifest", type=Path, default=None)
    ap.add_argument("--project-coverage", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repred = args.repred_dir
    args.full_prediction = args.full_prediction or repred / "predictions/tcga_full_scp682_predicted_phosphosite.parquet"
    args.v4_baseline = args.v4_baseline or repred / "predictions/scp682_v4_baseline.parquet"
    args.graph_delta = args.graph_delta or repred / "predictions/scp682_exact_scnet_gnn_delta.parquet"
    args.rna_matrix = args.rna_matrix or repred / "intermediate/tcga_gdc_star_tpm_model_genes_full.parquet"
    args.sample_manifest = args.sample_manifest or repred / "tables/tcga_scp682_prediction_sample_manifest.tsv"
    args.project_coverage = args.project_coverage or repred / "tables/tcga_scp682_prediction_project_coverage.tsv"
    outdir = args.output_dir
    ensure_dirs(outdir)
    logger = setup_logger(outdir)
    t0 = time.time()
    logger.info("Fig 5 support rebuild started")
    for p in [args.full_prediction, args.v4_baseline, args.graph_delta, args.rna_matrix, args.sample_manifest, args.project_coverage]:
        logger.info("Input exists=%s path=%s", p.exists(), p)
        if not p.exists():
            raise FileNotFoundError(str(p))
    full, baseline, graph_raw, rna, manifest = read_inputs(args, logger)
    full, baseline, graph_raw, rna, manifest = align_samples(full, baseline, graph_raw, rna, manifest)
    logger.info("Aligned samples=%d; sites=%d; RNA genes=%d; projects=%d", full.shape[0], full.shape[1], rna.shape[1], manifest["project_id"].nunique())
    arch = compute_effect_tables(full, baseline, graph_raw, rna, manifest, outdir, logger)
    project_summary = build_project_summary(arch, args.project_coverage, outdir)
    modules, weights, components = build_modules_and_nmf(arch, outdir, logger)
    shortlist = build_candidate_shortlist(modules, weights, outdir)
    build_fig5b_module_tables(arch, outdir)
    nmf_input_path = outdir / "tables/full_tcga_scp682_main_residual_module_nmf_input.tsv"
    nmf_rows = int(pd.read_csv(nmf_input_path, sep="\t", usecols=["module_key"]).shape[0]) if nmf_input_path.exists() else 0
    write_report(args, outdir, arch, project_summary, modules, nmf_rows, shortlist, time.time() - t0)
    logger.info("Fig 5 support rebuild finished in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
