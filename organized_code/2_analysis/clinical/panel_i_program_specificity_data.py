#!/usr/bin/env python3
"""Collapse phospho-specific modules into program-level specificity tables."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from statsmodels.stats.multitest import multipletests


ROOT = Path("E:/data/gongke/TCGA-TCPA")
FIG5_TABLES = ROOT / "paper_final" / "fig5" / "source_data" / "tables"
PTMSIGDB_GMT = (
    ROOT
    / "04_figures"
    / "20260528_fig5_v2"
    / "references"
    / "ptmsigdb_v2_0_0"
    / "ptm.sig.db.all.flanking.human.v2.0.0.gmt"
)
PHOSPHOSIGNOR_ALL = ROOT / "paper_final" / "fig5" / "references" / "PhosphoSIGNOR_all_Data.tsv"


PROGRAM_MAP = {
    "M10": ("Cell-cycle", "Cell-cycle / CDK-PLK", "cell cycle"),
    "M08": ("Cell-cycle", "Cell-cycle / CDK-PLK", "cell cycle"),
    "M05": ("Cell-cycle", "Cell-cycle / CDK-PLK", "cell cycle"),
    "M03": ("Cell-cycle", "Cell-cycle / CDK-PLK", "cell cycle"),
    "M04": ("PI3K-AKT", "PI3K-AKT", "growth survival"),
    "M12": ("PI3K-AKT", "PI3K-AKT", "growth survival"),
    "M13": ("MAPK", "MAPK / ERK", "MAPK"),
    "M14": ("MAPK", "MAPK / ERK", "MAPK"),
    "M11": ("PKC", "PKC / phorbol-response", "PKC"),
    "M06": ("PKC", "PKC / phorbol-response", "PKC"),
    "M09": ("PKC", "PKC / phorbol-response", "PKC"),
    "M07": ("EEF2K-translation-stress", "EEF2K / translation stress", "translation stress"),
    "M01": ("Unassigned", "weakly annotated", "unassigned"),
    "M02": ("Unassigned", "weakly annotated", "unassigned"),
}

PROGRAM_ORDER = [
    "Cell-cycle",
    "PI3K-AKT",
    "MAPK",
    "PKC",
    "EEF2K-translation-stress",
    "Unassigned",
]

PROGRAM_CLASS_COLORS = {
    "cell cycle": "#F6C8B6",
    "growth survival": "#DBDDEF",
    "MAPK": "#C1D8E9",
    "PKC": "#92B1D9",
    "translation stress": "#D4D4D4",
    "unassigned": "#BDBDBD",
}

CPTAC_TO_TCGA = {
    "CCRCC": "KIRC",
    "LUAD": "LUAD",
    "HNSCC": "HNSC",
    "LSCC": "LUSC",
    "PDA": "PAAD",
    "BRCA": "BRCA",
    "OV": "OV",
    "UCEC": "UCEC",
    "GBM": "GBM",
    "STAD": "STAD",
}

MIN_EVENTS_FOR_CALL = 50
PROGRAM_Q_CUTOFF = 0.05
SPECIFICITY_Q_CUTOFF = 0.05
MIN_ABS_PROGRAM_Z = 1.96
SHARED_MIN_CANCERS = 5


def bh_q(values) -> np.ndarray:
    pvals = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    out = np.full(pvals.shape[0], np.nan)
    ok = np.isfinite(pvals)
    if ok.sum() > 0:
        out[ok] = multipletests(pvals[ok], method="fdr_bh")[1]
    return out


def parse_phosphosignor_functional_signs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    rows = []
    site_re = re.compile(r"^([A-Za-z0-9_.-]+)_ph(Ser|Thr|Tyr)(\d+)$")
    aa = {"Ser": "S", "Thr": "T", "Tyr": "Y"}
    for _, row in df.iterrows():
        entity_a = str(row.get("entityA_name", "")).strip()
        entity_b = str(row.get("entityB_name", "")).strip()
        effect = str(row.get("effect", "")).strip().lower()
        match = site_re.match(entity_a)
        if match is None or "_ph" in entity_b:
            continue
        gene, residue, position = match.groups()
        if entity_b and entity_b.upper() != gene.upper():
            continue
        if effect == "up-regulates":
            sign = 1
        elif effect == "down-regulates":
            sign = -1
        else:
            continue
        rows.append(
            {
                "gene_site": f"{gene.upper()}|{aa[residue]}{position}",
                "functional_sign": sign,
                "effect": effect,
                "pmid": str(row.get("pmid", "")),
                "signor_id": str(row.get("signor_id", "")),
                "signor_score": float(row.get("signor-score", 0) or 0),
            }
        )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "gene_site",
                "functional_sign",
                "direction_call",
                "n_functional_edges",
                "n_up_edges",
                "n_down_edges",
                "weighted_direction_score",
                "pmids",
                "signor_ids",
                "direction_source",
            ]
        )
    out = []
    for site, grp in raw.groupby("gene_site"):
        weights = grp["signor_score"].fillna(0).to_numpy(dtype=float)
        weights = np.where(weights > 0, weights, 1.0)
        signs = grp["functional_sign"].to_numpy(dtype=float)
        score = float(np.sum(weights * signs))
        if score == 0:
            continue
        sign = 1 if score > 0 else -1
        out.append(
            {
                "gene_site": site,
                "functional_sign": sign,
                "direction_call": "activation" if sign > 0 else "inhibition",
                "n_functional_edges": int(grp.shape[0]),
                "n_up_edges": int((grp["functional_sign"] > 0).sum()),
                "n_down_edges": int((grp["functional_sign"] < 0).sum()),
                "weighted_direction_score": score,
                "pmids": ";".join(sorted(set(grp["pmid"].astype(str)))),
                "signor_ids": ";".join(sorted(set(grp["signor_id"].astype(str)))),
                "direction_source": "PhosphoSIGNOR functional phosphosite effect",
            }
        )
    return pd.DataFrame(out)


def orientation_audit(rows: pd.DataFrame, sites: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    functional_signs = parse_phosphosignor_functional_signs(PHOSPHOSIGNOR_ALL)
    sign_map = functional_signs.set_index("gene_site")["functional_sign"].to_dict()
    out = []
    for _, row in rows.iterrows():
        row_label = row["row_label"]
        signature = row["top_signature"]
        module_sites = sites.loc[sites["row_label"].eq(row_label)].copy()
        anchors = []
        anchor_sites = []
        for _, srow in module_sites.iterrows():
            site = str(srow["gene_site"]).upper()
            if site in sign_map:
                anchors.append(float(srow["site_weight"]) * int(sign_map[site]))
                anchor_sites.append(site)
        anchors = np.asarray(anchors, dtype=float)
        n = int(len(anchors))
        score = float(np.nansum(anchors)) if n else np.nan
        orientation_sign = int(np.sign(score)) if np.isfinite(score) and score != 0 else 1
        if n >= 2 and np.isfinite(score) and score != 0:
            jack = np.array([score - anchors[i] for i in range(n)])
            flip_rate = float(np.mean(np.sign(jack) != np.sign(score)))
            min_abs_jackknife_score = float(np.nanmin(np.abs(jack)))
        else:
            flip_rate = np.nan
            min_abs_jackknife_score = np.nan
        if n >= 10 and (not np.isfinite(flip_rate) or flip_rate == 0):
            confidence = "high"
        elif n >= 5 and (not np.isfinite(flip_rate) or flip_rate == 0):
            confidence = "moderate"
        else:
            confidence = "weak"
        module_id = row["module_id"]
        program_id, program_label, program_class = PROGRAM_MAP.get(
            module_id, ("Unassigned", "weakly annotated", "unassigned")
        )
        out.append(
            {
                "module_id": module_id,
                "row_label": row_label,
                "program_id": program_id,
                "program_label": program_label,
                "program_class": program_class,
                "top_signature": signature,
                "naming_source": "PTMsigDB member overlap; signature direction ignored",
                "direction_source": "PhosphoSIGNOR functional phosphosite effect",
                "orientation_sites": n,
                "orientation_score": score,
                "orientation_sign": orientation_sign,
                "functional_anchor_sites": ";".join(anchor_sites),
                "jackknife_flip_rate": flip_rate,
                "min_abs_jackknife_score": min_abs_jackknife_score,
                "orientation_confidence": confidence,
            }
        )
    return pd.DataFrame(out), functional_signs


def correlation_audit(effect: pd.DataFrame, module_map: pd.DataFrame) -> pd.DataFrame:
    orient = module_map.set_index("row_label")["orientation_sign"].to_dict()
    mat = effect.set_index("row_label").copy()
    for label, sign in orient.items():
        if label in mat.index and np.isfinite(sign):
            mat.loc[label] = pd.to_numeric(mat.loc[label], errors="coerce") * float(sign)
    rows = []
    for program_id, grp in module_map.groupby("program_id"):
        labels = grp["row_label"].tolist()
        if len(labels) < 2:
            continue
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                x = pd.to_numeric(mat.loc[labels[i]], errors="coerce")
                y = pd.to_numeric(mat.loc[labels[j]], errors="coerce")
                ok = x.notna() & y.notna()
                r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= 4 else np.nan
                rows.append(
                    {
                        "program_id": program_id,
                        "module_a": labels[i],
                        "module_b": labels[j],
                        "cross_cancer_effect_r": r,
                        "oversplit_if_abs_r_ge_0_7": bool(np.isfinite(r) and abs(r) >= 0.7),
                    }
                )
    return pd.DataFrame(rows)


def fixed_effect(beta: np.ndarray, se: np.ndarray) -> tuple[float, float, float, float]:
    beta = np.asarray(beta, dtype=float)
    se = np.asarray(se, dtype=float)
    ok = np.isfinite(beta) & np.isfinite(se) & (se > 0)
    if ok.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan
    w = 1.0 / np.square(se[ok])
    b = float(np.sum(w * beta[ok]) / np.sum(w))
    s = float(math.sqrt(1.0 / np.sum(w)))
    z = b / s if s > 0 else np.nan
    p = 2.0 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
    return b, s, z, p


def program_cox(cox: pd.DataFrame, module_map: pd.DataFrame) -> pd.DataFrame:
    dat = cox.merge(
        module_map[["row_label", "program_id", "program_label", "program_class", "orientation_sign", "orientation_confidence"]],
        on="row_label",
        how="left",
    )
    dat["orientation_sign"] = pd.to_numeric(dat["orientation_sign"], errors="coerce").fillna(1.0)
    dat["oriented_module_beta"] = dat["module_beta"] * dat["orientation_sign"]
    dat["oriented_module_z"] = dat["module_z"] * dat["orientation_sign"]
    rows = []
    for (program_id, cancer), grp in dat.groupby(["program_id", "cancer"], sort=False):
        b, s, z, p = fixed_effect(grp["oriented_module_beta"].to_numpy(), grp["module_se"].to_numpy())
        program_label = grp["program_label"].dropna().iloc[0]
        program_class = grp["program_class"].dropna().iloc[0]
        rows.append(
            {
                "program_id": program_id,
                "program_label": program_label,
                "program_class": program_class,
                "cancer": cancer,
                "n_modules": int(grp["row_label"].nunique()),
                "n": int(np.nanmax(grp["n"])),
                "events": int(np.nanmax(grp["events"])),
                "program_beta": b,
                "program_se": s,
                "program_z": z,
                "program_p": p,
                "modules": ";".join(grp["module_id"].dropna().astype(str).unique()) if "module_id" in grp.columns else "",
                "orientation_applied": "PhosphoSIGNOR functional sign",
            }
        )
    out = pd.DataFrame(rows)
    out["program_q"] = bh_q(out["program_p"])
    return out


def add_specificity(program: pd.DataFrame) -> pd.DataFrame:
    out = program.copy()
    out["cross_cancer_q_stat"] = np.nan
    out["cross_cancer_q_p"] = np.nan
    out["specificity_z"] = np.nan
    out["specificity_p"] = np.nan
    for program_id, grp in out.groupby("program_id"):
        valid = grp.loc[np.isfinite(grp["program_beta"]) & np.isfinite(grp["program_se"]) & (grp["program_se"] > 0)].copy()
        if valid.shape[0] >= 2:
            w = 1.0 / np.square(valid["program_se"].to_numpy(dtype=float))
            beta = valid["program_beta"].to_numpy(dtype=float)
            pooled = np.sum(w * beta) / np.sum(w)
            q_stat = float(np.sum(w * np.square(beta - pooled)))
            q_p = float(chi2.sf(q_stat, df=valid.shape[0] - 1))
            out.loc[out["program_id"].eq(program_id), "cross_cancer_q_stat"] = q_stat
            out.loc[out["program_id"].eq(program_id), "cross_cancer_q_p"] = q_p
        for idx, row in grp.iterrows():
            others = grp.loc[grp.index != idx].copy()
            others = others.loc[np.isfinite(others["program_beta"]) & np.isfinite(others["program_se"]) & (others["program_se"] > 0)]
            if not np.isfinite(row["program_beta"]) or not np.isfinite(row["program_se"]) or row["program_se"] <= 0 or others.empty:
                continue
            w = 1.0 / np.square(others["program_se"].to_numpy(dtype=float))
            other_beta = float(np.sum(w * others["program_beta"].to_numpy(dtype=float)) / np.sum(w))
            other_se = float(math.sqrt(1.0 / np.sum(w)))
            diff_se = math.sqrt(float(row["program_se"]) ** 2 + other_se**2)
            z = (float(row["program_beta"]) - other_beta) / diff_se
            out.loc[idx, "specificity_z"] = z
            out.loc[idx, "specificity_p"] = 2.0 * norm.sf(abs(z))
    out["cross_cancer_q_fdr"] = bh_q(out.drop_duplicates("program_id")["cross_cancer_q_p"]).repeat(
        out.groupby("program_id").size().reindex(out["program_id"].drop_duplicates()).to_numpy()
    ) if False else np.nan
    q_by_program = out.drop_duplicates("program_id")[["program_id", "cross_cancer_q_p"]].copy()
    q_by_program["cross_cancer_q_fdr"] = bh_q(q_by_program["cross_cancer_q_p"])
    out = out.drop(columns=["cross_cancer_q_fdr"]).merge(q_by_program[["program_id", "cross_cancer_q_fdr"]], on="program_id", how="left")
    out["specificity_q"] = bh_q(out["specificity_p"])
    out["program_call"] = "not supported"
    annotated = out["program_id"].ne("Unassigned")
    powered = out["events"].fillna(0) >= MIN_EVENTS_FOR_CALL
    sig = (
        (out["program_q"] < PROGRAM_Q_CUTOFF)
        & (out["program_z"].abs() >= MIN_ABS_PROGRAM_Z)
        & powered
        & annotated
    )
    restricted = sig & (out["specificity_q"] < SPECIFICITY_Q_CUTOFF)
    out.loc[restricted & (out["program_z"] > 0), "program_call"] = "cancer-restricted risk"
    out.loc[restricted & (out["program_z"] < 0), "program_call"] = "cancer-restricted protection"
    for program_id, grp in out.groupby("program_id"):
        if program_id == "Unassigned":
            continue
        grp_sig = (
            (grp["program_q"] < PROGRAM_Q_CUTOFF)
            & (grp["events"].fillna(0) >= MIN_EVENTS_FOR_CALL)
        )
        sig_pos = (grp_sig & (grp["program_z"] >= MIN_ABS_PROGRAM_Z)).sum()
        sig_neg = (grp_sig & (grp["program_z"] <= -MIN_ABS_PROGRAM_Z)).sum()
        if sig_pos >= SHARED_MIN_CANCERS:
            idx = grp.index[grp_sig & (grp["program_z"] >= MIN_ABS_PROGRAM_Z) & (grp["program_call"].eq("not supported"))]
            out.loc[idx, "program_call"] = "shared risk"
        if sig_neg >= SHARED_MIN_CANCERS:
            idx = grp.index[grp_sig & (grp["program_z"] <= -MIN_ABS_PROGRAM_Z) & (grp["program_call"].eq("not supported"))]
            out.loc[idx, "program_call"] = "shared protection"
    return out


def program_rows(program: pd.DataFrame, module_map: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for program_id in PROGRAM_ORDER:
        grp = program.loc[program["program_id"].eq(program_id)]
        if grp.empty:
            continue
        mm = module_map.loc[module_map["program_id"].eq(program_id)]
        n_restricted = int(grp["program_call"].astype(str).str.startswith("cancer-restricted").sum())
        n_shared = int(grp["program_call"].astype(str).str.startswith("shared").sum())
        conf_order = {"high": 3, "moderate": 2, "weak": 1}
        conf = mm["orientation_confidence"].map(conf_order).min()
        conf_label = {3: "high", 2: "moderate", 1: "weak"}.get(int(conf) if np.isfinite(conf) else 1, "weak")
        if program_id == "Unassigned":
            conf_label = "not interpreted"
        rows.append(
            {
                "program_id": program_id,
                "program_label": grp["program_label"].iloc[0],
                "program_class": grp["program_class"].iloc[0],
                "modules": ";".join(mm["module_id"]),
                "module_labels": "; ".join(mm["row_label"]),
                "n_modules": int(mm["module_id"].nunique()),
                "orientation_sites_total": int(mm["orientation_sites"].sum()),
                "orientation_confidence": conf_label,
                "max_abs_z": float(np.nanmax(np.abs(grp["program_z"]))),
                "mean_abs_z": float(np.nanmean(np.abs(grp["program_z"]))),
                "cross_cancer_q_p": float(grp["cross_cancer_q_p"].dropna().iloc[0]) if grp["cross_cancer_q_p"].notna().any() else np.nan,
                "cross_cancer_q_fdr": float(grp["cross_cancer_q_fdr"].dropna().iloc[0]) if grp["cross_cancer_q_fdr"].notna().any() else np.nan,
                "n_cancer_restricted": n_restricted,
                "n_shared": n_shared,
            }
        )
    return pd.DataFrame(rows)


def cptac_audit(module_map: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    concordance_path = (
        ROOT
        / "02_results"
        / "model_validation"
        / "20260427_cptac_adversarial_prediction_concordance_v1"
        / "tables"
        / "target_level_predicted_vs_measured_spearman_by_cancer.tsv"
    )
    survival_path = (
        ROOT
        / "02_results"
        / "model_validation"
        / "20260427_cptac_adversarial_survival_cohorts_v1"
        / "tables"
        / "predicted_phosphosite_os_logrank_screen_reliable_targets.tsv"
    )
    rows = []
    site_program = sites.merge(module_map[["row_label", "program_id", "program_label"]], on="row_label", how="left")
    site_program = site_program[["program_id", "program_label", "gene_site"]].drop_duplicates()
    if concordance_path.exists():
        conc = pd.read_csv(concordance_path, sep="\t")
        conc = conc.loc[conc["group"].isin(CPTAC_TO_TCGA.keys())].copy()
        conc["tcga_cancer"] = conc["group"].map(CPTAC_TO_TCGA)
        conc = conc.merge(site_program, left_on="target", right_on="gene_site", how="inner")
        for (program_id, label, tcga_cancer), grp in conc.groupby(["program_id", "program_label", "tcga_cancer"]):
            rows.append(
                {
                    "program_id": program_id,
                    "program_label": label,
                    "tcga_cancer": tcga_cancer,
                    "cptac_measure": "predicted_vs_measured_site_concordance",
                    "n_sites": int(grp["target"].nunique()),
                    "median_spearman": float(np.nanmedian(grp["spearman"])),
                    "fraction_positive_fdr_0_05": float(((grp["spearman"] > 0) & (grp["fdr"] < 0.05)).mean()),
                    "note": "site-level audit; not sample-level module reproduction",
                }
            )
    if survival_path.exists():
        surv = pd.read_csv(survival_path, sep="\t")
        surv = surv.loc[surv["cancer_label"].isin(CPTAC_TO_TCGA.keys())].copy()
        surv["tcga_cancer"] = surv["cancer_label"].map(CPTAC_TO_TCGA)
        surv = surv.merge(site_program, left_on="target", right_on="gene_site", how="inner")
        for (program_id, label, tcga_cancer), grp in surv.groupby(["program_id", "program_label", "tcga_cancer"]):
            rows.append(
                {
                    "program_id": program_id,
                    "program_label": label,
                    "tcga_cancer": tcga_cancer,
                    "cptac_measure": "measured_site_os_direction_screen",
                    "n_sites": int(grp["target"].nunique()),
                    "median_spearman": np.nan,
                    "fraction_positive_fdr_0_05": float(
                        ((grp["measured_high_vs_low_logrank_z"] * grp["pred_high_vs_low_logrank_z"] > 0)
                         & (grp["measured_logrank_p"] < 0.05)).mean()
                    ),
                    "note": "site-level OS direction audit; not sample-level module Cox",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    effect = pd.read_csv(FIG5_TABLES / "panel_i_phospho_specific_module_effect_matrix.tsv", sep="\t")
    rows = pd.read_csv(FIG5_TABLES / "panel_i_phospho_specific_module_rows.tsv", sep="\t")
    sites = pd.read_csv(FIG5_TABLES / "panel_i_phospho_specific_module_sites.tsv", sep="\t")
    cox = pd.read_csv(FIG5_TABLES / "panel_i_phospho_specific_module_sample_cox.tsv", sep="\t")
    cox["module_id"] = cox["row_label"].str.extract(r"^(M\\d+)")

    orient, functional_signs = orientation_audit(rows, sites)
    module_map = orient.copy()
    corr = correlation_audit(effect, module_map)
    program = program_cox(cox, module_map)
    program = add_specificity(program)
    row_meta = program_rows(program, module_map)
    cptac = cptac_audit(module_map, sites)

    cancers = [c for c in effect.columns if c != "row_label"]
    mat = program.pivot(index="program_id", columns="cancer", values="program_z").reindex(PROGRAM_ORDER)
    mat = mat.dropna(how="all")
    mat = mat.reindex(columns=cancers)
    module_signs = module_map.set_index("row_label")["orientation_sign"].to_dict()
    oriented_effect = effect.copy()
    for idx, row in oriented_effect.iterrows():
        sign = module_signs.get(row["row_label"], 1)
        for cancer in cancers:
            oriented_effect.loc[idx, cancer] = pd.to_numeric(oriented_effect.loc[idx, cancer], errors="coerce") * float(sign)

    call_mat = program.pivot(index="program_id", columns="cancer", values="program_call").reindex(mat.index).reindex(columns=cancers)
    spec_mat = program.pivot(index="program_id", columns="cancer", values="specificity_q").reindex(mat.index).reindex(columns=cancers)
    row_meta = row_meta.set_index("program_id").reindex(mat.index).reset_index()

    mat.reset_index(names="program_id").to_csv(FIG5_TABLES / "panel_i_program_effect_matrix.tsv", sep="\t", index=False)
    oriented_effect.to_csv(FIG5_TABLES / "panel_i_oriented_module_effect_matrix.tsv", sep="\t", index=False)
    call_mat.reset_index(names="program_id").to_csv(FIG5_TABLES / "panel_i_program_call_matrix.tsv", sep="\t", index=False)
    spec_mat.reset_index(names="program_id").to_csv(FIG5_TABLES / "panel_i_program_specificity_q_matrix.tsv", sep="\t", index=False)
    row_meta.to_csv(FIG5_TABLES / "panel_i_program_rows.tsv", sep="\t", index=False)
    program.to_csv(FIG5_TABLES / "panel_i_program_specificity.tsv", sep="\t", index=False)
    module_map.to_csv(FIG5_TABLES / "panel_i_module_orientation_audit.tsv", sep="\t", index=False)
    corr.to_csv(FIG5_TABLES / "panel_i_module_correlation_audit.tsv", sep="\t", index=False)
    functional_signs.to_csv(FIG5_TABLES / "panel_i_functional_direction_site_signs.tsv", sep="\t", index=False)
    cptac.to_csv(FIG5_TABLES / "panel_i_cptac_module_validation_audit.tsv", sep="\t", index=False)
    summary = {
        "programs": int(mat.shape[0]),
        "cancers": int(mat.shape[1]),
        "cancer_restricted_cells": int(program["program_call"].astype(str).str.startswith("cancer-restricted").sum()),
        "shared_cells": int(program["program_call"].astype(str).str.startswith("shared").sum()),
        "high_orientation_modules": int((module_map["orientation_confidence"] == "high").sum()),
        "moderate_orientation_modules": int((module_map["orientation_confidence"] == "moderate").sum()),
        "weak_orientation_modules": int((module_map["orientation_confidence"] == "weak").sum()),
        "cptac_audit_rows": int(cptac.shape[0]),
        "note": "Panel i is program-level; naming uses unsigned membership enrichment and direction uses PhosphoSIGNOR functional phosphosite effects.",
    }
    (FIG5_TABLES / "panel_i_program_specificity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
