import argparse
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib


DEFAULT_ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
DEFAULT_RAW = DEFAULT_ROOT / r"01_data\single_cell\raw\decryptm_pxd037285_v1"
DEFAULT_OUT = DEFAULT_ROOT / r"01_data\single_cell\intermediate\phospho_perturb\decryptm_pxd037285_curve_v1"


PATHWAY_GENES = {
    "EGFR_ERBB_axis": {"EGFR", "ERBB2", "ERBB3", "ERBB4", "GRB2", "SHC1", "GAB1", "CBL"},
    "RAS_MAPK_axis": {"KRAS", "NRAS", "HRAS", "BRAF", "RAF1", "ARAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "DUSP6", "ELK1", "FOS", "JUN"},
    "stress_MAPK_axis": {"MAPK8", "MAPK9", "MAPK14", "MAPKAPK2", "HSPB1", "ATF2", "JUN", "JUND"},
    "PI3K_AKT_mTOR_axis": {"PIK3CA", "PIK3CB", "PIK3CD", "PIK3R1", "PDPK1", "AKT1", "AKT2", "AKT3", "MTOR", "RPTOR", "RICTOR", "RPS6", "RPS6KB1", "EIF4EBP1", "GSK3A", "GSK3B", "FOXO1", "FOXO3"},
    "SHP2_RTK_axis": {"PTPN11", "GRB2", "SOS1", "SOS2", "GAB1", "GAB2", "EGFR", "ERBB2", "FGFR1", "FGFR2", "FGFR3"},
    "FGFR_axis": {"FGFR1", "FGFR2", "FGFR3", "FGFR4", "FRS2", "PLCg1", "PLCG1", "GRB2"},
    "BCR_BTK_axis": {"MS4A1", "CD79A", "CD79B", "SYK", "BTK", "LYN", "LCK", "BLK", "PLCG2", "CARD11", "NFKB1", "REL"},
    "ABL_SRC_axis": {"ABL1", "ABL2", "SRC", "YES1", "FYN", "LCK", "LYN", "HCK", "KIT", "PDGFRA", "PDGFRB"},
    "PLCG_PKC_axis": {"PLCG1", "PLCG2", "PRKCA", "PRKCB", "PRKCD", "PRKCE", "MARCKS", "PAK1", "PAK2"},
    "cell_cycle_DNA_damage": {"CDK1", "CDK2", "CDK4", "CDK6", "RB1", "TP53", "CHEK1", "CHEK2", "ATM", "ATR", "BRCA1", "MDC1", "H2AFX"},
    "proteasome_stress_axis": {"PSMB5", "PSMB1", "PSMB2", "PSMB7", "PSMA1", "PSMA2", "NFE2L2", "HSPA1A", "DDIT3"},
    "chromatin_acetylation_axis": {"EP300", "CREBBP", "BRD4", "HDAC1", "HDAC2", "HDAC3", "HDAC6", "KAT2A", "KAT5"},
    "cytoskeleton_adhesion": {"ABI1", "AHNAK", "LIMA1", "PLEC", "PLEC1", "MAP4", "KIF4A", "PXN", "VCL", "ZYX"},
}


DRUG_TARGET_GENES = {
    "dasatinib": {"ABL1", "SRC", "YES1", "LCK", "LYN", "HCK", "KIT", "PDGFRB"},
    "imatinib": {"ABL1", "KIT", "PDGFRA", "PDGFRB"},
    "nilotinib": {"ABL1", "KIT", "PDGFRA", "PDGFRB"},
    "bosutinib": {"ABL1", "SRC", "LYN", "HCK"},
    "ponatinib": {"ABL1", "FGFR1", "FGFR2", "FGFR3", "KIT", "PDGFRA"},
    "gefitinib": {"EGFR"},
    "erlotinib": {"EGFR"},
    "afatinib": {"EGFR", "ERBB2", "ERBB4"},
    "osimertinib": {"EGFR"},
    "lapatinib": {"EGFR", "ERBB2"},
    "trastuzumab": {"ERBB2"},
    "pertuzumab": {"ERBB2"},
    "rituximab": {"MS4A1"},
    "mk2206": {"AKT1", "AKT2", "AKT3"},
    "selumetinib": {"MAP2K1", "MAP2K2"},
    "pd325901": {"MAP2K1", "MAP2K2"},
    "trametinib": {"MAP2K1", "MAP2K2"},
    "shp099": {"PTPN11"},
    "azd4547": {"FGFR1", "FGFR2", "FGFR3"},
    "bortezomib": {"PSMB5"},
    "carfilzomib": {"PSMB5"},
    "dactolisib": {"PIK3CA", "PIK3CB", "PIK3CD", "MTOR"},
    "pictilisib": {"PIK3CA", "PIK3CB", "PIK3CD"},
    "azd-8055": {"MTOR"},
    "a485": {"EP300", "CREBBP"},
    "a486": {"EP300", "CREBBP"},
    "vorinostat": {"HDAC1", "HDAC2", "HDAC3", "HDAC6"},
    "panobinostat": {"HDAC1", "HDAC2", "HDAC3", "HDAC6"},
    "entinostat": {"HDAC1", "HDAC2", "HDAC3"},
    "curcumin": {"EP300", "CREBBP"},
    "methotrexat": {"DHFR", "TYMS"},
    "methotrexate": {"DHFR", "TYMS"},
    "cytarabine": {"DNA"},
}


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_token(value):
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.:+-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "NA"


def split_genes(value):
    genes = []
    for part in re.split(r"[;,/| ]+", str(value)):
        token = clean_token(part).upper()
        if token and token not in {"NA", "NAN", "CON__", "REV__"}:
            genes.append(token)
    return list(dict.fromkeys(genes))


def parse_float(value):
    if value is None:
        return math.nan
    text = str(value).strip()
    if text in {"", "-", "NA", "NaN", "nan", "None"}:
        return math.nan
    text = text.replace(",", ".")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return math.nan
    try:
        return float(match.group(0))
    except ValueError:
        return math.nan


def parse_time_hours(value):
    if value is None:
        return math.nan
    text = str(value).strip().lower().replace(" ", "")
    if text in {"", "-", "na", "nan", "none"}:
        return math.nan
    number = parse_float(text)
    if not np.isfinite(number):
        return math.nan
    if "min" in text or text.endswith("m"):
        return number / 60.0
    if "sec" in text or text.endswith("s"):
        return number / 3600.0
    if "day" in text or text.endswith("d"):
        return number * 24.0
    return number


def parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def modified_sequence_key(value):
    text = str(value)
    text = text.replace("(ph)", "p").replace("(ac)", "ac").replace("(gl)", "gl")
    return clean_token(text).upper()


def make_site_id(row):
    genes = split_genes(row.get("Gene names", "NA"))
    gene = genes[0] if genes else clean_token(row.get("Leading proteins", "NA")).upper()
    modseq = modified_sequence_key(row.get("Modified sequence", row.get("Sequence", "NA")))
    return f"{gene}__{modseq}"


def pathway_membership(genes):
    gene_set = {g.upper() for g in genes}
    hits = []
    for pathway, members in PATHWAY_GENES.items():
        if gene_set & {m.upper() for m in members}:
            hits.append(pathway)
    hits.append("global_phospho")
    return list(dict.fromkeys(hits))


def target_genes_for_drug(drug):
    genes = set()
    text = str(drug).lower()
    for part in re.split(r"[,;/+]| and ", text):
        key = re.sub(r"[^a-z0-9-]+", "", part.strip())
        if key in DRUG_TARGET_GENES:
            genes.update(DRUG_TARGET_GENES[key])
    for key, vals in DRUG_TARGET_GENES.items():
        if key in text:
            genes.update(vals)
    return sorted(genes)


def load_experiment_summary(raw_dir):
    raw_dir = Path(raw_dir)
    xlsx = raw_dir / "Experiment_summary.xlsx"
    if not xlsx.exists():
        zpath = raw_dir / "Experiment_summary.zip"
        if zpath.exists():
            with zipfile.ZipFile(zpath) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
                if names:
                    xlsx = zf.extract(names[0], raw_dir)
                    xlsx = Path(xlsx)
    if not Path(xlsx).exists():
        raise FileNotFoundError(f"missing Experiment_summary.xlsx in {raw_dir}")
    ptm = pd.read_excel(xlsx, sheet_name="PTMs")
    ptm = ptm.rename(columns={c: str(c).strip() for c in ptm.columns})
    ptm = ptm[~ptm["Experimental name"].astype(str).str.startswith("[", na=False)].copy()
    ptm = ptm[ptm["Experimental name"].notna()].copy()
    by_exp = {}
    for exp, sub in ptm.groupby(ptm["Experimental name"].astype(str).str.strip(), sort=False):
        row = sub.iloc[0].to_dict()
        row["n_summary_rows"] = int(len(sub))
        by_exp[str(exp).strip()] = row
    return by_exp


def read_pipeline_toml(zf, prefix):
    candidates = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith("pipeline.toml")]
    if not candidates:
        return {}
    raw = zf.read(candidates[0]).decode("utf-8", errors="replace")
    try:
        return tomllib.loads(raw)
    except Exception:
        return {}


def find_phospho_curve_entries(zf):
    entries = []
    for name in zf.namelist():
        lower = name.lower()
        if lower.endswith("curves.txt") and "phosphoproteome" in lower:
            prefix = name.rsplit("/", 1)[0] + "/"
            entries.append((name, prefix))
    return entries


def ratio_columns(columns):
    pairs = []
    for col in columns:
        match = re.match(r"TMT Channel Ratio (\d+)$", str(col))
        if match:
            pairs.append((int(match.group(1)), col))
    return [col for _, col in sorted(pairs)]


def channel_from_ratio_col(col):
    return int(re.search(r"(\d+)$", str(col)).group(1))


def dose_to_molar(value, scale):
    number = parse_float(value)
    if not np.isfinite(number):
        return math.nan
    scale_value = parse_float(scale)
    if np.isfinite(scale_value):
        return number * scale_value
    return math.nan


def channel_meta(exp_name, channel_number, n_channels, summary, pipeline):
    tmt = pipeline.get("TMT", {}) if isinstance(pipeline, dict) else {}
    processing = pipeline.get("Processing", {}) if isinstance(pipeline, dict) else {}
    idx0 = channel_number - 1
    doses = tmt.get("doses") or []
    times = tmt.get("time_points") or []
    dose = summary.get(f"Conc{idx0}", math.nan) if summary else math.nan
    time_value = summary.get(f"Time{idx0}", math.nan) if summary else math.nan
    if not np.isfinite(parse_float(dose)) and idx0 < len(doses):
        dose = doses[idx0]
    if not np.isfinite(parse_float(time_value)) and idx0 < len(times):
        time_value = times[idx0]
    exp_type = str(summary.get("Experiment-type", tmt.get("experiment_type", ""))).strip() if summary else ""
    fixed_time = summary.get("Treatment time", "") if summary else ""
    if str(exp_type).lower() == "dd" and not np.isfinite(parse_time_hours(time_value)):
        time_value = fixed_time
    control_channel = int(processing.get("control_channel") or 0)
    if not control_channel:
        control_channel = n_channels
    dose_scale = str(tmt.get("dose_scale", ""))
    dose_label = str(tmt.get("dose_label", ""))
    dose_molar = dose_to_molar(dose, dose_scale)
    return {
        "channel": int(channel_number),
        "dose": str(dose),
        "dose_unit": dose_label or "raw",
        "dose_molar": dose_molar,
        "time": str(time_value),
        "time_hours": parse_time_hours(time_value),
        "control_channel": int(control_channel),
    }


def condition_labels(exp_name, channel_number, n_channels, summary, pipeline):
    meta = channel_meta(exp_name, channel_number, n_channels, summary, pipeline)
    cell = clean_token(summary.get("Cell line", "NA") if summary else "NA")
    drug = clean_token(summary.get("Drug", "NA") if summary else "NA")
    exp_type = clean_token(summary.get("Experiment-type", "NA") if summary else "NA")
    system = clean_token(summary.get("System", "NA") if summary else "NA")
    if channel_number == meta["control_channel"]:
        condition = clean_token(f"decryptm__{cell}__{drug}__{exp_type}__control")
        control = condition
        perturbation_type = "control"
    else:
        dose_part = clean_token(f"{meta['dose']}{meta['dose_unit']}")
        time_part = clean_token(f"{meta['time']}h" if np.isfinite(meta["time_hours"]) else str(meta["time"]))
        condition = clean_token(f"decryptm__{cell}__{drug}__{exp_type}__{dose_part}__{time_part}")
        control = clean_token(f"decryptm__{cell}__{drug}__{exp_type}__control")
        perturbation_type = exp_type
    return condition, control, perturbation_type, system, meta


def iter_curve_tables(raw_dir):
    raw_dir = Path(raw_dir)
    zips = sorted(raw_dir.glob("*_Curves.zip"))
    for zpath in zips:
        with zipfile.ZipFile(zpath) as zf:
            for entry, prefix in find_phospho_curve_entries(zf):
                pipeline = read_pipeline_toml(zf, prefix)
                yield zpath, zf, entry, pipeline


def update_site_stats(stats, chunk, ratio_cols):
    if "Phosphoproteome" in chunk:
        chunk = chunk[chunk["Phosphoproteome"].map(parse_bool)].copy()
    if chunk.empty:
        return
    ratios = chunk[ratio_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    ratios[ratios <= 0] = np.nan
    log_ratio = np.log2(ratios)
    finite = np.isfinite(log_ratio)
    finite_n = finite.sum(axis=1)
    max_abs = np.where(finite_n > 0, np.max(np.where(finite, np.abs(log_ratio), 0.0), axis=1), np.nan)
    effect = pd.to_numeric(chunk.get("Curve effect size", pd.Series(np.nan, index=chunk.index)), errors="coerce").fillna(0.0).abs().to_numpy()
    regulation = chunk.get("Regulation", pd.Series("", index=chunk.index)).astype(str).str.strip()
    regulated = (~regulation.isin(["", "nan", "NaN", "None"])).to_numpy(dtype=np.int32)
    for i, (_, row) in enumerate(chunk.iterrows()):
        if finite_n[i] == 0:
            continue
        site_id = make_site_id(row)
        rec = stats.get(site_id)
        genes = split_genes(row.get("Gene names", "NA"))
        if rec is None:
            rec = {
                "site_id": site_id,
                "genes": genes,
                "primary_gene": genes[0] if genes else "NA",
                "modified_sequence": str(row.get("Modified sequence", "")),
                "sequence": str(row.get("Sequence", "")),
                "proteins": str(row.get("Proteins", "")),
                "leading_proteins": str(row.get("Leading proteins", "")),
                "protein_names": str(row.get("Protein names", "")),
                "phospho_sty": parse_float(row.get("Phospho (STY)", math.nan)),
                "finite_count": 0,
                "max_abs_log2_ratio": 0.0,
                "max_curve_effect_size": 0.0,
                "regulated_count": 0,
            }
            stats[site_id] = rec
        rec["finite_count"] += int(finite_n[i])
        if np.isfinite(max_abs[i]):
            rec["max_abs_log2_ratio"] = max(rec["max_abs_log2_ratio"], float(max_abs[i]))
        rec["max_curve_effect_size"] = max(rec["max_curve_effect_size"], float(effect[i]))
        rec["regulated_count"] += int(regulated[i])


def collect_samples(sample_rows, sample_key_to_idx, chunk, ratio_cols, summary_by_exp, pipeline, source_zip):
    n_channels = len(ratio_cols)
    for exp_name in chunk["Experiment"].dropna().astype(str).unique():
        summary = summary_by_exp.get(str(exp_name).strip(), {})
        for col in ratio_cols:
            channel = channel_from_ratio_col(col)
            key = (str(exp_name).strip(), int(channel))
            if key in sample_key_to_idx:
                continue
            condition, control, perturbation_type, system, meta = condition_labels(exp_name, channel, n_channels, summary, pipeline)
            cell = str(summary.get("Cell line", "NA"))
            drug = str(summary.get("Drug", "NA"))
            genes = target_genes_for_drug(drug)
            sample_key_to_idx[key] = len(sample_rows)
            sample_rows.append(
                {
                    "cell_id": clean_token(f"decryptm:{exp_name}:channel{channel}"),
                    "dataset_id": "decryptm_pxd037285_curve",
                    "source_zip": source_zip.name,
                    "experiment": str(exp_name).strip(),
                    "condition": condition,
                    "condition_label": condition,
                    "perturbation": drug if channel != meta["control_channel"] else "DMSO",
                    "perturbation_id": clean_token(drug if channel != meta["control_channel"] else "DMSO"),
                    "perturbation_type": perturbation_type,
                    "control_condition": control,
                    "time": meta["time"],
                    "time_unit": "h",
                    "time_hours": meta["time_hours"],
                    "dose": meta["dose"],
                    "dose_unit": meta["dose_unit"],
                    "dose_molar": meta["dose_molar"],
                    "cell_type": cell,
                    "system": system,
                    "target_genes": ";".join(genes),
                    "channel": int(channel),
                    "control_channel": int(meta["control_channel"]),
                    "is_simulated": False,
                }
            )


def scan_curves(raw_dir, summary_by_exp, chunksize):
    stats = {}
    sample_rows = []
    sample_key_to_idx = {}
    for zpath, zf, entry, pipeline in iter_curve_tables(raw_dir):
        print(f"scan {zpath.name}:{entry}", flush=True)
        with zf.open(entry) as handle:
            reader = pd.read_csv(handle, sep="\t", chunksize=chunksize, low_memory=False)
            for chunk in reader:
                rcols = ratio_columns(chunk.columns)
                if not rcols or "Experiment" not in chunk:
                    continue
                collect_samples(sample_rows, sample_key_to_idx, chunk, rcols, summary_by_exp, pipeline, zpath)
                update_site_stats(stats, chunk, rcols)
    return stats, sample_rows, sample_key_to_idx


def select_sites(stats, max_sites, min_finite):
    rows = []
    for rec in stats.values():
        if rec["finite_count"] < min_finite:
            continue
        score = rec["max_abs_log2_ratio"] + 0.25 * rec["max_curve_effect_size"] + 0.05 * math.log1p(rec["finite_count"]) + 0.50 * (rec["regulated_count"] > 0)
        row = dict(rec)
        row["selection_score"] = float(score)
        rows.append(row)
    rows = sorted(rows, key=lambda r: (r["selection_score"], r["finite_count"]), reverse=True)
    if max_sites > 0:
        rows = rows[:max_sites]
    rows = sorted(rows, key=lambda r: r["site_id"])
    for i, row in enumerate(rows):
        row["target_index"] = i
    return rows


def fill_matrix(raw_dir, summary_by_exp, selected_sites, sample_key_to_idx, sample_rows, chunksize, value_clip):
    site_to_idx = {row["site_id"]: i for i, row in enumerate(selected_sites)}
    sums = np.zeros((len(sample_rows), len(selected_sites)), dtype=np.float64)
    counts = np.zeros((len(sample_rows), len(selected_sites)), dtype=np.int16)
    for zpath, zf, entry, pipeline in iter_curve_tables(raw_dir):
        print(f"fill {zpath.name}:{entry}", flush=True)
        with zf.open(entry) as handle:
            reader = pd.read_csv(handle, sep="\t", chunksize=chunksize, low_memory=False)
            for chunk in reader:
                rcols = ratio_columns(chunk.columns)
                if not rcols or "Experiment" not in chunk:
                    continue
                if "Phosphoproteome" in chunk:
                    chunk = chunk[chunk["Phosphoproteome"].map(parse_bool)].copy()
                if chunk.empty:
                    continue
                site_ids = chunk.apply(make_site_id, axis=1)
                keep = site_ids.isin(site_to_idx)
                if not keep.any():
                    continue
                sub = chunk.loc[keep].copy()
                sub_site = site_ids.loc[keep].map(site_to_idx).to_numpy(dtype=np.int64)
                exp_names = sub["Experiment"].astype(str).str.strip().to_numpy()
                ratios = sub[rcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
                ratios[ratios <= 0] = np.nan
                log_ratio = np.log2(ratios)
                if value_clip > 0:
                    log_ratio = np.clip(log_ratio, -value_clip, value_clip)
                for j, col in enumerate(rcols):
                    channel = channel_from_ratio_col(col)
                    finite = np.isfinite(log_ratio[:, j])
                    if not finite.any():
                        continue
                    sample_idx = np.array([sample_key_to_idx.get((exp, channel), -1) for exp in exp_names], dtype=np.int64)
                    ok = finite & (sample_idx >= 0)
                    if not ok.any():
                        continue
                    np.add.at(sums, (sample_idx[ok], sub_site[ok]), log_ratio[ok, j].astype(np.float64))
                    np.add.at(counts, (sample_idx[ok], sub_site[ok]), 1)
    mask = counts > 0
    values = np.divide(sums, np.maximum(counts, 1), where=np.maximum(counts, 1) > 0).astype(np.float32)
    values[~mask] = 0.0
    return values, mask


def build_condition_table(meta):
    rows = []
    for condition, sub in meta.groupby("condition", sort=False):
        first = sub.iloc[0]
        rows.append(
            {
                "condition": condition,
                "condition_label": first.get("condition_label", condition),
                "dataset_id": first.get("dataset_id", "decryptm_pxd037285_curve"),
                "perturbation": first.get("perturbation", "NA"),
                "perturbation_id": first.get("perturbation_id", "NA"),
                "perturbation_type": first.get("perturbation_type", "NA"),
                "control_condition": first.get("control_condition", condition),
                "time": first.get("time", "NA"),
                "time_unit": first.get("time_unit", "h"),
                "time_hours": first.get("time_hours", math.nan),
                "dose": first.get("dose", "NA"),
                "dose_unit": first.get("dose_unit", "raw"),
                "dose_molar": first.get("dose_molar", math.nan),
                "cell_type": first.get("cell_type", "NA"),
                "system": first.get("system", "NA"),
                "target_genes": first.get("target_genes", ""),
                "n_cells": int(len(sub)),
                "is_simulated": False,
            }
        )
    return pd.DataFrame(rows)


def build_condition_prior(condition_table):
    rows = []
    for _, row in condition_table.iterrows():
        genes = split_genes(row.get("target_genes", ""))
        gene_set = set(genes)
        for pathway, members in PATHWAY_GENES.items():
            prior = 1.0 if gene_set & {m.upper() for m in members} else 0.0
            if prior:
                rows.append({"condition": row["condition"], "pathway": pathway, "prior": prior, "target_genes": ";".join(genes)})
    return pd.DataFrame(rows)


def write_outputs(out_dir, values, mask, sample_rows, selected_sites):
    out_dir = ensure_dir(out_dir)
    meta = pd.DataFrame(sample_rows)
    condition_table = build_condition_table(meta)
    target_rows = []
    pathway_rows = []
    for row in selected_sites:
        genes = row["genes"]
        target_rows.append(
            {
                "target_index": int(row["target_index"]),
                "raw_name": row["modified_sequence"],
                "target_id": row["site_id"],
                "molecule": "/".join(genes) if genes else row["primary_gene"],
                "site": row["modified_sequence"],
                "uniprot_id": row["leading_proteins"],
                "modified_peptide": row["modified_sequence"],
                "sequence": row["sequence"],
                "protein_names": row["protein_names"],
                "primary_pathway": pathway_membership(genes)[0],
                "finite_count": int(row["finite_count"]),
                "max_abs_log2_ratio": float(row["max_abs_log2_ratio"]),
                "max_curve_effect_size": float(row["max_curve_effect_size"]),
                "regulated_count": int(row["regulated_count"]),
                "selection_score": float(row["selection_score"]),
            }
        )
        for pathway in pathway_membership(genes):
            pathway_rows.append({"pathway": pathway, "raw_name": row["modified_sequence"], "target_index": int(row["target_index"]), "target_id": row["site_id"]})

    np.save(out_dir / "phospho_values.npy", values.astype(np.float32))
    np.save(out_dir / "phospho_raw.npy", values.astype(np.float32))
    np.save(out_dir / "target_mask.npy", mask.astype(bool))
    meta.to_csv(out_dir / "cell_metadata.tsv", sep="\t", index=False)
    condition_table.to_csv(out_dir / "condition_table.tsv", sep="\t", index=False)
    pd.DataFrame(target_rows).to_csv(out_dir / "phospho_target_table.tsv", sep="\t", index=False)
    pd.DataFrame(pathway_rows).to_csv(out_dir / "pathway_target_manifest.tsv", sep="\t", index=False)
    prior = build_condition_prior(condition_table)
    prior.to_csv(out_dir / "condition_pathway_prior.tsv", sep="\t", index=False)
    with (out_dir / "transform_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": "decryptM PXD037285 curve files",
                "transform": "log2 TMT Channel Ratio values from phosphoproteome curves; missing values retained in target_mask",
                "n_samples": int(values.shape[0]),
                "n_targets": int(values.shape[1]),
                "n_conditions": int(condition_table.shape[0]),
                "finite_fraction": float(mask.mean()),
                "n_condition_pathway_priors": int(prior.shape[0]),
            },
            handle,
            indent=2,
        )
    print(f"wrote {out_dir} samples={values.shape[0]} targets={values.shape[1]} conditions={condition_table.shape[0]} finite={mask.mean():.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--max-sites", type=int, default=8192)
    parser.add_argument("--min-finite", type=int, default=12)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--value-clip", type=float, default=6.0)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = ensure_dir(args.output_dir)
    summary_by_exp = load_experiment_summary(raw_dir)
    stats, sample_rows, sample_key_to_idx = scan_curves(raw_dir, summary_by_exp, args.chunksize)
    selected_sites = select_sites(stats, args.max_sites, args.min_finite)
    if not selected_sites:
        raise RuntimeError("no selected phospho sites")
    pd.DataFrame(selected_sites).drop(columns=["genes"], errors="ignore").to_csv(out_dir / "selected_site_stats.tsv", sep="\t", index=False)
    values, mask = fill_matrix(raw_dir, summary_by_exp, selected_sites, sample_key_to_idx, sample_rows, args.chunksize, args.value_clip)
    write_outputs(out_dir, values, mask, sample_rows, selected_sites)


if __name__ == "__main__":
    main()
