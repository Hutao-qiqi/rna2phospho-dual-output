import argparse
import json
import re
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
DECRYPTM = ROOT / r"01_data\single_cell\intermediate\phospho_perturb\decryptm_comparison_delta_v8"
COPHEE = ROOT / r"01_data\pathway_prior\intermediate\copheemap_20260519_scp682_ppko_v6"
KSTAR = ROOT / r"01_data\pathway_prior\intermediate\kstar_20260519"
RAW = ROOT / r"01_data\pathway_prior\raw\signed_regulatory_20260519"
OUT = ROOT / r"01_data\pathway_prior\intermediate\signed_phospho_regulatory_prior_v9"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def norm_gene(value):
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    text = text.split(";")[0].split(",")[0].strip()
    return re.sub(r"[^A-Za-z0-9_.-]", "", text).upper()


def pick_col(df, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"missing columns {candidates}; available={list(df.columns)}")
    return None


def download(url, path):
    ensure_dir(path.parent)
    if path.exists() and path.stat().st_size > 1024:
        return
    context = ssl._create_unverified_context() if url.startswith("https://") else None
    with urllib.request.urlopen(url, timeout=180, context=context) as r:
        path.write_bytes(r.read())


def read_tsv(path):
    return pd.read_csv(path, sep="\t", low_memory=False)


def load_sites():
    sites = read_tsv(DECRYPTM / "tables" / "site_table.tsv")
    if "target_index" not in sites.columns:
        sites = sites.copy()
        sites["target_index"] = np.arange(len(sites))
    gene_col = pick_col(sites, ["molecule", "gene", "gene_symbol", "protein", "Genes"], required=False)
    site_col = pick_col(sites, ["site", "position", "phosphosite"], required=False)
    tid_col = pick_col(sites, ["target_id", "id", "raw_name"], required=False)
    sites["model_gene"] = sites[gene_col].map(norm_gene) if gene_col else ""
    sites["model_site"] = sites[site_col].astype(str) if site_col else ""
    sites["model_target_id"] = sites[tid_col].astype(str) if tid_col else sites["target_index"].astype(str)
    return sites


def add_edge(rows, source, target_index, target_gene, target_site, sign, edge_type, source_db, score, evidence=""):
    source = norm_gene(source)
    target_gene = norm_gene(target_gene)
    if not source or target_index is None:
        return
    rows.append(
        {
            "source_regulator": source,
            "target_index": int(target_index),
            "target_gene": target_gene,
            "target_site": str(target_site or ""),
            "sign": int(sign),
            "edge_type": edge_type,
            "source_db": source_db,
            "evidence_score": float(score),
            "evidence": str(evidence or ""),
        }
    )


def kstar_edges(sites):
    path = KSTAR / "scp682_ppko_v5_kstar_kinase_site_edges.tsv"
    rows = []
    if not path.exists():
        return rows
    df = read_tsv(path)
    kinase_col = pick_col(df, ["kinase", "source", "source_regulator"])
    idx_col = pick_col(df, ["target_index"])
    score_col = pick_col(df, ["score", "edge_weight", "weight"], required=False)
    site_map = sites.set_index("target_index")
    for _, r in df.iterrows():
        idx = int(r[idx_col])
        if idx not in site_map.index:
            continue
        s = site_map.loc[idx]
        score = r[score_col] if score_col else 1.0
        add_edge(rows, r[kinase_col], idx, s["model_gene"], s["model_site"], 1, "kinase_to_phosphosite", "KSTAR", score)
    return rows


def cophee_edges(sites, n_sites):
    path = COPHEE / "tables" / "copheeksa_model_kinase_site_edges.tsv"
    rows = []
    if not path.exists():
        return rows
    df = read_tsv(path)
    kinase_col = pick_col(df, ["kinase", "source"])
    idx_col = pick_col(df, ["target_index"])
    score_col = pick_col(df, ["score", "cophee_score", "match_score"], required=False)
    site_map = sites.set_index("target_index")
    for _, r in df.iterrows():
        idx = int(r[idx_col])
        if idx >= n_sites or idx not in site_map.index:
            continue
        s = site_map.loc[idx]
        score = r[score_col] if score_col else 1.0
        add_edge(rows, r[kinase_col], idx, s["model_gene"], s["model_site"], 1, "kinase_to_phosphosite", "CoPheeKSA", score)
    return rows


def download_omnipath():
    interactions = RAW / "omnipath_interactions.tsv"
    enzsub = RAW / "omnipath_enzsub.tsv"
    download(
        "https://omnipathdb.org/interactions?genesymbols=1&fields=sources,references,curation_effort,n_references,n_resources&format=tsv",
        interactions,
    )
    download(
        "https://omnipathdb.org/enz_sub?genesymbols=1&fields=sources,references,isoforms&format=tsv",
        enzsub,
    )
    return interactions, enzsub


def download_depod():
    depod = RAW / "DEPOD_PPase_protSubtrates_201903.xls"
    download("https://depod.zellbiologie.uni-bonn.de/download/PPase_protSubtrates_201903.xls", depod)
    return depod


def omnipath_regulator_edges(sites, interactions_path):
    df = read_tsv(interactions_path)
    source_col = pick_col(df, ["source_genesymbol", "source"])
    target_col = pick_col(df, ["target_genesymbol", "target"])
    stim_col = pick_col(df, ["consensus_stimulation", "is_stimulation"], required=False)
    inhib_col = pick_col(df, ["consensus_inhibition", "is_inhibition"], required=False)
    directed_col = pick_col(df, ["consensus_direction", "is_directed"], required=False)
    res_col = pick_col(df, ["sources"], required=False)
    nres_col = pick_col(df, ["n_resources"], required=False)
    nref_col = pick_col(df, ["n_references"], required=False)
    rows = []
    for _, r in df.iterrows():
        if directed_col and int(float(r.get(directed_col, 0) or 0)) != 1:
            continue
        src = norm_gene(r[source_col])
        tgt = norm_gene(r[target_col])
        if not src or not tgt:
            continue
        stim = int(float(r.get(stim_col, 0) or 0)) if stim_col else 0
        inhib = int(float(r.get(inhib_col, 0) or 0)) if inhib_col else 0
        if stim == inhib:
            continue
        sign = 1 if stim > inhib else -1
        nres = float(r.get(nres_col, 1) or 1) if nres_col else 1.0
        nref = float(r.get(nref_col, 1) or 1) if nref_col else 1.0
        rows.append(
            {
                "source_regulator": src,
                "target_regulator": tgt,
                "sign": sign,
                "edge_type": "protein_to_protein",
                "source_db": "OmniPath",
                "evidence_score": float(np.log1p(nres) + 0.25 * np.log1p(nref)),
                "evidence": str(r.get(res_col, "")) if res_col else "",
            }
        )
    reg = pd.DataFrame(rows).drop_duplicates()

    # Project signed protein-protein effects to all measured sites on the target protein.
    site_rows = []
    sites_by_gene = {g: sub for g, sub in sites.groupby("model_gene") if g}
    for _, r in reg.iterrows():
        sub = sites_by_gene.get(r["target_regulator"])
        if sub is None:
            continue
        for _, s in sub.iterrows():
            add_edge(
                site_rows,
                r["source_regulator"],
                s["target_index"],
                s["model_gene"],
                s["model_site"],
                r["sign"],
                "protein_to_measured_phosphoprotein_site",
                "OmniPath",
                r["evidence_score"],
                r["evidence"],
            )
    return reg, site_rows


def omnipath_enzsub_edges(sites, enzsub_path):
    df = read_tsv(enzsub_path)
    enzyme_col = pick_col(df, ["enzyme_genesymbol", "enzyme"])
    substrate_col = pick_col(df, ["substrate_genesymbol", "substrate"])
    mod_col = pick_col(df, ["modification", "modification_type"], required=False)
    res_col = pick_col(df, ["residue_type", "residue"], required=False)
    off_col = pick_col(df, ["residue_offset", "offset", "position"], required=False)
    source_col = pick_col(df, ["sources"], required=False)
    site_lookup = {}
    for _, s in sites.iterrows():
        gene = s["model_gene"]
        site = str(s["model_site"])
        pos = "".join(re.findall(r"\d+", site))
        aa = "".join(re.findall(r"[A-Za-z]+", site)).upper()
        keys = [(gene, site.upper()), (gene, pos)]
        if aa and pos:
            keys.append((gene, f"{aa[0]}{pos}"))
        for key in keys:
            site_lookup.setdefault(key, []).append(s)
    rows = []
    for _, r in df.iterrows():
        mod = str(r.get(mod_col, "")).lower() if mod_col else ""
        if mod and "phosph" not in mod:
            continue
        enzyme = norm_gene(r[enzyme_col])
        substrate = norm_gene(r[substrate_col])
        residue = str(r.get(res_col, "") or "").upper()
        offset = str(r.get(off_col, "") or "")
        offset = "".join(re.findall(r"\d+", offset))
        candidates = []
        if residue and offset:
            candidates += site_lookup.get((substrate, f"{residue[0]}{offset}"), [])
        if offset:
            candidates += site_lookup.get((substrate, offset), [])
        for s in candidates:
            add_edge(
                rows,
                enzyme,
                s["target_index"],
                s["model_gene"],
                s["model_site"],
                1,
                "enzyme_phosphorylates_site",
                "OmniPath_enzsub",
                1.0,
                str(r.get(source_col, "")) if source_col else "",
            )
    return rows


def depod_phosphatase_edges(sites, depod_path):
    if not depod_path.exists():
        return []
    try:
        df = pd.read_excel(depod_path)
    except Exception:
        return []
    cols = {c.lower(): c for c in df.columns}
    phosphatase_col = None
    substrate_col = None
    site_col = None
    window_col = None
    score_col = None
    pubmed_col = None
    for c in df.columns:
        lc = c.lower()
        if phosphatase_col is None and ("phosphatase" in lc or "ppase" in lc):
            phosphatase_col = c
        if substrate_col is None and "substrate" in lc and ("entry" in lc or "gene" in lc or "name" in lc):
            substrate_col = c
        if site_col is None and ("dephosphorylation" in lc and "site" in lc):
            site_col = c
        if window_col is None and "window" in lc:
            window_col = c
        if score_col is None and "reliability" in lc:
            score_col = c
        if pubmed_col is None and "pubmed" in lc:
            pubmed_col = c
    if phosphatase_col is None or substrate_col is None:
        return []

    # DEPOD mostly uses UniProt entry names. Match by gene when the entry name starts with the gene;
    # otherwise fall back to substrate-level projection across measured sites.
    sites_by_gene = {g: sub for g, sub in sites.groupby("model_gene") if g}
    rows = []
    for _, r in df.iterrows():
        phosphatase = norm_gene(str(r.get(phosphatase_col, "")).split("_")[0])
        substrate_raw = str(r.get(substrate_col, ""))
        substrate = norm_gene(substrate_raw.split("_")[0])
        if not phosphatase or not substrate:
            continue
        site_text = str(r.get(site_col, "") or "")
        site_tokens = re.findall(r"[STY][0-9]+", site_text.upper())
        score = r.get(score_col, 1.0) if score_col else 1.0
        try:
            score = float(score)
        except Exception:
            score = 1.0
        evidence = f"DEPOD:{r.get(pubmed_col, '')}" if pubmed_col else "DEPOD"
        sub = sites_by_gene.get(substrate)
        if sub is None:
            continue
        matched = False
        for _, s in sub.iterrows():
            model_site = str(s["model_site"]).upper()
            if site_tokens and not any(tok in model_site for tok in site_tokens):
                continue
            add_edge(
                rows,
                phosphatase,
                s["target_index"],
                s["model_gene"],
                s["model_site"],
                -1,
                "phosphatase_dephosphorylates_site",
                "DEPOD",
                score,
                evidence,
            )
            matched = True
        if not matched and not site_tokens:
            for _, s in sub.iterrows():
                add_edge(
                    rows,
                    phosphatase,
                    s["target_index"],
                    s["model_gene"],
                    s["model_site"],
                    -1,
                    "phosphatase_to_measured_phosphoprotein_site",
                    "DEPOD",
                    score,
                    evidence,
                )
    return rows


def collapse_edges(edges):
    df = pd.DataFrame(edges)
    if df.empty:
        return df
    keys = ["source_regulator", "target_index", "target_gene", "target_site", "sign", "edge_type"]
    out = (
        df.groupby(keys, dropna=False)
        .agg(
            source_db=("source_db", lambda x: ";".join(sorted(set(map(str, x))))),
            evidence_score=("evidence_score", "max"),
            evidence=("evidence", lambda x: ";".join(sorted(set(map(str, x)))[:5])),
        )
        .reset_index()
    )
    return out.sort_values(["source_regulator", "target_index", "sign", "edge_type"])


def build_matrices(site_edges, sites):
    regs = sorted(set(site_edges["source_regulator"]))
    reg_to_i = {g: i for i, g in enumerate(regs)}
    n_sites = len(sites)
    mat = np.zeros((len(regs), n_sites), dtype=np.float32)
    for _, r in site_edges.iterrows():
        i = reg_to_i[r["source_regulator"]]
        j = int(r["target_index"])
        if 0 <= j < n_sites:
            mat[i, j] += float(r["sign"]) * float(r["evidence_score"])
    row_norm = np.maximum(1.0, np.abs(mat).sum(axis=1, keepdims=True))
    mat = mat / row_norm
    return regs, mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    ensure_dir(OUT / "tables")
    ensure_dir(OUT / "arrays")
    ensure_dir(OUT / "reports")
    ensure_dir(RAW)

    sites = load_sites()
    n_sites = len(sites)

    if args.skip_download:
        interactions_path = RAW / "omnipath_interactions.tsv"
        enzsub_path = RAW / "omnipath_enzsub.tsv"
        depod_path = RAW / "DEPOD_PPase_protSubtrates_201903.xls"
    else:
        interactions_path, enzsub_path = download_omnipath()
        depod_path = download_depod()

    site_edges = []
    site_edges += kstar_edges(sites)
    site_edges += cophee_edges(sites, n_sites)
    reg_edges, projected_rows = omnipath_regulator_edges(sites, interactions_path)
    site_edges += projected_rows
    site_edges += omnipath_enzsub_edges(sites, enzsub_path)
    site_edges += depod_phosphatase_edges(sites, depod_path)

    site_edges = collapse_edges(site_edges)
    regs, mat = build_matrices(site_edges, sites)
    reg_table = pd.DataFrame({"regulator_index": np.arange(len(regs)), "regulator": regs})

    site_edges.to_csv(OUT / "tables" / "signed_regulator_site_edges.tsv", sep="\t", index=False)
    reg_edges.to_csv(OUT / "tables" / "signed_regulator_regulator_edges.tsv", sep="\t", index=False)
    reg_table.to_csv(OUT / "tables" / "signed_regulator_table.tsv", sep="\t", index=False)
    sites[["target_index", "model_target_id", "model_gene", "model_site"]].to_csv(
        OUT / "tables" / "signed_prior_site_table.tsv", sep="\t", index=False
    )
    np.save(OUT / "arrays" / "signed_regulator_site_matrix.npy", mat)

    summary = {
        "n_sites": int(n_sites),
        "n_regulators": int(len(regs)),
        "n_signed_regulator_site_edges": int(len(site_edges)),
        "n_signed_regulator_regulator_edges": int(len(reg_edges)),
        "site_edge_source_counts": site_edges["source_db"].value_counts().head(30).to_dict(),
        "site_edge_type_counts": site_edges["edge_type"].value_counts().to_dict(),
        "sign_counts": site_edges["sign"].value_counts().to_dict(),
        "raw_files": {
            "omnipath_interactions": str(interactions_path),
            "omnipath_enzsub": str(enzsub_path),
            "depod_phosphatase_substrate": str(depod_path),
        },
    }
    (OUT / "reports" / "signed_phospho_regulatory_prior_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
