from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path("/data/lsy/Infinite_Stream")
SCP = ROOT / "SCP682-22/frozen_release/SCP682_22_paper_package_20260520"
OUT = ROOT / "02_results/model_validation/20260521_scp682_gnn_prior_audit/tables"
OUT.mkdir(parents=True, exist_ok=True)


def read_zip_tsv(zip_path: Path, nrows: int | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            return pd.read_csv(fh, sep="\t", nrows=nrows)


def norm_gene_site(gene: pd.Series, site: pd.Series) -> pd.Series:
    return gene.fillna("").astype(str).str.upper() + "|" + site.fillna("").astype(str).str.upper()


manifest = pd.read_csv(SCP / "training_set/phosphosite_target_manifest.tsv", sep="\t")
scp_sites = set(manifest["scp682_site_id"].astype(str).str.upper())
scp_parents = set(manifest["parent_gene"].astype(str).str.upper())

rows = []
examples = {}

cophee_s2 = ROOT / "01_data/pathway_prior/raw/copheemap/CoPheeMap/Supplementary_table/Table_S2_CoPheeMap.tsv.zip"
s2_head = read_zip_tsv(cophee_s2, nrows=5)
examples["Table_S2_CoPheeMap_head"] = s2_head.to_dict(orient="records")
examples["Table_S2_CoPheeMap_columns"] = list(s2_head.columns)
s2 = read_zip_tsv(cophee_s2)
site_cols = [c for c in s2.columns if "site" in c.lower() or "phosph" in c.lower() or "gene" in c.lower()]
edge_cols = list(s2.columns)

site_candidates = set()
for gene_col in [c for c in s2.columns if c.lower() in {"gene", "gene_symbol", "genesymbol", "substrate_gene"} or "gene" in c.lower()]:
    for site_col in [c for c in s2.columns if c.lower() in {"site", "position", "phosphosite"} or "site" in c.lower()]:
        cand = set(norm_gene_site(s2[gene_col], s2[site_col]))
        overlap = len(cand & scp_sites)
        if overlap > 0:
            site_candidates |= cand
if not site_candidates:
    for c in s2.columns:
        ser = s2[c].astype(str).str.upper()
        if ser.str.contains(r"^[A-Z0-9-]+\|[STY][0-9]", regex=True).any():
            site_candidates |= set(ser)

rows.append(
    {
        "source": "CoPheeMap_original_Table_S2",
        "path": str(cophee_s2),
        "n_rows": int(s2.shape[0]),
        "n_columns": int(s2.shape[1]),
        "candidate_site_ids": len(site_candidates),
        "overlap_scp682_sites": len(site_candidates & scp_sites),
        "coverage_fraction": len(site_candidates & scp_sites) / len(scp_sites),
        "site_related_columns": ";".join(site_cols[:30]),
        "all_columns_preview": ";".join(edge_cols[:30]),
    }
)

pos_ksa = ROOT / "01_data/pathway_prior/raw/copheemap/CoPheeMap/CoPheeKSA/positive_KSA.csv"
if pos_ksa.exists():
    pos = pd.read_csv(pos_ksa)
    examples["positive_KSA_columns"] = list(pos.columns)
    ksa_sites = set()
    for gene_col in [c for c in pos.columns if "gene" in c.lower() or "substrate" in c.lower()]:
        for site_col in [c for c in pos.columns if "site" in c.lower() or "position" in c.lower()]:
            ksa_sites |= set(norm_gene_site(pos[gene_col], pos[site_col]))
    rows.append(
        {
            "source": "CoPheeKSA_original_positive_KSA",
            "path": str(pos_ksa),
            "n_rows": int(pos.shape[0]),
            "n_columns": int(pos.shape[1]),
            "candidate_site_ids": len(ksa_sites),
            "overlap_scp682_sites": len(ksa_sites & scp_sites),
            "coverage_fraction": len(ksa_sites & scp_sites) / len(scp_sites),
            "site_related_columns": ";".join([c for c in pos.columns if "site" in c.lower() or "gene" in c.lower() or "kinase" in c.lower()]),
            "all_columns_preview": ";".join(list(pos.columns)[:30]),
        }
    )

kstar_hpp = ROOT / "01_data/pathway_prior/raw/kstar_20260516/extracted/RESOURCE_FILES/HumanPhosphoProteome.csv"
hpp = pd.read_csv(kstar_hpp)
examples["KSTAR_HumanPhosphoProteome_columns"] = list(hpp.columns)
kstar_hpp_sites = set()
for gene_col in [c for c in hpp.columns if "gene" in c.lower() or "symbol" in c.lower()]:
    for site_col in [c for c in hpp.columns if c.lower() in {"site", "mod_site"} or "site" in c.lower()]:
        kstar_hpp_sites |= set(norm_gene_site(hpp[gene_col], hpp[site_col]))
rows.append(
    {
        "source": "KSTAR_original_HumanPhosphoProteome",
        "path": str(kstar_hpp),
        "n_rows": int(hpp.shape[0]),
        "n_columns": int(hpp.shape[1]),
        "candidate_site_ids": len(kstar_hpp_sites),
        "overlap_scp682_sites": len(kstar_hpp_sites & scp_sites),
        "coverage_fraction": len(kstar_hpp_sites & scp_sites) / len(scp_sites),
        "site_related_columns": ";".join([c for c in hpp.columns if "site" in c.lower() or "gene" in c.lower() or "kinase" in c.lower() or "symbol" in c.lower()]),
        "all_columns_preview": ";".join(list(hpp.columns)[:30]),
    }
)

kstar_edges = ROOT / "01_data/pathway_prior/intermediate/kstar_20260516/kstar_default_network_edges_long.tsv"
edges = pd.read_csv(kstar_edges, sep="\t")
edge_gene_sites = set(norm_gene_site(edges["substrate_gene"], edges["site"]))
rows.append(
    {
        "source": "KSTAR_original_default_network_edges_long",
        "path": str(kstar_edges),
        "n_rows": int(edges.shape[0]),
        "n_columns": int(edges.shape[1]),
        "candidate_site_ids": len(edge_gene_sites),
        "overlap_scp682_sites": len(edge_gene_sites & scp_sites),
        "coverage_fraction": len(edge_gene_sites & scp_sites) / len(scp_sites),
        "n_kinases": int(edges["kinase"].nunique()),
        "site_related_columns": ";".join([c for c in edges.columns if "site" in c.lower() or "gene" in c.lower() or "kinase" in c.lower()]),
        "all_columns_preview": ";".join(list(edges.columns)[:30]),
    }
)

pd.DataFrame(rows).to_csv(OUT / "original_gnn_table_audit.tsv", sep="\t", index=False)
(OUT / "original_gnn_table_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2))
print(pd.DataFrame(rows).to_string(index=False))
