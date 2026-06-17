from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:/data/gongke/TCGA-TCPA")
SCP = ROOT / "SCP682-22/frozen_release/SCP682_22_paper_package_20260520"
OUT = ROOT / "remote_results/20260521_scp682_gnn_prior_audit_local"
OUT.mkdir(parents=True, exist_ok=True)


def read_zip_tsv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            return pd.read_csv(fh, sep="\t")


manifest = pd.read_csv(SCP / "training_set/phosphosite_target_manifest.tsv", sep="\t")
scp_sites = set(manifest["scp682_site_id"].astype(str).str.upper())

map_path = ROOT / "01_data/pathway_prior/processed/copheemap_v1/copheemap_site_id_to_model_gene_site.tsv"
if not map_path.exists():
    map_path = ROOT / "remote_results/20260521_tmp_missing.tsv"
site_map = pd.read_csv(map_path, sep="\t")
id_to_gene_site = dict(zip(site_map["cophee_site_id"].astype(str), site_map["gene_site_id"].astype(str).str.upper()))

raw_dir = ROOT / "01_data/pathway_prior/raw/copheemap_20260519_files"
s2 = read_zip_tsv(raw_dir / "Table_S2_CoPheeMap.tsv.zip")
site_ids = set(s2["site1"].astype(str)) | set(s2["site2"].astype(str))
gene_sites = {id_to_gene_site[x] for x in site_ids if x in id_to_gene_site}

ks = pd.read_csv(raw_dir / "K_S_CoPhee_llr55.csv")
ks_ids = set(ks["sites"].astype(str))
ks_gene_sites = {id_to_gene_site[x] for x in ks_ids if x in id_to_gene_site}

n2v = pd.read_csv(raw_dir / "n2v_networkST.csv")
first_col = n2v.columns[0]
n2v_ids = set(n2v[first_col].astype(str))
n2v_gene_sites = {id_to_gene_site[x] for x in n2v_ids if x in id_to_gene_site}

rows = [
    {
        "source": "CoPheeMap_original_Table_S2_site_site_edges",
        "n_rows": int(s2.shape[0]),
        "n_raw_cophee_site_ids": len(site_ids),
        "n_mapped_gene_site_ids": len(gene_sites),
        "overlap_scp682_sites": len(gene_sites & scp_sites),
        "coverage_fraction": len(gene_sites & scp_sites) / len(scp_sites),
    },
    {
        "source": "CoPheeKSA_original_K_S_CoPhee_llr55",
        "n_rows": int(ks.shape[0]),
        "n_raw_cophee_site_ids": len(ks_ids),
        "n_mapped_gene_site_ids": len(ks_gene_sites),
        "overlap_scp682_sites": len(ks_gene_sites & scp_sites),
        "coverage_fraction": len(ks_gene_sites & scp_sites) / len(scp_sites),
        "n_kinases": int(ks["kinase"].nunique()),
    },
    {
        "source": "CoPheeMap_original_n2v_networkST",
        "n_rows": int(n2v.shape[0]),
        "n_raw_cophee_site_ids": len(n2v_ids),
        "n_mapped_gene_site_ids": len(n2v_gene_sites),
        "overlap_scp682_sites": len(n2v_gene_sites & scp_sites),
        "coverage_fraction": len(n2v_gene_sites & scp_sites) / len(scp_sites),
    },
]
pd.DataFrame(rows).to_csv(OUT / "original_copheemap_local_audit.tsv", sep="\t", index=False)
(OUT / "original_copheemap_local_examples.json").write_text(json.dumps({
    "Table_S2_columns": list(s2.columns),
    "K_S_columns": list(ks.columns),
    "n2v_columns_first10": list(n2v.columns[:10]),
}, ensure_ascii=False, indent=2))
print(pd.DataFrame(rows).to_string(index=False))
