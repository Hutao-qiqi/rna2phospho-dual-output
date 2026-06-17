import json
import re
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\data\gongke\TCGA-TCPA")
COPHEE = ROOT / r"01_data\pathway_prior\raw\copheemap_20260519_files"
TARGET = ROOT / r"remote_scripts\phospho_target_table.tsv"
OUT = ROOT / r"02_results\single_cell\20260519_scp682_ppko_dataset_feature_audit"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_sequence(value):
    return re.sub(r"\[[^\]]*\]|\([^)]*\)|[^A-Za-z]", "", str(value)).upper()


def parse_cophee_site(site):
    parts = str(site).split("|")
    return {
        "site_id": site,
        "ensg": parts[0] if len(parts) > 0 else "",
        "ensp": parts[1] if len(parts) > 1 else "",
        "site": parts[2] if len(parts) > 2 else "",
        "window15": clean_sequence(parts[3]) if len(parts) > 3 else "",
    }


def build_model_peptides(targets):
    rows = []
    for _, row in targets.iterrows():
        peptide = clean_sequence(row.get("modified_peptide", ""))
        if len(peptide) >= 7:
            rows.append(
                {
                    "target_index": int(row["target_index"]),
                    "target_id": row.get("target_id", ""),
                    "molecule": row.get("molecule", ""),
                    "site": row.get("site", ""),
                    "peptide": peptide,
                }
            )
    return rows


def map_by_sequence(model_rows, cophee_sites):
    site_rows = [parse_cophee_site(s) for s in cophee_sites]
    mapped_model = set()
    mapped_cophee = set()
    pairs = []
    for m in model_rows:
        pep = m["peptide"]
        if len(pep) < 7:
            continue
        for s in site_rows:
            w = s["window15"]
            if not w:
                continue
            if pep in w or w in pep:
                mapped_model.add(m["target_index"])
                mapped_cophee.add(s["site_id"])
                if len(pairs) < 200000:
                    pairs.append(
                        {
                            "target_index": m["target_index"],
                            "target_id": m["target_id"],
                            "molecule": m["molecule"],
                            "model_site": m["site"],
                            "model_peptide": pep,
                            "cophee_site_id": s["site_id"],
                            "cophee_site": s["site"],
                            "cophee_window15": w,
                        }
                    )
    return mapped_model, mapped_cophee, pd.DataFrame(pairs)


def main():
    for sub in ("tables", "reports"):
        ensure_dir(OUT / sub)
    targets = pd.read_csv(TARGET, sep="\t")
    with zipfile.ZipFile(COPHEE / "Table_S2_CoPheeMap.tsv.zip") as zf:
        with zf.open("Table_S2_CoPheeMap.tsv") as handle:
            edge = pd.read_csv(handle, sep="\t")
    ksa = pd.read_csv(COPHEE / "K_S_CoPhee_llr55.csv")
    n2v = pd.read_csv(COPHEE / "n2v_networkST.csv", usecols=[0])

    cophee_sites = set(edge["site1"].astype(str)) | set(edge["site2"].astype(str))
    ksa_sites = set(ksa["sites"].astype(str))
    n2v_sites = set(n2v.iloc[:, 0].astype(str))
    model_rows = build_model_peptides(targets)
    mapped_model, mapped_cophee, pair_df = map_by_sequence(model_rows, cophee_sites | ksa_sites | n2v_sites)

    summary = {
        "copheemap_edges": int(len(edge)),
        "copheemap_unique_sites": int(len(cophee_sites)),
        "copheeksa_edges": int(len(ksa)),
        "copheeksa_unique_sites": int(len(ksa_sites)),
        "copheeksa_kinases": int(ksa["kinase"].nunique()),
        "n2v_sites": int(len(n2v_sites)),
        "model_sites": int(len(targets)),
        "model_sites_with_sequence": int(len(model_rows)),
        "model_sites_sequence_overlap_with_cophee": int(len(mapped_model)),
        "cophee_sites_sequence_overlap_with_model": int(len(mapped_cophee)),
    }
    pair_df.to_csv(OUT / "tables" / "copheemap_model_sequence_overlap_pairs.tsv", sep="\t", index=False)
    pd.DataFrame([summary]).to_csv(OUT / "tables" / "copheemap_overlap_summary.tsv", sep="\t", index=False)
    with (OUT / "reports" / "copheemap_overlap_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
