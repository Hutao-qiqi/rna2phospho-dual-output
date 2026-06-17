#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/data/lsy/Infinite_Stream")
RAW = ROOT / "01_data/pathway_prior/raw/copheemap"
REPO = RAW / "CoPheeMap"
OUT = ROOT / "01_data/pathway_prior/processed/copheemap_v1"
TARGETS = ROOT / "01_data/multi_omics/processed/pancancer_multi_task_locked_v1/phosphosite_gene_site_study_zscore_min20pct_targets.parquet"


def parse_site(raw: str, ensg_to_symbol: dict[str, str]) -> dict[str, str] | None:
    parts = str(raw).split("|")
    if len(parts) < 3:
        return None
    ensg = parts[0].split(".", 1)[0]
    site = parts[2]
    symbol = ensg_to_symbol.get(ensg)
    if not symbol:
        return None
    return {
        "cophee_site_id": raw,
        "ensembl_gene_id": ensg,
        "gene_symbol": symbol,
        "site": site,
        "gene_site_id": f"{symbol}|{site}",
        "peptide_15mer": parts[3] if len(parts) > 3 else "",
    }


def load_hgnc_mapping() -> dict[str, str]:
    hgnc = pd.read_csv(RAW / "hgnc_complete_set.txt", sep="\t", dtype=str)
    hgnc = hgnc.loc[hgnc["ensembl_gene_id"].notna() & hgnc["symbol"].notna(), ["ensembl_gene_id", "symbol"]]
    hgnc["ensembl_gene_id"] = hgnc["ensembl_gene_id"].str.split(".", n=1).str[0]
    return dict(zip(hgnc["ensembl_gene_id"], hgnc["symbol"]))


def build_site_map(target_set: set[str], ensg_to_symbol: dict[str, str]) -> pd.DataFrame:
    site_ids: set[str] = set()
    zip_path = REPO / "Supplementary_table/Table_S2_CoPheeMap.tsv.zip"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("Table_S2_CoPheeMap.tsv") as fh:
            for chunk in pd.read_csv(fh, sep="\t", chunksize=500_000):
                site_ids.update(chunk["site1"].astype(str))
                site_ids.update(chunk["site2"].astype(str))
    for path in [
        REPO / "CoPheeMap/data_construction/n2v_networkST.csv",
        REPO / "CoPheeKSA/results/K_S_CoPhee_llr55.csv",
        REPO / "CoPheeKSA/results/K_S_CoPhee_llr55_w_features.csv",
    ]:
        if path.exists():
            col = "sites" if "K_S" in path.name else None
            df = pd.read_csv(path, usecols=[0] if col is None else [col])
            if col is None:
                site_ids.update(df.iloc[:, 0].astype(str))
            else:
                site_ids.update(df[col].astype(str))

    rows = []
    for sid in sorted(site_ids):
        parsed = parse_site(sid, ensg_to_symbol)
        if parsed is not None:
            parsed["in_model_targets"] = parsed["gene_site_id"] in target_set
            rows.append(parsed)
    return pd.DataFrame(rows)


def build_edges(site_map: pd.DataFrame, target_set: set[str]) -> pd.DataFrame:
    id_to_target = dict(zip(site_map["cophee_site_id"], site_map["gene_site_id"]))
    rows = []
    zip_path = REPO / "Supplementary_table/Table_S2_CoPheeMap.tsv.zip"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("Table_S2_CoPheeMap.tsv") as fh:
            for chunk in pd.read_csv(fh, sep="\t", chunksize=500_000):
                a = chunk["site1"].map(id_to_target)
                b = chunk["site2"].map(id_to_target)
                ok = a.notna() & b.notna() & (a != b) & a.isin(target_set) & b.isin(target_set)
                if ok.any():
                    tmp = pd.DataFrame({"site_a": a[ok].astype(str), "site_b": b[ok].astype(str)})
                    rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["site_a", "site_b", "edge_weight"])
    edges = pd.concat(rows, ignore_index=True)
    ordered = np.sort(edges[["site_a", "site_b"]].to_numpy(dtype=str), axis=1)
    edges = pd.DataFrame({"site_a": ordered[:, 0], "site_b": ordered[:, 1]})
    edges = edges.drop_duplicates()
    edges["edge_weight"] = 1.0
    return edges


def build_embeddings(site_map: pd.DataFrame, target_set: set[str]) -> pd.DataFrame:
    n2v = pd.read_csv(REPO / "CoPheeMap/data_construction/n2v_networkST.csv")
    site_col = n2v.columns[0]
    emb_cols = [c for c in n2v.columns if c != site_col]
    mapped = site_map[["cophee_site_id", "gene_site_id"]].drop_duplicates()
    out = n2v.merge(mapped, left_on=site_col, right_on="cophee_site_id", how="inner")
    out = out.loc[out["gene_site_id"].isin(target_set), ["gene_site_id"] + emb_cols]
    out = out.drop_duplicates("gene_site_id")
    return out


def build_ksa(site_map: pd.DataFrame, target_set: set[str]) -> pd.DataFrame:
    ksa = pd.read_csv(REPO / "CoPheeKSA/results/K_S_CoPhee_llr55.csv")
    mapped = site_map[["cophee_site_id", "gene_site_id"]].drop_duplicates()
    out = ksa.merge(mapped, left_on="sites", right_on="cophee_site_id", how="inner")
    out = out.loc[out["gene_site_id"].isin(target_set), ["gene_site_id", "kinase", "scores", "cophee_site_id"]]
    out = out.rename(columns={"scores": "copheeksa_score"})
    out = out.sort_values(["gene_site_id", "copheeksa_score"], ascending=[True, False])
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    target_names = pd.read_parquet(TARGETS).columns.astype(str).tolist()
    target_set = set(target_names)
    ensg_to_symbol = load_hgnc_mapping()
    site_map = build_site_map(target_set, ensg_to_symbol)
    edges = build_edges(site_map, target_set)
    emb = build_embeddings(site_map, target_set)
    ksa = build_ksa(site_map, target_set)

    site_map.to_csv(OUT / "copheemap_site_id_to_model_gene_site.tsv", sep="\t", index=False)
    edges.to_csv(OUT / "copheemap_model_phosphosite_edges.tsv", sep="\t", index=False)
    emb.to_csv(OUT / "copheemap_model_phosphosite_node2vec.tsv", sep="\t", index=False)
    ksa.to_csv(OUT / "copheeksa_model_phosphosite_kinase_predictions.tsv", sep="\t", index=False)

    target_idx = {t: i for i, t in enumerate(target_names)}
    edge_idx = edges.assign(site_a_idx=edges["site_a"].map(target_idx), site_b_idx=edges["site_b"].map(target_idx))
    edge_idx = edge_idx.dropna(subset=["site_a_idx", "site_b_idx"])
    edge_idx[["site_a_idx", "site_b_idx", "edge_weight"]].astype({"site_a_idx": int, "site_b_idx": int}).to_csv(
        OUT / "copheemap_model_phosphosite_edges_indexed.tsv", sep="\t", index=False
    )

    lamb_neighbors = pd.DataFrame()
    if not edges.empty:
        lamb = "LAMB1|S1666"
        lamb_neighbors = edges.loc[(edges["site_a"] == lamb) | (edges["site_b"] == lamb)].copy()
        if not lamb_neighbors.empty:
            lamb_neighbors["neighbor"] = np.where(lamb_neighbors["site_a"] == lamb, lamb_neighbors["site_b"], lamb_neighbors["site_a"])
            lamb_neighbors.to_csv(OUT / "lamb1_s1666_copheemap_neighbors.tsv", sep="\t", index=False)
    lamb_ksa = ksa.loc[ksa["gene_site_id"].eq("LAMB1|S1666")].copy()
    lamb_ksa.to_csv(OUT / "lamb1_s1666_copheeksa_kinases.tsv", sep="\t", index=False)

    summary = {
        "n_model_phosphosite_targets": len(target_names),
        "n_copheemap_sites_mapped_to_symbols": int(site_map.shape[0]),
        "n_copheemap_sites_in_model_targets": int(site_map["in_model_targets"].sum()) if not site_map.empty else 0,
        "n_model_edges": int(edges.shape[0]),
        "n_model_nodes_with_edges": int(pd.unique(pd.concat([edges["site_a"], edges["site_b"]], ignore_index=True)).shape[0]) if not edges.empty else 0,
        "n_model_node2vec_embeddings": int(emb.shape[0]),
        "n_model_ksa_predictions": int(ksa.shape[0]),
        "n_model_sites_with_ksa": int(ksa["gene_site_id"].nunique()) if not ksa.empty else 0,
        "lamb1_s1666_edge_degree": int(lamb_neighbors.shape[0]) if not lamb_neighbors.empty else 0,
        "lamb1_s1666_ksa_count": int(lamb_ksa.shape[0]),
        "output_dir": str(OUT),
    }
    (OUT / "COPHEEMAP_PRIOR_DATA_CARD.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not lamb_ksa.empty:
        print(lamb_ksa.head(20).to_string(index=False))
    if not lamb_neighbors.empty:
        print(lamb_neighbors.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
