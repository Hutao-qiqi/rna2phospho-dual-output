import argparse
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]", re.IGNORECASE)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_sequence(value):
    text = str(value)
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", text)
    return AA_RE.sub("", text).upper()


def parse_cophee_site(value):
    parts = str(value).split("|")
    return {
        "cophee_site_id": str(value),
        "ensg": parts[0] if len(parts) > 0 else "",
        "ensp": parts[1] if len(parts) > 1 else "",
        "site": parts[2] if len(parts) > 2 else "",
        "window": clean_sequence(parts[3]) if len(parts) > 3 else "",
    }


def longest_common_substring_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, start=1):
            v = prev[j - 1] + 1 if ca == cb else 0
            cur.append(v)
            if v > best:
                best = v
        prev = cur
    return best


def sequence_match_score(peptide, window):
    if not peptide or not window:
        return 0.0
    if peptide in window or window in peptide:
        return 1.0
    lcs = longest_common_substring_len(peptide, window)
    return float(lcs / max(1, min(len(peptide), len(window))))


def build_kmer_index(cophee_sites, k):
    site_info = {}
    kmer_to_sites = defaultdict(set)
    for site_id in cophee_sites:
        info = parse_cophee_site(site_id)
        window = info["window"]
        if len(window) < k:
            continue
        site_info[site_id] = info
        for i in range(0, len(window) - k + 1):
            kmer_to_sites[window[i : i + k]].add(site_id)
    return site_info, kmer_to_sites


def model_peptide_rows(targets):
    rows = []
    for _, row in targets.iterrows():
        peptide = clean_sequence(row.get("modified_peptide", ""))
        if len(peptide) < 6:
            peptide = clean_sequence(row.get("raw_name", ""))
        if len(peptide) < 6:
            continue
        rows.append(
            {
                "target_index": int(row["target_index"]),
                "target_id": row.get("target_id", ""),
                "molecule": row.get("molecule", ""),
                "site": row.get("site", ""),
                "model_peptide": peptide,
            }
        )
    return rows


def map_model_to_cophee(targets, cophee_sites, args):
    site_info, kmer_to_sites = build_kmer_index(cophee_sites, args.kmer_size)
    model_to_cophee = defaultdict(list)
    cophee_to_model = defaultdict(list)
    rows = []
    for m in model_peptide_rows(targets):
        peptide = m["model_peptide"]
        candidates = set()
        if len(peptide) >= args.kmer_size:
            for i in range(0, len(peptide) - args.kmer_size + 1):
                candidates.update(kmer_to_sites.get(peptide[i : i + args.kmer_size], set()))
        scored = []
        for site_id in candidates:
            score = sequence_match_score(peptide, site_info[site_id]["window"])
            if score >= args.min_match_fraction:
                scored.append((score, site_id))
        scored.sort(reverse=True)
        for score, site_id in scored[: args.max_cophee_sites_per_model]:
            info = site_info[site_id]
            item = {
                **m,
                "cophee_site_id": site_id,
                "cophee_site": info["site"],
                "cophee_window": info["window"],
                "match_score": float(score),
            }
            rows.append(item)
            model_to_cophee[m["target_index"]].append((site_id, float(score)))
            cophee_to_model[site_id].append((m["target_index"], float(score)))
    return model_to_cophee, cophee_to_model, pd.DataFrame(rows)


def read_copheemap_edges(cophee_dir):
    path = Path(cophee_dir) / "Table_S2_CoPheeMap.tsv.zip"
    with zipfile.ZipFile(path) as zf:
        with zf.open("Table_S2_CoPheeMap.tsv") as handle:
            return pd.read_csv(handle, sep="\t")


def build_site_site_edges(edge_df, cophee_to_model, targets, args):
    pair = {}
    for row in edge_df.itertuples(index=False):
        s1 = str(getattr(row, "site1"))
        s2 = str(getattr(row, "site2"))
        m1 = cophee_to_model.get(s1)
        m2 = cophee_to_model.get(s2)
        if not m1 or not m2:
            continue
        for i, wi in m1:
            for j, wj in m2:
                if i == j:
                    continue
                for a, b in ((int(i), int(j)), (int(j), int(i))):
                    key = (a, b)
                    val = pair.get(key)
                    w = float(wi * wj)
                    if val is None:
                        pair[key] = [1, w]
                    else:
                        val[0] += 1
                        val[1] = max(val[1], w)
    rows = []
    for (i, j), (count, max_weight) in pair.items():
        score = math.log1p(count) * max_weight
        rows.append(
            {
                "target_index_1": int(i),
                "target_index_2": int(j),
                "edge_count": int(count),
                "edge_weight": float(score),
                "target_id_1": targets.iloc[int(i)].get("target_id", ""),
                "target_id_2": targets.iloc[int(j)].get("target_id", ""),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["target_index_1", "target_index_2", "edge_count", "edge_weight"])
    out = pd.DataFrame(rows)
    out = out.sort_values(["target_index_1", "edge_weight"], ascending=[True, False])
    out = out.groupby("target_index_1", as_index=False, group_keys=False).head(args.max_site_neighbors)
    return out.reset_index(drop=True)


def build_kinase_site_edges(ksa, cophee_to_model, targets, args):
    rows = []
    for row in ksa.itertuples(index=False):
        site_id = str(getattr(row, "sites"))
        kinase = str(getattr(row, "kinase")).upper()
        score = float(getattr(row, "scores"))
        mapped = cophee_to_model.get(site_id)
        if not mapped:
            continue
        for target_index, match_score in mapped:
            rows.append(
                {
                    "kinase": kinase,
                    "target_index": int(target_index),
                    "score": float(score * match_score),
                    "cophee_score": float(score),
                    "match_score": float(match_score),
                    "target_id": targets.iloc[int(target_index)].get("target_id", ""),
                    "molecule": targets.iloc[int(target_index)].get("molecule", ""),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["kinase", "target_index", "score"])
    out = pd.DataFrame(rows)
    out = out.sort_values(["target_index", "score"], ascending=[True, False])
    out = out.groupby(["kinase", "target_index"], as_index=False).agg(
        score=("score", "max"),
        cophee_score=("cophee_score", "max"),
        match_score=("match_score", "max"),
        target_id=("target_id", "first"),
        molecule=("molecule", "first"),
    )
    out = out.sort_values(["target_index", "score"], ascending=[True, False])
    out = out.groupby("target_index", as_index=False, group_keys=False).head(args.max_kinases_per_site)
    return out.reset_index(drop=True)


def build_site_embeddings(n2v, cophee_to_model, n_sites):
    dim_cols = list(n2v.columns[1:])
    accum = np.zeros((n_sites, len(dim_cols)), dtype=np.float64)
    weight = np.zeros(n_sites, dtype=np.float64)
    site_ids = n2v.iloc[:, 0].astype(str).tolist()
    vecs = n2v.iloc[:, 1:].to_numpy(dtype=np.float64)
    for site_id, vec in zip(site_ids, vecs):
        mapped = cophee_to_model.get(site_id)
        if not mapped:
            continue
        for target_index, match_score in mapped:
            accum[int(target_index)] += vec * float(match_score)
            weight[int(target_index)] += float(match_score)
    ok = weight > 0
    accum[ok] = accum[ok] / weight[ok, None]
    rows = []
    for i in range(n_sites):
        rec = {"target_index": int(i), "n_embedding_matches": float(weight[i])}
        for d in range(accum.shape[1]):
            rec[f"emb_{d}"] = float(accum[i, d])
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cophee-dir", required=True)
    parser.add_argument("--target-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kmer-size", type=int, default=7)
    parser.add_argument("--min-match-fraction", type=float, default=0.72)
    parser.add_argument("--max-cophee-sites-per-model", type=int, default=5)
    parser.add_argument("--max-site-neighbors", type=int, default=24)
    parser.add_argument("--max-kinases-per-site", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    for sub in ("tables", "reports"):
        ensure_dir(out_dir / sub)

    targets = pd.read_csv(args.target_table, sep="\t")
    edge_df = read_copheemap_edges(args.cophee_dir)
    ksa = pd.read_csv(Path(args.cophee_dir) / "K_S_CoPhee_llr55.csv")
    n2v = pd.read_csv(Path(args.cophee_dir) / "n2v_networkST.csv")
    cophee_sites = (
        set(edge_df["site1"].astype(str))
        | set(edge_df["site2"].astype(str))
        | set(ksa["sites"].astype(str))
        | set(n2v.iloc[:, 0].astype(str))
    )
    model_to_cophee, cophee_to_model, mapping = map_model_to_cophee(targets, cophee_sites, args)
    site_edges = build_site_site_edges(edge_df, cophee_to_model, targets, args)
    kinase_edges = build_kinase_site_edges(ksa, cophee_to_model, targets, args)
    embeddings = build_site_embeddings(n2v, cophee_to_model, len(targets))

    mapping.to_csv(out_dir / "tables" / "model_site_cophee_map.tsv", sep="\t", index=False)
    site_edges.to_csv(out_dir / "tables" / "copheemap_model_site_site_edges.tsv", sep="\t", index=False)
    kinase_edges.to_csv(out_dir / "tables" / "copheeksa_model_kinase_site_edges.tsv", sep="\t", index=False)
    embeddings.to_csv(out_dir / "tables" / "cophee_model_site_n2v_embeddings.tsv", sep="\t", index=False)

    summary = {
        "n_model_sites": int(len(targets)),
        "n_cophee_sites_total": int(len(cophee_sites)),
        "n_model_sites_mapped": int(mapping["target_index"].nunique()) if len(mapping) else 0,
        "n_cophee_sites_mapped": int(mapping["cophee_site_id"].nunique()) if len(mapping) else 0,
        "n_mapping_pairs": int(len(mapping)),
        "n_site_site_edges": int(len(site_edges)),
        "n_site_site_edge_sources": int(site_edges["target_index_1"].nunique()) if len(site_edges) else 0,
        "n_kinase_site_edges": int(len(kinase_edges)),
        "n_kinases": int(kinase_edges["kinase"].nunique()) if len(kinase_edges) else 0,
        "n_embedding_sites": int((embeddings["n_embedding_matches"] > 0).sum()),
        "kmer_size": int(args.kmer_size),
        "min_match_fraction": float(args.min_match_fraction),
        "max_site_neighbors": int(args.max_site_neighbors),
        "max_kinases_per_site": int(args.max_kinases_per_site),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "tables" / "copheemap_prior_summary.tsv", sep="\t", index=False)
    with (out_dir / "reports" / "copheemap_prior_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
