import argparse
import gzip
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
DECRYPTM_SITE_TABLE = (
    ROOT
    / r"01_data\single_cell\intermediate\phospho_perturb\decryptm_comparison_delta_v8\tables\site_table.tsv"
)
V9_DIR = ROOT / r"01_data\pathway_prior\intermediate\signed_phospho_regulatory_prior_v9"
COPHEE_DIR = ROOT / r"01_data\pathway_prior\intermediate\copheemap_20260519_scp682_ppko_v6"
RAW_STRING_DIR = ROOT / r"01_data\pathway_prior\raw\string_20260519"
OUT_DIR = ROOT / r"01_data\pathway_prior\intermediate\global_phosphoprotein_heterograph_v10"
RAW_UNIPROT_DIR = ROOT / r"01_data\pathway_prior\raw\uniprot_20260519"

STRING_VERSION = "12.0"
STRING_SPECIES = 9606
CALLER_IDENTITY = "SCP682-PPKO-1-global-phosphoprotein-heterograph-v10"

TRAINING_TARGETS = {
    "Gefitinib": ["EGFR"],
    "Afatinib": ["EGFR", "ERBB2", "ERBB4"],
    "Dasatinib": ["ABL1", "BCR", "SRC", "LCK", "YES1", "FYN", "KIT", "PDGFRB"],
    "Rituximab": ["MS4A1"],
}

SIGNED_REGULATOR_SITE_COLUMNS = [
    "source_protein_index",
    "source_regulator",
    "site_index",
    "target_index",
    "sign",
    "edge_type",
    "source_db",
    "edge_weight",
]

SIGNED_PROTEIN_PROTEIN_COLUMNS = [
    "source_protein_index",
    "target_protein_index",
    "source_regulator",
    "target_regulator",
    "sign",
    "edge_type",
    "source_db",
    "edge_weight",
]

UNSIGNED_PROTEIN_PROTEIN_COLUMNS = [
    "source_protein_index",
    "target_protein_index",
    "source_gene",
    "target_gene",
    "score",
    "combined_score",
    "source_db",
    "edge_type",
    "string_id_1",
    "string_id_2",
]

SITE_SITE_COPHEE_COLUMNS = [
    "site_index_1",
    "site_index_2",
    "target_index_1",
    "target_index_2",
    "edge_weight",
    "source_db",
    "edge_type",
]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_tsv(path):
    return pd.read_csv(path, sep="\t", low_memory=False)


def pick_col(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Missing required column. candidates={candidates}, columns={list(df.columns)}")
    return None


def norm_gene(value):
    if pd.isna(value):
        return ""
    gene = str(value).strip()
    if not gene or gene.lower() in {"nan", "none", "null"}:
        return ""
    if ";" in gene:
        gene = gene.split(";")[0]
    if "," in gene:
        gene = gene.split(",")[0]
    return gene.strip().upper()


def split_gene_symbols(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    parts = []
    for token in text.replace(",", ";").replace("/", ";").replace("|", ";").split(";"):
        gene = token.strip().upper()
        if gene and gene.lower() not in {"nan", "none", "null"}:
            parts.append(gene)
    return sorted(set(parts))


def safe_float(value, default=1.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def download_with_cache(url, path, timeout=300, min_bytes=1024):
    ensure_dir(path.parent)
    if path.exists() and path.stat().st_size >= min_bytes:
        return {"path": str(path), "downloaded": False, "ok": True, "error": ""}
    context = ssl._create_unverified_context() if url.startswith("https://") else None
    try:
        with urllib.request.urlopen(url, timeout=timeout, context=context) as response:
            path.write_bytes(response.read())
        return {"path": str(path), "downloaded": True, "ok": True, "error": ""}
    except Exception as exc:
        if path.exists() and path.stat().st_size >= min_bytes:
            return {"path": str(path), "downloaded": False, "ok": True, "error": str(exc)}
        return {"path": str(path), "downloaded": False, "ok": False, "error": str(exc)}


def string_get(url, cache_path, timeout=180, min_bytes=1):
    ensure_dir(cache_path.parent)
    if cache_path.exists() and cache_path.stat().st_size >= min_bytes:
        return cache_path.read_text(encoding="utf-8", errors="replace"), "cache", ""
    context = ssl._create_unverified_context() if url.startswith("https://") else None
    try:
        with urllib.request.urlopen(url, timeout=timeout, context=context) as response:
            text = response.read().decode("utf-8", errors="replace")
        cache_path.write_text(text, encoding="utf-8")
        return text, "download", ""
    except Exception as exc:
        if cache_path.exists() and cache_path.stat().st_size >= min_bytes:
            return cache_path.read_text(encoding="utf-8", errors="replace"), "cache_after_error", str(exc)
        return "", "failed", str(exc)


def build_uniprot_gene_map(accessions, raw_dir, timeout=180, batch_size=400):
    raw_dir = Path(raw_dir)
    ensure_dir(raw_dir)
    accessions = sorted({str(x).strip() for x in accessions if str(x).strip() and str(x).lower() != "nan"})
    mapping = {}
    statuses = []
    for batch_i, start in enumerate(range(0, len(accessions), batch_size)):
        batch = accessions[start : start + batch_size]
        query = " OR ".join([f"accession:{acc}" for acc in batch])
        params = {
            "query": f"({query}) AND organism_id:9606",
            "fields": "accession,gene_primary,gene_names",
            "format": "tsv",
            "size": str(len(batch)),
        }
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(params)
        cache = raw_dir / f"uniprot_accession_gene_batch_{batch_i:04d}.tsv"
        text, status, error = string_get(url, cache, timeout=timeout)
        statuses.append({"batch": batch_i, "n_query": len(batch), "status": status, "error": error})
        if not text:
            continue
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        header = lines[0].split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < len(header):
                continue
            acc = parts[col.get("Entry", col.get("From", 0))]
            primary = parts[col.get("Gene Names (primary)", -1)] if "Gene Names (primary)" in col else ""
            names = parts[col.get("Gene Names", -1)] if "Gene Names" in col else ""
            gene = norm_gene(primary) or (split_gene_symbols(names)[0] if split_gene_symbols(names) else "")
            if acc and gene:
                mapping[acc] = gene
    out = raw_dir / "uniprot_accession_gene_map.tsv"
    pd.DataFrame([{"uniprot_id": k, "gene": v} for k, v in sorted(mapping.items())]).to_csv(out, sep="\t", index=False)
    return mapping, {"n_accessions": len(accessions), "n_mapped": len(mapping), "batches": statuses, "path": str(out)}


def load_sites(site_table_path):
    sites = read_tsv(site_table_path).copy()
    if "target_index" not in sites.columns:
        sites["target_index"] = np.arange(len(sites), dtype=np.int64)
    sites["target_index"] = sites["target_index"].astype(int)

    gene_col = pick_col(sites, ["molecule", "gene", "gene_symbol", "protein", "Genes"], required=False)
    site_col = pick_col(sites, ["site", "position", "phosphosite"], required=False)
    target_id_col = pick_col(sites, ["target_id", "id", "feature_id"], required=False)

    if gene_col is None:
        raise KeyError("decryptM site_table.tsv lacks molecule/gene/protein column")

    sites["molecule"] = sites[gene_col].astype(str)
    sites["site_genes"] = sites[gene_col].map(split_gene_symbols)
    uniprot_col = pick_col(sites, ["uniprot_id", "accession", "UniProt", "protein_id"], required=False)
    unmapped = sites["site_genes"].map(len).eq(0)
    uniprot_status = {"n_accessions": 0, "n_mapped": 0}
    if unmapped.any() and uniprot_col:
        accessions = sites.loc[unmapped, uniprot_col].astype(str).tolist()
        gene_map, uniprot_status = build_uniprot_gene_map(accessions, RAW_UNIPROT_DIR)
        for idx in sites.index[unmapped]:
            acc = str(sites.at[idx, uniprot_col]).strip()
            gene = gene_map.get(acc, "")
            if gene:
                sites.at[idx, "site_genes"] = [gene]
                if not str(sites.at[idx, "molecule"]) or str(sites.at[idx, "molecule"]).lower() == "nan":
                    sites.at[idx, "molecule"] = gene
    unmapped = sites["site_genes"].map(len).eq(0)
    if unmapped.any() and uniprot_col:
        for idx in sites.index[unmapped]:
            acc = norm_gene(sites.at[idx, uniprot_col])
            if acc:
                sites.at[idx, "site_genes"] = [acc]
                sites.at[idx, "molecule"] = acc
    sites["site_gene_list"] = sites["site_genes"].map(lambda genes: ";".join(genes))
    missing_gene = sites["site_genes"].map(len).eq(0)
    if missing_gene.any():
        bad = sites.loc[missing_gene, "target_index"].head(20).tolist()
        raise ValueError(f"Missing phosphoprotein gene for site target_index examples: {bad}")

    sites["site"] = sites[site_col].astype(str) if site_col else ""
    if target_id_col:
        sites["target_id"] = sites[target_id_col].astype(str)
    else:
        sites["target_id"] = sites["molecule"] + "_" + sites["site"].astype(str)
    sites = sites.drop_duplicates("target_index").sort_values("target_index").reset_index(drop=True)
    sites["site_index"] = np.arange(len(sites), dtype=np.int64)
    sites.attrs["uniprot_status"] = uniprot_status
    return sites


def load_v9_regulators(v9_dir):
    table_path = v9_dir / "tables" / "signed_regulator_table.tsv"
    regs = set()
    if table_path.exists():
        df = read_tsv(table_path)
        col = pick_col(df, ["regulator", "gene", "source_regulator"], required=False)
        if col:
            regs.update(g for g in df[col].map(norm_gene) if g)

    site_edges_path = v9_dir / "tables" / "signed_regulator_site_edges.tsv"
    if site_edges_path.exists():
        df = read_tsv(site_edges_path)
        col = pick_col(df, ["source_regulator", "regulator", "source"], required=False)
        if col:
            regs.update(g for g in df[col].map(norm_gene) if g)

    pp_edges_path = v9_dir / "tables" / "signed_regulator_regulator_edges.tsv"
    if pp_edges_path.exists():
        df = read_tsv(pp_edges_path)
        for colname in ("source_regulator", "target_regulator", "source", "target"):
            if colname in df.columns:
                regs.update(g for g in df[colname].map(norm_gene) if g)
    return regs


def build_protein_nodes(sites, v9_regulators):
    sources = defaultdict(set)
    for genes in sites["site_genes"]:
        for gene in genes:
            sources[gene].add("decryptM_phosphoprotein")
    for gene in v9_regulators:
        if gene:
            sources[gene].add("v9_regulator")

    genes = sorted(sources)
    protein_nodes = pd.DataFrame(
        {
            "protein_index": np.arange(len(genes), dtype=np.int64),
            "gene": genes,
            "node_source": [";".join(sorted(sources[g])) for g in genes],
        }
    )
    return protein_nodes, {g: i for i, g in enumerate(genes)}


def build_site_nodes(sites):
    keep = ["site_index", "target_index", "molecule", "site", "target_id"]
    extra = [c for c in sites.columns if c not in keep]
    return sites[keep + extra]


def build_membership_edges(sites, protein_to_i):
    rows = []
    for row in sites.itertuples(index=False):
        genes = list(getattr(row, "site_genes"))
        if not genes:
            continue
        rows.append(
            {
                "protein_index": int(protein_to_i[genes[0]]),
                "gene": genes[0],
                "site_index": int(getattr(row, "site_index")),
                "target_index": int(getattr(row, "target_index")),
                "target_id": str(getattr(row, "target_id")),
                "edge_type": "protein_contains_site",
                "source_db": "decryptM",
                "membership_rank": 0,
                "molecule_raw": str(getattr(row, "molecule")),
            }
        )
        for rank, gene in enumerate(genes[1:], start=1):
            rows.append(
                {
                    "protein_index": int(protein_to_i[gene]),
                    "gene": gene,
                    "site_index": int(getattr(row, "site_index")),
                    "target_index": int(getattr(row, "target_index")),
                    "target_id": str(getattr(row, "target_id")),
                    "edge_type": "protein_contains_site",
                    "source_db": "decryptM",
                    "membership_rank": rank,
                    "molecule_raw": str(getattr(row, "molecule")),
                }
            )
    return pd.DataFrame(rows)


def load_signed_regulator_site_edges(v9_dir, site_index_by_target, protein_to_i):
    path = v9_dir / "tables" / "signed_regulator_site_edges.tsv"
    if not path.exists():
        return pd.DataFrame(columns=SIGNED_REGULATOR_SITE_COLUMNS)
    df = read_tsv(path).copy()
    source_col = pick_col(df, ["source_regulator", "regulator", "source"])
    target_col = pick_col(df, ["target_index"])
    sign_col = pick_col(df, ["sign"])
    score_col = pick_col(df, ["evidence_score", "score", "weight"], required=False)

    df["source_regulator"] = df[source_col].map(norm_gene)
    df["target_index"] = df[target_col].astype(int)
    df["site_index"] = df["target_index"].map(site_index_by_target)
    df["source_protein_index"] = df["source_regulator"].map(protein_to_i)
    df["sign"] = df[sign_col].map(safe_int).astype(int)
    df["edge_weight"] = df[score_col].map(safe_float) if score_col else 1.0
    df = df[df["site_index"].notna() & df["source_protein_index"].notna()].copy()
    df["site_index"] = df["site_index"].astype(int)
    df["source_protein_index"] = df["source_protein_index"].astype(int)

    for col, default in (("edge_type", "protein_to_site"), ("source_db", "V9")):
        if col not in df.columns:
            df[col] = default
    front = [
        "source_protein_index",
        "source_regulator",
        "site_index",
        "target_index",
        "sign",
        "edge_type",
        "source_db",
        "edge_weight",
    ]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest].sort_values(["source_protein_index", "site_index", "sign"]).reset_index(drop=True)


def load_signed_protein_protein_edges(v9_dir, protein_to_i):
    path = v9_dir / "tables" / "signed_regulator_regulator_edges.tsv"
    if not path.exists():
        return pd.DataFrame(columns=SIGNED_PROTEIN_PROTEIN_COLUMNS)
    df = read_tsv(path).copy()
    source_col = pick_col(df, ["source_regulator", "source"])
    target_col = pick_col(df, ["target_regulator", "target"])
    sign_col = pick_col(df, ["sign"])
    score_col = pick_col(df, ["evidence_score", "score", "weight"], required=False)

    df["source_regulator"] = df[source_col].map(norm_gene)
    df["target_regulator"] = df[target_col].map(norm_gene)
    df["source_protein_index"] = df["source_regulator"].map(protein_to_i)
    df["target_protein_index"] = df["target_regulator"].map(protein_to_i)
    df["sign"] = df[sign_col].map(safe_int).astype(int)
    df["edge_weight"] = df[score_col].map(safe_float) if score_col else 1.0
    df = df[df["source_protein_index"].notna() & df["target_protein_index"].notna()].copy()
    df["source_protein_index"] = df["source_protein_index"].astype(int)
    df["target_protein_index"] = df["target_protein_index"].astype(int)

    for col, default in (("edge_type", "protein_to_protein"), ("source_db", "V9")):
        if col not in df.columns:
            df[col] = default
    front = [
        "source_protein_index",
        "target_protein_index",
        "source_regulator",
        "target_regulator",
        "sign",
        "edge_type",
        "source_db",
        "edge_weight",
    ]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest].sort_values(["source_protein_index", "target_protein_index", "sign"]).reset_index(drop=True)


def string_download_paths(raw_dir, version):
    info = raw_dir / f"9606.protein.info.v{version}.txt.gz"
    links = raw_dir / f"9606.protein.links.v{version}.txt.gz"
    info_url = f"https://stringdb-downloads.org/download/protein.info.v{version}/9606.protein.info.v{version}.txt.gz"
    links_url = f"https://stringdb-downloads.org/download/protein.links.v{version}/9606.protein.links.v{version}.txt.gz"
    return info_url, info, links_url, links


def load_string_info_mapping(info_gz, genes):
    mapping = {}
    id_to_name = {}
    with gzip.open(info_gz, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        protein_idx = header.index("#string_protein_id") if "#string_protein_id" in header else header.index("string_protein_id")
        preferred_idx = header.index("preferred_name")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(protein_idx, preferred_idx):
                continue
            string_id = parts[protein_idx]
            preferred = norm_gene(parts[preferred_idx])
            if preferred in genes:
                mapping[preferred] = string_id
                id_to_name[string_id] = preferred
    return mapping, id_to_name


def query_string_ids_api(genes, raw_dir, batch_size, timeout, existing_mapping):
    missing = sorted(g for g in genes if g not in existing_mapping)
    mapping = dict(existing_mapping)
    id_to_name = {v: k for k, v in existing_mapping.items()}
    statuses = []
    for batch_i, start in enumerate(range(0, len(missing), batch_size)):
        batch = missing[start : start + batch_size]
        params = {
            "identifiers": "\r".join(batch),
            "species": str(STRING_SPECIES),
            "limit": "1",
            "echo_query": "1",
            "caller_identity": CALLER_IDENTITY,
        }
        url = "https://string-db.org/api/tsv/get_string_ids?" + urllib.parse.urlencode(params)
        cache = raw_dir / f"string_get_ids_batch_{batch_i:04d}.tsv"
        text, status, error = string_get(url, cache, timeout=timeout)
        statuses.append({"batch": batch_i, "n_query": len(batch), "status": status, "error": error})
        if not text:
            continue
        rows = [ln.split("\t") for ln in text.splitlines() if ln.strip()]
        if len(rows) < 2:
            continue
        header = rows[0]
        col = {name: i for i, name in enumerate(header)}
        for parts in rows[1:]:
            if len(parts) < len(header):
                continue
            query = norm_gene(parts[col.get("queryItem", col.get("preferredName", 0))])
            string_id = parts[col.get("stringId", -1)] if "stringId" in col else ""
            preferred = norm_gene(parts[col.get("preferredName", col.get("queryItem", 0))])
            gene = query if query in genes else preferred
            if gene in genes and string_id:
                mapping[gene] = string_id
                id_to_name[string_id] = gene
    return mapping, id_to_name, statuses


def build_string_edges_from_download(genes, raw_dir, required_score, version, timeout, api_batch_size):
    ensure_dir(raw_dir)
    info_url, info_path, links_url, links_path = string_download_paths(raw_dir, version)
    info_status = download_with_cache(info_url, info_path, timeout=timeout, min_bytes=1024)
    links_status = download_with_cache(links_url, links_path, timeout=timeout, min_bytes=1024)
    if not info_status["ok"] or not links_status["ok"]:
        return pd.DataFrame(), {
            "mode": "download",
            "ok": False,
            "info": info_status,
            "links": links_status,
            "error": "; ".join(x["error"] for x in (info_status, links_status) if x["error"]),
        }

    info_mapping, id_to_name = load_string_info_mapping(info_path, genes)
    api_mapping, api_id_to_name, api_statuses = query_string_ids_api(
        genes, raw_dir, api_batch_size, timeout, info_mapping
    )
    id_to_name.update(api_id_to_name)
    string_ids = set(api_mapping.values())

    rows = []
    with gzip.open(links_path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split()
        p1_idx = header.index("protein1")
        p2_idx = header.index("protein2")
        score_idx = header.index("combined_score")
        for line in handle:
            parts = line.rstrip("\n").split()
            if len(parts) <= score_idx:
                continue
            p1 = parts[p1_idx]
            p2 = parts[p2_idx]
            if p1 not in string_ids or p2 not in string_ids:
                continue
            raw_score = safe_float(parts[score_idx], 0.0)
            if raw_score < required_score:
                continue
            g1 = id_to_name.get(p1, "")
            g2 = id_to_name.get(p2, "")
            if not g1 or not g2 or g1 == g2:
                continue
            rows.append(
                {
                    "source_gene": g1,
                    "target_gene": g2,
                    "score": float(raw_score / 1000.0),
                    "combined_score": int(raw_score),
                    "source_db": "STRING",
                    "edge_type": "unsigned_protein_protein",
                    "string_id_1": p1,
                    "string_id_2": p2,
                }
            )
    edges = collapse_string_edges(pd.DataFrame(rows))
    status = {
        "mode": "download",
        "ok": True,
        "info": info_status,
        "links": links_status,
        "n_genes_requested": int(len(genes)),
        "n_genes_mapped": int(len(api_mapping)),
        "n_edges": int(len(edges)),
        "api_mapping_batches": api_statuses,
    }
    return edges, status


def build_string_edges_from_api(genes, raw_dir, batch_size, required_score, timeout):
    ensure_dir(raw_dir)
    rows = []
    statuses = []
    sorted_genes = sorted(genes)
    for batch_i, start in enumerate(range(0, len(sorted_genes), batch_size)):
        batch = sorted_genes[start : start + batch_size]
        params = {
            "identifiers": "\r".join(batch),
            "species": str(STRING_SPECIES),
            "required_score": str(int(required_score)),
            "caller_identity": CALLER_IDENTITY,
        }
        url = "https://string-db.org/api/tsv/network?" + urllib.parse.urlencode(params)
        cache = raw_dir / f"string_network_batch_{batch_i:04d}.tsv"
        text, status, error = string_get(url, cache, timeout=timeout)
        statuses.append({"batch": batch_i, "n_query": len(batch), "status": status, "error": error})
        if not text:
            continue
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        header = lines[0].split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < len(header):
                continue
            g1 = norm_gene(parts[col.get("preferredName_A", 0)])
            g2 = norm_gene(parts[col.get("preferredName_B", 1)])
            if g1 not in genes or g2 not in genes or g1 == g2:
                continue
            score = safe_float(parts[col["score"]], 0.0) if "score" in col else 0.0
            if score > 1.0:
                score = score / 1000.0
            rows.append(
                {
                    "source_gene": g1,
                    "target_gene": g2,
                    "score": float(score),
                    "combined_score": int(round(score * 1000.0)),
                    "source_db": "STRING",
                    "edge_type": "unsigned_protein_protein",
                    "string_id_1": parts[col["stringId_A"]] if "stringId_A" in col else "",
                    "string_id_2": parts[col["stringId_B"]] if "stringId_B" in col else "",
                }
            )
        time.sleep(0.2)
    edges = collapse_string_edges(pd.DataFrame(rows))
    ok = bool(len(edges)) or all(s["status"] != "failed" for s in statuses)
    status = {
        "mode": "api_network_chunked",
        "ok": ok,
        "n_genes_requested": int(len(genes)),
        "n_edges": int(len(edges)),
        "required_score": int(required_score),
        "batch_size": int(batch_size),
        "warning": "chunked network API can miss edges spanning different batches" if len(sorted_genes) > batch_size else "",
        "batches": statuses,
    }
    return edges, status


def collapse_string_edges(df):
    if df.empty:
        return pd.DataFrame(columns=UNSIGNED_PROTEIN_PROTEIN_COLUMNS)
    rows = []
    for row in df.itertuples(index=False):
        g1 = norm_gene(getattr(row, "source_gene"))
        g2 = norm_gene(getattr(row, "target_gene"))
        if not g1 or not g2 or g1 == g2:
            continue
        a, b = sorted([g1, g2])
        rows.append(
            {
                "source_gene": a,
                "target_gene": b,
                "score": float(getattr(row, "score")),
                "combined_score": int(getattr(row, "combined_score")),
                "source_db": "STRING",
                "edge_type": "unsigned_protein_protein",
                "string_id_1": getattr(row, "string_id_1", ""),
                "string_id_2": getattr(row, "string_id_2", ""),
            }
        )
    if not rows:
        return pd.DataFrame(columns=UNSIGNED_PROTEIN_PROTEIN_COLUMNS)
    out = pd.DataFrame(rows)
    out = (
        out.sort_values("score", ascending=False)
        .groupby(["source_gene", "target_gene"], as_index=False)
        .agg(
            score=("score", "max"),
            combined_score=("combined_score", "max"),
            source_db=("source_db", "first"),
            edge_type=("edge_type", "first"),
            string_id_1=("string_id_1", "first"),
            string_id_2=("string_id_2", "first"),
        )
    )
    return out.sort_values(["source_gene", "target_gene"]).reset_index(drop=True)


def load_string_edges(genes, protein_to_i, args, allowed_string_genes=None):
    if args.skip_string:
        return pd.DataFrame(columns=UNSIGNED_PROTEIN_PROTEIN_COLUMNS), {
            "ok": False,
            "skipped": True,
            "error": "skip_string requested",
        }

    raw_dir = Path(args.raw_string_dir)
    if args.string_mode in {"download", "auto"}:
        edges, status = build_string_edges_from_download(
            genes,
            raw_dir,
            args.string_required_score,
            args.string_version,
            args.string_timeout,
            args.string_batch_size,
        )
        if status.get("ok"):
            edges = attach_string_indices(edges, protein_to_i, allowed_string_genes, args.string_topk_per_gene)
            return edges, status
        if args.string_mode == "download":
            return attach_string_indices(edges, protein_to_i, allowed_string_genes, args.string_topk_per_gene), status

    edges, api_status = build_string_edges_from_api(
        genes,
        raw_dir,
        args.string_batch_size,
        args.string_required_score,
        args.string_timeout,
    )
    edges = attach_string_indices(edges, protein_to_i, allowed_string_genes, args.string_topk_per_gene)
    return edges, api_status


def attach_string_indices(edges, protein_to_i, allowed_genes=None, topk_per_gene=0):
    if edges.empty:
        return pd.DataFrame(columns=UNSIGNED_PROTEIN_PROTEIN_COLUMNS)
    edges = edges.copy()
    if allowed_genes is not None:
        allowed_genes = set(allowed_genes)
        edges = edges[edges["source_gene"].isin(allowed_genes) & edges["target_gene"].isin(allowed_genes)].copy()
    if edges.empty:
        return pd.DataFrame(columns=UNSIGNED_PROTEIN_PROTEIN_COLUMNS)
    if topk_per_gene and int(topk_per_gene) > 0:
        topk_per_gene = int(topk_per_gene)
        ranked = []
        left = edges.sort_values(["source_gene", "score"], ascending=[True, False]).groupby("source_gene").head(topk_per_gene)
        right = edges.sort_values(["target_gene", "score"], ascending=[True, False]).groupby("target_gene").head(topk_per_gene)
        ranked.append(left)
        ranked.append(right)
        edges = pd.concat(ranked, axis=0, ignore_index=True).drop_duplicates(["source_gene", "target_gene"])
    edges["source_protein_index"] = edges["source_gene"].map(protein_to_i)
    edges["target_protein_index"] = edges["target_gene"].map(protein_to_i)
    edges = edges[edges["source_protein_index"].notna() & edges["target_protein_index"].notna()].copy()
    edges["source_protein_index"] = edges["source_protein_index"].astype(int)
    edges["target_protein_index"] = edges["target_protein_index"].astype(int)
    front = [
        "source_protein_index",
        "target_protein_index",
        "source_gene",
        "target_gene",
        "score",
        "combined_score",
        "source_db",
        "edge_type",
    ]
    rest = [c for c in edges.columns if c not in front]
    return edges[front + rest].sort_values(["source_protein_index", "target_protein_index"]).reset_index(drop=True)


def load_site_site_cophee_edges(cophee_dir, site_index_by_target):
    path = cophee_dir / "tables" / "copheemap_model_site_site_edges.tsv"
    if not path.exists():
        return pd.DataFrame(columns=SITE_SITE_COPHEE_COLUMNS)
    df = read_tsv(path).copy()
    c1 = pick_col(df, ["target_index_1", "source_target_index", "source"])
    c2 = pick_col(df, ["target_index_2", "target_target_index", "target"])
    weight_col = pick_col(df, ["edge_weight", "score", "weight"], required=False)
    df["target_index_1"] = df[c1].astype(int)
    df["target_index_2"] = df[c2].astype(int)
    df["site_index_1"] = df["target_index_1"].map(site_index_by_target)
    df["site_index_2"] = df["target_index_2"].map(site_index_by_target)
    df = df[df["site_index_1"].notna() & df["site_index_2"].notna()].copy()
    df["site_index_1"] = df["site_index_1"].astype(int)
    df["site_index_2"] = df["site_index_2"].astype(int)
    df["edge_weight"] = df[weight_col].map(safe_float) if weight_col else 1.0
    df["source_db"] = "CoPheeMap"
    df["edge_type"] = "site_site"
    front = [
        "site_index_1",
        "site_index_2",
        "target_index_1",
        "target_index_2",
        "edge_weight",
        "source_db",
        "edge_type",
    ]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest].sort_values(["site_index_1", "edge_weight"], ascending=[True, False]).reset_index(drop=True)


def save_dense_matrices(out_dir, n_proteins, n_sites, membership, signed_rs, signed_pp, unsigned_pp):
    arrays = out_dir / "arrays"
    ensure_dir(arrays)

    membership_mat = np.zeros((n_proteins, n_sites), dtype=np.int8)
    for row in membership.itertuples(index=False):
        membership_mat[int(row.protein_index), int(row.site_index)] = 1
    np.save(arrays / "protein_site_membership.npy", membership_mat)

    rs_mat = np.zeros((n_proteins, n_sites), dtype=np.float32)
    if not signed_rs.empty:
        for row in signed_rs.itertuples(index=False):
            rs_mat[int(row.source_protein_index), int(row.site_index)] += float(row.sign) * float(row.edge_weight)
    np.save(arrays / "signed_regulator_site_matrix.npy", rs_mat)

    pp_mat = np.zeros((n_proteins, n_proteins), dtype=np.float32)
    if not signed_pp.empty:
        for row in signed_pp.itertuples(index=False):
            pp_mat[int(row.source_protein_index), int(row.target_protein_index)] += (
                float(row.sign) * float(row.edge_weight)
            )
    np.save(arrays / "signed_protein_protein_matrix.npy", pp_mat)

    unsigned_mat = np.zeros((n_proteins, n_proteins), dtype=np.float32)
    if not unsigned_pp.empty:
        for row in unsigned_pp.itertuples(index=False):
            i = int(row.source_protein_index)
            j = int(row.target_protein_index)
            score = float(row.score)
            if score > unsigned_mat[i, j]:
                unsigned_mat[i, j] = score
                unsigned_mat[j, i] = score
    np.save(arrays / "unsigned_protein_protein_matrix.npy", unsigned_mat)

    return {
        "protein_site_membership": str(arrays / "protein_site_membership.npy"),
        "signed_regulator_site": str(arrays / "signed_regulator_site_matrix.npy"),
        "signed_protein_protein": str(arrays / "signed_protein_protein_matrix.npy"),
        "unsigned_protein_protein": str(arrays / "unsigned_protein_protein_matrix.npy"),
    }


def save_site_site_matrix(out_dir, n_sites, edges):
    arrays = out_dir / "arrays"
    ensure_dir(arrays)
    npz_path = arrays / "site_site_cophee_matrix.npz"
    npy_path = arrays / "site_site_cophee_matrix.npy"
    try:
        from scipy import sparse

        if edges.empty:
            mat = sparse.csr_matrix((n_sites, n_sites), dtype=np.float32)
        else:
            ii = []
            jj = []
            vv = []
            for row in edges.itertuples(index=False):
                i = int(row.site_index_1)
                j = int(row.site_index_2)
                w = float(row.edge_weight)
                ii.extend([i, j])
                jj.extend([j, i])
                vv.extend([w, w])
            mat = sparse.coo_matrix((vv, (ii, jj)), shape=(n_sites, n_sites), dtype=np.float32).tocsr()
            mat.eliminate_zeros()
        sparse.save_npz(npz_path, mat)
        return {"path": str(npz_path), "format": "scipy_sparse_npz", "nnz": int(mat.nnz)}
    except Exception as exc:
        mat = np.zeros((n_sites, n_sites), dtype=np.float32)
        if not edges.empty:
            for row in edges.itertuples(index=False):
                i = int(row.site_index_1)
                j = int(row.site_index_2)
                w = float(row.edge_weight)
                if w > mat[i, j]:
                    mat[i, j] = w
                    mat[j, i] = w
        np.save(npy_path, mat)
        return {"path": str(npy_path), "format": "dense_npy", "error": str(exc)}


def write_tables(out_dir, protein_nodes, site_nodes, membership, signed_rs, signed_pp, unsigned_pp, site_site):
    tables = out_dir / "tables"
    ensure_dir(tables)
    protein_nodes.to_csv(tables / "protein_nodes.tsv", sep="\t", index=False)
    site_nodes.to_csv(tables / "site_nodes.tsv", sep="\t", index=False)
    membership.to_csv(tables / "protein_site_membership_edges.tsv", sep="\t", index=False)
    signed_rs.to_csv(tables / "signed_regulator_site_edges.tsv", sep="\t", index=False)
    signed_pp.to_csv(tables / "signed_protein_protein_edges.tsv", sep="\t", index=False)
    unsigned_pp.to_csv(tables / "unsigned_protein_protein_edges.tsv", sep="\t", index=False)
    site_site.to_csv(tables / "site_site_cophee_edges.tsv", sep="\t", index=False)


def compute_isolated_proteins(n_proteins, membership, signed_rs, signed_pp, unsigned_pp):
    degree = np.zeros(n_proteins, dtype=np.int64)
    for df, cols in (
        (membership, ["protein_index"]),
        (signed_rs, ["source_protein_index"]),
        (signed_pp, ["source_protein_index", "target_protein_index"]),
        (unsigned_pp, ["source_protein_index", "target_protein_index"]),
    ):
        if df.empty:
            continue
        for col in cols:
            if col in df.columns:
                vals = df[col].dropna().astype(int).to_numpy()
                degree[vals] += 1
    isolated = np.flatnonzero(degree == 0)
    return isolated, degree


def optional_p100_shared_site_coverage(sites, root):
    candidates = [
        root / r"01_data\pathway_prior\intermediate\lincs_p100_comparison_delta_v7\tables\shared_sites.tsv",
        root / r"01_data\pathway_prior\intermediate\lincs_p100_comparison_delta_v7\tables\p100_shared_sites.tsv",
        root / r"01_data\single_cell\intermediate\phospho_perturb\joint_decryptm_p100\tables\shared_sites.tsv",
    ]
    target_ids = set(sites["target_id"].astype(str))
    target_indices = set(sites["target_index"].astype(int))
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = read_tsv(path)
            if "target_index" in df.columns:
                n_total = int(df["target_index"].nunique())
                n_hit = int(df["target_index"].astype(int).isin(target_indices).sum())
            elif "target_id" in df.columns:
                n_total = int(df["target_id"].nunique())
                n_hit = int(df["target_id"].astype(str).isin(target_ids).sum())
            else:
                continue
            return {
                "available": True,
                "path": str(path),
                "n_p100_shared_sites": n_total,
                "n_covered": n_hit,
                "coverage": float(n_hit / n_total) if n_total else None,
            }
        except Exception as exc:
            return {"available": False, "path": str(path), "error": str(exc)}
    return {"available": False}


def build_summary(
    args,
    sites,
    protein_nodes,
    v9_regulators,
    membership,
    signed_rs,
    signed_pp,
    unsigned_pp,
    site_site,
    string_status,
    matrix_paths,
    site_site_matrix_info,
):
    n_proteins = int(len(protein_nodes))
    n_sites = int(len(sites))
    decryptm_genes = set(g for genes in sites["site_genes"] for g in genes)
    protein_genes = set(protein_nodes["gene"].map(norm_gene))
    isolated, degree = compute_isolated_proteins(n_proteins, membership, signed_rs, signed_pp, unsigned_pp)

    training_targets = {}
    for drug, genes in TRAINING_TARGETS.items():
        training_targets[drug] = {gene: bool(norm_gene(gene) in protein_genes) for gene in genes}

    covered_sites = int(membership["site_index"].nunique()) if not membership.empty else 0
    summary = {
        "n_proteins": n_proteins,
        "n_sites": n_sites,
        "site_membership_coverage": float(covered_sites / n_sites) if n_sites else 0.0,
        "n_site_membership_edges": int(len(membership)),
        "n_decryptm_phosphoproteins": int(len(decryptm_genes)),
        "n_decryptm_phosphoproteins_in_network": int(len(decryptm_genes & protein_genes)),
        "decryptM_phosphoprotein_network_rate": float(len(decryptm_genes & protein_genes) / len(decryptm_genes))
        if decryptm_genes
        else 0.0,
        "n_v9_regulators": int(len(v9_regulators)),
        "n_v9_regulators_in_network": int(len(set(v9_regulators) & protein_genes)),
        "v9_regulator_network_rate": float(len(set(v9_regulators) & protein_genes) / len(v9_regulators))
        if v9_regulators
        else 0.0,
        "n_signed_regulator_site_edges": int(len(signed_rs)),
        "n_signed_protein_protein_edges": int(len(signed_pp)),
        "n_unsigned_protein_protein_edges": int(len(unsigned_pp)),
        "n_site_site_cophee_edges": int(len(site_site)),
        "n_isolated_proteins": int(len(isolated)),
        "isolated_protein_ratio": float(len(isolated) / n_proteins) if n_proteins else 0.0,
        "training_target_genes_in_network": training_targets,
        "string_failed": bool(not string_status.get("ok", False)),
        "string_status": string_status,
        "matrix_paths": matrix_paths,
        "site_site_matrix": site_site_matrix_info,
        "p100_shared_site_coverage": optional_p100_shared_site_coverage(sites, Path(args.root)),
        "source_paths": {
            "decryptm_site_table": str(args.site_table),
            "v9_dir": str(args.v9_dir),
            "cophee_dir": str(args.cophee_dir),
            "raw_string_dir": str(args.raw_string_dir),
            "output_dir": str(args.output_dir),
        },
        "uniprot_mapping_status": sites.attrs.get("uniprot_status", {}),
        "degree_summary": {
            "protein_degree_min": int(degree.min()) if len(degree) else 0,
            "protein_degree_median": float(np.median(degree)) if len(degree) else 0.0,
            "protein_degree_max": int(degree.max()) if len(degree) else 0,
        },
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--site-table", default=str(DECRYPTM_SITE_TABLE))
    parser.add_argument("--v9-dir", default=str(V9_DIR))
    parser.add_argument("--cophee-dir", default=str(COPHEE_DIR))
    parser.add_argument("--raw-string-dir", default=str(RAW_STRING_DIR))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--string-mode", choices=["auto", "download", "api"], default="auto")
    parser.add_argument("--string-version", default=STRING_VERSION)
    parser.add_argument("--string-required-score", type=int, default=400)
    parser.add_argument("--string-batch-size", type=int, default=500)
    parser.add_argument("--string-timeout", type=int, default=300)
    parser.add_argument("--string-scope", choices=["all_network", "decryptm_phosphoproteins"], default="all_network")
    parser.add_argument("--string-topk-per-gene", type=int, default=0)
    parser.add_argument("--skip-string", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.root = Path(args.root)
    args.site_table = Path(args.site_table)
    args.v9_dir = Path(args.v9_dir)
    args.cophee_dir = Path(args.cophee_dir)
    args.raw_string_dir = Path(args.raw_string_dir)
    args.output_dir = Path(args.output_dir)

    for sub in ("tables", "arrays", "reports"):
        ensure_dir(args.output_dir / sub)

    sites = load_sites(args.site_table)
    site_nodes = build_site_nodes(sites)
    site_index_by_target = dict(zip(sites["target_index"].astype(int), sites["site_index"].astype(int)))

    v9_regulators = load_v9_regulators(args.v9_dir)
    protein_nodes, protein_to_i = build_protein_nodes(sites, v9_regulators)
    membership = build_membership_edges(sites, protein_to_i)

    signed_rs = load_signed_regulator_site_edges(args.v9_dir, site_index_by_target, protein_to_i)
    signed_pp = load_signed_protein_protein_edges(args.v9_dir, protein_to_i)
    decryptm_gene_set = set(g for genes in sites["site_genes"] for g in genes)
    allowed_string_genes = decryptm_gene_set if args.string_scope == "decryptm_phosphoproteins" else None
    unsigned_pp, string_status = load_string_edges(set(protein_nodes["gene"]), protein_to_i, args, allowed_string_genes)
    string_status["string_scope"] = args.string_scope
    string_status["string_topk_per_gene"] = int(args.string_topk_per_gene)
    string_status["n_allowed_string_genes"] = int(len(allowed_string_genes)) if allowed_string_genes is not None else int(len(protein_nodes))
    site_site = load_site_site_cophee_edges(args.cophee_dir, site_index_by_target)

    write_tables(args.output_dir, protein_nodes, site_nodes, membership, signed_rs, signed_pp, unsigned_pp, site_site)
    matrix_paths = save_dense_matrices(
        args.output_dir,
        len(protein_nodes),
        len(site_nodes),
        membership,
        signed_rs,
        signed_pp,
        unsigned_pp,
    )
    site_site_matrix_info = save_site_site_matrix(args.output_dir, len(site_nodes), site_site)

    summary = build_summary(
        args,
        sites,
        protein_nodes,
        v9_regulators,
        membership,
        signed_rs,
        signed_pp,
        unsigned_pp,
        site_site,
        string_status,
        matrix_paths,
        site_site_matrix_info,
    )
    report_path = args.output_dir / "reports" / "global_heterograph_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
