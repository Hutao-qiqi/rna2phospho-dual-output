#!/usr/bin/env python3
import csv
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path("/data/lsy/Infinite_Stream")
BASE = ROOT / "01_data/pathway_prior/intermediate/open_kinase_substrate"
META = ROOT / "01_data/pathway_prior/metadata"
RESULT = ROOT / "02_results/pathway_prior/20260425_open_kinase_prior_audit"
TABLES = RESULT / "tables"
LOGS = RESULT / "logs"

MAIN = BASE / "open_kinase_substrate_prior_strict.tsv"
PN = BASE / "phosphonetworks_kinase_substrate_long.tsv"
OP = BASE / "omnipath_enzsub_no_phosphosite.tsv"
KINHUB_HTML = META / "kinhub_human_kinases.html"
KINHUB_TSV = META / "kinhub_human_kinases.tsv"


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_tbody = False
        self.in_tr = False
        self.in_td = False
        self.cur = []
        self.row = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == "tbody":
            self.in_tbody = True
        elif tag == "tr":
            self.in_tr = True
            self.row = []
        elif tag == "td":
            self.in_td = True
            self.cur = []

    def handle_endtag(self, tag):
        if tag == "td":
            self.in_td = False
            self.row.append("".join(self.cur).strip())
        elif tag == "tr":
            if self.in_tbody and self.row:
                self.rows.append(self.row)
            self.in_tr = False
        elif tag == "tbody":
            self.in_tbody = False

    def handle_data(self, data):
        if self.in_td:
            self.cur.append(data)


def read_rows(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path, rows, fields):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def is_phos(row):
    return row.get("modification", "").strip().lower() == "phosphorylation"


def site_key(row):
    site = row.get("substrate_site", "").strip()
    if site:
        return f"{row['substrate_gene']}:{site}"
    residue_type = row.get("residue_type", "").strip()
    residue_offset = row.get("residue_offset", "").strip()
    if residue_type and residue_offset:
        return f"{row['substrate_gene']}:{residue_type}{residue_offset}"
    return ""


def edge_gene(row):
    return (row["kinase_gene"].strip(), row["substrate_gene"].strip())


def edge_site(row):
    target = site_key(row) or f"{row['substrate_gene'].strip()}:NO_SITE"
    return (row["kinase_gene"].strip(), target)


def parse_kinhub():
    parser = TableParser()
    parser.feed(KINHUB_HTML.read_text(errors="ignore"))
    fields = ["xName", "Manning", "HGNC", "Kinase_Name", "Group", "Family", "SubFamily", "UniprotID"]
    rows = []
    for row in parser.rows:
        if len(row) >= 8:
            rows.append(dict(zip(fields, row[:8])))
    write_tsv(KINHUB_TSV, rows, fields)
    return rows


def kinome_filtered_stats(label, rows, kinases):
    phos = [r for r in rows if is_phos(r)]
    kinase_rows = [r for r in phos if r.get("kinase_gene", "").strip() in kinases]
    dropped = [r for r in phos if r.get("kinase_gene", "").strip() not in kinases]
    kset = {r["kinase_gene"].strip() for r in kinase_rows}
    sites = {site_key(r) for r in kinase_rows if site_key(r)}
    substrates = {r["substrate_gene"].strip() for r in kinase_rows if r.get("substrate_gene", "").strip()}
    gene_edges = {edge_gene(r) for r in kinase_rows}
    site_edges = {edge_site(r) for r in kinase_rows}
    return {
        "subset": label,
        "phosphorylation_records_before_filter": len(phos),
        "records_after_kinhub_filter": len(kinase_rows),
        "records_dropped_as_non_kinhub_enzyme": len(dropped),
        "kinhub_kinase_count": len(kinases),
        "covered_kinhub_kinases": len(kset),
        "kinhub_coverage_percent": f"{100 * len(kset) / len(kinases):.1f}",
        "unique_substrate_genes": len(substrates),
        "unique_positioned_substrate_sites": len(sites),
        "unique_gene_level_edges": len(gene_edges),
        "unique_site_aware_edges": len(site_edges),
    }


def rtk_filtered(rows, kinases):
    out = []
    phos = [r for r in rows if is_phos(r) and r.get("kinase_gene", "").strip() in kinases]
    for kinase in ["EGFR", "ERBB2", "MET", "PDGFRA"]:
        kr = [r for r in phos if r.get("kinase_gene", "").strip() == kinase]
        sites = {site_key(r) for r in kr if site_key(r)}
        subs = {r["substrate_gene"].strip() for r in kr if r.get("substrate_gene", "").strip()}
        if len(sites) < 10:
            flag = "HIGH_RISK_LT10"
        elif len(sites) < 20:
            flag = "WEAK_LT20"
        else:
            flag = "OK"
        out.append({
            "kinase": kinase,
            "records": len(kr),
            "unique_substrate_genes": len(subs),
            "unique_positioned_substrate_sites": len(sites),
            "unique_site_aware_edges": len({edge_site(r) for r in kr}),
            "flag": flag,
        })
    return out


def top_nonkinase(rows, kinases):
    phos = [r for r in rows if is_phos(r)]
    outdeg = Counter()
    for r in phos:
        k = r.get("kinase_gene", "").strip()
        if k and k not in kinases:
            outdeg[k] += 1
    return [{"enzyme_symbol": k, "phosphorylation_records": n} for k, n in outdeg.most_common(50)]


def degree_filtered(rows, kinases):
    phos = [r for r in rows if is_phos(r) and r.get("kinase_gene", "").strip() in kinases]
    edges = {edge_site(r) for r in phos}
    kinase_nodes = {k for k, _ in edges}
    target_nodes = {t for _, t in edges}
    outdeg = Counter(k for k, _ in edges)
    indeg = Counter(t for _, t in edges)
    node_n = len(kinase_nodes) + len(target_nodes)
    avg_degree = 2 * len(edges) / node_n if node_n else 0
    density = len(edges) / (len(kinase_nodes) * len(target_nodes)) if kinase_nodes and target_nodes else 0
    top = []
    for k, n in outdeg.most_common(30):
        top.append({
            "kinase": k,
            "outdegree": n,
            "flag": "SUPER_NODE_GT1000" if n > 1000 else ("BROAD_GT500" if n > 500 else ""),
        })
    summary = [{
        "subset": "main_phosphorylation_kinhub_filtered",
        "unique_site_aware_edges": len(edges),
        "kinase_nodes": len(kinase_nodes),
        "substrate_site_nodes": len(target_nodes),
        "average_node_degree": f"{avg_degree:.3f}",
        "bipartite_density": f"{density:.8f}",
        "max_kinase_outdegree": max(outdeg.values()) if outdeg else 0,
        "max_target_indegree": max(indeg.values()) if indeg else 0,
    }]
    return summary, top


def overlap_filtered(pn, op, kinases):
    pn_phos = [r for r in pn if is_phos(r) and r.get("kinase_gene", "").strip() in kinases]
    op_phos = [r for r in op if is_phos(r) and r.get("kinase_gene", "").strip() in kinases]
    rows = []
    for label, keyfunc, require_site in [
        ("gene_pair", edge_gene, False),
        ("positioned_site_only", edge_site, True),
    ]:
        pset = {keyfunc(r) for r in pn_phos if not require_site or site_key(r)}
        oset = {keyfunc(r) for r in op_phos if not require_site or site_key(r)}
        inter = pset & oset
        union = pset | oset
        j = len(inter) / len(union) if union else 0
        rows.append({
            "edge_level": label,
            "phosphonetworks_edges": len(pset),
            "omnipath_strict_edges": len(oset),
            "intersection_edges": len(inter),
            "union_edges": len(union),
            "jaccard_percent": f"{100*j:.1f}",
            "phosphonetworks_overlap_percent": f"{100*len(inter)/len(pset):.1f}" if pset else "0.0",
            "omnipath_overlap_percent": f"{100*len(inter)/len(oset):.1f}" if oset else "0.0",
            "flag": "LOW_LT10" if j < 0.10 else ("TARGET_10_TO_40" if j <= 0.40 else "HIGH_GT40"),
        })
    return rows


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    kinhub = parse_kinhub()
    kinases = {r["HGNC"].strip() for r in kinhub if r.get("HGNC", "").strip()}
    main = read_rows(MAIN)
    pn = read_rows(PN)
    op = read_rows(OP)

    coverage = [
        kinome_filtered_stats("main_phosphorylation_kinhub_filtered", main, kinases),
        kinome_filtered_stats("phosphonetworks_phosphorylation_kinhub_filtered", pn, kinases),
        kinome_filtered_stats("omnipath_strict_phosphorylation_kinhub_filtered", op, kinases),
    ]
    write_tsv(TABLES / "coverage_summary_kinhub_filtered.tsv", coverage, list(coverage[0].keys()))
    rtk = rtk_filtered(main, kinases)
    write_tsv(TABLES / "rtk_coverage_kinhub_filtered.tsv", rtk, list(rtk[0].keys()))
    top_non = top_nonkinase(main, kinases)
    write_tsv(TABLES / "top_non_kinhub_enzymes.tsv", top_non, list(top_non[0].keys()))
    degree_summary, top_k = degree_filtered(main, kinases)
    write_tsv(TABLES / "graph_degree_summary_kinhub_filtered.tsv", degree_summary, list(degree_summary[0].keys()))
    write_tsv(TABLES / "top_kinases_by_outdegree_kinhub_filtered.tsv", top_k, list(top_k[0].keys()))
    overlap = overlap_filtered(pn, op, kinases)
    write_tsv(TABLES / "source_overlap_kinhub_filtered.tsv", overlap, list(overlap[0].keys()))
    warnings = []
    main_cov = float(coverage[0]["kinhub_coverage_percent"])
    if main_cov < 70:
        warnings.append(f"KinHub-filtered kinase coverage below 70 percent: {main_cov}%")
    for row in rtk:
        if row["flag"] != "OK":
            warnings.append(f"{row['kinase']} has weak positioned site support: {row['unique_positioned_substrate_sites']} ({row['flag']})")
    for row in overlap:
        if row["edge_level"] == "gene_pair" and row["flag"] == "LOW_LT10":
            warnings.append(f"KinHub-filtered source overlap low at gene-pair level: {row['jaccard_percent']}%")
    (LOGS / "audit_warnings_kinhub_filtered.txt").write_text("\n".join(warnings) + "\n")
    print(f"kinhub_rows={len(kinhub)}")
    print(f"kinhub_symbols={len(kinases)}")
    print(f"warnings={len(warnings)}")


if __name__ == "__main__":
    main()
