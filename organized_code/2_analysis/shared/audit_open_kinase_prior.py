#!/usr/bin/env python3
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path("/data/lsy/Infinite_Stream")
BASE = PROJECT_ROOT / "01_data/pathway_prior/intermediate/open_kinase_substrate"
RESULT = PROJECT_ROOT / "02_results/pathway_prior/20260425_open_kinase_prior_audit"
TABLES = RESULT / "tables"
LOGS = RESULT / "logs"

MAIN = BASE / "open_kinase_substrate_prior_strict.tsv"
PN = BASE / "phosphonetworks_kinase_substrate_long.tsv"
OP = BASE / "omnipath_enzsub_no_phosphosite.tsv"

HUMAN_KINOME_DENOM = 518
KINOME_WARNING_COVERAGE = 0.70
RTK_CORE = ["EGFR", "ERBB2", "MET", "PDGFRA"]


def read_rows(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def nonempty(value):
    return value is not None and str(value).strip() != ""


def is_phosphorylation(row):
    return row.get("modification", "").strip().lower() == "phosphorylation"


def site_key(row):
    site = row.get("substrate_site", "").strip()
    residue_type = row.get("residue_type", "").strip()
    residue_offset = row.get("residue_offset", "").strip()
    if site:
        return f"{row['substrate_gene']}:{site}"
    if residue_type and residue_offset:
        return f"{row['substrate_gene']}:{residue_type}{residue_offset}"
    return ""


def edge_gene(row):
    return (row["kinase_gene"].strip(), row["substrate_gene"].strip())


def edge_site(row):
    target = site_key(row)
    if not target:
        target = f"{row['substrate_gene'].strip()}:NO_SITE"
    return (row["kinase_gene"].strip(), target)


def subset_stats(rows, label):
    kinases = {r["kinase_gene"].strip() for r in rows if nonempty(r.get("kinase_gene"))}
    substrates = {r["substrate_gene"].strip() for r in rows if nonempty(r.get("substrate_gene"))}
    sites = {site_key(r) for r in rows if site_key(r)}
    gene_edges = {edge_gene(r) for r in rows if nonempty(r.get("kinase_gene")) and nonempty(r.get("substrate_gene"))}
    site_edges = {edge_site(r) for r in rows if nonempty(r.get("kinase_gene")) and nonempty(r.get("substrate_gene"))}
    mod_counts = Counter(r.get("modification", "").strip() or "NA" for r in rows)
    coverage = len(kinases) / HUMAN_KINOME_DENOM
    return {
        "subset": label,
        "records": len(rows),
        "unique_kinases_or_enzymes": len(kinases),
        "human_kinome_denominator": HUMAN_KINOME_DENOM,
        "kinome_coverage_fraction": f"{coverage:.4f}",
        "kinome_coverage_percent": f"{100 * coverage:.1f}",
        "below_70_percent_flag": "YES" if coverage < KINOME_WARNING_COVERAGE else "NO",
        "unique_substrate_genes": len(substrates),
        "unique_substrate_phosphosites_with_position": len(sites),
        "unique_gene_level_edges": len(gene_edges),
        "unique_site_aware_edges": len(site_edges),
        "modification_counts": ";".join(f"{k}:{v}" for k, v in sorted(mod_counts.items())),
    }


def rtk_stats(rows, label):
    out = []
    for kinase in RTK_CORE:
        kr = [r for r in rows if r.get("kinase_gene", "").strip() == kinase]
        site_edges = {edge_site(r) for r in kr if nonempty(r.get("substrate_gene"))}
        positioned_sites = {site_key(r) for r in kr if site_key(r)}
        substrate_genes = {r["substrate_gene"].strip() for r in kr if nonempty(r.get("substrate_gene"))}
        source_counts = Counter()
        for r in kr:
            for s in (r.get("source_record", "") + ";" + r.get("sources", "")).split(";"):
                s = s.strip()
                if s:
                    source_counts[s] += 1
        n_site = len(positioned_sites)
        if n_site < 10:
            flag = "HIGH_RISK_LT10"
        elif n_site < 20:
            flag = "WEAK_LT20"
        else:
            flag = "OK"
        out.append({
            "subset": label,
            "kinase": kinase,
            "records": len(kr),
            "unique_substrate_genes": len(substrate_genes),
            "unique_substrate_sites_with_position": n_site,
            "unique_site_aware_edges": len(site_edges),
            "coverage_flag": flag,
            "top_sources": ";".join(f"{k}:{v}" for k, v in source_counts.most_common(8)),
        })
    return out


def overlap_stats(pn_rows, op_rows):
    rows = []
    pn_phos = [r for r in pn_rows if is_phosphorylation(r)]
    op_phos = [r for r in op_rows if is_phosphorylation(r)]
    for label, keyfunc, require_site in [
        ("gene_pair", edge_gene, False),
        ("site_aware_with_no_site_bucket", edge_site, False),
        ("positioned_site_only", edge_site, True),
    ]:
        pset = set()
        oset = set()
        for r in pn_phos:
            if require_site and not site_key(r):
                continue
            pset.add(keyfunc(r))
        for r in op_phos:
            if require_site and not site_key(r):
                continue
            oset.add(keyfunc(r))
        inter = pset & oset
        union = pset | oset
        pn_overlap = len(inter) / len(pset) if pset else 0
        op_overlap = len(inter) / len(oset) if oset else 0
        jaccard = len(inter) / len(union) if union else 0
        if jaccard < 0.10:
            flag = "LOW_LT10"
        elif jaccard <= 0.40:
            flag = "TARGET_10_TO_40"
        else:
            flag = "HIGH_GT40"
        rows.append({
            "edge_level": label,
            "phosphonetworks_edges": len(pset),
            "omnipath_strict_edges": len(oset),
            "intersection_edges": len(inter),
            "union_edges": len(union),
            "jaccard_fraction": f"{jaccard:.4f}",
            "jaccard_percent": f"{100*jaccard:.1f}",
            "phosphonetworks_overlap_percent": f"{100*pn_overlap:.1f}",
            "omnipath_overlap_percent": f"{100*op_overlap:.1f}",
            "overlap_flag": flag,
        })
    return rows


def degree_stats(rows, label):
    phos = [r for r in rows if is_phosphorylation(r)]
    edges = {edge_site(r) for r in phos if nonempty(r.get("kinase_gene")) and nonempty(r.get("substrate_gene"))}
    kinases = {k for k, _ in edges}
    targets = {t for _, t in edges}
    outdeg = Counter(k for k, _ in edges)
    indeg = Counter(t for _, t in edges)
    node_degree = Counter()
    for k, t in edges:
        node_degree[k] += 1
        node_degree[t] += 1
    nodes = len(kinases) + len(targets)
    avg_degree = (2 * len(edges) / nodes) if nodes else 0
    density = (len(edges) / (len(kinases) * len(targets))) if kinases and targets else 0
    out_values = sorted(outdeg.values())
    in_values = sorted(indeg.values())

    def pct(values, q):
        if not values:
            return 0
        idx = math.ceil(q * len(values)) - 1
        idx = max(0, min(idx, len(values) - 1))
        return values[idx]

    summary = [{
        "subset": label,
        "phosphorylation_records": len(phos),
        "unique_site_aware_edges": len(edges),
        "kinase_nodes": len(kinases),
        "substrate_site_nodes": len(targets),
        "total_bipartite_nodes": nodes,
        "average_node_degree": f"{avg_degree:.3f}",
        "bipartite_density": f"{density:.8f}",
        "median_kinase_outdegree": pct(out_values, 0.50),
        "p90_kinase_outdegree": pct(out_values, 0.90),
        "p95_kinase_outdegree": pct(out_values, 0.95),
        "p99_kinase_outdegree": pct(out_values, 0.99),
        "max_kinase_outdegree": max(out_values) if out_values else 0,
        "median_target_indegree": pct(in_values, 0.50),
        "p95_target_indegree": pct(in_values, 0.95),
        "max_target_indegree": max(in_values) if in_values else 0,
    }]
    top_nodes = []
    for node, deg in node_degree.most_common(50):
        top_nodes.append({
            "subset": label,
            "node": node,
            "node_type": "kinase" if node in kinases else "substrate_site",
            "degree": deg,
        })
    top_kinases = []
    for kinase, deg in outdeg.most_common(50):
        flag = "SUPER_NODE_GT1000" if deg > 1000 else ("BROAD_GT500" if deg > 500 else "")
        top_kinases.append({
            "subset": label,
            "kinase": kinase,
            "outdegree": deg,
            "flag": flag,
        })
    return summary, top_nodes, top_kinases


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    main_rows = read_rows(MAIN)
    pn_rows = read_rows(PN)
    op_rows = read_rows(OP)
    main_phos = [r for r in main_rows if is_phosphorylation(r)]
    pn_phos = [r for r in pn_rows if is_phosphorylation(r)]
    op_phos = [r for r in op_rows if is_phosphorylation(r)]

    coverage = [
        subset_stats(main_rows, "main_all_modifications"),
        subset_stats(main_phos, "main_phosphorylation_only"),
        subset_stats(pn_phos, "phosphonetworks_phosphorylation"),
        subset_stats(op_phos, "omnipath_strict_phosphorylation"),
    ]
    write_tsv(TABLES / "coverage_summary.tsv", coverage, list(coverage[0].keys()))

    rtk = []
    rtk.extend(rtk_stats(main_phos, "main_phosphorylation_only"))
    rtk.extend(rtk_stats(pn_phos, "phosphonetworks_phosphorylation"))
    rtk.extend(rtk_stats(op_phos, "omnipath_strict_phosphorylation"))
    write_tsv(TABLES / "rtk_coverage.tsv", rtk, list(rtk[0].keys()))

    overlap = overlap_stats(pn_rows, op_rows)
    write_tsv(TABLES / "source_overlap.tsv", overlap, list(overlap[0].keys()))

    degree_summary, top_nodes, top_kinases = degree_stats(main_rows, "main_phosphorylation_only")
    write_tsv(TABLES / "graph_degree_summary.tsv", degree_summary, list(degree_summary[0].keys()))
    write_tsv(TABLES / "top_nodes_by_degree.tsv", top_nodes, list(top_nodes[0].keys()))
    write_tsv(TABLES / "top_kinases_by_outdegree.tsv", top_kinases, list(top_kinases[0].keys()))

    # Distribution of modifications is useful for detecting non-kinase PTM contamination.
    mod_rows = []
    for label, rows in [
        ("main_all_modifications", main_rows),
        ("phosphonetworks", pn_rows),
        ("omnipath_strict", op_rows),
    ]:
        for mod, n in Counter(r.get("modification", "").strip() or "NA" for r in rows).most_common():
            mod_rows.append({"subset": label, "modification": mod, "records": n})
    write_tsv(TABLES / "modification_distribution.tsv", mod_rows, ["subset", "modification", "records"])

    warnings = []
    cov = float(coverage[1]["kinome_coverage_fraction"])
    if cov < KINOME_WARNING_COVERAGE:
        warnings.append(f"Kinase coverage below 70 percent: {coverage[1]['kinome_coverage_percent']}%")
    for row in rtk:
        if row["subset"] == "main_phosphorylation_only" and row["coverage_flag"] != "OK":
            warnings.append(f"{row['kinase']} RTK positioned substrate sites weak: {row['unique_substrate_sites_with_position']} ({row['coverage_flag']})")
    for row in overlap:
        if row["edge_level"] == "gene_pair" and row["overlap_flag"] == "LOW_LT10":
            warnings.append(f"Source overlap low at gene-pair level: {row['jaccard_percent']}%")
    for row in top_kinases:
        if row["flag"] == "SUPER_NODE_GT1000":
            warnings.append(f"Super-node kinase candidate: {row['kinase']} outdegree {row['outdegree']}")
    (LOGS / "audit_warnings.txt").write_text("\n".join(warnings) + ("\n" if warnings else "No hard warning.\n"))
    print(f"results={RESULT}")
    print(f"warnings={len(warnings)}")


if __name__ == "__main__":
    main()
