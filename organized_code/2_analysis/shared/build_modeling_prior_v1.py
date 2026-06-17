#!/usr/bin/env python3
import csv
import hashlib
import math
from collections import Counter
from datetime import date
from pathlib import Path


ROOT = Path("/data/lsy/Infinite_Stream")
BASE = ROOT / "01_data/pathway_prior/intermediate/open_kinase_substrate"
META = ROOT / "01_data/pathway_prior/metadata"
OUTDIR = ROOT / "01_data/pathway_prior/processed"
RESULT = ROOT / "02_results/pathway_prior/20260425_modeling_prior_v1"
CODE = ROOT / "03_code/pathway_prior"

PN = BASE / "phosphonetworks_kinase_substrate_long.tsv"
OP = BASE / "omnipath_enzsub_no_phosphosite.tsv"
KINHUB = META / "kinhub_human_kinases.tsv"
OUT = OUTDIR / "kinase_substrate_prior_for_modeling_v1.tsv"
SUMMARY = OUTDIR / "kinase_substrate_prior_for_modeling_v1_summary.tsv"
DATACARD = OUTDIR / "kinase_substrate_prior_for_modeling_v1_data_card.md"
RESULT_TABLE = RESULT / "tables" / "kinase_substrate_prior_for_modeling_v1_summary.tsv"

RTK_CORE = {"EGFR", "ERBB2", "MET", "PDGFRA"}
BROAD_KINASES = {"GSK3B", "MAPK8", "MAPK1", "PRKCA", "MAPK14", "CDK1", "PRKACA", "CSNK2A1"}


def read_rows(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def site_key(row):
    site = (row.get("substrate_site") or "").strip()
    if site:
        return site
    residue_type = (row.get("residue_type") or "").strip()
    residue_offset = (row.get("residue_offset") or "").strip()
    if residue_type and residue_offset:
        return f"{residue_type}{residue_offset}"
    return ""


def source_label(row):
    sr = (row.get("source_record") or "").lower()
    src = (row.get("sources") or "").lower()
    joined = sr + ";" + src
    labels = set()
    if "phosphonetworks" in joined:
        labels.add("phosphonetworks")
    if "omnipath" in joined:
        labels.add("omnipath")
    return labels


def normalize_rows(rows, kinases):
    for row in rows:
        if (row.get("modification") or "").strip().lower() != "phosphorylation":
            continue
        kinase = (row.get("kinase_gene") or "").strip()
        substrate = (row.get("substrate_gene") or "").strip()
        if not kinase or not substrate:
            continue
        if kinase not in kinases:
            continue
        site = site_key(row)
        labels = source_label(row)
        if not labels:
            continue
        yield {
            "kinase_gene": kinase,
            "substrate_gene": substrate,
            "substrate_site": site,
            "residue_type": (row.get("residue_type") or (site[:1] if site else "")).strip(),
            "residue_offset": (row.get("residue_offset") or (site[1:] if len(site) > 1 else "")).strip(),
            "has_site": bool(site),
            "source_labels": labels,
            "source_records": {(row.get("source_record") or "").strip()},
            "original_sources": {(row.get("sources") or "").strip()},
            "references": {(row.get("references") or "").strip()},
            "raw_scores": {(row.get("score") or "").strip()},
        }


def merge_records(records):
    merged = {}
    for rec in records:
        if rec["has_site"]:
            key = ("site", rec["kinase_gene"], rec["substrate_gene"], rec["substrate_site"])
        else:
            key = ("gene", rec["kinase_gene"], rec["substrate_gene"], "")
        if key not in merged:
            merged[key] = rec
        else:
            dst = merged[key]
            dst["source_labels"].update(rec["source_labels"])
            dst["source_records"].update(rec["source_records"])
            dst["original_sources"].update(rec["original_sources"])
            dst["references"].update(rec["references"])
            dst["raw_scores"].update(rec["raw_scores"])
            if not dst["residue_type"] and rec["residue_type"]:
                dst["residue_type"] = rec["residue_type"]
            if not dst["residue_offset"] and rec["residue_offset"]:
                dst["residue_offset"] = rec["residue_offset"]
    return merged


def clean_join(values):
    vals = sorted({v for v in values if v})
    return ";".join(vals)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (RESULT / "tables").mkdir(parents=True, exist_ok=True)
    (RESULT / "logs").mkdir(parents=True, exist_ok=True)

    kin_rows = read_rows(KINHUB)
    kinases = {r["HGNC"].strip() for r in kin_rows if r.get("HGNC", "").strip()}
    pn_rows = list(normalize_rows(read_rows(PN), kinases))
    op_rows = list(normalize_rows(read_rows(OP), kinases))
    merged = merge_records(pn_rows + op_rows)

    out_degree = Counter()
    for rec in merged.values():
        out_degree[rec["kinase_gene"]] += 1

    rows = []
    for rec in merged.values():
        labels = rec["source_labels"]
        if labels == {"phosphonetworks", "omnipath"}:
            source = "both"
        elif "phosphonetworks" in labels:
            source = "phosphonetworks"
        else:
            source = "omnipath"
        edge_level = "high_confidence" if rec["has_site"] else "gene_level_only"
        weight = 1.0 if rec["has_site"] else 0.3
        degree = out_degree[rec["kinase_gene"]]
        rows.append({
            "prior_version": "v1",
            "kinase_gene": rec["kinase_gene"],
            "substrate_gene": rec["substrate_gene"],
            "substrate_site": rec["substrate_site"],
            "residue_type": rec["residue_type"],
            "residue_offset": rec["residue_offset"],
            "modification": "phosphorylation",
            "edge_level": edge_level,
            "source": source,
            "source_records": clean_join(rec["source_records"]),
            "original_sources": clean_join(rec["original_sources"]),
            "references": clean_join(rec["references"]),
            "has_site": "True" if rec["has_site"] else "False",
            "weight": f"{weight:.3f}",
            "kinase_out_degree": degree,
            "attention_degree_scale": f"{1 / math.sqrt(degree):.8f}",
            "broad_kinase_flag": "True" if rec["kinase_gene"] in BROAD_KINASES else "False",
            "rtk_core_flag": "True" if rec["kinase_gene"] in RTK_CORE else "False",
        })
    rows.sort(key=lambda r: (r["kinase_gene"], r["substrate_gene"], r["has_site"] != "True", r["substrate_site"]))

    fields = [
        "prior_version",
        "kinase_gene",
        "substrate_gene",
        "substrate_site",
        "residue_type",
        "residue_offset",
        "modification",
        "edge_level",
        "source",
        "has_site",
        "weight",
        "kinase_out_degree",
        "attention_degree_scale",
        "broad_kinase_flag",
        "rtk_core_flag",
        "source_records",
        "original_sources",
        "references",
    ]
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    by_source = Counter(r["source"] for r in rows)
    by_level = Counter(r["edge_level"] for r in rows)
    rtk_rows = []
    for kinase in ["EGFR", "ERBB2", "MET", "PDGFRA"]:
        kr = [r for r in rows if r["kinase_gene"] == kinase]
        rtk_rows.append({
            "kinase": kinase,
            "edges": len(kr),
            "substrate_genes": len({r["substrate_gene"] for r in kr}),
            "site_edges": sum(1 for r in kr if r["has_site"] == "True"),
            "gene_level_only_edges": sum(1 for r in kr if r["has_site"] == "False"),
        })

    summary_rows = [
        ["metric", "value"],
        ["prior_version", "v1"],
        ["build_date", str(date.today())],
        ["output", str(OUT)],
        ["sha256", sha256(OUT)],
        ["total_edges", str(total)],
        ["high_confidence_site_edges", str(by_level["high_confidence"])],
        ["gene_level_only_edges", str(by_level["gene_level_only"])],
        ["source_phosphonetworks", str(by_source["phosphonetworks"])],
        ["source_omnipath", str(by_source["omnipath"])],
        ["source_both", str(by_source["both"])],
        ["kinases", str(len({r["kinase_gene"] for r in rows}))],
        ["substrate_genes", str(len({r["substrate_gene"] for r in rows}))],
        ["substrate_sites_with_position", str(len({r["substrate_gene"] + ':' + r["substrate_site"] for r in rows if r["has_site"] == "True"}))],
        ["kinhub_whitelist_kinases", str(len(kinases))],
        ["degree_scale_formula", "1 / sqrt(kinase_out_degree)"],
        ["gene_level_only_weight", "0.3"],
    ]
    with SUMMARY.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerows(summary_rows)
    RESULT_TABLE.write_text(SUMMARY.read_text())

    rtk_path = RESULT / "tables" / "rtk_core_coverage_in_modeling_prior_v1.tsv"
    with rtk_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rtk_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rtk_rows)

    card = f"""# kinase_substrate_prior_for_modeling_v1 数据卡

版本：v1
日期：{date.today()}
固定表：`{OUT}`
校验值：`{sha256(OUT)}`

## 来源

- PhosphoNetworks：`rawKSI.csv`、`refKSI.csv`、`comKSI.csv`、`highResolutionNetwork.csv`
- OmniPath：`enzsub` 接口，下载参数为 `format=tsv&genesymbols=yes&fields=sources,references`
- 人类激酶白名单：KinHub 人类激酶表，517 个 HGNC 符号

## 过滤规则

- 只保留 `modification = phosphorylation`。
- `kinase_gene` 必须在 KinHub 人类激酶白名单内。
- 位点级边按 `kinase_gene + substrate_gene + substrate_site` 合并。
- 无位点基因级边按 `kinase_gene + substrate_gene` 合并，不和位点级边互相覆盖。
- `source` 取值为 `phosphonetworks`、`omnipath`、`both`。
- `has_site = True` 的边标记为 `high_confidence`，`weight = 1.0`。
- `has_site = False` 的边标记为 `gene_level_only`，`weight = 0.3`。
- 每条边写入 `kinase_out_degree` 和 `attention_degree_scale = 1 / sqrt(kinase_out_degree)`，用于图注意力归一化。

## 规模

- 总边数：{total}
- 位点级高置信边：{by_level['high_confidence']}
- 基因级低权重边：{by_level['gene_level_only']}
- 覆盖激酶数：{len({r['kinase_gene'] for r in rows})}
- 覆盖底物基因数：{len({r['substrate_gene'] for r in rows})}
- 覆盖带位置底物磷酸化位点数：{len({r['substrate_gene'] + ':' + r['substrate_site'] for r in rows if r['has_site'] == 'True'})}

## 使用限制

MET 和 PDGFRA 在开放先验中的带位置底物位点支撑偏弱，后续亚型分析必须保留有先验和无先验两版消融结果。宽连接激酶 GSK3B、MAPK8、MAPK1、PRKCA、MAPK14、CDK1、PRKACA、CSNK2A1 不删除，但训练时必须使用 `attention_degree_scale` 做 softmax 前缩放。
"""
    DATACARD.write_text(card)
    (RESULT / "tables" / DATACARD.name).write_text(card)

    print(f"output={OUT}")
    print(f"edges={total}")
    print(f"site_edges={by_level['high_confidence']}")
    print(f"gene_level_only={by_level['gene_level_only']}")
    print(f"sha256={sha256(OUT)}")


if __name__ == "__main__":
    main()
