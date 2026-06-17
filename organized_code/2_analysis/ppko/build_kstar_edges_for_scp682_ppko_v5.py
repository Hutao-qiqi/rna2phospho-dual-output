import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_fasta(path):
    seq_by_acc = {}
    acc_by_gene = defaultdict(list)
    current_acc = None
    current_gene = None
    chunks = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_acc and chunks:
                    seq = "".join(chunks).upper()
                    seq_by_acc[current_acc] = seq
                    if current_gene:
                        acc_by_gene[current_gene].append(current_acc)
                parts = line.split("|")
                current_acc = parts[1].upper() if len(parts) > 1 else line[1:].split()[0].upper()
                match = re.search(r"\bGN=([A-Za-z0-9_.-]+)", line)
                current_gene = match.group(1).upper() if match else None
                chunks = []
            else:
                chunks.append(line)
    if current_acc and chunks:
        seq = "".join(chunks).upper()
        seq_by_acc[current_acc] = seq
        if current_gene:
            acc_by_gene[current_gene].append(current_acc)
    return seq_by_acc, acc_by_gene


def split_accessions(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    out = []
    for item in re.split(r"[;,| ]+", str(text)):
        item = item.strip().upper()
        if not item or item == "NAN":
            continue
        out.append(item.split("-", 1)[0])
    return list(dict.fromkeys(out))


def parse_modified_peptide(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return "", []
    s = str(text)
    residues = []
    phospho_offsets = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isalpha() and ch.upper() in set("ACDEFGHIKLMNPQRSTVWY"):
            residues.append(ch.upper())
            offset = len(residues)
            if s[i + 1 : i + 5].lower() == "(ph)":
                phospho_offsets.append(offset)
                i += 5
                continue
            if s[i + 1 : i + 6] == "[+80]":
                phospho_offsets.append(offset)
                i += 6
                continue
            i += 1
            continue
        i += 1
    return "".join(residues), phospho_offsets


def site_from_target_id(text):
    value = str(text).upper()
    matches = re.findall(r"([STY])(\d+)", value)
    return [f"{aa}{pos}" for aa, pos in matches]


def build_target_site_map(targets, seq_by_acc, acc_by_gene):
    rows = []
    key_to_targets = defaultdict(set)
    for _, row in targets.iterrows():
        target_index = int(row["target_index"])
        molecule = str(row.get("molecule", "")).upper()
        accessions = split_accessions(row.get("uniprot_id", ""))
        if molecule in acc_by_gene:
            accessions.extend(acc_by_gene[molecule])
        accessions = list(dict.fromkeys(a for a in accessions if a in seq_by_acc))

        direct_sites = []
        site = str(row.get("site", "")).upper()
        if re.fullmatch(r"[STY]\d+", site):
            direct_sites.append(site)
        direct_sites.extend(site_from_target_id(row.get("target_id", "")))
        for acc in accessions:
            for ksite in sorted(set(direct_sites)):
                key = f"{acc}|{ksite}"
                key_to_targets[key].add(target_index)
                rows.append(
                    {
                        "target_index": target_index,
                        "target_id": row.get("target_id", ""),
                        "molecule": row.get("molecule", ""),
                        "accession": acc,
                        "kstar_site": ksite,
                        "mapping_source": "target_site_or_id",
                    }
                )

        peptide, offsets = parse_modified_peptide(row.get("modified_peptide", row.get("raw_name", "")))
        if not peptide or not offsets:
            peptide, offsets = parse_modified_peptide(row.get("raw_name", ""))
        if peptide and offsets:
            for acc in accessions:
                seq = seq_by_acc.get(acc, "")
                start = seq.find(peptide)
                if start < 0:
                    continue
                for off in offsets:
                    pos = start + off
                    aa = peptide[off - 1]
                    if aa not in {"S", "T", "Y"}:
                        continue
                    ksite = f"{aa}{pos}"
                    key = f"{acc}|{ksite}"
                    key_to_targets[key].add(target_index)
                    rows.append(
                        {
                            "target_index": target_index,
                            "target_id": row.get("target_id", ""),
                            "molecule": row.get("molecule", ""),
                            "accession": acc,
                            "kstar_site": ksite,
                            "mapping_source": "modified_peptide_sequence",
                        }
                    )

    site_map = pd.DataFrame(rows).drop_duplicates()
    return site_map, key_to_targets


def network_files(network_root):
    root = Path(network_root)
    files = []
    for phospho_type in ("ST", "Y"):
        d = root / "NETWORKS" / phospho_type / "Default" / "INDIVIDUAL_NETWORKS"
        files.extend((phospho_type, p) for p in sorted(d.glob("*.tsv")))
    return files


def build_edges(network_root, key_to_targets):
    rows = []
    counts = defaultdict(int)
    n_networks = defaultdict(int)
    wanted = set(key_to_targets)
    for phospho_type, path in network_files(network_root):
        n_networks[phospho_type] += 1
        df = pd.read_csv(path, sep="\t", dtype=str)
        df["model_key"] = df["KSTAR_ACCESSION"].str.upper() + "|" + df["KSTAR_SITE"].str.upper()
        sub = df[df["model_key"].isin(wanted)]
        for _, edge in sub.iterrows():
            key = edge["model_key"]
            kinase = str(edge["KSTAR_KINASE"]).upper()
            for target_index in key_to_targets[key]:
                counts[(phospho_type, kinase, int(target_index), key)] += 1
    for (phospho_type, kinase, target_index, key), count in counts.items():
        acc, site = key.split("|", 1)
        denom = max(1, n_networks[phospho_type])
        rows.append(
            {
                "phospho_type": phospho_type,
                "kinase": kinase,
                "target_index": target_index,
                "kstar_accession": acc,
                "kstar_site": site,
                "network_count": count,
                "n_networks": denom,
                "edge_frequency": count / denom,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-table", required=True)
    parser.add_argument("--extracted-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-edge-frequency", type=float, default=0.02)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = Path(args.extracted_root)
    targets = pd.read_csv(args.target_table, sep="\t")
    seq_by_acc, acc_by_gene = parse_fasta(extracted / "RESOURCE_FILES" / "humanProteome.fasta")
    site_map, key_to_targets = build_target_site_map(targets, seq_by_acc, acc_by_gene)
    edge_table = build_edges(extracted, key_to_targets)
    if len(edge_table):
        edge_table = edge_table[edge_table["edge_frequency"] >= args.min_edge_frequency].copy()
        edge_table = edge_table.sort_values(["target_index", "kinase", "edge_frequency"], ascending=[True, True, False])
    site_map.to_csv(out_dir / "scp682_ppko_v5_target_kstar_site_map.tsv", sep="\t", index=False)
    edge_table.to_csv(out_dir / "scp682_ppko_v5_kstar_kinase_site_edges.tsv", sep="\t", index=False)
    summary = {
        "n_targets": int(len(targets)),
        "n_mapped_target_site_rows": int(len(site_map)),
        "n_mapped_targets": int(site_map["target_index"].nunique()) if len(site_map) else 0,
        "n_kstar_keys": int(len(key_to_targets)),
        "n_edges": int(len(edge_table)),
        "n_edge_targets": int(edge_table["target_index"].nunique()) if len(edge_table) else 0,
        "n_kinases": int(edge_table["kinase"].nunique()) if len(edge_table) else 0,
        "min_edge_frequency": float(args.min_edge_frequency),
    }
    (out_dir / "scp682_ppko_v5_kstar_edge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
