# Build parent-protein query lists for SCP682-SC12 protein-residual training.
# The script is intentionally lightweight: it does not import torch or SC11.

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_TARGET_IDS = [
    "RPS6_pSitePending",
    "STAT3_Y705",
    "MAPK14_pSitePending",
    "AKT_pSitePending",
    "BTK_pSitePending",
    "CREB1_S133_ATF1_S63",
    "FOS_S32",
    "JUN_pSitePending",
    "MAPK1_MAPK3_pSitePending",
    "RELA_S536",
    "SYK_pSitePending",
    "BLNK_pSitePending",
    "PLCG2_Y759",
    "ZAP70_Y493",
    "LAT_Y226",
    "RB1_pSitePending",
]


ALIASES = {
    "P-P65": ["RELA"],
    "P-RB": ["RB1"],
    "P-BLNK": ["BLNK"],
    "P-CD79A": ["CD79A"],
    "P-HISTONH2A.X": ["H2AFX"],
    "P-HISTONE_H2A.X": ["H2AFX"],
    "P-HISTONH3": ["H3C1", "H3F3A", "H3F3B"],
    "P-PKC-B1": ["PRKCB"],
    "P-SRC": ["SRC", "LYN", "FYN", "LCK"],
    "P-TOR": ["MTOR"],
    "P-C-JUN": ["JUN"],
    "P-JNK": ["MAPK8", "MAPK9"],
    "P-IKKA/B": ["CHUK", "IKBKB"],
    "P-AMPK-A1/2": ["PRKAA1", "PRKAA2"],
    "P-AMPK-B1": ["PRKAB1"],
    "P-CDK1": ["CDK1"],
    "P-CDK4": ["CDK4"],
    "P-IRAK4": ["IRAK4"],
    "P-JAK1": ["JAK1"],
    "P-STAT5": ["STAT5A", "STAT5B"],
    "P-STAT6": ["STAT6"],
    "P-C-JUN": ["JUN"],
    "P44/42": ["MAPK1", "MAPK3"],
    "PHOSPHO_P44_42": ["MAPK1", "MAPK3"],
    "HM0025": ["MAPK1", "MAPK3"],
}


DROP_TOKENS = {
    "",
    "NAN",
    "NA",
    "NULL",
    "NONE",
    "UNKNOWN",
    "PENDING",
    "PENDING_ANTIBODY_CLONE",
}


def parse_list(text):
    if text is None:
        return []
    items = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if token:
            items.append(token)
    return items


def normalize_symbol(value):
    text = str(value).strip().upper()
    text = text.replace("PHOSPHO-", "").replace("PHOSPHO_", "")
    text = text.replace("ANTI-", "").replace("ANTI_", "")
    if text.startswith("P-") and len(text) > 2:
        text = text[2:]
    text = text.strip("-_ ")
    if text in DROP_TOKENS:
        return ""
    if text.isdigit() or len(text) <= 1:
        return ""
    synonym = {
        "C-JUN": "JUN",
        "ERK1": "MAPK3",
        "ERK2": "MAPK1",
    }
    text = synonym.get(text, text)
    if text in {
        "P65",
        "IKKA",
        "IKKB",
        "AMPK-A1",
        "AMPK-A2",
        "AMPK-B1",
        "HISTONH2A.X",
        "HISTONH3",
        "JNK",
        "TOR",
        "PKC-B1",
        "STAT5",
    }:
        return ""
    if text.startswith(("INTRA", "HM", "BHM")) or "IGG" in text:
        return ""
    if len(text) > 15:
        return ""
    if "." in text:
        return ""
    if len(text) >= 2 and text[0] in {"S", "T", "Y"} and text[1:].isdigit():
        return ""
    return text


def split_symbols(value):
    symbols = []
    text = str(value)
    for sep in ["/", "|", ","]:
        text = text.replace(sep, ";")
    for token in text.split(";"):
        sym = normalize_symbol(token)
        if sym and sym.replace("-", "").replace(".", "").isalnum():
            symbols.append(sym)
    return symbols


def symbols_for_target(row):
    symbols = []
    for col in ["protein_symbol", "gene", "gene_symbol"]:
        if col in row:
            symbols.extend(split_symbols(row.get(col, "")))

    joined = " ".join(
        str(row.get(col, "")).upper()
        for col in ["target_id", "canonical_label", "feature_id", "protein_symbol"]
    )
    for key, vals in ALIASES.items():
        if key in joined:
            symbols.extend(vals)
    if not symbols:
        for col in ["target_id", "canonical_label"]:
            if col in row:
                symbols.extend(split_symbols(row.get(col, "")))

    clean = []
    for sym in symbols:
        sym = normalize_symbol(sym)
        if not sym:
            continue
        if sym.startswith("P-"):
            continue
        if sym in DROP_TOKENS:
            continue
        clean.append(sym)
    return sorted(set(clean))


def mapping_level(row, n_symbols):
    joined = " ".join(
        str(row.get(col, "")).upper()
        for col in ["target_id", "canonical_label", "feature_id", "residue"]
    )
    if n_symbols <= 0:
        return "unmapped_manual_review"
    if "PSITEPENDING" in joined or "PENDING_ANTIBODY_CLONE" in joined:
        return "parent_only_site_pending"
    if any(token in joined for token in ["_S", "_T", "_Y", "|S", "|T", "|Y"]):
        return "parent_only_site_named"
    return "parent_only"


def load_target_table(args):
    if args.target_table:
        path = Path(args.target_table)
    else:
        path = Path(args.input_dir) / "phospho_target_table.tsv"
    if not path.exists():
        raise FileNotFoundError(f"target table not found: {path}")
    return path, pd.read_csv(path, sep="\t")


def choose_targets(target_table, target_ids):
    tt = target_table.copy()
    if "target_id" not in tt.columns:
        raise ValueError("target table must contain target_id")
    if target_ids == ["include_in_loss"]:
        if "include_in_loss" not in tt.columns:
            raise ValueError("target table has no include_in_loss column")
        tt = tt[tt["include_in_loss"].astype(str).str.lower().eq("true")].copy()
    elif target_ids:
        wanted = set(target_ids)
        tt = tt[tt["target_id"].astype(str).isin(wanted)].copy()
        missing = sorted(wanted - set(tt["target_id"].astype(str)))
        if missing:
            raise ValueError("missing target_id in target table: " + ", ".join(missing))
    if "target_index" in tt.columns:
        rows = []
        seen = set()
        for _, row in tt.sort_values(["target_index"]).iterrows():
            idx = int(row["target_index"])
            if idx in seen:
                continue
            rows.append(row)
            seen.add(idx)
        tt = pd.DataFrame(rows)
    return tt.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=r"D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_masked_multisite_v1")
    parser.add_argument("--target-table", default="")
    parser.add_argument("--target-ids", default="")
    parser.add_argument("--use-default-targets", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_table_path, target_table = load_target_table(args)
    target_ids = parse_list(args.target_ids)
    if args.use_default_targets and not target_ids:
        target_ids = DEFAULT_TARGET_IDS
    targets = choose_targets(target_table, target_ids)

    mapping_rows = []
    protein_to_targets = {}
    for order, row in targets.iterrows():
        row_dict = row.to_dict()
        symbols = symbols_for_target(row_dict)
        for sym in symbols:
            protein_to_targets.setdefault(sym, []).append(str(row_dict.get("target_id", "")))
        mapping_rows.append(
            {
                "target_order": int(order),
                "target_id": row_dict.get("target_id", ""),
                "target_index": row_dict.get("target_index", order),
                "feature_id": row_dict.get("feature_id", ""),
                "protein_symbol": row_dict.get("protein_symbol", ""),
                "residue": row_dict.get("residue", ""),
                "canonical_label": row_dict.get("canonical_label", ""),
                "candidate_parent_proteins": ";".join(symbols),
                "primary_parent_gene": symbols[0] if symbols else "",
                "parent_candidate_count": int(len(symbols)),
                "mapping_level": mapping_level(row_dict, len(symbols)),
                "manual_review_required": bool(len(symbols) == 0),
            }
        )

    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(out_dir / "sc12_target_parent_protein_mapping.tsv", sep="\t", index=False)

    protein_rows = []
    for i, protein in enumerate(sorted(protein_to_targets)):
        protein_rows.append(
            {
                "protein_order": i,
                "protein_symbol": protein,
                "n_targets_using_parent": len(set(protein_to_targets[protein])),
                "target_ids": ";".join(sorted(set(protein_to_targets[protein]))),
            }
        )
    protein_table = pd.DataFrame(protein_rows)
    protein_table.to_csv(out_dir / "sc12_required_parent_proteins.tsv", sep="\t", index=False)

    with (out_dir / "scTranslator_query_proteins.txt").open("w", encoding="utf-8") as fh:
        for protein in protein_table["protein_symbol"].tolist():
            fh.write(str(protein) + "\n")
    protein_table.rename(columns={"protein_symbol": "gene"}).to_csv(
        out_dir / "scTranslator_query_proteins.tsv", sep="\t", index=False
    )

    with (out_dir / "scProTrans_query_proteins.txt").open("w", encoding="utf-8") as fh:
        for protein in protein_table["protein_symbol"].tolist():
            fh.write(str(protein) + "\n")
    protein_table.rename(columns={"protein_symbol": "protein"}).to_csv(
        out_dir / "scProTrans_query_proteins.tsv", sep="\t", index=False
    )

    missing = mapping[mapping["parent_candidate_count"].le(0)].copy()
    if len(missing):
        missing.to_csv(out_dir / "sc12_targets_without_parent_protein.tsv", sep="\t", index=False)

    manifest = {
        "target_table": str(target_table_path),
        "n_targets": int(len(mapping)),
        "n_parent_proteins": int(len(protein_table)),
        "n_targets_without_parent_protein": int(len(missing)),
        "outputs": [
            "sc12_target_parent_protein_mapping.tsv",
            "sc12_required_parent_proteins.tsv",
            "scTranslator_query_proteins.txt",
            "scTranslator_query_proteins.tsv",
            "scProTrans_query_proteins.txt",
            "scProTrans_query_proteins.tsv",
        ],
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
