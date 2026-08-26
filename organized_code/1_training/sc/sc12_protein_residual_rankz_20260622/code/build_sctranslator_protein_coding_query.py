from __future__ import annotations

import json
import pickle
import urllib.request
from pathlib import Path

import pandas as pd


REPO = Path(r"D:\data\lsy\models\scTranslator")
OUT = Path(r"D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\protein_prediction_cache\scp682_sc12_all_predictable_proteins_v1")
HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"


def myid_for_symbol(symbol: str, hgs_to_entrez: dict, entrez_to_myid: dict) -> int | None:
    entrez = hgs_to_entrez.get(symbol.upper())
    if entrez is None:
        return None
    vals = list(entrez) if isinstance(entrez, (list, tuple, set)) else [entrez]
    for val in vals:
        for key in [val, str(val), str(val).split(".")[0]]:
            if key in entrez_to_myid:
                return int(entrez_to_myid[key])
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hgnc_path = OUT / "hgnc_complete_set.txt"
    if not hgnc_path.exists():
        urllib.request.urlretrieve(HGNC_URL, hgnc_path)

    id_dir = REPO / r"code\model\ID_dic"
    with (id_dir / "hgs_to_EntrezID.pkl").open("rb") as fh:
        hgs_to_entrez = pickle.load(fh)
    with (id_dir / "EntrezID_to_myID.pkl").open("rb") as fh:
        entrez_to_myid = pickle.load(fh)

    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str).fillna("")
    protein_coding = hgnc[
        (hgnc["status"].str.lower() == "approved")
        & (hgnc["locus_group"].str.lower() == "protein-coding gene")
    ].copy()

    rows = []
    seen_myid = set()
    for _, row in protein_coding.iterrows():
        symbol = row["symbol"].strip().upper()
        my_id = myid_for_symbol(symbol, hgs_to_entrez, entrez_to_myid)
        if my_id is None:
            continue
        if my_id in seen_myid:
            continue
        seen_myid.add(my_id)
        rows.append(
            {
                "protein_symbol": symbol,
                "my_id": my_id,
                "hgnc_id": row.get("hgnc_id", ""),
                "name": row.get("name", ""),
                "entrez_id": row.get("entrez_id", ""),
                "ensembl_gene_id": row.get("ensembl_gene_id", ""),
            }
        )
    rows = sorted(rows, key=lambda r: r["protein_symbol"])

    txt = OUT / "scTranslator_hgnc_protein_coding_query.txt"
    tsv = OUT / "scTranslator_hgnc_protein_coding_query.tsv"
    pd.DataFrame(rows).to_csv(tsv, sep="\t", index=False)
    txt.write_text("\n".join([r["protein_symbol"] for r in rows]) + "\n", encoding="utf-8")

    manifest = {
        "source": "HGNC approved protein-coding genes intersected with scTranslator decoder ID dictionary",
        "hgnc_url": HGNC_URL,
        "hgnc_file": str(hgnc_path),
        "n_hgnc_approved_protein_coding": int(protein_coding.shape[0]),
        "n_sctranslator_predictable_protein_coding": len(rows),
        "query_file": str(txt),
        "table_file": str(tsv),
    }
    (OUT / "protein_coding_query_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
