from __future__ import annotations

import json
import pickle
import re
from pathlib import Path


REPO = Path(r"D:\data\lsy\models\scTranslator")
OUT = Path(r"D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\protein_prediction_cache\scp682_sc12_all_predictable_proteins_v1")


def symbol_score(symbol: str) -> tuple[int, int, str]:
    s = symbol.upper()
    # Prefer common official-looking protein coding symbols over aliases and loci.
    bad_prefix = ("LOC", "LINC", "MIR", "SNOR", "SNORD", "SNORA", "RNU", "RNA", "MT-", "HLA-")
    bad = int(s.startswith(bad_prefix))
    has_dash = int("-" in s)
    weird = int(re.search(r"[^A-Z0-9-]", s) is not None)
    length_penalty = abs(len(s) - 5)
    return (bad, weird, has_dash, length_penalty, s)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    id_dir = REPO / r"code\model\ID_dic"
    with (id_dir / "hgs_to_EntrezID.pkl").open("rb") as fh:
        hgs_to_entrez = pickle.load(fh)
    with (id_dir / "EntrezID_to_myID.pkl").open("rb") as fh:
        entrez_to_myid = pickle.load(fh)

    by_myid: dict[int, set[str]] = {}
    for gene, entrez in hgs_to_entrez.items():
        symbol = str(gene).strip().upper()
        if not symbol:
            continue
        vals = list(entrez) if isinstance(entrez, (list, tuple, set)) else [entrez]
        for val in vals:
            my_id = None
            for key in [val, str(val), str(val).split(".")[0]]:
                if key in entrez_to_myid:
                    my_id = int(entrez_to_myid[key])
                    break
            if my_id is not None:
                by_myid.setdefault(my_id, set()).add(symbol)

    rows = []
    for my_id, symbols in by_myid.items():
        preferred = sorted(symbols, key=symbol_score)[0]
        rows.append((preferred, my_id, len(symbols), ";".join(sorted(symbols)[:20])))
    rows.sort(key=lambda x: x[0])

    # This is the broadest scTranslator-decodable gene/protein vocabulary after alias collapse.
    txt = OUT / "scTranslator_all_predictable_unique_gene_query.txt"
    tsv = OUT / "scTranslator_all_predictable_unique_gene_query.tsv"
    with txt.open("w", encoding="utf-8") as fh:
        for symbol, _, _, _ in rows:
            fh.write(symbol + "\n")
    with tsv.open("w", encoding="utf-8") as fh:
        fh.write("protein_symbol\tmy_id\tn_aliases\talias_examples\n")
        for symbol, my_id, n_aliases, alias_examples in rows:
            fh.write(f"{symbol}\t{my_id}\t{n_aliases}\t{alias_examples}\n")
    (OUT / "query_manifest.json").write_text(
        json.dumps(
            {
                "source": "scTranslator hgs_to_EntrezID and EntrezID_to_myID dictionaries",
                "n_raw_symbols": len(hgs_to_entrez),
                "n_unique_my_id": len(rows),
                "query_file": str(txt),
                "table_file": str(tsv),
                "note": "Alias-collapsed broad scTranslator-decodable vocabulary; each row has one unique decoder my_id.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"n_unique_my_id": len(rows), "txt": str(txt), "tsv": str(tsv)}, indent=2))


if __name__ == "__main__":
    main()
