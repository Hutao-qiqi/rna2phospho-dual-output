from pathlib import Path
import pandas as pd

meta = Path(r"D:\lsy\01_data\shared\metadata")
result = Path(r"D:\lsy\02_results\single_cell\20260510_schema_unification\tables")

phospho_path = meta / "phospho_target_schema.tsv"
common_path = result / "common_phospho_targets.tsv"

phospho = pd.read_csv(phospho_path, sep="\t")
row = {
    "source_label": "pRPS6",
    "target_id": "RPS6_pSitePending",
    "protein_symbol": "RPS6",
    "hgnc_symbol": "RPS6",
    "uniprot_accession": "P62753",
    "residue": "pending_antibody_clone",
    "canonical_label": "RPS6_P62753_pSitePending",
    "datasets_confirmed": "phospho_seq_blair_2025",
}
phospho = phospho[phospho["target_id"] != row["target_id"]]
phospho = pd.concat([phospho, pd.DataFrame([row])], ignore_index=True)
phospho.to_csv(phospho_path, sep="\t", index=False)

common = pd.read_csv(common_path, sep="\t")
common = common[common["target_id"] != row["target_id"]]
common = pd.concat(
    [
        common,
        pd.DataFrame(
            [
                {
                    "target_id": row["target_id"],
                    "canonical_label": row["canonical_label"],
                    "datasets_confirmed": row["datasets_confirmed"],
                    "shared_across_training_datasets": "no",
                    "reason": "Blair Phospho-Multi has pRPS6 ADT; exact residue needs antibody annotation before cross-dataset matching.",
                }
            ]
        ),
    ],
    ignore_index=True,
)
common.to_csv(common_path, sep="\t", index=False)
print("patched pRPS6 into phospho schema")
