from pathlib import Path
import pandas as pd

root = Path(r"D:\lsy")
tables = root / "02_results" / "single_cell" / "20260510_data_coverage" / "tables"
schema_tables = root / "02_results" / "single_cell" / "20260510_schema_unification" / "tables"

anchor = pd.read_csv(tables / "phospho_anchor_candidates.tsv", sep="\t")
direct = anchor[anchor["anchor_status"] == "direct_anchor"].copy()
direct = direct.rename(
    columns={
        "target_id": "target_id",
        "source_labels": "source_labels",
        "datasets": "datasets_confirmed",
        "residues": "residue",
        "paired_statuses": "paired_statuses",
    }
)
direct["shared_across_training_datasets"] = "yes"
direct["reason"] = "At least two direct paired datasets cover this canonical phosphosite/protein target after antibody synonym audit."
direct[
    [
        "target_id",
        "datasets_confirmed",
        "source_labels",
        "residue",
        "n_direct_paired_datasets",
        "shared_across_training_datasets",
        "reason",
    ]
].to_csv(schema_tables / "common_phospho_targets.tsv", sep="\t", index=False)
print(direct[["target_id", "datasets_confirmed", "source_labels", "residue", "n_direct_paired_datasets"]].to_string(index=False))
