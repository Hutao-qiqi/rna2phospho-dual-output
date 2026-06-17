from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
BASE = ROOT / r"01_data\pathway_prior\intermediate\signed_phospho_regulatory_prior_v9"

site = pd.read_csv(BASE / "tables" / "signed_regulator_site_edges.tsv", sep="\t")
reg = pd.read_csv(BASE / "tables" / "signed_regulator_regulator_edges.tsv", sep="\t")

print("site_edges_head")
print(site.head(12).to_string(index=False))
print("\nsite_edges_negative_head")
print(site[site["sign"] < 0].head(20).to_string(index=False))
print("\nsite_edge_counts")
print(site.groupby(["edge_type", "sign"]).size().reset_index(name="n").to_string(index=False))
print("\nregulator_edge_counts")
print(reg["sign"].value_counts().to_string())
print("\nregulator_edges_negative_head")
print(reg[reg["sign"] < 0].head(20).to_string(index=False))
