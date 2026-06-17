import pandas as pd
import re

blair = r"D:\lsy\01_data\single_cell\intermediate\paired_matrices\phospho_seq_blair_2025_phospho_multi\adt_counts.tsv"
cols = pd.read_csv(blair, sep="\t", nrows=0).columns[1:]
print("Blair phospho-like by simple pattern:")
for c in cols:
    if re.search(r"^p|phos", c, re.I):
        print(c)

print("\nRNA gene overlap:")
print(pd.read_csv(r"D:\lsy\02_results\single_cell\20260510_data_coverage\tables\rna_gene_symbol_overlap.tsv", sep="\t").to_string(index=False))

print("\nPhospho target coverage:")
print(pd.read_csv(r"D:\lsy\02_results\single_cell\20260510_data_coverage\tables\phospho_target_dataset_coverage.tsv", sep="\t").to_string(index=False))
