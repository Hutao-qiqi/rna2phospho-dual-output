import pandas as pd

p = r"D:\lsy\01_data\single_cell\intermediate\paired_matrices\phospho_seq_blair_2025_phospho_multi\adt_counts.tsv"
df = pd.read_csv(p, sep="\t", nrows=1)
for c in df.columns:
    print(c)
