import gzip
import csv
from pathlib import Path

base = Path(r"D:\lsy\01_data\single_cell\raw\phospho_seq_blair_2025\extracted")
for path in sorted(base.glob("*.csv.gz")):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = []
        for _, row in zip(range(5), reader):
            rows.append(row[:8])
    print(path.name)
    print("n_columns", len(header))
    print("header_first", header[:12])
    print("rows_first", rows[:2])
