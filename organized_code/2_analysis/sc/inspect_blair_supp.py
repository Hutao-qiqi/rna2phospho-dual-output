from pathlib import Path
import pandas as pd

xlsx = Path(r"E:\data\gongke\TCGA-TCPA\_downloads\blair_supp\supplementary_dataset_1.xlsx")
xl = pd.ExcelFile(xlsx)
print("sheets", xl.sheet_names)
for sheet in xl.sheet_names:
    df = pd.read_excel(xlsx, sheet_name=sheet)
    print("\nSHEET", sheet, df.shape)
    print(df.head(5).to_string())
    text = df.astype(str).agg(" ".join, axis=1)
    hits = df[text.str.contains("RPS6|pRPS6|phospho|pMAPK|STAT3|ERK|MAPK|S6", case=False, na=False)]
    if not hits.empty:
        print("HITS")
        print(hits.to_string(index=False))
