from pathlib import Path
import pandas as pd

root = Path(r"E:\data\gongke\TCGA-TCPA")
xlsx = root / "_downloads" / "blair_supp" / "supplementary_dataset_1.xlsx"
out = root / "_downloads" / "coverage_audit"
out.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(xlsx, sheet_name=0, header=None)
blocks = [
    ("pilot", 0),
    ("benchmark", 6),
    ("retinal_organoid", 12),
    ("retinal_organoid_multi", 18),
    ("cerebral_organoid", 24),
]

rows = []
for exp, start in blocks:
    for i in range(1, df.shape[0]):
        antibody = df.iat[i, start] if start < df.shape[1] else None
        site = df.iat[i, start + 1] if start + 1 < df.shape[1] else None
        vendor = df.iat[i, start + 2] if start + 2 < df.shape[1] else None
        catalog = df.iat[i, start + 3] if start + 3 < df.shape[1] else None
        tsb = df.iat[i, start + 4] if start + 4 < df.shape[1] else None
        if pd.isna(antibody):
            continue
        antibody_s = str(antibody).strip()
        site_s = "" if pd.isna(site) else str(site).strip()
        if antibody_s.lower() == "antibody":
            continue
        rows.append(
            {
                "dataset_id": "phospho_seq_blair_2025",
                "experiment": exp,
                "feature_label": antibody_s,
                "phospho_site": site_s,
                "vendor": "" if pd.isna(vendor) else str(vendor).strip(),
                "catalog": "" if pd.isna(catalog) else str(catalog).strip(),
                "tsb_index": "" if pd.isna(tsb) else str(tsb).strip(),
            }
        )

blair = pd.DataFrame(rows)
site_nonempty = blair["phospho_site"].fillna("").str.strip().ne("")
label_phospho = blair["feature_label"].str.contains(r"^p[A-Z0-9-]|phospho", case=True, regex=True)
blair["is_phospho"] = label_phospho | site_nonempty
blair.to_csv(out / "blair_supplementary_antibody_table.tsv", sep="\t", index=False)

phospho = blair[blair["is_phospho"]].copy()
phospho.to_csv(out / "blair_phospho_antibody_table.tsv", sep="\t", index=False)

print(phospho.to_string(index=False))
