from pathlib import Path

import numpy as np
import pandas as pd


indir = Path(r"D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\phospho_perturb\lincs_p100_lvl4_v1")
rawdir = Path(r"D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\raw\lincs_p100_lvl4_v1")

meta = pd.read_csv(indir / "cell_metadata.tsv", sep="\t")
cond = pd.read_csv(indir / "condition_table.tsv", sep="\t")
targets = pd.read_csv(indir / "phospho_target_table.tsv", sep="\t")
manifest = pd.read_csv(indir / "pathway_target_manifest.tsv", sep="\t")
mask = np.load(indir / "target_mask.npy")
vals = np.load(indir / "phospho_values.npy")

print("P100_LOCAL")
print("raw_gct_files", len(list(rawdir.glob("*.gct"))))
print("matrix_shape", vals.shape)
print("cell_metadata_rows", len(meta))
print("condition_rows", len(cond))
print("targets", len(targets))
print("pathways", manifest["pathway"].nunique() if "pathway" in manifest.columns else "NA")
print("finite_fraction", float(mask.mean()))

for prefix, frame in [("meta", meta), ("cond", cond)]:
    for col in ["condition", "control_condition", "perturbation", "perturbation_type", "cell_line", "dose", "time", "plate"]:
        if col in frame.columns:
            print(f"{prefix}_unique_{col}", frame[col].nunique(dropna=True))

print("meta_columns", "|".join(meta.columns))
print("cond_columns", "|".join(cond.columns))

if "perturbation" in cond.columns:
    vc = cond["perturbation"].astype(str).value_counts()
    print("n_drugs_excluding_DMSO_like", int(sum(~vc.index.str.lower().isin(["dmso", "nan"]))))
    print("top_drugs_by_condition_count")
    print(vc.head(12).to_string())

for col in ["cell_line", "time", "dose", "perturbation_type"]:
    if col in cond.columns:
        print(f"cond_{col}_counts")
        print(cond[col].astype(str).value_counts().head(20).to_string())

print("target_table_head")
print(targets.head(5).to_string(index=False))

if "target_id" in targets.columns:
    genes = targets["target_id"].astype(str).str.split("_").str[0]
    print("targets_with_gene_like", int(targets["target_id"].astype(str).str.contains("_").sum()))
    print("unique_target_gene_prefix", int(genes.nunique()))

print("per_site_detected_fraction_summary")
site_frac = mask.mean(axis=0)
print(pd.Series(site_frac).describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())

print("per_sample_detected_count_summary")
sample_count = mask.sum(axis=1)
print(pd.Series(sample_count).describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())
