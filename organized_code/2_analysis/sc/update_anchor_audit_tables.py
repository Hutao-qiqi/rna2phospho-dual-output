from pathlib import Path
import pandas as pd
from scipy.io import mmread

root = Path(r"D:\lsy")
meta = root / "01_data" / "shared" / "metadata"
paired = root / "01_data" / "single_cell" / "intermediate" / "paired_matrices"
out = root / "02_results" / "single_cell" / "20260510_data_coverage"
for sub in ["tables", "reports"]:
    (out / sub).mkdir(parents=True, exist_ok=True)

blair_supp = meta / "blair_phospho_antibody_table.tsv"
blair = pd.read_csv(blair_supp, sep="\t")

blair_map = {
    "pRPS6": ("RPS6_S235_S236", "RPS6", "P62753", "S235/S236"),
    "pSTAT3-705": ("STAT3_Y705", "STAT3", "P40763", "Y705"),
    "pSTAT3-727": ("STAT3_S727", "STAT3", "P40763", "S727"),
    "pFOXO1": ("FOXO1_S319", "FOXO1", "Q12778", "S319"),
    "pMEK": ("MAP2K1_S298", "MAP2K1", "Q02750", "S298"),
    "pSTAT1": ("STAT1_Y701", "STAT1", "P42224", "Y701"),
    "pMAPK": ("MAPK14_T180_Y182", "MAPK14", "Q16539", "T180/Y182"),
    "pERK1/2": ("MAPK1_MAPK3_T202_Y204", "MAPK1;MAPK3", "P28482;P27361", "T202/Y204;T185/Y187"),
    "pJAK-1": ("JAK1_Y1022_Y1023", "JAK1", "P23458", "Y1022/Y1023"),
    "pPLCy": ("PLCG1_Y783_or_PLCG2_Y759_pending", "PLCG1;PLCG2", "P19174;P16885", "pending_label_says_Y759"),
    "pPKC-b1": ("PRKCB_T642", "PRKCB", "P05771", "T642"),
    "pSMAD1/5": ("SMAD1_SMAD5_S463_S465", "SMAD1;SMAD5", "Q15797;Q99717", "S463/S465"),
    "pBCatenin": ("CTNNB1_S33_S37", "CTNNB1", "P35222", "S33/S37"),
    "p-AMPK-a1/2": ("PRKAA1_PRKAA2_T172", "PRKAA1;PRKAA2", "Q13131;P54646", "T172"),
    "pTSC2-1837": ("TSC2_S1387", "TSC2", "P49815", "S1387"),
    "p38-MAPK": ("MAPK14_total", "MAPK14", "Q16539", "none"),
    "p65-nFKb": ("RELA_total", "RELA", "Q04206", "none"),
    "pSMAD2": ("SMAD2_S250", "SMAD2", "Q15796", "S250"),
    "pPKCa": ("PRKCA_T638", "PRKCA", "P17252", "T638"),
    "pJNK1": ("MAPK8_T183_Y185_or_T183_Y221_pending", "MAPK8", "P45983", "pending_label_says_T183/221"),
    "pGSK3B": ("GSK3B_S9", "GSK3B", "P49841", "S9"),
    "pPI3K": ("PI3K_pSitePending", "PIK3_family", "pending", "pending"),
    "pSTAT3": ("STAT3_S727", "STAT3", "P40763", "S727"),
    "pVim": ("VIM_S39", "VIM", "P08670", "S39"),
}

blair_rows = []
for _, r in blair.iterrows():
    tid, symbol, uniprot, residue = blair_map.get(r["feature_label"], (r["feature_label"], "pending", "pending", str(r["phospho_site"])))
    paired_status = "direct_RNA_ADT" if r["experiment"] == "retinal_organoid_multi" else "not_direct_RNA_ADT"
    if r["experiment"] == "cerebral_organoid":
        paired_status = "ADT_ATAC_plus_bridge_or_imputed_RNA"
    blair_rows.append(
        {
            "dataset_id": "phospho_seq_blair_2025",
            "experiment": r["experiment"],
            "source_label": r["feature_label"],
            "target_id": tid,
            "protein_symbol": symbol,
            "uniprot_accession": uniprot,
            "residue": residue,
            "vendor": r["vendor"],
            "catalog": r["catalog"],
            "paired_status": paired_status,
        }
    )

vivo_rows = [
    ("phos-STAT3", "STAT3_Y705", "STAT3", "P40763", "Y705", "unknown", "unknown"),
    ("phos-p65", "RELA_S536", "RELA", "Q04206", "S536", "unknown", "unknown"),
    ("phos-FOS", "FOS_S32", "FOS", "P01100", "S32", "unknown", "unknown"),
    ("phos-ERK1/2", "MAPK1_MAPK3_T202_Y204", "MAPK1;MAPK3", "P28482;P27361", "T202/Y204;T185/Y187", "unknown", "unknown"),
]
vivo_rows = [
    {
        "dataset_id": "vivo_seq_th17_2025",
        "experiment": "Th17_Vivo_seq",
        "source_label": a,
        "target_id": b,
        "protein_symbol": c,
        "uniprot_accession": d,
        "residue": e,
        "vendor": f,
        "catalog": g,
        "paired_status": "direct_RNA_phospho",
    }
    for a, b, c, d, e, f, g in vivo_rows
]

icc_rows_raw = [
    ("intra-B1126-RPS6Pho", "RPS6_S235_S236", "RPS6", "P62753", "S235/S236", "BioLegend_or_panel_pending", "B1126"),
    ("intra-BHMIgG1003-phospho-p38-A16016A", "MAPK14_pSitePending", "MAPK14", "Q16539", "pending", "pending", "BHMIgG1003"),
    ("intra-BHM0007-phospho-p38-D3F9", "MAPK14_pSitePending_D3F9", "MAPK14", "Q16539", "pending", "pending", "BHM0007"),
    ("intra-BHM0008-phospho-p65-93H1", "RELA_pSitePending_93H1", "RELA", "Q04206", "pending", "pending", "BHM0008"),
    ("intra-BHM0010-phospho-c-Jun", "JUN_pSitePending", "JUN", "P05412", "pending", "pending", "BHM0010"),
    ("intra-phospho-LAT-pY226-J96-1238.58.93", "LAT_Y226", "LAT", "O43561", "Y226", "pending", "J96-1238.58.93"),
    ("intra-phospho-CREB-pS133-ATF-1-pS63-J151-21", "CREB1_S133_ATF1_S63", "CREB1;ATF1", "P16220;P18846", "S133;S63", "pending", "J151-21"),
    ("intra-phospho-Stat3-Tyr705-D3A7", "STAT3_Y705", "STAT3", "P40763", "Y705", "Cell Signaling Technology", "D3A7"),
    ("intra-phospho-SLP-76-Ser376-E3G9U", "LCP2_S376", "LCP2", "Q13094", "S376", "pending", "E3G9U"),
    ("intra-phospho-MEK1-pS298-J114-64", "MAP2K1_S298", "MAP2K1", "Q02750", "S298", "pending", "J114-64"),
    ("intra-BHMIgG1017-pSTAT5-Y6947", "STAT5A_STAT5B_Y694_Y699", "STAT5A;STAT5B", "P42229;P51692", "Y694;Y699", "pending", "BHMIgG1017"),
    ("intra-HM0025-phospho-p44-42-197G2", "MAPK1_MAPK3_T202_Y204", "MAPK1;MAPK3", "P28482;P27361", "T202/Y204;T185/Y187", "Cell Signaling Technology", "197G2"),
]
icc_rows = [
    {
        "dataset_id": "iccite_seq_tcell_2025",
        "experiment": "Perturb_icCITE_seq",
        "source_label": a,
        "target_id": b,
        "protein_symbol": c,
        "uniprot_accession": d,
        "residue": e,
        "vendor": f,
        "catalog": g,
        "paired_status": "direct_RNA_intra_ADT",
    }
    for a, b, c, d, e, f, g in icc_rows_raw
]

all_features = pd.DataFrame(blair_rows + vivo_rows + icc_rows)
all_features.to_csv(out / "tables" / "all_phospho_features_expanded.tsv", sep="\t", index=False)

coverage = []

vivo = pd.read_csv(paired / "vivo_seq_th17_2025" / "phospho_counts.tsv", sep="\t", index_col=0)
vivo_matrix_label = {
    "phos-STAT3": "STAT3_P40763_Y705",
    "phos-p65": "RELA_Q04206_S536",
    "phos-FOS": "FOS_P01100_S32",
    "phos-ERK1/2": "MAPK1_P28482_T202_Y204__MAPK3_P27361_T185_Y187",
}
for r in vivo_rows:
    s = pd.to_numeric(vivo[vivo_matrix_label[r["source_label"]]], errors="coerce")
    coverage.append({**r, "n_cells_total": len(s), "n_measured": int(s.notna().sum()), "n_detected_nonzero": int((s.fillna(0) > 0).sum())})

blair_adt = pd.read_csv(paired / "phospho_seq_blair_2025_phospho_multi" / "adt_counts.tsv", sep="\t", index_col=0)
if "pRPS6" in blair_adt.columns:
    s = pd.to_numeric(blair_adt["pRPS6"], errors="coerce")
    r = [x for x in blair_rows if x["experiment"] == "retinal_organoid_multi" and x["source_label"] == "pRPS6"][0]
    coverage.append({**r, "n_cells_total": len(s), "n_measured": int(s.notna().sum()), "n_detected_nonzero": int((s.fillna(0) > 0).sum())})

icc_features = pd.read_csv(paired / "iccite_seq_tcell_2025" / "phospho_counts" / "phospho_counts_features.tsv", sep="\t", header=None)[0].tolist()
icc_m = mmread(paired / "iccite_seq_tcell_2025" / "phospho_counts" / "phospho_counts.mtx").tocsr()
for i, label in enumerate(icc_features):
    r = [x for x in icc_rows if x["source_label"] == label][0]
    row = icc_m.getrow(i)
    coverage.append({**r, "n_cells_total": icc_m.shape[1], "n_measured": icc_m.shape[1], "n_detected_nonzero": int(row.nnz)})

cov = pd.DataFrame(coverage)
cov["detected_rate_among_total"] = cov["n_detected_nonzero"] / cov["n_cells_total"]
cov.to_csv(out / "tables" / "phospho_feature_cell_coverage.tsv", sep="\t", index=False)

def direct_dataset_count(sub: pd.DataFrame) -> int:
    direct_status = {"direct_RNA_ADT", "direct_RNA_intra_ADT", "direct_RNA_phospho"}
    return len(set(sub.loc[sub["paired_status"].isin(direct_status), "dataset_id"]))

anchor = (
    all_features.groupby("target_id")
    .agg(
        n_datasets=("dataset_id", lambda x: len(set(x))),
        datasets=("dataset_id", lambda x: ";".join(sorted(set(x)))),
        source_labels=("source_label", lambda x: ";".join(sorted(set(x)))),
        residues=("residue", lambda x: ";".join(sorted(set(map(str, x))))),
        paired_statuses=("paired_status", lambda x: ";".join(sorted(set(x)))),
    )
    .reset_index()
)
direct_counts = all_features.groupby("target_id").apply(direct_dataset_count, include_groups=False).rename("n_direct_paired_datasets").reset_index()
anchor = anchor.merge(direct_counts, on="target_id", how="left")
anchor["anchor_status"] = anchor.apply(
    lambda r: "direct_anchor" if r["n_direct_paired_datasets"] >= 2 else ("candidate_not_direct_paired" if r["n_datasets"] >= 2 else "single_dataset"),
    axis=1,
)
anchor.to_csv(out / "tables" / "phospho_anchor_candidates.tsv", sep="\t", index=False)

sample_rows = []
vivo_meta = pd.read_csv(paired / "vivo_seq_th17_2025" / "cell_metadata.tsv", sep="\t")
for cols in [["culture"], ["panel"], ["stim"], ["culture", "panel", "stim"]]:
    tab = vivo_meta.groupby(cols).size().reset_index(name="n_cells")
    tab.insert(0, "dataset_id", "vivo_seq_th17_2025")
    tab.insert(1, "grouping", "+".join(cols))
    sample_rows.append(tab)

icc_meta = pd.read_csv(paired / "iccite_seq_tcell_2025" / "cell_metadata.tsv", sep="\t", usecols=["cell_id", "orig.ident", "Donor", "PerturbedGene", "gRNA"])
for cols in [["Donor"], ["orig.ident"], ["PerturbedGene"]]:
    tab = icc_meta.groupby(cols).size().reset_index(name="n_cells").sort_values("n_cells", ascending=False)
    tab.insert(0, "dataset_id", "iccite_seq_tcell_2025")
    tab.insert(1, "grouping", "+".join(cols))
    sample_rows.append(tab.head(50))

blair_meta = pd.read_csv(paired / "phospho_seq_blair_2025_phospho_multi" / "cell_metadata.tsv", sep="\t")
tab = blair_meta.groupby(["assay_subset", "cell_type_label", "cell_ontology_id"]).size().reset_index(name="n_cells")
tab.insert(0, "dataset_id", "phospho_seq_blair_2025_phospho_multi")
tab.insert(1, "grouping", "assay_subset+cell_type_label")
sample_rows.append(tab)

sample_distribution = pd.concat(sample_rows, ignore_index=True, sort=False)
sample_distribution.to_csv(out / "tables" / "cell_sample_distribution.tsv", sep="\t", index=False)

platform = pd.DataFrame(
    [
        {
            "dataset_id": "vivo_seq_th17_2025",
            "platform": "10x Genomics 3' v3 scRNA-seq with inCITE/Vivo-seq phospho antibody readout",
            "evidence": "GEO and article state all samples were hashed into one 10x Genomics 3' V3 well.",
            "compatibility_note": "mouse; needs ortholog mapping before human cross-dataset RNA training",
        },
        {
            "dataset_id": "phospho_seq_blair_2025_phospho_multi",
            "platform": "10x Genomics Multiome ATAC + RNA with Phospho-seq ADT",
            "evidence": "GEO and article describe Phospho-seq Multi using 10x scATAC + RNA multiome kit.",
            "compatibility_note": "human; RNA direct paired only in Phospho-Multi subset, 1474 cells",
        },
        {
            "dataset_id": "iccite_seq_tcell_2025",
            "platform": "10x-style single-cell RNA with ADT/intra/CROP assays; Illumina NovaSeq 6000 sequencing",
            "evidence": "Processed Seurat object has RNA, ADT, intra and CROP assays; ENA run metadata lists Illumina NovaSeq 6000.",
            "compatibility_note": "human; best direct RNA + phospho coverage",
        },
    ]
)
platform.to_csv(out / "tables" / "platform_compatibility.tsv", sep="\t", index=False)

lines = [
    "# 位点与覆盖审计",
    "",
    "Blair pRPS6 已确认是 BioLegend 608602，phospho site 为 S235/236。这个结论来自 Blair Nature Communications 2025 Supplementary Dataset 1。",
    "",
    "直接 paired anchor：",
]
direct = anchor[anchor["anchor_status"] == "direct_anchor"].sort_values("target_id")
for _, r in direct.iterrows():
    lines.append(f"- {r['target_id']}: {r['datasets']}；labels={r['source_labels']}；residues={r['residues']}")
lines.extend(
    [
        "",
        "非直接 paired 或需谨慎的候选：",
    ]
)
for _, r in anchor[anchor["anchor_status"] == "candidate_not_direct_paired"].sort_values("target_id").iterrows():
    lines.append(f"- {r['target_id']}: {r['datasets']}；paired_status={r['paired_statuses']}")
lines.extend(
    [
        "",
        "限制：Blair cerebral organoid 的丰富 phospho panel 主要是 ADT + ATAC，并非直接 RNA + phospho；如果使用 bridge/imputed RNA，必须单独标注，不能与 direct paired 混合评估。",
    ]
)
(out / "reports" / "anchor_and_platform_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print("wrote anchor audit tables")
