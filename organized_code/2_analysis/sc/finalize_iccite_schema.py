from pathlib import Path
import pandas as pd

root = Path(r"D:\lsy")
meta = root / "01_data" / "shared" / "metadata"
tables = root / "02_results" / "single_cell" / "20260510_schema_unification" / "tables"
report_dir = root / "02_results" / "single_cell" / "20260510_schema_unification" / "reports"
matrix_dir = root / "01_data" / "single_cell" / "intermediate" / "paired_matrices" / "iccite_seq_tcell_2025"

schema_path = meta / "unified_dataset_schema.tsv"
schema = pd.read_csv(schema_path, sep="\t")
schema = schema[schema["dataset_id"] != "iccite_seq_tcell_2025"]
schema = pd.concat(
    [
        schema,
        pd.DataFrame(
            [
                {
                    "dataset_id": "iccite_seq_tcell_2025",
                    "organism": "Homo sapiens",
                    "matrix_status": "exported_full_rna",
                    "paired_rna_phospho": "yes",
                    "rna_matrix": str(matrix_dir / "rna_full_counts" / "rna_full_counts.mtx"),
                    "phospho_matrix": str(matrix_dir / "phospho_counts" / "phospho_counts.mtx"),
                    "cell_metadata": str(matrix_dir / "cell_metadata.tsv"),
                    "notes": "Exported from Zenodo Seurat object. RNA is full gene count matrix, intra assay contains intracellular proteins, phospho matrix is subset of intra features with phospho-like labels.",
                }
            ]
        ),
    ],
    ignore_index=True,
)
schema.to_csv(schema_path, sep="\t", index=False)

features = pd.read_csv(matrix_dir / "phospho_counts" / "phospho_counts_features.tsv", sep="\t", header=None, names=["source_label"])
known = {
    "intra-B1126-RPS6Pho": ("RPS6_pSitePending", "RPS6", "P62753", "pending_antibody_clone"),
    "intra-BHMIgG1003-phospho-p38-A16016A": ("MAPK14_pSitePending", "MAPK14", "Q16539", "pending_antibody_clone"),
    "intra-BHM0007-phospho-p38-D3F9": ("MAPK14_pSitePending_D3F9", "MAPK14", "Q16539", "pending_antibody_clone"),
    "intra-BHM0008-phospho-p65-93H1": ("RELA_pSitePending_93H1", "RELA", "Q04206", "pending_antibody_clone"),
    "intra-BHM0010-phospho-c-Jun": ("JUN_pSitePending", "JUN", "P05412", "pending_antibody_clone"),
    "intra-phospho-LAT-pY226-J96-1238.58.93": ("LAT_Y226", "LAT", "O43561", "Y226"),
    "intra-phospho-CREB-pS133-ATF-1-pS63-J151-21": ("CREB1_S133_ATF1_S63", "CREB1;ATF1", "P16220;P18846", "S133;S63"),
    "intra-phospho-Stat3-Tyr705-D3A7": ("STAT3_Y705", "STAT3", "P40763", "Y705"),
    "intra-phospho-SLP-76-Ser376-E3G9U": ("LCP2_S376", "LCP2", "Q13094", "S376"),
    "intra-phospho-MEK1-pS298-J114-64": ("MAP2K1_S298", "MAP2K1", "Q02750", "S298"),
    "intra-BHMIgG1017-pSTAT5-Y6947": ("STAT5A_STAT5B_Y694_Y699", "STAT5A;STAT5B", "P42229;P51692", "Y694;Y699"),
    "intra-HM0025-phospho-ZAP70-SYK-pY319-pY352": ("ZAP70_Y319_SYK_Y352", "ZAP70;SYK", "P43403;P43405", "Y319;Y352"),
}

phospho_path = meta / "phospho_target_schema.tsv"
phospho = pd.read_csv(phospho_path, sep="\t")
rows = []
for label in features["source_label"]:
    target_id, symbol, uniprot, residue = known.get(label, (label.replace("intra-", "").replace("-", "_"), "pending", "pending", "pending"))
    rows.append(
        {
            "source_label": label,
            "target_id": target_id,
            "protein_symbol": symbol,
            "hgnc_symbol": symbol,
            "uniprot_accession": uniprot,
            "residue": residue,
            "canonical_label": f"{symbol}_{uniprot}_{residue}",
            "datasets_confirmed": "iccite_seq_tcell_2025",
        }
    )
new = pd.DataFrame(rows)
phospho = phospho[~phospho["source_label"].isin(set(new["source_label"]))]
phospho = pd.concat([phospho, new], ignore_index=True)
phospho.to_csv(phospho_path, sep="\t", index=False)

common_path = tables / "common_phospho_targets.tsv"
common = pd.DataFrame(
    [
        {
            "target_id": "STAT3_Y705",
            "canonical_label": "STAT3_P40763_Y705",
            "datasets_confirmed": "vivo_seq_th17_2025;iccite_seq_tcell_2025",
            "shared_across_training_datasets": "yes",
            "reason": "Vivo has phos-STAT3 mapped to STAT3 Y705; icCITE has intra-phospho-Stat3-Tyr705-D3A7.",
        },
        {
            "target_id": "RPS6_pSitePending",
            "canonical_label": "RPS6_P62753_pending_antibody_clone",
            "datasets_confirmed": "phospho_seq_blair_2025;iccite_seq_tcell_2025",
            "shared_across_training_datasets": "pending_exact_site",
            "reason": "Blair has pRPS6 and icCITE has RPS6Pho; exact epitope/site must be checked before treating as the same phosphosite.",
        },
    ]
)
common.to_csv(common_path, sep="\t", index=False)

status_path = tables / "paired_matrix_export_status.tsv"
status = pd.read_csv(status_path, sep="\t")
status = status[status["dataset_id"] != "iccite_seq_tcell_2025"]
status = pd.concat(
    [
        status,
        pd.DataFrame(
            [
                {
                    "dataset_id": "iccite_seq_tcell_2025",
                    "status": "paired_matrix_exported_full_rna",
                    "n_cells": 114398,
                    "n_genes": 29297,
                    "n_phospho_targets": len(features),
                    "matrix_dir": str(matrix_dir),
                }
            ]
        ),
    ],
    ignore_index=True,
)
status.to_csv(status_path, sep="\t", index=False)

lines = [
    "# icCITE-seq 导出记录",
    "",
    "已从 Zenodo Seurat 对象导出全基因 RNA 矩阵、intra 矩阵、phospho 子矩阵和细胞元数据。",
    "",
    "RNA: 29,297 基因 × 114,398 细胞。",
    f"phospho features: {len(features)}。",
    "",
    "当前明确跨数据集重叠：STAT3 Y705 覆盖 Vivo-seq 与 icCITE-seq。",
    "RPS6 在 Blair 与 icCITE-seq 都出现，但 Blair 的 pRPS6 精确位点还没有抗体注释，不能直接并入共同位点。",
]
(report_dir / "iccite_export_20260510.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("finalized icCITE schema")
