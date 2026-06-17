from pathlib import Path
import gzip
import re
import pandas as pd
from scipy.io import mmread

root = Path(r"D:\lsy")
paired = root / "01_data" / "single_cell" / "intermediate" / "paired_matrices"
meta = root / "01_data" / "shared" / "metadata"
out = root / "02_results" / "single_cell" / "20260510_data_coverage"
for sub in ["tables", "reports"]:
    (out / sub).mkdir(parents=True, exist_ok=True)


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def mtx_shape(path: Path) -> tuple[int, int, int]:
    m = mmread(path)
    return int(m.shape[0]), int(m.shape[1]), int(m.nnz)


def csv_gz_shape(path: Path) -> tuple[int, int]:
    n_rows = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split(",")
        for _ in handle:
            n_rows += 1
    return n_rows, max(0, len(header) - 1)


def count_nonzero_tsv(path: Path) -> tuple[int, int, int]:
    df = pd.read_csv(path, sep="\t", index_col=0)
    values = df.apply(pd.to_numeric, errors="coerce")
    non_missing = int(values.notna().sum().sum())
    nonzero = int((values.fillna(0) != 0).sum().sum())
    total = int(values.shape[0] * values.shape[1])
    return total, non_missing, nonzero


datasets = []

vivo_dir = paired / "vivo_seq_th17_2025"
vivo_genes = set(read_lines(vivo_dir / "genes.tsv"))
vivo_barcodes = read_lines(vivo_dir / "barcodes.tsv")
vivo_r, vivo_c, vivo_nnz = mtx_shape(vivo_dir / "rna_counts.mtx")
vivo_total, vivo_non_missing, vivo_nonzero = count_nonzero_tsv(vivo_dir / "phospho_counts.tsv")
vivo_phospho = list(pd.read_csv(vivo_dir / "phospho_counts.tsv", sep="\t", nrows=0).columns[1:])
datasets.append(
    {
        "dataset_id": "vivo_seq_th17_2025",
        "paired_status": "exported",
        "n_cells": len(vivo_barcodes),
        "n_rna_genes": len(vivo_genes),
        "rna_nnz": vivo_nnz,
        "rna_density": vivo_nnz / (vivo_r * vivo_c),
        "n_phospho_features": len(vivo_phospho),
        "phospho_total_values": vivo_total,
        "phospho_non_missing": vivo_non_missing,
        "phospho_nonzero": vivo_nonzero,
        "phospho_non_missing_rate": vivo_non_missing / vivo_total,
        "phospho_nonzero_rate": vivo_nonzero / vivo_total,
    }
)

blair_dir = paired / "phospho_seq_blair_2025_phospho_multi"
blair_barcodes = read_lines(blair_dir / "barcodes.tsv")
blair_raw = root / "01_data" / "single_cell" / "raw" / "phospho_seq_blair_2025" / "extracted"
blair_gene_n, blair_cell_n = csv_gz_shape(blair_raw / "GSM8726041_Phospho-Multi_RNA.csv.gz")
blair_adt = pd.read_csv(blair_dir / "adt_counts.tsv", sep="\t", index_col=0)
blair_adt_features = list(blair_adt.columns)
blair_phospho = [x for x in blair_adt_features if re.search(r"^p|phos", str(x), re.I)]
datasets.append(
    {
        "dataset_id": "phospho_seq_blair_2025_phospho_multi",
        "paired_status": "rna_adt_barcode_aligned_site_pending",
        "n_cells": len(blair_barcodes),
        "n_rna_genes": blair_gene_n,
        "rna_nnz": "not_counted_csv_gz",
        "rna_density": "not_counted_csv_gz",
        "n_phospho_features": len(blair_phospho),
        "phospho_total_values": int(blair_adt[blair_phospho].shape[0] * max(1, len(blair_phospho))) if blair_phospho else 0,
        "phospho_non_missing": int(blair_adt[blair_phospho].notna().sum().sum()) if blair_phospho else 0,
        "phospho_nonzero": int((blair_adt[blair_phospho].fillna(0) != 0).sum().sum()) if blair_phospho else 0,
        "phospho_non_missing_rate": 1.0 if blair_phospho else 0.0,
        "phospho_nonzero_rate": float((blair_adt[blair_phospho].fillna(0) != 0).sum().sum() / (blair_adt[blair_phospho].shape[0] * len(blair_phospho))) if blair_phospho else 0.0,
    }
)

icc_dir = paired / "iccite_seq_tcell_2025"
icc_genes = set(read_lines(icc_dir / "rna_full_counts" / "rna_full_counts_features.tsv"))
icc_barcodes = read_lines(icc_dir / "rna_full_counts" / "rna_full_counts_barcodes.tsv")
icc_r, icc_c, icc_nnz = mtx_shape(icc_dir / "rna_full_counts" / "rna_full_counts.mtx")
icc_pr, icc_pc, icc_pnnz = mtx_shape(icc_dir / "phospho_counts" / "phospho_counts.mtx")
icc_phospho = read_lines(icc_dir / "phospho_counts" / "phospho_counts_features.tsv")
datasets.append(
    {
        "dataset_id": "iccite_seq_tcell_2025",
        "paired_status": "exported_full_rna",
        "n_cells": len(icc_barcodes),
        "n_rna_genes": len(icc_genes),
        "rna_nnz": icc_nnz,
        "rna_density": icc_nnz / (icc_r * icc_c),
        "n_phospho_features": len(icc_phospho),
        "phospho_total_values": icc_pr * icc_pc,
        "phospho_non_missing": icc_pr * icc_pc,
        "phospho_nonzero": icc_pnnz,
        "phospho_non_missing_rate": 1.0,
        "phospho_nonzero_rate": icc_pnnz / (icc_pr * icc_pc),
    }
)

coverage = pd.DataFrame(datasets)
coverage.to_csv(out / "tables" / "dataset_matrix_coverage.tsv", sep="\t", index=False)

gene_sets = {
    "vivo_seq_th17_2025_mouse_symbols": vivo_genes,
    "phospho_seq_blair_2025_human_symbols": set(pd.read_csv(blair_raw / "GSM8726041_Phospho-Multi_RNA.csv.gz", usecols=[0]).iloc[:, 0].astype(str)),
    "iccite_seq_tcell_2025_human_symbols": icc_genes,
}

gene_overlap_rows = []
keys = list(gene_sets)
for i, a in enumerate(keys):
    for b in keys[i + 1 :]:
        inter = gene_sets[a] & gene_sets[b]
        union = gene_sets[a] | gene_sets[b]
        gene_overlap_rows.append(
            {
                "set_a": a,
                "set_b": b,
                "n_a": len(gene_sets[a]),
                "n_b": len(gene_sets[b]),
                "n_intersection": len(inter),
                "jaccard": len(inter) / len(union),
            }
        )
pd.DataFrame(gene_overlap_rows).to_csv(out / "tables" / "rna_gene_symbol_overlap.tsv", sep="\t", index=False)

phospho_schema = pd.read_csv(meta / "phospho_target_schema.tsv", sep="\t")
phospho_schema.to_csv(out / "tables" / "phospho_target_schema_snapshot.tsv", sep="\t", index=False)

target_rows = []
for target, sub in phospho_schema.groupby("target_id"):
    ds = sorted(set(";".join(sub["datasets_confirmed"].dropna().astype(str)).split(";")))
    ds = [x for x in ds if x]
    target_rows.append(
        {
            "target_id": target,
            "n_datasets": len(ds),
            "datasets": ";".join(ds),
            "residue": ";".join(sorted(set(sub["residue"].dropna().astype(str)))),
            "status": "shared" if len(ds) >= 2 else "single_dataset",
        }
    )
target_coverage = pd.DataFrame(target_rows).sort_values(["status", "target_id"])
target_coverage.to_csv(out / "tables" / "phospho_target_dataset_coverage.tsv", sep="\t", index=False)

usable = target_coverage[target_coverage["status"] == "shared"].copy()
usable.to_csv(out / "tables" / "usable_cross_dataset_phospho_targets.tsv", sep="\t", index=False)

lines = [
    "# 数据覆盖审计",
    "",
    "结论：当前覆盖不足以支撑“10-15 个共同 phospho 位点跨多个数据集都有”。真实可直接跨数据集使用的位点只有 STAT3 Y705；RPS6 在 Blair 与 icCITE 都出现，但 Blair 的精确位点未确认，只能列为待确认。",
    "",
    "矩阵覆盖：",
]
for row in datasets:
    lines.append(
        f"- {row['dataset_id']}: {row['n_cells']} 细胞，{row['n_rna_genes']} RNA 基因，"
        f"{row['n_phospho_features']} phospho-like 特征，phospho 非零率 {row['phospho_nonzero_rate']:.4f}"
        if isinstance(row["phospho_nonzero_rate"], float)
        else f"- {row['dataset_id']}: {row}"
    )
lines.extend(
    [
        "",
        "问题点：",
        "- Vivo 是小样本同细胞 RNA + phospho，只有 2230 细胞。",
        "- Blair Phospho-Multi 有 RNA + ADT 对齐，但 ADT 里只有 pRPS6 明显是 phospho，且位点未定。",
        "- icCITE-seq 覆盖最好，114398 细胞、29297 基因、12 个 phospho-like intracellular features。",
        "- Vivo 是 mouse，Blair 和 icCITE 是 human。跨物种训练前必须做人鼠 ortholog 映射，不能直接拿 gene symbol 交集当同源基因集。",
    ]
)
(out / "reports" / "data_coverage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(coverage.to_string(index=False))
print("\nusable shared phospho targets")
print(usable.to_string(index=False))
