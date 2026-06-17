from pathlib import Path
import pandas as pd

root = Path(r"D:\lsy")
meta = root / "01_data" / "shared" / "metadata"
result = root / "02_results" / "single_cell" / "20260510_iccite_download"
for sub in ["tables", "reports"]:
    (result / sub).mkdir(parents=True, exist_ok=True)

run_table = meta / "PRJDB16517_ena_read_run.tsv"
df = pd.read_csv(run_table, sep="\t")

def sum_bytes(series: pd.Series) -> int:
    total = 0
    for value in series.dropna().astype(str):
        for item in value.split(";"):
            if item.strip():
                total += int(item)
    return total

summary = (
    df.groupby(["library_strategy", "library_source", "library_selection"], dropna=False)
    .agg(
        n_runs=("run_accession", "nunique"),
        n_samples=("sample_accession", "nunique"),
        n_experiments=("experiment_accession", "nunique"),
        fastq_bytes=("fastq_bytes", sum_bytes),
    )
    .reset_index()
)
summary["fastq_gb"] = (summary["fastq_bytes"] / 1024**3).round(2)
summary.to_csv(result / "tables" / "PRJDB16517_run_summary.tsv", sep="\t", index=False)

df.to_csv(result / "tables" / "PRJDB16517_ena_read_run.tsv", sep="\t", index=False)

rna = df[df["library_strategy"].astype(str).str.contains("RNA", case=False, na=False)].copy()
rna.to_csv(result / "tables" / "PRJDB16517_rna_runs.tsv", sep="\t", index=False)

lines = [
    "# PRJDB16517 下载记录",
    "",
    f"ENA run 表共 {len(df)} 个 run。",
    "",
    "按测序类型汇总：",
    "",
]
for _, row in summary.iterrows():
    lines.append(
        f"- {row['library_strategy']} / {row['library_source']} / {row['library_selection']}: "
        f"{row['n_runs']} runs, {row['fastq_gb']} GB"
    )
lines.extend(
    [
        "",
        "作者处理后对象来自 Zenodo: https://zenodo.org/records/16020737",
        "文件名: Perturb_icCITE_seq_FOXP3_regulators.rds",
        "大小: 28.0 GB",
        "",
        "判断：优先下载处理后 Seurat 对象，不直接全量下载 DRA FASTQ。DRA 全量包含 ChIP-Seq、ATAC-seq、RNA-Seq 和其它 cDNA/oligo-dT 文库，体量远大于当前矩阵构建需求。",
    ]
)
(result / "reports" / "PRJDB16517_download_strategy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary.to_string(index=False))
