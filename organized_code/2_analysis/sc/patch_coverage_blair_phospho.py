from pathlib import Path
import pandas as pd

root = Path(r"D:\lsy")
out = root / "02_results" / "single_cell" / "20260510_data_coverage"
coverage_path = out / "tables" / "dataset_matrix_coverage.tsv"

coverage = pd.read_csv(coverage_path, sep="\t")
mask = coverage["dataset_id"] == "phospho_seq_blair_2025_phospho_multi"
coverage.loc[mask, "n_phospho_features"] = 1
coverage.loc[mask, "phospho_total_values"] = 1474
coverage.loc[mask, "phospho_non_missing"] = 1474
coverage.loc[mask, "phospho_nonzero"] = 1027
coverage.loc[mask, "phospho_non_missing_rate"] = 1.0
coverage.loc[mask, "phospho_nonzero_rate"] = 1027 / 1474
coverage.to_csv(coverage_path, sep="\t", index=False)

report = out / "reports" / "data_coverage_audit.md"
text = report.read_text(encoding="utf-8")
text = text.replace(
    "phospho 非零率 0.5078",
    "phospho 非零率 0.6967；明显 phospho 特征只有 pRPS6",
)
text += "\n修正：Blair ADT 中 `PAX6`、`PKCa`、`PCNA` 只是以 P 开头，不是 phospho 命名；覆盖统计只保留 `pRPS6`。\n"
report.write_text(text, encoding="utf-8")
print("patched Blair phospho coverage")
