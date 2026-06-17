from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(r"E:\data\gongke\TCGA-TCPA")
RAW = ROOT / "02_results" / "raw_external" / "v10b_p100_validation"
OUT = ROOT / "02_results" / "figure_sources" / "20260528_fig4_locked_p100_v10b_cosine_direction"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def sem(values):
    x = pd.Series(values).dropna().to_numpy(float)
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def ci_low(values):
    x = pd.Series(values).dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan
    if len(x) == 1:
        return float(x[0])
    return float(np.mean(x) - 1.96 * sem(x))


def ci_high(values):
    x = pd.Series(values).dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan
    if len(x) == 1:
        return float(x[0])
    return float(np.mean(x) + 1.96 * sem(x))


def drug_class(perturbation, target_genes):
    p = str(perturbation).strip().lower()
    t = str(target_genes).upper()
    if "HDAC" in t or p in {"vorinostat"}:
        return "HDAC"
    if "PSMB" in t or p in {"bortezomib", "carfilzomib"}:
        return "蛋白酶体"
    if "MAP2K" in t or p in {"trametinib", "selumetinib"}:
        return "MEK"
    if "EGFR" in t or "ERBB2" in t:
        return "EGFR/HER2"
    if any(g in t for g in ["ABL1", "SRC", "LYN", "HCK"]):
        return "ABL/SRC"
    if "MTOR" in t:
        return "mTOR"
    return "其他"


def summarize(grouped, value_col):
    return grouped.agg(
        n=(value_col, "count"),
        mean=(value_col, "mean"),
        median=(value_col, "median"),
        sem=(value_col, sem),
        ci95_low=(value_col, ci_low),
        ci95_high=(value_col, ci_high),
    ).reset_index()


metrics = pd.read_csv(RAW / "global_graph_v9_p100_metrics.tsv", sep="\t")
drug_summary = pd.read_csv(RAW / "global_graph_v9_p100_drug_summary.tsv", sep="\t")

for df in [metrics, drug_summary]:
    df["perturbation"] = df["perturbation"].astype(str).str.strip()
    df["target_genes"] = df["target_genes"].astype(str)
    df["drug_class"] = [drug_class(p, t) for p, t in zip(df["perturbation"], df["target_genes"])]

metrics = metrics[metrics["mode"].isin(["true", "zero"])].copy()
metrics["comparison_id"] = metrics.groupby("mode").cumcount() + 1

true = metrics[metrics["mode"].eq("true")].copy()
zero = metrics[metrics["mode"].eq("zero")].copy()

panel_b_rows = []
for site_set, metric_col, label in [
    ("all_sites", "all_cosine", "余弦"),
    ("all_sites", "all_direction", "方向准确率"),
    ("responsive_top20", "responsive20_cosine", "余弦"),
    ("responsive_top20", "responsive20_direction", "方向准确率"),
]:
    vals = true[metric_col]
    panel_b_rows.append({
        "panel": "b",
        "mode": "true",
        "site_set": site_set,
        "metric": label,
        "metric_column": metric_col,
        "n": int(vals.count()),
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "sem": sem(vals),
        "ci95_low": ci_low(vals),
        "ci95_high": ci_high(vals),
    })
panel_b = pd.DataFrame(panel_b_rows)
panel_b.to_csv(TABLES / "panel_b_p100_true_overall_bars_cosine_direction.tsv", sep="\t", index=False)

panel_c_wide = true[[
    "comparison_id",
    "perturbation",
    "target_genes",
    "drug_class",
    "n_shared_sites",
    "responsive20_cosine",
    "responsive20_direction",
]].rename(columns={
    "responsive20_cosine": "responsive_top20_cosine",
    "responsive20_direction": "responsive_top20_direction_accuracy",
})
panel_c_wide.to_csv(TABLES / "panel_c_p100_true_comparison_distribution_wide.tsv", sep="\t", index=False)

panel_c_long = panel_c_wide.melt(
    id_vars=["comparison_id", "perturbation", "target_genes", "drug_class", "n_shared_sites"],
    value_vars=["responsive_top20_cosine", "responsive_top20_direction_accuracy"],
    var_name="metric",
    value_name="value",
)
panel_c_long.to_csv(TABLES / "panel_c_p100_true_comparison_distribution_long.tsv", sep="\t", index=False)

panel_d_rows = []
for cls, sub in true.groupby("drug_class"):
    for site_set, metric_col, label in [
        ("all_sites", "all_cosine", "余弦"),
        ("all_sites", "all_direction", "方向准确率"),
        ("responsive_top20", "responsive20_cosine", "余弦"),
        ("responsive_top20", "responsive20_direction", "方向准确率"),
    ]:
        vals = sub[metric_col]
        panel_d_rows.append({
            "panel": "d",
            "drug_class": cls,
            "site_set": site_set,
            "metric": label,
            "metric_column": metric_col,
            "n": int(vals.count()),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "sem": sem(vals),
            "ci95_low": ci_low(vals),
            "ci95_high": ci_high(vals),
            "n_shared_sites_median": float(sub["n_shared_sites"].median()),
        })
panel_d = pd.DataFrame(panel_d_rows)
order = ["EGFR/HER2", "ABL/SRC", "HDAC", "mTOR", "MEK", "蛋白酶体", "其他"]
panel_d["drug_class"] = pd.Categorical(panel_d["drug_class"], categories=order, ordered=True)
panel_d = panel_d.sort_values(["drug_class", "site_set", "metric"]).copy()
panel_d["drug_class"] = panel_d["drug_class"].astype(str)
panel_d.to_csv(TABLES / "panel_d_p100_true_drug_class_cosine_direction.tsv", sep="\t", index=False)

drug_true = drug_summary[drug_summary["mode"].eq("true")].copy()
panel_e = drug_true[[
    "perturbation",
    "target_genes",
    "drug_class",
    "n",
    "all_cosine",
    "responsive20_cosine",
    "all_direction",
    "responsive20_direction",
]].rename(columns={
    "n": "n_comparisons",
    "all_cosine": "all_sites_cosine",
    "responsive20_cosine": "responsive_top20_cosine",
    "all_direction": "all_sites_direction_accuracy",
    "responsive20_direction": "responsive_top20_direction_accuracy",
})
n_shared = true.groupby("perturbation", as_index=False).agg(n_shared_sites_median=("n_shared_sites", "median"))
panel_e = panel_e.merge(n_shared, on="perturbation", how="left")
panel_e = panel_e.sort_values("responsive_top20_cosine", ascending=False).reset_index(drop=True)
panel_e["heatmap_row_order"] = np.arange(1, len(panel_e) + 1)
panel_e.to_csv(TABLES / "panel_e_p100_true_drug_heatmap_multi_metrics.tsv", sep="\t", index=False)

paired = true[[
    "comparison_id",
    "perturbation",
    "target_genes",
    "drug_class",
    "n_shared_sites",
    "all_cosine",
    "responsive20_cosine",
    "all_direction",
    "responsive20_direction",
]].merge(
    zero[[
        "comparison_id",
        "all_cosine",
        "responsive20_cosine",
        "all_direction",
        "responsive20_direction",
    ]],
    on="comparison_id",
    suffixes=("_true", "_zero"),
)
paired["all_sites_cosine_delta_true_minus_zero"] = paired["all_cosine_true"] - paired["all_cosine_zero"]
paired["responsive_top20_cosine_delta_true_minus_zero"] = paired["responsive20_cosine_true"] - paired["responsive20_cosine_zero"]
paired["all_sites_direction_delta_true_minus_zero"] = paired["all_direction_true"] - paired["all_direction_zero"]
paired["responsive_top20_direction_delta_true_minus_zero"] = paired["responsive20_direction_true"] - paired["responsive20_direction_zero"]
paired.to_csv(TABLES / "panel_f_p100_true_vs_zero_paired.tsv", sep="\t", index=False)

panel_f_long = []
for _, row in paired.iterrows():
    for site_set, true_col, zero_col in [
        ("all_sites", "all_cosine_true", "all_cosine_zero"),
        ("responsive_top20", "responsive20_cosine_true", "responsive20_cosine_zero"),
    ]:
        panel_f_long.append({
            "comparison_id": row["comparison_id"],
            "perturbation": row["perturbation"],
            "drug_class": row["drug_class"],
            "site_set": site_set,
            "mode": "true",
            "cosine": row[true_col],
        })
        panel_f_long.append({
            "comparison_id": row["comparison_id"],
            "perturbation": row["perturbation"],
            "drug_class": row["drug_class"],
            "site_set": site_set,
            "mode": "zero",
            "cosine": row[zero_col],
        })
pd.DataFrame(panel_f_long).to_csv(TABLES / "panel_f_p100_true_vs_zero_paired_long.tsv", sep="\t", index=False)

panel_f_summary = pd.DataFrame([
    {
        "site_set": "all_sites",
        "metric": "余弦",
        "n_pairs": int(len(paired)),
        "true_mean": float(paired["all_cosine_true"].mean()),
        "zero_mean": float(paired["all_cosine_zero"].mean()),
        "mean_delta_true_minus_zero": float(paired["all_sites_cosine_delta_true_minus_zero"].mean()),
        "median_delta_true_minus_zero": float(paired["all_sites_cosine_delta_true_minus_zero"].median()),
        "delta_ci95_low": ci_low(paired["all_sites_cosine_delta_true_minus_zero"]),
        "delta_ci95_high": ci_high(paired["all_sites_cosine_delta_true_minus_zero"]),
    },
    {
        "site_set": "responsive_top20",
        "metric": "余弦",
        "n_pairs": int(len(paired)),
        "true_mean": float(paired["responsive20_cosine_true"].mean()),
        "zero_mean": float(paired["responsive20_cosine_zero"].mean()),
        "mean_delta_true_minus_zero": float(paired["responsive_top20_cosine_delta_true_minus_zero"].mean()),
        "median_delta_true_minus_zero": float(paired["responsive_top20_cosine_delta_true_minus_zero"].median()),
        "delta_ci95_low": ci_low(paired["responsive_top20_cosine_delta_true_minus_zero"]),
        "delta_ci95_high": ci_high(paired["responsive_top20_cosine_delta_true_minus_zero"]),
    },
])
panel_f_summary.to_csv(TABLES / "panel_f_p100_true_vs_zero_paired_summary.tsv", sep="\t", index=False)

xlsx = OUT / "SCP682_PPKO_V10B_Fig4_locked_P100_tables.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
    panel_b.to_excel(writer, sheet_name="panel_b_bars", index=False)
    panel_c_wide.to_excel(writer, sheet_name="panel_c_wide", index=False)
    panel_c_long.to_excel(writer, sheet_name="panel_c_long", index=False)
    panel_d.to_excel(writer, sheet_name="panel_d_classes", index=False)
    panel_e.to_excel(writer, sheet_name="panel_e_heatmap", index=False)
    paired.to_excel(writer, sheet_name="panel_f_paired", index=False)
    panel_f_summary.to_excel(writer, sheet_name="panel_f_summary", index=False)

manifest = {
    "source": str(RAW),
    "output": str(OUT),
    "model": "SCP682-PPKO V10B",
    "external_validation": "P100 only",
    "n_true_comparisons": int(len(true)),
    "n_zero_comparisons": int(len(zero)),
    "shuffled_removed": True,
    "primary_metrics": ["cosine", "direction_accuracy"],
    "drug_classes_present": sorted(true["drug_class"].unique().tolist()),
    "tables": sorted(p.name for p in TABLES.glob("*.tsv")),
    "workbook": str(xlsx),
}
(REPORTS / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

readme = [
    "# Fig 4 锁定版 P100 作图源表",
    "",
    "范围：只用 P100；主指标为余弦和方向准确率；shuffled 已移除。",
    "",
    "## 面板对应表",
    "",
    "- panel_b_p100_true_overall_bars_cosine_direction.tsv：true 模式下 all_sites 与 responsive_top20 两档，余弦与方向准确率共四根柱。",
    "- panel_c_p100_true_comparison_distribution_wide.tsv：逐 comparison 宽表，n=125。",
    "- panel_c_p100_true_comparison_distribution_long.tsv：逐 comparison 长表，用于小提琴图或箱线图。",
    "- panel_d_p100_true_drug_class_cosine_direction.tsv：按药物类别拆分，含余弦与方向准确率。",
    "- panel_e_p100_true_drug_heatmap_multi_metrics.tsv：药物热图表，含 16 个药物、多指标和 n_shared_sites 中位数。",
    "- panel_f_p100_true_vs_zero_paired.tsv：true vs zero 配对宽表。",
    "- panel_f_p100_true_vs_zero_paired_long.tsv：true vs zero 配对长表，用于连线图。",
    "- panel_f_p100_true_vs_zero_paired_summary.tsv：true-zero 差值摘要。",
    "",
    "合并工作簿：SCP682_PPKO_V10B_Fig4_locked_P100_tables.xlsx",
]
(OUT / "MANIFEST.md").write_text("\n".join(readme), encoding="utf-8")

print(json.dumps(manifest, ensure_ascii=False, indent=2))
print(panel_b.to_string(index=False))
print(panel_f_summary.to_string(index=False))
