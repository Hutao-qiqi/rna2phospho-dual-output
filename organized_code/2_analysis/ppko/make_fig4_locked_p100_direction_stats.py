from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"E:\data\gongke\TCGA-TCPA")
FIG = ROOT / "02_results" / "figure_sources" / "20260528_fig4_locked_p100_v10b_cosine_direction"
TABLES = FIG / "tables"
REPORTS = FIG / "reports"


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


def iqr_low(values):
    x = pd.Series(values).dropna().to_numpy(float)
    return float(np.quantile(x, 0.25)) if len(x) else np.nan


def iqr_high(values):
    x = pd.Series(values).dropna().to_numpy(float)
    return float(np.quantile(x, 0.75)) if len(x) else np.nan


def paired_wilcoxon(a, b):
    x = pd.Series(a).to_numpy(float)
    y = pd.Series(b).to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return np.nan
    diff = x[ok] - y[ok]
    if np.allclose(diff, 0):
        return 1.0
    return float(stats.wilcoxon(x[ok], y[ok], alternative="two-sided", zero_method="wilcox").pvalue)


panel_b = pd.read_csv(TABLES / "panel_b_p100_true_overall_bars_cosine_direction.tsv", sep="\t", dtype={"mode": str})
panel_c = pd.read_csv(TABLES / "panel_c_p100_true_comparison_distribution_wide.tsv", sep="\t")
panel_d = pd.read_csv(TABLES / "panel_d_p100_true_drug_class_cosine_direction.tsv", sep="\t")
panel_e = pd.read_csv(TABLES / "panel_e_p100_true_drug_heatmap_multi_metrics.tsv", sep="\t")
panel_f = pd.read_csv(TABLES / "panel_f_p100_true_vs_zero_paired.tsv", sep="\t")

# b：两档方向准确率均值与置信区间
panel_b_direction = panel_b[panel_b["metric"].eq("方向准确率")].copy()
panel_b_direction["mode"] = "true"
panel_b_direction.to_csv(TABLES / "panel_b_direction_accuracy_mean_ci.tsv", sep="\t", index=False)

# c：逐比较项方向准确率分布、四分位数、两档配对检验
panel_c_direction_wide = panel_c[[
    "comparison_id",
    "perturbation",
    "target_genes",
    "drug_class",
    "n_shared_sites",
]].copy()
panel_c_direction_wide["all_sites_direction_accuracy"] = pd.read_csv(
    TABLES / "panel_f_p100_true_vs_zero_paired.tsv", sep="\t"
)["all_direction_true"].to_numpy(float)
panel_c_direction_wide["responsive_top20_direction_accuracy"] = panel_c[
    "responsive_top20_direction_accuracy"
].to_numpy(float)
panel_c_direction_wide.to_csv(TABLES / "panel_c_direction_accuracy_distribution_wide.tsv", sep="\t", index=False)

panel_c_direction_long = panel_c_direction_wide.melt(
    id_vars=["comparison_id", "perturbation", "target_genes", "drug_class", "n_shared_sites"],
    value_vars=["all_sites_direction_accuracy", "responsive_top20_direction_accuracy"],
    var_name="site_set",
    value_name="direction_accuracy",
)
panel_c_direction_long["site_set"] = panel_c_direction_long["site_set"].replace({
    "all_sites_direction_accuracy": "all_sites",
    "responsive_top20_direction_accuracy": "responsive_top20",
})
panel_c_direction_long.to_csv(TABLES / "panel_c_direction_accuracy_distribution_long.tsv", sep="\t", index=False)

panel_c_direction_summary = (
    panel_c_direction_long.groupby("site_set", as_index=False)
    .agg(
        n=("direction_accuracy", "count"),
        mean=("direction_accuracy", "mean"),
        median=("direction_accuracy", "median"),
        iqr_low=("direction_accuracy", iqr_low),
        iqr_high=("direction_accuracy", iqr_high),
        sem=("direction_accuracy", sem),
        ci95_low=("direction_accuracy", ci_low),
        ci95_high=("direction_accuracy", ci_high),
    )
)
panel_c_direction_summary.to_csv(TABLES / "panel_c_direction_accuracy_median_iqr.tsv", sep="\t", index=False)

panel_c_wilcoxon = pd.DataFrame([{
    "comparison": "responsive_top20_direction_accuracy_vs_all_sites_direction_accuracy",
    "n_pairs": int(len(panel_c_direction_wide)),
    "all_sites_mean": float(panel_c_direction_wide["all_sites_direction_accuracy"].mean()),
    "responsive_top20_mean": float(panel_c_direction_wide["responsive_top20_direction_accuracy"].mean()),
    "mean_delta_responsive_minus_all": float(
        (panel_c_direction_wide["responsive_top20_direction_accuracy"] - panel_c_direction_wide["all_sites_direction_accuracy"]).mean()
    ),
    "wilcoxon_two_sided_p": paired_wilcoxon(
        panel_c_direction_wide["responsive_top20_direction_accuracy"],
        panel_c_direction_wide["all_sites_direction_accuracy"],
    ),
}])
panel_c_wilcoxon.to_csv(TABLES / "panel_c_direction_accuracy_wilcoxon.tsv", sep="\t", index=False)

# d：按类别的方向准确率均值和置信区间；Kruskal-Wallis
panel_d_direction = panel_d[panel_d["metric"].eq("方向准确率")].copy()
panel_d_direction.to_csv(TABLES / "panel_d_drug_class_direction_accuracy_mean_ci.tsv", sep="\t", index=False)

kruskal_rows = []
for metric_col, site_set in [
    ("all_sites_direction_accuracy", "all_sites"),
    ("responsive_top20_direction_accuracy", "responsive_top20"),
]:
    values = []
    for _, sub in panel_c_direction_wide.groupby("drug_class"):
        x = sub[metric_col].dropna().to_numpy(float)
        if len(x):
            values.append(x)
    if len(values) >= 2:
        kw = stats.kruskal(*values)
        p = float(kw.pvalue)
        h = float(kw.statistic)
    else:
        p = np.nan
        h = np.nan
    kruskal_rows.append({
        "site_set": site_set,
        "metric": "方向准确率",
        "n_classes": int(panel_c_direction_wide["drug_class"].nunique()),
        "kruskal_wallis_H": h,
        "kruskal_wallis_p": p,
    })
panel_d_kruskal = pd.DataFrame(kruskal_rows)
panel_d_kruskal.to_csv(TABLES / "panel_d_drug_class_direction_kruskal_wallis.tsv", sep="\t", index=False)

# e：16 个药物各自方向准确率
panel_e_direction = panel_e[[
    "heatmap_row_order",
    "perturbation",
    "target_genes",
    "drug_class",
    "n_comparisons",
    "n_shared_sites_median",
    "all_sites_direction_accuracy",
    "responsive_top20_direction_accuracy",
]].copy()
panel_e_direction.to_csv(TABLES / "panel_e_drug_direction_accuracy.tsv", sep="\t", index=False)

# f：true vs zero 的配对 Wilcoxon 和对角线上方数量
panel_f_stats_rows = []
for site_set, true_col, zero_col in [
    ("all_sites", "all_cosine_true", "all_cosine_zero"),
    ("responsive_top20", "responsive20_cosine_true", "responsive20_cosine_zero"),
]:
    true_v = panel_f[true_col].to_numpy(float)
    zero_v = panel_f[zero_col].to_numpy(float)
    delta = true_v - zero_v
    panel_f_stats_rows.append({
        "site_set": site_set,
        "metric": "余弦",
        "n_pairs": int(np.sum(np.isfinite(true_v) & np.isfinite(zero_v))),
        "true_mean": float(np.nanmean(true_v)),
        "zero_mean": float(np.nanmean(zero_v)),
        "mean_delta_true_minus_zero": float(np.nanmean(delta)),
        "median_delta_true_minus_zero": float(np.nanmedian(delta)),
        "n_above_diagonal_true_gt_zero": int(np.sum(delta > 0)),
        "n_on_diagonal_true_eq_zero": int(np.sum(np.isclose(delta, 0))),
        "n_below_diagonal_true_lt_zero": int(np.sum(delta < 0)),
        "wilcoxon_two_sided_p": paired_wilcoxon(true_v, zero_v),
    })
panel_f_stats = pd.DataFrame(panel_f_stats_rows)
panel_f_stats.to_csv(TABLES / "panel_f_true_vs_zero_wilcoxon_above_diagonal.tsv", sep="\t", index=False)

# 更新工作簿
xlsx = FIG / "SCP682_PPKO_V10B_Fig4_locked_P100_tables_with_direction_stats.xlsx"
sheet_map = {
    "panel_b_bars": TABLES / "panel_b_p100_true_overall_bars_cosine_direction.tsv",
    "panel_b_direction": TABLES / "panel_b_direction_accuracy_mean_ci.tsv",
    "panel_c_wide": TABLES / "panel_c_p100_true_comparison_distribution_wide.tsv",
    "panel_c_direction_long": TABLES / "panel_c_direction_accuracy_distribution_long.tsv",
    "panel_c_direction_summary": TABLES / "panel_c_direction_accuracy_median_iqr.tsv",
    "panel_c_wilcoxon": TABLES / "panel_c_direction_accuracy_wilcoxon.tsv",
    "panel_d_classes": TABLES / "panel_d_p100_true_drug_class_cosine_direction.tsv",
    "panel_d_direction": TABLES / "panel_d_drug_class_direction_accuracy_mean_ci.tsv",
    "panel_d_kruskal": TABLES / "panel_d_drug_class_direction_kruskal_wallis.tsv",
    "panel_e_heatmap": TABLES / "panel_e_p100_true_drug_heatmap_multi_metrics.tsv",
    "panel_e_direction": TABLES / "panel_e_drug_direction_accuracy.tsv",
    "panel_f_paired": TABLES / "panel_f_p100_true_vs_zero_paired.tsv",
    "panel_f_stats": TABLES / "panel_f_true_vs_zero_wilcoxon_above_diagonal.tsv",
}
with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
    for sheet, path in sheet_map.items():
        pd.read_csv(path, sep="\t").to_excel(writer, sheet_name=sheet[:31], index=False)

manifest = {
    "output": str(FIG),
    "new_tables": [
        "panel_b_direction_accuracy_mean_ci.tsv",
        "panel_c_direction_accuracy_distribution_wide.tsv",
        "panel_c_direction_accuracy_distribution_long.tsv",
        "panel_c_direction_accuracy_median_iqr.tsv",
        "panel_c_direction_accuracy_wilcoxon.tsv",
        "panel_d_drug_class_direction_accuracy_mean_ci.tsv",
        "panel_d_drug_class_direction_kruskal_wallis.tsv",
        "panel_e_drug_direction_accuracy.tsv",
        "panel_f_true_vs_zero_wilcoxon_above_diagonal.tsv",
    ],
    "workbook": str(xlsx),
}
(REPORTS / "direction_stats_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps(manifest, ensure_ascii=False, indent=2))
print("\n[b]")
print(panel_b_direction.to_string(index=False))
print("\n[c summary]")
print(panel_c_direction_summary.to_string(index=False))
print("\n[c wilcoxon]")
print(panel_c_wilcoxon.to_string(index=False))
print("\n[d kruskal]")
print(panel_d_kruskal.to_string(index=False))
print("\n[f]")
print(panel_f_stats.to_string(index=False))
