from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(r"E:\data\gongke\TCGA-TCPA")
FIG = ROOT / "02_results" / "figure_sources" / "20260528_fig4_locked_p100_v10b_cosine_direction"
TCGA = FIG / "tcga_validation"
TABLES = FIG / "tables"
REPORTS = FIG / "reports"
TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

PRIMARY_SCORE = "ppko_target_prior_abs_mean"
SCORE_LABELS = {
    "ppko_target_prior_abs_mean": "V10B 靶点先验相关位点预测扰动幅度",
    "ppko_abs_delta_top200_mean": "V10B 预测前二百位点扰动幅度",
    "ppko_observed_site_abs_mean": "V10B 外部观测位点扰动幅度",
    "control_hand_pathway_score": "手工靶点通路分数",
    "control_global_phospho_mean": "全局磷酸化均值",
    "control_global_phospho_abs_mean": "全局绝对磷酸化负荷",
    "control_mapped_marker_mean": "映射磷酸化抗体均值",
    "control_mapped_marker_abs_mean": "映射磷酸化抗体绝对均值",
    "control_target_total_mean": "靶点总蛋白均值",
    "control_observed_marker_count": "观测抗体数量",
}


def roc_curve_points(y_true, score):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s) & np.isfinite(y)
    y = y[ok]
    s = s[ok]
    thresholds = np.r_[np.inf, np.sort(np.unique(s))[::-1], -np.inf]
    rows = []
    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    for thr in thresholds:
        pred = s >= thr
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        tn = int(np.sum((~pred) & (y == 0)))
        fn = int(np.sum((~pred) & (y == 1)))
        rows.append({
            "threshold": float(thr) if np.isfinite(thr) else str(thr),
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "tpr": tp / pos,
            "fpr": fp / neg,
            "specificity": tn / neg,
        })
    return pd.DataFrame(rows)


def median_iqr_table(df, score_col):
    rows = []
    for label, sub in df.groupby("response_label"):
        x = sub[score_col].dropna().to_numpy(float)
        rows.append({
            "response_label": label,
            "n": int(len(x)),
            "mean": float(np.mean(x)) if len(x) else np.nan,
            "median": float(np.median(x)) if len(x) else np.nan,
            "iqr_low": float(np.quantile(x, 0.25)) if len(x) else np.nan,
            "iqr_high": float(np.quantile(x, 0.75)) if len(x) else np.nan,
        })
    return pd.DataFrame(rows)


pred_all = pd.read_csv(TCGA / "tables" / "all_model_patient_predictions.tsv", sep="\t")
pred_v10b = pd.read_csv(TCGA / "tables" / "v10b_300_patient_predictions.tsv", sep="\t")
model_auc = pd.read_csv(TCGA / "tables" / "model_score_auc_ci_permutation.tsv", sep="\t")
class_auc = pd.read_csv(TCGA / "tables" / "model_drug_class_summary.tsv", sep="\t")
drug_auc = pd.read_csv(TCGA / "tables" / "model_drug_summary.tsv", sep="\t")
general_controls = pd.read_csv(TCGA / "tables" / "v10b_general_score_controls.tsv", sep="\t")
general_auc = pd.read_csv(TCGA / "tables" / "v10b_general_score_control_auc.tsv", sep="\t")
random_auc = pd.read_csv(TCGA / "tables" / "v10b_random_marker_control_auc.tsv", sep="\t")

for df in [pred_all, pred_v10b, general_controls]:
    df["response_label"] = np.where(df["response_binary"].astype(int).eq(1), "反应者", "非反应者")
    df["patient_drug_id"] = (
        df["patient12"].astype(str) + "|" + df["drug_name"].astype(str) + "|" + df["sample_id"].astype(str)
    )

# g：V10B 患者层箱线图和 ROC
panel_g_box = pred_v10b[[
    "patient_drug_id",
    "patient12",
    "sample_id",
    "Cancer",
    "drug_name",
    "drug_class",
    "target_genes",
    "response_binary",
    "response_label",
    "measure_of_response",
    PRIMARY_SCORE,
    "ppko_abs_delta_top200_mean",
    "ppko_observed_site_abs_mean",
    "n_observed_projected_sites",
    "n_tcpa_markers_projected",
]].copy()
panel_g_box = panel_g_box.rename(columns={PRIMARY_SCORE: "tcga_v10b_primary_score"})
panel_g_box.to_csv(TABLES / "panel_g_tcga_v10b_patient_score_boxplot.tsv", sep="\t", index=False)

panel_g_box_summary = median_iqr_table(
    panel_g_box.rename(columns={"tcga_v10b_primary_score": PRIMARY_SCORE}), PRIMARY_SCORE
)
panel_g_box_summary.to_csv(TABLES / "panel_g_tcga_v10b_patient_score_boxplot_summary.tsv", sep="\t", index=False)

roc_tables = []
for score_col in [PRIMARY_SCORE, "ppko_abs_delta_top200_mean", "ppko_observed_site_abs_mean"]:
    roc = roc_curve_points(pred_v10b["response_binary"], pred_v10b[score_col])
    roc["score"] = score_col
    roc["score_label"] = SCORE_LABELS[score_col]
    roc_tables.append(roc)
panel_g_roc = pd.concat(roc_tables, ignore_index=True)
panel_g_roc.to_csv(TABLES / "panel_g_tcga_v10b_roc_curve.tsv", sep="\t", index=False)

panel_g_auc = model_auc[
    model_auc["model_name"].eq("v10b_300")
    & model_auc["score"].isin([PRIMARY_SCORE, "ppko_abs_delta_top200_mean", "ppko_observed_site_abs_mean"])
].copy()
panel_g_auc["score_label"] = panel_g_auc["score"].map(SCORE_LABELS)
panel_g_auc.to_csv(TABLES / "panel_g_tcga_v10b_auc_ci_permutation.tsv", sep="\t", index=False)

# h：V10B 与旧 V10 的模型分数对照
panel_h = model_auc.copy()
panel_h["score_label"] = panel_h["score"].map(SCORE_LABELS).fillna(panel_h["score"])
panel_h.to_csv(TABLES / "panel_h_tcga_model_auc_comparison.tsv", sep="\t", index=False)

panel_h_primary = panel_h[panel_h["score"].eq(PRIMARY_SCORE)].copy()
panel_h_primary.to_csv(TABLES / "panel_h_tcga_primary_score_v10_vs_v10b.tsv", sep="\t", index=False)

# i：普通评分对照和随机抗体对照
panel_i_controls = general_auc.copy()
panel_i_controls["score_label"] = panel_i_controls["score"].map(SCORE_LABELS).fillna(panel_i_controls["score"])
panel_i_controls["score_group"] = np.where(
    panel_i_controls["score"].str.startswith("ppko_"), "SCP682-PPKO V10B", "普通评分"
)
panel_i_controls.to_csv(TABLES / "panel_i_tcga_general_score_control_auc.tsv", sep="\t", index=False)

panel_i_controls_box = general_controls[[
    "patient_drug_id",
    "patient12",
    "sample_id",
    "Cancer",
    "drug_name",
    "drug_class",
    "target_genes",
    "response_binary",
    "response_label",
    "measure_of_response",
    PRIMARY_SCORE,
    "control_hand_pathway_score",
    "control_global_phospho_mean",
    "control_mapped_marker_mean",
    "control_target_total_mean",
]].copy()
panel_i_controls_box.to_csv(TABLES / "panel_i_tcga_general_score_patient_values.tsv", sep="\t", index=False)

panel_i_random = random_auc.copy()
panel_i_random.to_csv(TABLES / "panel_i_tcga_random_marker_control_auc.tsv", sep="\t", index=False)

# 可选分层表：药物类别和药物层面
class_rows = []
for _, row in class_auc[class_auc["model_name"].eq("v10b_300")].iterrows():
    class_rows.append({
        "model_name": row["model_name"],
        "drug_class": row["drug_class"],
        "n": row["n"],
        "n_responder": row["n_responder"],
        "n_non_responder": row["n_non_responder"],
        "primary_auc": row[f"{PRIMARY_SCORE}_auc"],
        "primary_mean_responder": row[f"{PRIMARY_SCORE}_mean_responder"],
        "primary_mean_non_responder": row[f"{PRIMARY_SCORE}_mean_non_responder"],
    })
panel_tcga_class = pd.DataFrame(class_rows)
panel_tcga_class.to_csv(TABLES / "panel_tcga_v10b_drug_class_auc.tsv", sep="\t", index=False)

drug_rows = []
for _, row in drug_auc[drug_auc["model_name"].eq("v10b_300")].iterrows():
    drug_rows.append({
        "model_name": row["model_name"],
        "drug_name": row["drug_name"],
        "n": row["n"],
        "n_responder": row["n_responder"],
        "n_non_responder": row["n_non_responder"],
        "primary_auc": row[f"{PRIMARY_SCORE}_auc"],
        "primary_mean_responder": row[f"{PRIMARY_SCORE}_mean_responder"],
        "primary_mean_non_responder": row[f"{PRIMARY_SCORE}_mean_non_responder"],
    })
panel_tcga_drug = pd.DataFrame(drug_rows)
panel_tcga_drug.to_csv(TABLES / "panel_tcga_v10b_drug_auc.tsv", sep="\t", index=False)

# 合并工作簿
xlsx = FIG / "SCP682_PPKO_V10B_TCGA_validation_plot_tables.xlsx"
sheet_map = {
    "g_box": TABLES / "panel_g_tcga_v10b_patient_score_boxplot.tsv",
    "g_box_summary": TABLES / "panel_g_tcga_v10b_patient_score_boxplot_summary.tsv",
    "g_roc": TABLES / "panel_g_tcga_v10b_roc_curve.tsv",
    "g_auc": TABLES / "panel_g_tcga_v10b_auc_ci_permutation.tsv",
    "h_model_auc": TABLES / "panel_h_tcga_model_auc_comparison.tsv",
    "h_primary": TABLES / "panel_h_tcga_primary_score_v10_vs_v10b.tsv",
    "i_controls_auc": TABLES / "panel_i_tcga_general_score_control_auc.tsv",
    "i_controls_values": TABLES / "panel_i_tcga_general_score_patient_values.tsv",
    "i_random": TABLES / "panel_i_tcga_random_marker_control_auc.tsv",
    "tcga_class": TABLES / "panel_tcga_v10b_drug_class_auc.tsv",
    "tcga_drug": TABLES / "panel_tcga_v10b_drug_auc.tsv",
}
with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
    for sheet, path in sheet_map.items():
        pd.read_csv(path, sep="\t").to_excel(writer, sheet_name=sheet[:31], index=False)

manifest = {
    "tcga_source_dir": str(TCGA),
    "figure_source_dir": str(FIG),
    "n_patient_drug_records": int(len(pred_v10b)),
    "n_responder": int(pred_v10b["response_binary"].eq(1).sum()),
    "n_non_responder": int(pred_v10b["response_binary"].eq(0).sum()),
    "primary_score": PRIMARY_SCORE,
    "primary_score_label": SCORE_LABELS[PRIMARY_SCORE],
    "new_tables": [path.name for path in sheet_map.values()],
    "workbook": str(xlsx),
}
(REPORTS / "tcga_validation_plot_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)

print(json.dumps(manifest, ensure_ascii=False, indent=2))
print(panel_g_auc.to_string(index=False))
print(panel_i_controls[["score", "auc", "bootstrap_ci_low", "bootstrap_ci_high", "permutation_p_right"]].to_string(index=False))
