import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, pearsonr, spearmanr


PALETTE = {
    "pT1a": "#92B1D9",
    "pT3a": "#F6C8B6",
    "Negative": "#92B1D9",
    "Positive": "#F6C8B6",
    "Tumor sample all cells": "#92B1D9",
    "Malignant tumor cells": "#F6C8B6",
}

TARGET = "predicted_RPS6_pS235_S236"


def ensure_dirs(out_dir: Path) -> None:
    for name in ["tables", "figures", "reports"]:
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def clean_stage(value) -> str:
    if pd.isna(value):
        return np.nan
    text = str(value)
    if text.startswith("H_"):
        text = text[2:]
    return text if text in {"pT1a", "pT3a"} else np.nan


def clean_necrosis(value) -> str:
    if pd.isna(value):
        return np.nan
    text = str(value)
    return text if text in {"Negative", "Positive"} else np.nan


def first_non_null(series: pd.Series):
    values = series.dropna().astype(str)
    values = values[values != ""]
    if values.empty:
        return np.nan
    return values.iloc[0]


def patient_clinic_from_h5ad(h5ad_path: Path) -> pd.DataFrame:
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    required = [
        "patient",
        "pT stage",
        "necrosis",
        "tumor size, mm",
        "age",
        "sex",
        "operation",
        "tissue",
    ]
    missing = [col for col in required if col not in obs.columns]
    if missing:
        raise ValueError(f"Missing h5ad obs columns: {missing}")

    rows = []
    for patient_id, sub in obs.groupby("patient", observed=False):
        tumor = sub[sub["tissue"].astype(str) == "Tumor"]
        source = tumor if not tumor.empty else sub

        stage_values = source["pT stage"].map(clean_stage).dropna().unique()
        necrosis_values = source["necrosis"].map(clean_necrosis).dropna().unique()
        tumor_size_values = pd.to_numeric(
            source["tumor size, mm"].replace("Healthy", np.nan), errors="coerce"
        ).dropna()

        rows.append(
            {
                "patient_id": str(patient_id),
                "pT_stage": stage_values[0] if len(stage_values) else np.nan,
                "necrosis": necrosis_values[0] if len(necrosis_values) else np.nan,
                "tumor_size_mm": float(tumor_size_values.iloc[0])
                if not tumor_size_values.empty
                else np.nan,
                "age": first_non_null(source["age"].replace("Healthy", np.nan)),
                "sex": first_non_null(source["sex"].replace("Healthy", np.nan)),
                "operation": first_non_null(source["operation"].replace("Healthy", np.nan)),
                "n_cells_total_h5ad": int(len(sub)),
                "n_tumor_sample_cells_h5ad": int(len(tumor)),
                "n_healthy_cells_h5ad": int((sub["tissue"].astype(str) == "Healthy").sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("patient_id")


def add_clinical_columns(cell: pd.DataFrame, clinic: pd.DataFrame) -> pd.DataFrame:
    merged = cell.merge(
        clinic[
            [
                "patient_id",
                "pT_stage",
                "necrosis",
                "tumor_size_mm",
                "age",
                "sex",
                "operation",
            ]
        ],
        on="patient_id",
        how="left",
    )
    merged["stage_clean_from_cell"] = merged["stage"].map(clean_stage)
    merged["necrosis_from_cell"] = merged["survival"].map(clean_necrosis)
    return merged


def scoped_cells(cell: pd.DataFrame) -> dict:
    tumor_sample = cell[cell["tissue"].astype(str) == "Tumor"].copy()
    malignant = cell[cell["malignant_status"].astype(str) == "malignant_inferred"].copy()
    return {
        "Tumor sample all cells": tumor_sample,
        "Malignant tumor cells": malignant,
    }


def patient_scope_summary(data: pd.DataFrame, scope: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    rows = []
    for patient_id, sub in data.groupby("patient_id", observed=False):
        rows.append(
            {
                "scope": scope,
                "patient_id": patient_id,
                "pT_stage": first_non_null(sub["pT_stage"]),
                "necrosis": first_non_null(sub["necrosis"]),
                "tumor_size_mm": pd.to_numeric(sub["tumor_size_mm"], errors="coerce").dropna().iloc[0]
                if pd.to_numeric(sub["tumor_size_mm"], errors="coerce").notna().any()
                else np.nan,
                "n_cells": int(len(sub)),
                "median_predicted_RPS6_pS235_S236": float(sub[TARGET].median()),
                "mean_predicted_RPS6_pS235_S236": float(sub[TARGET].mean()),
                "median_mTOR_S6_score": float(sub["mTOR_S6_score"].median()),
                "median_RPS6_mRNA": float(sub["RPS6_mRNA"].median()),
            }
        )
    return pd.DataFrame(rows)


def mwu_pvalue(values_a: pd.Series, values_b: pd.Series):
    values_a = pd.to_numeric(values_a, errors="coerce").dropna()
    values_b = pd.to_numeric(values_b, errors="coerce").dropna()
    if len(values_a) == 0 or len(values_b) == 0:
        return np.nan
    return float(mannwhitneyu(values_a, values_b, alternative="two-sided").pvalue)


def group_stats(scopes: dict, patient_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    configs = [
        ("pT_stage", ["pT1a", "pT3a"]),
        ("necrosis", ["Negative", "Positive"]),
    ]
    for scope, data in scopes.items():
        for variable, levels in configs:
            sub = data[data[variable].isin(levels)].copy()
            patient_sub = patient_summary[
                (patient_summary["scope"] == scope) & (patient_summary[variable].isin(levels))
            ].copy()

            cell_p = mwu_pvalue(
                sub.loc[sub[variable] == levels[0], TARGET],
                sub.loc[sub[variable] == levels[1], TARGET],
            )
            patient_p = mwu_pvalue(
                patient_sub.loc[
                    patient_sub[variable] == levels[0],
                    "median_predicted_RPS6_pS235_S236",
                ],
                patient_sub.loc[
                    patient_sub[variable] == levels[1],
                    "median_predicted_RPS6_pS235_S236",
                ],
            )

            for level in levels:
                part = sub[sub[variable] == level]
                patient_part = patient_sub[patient_sub[variable] == level]
                rows.append(
                    {
                        "scope": scope,
                        "variable": variable,
                        "group": level,
                        "n_cells": int(len(part)),
                        "n_patients": int(patient_part["patient_id"].nunique()),
                        "cell_median": float(part[TARGET].median()) if len(part) else np.nan,
                        "cell_mean": float(part[TARGET].mean()) if len(part) else np.nan,
                        "cell_q25": float(part[TARGET].quantile(0.25)) if len(part) else np.nan,
                        "cell_q75": float(part[TARGET].quantile(0.75)) if len(part) else np.nan,
                        "patient_median_of_medians": float(
                            patient_part["median_predicted_RPS6_pS235_S236"].median()
                        )
                        if len(patient_part)
                        else np.nan,
                        "cell_level_mannwhitney_p": cell_p,
                        "patient_level_mannwhitney_p": patient_p,
                    }
                )
    return pd.DataFrame(rows)


def tumor_size_correlation(patient_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, sub in patient_summary.groupby("scope", observed=False):
        part = sub.dropna(subset=["tumor_size_mm", "median_predicted_RPS6_pS235_S236"])
        if len(part) >= 3:
            spearman = spearmanr(
                part["tumor_size_mm"], part["median_predicted_RPS6_pS235_S236"]
            )
            pearson = pearsonr(
                part["tumor_size_mm"], part["median_predicted_RPS6_pS235_S236"]
            )
            spearman_r = float(spearman.statistic)
            spearman_p = float(spearman.pvalue)
            pearson_r = float(pearson.statistic)
            pearson_p = float(pearson.pvalue)
        else:
            spearman_r = spearman_p = pearson_r = pearson_p = np.nan
        rows.append(
            {
                "scope": scope,
                "n_patients": int(len(part)),
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
            }
        )
    return pd.DataFrame(rows)


def plot_group_box(
    scopes: dict,
    patient_summary: pd.DataFrame,
    variable: str,
    levels: list[str],
    output_prefix: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    y_candidates = []
    for scope, data in scopes.items():
        sub_for_limit = data[data[variable].isin(levels)]
        if not sub_for_limit.empty:
            y_candidates.append(float(np.nanpercentile(sub_for_limit[TARGET], 99.0)))
        patient_for_limit = patient_summary[
            (patient_summary["scope"] == scope) & (patient_summary[variable].isin(levels))
        ]
        if not patient_for_limit.empty:
            y_candidates.append(
                float(patient_for_limit["median_predicted_RPS6_pS235_S236"].max())
            )
    y_upper = max(y_candidates) * 1.18 if y_candidates else 1.0
    y_upper = max(y_upper, 120.0)
    for ax, (scope, data) in zip(axes, scopes.items()):
        sub = data[data[variable].isin(levels)].copy()
        patient_sub = patient_summary[
            (patient_summary["scope"] == scope) & (patient_summary[variable].isin(levels))
        ].copy()
        sns.boxplot(
            data=sub,
            x=variable,
            y=TARGET,
            order=levels,
            palette=[PALETTE[level] for level in levels],
            fliersize=0,
            linewidth=1.0,
            ax=ax,
        )
        sns.stripplot(
            data=patient_sub,
            x=variable,
            y="median_predicted_RPS6_pS235_S236",
            order=levels,
            color="#2B2B2B",
            size=5,
            jitter=0.10,
            ax=ax,
        )
        for idx, level in enumerate(levels):
            n_cells = int((sub[variable] == level).sum())
            n_patients = int(patient_sub.loc[patient_sub[variable] == level, "patient_id"].nunique())
            ax.text(
                idx,
                y_upper * 0.025,
                f"cells={n_cells}\npatients={n_patients}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )
        ax.set_ylim(0, y_upper)
        ax.set_title(scope, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Predicted RPS6 pS235/S236" if ax is axes[0] else "")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_prefix)


def plot_tumor_size(patient_summary: pd.DataFrame, corr: pd.DataFrame, output_prefix: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    all_points = patient_summary.dropna(
        subset=["tumor_size_mm", "median_predicted_RPS6_pS235_S236"]
    )
    if not all_points.empty:
        ymin = float(all_points["median_predicted_RPS6_pS235_S236"].min())
        ymax = float(all_points["median_predicted_RPS6_pS235_S236"].max())
        yrange = max(ymax - ymin, 10.0)
        y_limits = (ymin - 0.15 * yrange, ymax + 0.20 * yrange)
    else:
        y_limits = (0.0, 1.0)
    for ax, scope in zip(axes, ["Tumor sample all cells", "Malignant tumor cells"]):
        sub = patient_summary[
            (patient_summary["scope"] == scope)
            & patient_summary["tumor_size_mm"].notna()
            & patient_summary["median_predicted_RPS6_pS235_S236"].notna()
        ].copy()
        ax.scatter(
            sub["tumor_size_mm"],
            sub["median_predicted_RPS6_pS235_S236"],
            s=48,
            color=PALETTE[scope],
            edgecolor="#222222",
            linewidth=0.6,
        )
        if len(sub) >= 3:
            sns.regplot(
                data=sub,
                x="tumor_size_mm",
                y="median_predicted_RPS6_pS235_S236",
                scatter=False,
                color="#555555",
                ci=None,
                line_kws={"linewidth": 1.0},
                ax=ax,
            )
        for _, row in sub.iterrows():
            ax.text(
                row["tumor_size_mm"] + 0.8,
                row["median_predicted_RPS6_pS235_S236"],
                row["patient_id"],
                fontsize=7,
                va="center",
            )
        rrow = corr[corr["scope"] == scope]
        if not rrow.empty:
            text = f"Spearman r={rrow['spearman_r'].iloc[0]:.2f}, p={rrow['spearman_p'].iloc[0]:.3g}"
            ax.text(
                0.97,
                0.05,
                text,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )
        ax.set_title(scope, fontsize=11)
        ax.set_xlabel("Tumor size, mm")
        ax.set_ylabel("Patient median predicted RPS6 pS235/S236" if ax is axes[0] else "")
        ax.set_ylim(*y_limits)
        ax.grid(color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Tumor size correlation", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_prefix)


def save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    for ext in ["png", "pdf", "svg"]:
        path = output_prefix.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_path: Path,
    patient_tumor_summary: pd.DataFrame,
    stats: pd.DataFrame,
    corr: pd.DataFrame,
) -> None:
    lines = [
        "# KIRC RPS6 临床病理分层补充分析",
        "",
        "## 数据字段",
        "",
        "- 本数据集没有 OS、PFS、DSS、死亡状态或随访时间。",
        "- 可用临床病理字段包括 pT stage、necrosis、tumor size, mm、age、sex、operation。",
        "- 原输出表中的 survival 字段实际来自原始 necrosis；后续解释应写 necrosis。",
        "- 肿瘤细胞限定为 malignant_status == malignant_inferred。",
        "",
        "## 患者级肿瘤细胞汇总",
        "",
        f"- 有 malignant_inferred 肿瘤细胞的患者数: {patient_tumor_summary['patient_id'].nunique()}",
        "",
        "## pT stage",
        "",
    ]

    def stat_line(variable: str, scope: str) -> str:
        sub = stats[(stats["variable"] == variable) & (stats["scope"] == scope)]
        parts = []
        for _, row in sub.iterrows():
            parts.append(
                f"{row['group']}: cells={int(row['n_cells'])}, patients={int(row['n_patients'])}, "
                f"cell median={row['cell_median']:.3f}, patient median={row['patient_median_of_medians']:.3f}"
            )
        p = sub["patient_level_mannwhitney_p"].dropna()
        p_text = f"; patient-level p={p.iloc[0]:.3g}" if not p.empty else ""
        return f"- {scope}: " + "; ".join(parts) + p_text

    lines.append(stat_line("pT_stage", "Tumor sample all cells"))
    lines.append(stat_line("pT_stage", "Malignant tumor cells"))
    lines.extend(["", "## necrosis", ""])
    lines.append(stat_line("necrosis", "Tumor sample all cells"))
    lines.append(stat_line("necrosis", "Malignant tumor cells"))
    lines.extend(["", "## tumor size, mm", ""])
    for _, row in corr.iterrows():
        lines.append(
            f"- {row['scope']}: n={int(row['n_patients'])}, "
            f"Spearman r={row['spearman_r']:.3f}, p={row['spearman_p']:.3g}"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- 该补充分析只能作为病理预后相关分层，不能替代真正生存分析。",
            "- 细胞级检验受同一患者内细胞重复采样影响，患者级结果更适合写入正文或图注。",
            "- predicted RPS6 pS235/S236 是模型推断状态，不是直接测得的磷酸化。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-table", required=True)
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cell_table = Path(args.cell_table)
    h5ad_path = Path(args.h5ad)
    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)

    cell = pd.read_csv(cell_table, sep="\t")
    clinic = patient_clinic_from_h5ad(h5ad_path)
    merged = add_clinical_columns(cell, clinic)
    scopes = scoped_cells(merged)

    patient_summaries = []
    for scope, data in scopes.items():
        patient_summaries.append(patient_scope_summary(data, scope))
    patient_summary = pd.concat(patient_summaries, ignore_index=True)

    patient_tumor_summary = patient_summary[
        patient_summary["scope"] == "Malignant tumor cells"
    ].copy()
    patient_tumor_summary = patient_tumor_summary[
        [
            "patient_id",
            "pT_stage",
            "necrosis",
            "tumor_size_mm",
            "n_cells",
            "median_predicted_RPS6_pS235_S236",
            "mean_predicted_RPS6_pS235_S236",
            "median_mTOR_S6_score",
            "median_RPS6_mRNA",
        ]
    ].rename(columns={"n_cells": "n_malignant_tumor_cells"})

    stats = group_stats(scopes, patient_summary)
    corr = tumor_size_correlation(patient_summary)

    clinic.to_csv(out_dir / "tables" / "kirc_rps6_patient_clinicopathology_metadata.tsv", sep="\t", index=False)
    patient_summary.to_csv(out_dir / "tables" / "kirc_rps6_patient_scope_summary.tsv", sep="\t", index=False)
    patient_tumor_summary.to_csv(out_dir / "tables" / "kirc_rps6_patient_tumor_summary.tsv", sep="\t", index=False)
    stats.to_csv(out_dir / "tables" / "kirc_rps6_clinicopathology_group_stats.tsv", sep="\t", index=False)
    corr.to_csv(out_dir / "tables" / "kirc_rps6_tumor_size_patient_correlation.tsv", sep="\t", index=False)

    plot_group_box(
        scopes,
        patient_summary,
        "pT_stage",
        ["pT1a", "pT3a"],
        out_dir / "figures" / "kirc_rps6_pt_stage_all_vs_tumor",
        "pT stage",
    )
    plot_group_box(
        scopes,
        patient_summary,
        "necrosis",
        ["Negative", "Positive"],
        out_dir / "figures" / "kirc_rps6_necrosis_all_vs_tumor",
        "Necrosis",
    )
    plot_tumor_size(
        patient_summary,
        corr,
        out_dir / "figures" / "kirc_rps6_tumor_size_patient_correlation_all_vs_tumor",
    )

    write_report(
        out_dir / "reports" / "clinicopathology_summary.md",
        patient_tumor_summary,
        stats,
        corr,
    )

    run_meta = {
        "cell_table": str(cell_table),
        "h5ad": str(h5ad_path),
        "out_dir": str(out_dir),
        "n_cells_loaded": int(len(cell)),
        "n_patients_clinical": int(clinic["patient_id"].nunique()),
        "n_patients_with_malignant_tumor_cells": int(patient_tumor_summary["patient_id"].nunique()),
        "target": TARGET,
        "clinical_fields": ["pT_stage", "necrosis", "tumor_size_mm", "age", "sex", "operation"],
    }
    (out_dir / "clinicopathology_run_metadata.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
