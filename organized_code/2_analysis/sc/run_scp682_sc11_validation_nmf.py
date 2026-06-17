#!/usr/bin/env python
# 模型: SCP682-SC
# 作用: 对外部验证队列的细胞/样本 × 磷酸化读数矩阵做 NMF，并比较预测模块与实测模块。
# 输入: paper_materials_SCP682_SC11/02_data_tables/external_predicted_observed 中的 predicted-observed 长表。
# 输出: 04_figure_source_data/sc11_validation_nmf_v1 下的 NMF 图源表、总览图和报告。
# 依赖: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn。
# 原始路径: remote_scripts/run_scp682_sc11_validation_nmf.py
# 原始版本: 2026-05-27 validation NMF v1

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "paper_materials_SCP682_SC11").exists():
            return candidate
        if candidate.name == "paper_materials_SCP682_SC11":
            return candidate.parent
    return Path.cwd()


def snake_case(name: str) -> str:
    text = str(name)
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or "column"


ROOT = find_project_root()
PM = ROOT / "paper_materials_SCP682_SC11"
PRED_DIR = PM / "02_data_tables" / "external_predicted_observed"
OUT = PM / "04_figure_source_data" / "sc11_validation_nmf_v1"
FIG = OUT / "figures"
SD = OUT / "source_data"
REP = OUT / "reports"

for folder in [OUT, FIG, SD, REP]:
    folder.mkdir(parents=True, exist_ok=True)


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
    }
)

CMAP_SEQ = LinearSegmentedColormap.from_list("nmf_seq", ["#F7F7F7", "#C8D7D2", "#6CBFB5", "#1F3A5F"], N=256)
CMAP_DIV = LinearSegmentedColormap.from_list("nmf_div", ["#92B1D9", "#F7F7F7", "#D98973"], N=256)


COHORT_DISPLAY = {
    "gse300551_iccite_plex_kinase_2025": "GSE300551",
    "phospho_seq_blair_2025_phospho_multi": "Blair",
    "vivo_seq_th17_2025": "Vivo-seq Th17",
    "signal_seq_gse256403_hela_2024": "SIGNAL-seq HeLa",
    "signal_seq_gse256404_pdo_caf_2024": "SIGNAL-seq PDO/CAF",
}


def savefig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=350, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_table(df: pd.DataFrame, name: str, desc: str) -> Path:
    out = df.copy()
    out.columns = [snake_case(c) for c in out.columns]
    out = out.fillna("NA")
    path = SD / name
    out.to_csv(path, sep="\t", index=False, na_rep="NA")
    path.with_suffix(".md").write_text(desc + "\n", encoding="utf-8")
    return path


def shorten_target(target: str, n: int = 18) -> str:
    text = str(target).replace("_pSitePending", "").replace("MAPK1_MAPK3", "ERK1/2")
    text = text.replace("RPS6_pSitePending", "RPS6")
    text = text.replace("RELA_pSitePending_93H1", "RELA_93H1")
    return text if len(text) <= n else text[: n - 1] + "."


def normalize_columns(mat: pd.DataFrame) -> pd.DataFrame:
    """Per-target robust min-max transform. This keeps NMF nonnegative without treating missing values as zero."""
    out = pd.DataFrame(index=mat.index)
    for col in mat.columns:
        x = pd.to_numeric(mat[col], errors="coerce").astype(float)
        med = float(np.nanmedian(x)) if np.isfinite(np.nanmedian(x)) else 0.0
        x = x.fillna(med)
        lo = float(np.nanquantile(x, 0.01))
        hi = float(np.nanquantile(x, 0.99))
        if not np.isfinite(lo):
            lo = float(np.nanmin(x))
        if not np.isfinite(hi):
            hi = float(np.nanmax(x))
        if hi <= lo:
            out[col] = 0.0
        else:
            y = x.clip(lo, hi)
            out[col] = (y - lo) / (hi - lo)
    return out.astype(np.float32)


def select_k(n_targets: int) -> int:
    if n_targets < 2:
        return 0
    if n_targets < 5:
        return 2
    return 3


def fit_nmf(mat: pd.DataFrame, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    x = mat.to_numpy(dtype=np.float32)
    model = NMF(
        n_components=k,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=1000,
        tol=1e-4,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        w = model.fit_transform(x)
    h = model.components_
    denom = float(np.linalg.norm(x, ord="fro"))
    rel_error = float(model.reconstruction_err_ / denom) if denom > 0 else np.nan
    return w, h, rel_error


def normalize_h(h: np.ndarray) -> np.ndarray:
    denom = h.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return h / denom


def top_targets(targets: list[str], loadings: np.ndarray, top_n: int = 5) -> str:
    order = np.argsort(-loadings)[:top_n]
    return ";".join(f"{targets[i]}:{loadings[i]:.3f}" for i in order)


def process_file(path: Path, seed: int = 682) -> dict[str, pd.DataFrame | list[dict] | str]:
    usecols = [
        "cohort_id",
        "cell_id",
        "row_index",
        "target_id",
        "predicted_raw_scale",
        "observed_raw_scale",
    ]
    df = pd.read_csv(path, sep="\t", usecols=usecols)
    cohort = str(df["cohort_id"].iloc[0])
    display = COHORT_DISPLAY.get(cohort, cohort)

    obs = df.pivot(index=["cell_id", "row_index"], columns="target_id", values="observed_raw_scale")
    pred = df.pivot(index=["cell_id", "row_index"], columns="target_id", values="predicted_raw_scale")
    common_targets = [t for t in obs.columns if t in pred.columns]

    target_rows = []
    usable_targets = []
    for target in common_targets:
        x = pd.to_numeric(obs[target], errors="coerce")
        unique_count = int(x.nunique(dropna=True))
        variance = float(x.var()) if x.notna().sum() > 1 else np.nan
        usable = bool(unique_count > 1 and np.isfinite(variance) and variance > 1e-12)
        target_rows.append(
            {
                "cohort_id": cohort,
                "display_cohort": display,
                "target_id": target,
                "observed_unique_values": unique_count,
                "observed_variance": variance,
                "used_for_nmf": usable,
            }
        )
        if usable:
            usable_targets.append(target)

    k = select_k(len(usable_targets))
    if k == 0:
        return {
            "cohort_id": cohort,
            "summary_rows": [
                {
                    "cohort_id": cohort,
                    "display_cohort": display,
                    "status": "skipped",
                    "n_cells": int(obs.shape[0]),
                    "n_targets_total": int(len(common_targets)),
                    "n_targets_used": int(len(usable_targets)),
                    "n_components": 0,
                    "observed_relative_reconstruction_error": "NA",
                    "predicted_relative_reconstruction_error": "NA",
                    "mean_matched_h_cosine": "NA",
                    "mean_matched_w_spearman": "NA",
                    "skipped_reason": "usable_targets_less_than_2",
                    "source_file": path.name,
                }
            ],
            "target_rows": target_rows,
            "h_rows": [],
            "w_df": pd.DataFrame(),
            "alignment_rows": [],
            "top_rows": [],
        }

    obs_norm = normalize_columns(obs[usable_targets])
    pred_norm = normalize_columns(pred[usable_targets])
    w_obs, h_obs, err_obs = fit_nmf(obs_norm, k, seed)
    w_pred, h_pred, err_pred = fit_nmf(pred_norm, k, seed + 17)
    h_obs_n = normalize_h(h_obs)
    h_pred_n = normalize_h(h_pred)

    cosine = 1.0 - cdist(h_obs_n, h_pred_n, metric="cosine")
    row_ind, col_ind = linear_sum_assignment(-cosine)

    h_rows = []
    top_rows = []
    for matrix_type, h, h_n in [
        ("observed", h_obs, h_obs_n),
        ("predicted", h_pred, h_pred_n),
    ]:
        for comp_idx in range(k):
            comp = f"nmf{comp_idx + 1}"
            top_rows.append(
                {
                    "cohort_id": cohort,
                    "display_cohort": display,
                    "matrix_type": matrix_type,
                    "component": comp,
                    "top_targets": top_targets(usable_targets, h_n[comp_idx]),
                }
            )
            for j, target in enumerate(usable_targets):
                h_rows.append(
                    {
                        "cohort_id": cohort,
                        "display_cohort": display,
                        "matrix_type": matrix_type,
                        "component": comp,
                        "target_id": target,
                        "loading": float(h[comp_idx, j]),
                        "normalized_loading": float(h_n[comp_idx, j]),
                    }
                )

    alignment_rows = []
    w_cols = []
    for i, j in zip(row_ind, col_ind):
        rho = spearmanr(w_obs[:, i], w_pred[:, j], nan_policy="omit").statistic
        alignment_rows.append(
            {
                "cohort_id": cohort,
                "display_cohort": display,
                "observed_component": f"nmf{i + 1}",
                "predicted_component": f"nmf{j + 1}",
                "h_cosine": float(cosine[i, j]),
                "w_spearman": float(rho) if np.isfinite(rho) else "NA",
                "observed_top_targets": top_targets(usable_targets, h_obs_n[i]),
                "predicted_top_targets": top_targets(usable_targets, h_pred_n[j]),
            }
        )

    cells = obs_norm.index.to_frame(index=False)
    w_df = pd.DataFrame(
        {
            "cohort_id": cohort,
            "display_cohort": display,
            "cell_id": cells["cell_id"].astype(str),
            "row_index": cells["row_index"].astype(int),
        }
    )
    for i in range(k):
        w_df[f"observed_nmf{i + 1}"] = w_obs[:, i]
        w_df[f"predicted_nmf{i + 1}"] = w_pred[:, i]
        w_cols.extend([f"observed_nmf{i + 1}", f"predicted_nmf{i + 1}"])

    valid_w = [r["w_spearman"] for r in alignment_rows if isinstance(r["w_spearman"], float)]
    summary_rows = [
        {
            "cohort_id": cohort,
            "display_cohort": display,
            "status": "complete",
            "n_cells": int(obs_norm.shape[0]),
            "n_targets_total": int(len(common_targets)),
            "n_targets_used": int(len(usable_targets)),
            "n_components": int(k),
            "observed_relative_reconstruction_error": float(err_obs),
            "predicted_relative_reconstruction_error": float(err_pred),
            "mean_matched_h_cosine": float(np.mean([cosine[i, j] for i, j in zip(row_ind, col_ind)])),
            "mean_matched_w_spearman": float(np.mean(valid_w)) if valid_w else "NA",
            "skipped_reason": "NA",
            "source_file": path.name,
        }
    ]

    return {
        "cohort_id": cohort,
        "summary_rows": summary_rows,
        "target_rows": target_rows,
        "h_rows": h_rows,
        "w_df": w_df,
        "alignment_rows": alignment_rows,
        "top_rows": top_rows,
    }


def draw_loadings_heatmap(h_long: pd.DataFrame, summary: pd.DataFrame) -> None:
    complete = summary[summary["status"] == "complete"]["cohort_id"].tolist()
    n = len(complete)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(10.2, 2.35 * n + 1.2), squeeze=False)
    for row, cohort in enumerate(complete):
        display = summary.loc[summary["cohort_id"] == cohort, "display_cohort"].iloc[0]
        for col, matrix_type in enumerate(["observed", "predicted"]):
            ax = axes[row, col]
            sub = h_long[(h_long["cohort_id"] == cohort) & (h_long["matrix_type"] == matrix_type)]
            mat = sub.pivot(index="component", columns="target_id", values="normalized_loading")
            mat = mat.loc[sorted(mat.index, key=lambda x: int(x.replace("nmf", "")))]
            sns.heatmap(
                mat,
                ax=ax,
                cmap=CMAP_SEQ,
                vmin=0,
                vmax=float(max(0.25, h_long["normalized_loading"].quantile(0.98))),
                cbar=row == 0 and col == 1,
                cbar_kws={"label": "normalized loading", "shrink": 0.5},
                linewidths=0.2,
                linecolor="#F0F0F0",
            )
            ax.set_title(f"{display} · {matrix_type}", pad=10)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels([shorten_target(t, 13) for t in mat.columns], rotation=90, ha="center", va="top")
            ax.tick_params(axis="x", pad=2)
    fig.suptitle("Validation cohort NMF phospho modules", y=0.985, fontsize=11)
    fig.subplots_adjust(left=0.075, right=0.96, top=0.92, bottom=0.08, hspace=1.15, wspace=0.22)
    savefig(fig, "validation_nmf_loadings_heatmap")


def draw_alignment(alignment: pd.DataFrame, summary: pd.DataFrame) -> None:
    if alignment.empty:
        return
    align = alignment.copy()
    align["component_pair"] = align["display_cohort"] + " " + align["observed_component"] + "→" + align["predicted_component"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, max(3.2, 0.31 * len(align))), sharey=False)
    y = np.arange(len(align))
    axes[0].barh(y, align["h_cosine"], color="#6CBFB5", edgecolor="white")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(align["component_pair"])
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("H loading cosine")
    axes[0].axvline(0.5, color="#A0A0A0", lw=0.7, ls="--")
    ws = pd.to_numeric(align["w_spearman"], errors="coerce")
    axes[1].barh(y, ws, color="#D4A56B", edgecolor="white")
    axes[1].set_xlim(-1, 1)
    axes[1].set_xlabel("cell score Spearman")
    axes[1].axvline(0, color="#606060", lw=0.7)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    fig.suptitle("Observed-predicted NMF component alignment", y=0.995, fontsize=11)
    fig.subplots_adjust(left=0.34, right=0.98, top=0.9, bottom=0.14, wspace=0.18)
    savefig(fig, "validation_nmf_component_alignment")


def draw_summary(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    complete = summary[summary["status"] == "complete"].copy()
    if complete.empty:
        return
    x = np.arange(len(complete))
    ax.bar(x - 0.18, complete["mean_matched_h_cosine"], width=0.36, color="#6CBFB5", label="loading cosine")
    ax.bar(x + 0.18, complete["mean_matched_w_spearman"], width=0.36, color="#D4A56B", label="cell score Spearman")
    ax.set_xticks(x)
    labels = [
        f"{row['display_cohort']}\n{int(row['n_targets_used'])} targets, k={int(row['n_components'])}"
        for _, row in complete.iterrows()
    ]
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("matched component agreement")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    savefig(fig, "validation_nmf_summary")


def main() -> int:
    files = sorted(
        p
        for p in PRED_DIR.glob("scp682_sc11_predicted_observed_*.tsv")
        if "manifest" not in p.name and "per_target" not in p.name
    )
    all_summary = []
    all_targets = []
    all_h = []
    all_alignment = []
    all_top = []
    all_w = []

    for path in files:
        result = process_file(path)
        all_summary.extend(result["summary_rows"])
        all_targets.extend(result["target_rows"])
        all_h.extend(result["h_rows"])
        all_alignment.extend(result["alignment_rows"])
        all_top.extend(result["top_rows"])
        w_df = result["w_df"]
        if isinstance(w_df, pd.DataFrame) and not w_df.empty:
            all_w.append(w_df)

    summary = pd.DataFrame(all_summary)
    targets = pd.DataFrame(all_targets)
    h_long = pd.DataFrame(all_h)
    alignment = pd.DataFrame(all_alignment)
    top = pd.DataFrame(all_top)
    w_scores = pd.concat(all_w, ignore_index=True, sort=False) if all_w else pd.DataFrame()

    write_table(summary, "validation_nmf_summary.tsv", "每个外部验证队列的 NMF 状态、目标数、分解维度、重构误差以及预测/实测模块对齐指标。")
    write_table(targets, "validation_nmf_target_filter.tsv", "每个队列每个位点是否进入 NMF；实测值为常数或方差为 0 的位点不进入。")
    write_table(h_long, "validation_nmf_h_loadings_long.tsv", "NMF H 矩阵长表；matrix_type 区分 observed 与 predicted，normalized_loading 为每个成分内归一化后的位点载荷。")
    write_table(alignment, "validation_nmf_component_alignment.tsv", "预测 NMF 成分与实测 NMF 成分的最优匹配；H 用 cosine，W 用细胞得分 Spearman。")
    write_table(top, "validation_nmf_top_targets.tsv", "每个 NMF 成分载荷最高的位点摘要。")
    if not w_scores.empty:
        write_table(w_scores, "validation_nmf_w_scores_wide.tsv", "每个细胞/样本的 observed 和 predicted NMF 成分得分宽表。")

    if not h_long.empty:
        draw_loadings_heatmap(h_long, summary)
    if not alignment.empty:
        draw_alignment(alignment, summary)
        draw_summary(summary)

    manifest = {
        "output_dir": str(OUT),
        "input_dir": str(PRED_DIR.relative_to(PM)),
        "n_input_files": len(files),
        "figures": sorted(p.name for p in FIG.glob("*.png")),
        "source_tables": sorted(p.name for p in SD.glob("*.tsv")),
        "notes": [
            "NMF 使用 raw scale 的 observed/predicted 矩阵，并对每个位点做 1%-99% 分位裁剪后的非负归一化。",
            "只用实测值有变化的位点进入 NMF；Blair 只有 1 个可用 readout，因此不能做多位点 NMF。",
            "component_alignment 通过 H 载荷 cosine 做最优匹配，W Spearman 评估同一细胞上的模块得分一致性。",
        ],
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# SCP682-SC validation NMF\n\n"
        "本目录保存所有外部验证队列的 NMF 分解结果。NMF 不把缺失当 0；输入矩阵先在每个位点内做非负归一化。\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
