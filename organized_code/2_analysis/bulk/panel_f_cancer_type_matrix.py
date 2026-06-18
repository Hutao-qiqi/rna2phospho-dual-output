#!/usr/bin/env python
"""panel_f_cancer_type_matrix.py — 重算 panel f 矩阵：列从器官组 → 癌种。

读：
  - paper_materials_SCP682/02_data_tables/oof_branch_predictions/observed_phosphosite.parquet
  - paper_materials_SCP682/02_data_tables/oof_branch_predictions/sample_manifest.tsv  (含 cancer_label)
  - SCP682_PORTABLE/predictions/scp682_main_oof_phosphosite.parquet
  - _scripts/panel_f_heatmap_matrix.tsv  (2000 个目标位点 row_order/target/gene/direction)

算：
  - 折叠 cancer_label 子批次 (BRCA_PROSPECTIVE + BRCA_TCGA → BRCA 等)
  - 对每 (cancer_type, target) 做 Spearman ρ (predicted vs observed)
  - 输出 2000 行 × (cancer_type 列们) 的宽矩阵

输出：
  panel_f_heatmap_matrix_cancer_type.tsv  (替换原版 panel_f_heatmap_matrix.tsv)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("E:/data/gongke/TCGA-TCPA/paper_materials_SCP682")
OBS_PQ = ROOT / "02_data_tables" / "oof_branch_predictions" / "observed_phosphosite.parquet"
MANI_TSV = ROOT / "02_data_tables" / "oof_branch_predictions" / "sample_manifest.tsv"
PRED_PQ = Path("E:/data/gongke/TCGA-TCPA/SCP682_PORTABLE/predictions/scp682_main_oof_phosphosite.parquet")

SCRIPTS = ROOT / "04_figure_source_data" / "fig2" / "_scripts"
TARGET_TSV = SCRIPTS / "panel_f_heatmap_matrix.tsv"
OUT_TSV    = SCRIPTS / "panel_f_heatmap_matrix_cancer_type.tsv"

# Cancer label 折叠：把 PROSPECTIVE/TCGA/CONFIRMATORY/DISCOVERY 等批次后缀合并
COLLAPSE = {
    "BRCA_PROSPECTIVE": "BRCA",
    "BRCA_TCGA":         "BRCA",
    "CCRCC":             "CCRCC",
    "NON_CCRCC":         "ccPRCC",   # papillary RCC 不是 CCRCC，单列
    "COAD_PROSPECTIVE":  "COAD",
    "GBM_CONFIRMATORY":  "GBM",
    "GBM_DISCOVERY":     "GBM",
    "HNSCC":             "HNSCC",
    "LSCC":              "LSCC",
    "LUAD":              "LUAD",
    "LUAD_CONFIRM":      "LUAD",
    "OV_PROSPECTIVE":    "OV",
    "OV_TCGA":           "OV",
    "PDA":               "PDA",
    "STAD":              "STAD",
    "UCEC":              "UCEC",
    "UCEC_CONFIRM":      "UCEC",
}

# 显示顺序（按癌种字母）
DISPLAY_ORDER = [
    "BRCA", "CCRCC", "ccPRCC", "COAD", "GBM",
    "HNSCC", "LSCC", "LUAD", "OV", "PDA", "STAD", "UCEC"
]

MIN_SAMPLES_PER_SITE = 10   # 单癌种内最少有效 (pred, obs) 才算 ρ


def main() -> int:
    print("Reading parquet files ...")
    pred = pd.read_parquet(PRED_PQ).astype(np.float32)
    obs  = pd.read_parquet(OBS_PQ).astype(np.float32)
    mani = pd.read_csv(MANI_TSV, sep="\t")

    # 对齐 index/cols
    common_samples = pred.index.intersection(obs.index)
    common_sites   = pred.columns.intersection(obs.columns)
    pred = pred.loc[common_samples, common_sites]
    obs  = obs.loc[common_samples, common_sites]
    print(f"after align: {pred.shape}")

    # 把 sample → cancer_type
    mani_idx = mani.set_index("index")
    sample_cancer = mani_idx.loc[common_samples, "cancer_label"].map(COLLAPSE)
    sample_cancer = sample_cancer.dropna()
    pred = pred.loc[sample_cancer.index]
    obs  = obs.loc[sample_cancer.index]

    cancer_types = sorted(sample_cancer.unique(), key=lambda x: DISPLAY_ORDER.index(x))
    print("cancer types: ", cancer_types)
    print("samples per cancer:")
    print(sample_cancer.value_counts())

    # 只保留 panel_f 关心的 2000 个 target
    tgt = pd.read_csv(TARGET_TSV, sep="\t")
    target_set = [t for t in tgt["target"] if t in pred.columns]
    print(f"target sites in panel f: {len(tgt)}; in prediction matrix: {len(target_set)}")
    pred = pred[target_set]
    obs  = obs[target_set]

    # 算每个 (cancer_type, target) 的 Spearman ρ
    rows = []
    for ct in cancer_types:
        mask = (sample_cancer == ct).values
        sub_pred = pred.iloc[mask]
        sub_obs  = obs.iloc[mask]
        if sub_pred.shape[0] < MIN_SAMPLES_PER_SITE:
            continue
        # 向量化：每列分别做 spearmanr
        ct_rhos = np.full(len(target_set), np.nan, dtype=np.float64)
        for j, site in enumerate(target_set):
            p = sub_pred.iloc[:, j].values
            o = sub_obs.iloc[:, j].values
            both = ~(np.isnan(p) | np.isnan(o))
            n = int(both.sum())
            if n < MIN_SAMPLES_PER_SITE:
                continue
            try:
                rho, _ = spearmanr(p[both], o[both])
                if np.isfinite(rho):
                    ct_rhos[j] = rho
            except Exception:
                pass
        rows.append({"cancer_type": ct, "n_samples": int(mask.sum()),
                     **{site: rho for site, rho in zip(target_set, ct_rhos)}})
        n_valid = int(np.sum(~np.isnan(ct_rhos)))
        print(f"  {ct}: n={mask.sum()}, sites with rho = {n_valid}/{len(target_set)}")

    out = pd.DataFrame(rows).set_index("cancer_type")
    print(f"\nfinal matrix: {out.shape}")

    # 转置成 target × cancer_type 长 → 宽，按目标顺序
    long = (out
            .drop(columns=["n_samples"])
            .T  # rows = sites
            .reset_index()
            .rename(columns={"index": "target"}))

    # 加回 panel_f_heatmap_matrix.tsv 的 row_order/gene/direction 元信息
    meta = tgt[["row_order", "target", "gene", "direction", "CPTAC_all"]].copy()
    merged = meta.merge(long, on="target", how="left")
    merged.to_csv(OUT_TSV, sep="\t", index=False, float_format="%.6f")
    print(f"\nwrote {OUT_TSV}")

    # NaN 统计
    print("\nNaN per cancer type (out of 2000 sites):")
    for ct in cancer_types:
        if ct in merged.columns:
            n_nan = merged[ct].isna().sum()
            print(f"  {ct:7s} n_samples={out.loc[ct, 'n_samples']:>3}  NaN={n_nan:>4}  ({100*n_nan/len(merged):.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
