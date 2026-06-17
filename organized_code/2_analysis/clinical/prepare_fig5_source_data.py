#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("E:/data/gongke/TCGA-TCPA")
FIG5 = ROOT / "paper_final/fig5"
OUT = FIG5 / "source_data/tables"
RES = ROOT / "02_results/model_validation/20260530_fig5_rps6_ps6_panel_v1"
EXACT = ROOT / "02_results/model_validation/20260530_fig5_exact_site_anchor_search_v1"


def copy_table(src: Path, name: str) -> None:
    dst = OUT / name
    if src.suffix == ".gz":
        df = pd.read_csv(src, sep="\t")
        df.to_csv(dst, sep="\t", index=False)
    else:
        shutil.copyfile(src, dst)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    copy_table(
        RES / "tables/tcga_kirc_predicted_rps6_and_predicted_mtor_phospho_state.tsv",
        "panel_a_tcga_predicted_pS6_mRNA_controls.tsv",
    )
    copy_table(
        EXACT / "tables/tcga_kirc_rps6_s235_s236_sample_table.tsv.gz",
        "panel_b_tcga_kirc_rps6_survival_samples.tsv",
    )
    copy_table(
        EXACT / "tables/tcga_kirc_rps6_s235_s236_parent_mrna_cox.tsv",
        "panel_b_tcga_kirc_rps6_cox_forest.tsv",
    )
    copy_table(
        EXACT / "tables/cptac_ccrcc_rps6_s235_s236_sample_table.tsv.gz",
        "panel_c_cptac_ccrcc_rps6_prediction_measurement.tsv",
    )
    copy_table(
        EXACT / "tables/cptac_ccrcc_rps6_s235_s236_measurement_correlations.tsv",
        "panel_c_cptac_ccrcc_rps6_correlations.tsv",
    )
    copy_table(
        EXACT / "tables/cptac_ccrcc_rps6_s235_s236_partial_correlations.tsv",
        "panel_c_cptac_ccrcc_rps6_partial_correlations.tsv",
    )
    copy_table(
        RES / "tables/cptac_ccrcc_rps6_ps6_bubble_partial_correlations.tsv",
        "panel_d_cptac_ccrcc_rps6_bubble_partial_correlations.tsv",
    )
    copy_table(
        RES / "tables/tcga_kirc_rps6_ps6_site_over_parent_waterfall_data.tsv",
        "panel_e_tcga_kirc_site_over_parent_waterfall.tsv",
    )
    copy_table(
        RES / "tables/panel_c_plotted_drug_correlations.tsv",
        "panel_f_depmap_rcc_mtor_drug_correlations.tsv",
    )
    copy_table(
        RES / "tables/depmap_rcc_rps6_ps6_mtor_drug_scatter_long.tsv.gz",
        "panel_f_depmap_rcc_mtor_drug_scatter_long.tsv",
    )

    # Add a compact manifest for the figure legend and reproducibility audit.
    manifest = {
        "figure": "Fig 5",
        "site": "RPS6|S235_S236",
        "cancer": "TCGA-KIRC / ccRCC",
        "tcga_predicted_mtor_phospho_state_features": [
            "MTOR|S2481",
            "EIF4EBP1|S65",
            "EIF4EBP1|T70",
            "RPS6KB1|T421_S424",
        ],
        "source_result_dir": str(RES),
        "exact_anchor_result_dir": str(EXACT),
    }
    (OUT / "fig5_source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Small summary table used by captions.
    a = pd.read_csv(OUT / "panel_a_tcga_predicted_pS6_mRNA_controls.tsv", sep="\t")
    rows = []
    for variable in ["predicted_rps6_s235_s236", "RPS6_mrna"]:
        y = np.log2(a["RPS6_mrna"] + 1) if variable == "RPS6_mrna" else a[variable]
        label = "RPS6 mRNA log2(TPM+1)" if variable == "RPS6_mrna" else "Predicted RPS6 pS235/S236"
        for group_col, order in [
            ("OS status", ["Alive/censored", "Deceased"]),
            ("Grade", ["G1/G2", "G3/G4"]),
            ("predicted mTOR phospho-state", ["Low", "High"]),
        ]:
            d = pd.DataFrame({"group": a[group_col], "value": y}).dropna()
            vals = [d.loc[d["group"] == g, "value"].values for g in order]
            if len(vals[0]) and len(vals[1]):
                from scipy import stats

                p = stats.mannwhitneyu(vals[0], vals[1], alternative="two-sided").pvalue
                rows.append(
                    {
                        "variable": label,
                        "group": group_col,
                        "first": order[0],
                        "second": order[1],
                        "n_first": len(vals[0]),
                        "n_second": len(vals[1]),
                        "median_shift_second_minus_first": float(np.median(vals[1]) - np.median(vals[0])),
                        "p": float(p),
                    }
                )
    pd.DataFrame(rows).to_csv(OUT / "panel_a_distribution_tests.tsv", sep="\t", index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
