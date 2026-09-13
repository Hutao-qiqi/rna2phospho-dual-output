#!/usr/bin/env python3
"""PhosphoSCT-S1 使用锁定 M2.2 PT7 参数进行外部评价。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
FUSION_SCRIPT = ROOT / "03_code/model_validation/scp682_v2_m22_zero_aware_pt7/run_m22_f_pt7_prediction_fusion.py"
S1 = ROOT / "02_results/model_validation/scp682_v2_m23_phosphosct_s1_20260831/models/phosphosct_s1_model.npz"
M2_PREDICTION = ROOT / "02_results/external_validation/scp682_v2_m22_f_pt7_fusion_20260831_v2/predictions/fusion_selected_predictions.npz"
OUT = ROOT / "02_results/external_validation/scp682_v2_m23_phosphosct_s1_pt7_20260831"
PT7 = (0.10, 1.0, 100.0)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    for name in ("tables", "predictions", "reports", "logs"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    fusion = load_module(FUSION_SCRIPT, "fusion_pipeline")
    module = fusion.load_pipeline()
    core = module.core
    core.LATENT = module.LATENT

    primary_rna = core.numeric_frame(module.PRIMARY / "rna_reference_quantile.parquet")
    primary_protein = core.numeric_frame(module.PRIMARY / "protein_prediction.parquet").reindex(primary_rna.index)
    primary_truth = core.numeric_frame(module.PRIMARY / "phosphosite_logscale_aligned.parquet").reindex(primary_rna.index)
    primary_split = pd.read_csv(module.PRIMARY / "split_manifest.tsv", sep="\t").set_index("sample_id").reindex(primary_rna.index)
    primary_train = primary_split.role.eq("selection_train").to_numpy()
    external_rna = core.numeric_frame(module.EXTERNAL / "rna_reference_quantile.parquet")
    external_protein = core.numeric_frame(module.EXTERNAL / "protein_prediction.parquet").reindex(external_rna.index)
    external_truth = core.numeric_frame(module.EXTERNAL / "phosphosite_logscale_aligned.parquet").reindex(external_rna.index)
    external_split = pd.read_csv(module.EXTERNAL / "split_manifest.tsv", sep="\t").set_index("sample_id").reindex(external_rna.index)
    external_meta = pd.read_csv(module.EXTERNAL / "sample_metadata.tsv", sep="\t").set_index("sample_id").reindex(external_rna.index)
    calibration_ids = external_rna.index[external_split.role.eq("selection_train")]
    test_ids = external_rna.index[external_split.role.eq("selection_validation")]
    all_ids = calibration_ids.append(test_ids)
    calibration_mask = external_rna.index.isin(calibration_ids)

    manifest = pd.read_csv(module.MANIFEST, sep="\t")
    protein_lookup = {str(gene).upper(): index for index, gene in enumerate(primary_protein.columns.astype(str))}
    parent_index = np.asarray([protein_lookup[str(gene).upper()] for gene in manifest.total_protein_gene], dtype=np.int64)
    beta = pd.read_csv(module.M1 / "parent_beta.tsv", sep="\t").beta.to_numpy(np.float32)
    primary_y = primary_truth.to_numpy(np.float32)
    primary_p = primary_protein.to_numpy(np.float32)
    primary_mean = np.nanmean(primary_y[primary_train], axis=0).astype(np.float32)
    primary_count = np.isfinite(primary_y[primary_train]).sum(axis=0)
    primary_protein_center = np.nanmean(primary_p[primary_train], axis=0).astype(np.float32)

    old_package = np.load(module.LATENT / "models/input_projection_and_ridge.npz")
    s1 = np.load(S1)
    feature = fusion.transform_inputs(
        external_rna.to_numpy(np.float32), external_protein.to_numpy(np.float32), old_package
    )
    coordinate = (feature @ s1["ridge_coef"].T + s1["ridge_intercept"][None, :]).astype(np.float32)
    basis = s1["basis"].astype(np.float32)

    truth_cal = external_truth.loc[calibration_ids].to_numpy(np.float32)
    protein_cal = external_protein.loc[calibration_ids].to_numpy(np.float32)
    studies_cal = external_meta.loc[calibration_ids].study.astype(str).to_numpy()
    fold_id = pd.read_csv(module.PROTOCOL / "calibration_fold_assignments.tsv", sep="\t").set_index("sample_id").reindex(calibration_ids).fold.to_numpy(np.int64)
    s1_oof = fusion.pt7_oof(
        module, coordinate[calibration_mask], truth_cal, protein_cal, studies_cal, fold_id,
        basis, primary_mean, primary_count, primary_protein_center, parent_index, beta,
        PT7, "PhosphoSCT-S1",
    )

    order = external_rna.index.get_indexer(all_ids)
    truth_ordered = external_truth.loc[all_ids].to_numpy(np.float32)
    protein_ordered = external_protein.loc[all_ids].to_numpy(np.float32)
    studies_ordered = external_meta.loc[all_ids].study.astype(str).to_numpy()
    s1_test = fusion.pt7_full(
        module, coordinate[order], truth_ordered, protein_ordered, studies_ordered,
        len(calibration_ids), basis, primary_mean, primary_count, primary_protein_center,
        parent_index, beta, PT7,
    )

    m2 = np.load(M2_PREDICTION, allow_pickle=True)
    if not np.array_equal(m2["calibration_ids"].astype(str), calibration_ids.astype(str)):
        raise ValueError("calibration order differs")
    if not np.array_equal(m2["test_ids"].astype(str), test_ids.astype(str)):
        raise ValueError("test order differs")
    m2_oof = m2["m2_oof"].astype(np.float32)
    m2_test = m2["selected_test"].astype(np.float32)
    truth_test = external_truth.loc[test_ids].to_numpy(np.float32)
    studies_test = external_meta.loc[test_ids].study.astype(str).to_numpy()
    rows = []
    for model_name, scope, truth, prediction, studies in (
        ("M2.2_PT7", "external_247_OOF", truth_cal, m2_oof, studies_cal),
        ("PhosphoSCT_S1_PT7", "external_247_OOF", truth_cal, s1_oof, studies_cal),
        ("M2.2_PT7", "external_106_locked_test", truth_test, m2_test, studies_test),
        ("PhosphoSCT_S1_PT7", "external_106_locked_test", truth_test, s1_test, studies_test),
    ):
        rows.extend(module.metric_pair(truth, prediction, studies, model_name, scope))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "tables/phosphosct_s1_external_pt7_metrics.tsv", sep="\t", index=False)
    np.savez_compressed(
        OUT / "predictions/phosphosct_s1_external_pt7_predictions.npz",
        calibration_ids=np.asarray(calibration_ids, str),
        test_ids=np.asarray(test_ids, str),
        site_ids=external_truth.columns.astype(str).to_numpy(),
        s1_oof=s1_oof,
        s1_test=s1_test,
    )
    centered = table.query("coordinate == 'study_centered'")
    report = "\n".join(
        [
            "# PhosphoSCT-S1 外部 PT7",
            "",
            "| 数据 | 模型 | 患者 Pearson | 患者 Spearman | 位点 Pearson | 位点 Spearman | MSE |",
            "|---|---|---:|---:|---:|---:|---:|",
            *[
                f"| {row.scope} | {row.model} | {row.patient_pearson_median:.6f} | {row.patient_spearman_median:.6f} | {row.site_pearson_median:.6f} | {row.site_spearman_median:.6f} | {row.patient_equal_mse:.6f} |"
                for _, row in centered.iterrows()
            ],
            "",
        ]
    )
    (OUT / "RESULTS_20260831.md").write_text(report, encoding="utf-8")
    (OUT / "reports/run_summary.json").write_text(
        json.dumps({"status": "complete", "pt7": PT7, "metrics": table.to_dict("records")}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "SUCCESS").touch()
    print(centered.to_string(index=False), flush=True)
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
