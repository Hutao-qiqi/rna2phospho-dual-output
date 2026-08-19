from pathlib import Path
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cvroot", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    INPUT = args.input
    CVROOT = args.cvroot
    RESULTS = args.results

    target = pd.read_parquet(INPUT / 'phosphosite_logscale_aligned.parquet')
    metadata = pd.read_csv(INPUT / 'sample_metadata.tsv', sep='\t').set_index('sample_id')
    rows = []
    for fold in range(5):
        split = pd.read_csv(CVROOT / f'fold{fold}/split_manifest.tsv', sep='\t')
        ids = split.sample_id.astype(str).tolist()
        train_ids = split.loc[split.role.eq('selection_train'), 'sample_id'].astype(str).tolist()
        validation_ids = split.loc[split.role.eq('selection_validation'), 'sample_id'].astype(str).tolist()
        y = target.loc[ids].to_numpy(np.float64)
        labels = metadata.loc[ids, 'study'].to_numpy(str)
        train_rows = np.flatnonzero(split.role.to_numpy() == 'selection_train')
        val_rows = np.flatnonzero(split.role.to_numpy() == 'selection_validation')
        observed = np.isfinite(y)
        global_mean = np.nanmean(y[train_rows], axis=0)
        global_mean = np.nan_to_num(global_mean, nan=0.0)
        global_count = np.isfinite(y[train_rows]).sum(axis=0)
        for tau in (0, 2, 5, 10, 20, 50):
            center = np.zeros_like(y)
            for study in np.unique(labels):
                study_rows = np.flatnonzero(labels == study)
                train_study_rows = np.intersect1d(study_rows, train_rows)
                if train_study_rows.size == 0:
                    center[study_rows] = global_mean
                    continue
                study_values = y[train_study_rows]
                study_mean = np.nanmean(study_values, axis=0)
                study_mean = np.nan_to_num(study_mean, nan=0.0)
                site_n = np.isfinite(study_values).sum(axis=0)
                common = (site_n >= 5) & (global_count >= 5)
                offset = float(np.median(study_mean[common] - global_mean[common])) if common.any() else 0.0
                weight = site_n / (site_n + tau) if tau > 0 else (site_n > 0).astype(float)
                center[study_rows] = weight * study_mean + (1.0 - weight) * (global_mean + offset)
            run = RESULTS / f'scp682main1_tau_inner_cv_f{fold}_tau{tau}_u32_20260819'
            pred_path = run / 'predictions/validation_raw_log_profile_spearman_best.parquet'
            pred = pd.read_parquet(pred_path).loc[validation_ids].to_numpy(np.float64)
            truth = y[val_rows]
            baseline = center[val_rows]
            mask = np.isfinite(truth) & np.isfinite(pred)
            mse_values = []
            r2_values = []
            spearman_values = []
            by_study = {}
            for i in range(len(validation_ids)):
                ok = mask[i]
                err = truth[i, ok] - pred[i, ok]
                residual = truth[i, ok] - baseline[i, ok]
                sse_model = float(np.sum(err * err))
                sse_zero = float(np.sum(residual * residual))
                r2 = 1.0 - sse_model / sse_zero if sse_zero > 1.0e-12 else np.nan
                rho = float(spearmanr(truth[i, ok], pred[i, ok]).statistic)
                mse_values.append(float(np.mean(err * err)))
                r2_values.append(r2)
                spearman_values.append(rho)
                by_study.setdefault(labels[val_rows[i]], []).append(rho)
            study_macro = float(np.nanmean([np.nanmedian(v) for v in by_study.values()]))
            rows.append({
                'fold': fold,
                'tau': tau,
                'raw_spearman': float(np.nanmedian(spearman_values)),
                'raw_mse': float(np.mean(mse_values)),
                'residual_r2': float(np.nanmedian(r2_values)),
                'r2_positive': float(np.mean(np.asarray(r2_values) > 0)),
                'study_macro_spearman': study_macro,
            })
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / 'scp682main1_tau_inner_cv_20260819/tau_inner_cv_summary.tsv', sep='\t', index=False)
    summary = result.groupby('tau').agg(
        folds=('fold', 'count'),
        raw_spearman_mean=('raw_spearman', 'mean'),
        raw_spearman_sd=('raw_spearman', 'std'),
        raw_mse_mean=('raw_mse', 'mean'),
        residual_r2_mean=('residual_r2', 'mean'),
        r2_positive_mean=('r2_positive', 'mean'),
        study_macro_spearman_mean=('study_macro_spearman', 'mean'),
    ).sort_values('raw_spearman_mean', ascending=False)
    summary.to_csv(RESULTS / 'scp682main1_tau_inner_cv_20260819/tau_inner_cv_by_tau.tsv', sep='\t')
    print(summary.to_string())


if __name__ == '__main__':
    main()
