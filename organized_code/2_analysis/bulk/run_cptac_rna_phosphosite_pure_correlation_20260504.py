#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('/data/lsy/Infinite_Stream')
DATA = ROOT / '01_data/multi_omics/processed/pancancer_multi_task_locked_v2'
RESULT = ROOT / '02_results/biological_validation/20260504_cptac_rna_phosphosite_pure_correlation_v1'
TABLE = RESULT / 'tables'
LOG = RESULT / 'logs'
REPORT = RESULT / 'reports'

RNA_PATH = DATA / 'rna_log2_tpm_paired.parquet'
PHOS_Z_PATH = DATA / 'phosphosite_gene_site_study_zscore_min20pct_targets.parquet'
PHOS_LOGR_PATH = DATA / 'phosphosite_gene_site_logratio_min20pct_targets.parquet'
MANIFEST_PATH = DATA / 'sample_manifest.tsv'
TARGET_MANIFEST_PATH = DATA / 'target_manifest_gene_site_locked_v2.tsv'

MIN_N = 10


def mkdirs() -> None:
    for p in [TABLE, LOG, REPORT]:
        p.mkdir(parents=True, exist_ok=True)


def pearson_spearman(x: pd.Series, y: pd.Series) -> dict:
    tmp = pd.DataFrame({'x': x, 'y': y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(tmp))
    out = {'n': n, 'pearson': np.nan, 'pearson_p': np.nan, 'spearman': np.nan, 'spearman_p': np.nan}
    if n >= MIN_N and tmp['x'].nunique() > 1 and tmp['y'].nunique() > 1:
        pr = stats.pearsonr(tmp['x'], tmp['y'])
        sr = stats.spearmanr(tmp['x'], tmp['y'])
        out.update({'pearson': float(pr.statistic), 'pearson_p': float(pr.pvalue), 'spearman': float(sr.statistic), 'spearman_p': float(sr.pvalue)})
    return out


def site_parent_correlations(rna: pd.DataFrame, phos: pd.DataFrame, target_manifest: pd.DataFrame, manifest: pd.DataFrame, value_layer: str) -> pd.DataFrame:
    rows = []
    sample_ids = phos.index.intersection(rna.index)
    phos = phos.loc[sample_ids]
    rna = rna.loc[sample_ids]
    sample_cancer = manifest.set_index('sample_id').loc[sample_ids, 'cancer_label']
    target_manifest = target_manifest[target_manifest['gene_site_id'].isin(phos.columns)].copy()
    for _, row in target_manifest.iterrows():
        site_id = str(row['gene_site_id'])
        gene = str(row['gene'])
        if gene not in rna.columns or site_id not in phos.columns:
            continue
        base = {
            'value_layer': value_layer,
            'aggregation': 'gene_site_original',
            'phosphosite_id': site_id,
            'gene': gene,
            'site': str(row['site_canonical']),
            'n_observed_matrix': int(row.get('n_observed', phos[site_id].notna().sum())),
            'sample_coverage_matrix': float(row.get('sample_coverage', phos[site_id].notna().mean())),
        }
        all_res = pearson_spearman(rna[gene], phos[site_id])
        rows.append({**base, 'cohort': 'PAN_CPTAC_PDC', **all_res})
        for cancer, ids in sample_cancer.groupby(sample_cancer).groups.items():
            ids = list(ids)
            res = pearson_spearman(rna.loc[ids, gene], phos.loc[ids, site_id])
            rows.append({**base, 'cohort': str(cancer), **res})
    return pd.DataFrame(rows)


def build_residue_sum_matrix(phos: pd.DataFrame, target_manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_manifest = target_manifest[target_manifest['gene_site_id'].isin(phos.columns)].copy()
    residue_to_cols: dict[str, list[str]] = {}
    residue_rows = []
    for _, row in target_manifest.iterrows():
        col = str(row['gene_site_id'])
        gene = str(row['gene'])
        sites = [s for s in str(row['site_canonical']).split('_') if s]
        for residue in sites:
            rid = f'{gene}|{residue}'
            residue_to_cols.setdefault(rid, []).append(col)
    out = {}
    for rid, cols in residue_to_cols.items():
        vals = phos[cols].sum(axis=1, skipna=True, min_count=1)
        out[rid] = vals.astype(np.float32)
        gene, residue = rid.split('|', 1)
        residue_rows.append({'residue_id': rid, 'gene': gene, 'residue': residue, 'n_source_gene_site_features': len(cols), 'source_gene_site_features': ','.join(cols)})
    residue_mat = pd.DataFrame(out, index=phos.index)
    residue_manifest = pd.DataFrame(residue_rows)
    return residue_mat, residue_manifest


def residue_parent_correlations(rna: pd.DataFrame, residue_mat: pd.DataFrame, residue_manifest: pd.DataFrame, manifest: pd.DataFrame, value_layer: str) -> pd.DataFrame:
    rows = []
    sample_ids = residue_mat.index.intersection(rna.index)
    residue_mat = residue_mat.loc[sample_ids]
    rna = rna.loc[sample_ids]
    sample_cancer = manifest.set_index('sample_id').loc[sample_ids, 'cancer_label']
    for _, row in residue_manifest.iterrows():
        rid = str(row['residue_id'])
        gene = str(row['gene'])
        if gene not in rna.columns or rid not in residue_mat.columns:
            continue
        base = {
            'value_layer': value_layer,
            'aggregation': 'single_residue_sum_from_all_features_containing_residue',
            'phosphosite_id': rid,
            'gene': gene,
            'site': str(row['residue']),
            'n_source_gene_site_features': int(row['n_source_gene_site_features']),
            'source_gene_site_features': row['source_gene_site_features'],
            'n_observed_matrix': int(residue_mat[rid].notna().sum()),
            'sample_coverage_matrix': float(residue_mat[rid].notna().mean()),
        }
        rows.append({**base, 'cohort': 'PAN_CPTAC_PDC', **pearson_spearman(rna[gene], residue_mat[rid])})
        for cancer, ids in sample_cancer.groupby(sample_cancer).groups.items():
            ids = list(ids)
            rows.append({**base, 'cohort': str(cancer), **pearson_spearman(rna.loc[ids, gene], residue_mat.loc[ids, rid])})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for (value_layer, aggregation, cohort), sub in df.groupby(['value_layer', 'aggregation', 'cohort'], dropna=False):
        vals_p = sub['pearson'].dropna()
        vals_s = sub['spearman'].dropna()
        rows.append({
            'analysis': name,
            'value_layer': value_layer,
            'aggregation': aggregation,
            'cohort': cohort,
            'n_features_tested': int(len(sub)),
            'n_features_valid_pearson': int(vals_p.shape[0]),
            'pearson_median': float(vals_p.median()) if len(vals_p) else np.nan,
            'pearson_q25': float(vals_p.quantile(0.25)) if len(vals_p) else np.nan,
            'pearson_q75': float(vals_p.quantile(0.75)) if len(vals_p) else np.nan,
            'pearson_abs_median': float(vals_p.abs().median()) if len(vals_p) else np.nan,
            'pearson_ge_0_1': int((vals_p >= 0.1).sum()),
            'pearson_ge_0_2': int((vals_p >= 0.2).sum()),
            'pearson_ge_0_3': int((vals_p >= 0.3).sum()),
            'pearson_le_minus_0_1': int((vals_p <= -0.1).sum()),
            'n_features_valid_spearman': int(vals_s.shape[0]),
            'spearman_median': float(vals_s.median()) if len(vals_s) else np.nan,
            'spearman_q25': float(vals_s.quantile(0.25)) if len(vals_s) else np.nan,
            'spearman_q75': float(vals_s.quantile(0.75)) if len(vals_s) else np.nan,
            'spearman_abs_median': float(vals_s.abs().median()) if len(vals_s) else np.nan,
            'spearman_ge_0_1': int((vals_s >= 0.1).sum()),
            'spearman_ge_0_2': int((vals_s >= 0.2).sum()),
            'spearman_ge_0_3': int((vals_s >= 0.3).sum()),
            'spearman_le_minus_0_1': int((vals_s <= -0.1).sum()),
        })
    return pd.DataFrame(rows)


def key_site_table(*dfs: pd.DataFrame) -> pd.DataFrame:
    keys = ['ERBB2', 'AKT1', 'AKT2', 'AKT3', 'AKT1S1', 'MTOR', 'RPTOR', 'RICTOR', 'RPS6', 'RPS6KB1', 'RPS6KA1', 'RPS6KA3', 'GSK3A', 'GSK3B', 'TSC2', 'PDPK1', 'PIK3CA', 'PIK3CB', 'PIK3R1']
    out = []
    for df in dfs:
        sub = df[(df['cohort'] == 'PAN_CPTAC_PDC') & (df['gene'].isin(keys))].copy()
        out.append(sub)
    res = pd.concat(out, ignore_index=True)
    return res.sort_values(['gene', 'site', 'aggregation', 'value_layer'])


def main() -> int:
    mkdirs()
    rna = pd.read_parquet(RNA_PATH)
    phos_z = pd.read_parquet(PHOS_Z_PATH)
    phos_logr = pd.read_parquet(PHOS_LOGR_PATH)
    manifest = pd.read_csv(MANIFEST_PATH, sep='\t')
    target_manifest = pd.read_csv(TARGET_MANIFEST_PATH, sep='\t')
    common = rna.index.intersection(phos_z.index).intersection(phos_logr.index)
    rna = rna.loc[common]
    phos_z = phos_z.loc[common]
    phos_logr = phos_logr.loc[common]
    manifest = manifest[manifest['sample_id'].isin(common)].copy()

    site_z = site_parent_correlations(rna, phos_z, target_manifest, manifest, 'study_zscore')
    site_logr = site_parent_correlations(rna, phos_logr, target_manifest, manifest, 'logratio')
    site_all = pd.concat([site_z, site_logr], ignore_index=True, sort=False)
    site_all.to_csv(TABLE / 'rna_parent_vs_phosphosite_original_gene_site_correlations.tsv', sep='\t', index=False)

    residue_z_mat, residue_manifest = build_residue_sum_matrix(phos_z, target_manifest)
    residue_z_mat.to_parquet(TABLE / 'phosphosite_single_residue_sum_matrix_study_zscore.parquet')
    residue_manifest.to_csv(TABLE / 'phosphosite_single_residue_sum_manifest.tsv', sep='\t', index=False)
    residue_z = residue_parent_correlations(rna, residue_z_mat, residue_manifest, manifest, 'study_zscore')

    residue_logr_mat, residue_manifest_logr = build_residue_sum_matrix(phos_logr, target_manifest)
    residue_logr = residue_parent_correlations(rna, residue_logr_mat, residue_manifest_logr, manifest, 'logratio')
    residue_all = pd.concat([residue_z, residue_logr], ignore_index=True, sort=False)
    residue_all.to_csv(TABLE / 'rna_parent_vs_phosphosite_single_residue_sum_correlations.tsv', sep='\t', index=False)

    summary = pd.concat([summarize(site_all, 'original_gene_site'), summarize(residue_all, 'single_residue_sum')], ignore_index=True)
    summary.to_csv(TABLE / 'rna_phosphosite_pure_correlation_summary.tsv', sep='\t', index=False)

    key = key_site_table(site_all, residue_all)
    key.to_csv(TABLE / 'rna_phosphosite_key_pathway_site_correlations.tsv', sep='\t', index=False)

    run = {
        'script': str(Path(__file__)),
        'result_dir': str(RESULT),
        'rna_matrix': str(RNA_PATH),
        'phosphosite_study_zscore_matrix': str(PHOS_Z_PATH),
        'phosphosite_logratio_matrix': str(PHOS_LOGR_PATH),
        'n_samples': int(len(common)),
        'n_rna_genes': int(rna.shape[1]),
        'n_original_gene_site_features': int(phos_z.shape[1]),
        'n_single_residue_sum_features': int(residue_z_mat.shape[1]),
        'minimum_valid_pairs_per_correlation': MIN_N,
        'single_residue_sum_rule': 'split site_canonical by underscore; each original feature contributes to every residue it contains; values are summed per sample for the same gene|residue before RNA correlation',
        'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    (LOG / 'run_config.json').write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding='utf-8')

    main_summary = summary[(summary['cohort'] == 'PAN_CPTAC_PDC') & (summary['value_layer'] == 'study_zscore')]
    lines = ['# RNA parent gene vs phosphosite pure correlation', '', main_summary.to_markdown(index=False)]
    lines.append('')
    lines.append('## Key pathway examples')
    show_cols = ['value_layer','aggregation','phosphosite_id','gene','site','n','pearson','spearman','n_source_gene_site_features']
    existing = [c for c in show_cols if c in key.columns]
    lines.append(key[(key['cohort']=='PAN_CPTAC_PDC') & (key['value_layer']=='study_zscore')].head(80)[existing].to_markdown(index=False))
    (REPORT / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(run, indent=2, ensure_ascii=False), flush=True)
    print(main_summary.to_string(index=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
