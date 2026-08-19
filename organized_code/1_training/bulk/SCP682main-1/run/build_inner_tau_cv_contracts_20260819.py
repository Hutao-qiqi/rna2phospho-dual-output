"""Build study-stratified 5-fold inner cross-validation contracts for tau selection.

Reads the locked random70 reference RNA/protein/phosphosite inputs and writes
a train/validation contract per inner fold. Paths come from the command line;
no local paths are hardcoded.

Usage:
    python build_inner_tau_cv_contracts_20260819.py \
        --input  <random70_reference_rna_protein_..._inputs_20260815> \
        --output <output_dir>
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


SEED = 20260819
FOLDS = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    INPUT = args.input
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=False)
    split = pd.read_csv(INPUT / 'split_manifest.tsv', sep='\t')
    meta = pd.read_csv(INPUT / 'sample_metadata.tsv', sep='\t').set_index('sample_id')
    prov = pd.read_csv(INPUT / 'protein_prediction_provenance.tsv', sep='\t').set_index('sample_id')
    protein = pd.read_parquet(INPUT / 'protein_prediction.parquet')
    train = split.loc[split.role.eq('selection_train'), 'sample_id'].astype(str).to_numpy()
    studies = meta.loc[train, 'study'].astype(str).to_numpy()
    rng = np.random.default_rng(SEED)
    fold = np.full(train.size, -1, dtype=np.int64)
    for study in np.unique(studies):
        ix = np.flatnonzero(studies == study)
        rng.shuffle(ix)
        for j, pos in enumerate(ix):
            fold[pos] = j % FOLDS
    pd.DataFrame({'sample_id': train, 'inner_fold': fold}).to_csv(OUT / 'inner_fold_assignment.tsv', sep='\t', index=False)
    for k in range(FOLDS):
        val = train[fold == k]
        tr = train[fold != k]
        ids = np.concatenate([tr, val])
        fold_dir = OUT / f'fold{k}'
        fold_dir.mkdir()
        pd.DataFrame({'sample_id': ids, 'role': ['selection_train'] * len(tr) + ['selection_validation'] * len(val)}).to_csv(fold_dir / 'split_manifest.tsv', sep='\t', index=False)
        meta.loc[ids].reset_index().to_csv(fold_dir / 'sample_metadata.tsv', sep='\t', index=False)
        p = prov.loc[ids].copy()
        p['prediction_role'] = ['cross_fitted'] * len(tr) + ['selection_train_only'] * len(val)
        p.reset_index().to_csv(fold_dir / 'protein_prediction_provenance.tsv', sep='\t', index=False)
        protein.loc[ids].to_parquet(fold_dir / 'protein_prediction.parquet')
        pd.DataFrame({'train_samples': [len(tr)], 'validation_samples': [len(val)]}).to_csv(fold_dir / 'sizes.tsv', sep='\t', index=False)
    (OUT / 'SUCCESS').touch()


if __name__ == '__main__':
    main()
