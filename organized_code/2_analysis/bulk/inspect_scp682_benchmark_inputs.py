from pathlib import Path

import pandas as pd


def show(path: Path) -> None:
    print(f"\n### {path}")
    print(f"exists={path.exists()} size={path.stat().st_size if path.exists() else 'NA'}")
    if path.exists() and path.suffix == ".parquet":
        df = pd.read_parquet(path)
        print(f"shape={df.shape}")
        print(f"index_head={list(df.index[:3])}")
        print(f"columns_head={list(df.columns[:8])}")
    elif path.exists() and path.suffix in {".tsv", ".csv"}:
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, nrows=5)
        print(f"columns={list(df.columns)}")
        print(df.head().to_string(index=False))


base = Path("/data/lsy/Infinite_Stream")
pkg = base / "SCP682_PORTABLE"

paths = [
    pkg / "training_set/observed_phosphosite.parquet",
    pkg / "training_set/oof_candidate_parent_only_phosphosite.parquet",
    pkg / "training_set/oof_candidate_ridge_direct_phosphosite.parquet",
    pkg / "training_set/oof_candidate_rna_direct_phosphosite.parquet",
    pkg / "training_set/sample_manifest.tsv",
    pkg / "training_set/phosphosite_target_manifest.tsv",
    pkg / "predictions/scp682_main_oof_phosphosite.parquet",
    pkg / "performance/per_site_spearman.tsv",
    base / "SCP682/02_results/model_validation/20260503_cptac_phosphosite_robust_stacking_v3_9/predictions/oof_candidate_v3_9_sample_centered_robust_equal_plus_cvae_0_2_phosphosite.parquet",
    base / "01_data/pathway_prior/processed/kinase_substrate_prior_for_modeling_v1.tsv",
    base / "01_data/pathway_prior/processed/copheemap_v1/copheeksa_model_phosphosite_kinase_predictions.tsv",
]

for path in paths:
    show(path)

