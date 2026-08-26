@echo off
setlocal

set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set PY=D:\Tools\anaconda3\python.exe
set SCRIPT=%ROOT%\03_code\single_cell\modeling\train_scp682_sc12_protein_residual.py
set OUT=02_results\single_cell\20260621_scp682_sc12_protein_residual_smoke_v1
set PROTEIN_CACHE=01_data\single_cell\intermediate\protein_prediction_cache\scTranslator_sc12_all_protein_cache_merged_v1

"%PY%" "%SCRIPT%" ^
  --root "%ROOT%" ^
  --model-input-dir "01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_gse300551_signal_seq_multidomain_v1" ^
  --pathway-manifest "02_results\single_cell\20260519_scp682_sc3_multidomain_features_v1\intermediate\pathway_gene_manifest.tsv" ^
  --protein-cache-dir "%PROTEIN_CACHE%" ^
  --output-dir "%OUT%" ^
  --train-datasets "iccite_seq_tcell_2025,qurie_seq_bjab_2021" ^
  --holdout-datasets "gse300551_iccite_plex_kinase_2025,phospho_seq_blair_2025_phospho_multi,vivo_seq_th17_2025,signal_seq_gse256403_hela_2024,signal_seq_gse256404_pdo_caf_2024" ^
  --target-ids include_in_loss ^
  --epochs 2 ^
  --patience 2 ^
  --smoke ^
  --smoke-cells 2048 ^
  --batch-size 512 ^
  --eval-batch-size 1024 ^
  --prior-steps-per-epoch 1 ^
  --site-graph-candidate-limit 32 ^
  --site-graph-max-aux-nodes 1024 ^
  --device cuda:1

exit /b %ERRORLEVEL%
