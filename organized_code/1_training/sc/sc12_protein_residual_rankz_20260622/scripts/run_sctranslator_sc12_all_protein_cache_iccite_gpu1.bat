@echo off
setlocal enabledelayedexpansion

set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set PY=D:\Tools\anaconda3\python.exe
set SCRIPT=%ROOT%\03_code\single_cell\modeling\run_sctranslator_sc12_all_protein_inference.py
set CKPT=D:\data\lsy\models\scTranslator\checkpoint\scTranslator_2M.pt
set PROTEINS=%ROOT%\01_data\single_cell\intermediate\protein_prediction_cache\scp682_sc12_all_predictable_proteins_v1\scTranslator_hgnc_protein_coding_query.txt
set H5AD_DIR=%ROOT%\01_data\single_cell\intermediate\foundation_model_h5ad_inputs_v1
set OUT_ROOT=%ROOT%\01_data\single_cell\intermediate\protein_prediction_cache\scTranslator_sc12_all_protein_cache_v1
set LOG_DIR=%OUT_ROOT%\_logs

set CUDA_VISIBLE_DEVICES=1

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

call :run_one iccite_seq_tcell_2025
exit /b %ERRORLEVEL%

:run_one
set DATASET=%~1
if exist "%OUT_ROOT%\%DATASET%\manifest.json" (
  echo [%date% %time%] skip %DATASET%, manifest exists>>"%LOG_DIR%\all_protein_gpu1.log"
  exit /b 0
)
"%PY%" "%SCRIPT%" --checkpoint "%CKPT%" --rna-h5ad "%H5AD_DIR%\%DATASET%.h5ad" --protein-list "%PROTEINS%" --output-dir "%OUT_ROOT%\%DATASET%" --chunk-size 128 --batch-size 64 --protein-shard-size 1000 --resume 1>>"%LOG_DIR%\all_protein_gpu1.log" 2>>"%LOG_DIR%\all_protein_gpu1.err.log"
exit /b %ERRORLEVEL%
