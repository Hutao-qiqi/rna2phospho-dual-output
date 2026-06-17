@echo off
setlocal EnableDelayedExpansion
set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set CODE=%ROOT%\03_code\single_cell\modeling
set TEMPLATE=%ROOT%\01_data\single_cell\intermediate\phospho_model_inputs\scgpt_frozen_gse300551_signal_seq_multidomain_v1
set H5AD_MANIFEST=%ROOT%\01_data\single_cell\intermediate\foundation_model_h5ad_inputs_v1\foundation_h5ad_manifest.tsv
set PY=D:\Tools\anaconda3\python.exe
set LOGDIR=%ROOT%\02_results\single_cell\20260601_remaining_foundation_models_linear_regression_v1\logs
set DATASETS=iccite_seq_tcell_2025 phospho_seq_blair_2025_phospho_multi qurie_seq_bjab_2021 vivo_seq_th17_2025 gse300551_iccite_plex_kinase_2025 signal_seq_gse256403_hela_2024
mkdir "%LOGDIR%" 2>nul

echo [%date% %time%] remaining foundation models start > "%LOGDIR%\run_remaining.log"

echo [%date% %time%] waiting scimilarity model download >> "%LOGDIR%\run_remaining.log"
:WAIT_SCIM
if exist "D:\data\lsy\models\scimilarity\model_v1.1.tar.gz.aria2" (
  timeout /t 300 /nobreak >nul
  goto WAIT_SCIM
)
if not exist "D:\data\lsy\models\scimilarity\model_v1.1\encoder.ckpt" (
  tar -xf "D:\data\lsy\models\scimilarity\model_v1.1.tar.gz" -C "D:\data\lsy\models\scimilarity" >> "%LOGDIR%\run_remaining.log" 2>&1
)
if exist "D:\data\lsy\models\scimilarity\model_v1.1\encoder.ckpt" (
  echo [%date% %time%] running scimilarity embeddings >> "%LOGDIR%\run_remaining.log"
  "%PY%" "%CODE%\precompute_scimilarity_embeddings_multidomain.py" --use-gpu --skip-existing --datasets %DATASETS% >> "%LOGDIR%\scimilarity_embedding.log" 2>&1
  set EC=!ERRORLEVEL!
  echo !EC! > "%LOGDIR%\scimilarity_embedding_exit_code.txt"
  if "!EC!"=="0" (
    "%PY%" "%CODE%\extract_obsm_embeddings_to_model_input.py" ^
      --template-input-dir "%TEMPLATE%" ^
      --h5ad-manifest "%ROOT%\01_data\single_cell\intermediate\foundation_model_embeddings\scimilarity_v1_1_multidomain_v1\scimilarity_embedding_manifest.tsv" ^
      --output-dir "%ROOT%\01_data\single_cell\intermediate\phospho_model_inputs\scimilarity_v1_1_gse300551_signal_seq_multidomain_v1" ^
      --obsm-key X_scimilarity ^
      --datasets %DATASETS% ^
      --method-name scimilarity_v1_1 >> "%LOGDIR%\scimilarity_assemble.log" 2>&1
    set EC=!ERRORLEVEL!
    echo !EC! > "%LOGDIR%\scimilarity_assemble_exit_code.txt"
    if "!EC!"=="0" (
      "%PY%" "%CODE%\run_foundation_multidomain_persite_linear_regression.py" ^
        --input-dir "%ROOT%\01_data\single_cell\intermediate\phospho_model_inputs\scimilarity_v1_1_gse300551_signal_seq_multidomain_v1" ^
        --output "%ROOT%\02_results\single_cell\20260601_scimilarity_v1_1_linear_regression_benchmark_v1" ^
        --method-name scimilarity_v1_1_ordinary_linear_regression >> "%LOGDIR%\scimilarity_linear_regression.log" 2>&1
      echo !ERRORLEVEL! > "%LOGDIR%\scimilarity_linear_regression_exit_code.txt"
    )
  )
) else (
  echo scimilarity model missing after download > "%LOGDIR%\scimilarity_missing.txt"
)

echo [%date% %time%] remaining foundation models done > "%LOGDIR%\done.txt"
endlocal
exit /b 0

echo [%date% %time%] waiting TranscriptFormer download >> "%LOGDIR%\run_remaining.log"
:WAIT_TF
if not exist "D:\data\lsy\models\transcriptformer\_logs\tf_sapiens_done.txt" (
  timeout /t 300 /nobreak >nul
  goto WAIT_TF
)
if exist "D:\data\lsy\models\transcriptformer\tf_sapiens\model_weights.pt" (
  echo [%date% %time%] running TranscriptFormer embeddings >> "%LOGDIR%\run_remaining.log"
  "%PY%" "%CODE%\prepare_transcriptformer_h5ad_inputs.py" --skip-existing --datasets %DATASETS% >> "%LOGDIR%\transcriptformer_prepare_h5ad.log" 2>&1
  set EC=!ERRORLEVEL!
  echo !EC! > "%LOGDIR%\transcriptformer_prepare_h5ad_exit_code.txt"
  if not "!EC!"=="0" goto SKIP_TF
  "%PY%" "%CODE%\run_transcriptformer_multidomain_embeddings.py" ^
    --h5ad-dir "01_data\single_cell\intermediate\foundation_model_h5ad_inputs_transcriptformer_geneid_v1" ^
    --datasets %DATASETS% ^
    --skip-existing >> "%LOGDIR%\transcriptformer_embedding.log" 2>&1
  set EC=!ERRORLEVEL!
  echo !EC! > "%LOGDIR%\transcriptformer_embedding_exit_code.txt"
  if "!EC!"=="0" (
    "%PY%" "%CODE%\extract_obsm_embeddings_to_model_input.py" ^
      --template-input-dir "%TEMPLATE%" ^
      --h5ad-manifest "%ROOT%\01_data\single_cell\intermediate\foundation_model_embeddings\transcriptformer_tf_sapiens_multidomain_v1\transcriptformer_embedding_manifest.tsv" ^
      --output-dir "%ROOT%\01_data\single_cell\intermediate\phospho_model_inputs\transcriptformer_tf_sapiens_gse300551_signal_seq_multidomain_v1" ^
      --obsm-key embeddings ^
      --datasets %DATASETS% ^
      --method-name transcriptformer_tf_sapiens >> "%LOGDIR%\transcriptformer_assemble.log" 2>&1
    set EC=!ERRORLEVEL!
    echo !EC! > "%LOGDIR%\transcriptformer_assemble_exit_code.txt"
    if "!EC!"=="0" (
      "%PY%" "%CODE%\run_foundation_multidomain_persite_linear_regression.py" ^
        --input-dir "%ROOT%\01_data\single_cell\intermediate\phospho_model_inputs\transcriptformer_tf_sapiens_gse300551_signal_seq_multidomain_v1" ^
        --output "%ROOT%\02_results\single_cell\20260601_transcriptformer_tf_sapiens_linear_regression_benchmark_v1" ^
        --method-name transcriptformer_tf_sapiens_ordinary_linear_regression >> "%LOGDIR%\transcriptformer_linear_regression.log" 2>&1
      echo !ERRORLEVEL! > "%LOGDIR%\transcriptformer_linear_regression_exit_code.txt"
    )
  )
  :SKIP_TF
) else (
  echo TranscriptFormer model missing after download > "%LOGDIR%\transcriptformer_missing.txt"
)

echo [%date% %time%] remaining foundation models done > "%LOGDIR%\done.txt"
endlocal
