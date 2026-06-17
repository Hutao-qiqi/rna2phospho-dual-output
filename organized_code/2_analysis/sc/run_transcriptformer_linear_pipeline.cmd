@echo off
setlocal EnableDelayedExpansion
set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set CODE=%ROOT%\03_code\single_cell\modeling
set TEMPLATE=%ROOT%\01_data\single_cell\intermediate\phospho_model_inputs\scgpt_frozen_gse300551_signal_seq_multidomain_v1
set PY=D:\Tools\anaconda3\python.exe
set LOGDIR=%ROOT%\02_results\single_cell\20260601_transcriptformer_tf_sapiens_linear_regression_benchmark_v1\logs
set DATASETS=iccite_seq_tcell_2025 phospho_seq_blair_2025_phospho_multi qurie_seq_bjab_2021 vivo_seq_th17_2025 gse300551_iccite_plex_kinase_2025 signal_seq_gse256403_hela_2024
mkdir "%LOGDIR%" 2>nul

echo [%date% %time%] TranscriptFormer linear task start > "%LOGDIR%\formal_stdout.log"
:WAIT_TF
if not exist "D:\data\lsy\models\transcriptformer\_logs\tf_sapiens_done.txt" (
  timeout /t 120 /nobreak >nul
  goto WAIT_TF
)
if not exist "D:\data\lsy\models\transcriptformer\tf_sapiens\model_weights.pt" (
  echo TranscriptFormer model missing after download > "%LOGDIR%\fatal.log"
  exit /b 2
)

"%PY%" "%CODE%\prepare_transcriptformer_h5ad_inputs.py" --skip-existing --datasets %DATASETS% >> "%LOGDIR%\formal_stdout.log" 2> "%LOGDIR%\formal_stderr.log"
set EC=%ERRORLEVEL%
if not "%EC%"=="0" (
  echo %EC% > "%LOGDIR%\formal_exit_code.txt"
  exit /b %EC%
)

"%PY%" "%CODE%\run_transcriptformer_multidomain_embeddings.py" ^
  --h5ad-dir "01_data\single_cell\intermediate\foundation_model_h5ad_inputs_transcriptformer_geneid_v1" ^
  --datasets %DATASETS% ^
  --batch-size 32 ^
  --cuda-visible-devices 1 ^
  --num-workers 8 ^
  --cpu-threads 16 ^
  --skip-existing >> "%LOGDIR%\formal_stdout.log" 2>> "%LOGDIR%\formal_stderr.log"
set EC=%ERRORLEVEL%
if not "%EC%"=="0" (
  echo %EC% > "%LOGDIR%\formal_exit_code.txt"
  exit /b %EC%
)

"%PY%" "%CODE%\extract_obsm_embeddings_to_model_input.py" ^
  --template-input-dir "%TEMPLATE%" ^
  --h5ad-manifest "%ROOT%\01_data\single_cell\intermediate\foundation_model_embeddings\transcriptformer_tf_sapiens_multidomain_v1\transcriptformer_embedding_manifest.tsv" ^
  --output-dir "%ROOT%\01_data\single_cell\intermediate\phospho_model_inputs\transcriptformer_tf_sapiens_gse300551_signal_seq_multidomain_v1" ^
  --obsm-key embeddings ^
  --datasets %DATASETS% ^
  --method-name transcriptformer_tf_sapiens >> "%LOGDIR%\formal_stdout.log" 2>> "%LOGDIR%\formal_stderr.log"
set EC=%ERRORLEVEL%
if not "%EC%"=="0" (
  echo %EC% > "%LOGDIR%\formal_exit_code.txt"
  exit /b %EC%
)

"%PY%" "%CODE%\run_foundation_multidomain_persite_linear_regression.py" ^
  --input-dir "%ROOT%\01_data\single_cell\intermediate\phospho_model_inputs\transcriptformer_tf_sapiens_gse300551_signal_seq_multidomain_v1" ^
  --output "%ROOT%\02_results\single_cell\20260601_transcriptformer_tf_sapiens_linear_regression_benchmark_v1" ^
  --method-name transcriptformer_tf_sapiens_ordinary_linear_regression >> "%LOGDIR%\formal_stdout.log" 2>> "%LOGDIR%\formal_stderr.log"
set EC=%ERRORLEVEL%
echo %EC% > "%LOGDIR%\formal_exit_code.txt"
if "%EC%"=="0" echo [%date% %time%] done > "%LOGDIR%\done.txt"
endlocal
