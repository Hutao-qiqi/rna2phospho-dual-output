@echo off
setlocal

set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set PY=D:\Tools\anaconda3\python.exe
set MODEL_INPUT=%ROOT%\01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_gse300551_signal_seq_multidomain_v1
set CACHE_ROOT=%ROOT%\01_data\single_cell\intermediate\protein_prediction_cache\scTranslator_sc12_all_protein_cache_v1
set MERGED=%ROOT%\01_data\single_cell\intermediate\protein_prediction_cache\scTranslator_sc12_all_protein_cache_merged_v1
set RESULT=%ROOT%\02_results\single_cell\20260621_scp682_sc12_protein_residual_v1
set LOG_DIR=%RESULT%\logs

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

"%PY%" "%ROOT%\03_code\single_cell\modeling\audit_sc12_all_protein_cache.py" 1>"%LOG_DIR%\merge_all_protein_audit_stdout.log" 2>"%LOG_DIR%\merge_all_protein_audit_stderr.log"
if errorlevel 1 (
  echo %ERRORLEVEL%>"%LOG_DIR%\merge_all_protein_exit_code.txt"
  exit /b %ERRORLEVEL%
)

"%PY%" "%ROOT%\03_code\single_cell\modeling\merge_sc12_protein_cache_by_cell_metadata.py" --model-input-dir "%MODEL_INPUT%" --cache-root "%CACHE_ROOT%" --output-dir "%MERGED%" 1>"%LOG_DIR%\merge_all_protein_stdout.log" 2>"%LOG_DIR%\merge_all_protein_stderr.log"
set CODE=%ERRORLEVEL%
echo %CODE%>"%LOG_DIR%\merge_all_protein_exit_code.txt"
if not "%CODE%"=="0" exit /b %CODE%

"%PY%" "%ROOT%\03_code\single_cell\modeling\sc12_all_protein_preflight_summary.py" 1>"%LOG_DIR%\all_protein_preflight_stdout.log" 2>"%LOG_DIR%\all_protein_preflight_stderr.log"
set PREFLIGHT_CODE=%ERRORLEVEL%
echo %PREFLIGHT_CODE%>"%LOG_DIR%\all_protein_preflight_exit_code.txt"
if not "%PREFLIGHT_CODE%"=="0" exit /b %PREFLIGHT_CODE%

echo merged_all_protein_cache_ready>"%RESULT%\sc12_all_protein_cache_ready.txt"
exit /b 0
