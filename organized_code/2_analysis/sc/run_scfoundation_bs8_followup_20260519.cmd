@echo off
setlocal enabledelayedexpansion
set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set PYTHON=D:\data\lsy\envs\scgpt\python.exe
set CODEDIR=%ROOT%\03_code\single_cell\modeling
set LOGDIR=%ROOT%\02_results\single_cell\20260519_scfoundation_ridge_gse300551_signal_v3\logs
set RESULT=%ROOT%\02_results\single_cell\20260519_scfoundation_ridge_gse300551_signal_v3
set SHARDBASE=01_data\single_cell\intermediate\scfoundation_embeddings\cap12000_gse300551_signal_seq_v1_shards
set MERGED=01_data\single_cell\intermediate\scfoundation_embeddings\cap12000_gse300551_signal_seq_v1
set INPUTOUT=01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_gse300551_signal_seq_multidomain_v1
set TEMPLATE=01_data\single_cell\intermediate\phospho_model_inputs\geneformer_gse300551_signal_seq_multidomain_v1
set OLDEMB=01_data\single_cell\intermediate\scfoundation_embeddings\cap12000_cell_all_v1

if not exist "%LOGDIR%" mkdir "%LOGDIR%"
echo [%date% %time%] bs8 followup start >> "%LOGDIR%\bs8_followup.log"

:WAIT_BS8
if not exist "%LOGDIR%\embedding_shard1_bs8.exit_code.txt" (
  echo [%date% %time%] waiting shard1 bs8 >> "%LOGDIR%\bs8_followup.log"
  timeout /t 300 /nobreak >nul
  goto WAIT_BS8
)

set /p SHARD1_CODE=<"%LOGDIR%\embedding_shard1_bs8.exit_code.txt"
set SHARD1_CODE=%SHARD1_CODE: =%
if not "%SHARD1_CODE%"=="0" (
  echo [%date% %time%] shard1 bs8 failed code=%SHARD1_CODE% >> "%LOGDIR%\fatal.log"
  echo 1 > "%LOGDIR%\formal_exit_code.txt"
  exit /b 1
)
echo 0 > "%LOGDIR%\embedding_shard1.exit_code.txt"
echo [%date% %time%] shard1 bs8 complete >> "%LOGDIR%\bs8_followup.log"

"%PYTHON%" "%CODEDIR%\merge_scfoundation_multidomain_shards.py" --root "%ROOT%" --datasets gse300551_iccite_plex_kinase_2025 signal_seq_gse256403_hela_2024 signal_seq_gse256404_pdo_caf_2024 --shard-dirs "%SHARDBASE%\shard0_of_2" "%SHARDBASE%\shard1_of_2" --output-dir "%MERGED%" 1>> "%LOGDIR%\formal_stdout.log" 2>> "%LOGDIR%\formal_stderr.log"
if errorlevel 1 (
  echo [%date% %time%] merge failed >> "%LOGDIR%\fatal.log"
  echo 1 > "%LOGDIR%\formal_exit_code.txt"
  exit /b 1
)
echo [%date% %time%] merge complete >> "%LOGDIR%\bs8_followup.log"

"%PYTHON%" "%CODEDIR%\build_scfoundation_multidomain_model_input.py" --root "%ROOT%" --template-input "%TEMPLATE%" --source-dirs "%OLDEMB%" "%MERGED%" --output-dir "%INPUTOUT%" 1>> "%LOGDIR%\formal_stdout.log" 2>> "%LOGDIR%\formal_stderr.log"
if errorlevel 1 (
  echo [%date% %time%] assemble input failed >> "%LOGDIR%\fatal.log"
  echo 1 > "%LOGDIR%\formal_exit_code.txt"
  exit /b 1
)
echo [%date% %time%] input assemble complete >> "%LOGDIR%\bs8_followup.log"

"%PYTHON%" "%CODEDIR%\run_scfoundation_multidomain_persite_ridge.py" --input-dir "%ROOT%\%INPUTOUT%" --output "%RESULT%" --train-datasets iccite_seq_tcell_2025,qurie_seq_bjab_2021 --test-datasets gse300551_iccite_plex_kinase_2025,signal_seq_gse256403_hela_2024,signal_seq_gse256404_pdo_caf_2024,phospho_seq_blair_2025_phospho_multi,vivo_seq_th17_2025 --method-name scfoundation_cap12000_persite_ridge_iccite_qurie 1>> "%LOGDIR%\formal_stdout.log" 2>> "%LOGDIR%\formal_stderr.log"
if errorlevel 1 (
  echo [%date% %time%] ridge failed >> "%LOGDIR%\fatal.log"
  echo 1 > "%LOGDIR%\formal_exit_code.txt"
  exit /b 1
)

echo 0 > "%LOGDIR%\formal_exit_code.txt"
echo [%date% %time%] all done >> "%LOGDIR%\bs8_followup.log"
exit /b 0
