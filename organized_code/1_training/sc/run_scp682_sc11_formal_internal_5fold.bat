@echo off
set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set SCP682_PORTABLE=%ROOT%\SCP682_PORTABLE
set TRANSFER=%ROOT%\01_data\single_cell\intermediate\scp682_main_sc_transfer_prior_v1
set BASE=%ROOT%\02_results\single_cell\20260529_scp682_sc11_current_main_internal_5fold_v1
set LOG=%BASE%\logs
if not exist "%BASE%" mkdir "%BASE%"
if not exist "%LOG%" mkdir "%LOG%"
if exist "%LOG%\formal_exit_code.txt" del "%LOG%\formal_exit_code.txt"
echo %DATE% %TIME% > "%LOG%\formal_started.txt"

"D:\Tools\anaconda3\python.exe" "%ROOT%\03_code\single_cell\modeling\export_scp682_main_sc_transfer_prior.py" ^
  --scp682-main-dir "%SCP682_PORTABLE%" ^
  --output-dir "%TRANSFER%" ^
  --device cuda:1 ^
  > "%LOG%\export_scp682_main_stdout.log" 2> "%LOG%\export_scp682_main_stderr.log"
if errorlevel 1 (
  echo %ERRORLEVEL% > "%LOG%\formal_exit_code.txt"
  echo %DATE% %TIME% > "%LOG%\formal_finished.txt"
  exit /b %ERRORLEVEL%
)

for %%F in (1 2 3 4 5) do (
  set OUT=%BASE%\fold_%%F
  call :RUNFOLD %%F
  if errorlevel 1 (
    echo fold %%F failed with errorlevel %ERRORLEVEL% >> "%LOG%\formal_stderr.log"
    echo %ERRORLEVEL% > "%LOG%\formal_exit_code.txt"
    echo %DATE% %TIME% > "%LOG%\formal_finished.txt"
    exit /b %ERRORLEVEL%
  )
)

"D:\Tools\anaconda3\python.exe" "%ROOT%\03_code\single_cell\modeling\summarize_scp682_sc11_internal_5fold.py" ^
  --base-dir "%BASE%" ^
  > "%LOG%\summary_stdout.log" 2> "%LOG%\summary_stderr.log"

echo %ERRORLEVEL% > "%LOG%\formal_exit_code.txt"
echo %DATE% %TIME% > "%LOG%\formal_finished.txt"
exit /b %ERRORLEVEL%

:RUNFOLD
set FOLD=%1
set OUT=%BASE%\fold_%FOLD%
set FLOG=%OUT%\logs
if not exist "%OUT%" mkdir "%OUT%"
if not exist "%FLOG%" mkdir "%FLOG%"
if not exist "%OUT%\models" mkdir "%OUT%\models"
if not exist "%OUT%\tables" mkdir "%OUT%\tables"
if not exist "%OUT%\reports" mkdir "%OUT%\reports"
echo %DATE% %TIME% > "%FLOG%\formal_started.txt"

"D:\Tools\anaconda3\python.exe" "%ROOT%\03_code\single_cell\modeling\train_scp682_sc11_expanded_scnet_site_gnn.py" ^
  --root "%ROOT%" ^
  --output-dir "%OUT%" ^
  --train-datasets iccite_seq_tcell_2025,qurie_seq_bjab_2021 ^
  --train-control-only-datasets __none__ ^
  --holdout-datasets gse300551_iccite_plex_kinase_2025,phospho_seq_blair_2025_phospho_multi,vivo_seq_th17_2025,signal_seq_gse256403_hela_2024,signal_seq_gse256404_pdo_caf_2024 ^
  --include-drug-delta-eval ^
  --epochs 140 ^
  --patience 26 ^
  --batch-size 2048 ^
  --batch-log-interval 20 ^
  --eval-batch-size 4096 ^
  --context-cells 1024 ^
  --max-eval-train-cells 40000 ^
  --hidden 256 ^
  --pathway-layers 2 ^
  --attention-heads 4 ^
  --dropout 0.12 ^
  --lr 0.00018 ^
  --weight-decay 0.0001 ^
  --recon-weight 1.0 ^
  --prior-weight 0.03 ^
  --prior-neighbors 12 ^
  --prior-temperature 0.08 ^
  --prior-min-similarity 0.15 ^
  --prior-steps-per-epoch 12 ^
  --prior-batch-size 1024 ^
  --prior-datasets iccite_seq_tcell_2025,qurie_seq_bjab_2021 ^
  --scp682-main-transfer-dir 01_data\single_cell\intermediate\scp682_main_sc_transfer_prior_v1 ^
  --transfer-alpha 0.55 ^
  --transfer-attention-weight 0.06 ^
  --full-transfer-scale 0.45 ^
  --teacher-distill-weight 0.08 ^
  --site-graph-weight 0.03 ^
  --site-graph-scale 0.25 ^
  --site-graph-topk 12 ^
  --site-graph-max-aux-nodes 12000 ^
  --huber-beta 0.5 ^
  --loss-type huber ^
  --warmup-epochs 8 ^
  --val-fraction 0 ^
  --val-cells 0 ^
  --cv-folds 5 ^
  --cv-fold %FOLD% ^
  --max-target-weight 4.0 ^
  --balance-dataset-size 60000 ^
  --grad-clip 1.0 ^
  --seed 682522 ^
  --device cuda:1 ^
  --attention-cells 20000 ^
  > "%FLOG%\formal_stdout.log" 2> "%FLOG%\formal_stderr.log"

echo %ERRORLEVEL% > "%FLOG%\formal_exit_code.txt"
echo %DATE% %TIME% > "%FLOG%\formal_finished.txt"
exit /b %ERRORLEVEL%
