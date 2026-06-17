@echo off
setlocal enabledelayedexpansion

set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set TRANSFER=%ROOT%\01_data\single_cell\intermediate\scp682_main_pathway_token_transfer_prior_v1
set WARM=%ROOT%\02_results\single_cell\20260520_scp682_sc7_scfoundation_scp682_main_transfer_v1
set BASE_OUT=%ROOT%\02_results\single_cell\20260602_scp682_sc11_m4_graph_controls
set PY=D:\Tools\anaconda3\python.exe
set SCRIPT=%ROOT%\03_code\single_cell\modeling\train_scp682_sc11_expanded_scnet_site_gnn.py

if not exist "%BASE_OUT%" mkdir "%BASE_OUT%"
if not exist "%BASE_OUT%\logs" mkdir "%BASE_OUT%\logs"
echo %DATE% %TIME% > "%BASE_OUT%\logs\m4_started.txt"

if not exist "%TRANSFER%\tables" (
  echo missing SCP682 main pathway-token transfer package > "%BASE_OUT%\logs\fatal.log"
  exit /b 1
)

if not exist "%WARM%\models\scp682_sc7_final.pt" (
  echo missing SCP682-SC warm-start package > "%BASE_OUT%\logs\fatal.log"
  exit /b 1
)

call :run_one rewired_all rewired_all
call :run_one no_copheemap no_copheemap
call :run_one no_copheeksa no_copheeksa
call :run_one no_kstar no_kstar

echo %DATE% %TIME% > "%BASE_OUT%\done.txt"
exit /b 0

:run_one
set RUN_NAME=%1
set EDGE_MODE=%2
set OUT=%BASE_OUT%\%RUN_NAME%
set LOG=%OUT%\logs
if not exist "%OUT%" mkdir "%OUT%"
if not exist "%OUT%\models" mkdir "%OUT%\models"
if not exist "%OUT%\tables" mkdir "%OUT%\tables"
if not exist "%OUT%\reports" mkdir "%OUT%\reports"
if not exist "%LOG%" mkdir "%LOG%"

echo %DATE% %TIME% start %RUN_NAME% edge_mode=%EDGE_MODE% >> "%BASE_OUT%\logs\run_status.txt"
"%PY%" "%SCRIPT%" ^
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
  --scp682-main-pathway-token-transfer-dir 01_data\single_cell\intermediate\scp682_main_pathway_token_transfer_prior_v1 ^
  --transfer-alpha 0.55 ^
  --transfer-attention-weight 0.06 ^
  --full-transfer-scale 0.45 ^
  --teacher-distill-weight 0.08 ^
  --site-graph-weight 0.03 ^
  --site-graph-scale 0.25 ^
  --site-graph-topk 12 ^
  --site-graph-max-aux-nodes 12000 ^
  --site-graph-edge-mode %EDGE_MODE% ^
  --site-graph-rewire-seed 20260602 ^
  --site-graph-rewire-swaps-per-edge 10 ^
  --warm-start-model 02_results\single_cell\20260520_scp682_sc7_scfoundation_scp682_main_transfer_v1\models\scp682_sc7_final.pt ^
  --huber-beta 0.5 ^
  --loss-type huber ^
  --warmup-epochs 8 ^
  --val-fraction 0.12 ^
  --val-cells 30000 ^
  --max-target-weight 4.0 ^
  --balance-dataset-size 60000 ^
  --grad-clip 1.0 ^
  --seed 682522 ^
  --device cuda:1 ^
  --attention-cells 20000 ^
  > "%LOG%\stdout.log" 2> "%LOG%\stderr.log"

if errorlevel 1 (
  echo %DATE% %TIME% failed %RUN_NAME% >> "%BASE_OUT%\logs\run_status.txt"
  exit /b %ERRORLEVEL%
)
echo %DATE% %TIME% done %RUN_NAME% >> "%BASE_OUT%\logs\run_status.txt"
exit /b 0
