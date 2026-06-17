@echo off
setlocal
set CUDA_VISIBLE_DEVICES=0
set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set PKG=%ROOT%\SCP682_MAIN
set OUT=%ROOT%\02_results\model_validation\20260603_scp682_consistent_shrinkage_loss_e160_gpu0
set PY=D:\data\lsy\envs\scgpt\python.exe
mkdir "%OUT%" 2>NUL
mkdir "%OUT%\logs" 2>NUL
"%PY%" "%PKG%\scripts\train_scp682_main_v4_exact_scnet_gnn_consistent_loss.py" ^
  --package-dir "%PKG%" ^
  --prior-root "%PKG%\priors" ^
  --output-dir "%OUT%" ^
  --device cuda:0 ^
  --epochs 160 ^
  --lr 8e-5 ^
  --knn 10 ^
  --reduce-interval 30 ^
  --min-connect 5 ^
  --shrinkage 0.3 ^
  --v4-baseline-path "%PKG%\training_set\v4_phosphosite_baseline.parquet" ^
  --seed 20260522 ^
  --ppi-weight 0.08 ^
  --baseline-weight 0.08 ^
  --attention-l1 0.004 ^
  --batch-size 4 ^
  --hidden 64 ^
  --latent 32 ^
  --inter-dim 96 ^
  --embd-dim 32 ^
  --num-layers 1 ^
  > "%OUT%\logs\run.log" 2> "%OUT%\logs\run.stderr.log"
set EXITCODE=%ERRORLEVEL%
if %EXITCODE% EQU 0 (
  echo done>"%OUT%\done.txt"
) else (
  echo %EXITCODE%>"%OUT%\fatal.log"
)
exit /b %EXITCODE%
