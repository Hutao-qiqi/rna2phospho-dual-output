@echo off
set SHARD_INDEX=%1
set DEVICE=%2
set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set PYTHON=D:\data\lsy\envs\scgpt\python.exe
set CODEDIR=%ROOT%\03_code\single_cell\modeling
set LOGDIR=%ROOT%\02_results\single_cell\20260519_scfoundation_ridge_gse300551_signal_v3\logs
set SHARDBASE=01_data\single_cell\intermediate\scfoundation_embeddings\cap12000_gse300551_signal_seq_v1_shards
set OUTDIR=%SHARDBASE%\shard%SHARD_INDEX%_of_2

if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set CUDA_VISIBLE_DEVICES=%DEVICE%
set PYTORCH_ALLOC_CONF=expandable_segments:True
set PYTHONWARNINGS=ignore

"%PYTHON%" "%CODEDIR%\precompute_scfoundation_embeddings_multidomain.py" --root "%ROOT%" --datasets gse300551_iccite_plex_kinase_2025 signal_seq_gse256403_hela_2024 signal_seq_gse256404_pdo_caf_2024 --batch-size 8 --max-gene-tokens 12000 --num-shards 2 --shard-index %SHARD_INDEX% --skip-existing --output-dir "%OUTDIR%" 1> "%LOGDIR%\embedding_shard%SHARD_INDEX%_bs8.log" 2> "%LOGDIR%\embedding_shard%SHARD_INDEX%_bs8.err"
set CODE=%ERRORLEVEL%
echo %CODE% > "%LOGDIR%\embedding_shard%SHARD_INDEX%_bs8.exit_code.txt"
exit /b %CODE%

