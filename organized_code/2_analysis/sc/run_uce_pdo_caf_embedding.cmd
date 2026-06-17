@echo off
set CUDA_VISIBLE_DEVICES=1
set PY=D:\data\lsy\envs\scgpt\python.exe
set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set UCE_REPO=D:\data\lsy\repos\UCE
set SCRIPT=D:\data\lsy\vm_lsy_parent\lsy\03_code\single_cell\modeling\run_uce_multidomain_embeddings.py
set LOGDIR=D:\data\lsy\vm_lsy_parent\lsy\01_data\single_cell\intermediate\foundation_model_embeddings\uce_4layer_multidomain_v1\_logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
"%PY%" "%SCRIPT%" --root "%ROOT%" --uce-repo "%UCE_REPO%" --datasets signal_seq_gse256404_pdo_caf_2024 --batch-size 100 --skip-existing > "%LOGDIR%\pdo_caf_stdout.log" 2> "%LOGDIR%\pdo_caf_stderr.log"
echo %ERRORLEVEL% > "%LOGDIR%\pdo_caf_exit_code.txt"
