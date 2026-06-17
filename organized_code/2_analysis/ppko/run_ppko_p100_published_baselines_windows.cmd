@echo off
setlocal
set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set PY=D:\Tools\anaconda3\python.exe
set SCRIPT=%ROOT%\03_code\single_cell\modeling\evaluate_ppko_p100_published_baselines.py

"%PY%" "%SCRIPT%" --data-root "%ROOT%" --device cuda:0
