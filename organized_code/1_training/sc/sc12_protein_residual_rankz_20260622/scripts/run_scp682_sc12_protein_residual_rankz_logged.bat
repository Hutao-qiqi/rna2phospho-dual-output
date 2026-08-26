@echo off
setlocal

set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set OUT=%ROOT%\02_results\single_cell\20260622_scp682_sc12_protein_residual_rankz_v1
set LOG=%OUT%\logs
if not exist "%LOG%" mkdir "%LOG%"

call "%ROOT%\03_code\single_cell\modeling\run_scp682_sc12_protein_residual_rankz.bat" > "%LOG%\formal_stdout.log" 2> "%LOG%\formal_stderr.log"
echo %ERRORLEVEL% > "%LOG%\formal_exit_code.txt"
exit /b %ERRORLEVEL%
