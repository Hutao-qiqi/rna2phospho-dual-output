@echo off
setlocal

set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set OUT=%ROOT%\02_results\single_cell\20260621_scp682_sc12_protein_residual_v1
set LOG_DIR=%OUT%\logs
set RUNNER=%ROOT%\03_code\single_cell\modeling\run_scp682_sc12_protein_residual_formal.bat

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [%date% %time%] start SCP682-SC12 formal training > "%LOG_DIR%\formal_stdout.log"
call "%RUNNER%" >> "%LOG_DIR%\formal_stdout.log" 2> "%LOG_DIR%\formal_stderr.log"
set CODE=%ERRORLEVEL%
echo %CODE% > "%LOG_DIR%\formal_exit_code.txt"
echo [%date% %time%] exit_code=%CODE% >> "%LOG_DIR%\formal_stdout.log"
exit /b %CODE%
