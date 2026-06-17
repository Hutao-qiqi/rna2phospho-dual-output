@echo off
set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set OUTPARENT=%ROOT%\02_results\model_validation
set OUT=%OUTPARENT%\20260602_m4_knowledge_graph_controls_e160_wsl
if exist "%OUT%" rmdir /s /q "%OUT%"
if exist "%OUTPARENT%\20260602_m4_wsl_process.stdout.log" del /q "%OUTPARENT%\20260602_m4_wsl_process.stdout.log"
if exist "%OUTPARENT%\20260602_m4_wsl_process.stderr.log" del /q "%OUTPARENT%\20260602_m4_wsl_process.stderr.log"
if exist "%OUTPARENT%\20260602_m4_wsl_process.exit.txt" del /q "%OUTPARENT%\20260602_m4_wsl_process.exit.txt"
wsl.exe -e bash /mnt/d/data/lsy/vm_lsy_parent/lsy/remote_scripts/launch_scp682_m4_wsl_parallel.sh > "%OUTPARENT%\20260602_m4_wsl_process.stdout.log" 2> "%OUTPARENT%\20260602_m4_wsl_process.stderr.log"
echo exit %ERRORLEVEL% > "%OUTPARENT%\20260602_m4_wsl_process.exit.txt"
