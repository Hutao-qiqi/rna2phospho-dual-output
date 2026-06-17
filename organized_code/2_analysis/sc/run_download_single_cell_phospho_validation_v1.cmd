@echo off
set root=D:\data\lsy\vm_lsy_parent\lsy
set out=%root%\01_data\single_cell\raw\external_single_cell_phospho_validation_v1
set logs=%out%\_logs
if not exist "%logs%" mkdir "%logs%"
cd /d "%root%"
echo %date% %time% > "%logs%\started.txt"
"D:\Tools\anaconda3\python.exe" "03_code\single_cell\data_download\download_single_cell_phospho_validation_v1.py" > "%logs%\formal_stdout.log" 2> "%logs%\formal_stderr.log"
set code=%ERRORLEVEL%
echo %code% > "%logs%\formal_exit_code.txt"
exit /b %code%
