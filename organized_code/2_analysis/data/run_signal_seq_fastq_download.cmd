@echo off
set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set OUT=%ROOT%\01_data\single_cell\raw\external_single_cell_phospho_validation_v1\SIGNAL-seq_GSE256405
set LOGDIR=%OUT%\_logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
cd /d "%ROOT%"
"D:\Tools\anaconda3\python.exe" "%ROOT%\03_code\single_cell\data_download\download_signal_seq_v1.py" --root "%ROOT%" --download-fastq >> "%LOGDIR%\download_fastq_stdout.log" 2>> "%LOGDIR%\download_fastq_stderr.log"
echo %ERRORLEVEL% > "%LOGDIR%\download_fastq_exit_code.txt"
