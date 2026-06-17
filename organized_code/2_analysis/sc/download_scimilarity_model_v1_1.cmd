@echo off
set OUTDIR=D:\data\lsy\models\scimilarity
set LOGDIR=D:\data\lsy\models\scimilarity\_logs
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
D:\Tools\aria2-1.37.0-win-64bit-build1\aria2c.exe -x 8 -s 8 -k 4M --continue=true --file-allocation=none --dir="%OUTDIR%" --out=model_v1.1.tar.gz "https://zenodo.org/records/10685499/files/model_v1.1.tar.gz?download=1" > "%LOGDIR%\download_stdout.log" 2> "%LOGDIR%\download_stderr.log"
echo %ERRORLEVEL% > "%LOGDIR%\download_exit_code.txt"
