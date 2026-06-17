@echo off
setlocal
set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set MODEL_DIR=D:\data\lsy\models\transcriptformer
set LOG_DIR=%MODEL_DIR%\_logs
mkdir "%LOG_DIR%" 2>nul
mkdir "%MODEL_DIR%" 2>nul
cd /d "%MODEL_DIR%"

echo [%date% %time%] download tf_sapiens start > "%LOG_DIR%\tf_sapiens_download.log"
D:\Tools\aria2-1.37.0-win-64bit-build1\aria2c.exe ^
  -x 8 -s 8 -k 1M --continue=true --auto-file-renaming=false --allow-overwrite=true ^
  -d "%MODEL_DIR%" ^
  -o "tf_sapiens.tar.gz" ^
  "https://czi-transcriptformer.s3.amazonaws.com/weights/tf_sapiens.tar.gz" ^
  >> "%LOG_DIR%\tf_sapiens_download.log" 2>&1
set EC=%ERRORLEVEL%
echo %EC% > "%LOG_DIR%\tf_sapiens_download_exit_code.txt"
if not "%EC%"=="0" exit /b %EC%

echo [%date% %time%] extract tf_sapiens >> "%LOG_DIR%\tf_sapiens_download.log"
tar -xf "%MODEL_DIR%\tf_sapiens.tar.gz" -C "%MODEL_DIR%" >> "%LOG_DIR%\tf_sapiens_download.log" 2>&1
set EC=%ERRORLEVEL%
echo %EC% > "%LOG_DIR%\tf_sapiens_extract_exit_code.txt"
if not "%EC%"=="0" exit /b %EC%
echo [%date% %time%] done > "%LOG_DIR%\tf_sapiens_done.txt"
endlocal
