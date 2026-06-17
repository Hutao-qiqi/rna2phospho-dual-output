@echo off
set ROOT=D:\data\lsy\vm_lsy_parent\lsy
set PYTHON=D:\Tools\anaconda3\envs\scvi-env\python.exe
set SCRIPT=%ROOT%\remote_scripts\launch_scp682_consistent_graph_controls_e160_windows.py
set OUT=%ROOT%\02_results\model_validation\20260603_consistent_graph_controls_e160_windows
if not exist "%OUT%\logs" mkdir "%OUT%\logs"
echo start %DATE% %TIME% > "%OUT%\logs\launcher.start.txt"
start "scp682_consistent_graph_controls" /min cmd /c ""%PYTHON%" "%SCRIPT%" > "%OUT%\logs\launcher.stdout.log" 2> "%OUT%\logs\launcher.stderr.log""
