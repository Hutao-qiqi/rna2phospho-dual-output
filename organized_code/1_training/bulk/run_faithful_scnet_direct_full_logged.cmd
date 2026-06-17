@echo off
set OUT=D:\data\lsy\vm_lsy_parent\lsy\02_results\model_validation\20260602_faithful_scnet_direct_bulk_c1
if not exist %OUT% mkdir %OUT%
if not exist %OUT%\logs mkdir %OUT%\logs
echo started %DATE% %TIME% > %OUT%\run_status.txt
call D:\data\lsy\vm_lsy_parent\lsy\SCP682_MAIN\scripts\run_faithful_scnet_direct_full.cmd > %OUT%\logs\run.log 2> %OUT%\logs\run.stderr.log
echo finished %DATE% %TIME% >> %OUT%\run_status.txt
