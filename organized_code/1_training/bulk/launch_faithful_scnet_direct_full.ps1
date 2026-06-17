$out = "D:\data\lsy\vm_lsy_parent\lsy\02_results\model_validation\20260602_faithful_scnet_direct_bulk_c1"
New-Item -ItemType Directory -Force -Path $out | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $out "logs") | Out-Null
$cmd = "D:\data\lsy\vm_lsy_parent\lsy\SCP682_MAIN\scripts\run_faithful_scnet_direct_full_logged.cmd"
$p = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $cmd) -WindowStyle Hidden -PassThru
$p.Id | Out-File -Encoding ascii (Join-Path $out "run.pid")
"launched PID=$($p.Id) at $(Get-Date -Format o)" | Out-File -Encoding utf8 (Join-Path $out "run_status.txt")
