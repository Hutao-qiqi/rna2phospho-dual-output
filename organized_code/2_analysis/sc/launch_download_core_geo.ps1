$ErrorActionPreference = "Stop"

$script = "D:\lsy\03_code\single_cell\download\download_core_geo.ps1"
$logDir = "D:\lsy\02_results\single_cell\20260509_data_inventory\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stdout = Join-Path $logDir "download_core_geo.stdout.log"
$stderr = Join-Path $logDir "download_core_geo.stderr.log"

$p = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script) `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru `
    -WindowStyle Hidden

$p.Id | Set-Content (Join-Path $logDir "download_core_geo.pid")
"started_pid=$($p.Id)"

