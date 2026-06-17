$ErrorActionPreference = "Stop"

$Root = "D:\data\lsy\vm_lsy_parent\lsy"
$CodeDir = Join-Path $Root "03_code\single_cell\modeling"
$RunScript = Join-Path $CodeDir "run_external_bulk_pride_large_assets_v2_download.ps1"
$OutRoot = Join-Path $Root "01_data\single_cell\raw\external_bulk_phospho_validation_v1"
$LogDir = Join-Path $OutRoot "_logs\large_assets_v2"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$TaskName = "SCP682_EXTERNAL_BULK_PRIDE_LARGE_ASSETS_V2"
$TaskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$RunScript`""
$TaskLog = Join-Path $LogDir "pride_schtask_launch.log"

"create $(Get-Date -Format o) $TaskCommand" | Set-Content -Encoding UTF8 $TaskLog
& schtasks.exe /Create /TN $TaskName /TR $TaskCommand /SC ONCE /ST 23:59 /F | Add-Content -Encoding UTF8 $TaskLog
& schtasks.exe /Run /TN $TaskName | Add-Content -Encoding UTF8 $TaskLog
"run $(Get-Date -Format o)" | Add-Content -Encoding UTF8 $TaskLog
