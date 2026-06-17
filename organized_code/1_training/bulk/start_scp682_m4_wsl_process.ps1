$ErrorActionPreference = "Stop"

$Root = "D:\data\lsy\vm_lsy_parent\lsy"
$OutParent = Join-Path $Root "02_results\model_validation"
$Out = Join-Path $OutParent "20260602_m4_knowledge_graph_controls_e160_wsl"
$Stdout = Join-Path $OutParent "20260602_m4_wsl_process.stdout.log"
$Stderr = Join-Path $OutParent "20260602_m4_wsl_process.stderr.log"
$PidFile = Join-Path $OutParent "20260602_m4_wsl_process.pid"

Remove-Item -Recurse -Force $Out -ErrorAction SilentlyContinue
Remove-Item -Force $Stdout,$Stderr,$PidFile -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $OutParent | Out-Null

$Arg = "-e bash /mnt/d/data/lsy/vm_lsy_parent/lsy/remote_scripts/launch_scp682_m4_wsl_parallel.sh"
$Process = Start-Process -FilePath "wsl.exe" `
    -ArgumentList $Arg `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden `
    -PassThru

$Process.Id | Set-Content -Path $PidFile
Write-Output "started wsl.exe pid=$($Process.Id)"
