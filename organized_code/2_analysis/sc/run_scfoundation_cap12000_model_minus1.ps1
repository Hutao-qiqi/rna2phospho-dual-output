$ErrorActionPreference = "Stop"

$root = "D:\data\lsy\vm_lsy_parent\lsy"
$python = "D:\data\lsy\envs\scgpt\python.exe"
$codeDir = Join-Path $root "03_code\single_cell\modeling"
$runDir = Join-Path $root "02_results\single_cell\20260512_scfoundation_cap12000_model_minus1_cell_state_baseline"
$logDir = Join-Path $runDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$transcript = Join-Path $logDir "run.log"
$fatal = Join-Path $logDir "fatal.log"
$done = Join-Path $logDir "done.txt"
if (Test-Path $fatal) { Remove-Item -LiteralPath $fatal -Force }
if (Test-Path $done) { Remove-Item -LiteralPath $done -Force }

try {
    Start-Transcript -Path $transcript -Append | Out-Null
    Write-Host "start_time=$(Get-Date -Format o)"
    & $python (Join-Path $codeDir "run_cell_state_baseline.py") `
        --input-dir (Join-Path $root "01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_masked_multisite_v1") `
        --output $runDir `
        --seed 13
    if ($LASTEXITCODE -ne 0) {
        throw "model_minus1 failed with exit code $LASTEXITCODE"
    }
    "done_time=$(Get-Date -Format o)" | Set-Content -Path $done -Encoding utf8
    Write-Host "done_time=$(Get-Date -Format o)"
    Stop-Transcript | Out-Null
    exit 0
}
catch {
    $_ | Out-String | Set-Content -Path $fatal -Encoding utf8
    Write-Host $_
    try { Stop-Transcript | Out-Null } catch {}
    exit 1
}
