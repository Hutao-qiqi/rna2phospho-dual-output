$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
if (Test-Path Variable:\PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Root = "D:\data\lsy\vm_lsy_parent\lsy"
$CodeDir = Join-Path $Root "03_code\single_cell\modeling"
$OutRoot = Join-Path $Root "01_data\single_cell\raw\external_bulk_phospho_validation_v1"
$LogDir = Join-Path $OutRoot "_logs\large_assets_v2"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stdout = Join-Path $LogDir "massive_formal_stdout.log"
$Stderr = Join-Path $LogDir "massive_formal_stderr.log"
$ExitCode = Join-Path $LogDir "massive_formal_exit_code.txt"
$Started = Join-Path $LogDir "massive_formal_started.txt"

"started $(Get-Date -Format o)" | Set-Content -Encoding UTF8 $Started
Remove-Item -ErrorAction SilentlyContinue $Stdout, $Stderr, $ExitCode

Set-Location $Root
try {
    & python (Join-Path $CodeDir "download_external_bulk_large_validation_assets_v2.py") `
        --source massive `
        --include-raw `
        --reuse-massive-manifest `
        --skip-massive-rel-path "quant/CurveCurator/Cysteine.zip" `
        --skip-massive-rel-path "quant/CurveCurator/FullProteome.zip" `
        --skip-massive-rel-path "quant/CurveCurator/Phospho.zip" `
        --skip-massive-rel-path "quant/CurveCurator/Ubi.zip" `
        --max-tries 20 `
        --massive-file-tries 2 `
        --massive-connect-tries 2 `
        --massive-ftp-timeout 60 `
        --massive-read-timeout 30 `
        --massive-process-stall-timeout 90 `
        --massive-process-poll 10 `
        --retry-wait 10 `
        1> $Stdout 2> $Stderr
    $Code = $LASTEXITCODE
}
catch {
    $_ | Out-File -Encoding UTF8 -Append $Stderr
    $Code = 1
}
"$Code" | Set-Content -Encoding UTF8 $ExitCode
exit $Code
