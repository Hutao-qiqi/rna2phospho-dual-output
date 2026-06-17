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

$Stdout = Join-Path $LogDir "pride_formal_stdout.log"
$Stderr = Join-Path $LogDir "pride_formal_stderr.log"
$ExitCode = Join-Path $LogDir "pride_formal_exit_code.txt"
$Started = Join-Path $LogDir "pride_formal_started.txt"

"started $(Get-Date -Format o)" | Set-Content -Encoding UTF8 $Started
Remove-Item -ErrorAction SilentlyContinue $Stdout, $Stderr, $ExitCode

Set-Location $Root
try {
    & python (Join-Path $CodeDir "download_external_bulk_large_validation_assets_v2.py") `
        --source pride `
        --skip-raw `
        --pride-max-file-gb 5 `
        --pride-parallel 4 `
        --pride-connections 8 `
        --max-tries 20 `
        --retry-wait 20 `
        1> $Stdout 2> $Stderr
    $Code = $LASTEXITCODE
}
catch {
    $_ | Out-File -Encoding UTF8 -Append $Stderr
    $Code = 1
}
"$Code" | Set-Content -Encoding UTF8 $ExitCode
exit $Code
