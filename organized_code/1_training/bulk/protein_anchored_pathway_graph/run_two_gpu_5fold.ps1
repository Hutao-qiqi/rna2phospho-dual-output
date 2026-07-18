param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutputDir "logs") | Out-Null

$Common = @(
    (Join-Path $ScriptDir "train.py"),
    "--project-root", $ProjectRoot,
    "--output-dir", $OutputDir
)

$Gpu0 = Start-Process -FilePath $Python -ArgumentList ($Common + @("--folds", "0,2,4", "--device", "cuda:0")) `
    -RedirectStandardOutput (Join-Path $OutputDir "logs\gpu0.stdout.log") `
    -RedirectStandardError (Join-Path $OutputDir "logs\gpu0.stderr.log") -PassThru -WindowStyle Hidden
$Gpu1 = Start-Process -FilePath $Python -ArgumentList ($Common + @("--folds", "1,3", "--device", "cuda:1")) `
    -RedirectStandardOutput (Join-Path $OutputDir "logs\gpu1.stdout.log") `
    -RedirectStandardError (Join-Path $OutputDir "logs\gpu1.stderr.log") -PassThru -WindowStyle Hidden

$Gpu0.WaitForExit()
$Gpu1.WaitForExit()
if ($Gpu0.ExitCode -ne 0 -or $Gpu1.ExitCode -ne 0) {
    throw "Training failed: GPU0 exit=$($Gpu0.ExitCode), GPU1 exit=$($Gpu1.ExitCode)"
}

& $Python (Join-Path $ScriptDir "summarize.py") --project-root $ProjectRoot --result-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Fold summarization failed with exit code $LASTEXITCODE"
}
