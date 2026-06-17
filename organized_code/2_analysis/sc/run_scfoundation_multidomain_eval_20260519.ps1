$ErrorActionPreference = "Stop"

$root = "D:\data\lsy\vm_lsy_parent\lsy"
$python = "D:\data\lsy\envs\scgpt\python.exe"
$codeDir = Join-Path $root "03_code\single_cell\modeling"
$logDir = Join-Path $root "02_results\single_cell\20260519_scfoundation_ridge_gse300551_signal_v3\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:PYTORCH_ALLOC_CONF = "expandable_segments:True"

function Run-Step {
    param(
        [string]$Name,
        [scriptblock]$Block
    )
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] START $Name" | Tee-Object -FilePath (Join-Path $logDir "formal_stdout.log") -Append
    & $Block *>> (Join-Path $logDir "formal_stdout.log")
    if ($LASTEXITCODE -ne 0) {
        $code = $LASTEXITCODE
        "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] FAIL $Name exit=$code" | Tee-Object -FilePath (Join-Path $logDir "fatal.log") -Append
        Set-Content -Path (Join-Path $logDir "formal_exit_code.txt") -Value $code
        exit $code
    }
    "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] DONE $Name" | Tee-Object -FilePath (Join-Path $logDir "formal_stdout.log") -Append
}

$shardBase = "01_data\single_cell\intermediate\scfoundation_embeddings\cap12000_gse300551_signal_seq_v1_shards"
$merged = "01_data\single_cell\intermediate\scfoundation_embeddings\cap12000_gse300551_signal_seq_v1"
$inputOut = "01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_gse300551_signal_seq_multidomain_v1"
$template = "01_data\single_cell\intermediate\phospho_model_inputs\geneformer_gse300551_signal_seq_multidomain_v1"
$oldEmb = "01_data\single_cell\intermediate\scfoundation_embeddings\cap12000_cell_all_v1"
$result = Join-Path $root "02_results\single_cell\20260519_scfoundation_ridge_gse300551_signal_v3"

$procs = @()
for ($i = 0; $i -lt 2; $i++) {
    $gpu = "$i"
    $child = Join-Path $codeDir "run_scfoundation_embedding_shard_20260519.cmd"
    $procs += Start-Process -FilePath $child `
        -ArgumentList @("$i", $gpu) `
        -WindowStyle Hidden `
        -PassThru
}

Wait-Process -InputObject $procs
$failed = $false
for ($i = 0; $i -lt 2; $i++) {
    $codePath = Join-Path $logDir "embedding_shard${i}.exit_code.txt"
    if (-not (Test-Path $codePath)) {
        $failed = $true
        continue
    }
    $code = [int](Get-Content $codePath -Raw)
    if ($code -ne 0) {
        $failed = $true
    }
}
if ($failed) {
    "embedding shard failed" | Tee-Object -FilePath (Join-Path $logDir "fatal.log") -Append
    Set-Content -Path (Join-Path $logDir "formal_exit_code.txt") -Value 1
    exit 1
}

Run-Step "merge_embeddings" {
    & $python (Join-Path $codeDir "merge_scfoundation_multidomain_shards.py") `
        --root $root `
        --datasets "gse300551_iccite_plex_kinase_2025" "signal_seq_gse256403_hela_2024" "signal_seq_gse256404_pdo_caf_2024" `
        --shard-dirs "$shardBase\shard0_of_2" "$shardBase\shard1_of_2" `
        --output-dir $merged
}

Run-Step "assemble_model_input" {
    & $python (Join-Path $codeDir "build_scfoundation_multidomain_model_input.py") `
        --root $root `
        --template-input $template `
        --source-dirs $oldEmb $merged `
        --output-dir $inputOut
}

Run-Step "ridge_external_eval" {
    & $python (Join-Path $codeDir "run_scfoundation_multidomain_persite_ridge.py") `
        --input-dir (Join-Path $root $inputOut) `
        --output $result `
        --train-datasets "iccite_seq_tcell_2025,qurie_seq_bjab_2021" `
        --test-datasets "gse300551_iccite_plex_kinase_2025,signal_seq_gse256403_hela_2024,signal_seq_gse256404_pdo_caf_2024,phospho_seq_blair_2025_phospho_multi,vivo_seq_th17_2025" `
        --method-name "scfoundation_cap12000_persite_ridge_iccite_qurie"
}

Set-Content -Path (Join-Path $logDir "formal_exit_code.txt") -Value 0
"[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] ALL_DONE" | Tee-Object -FilePath (Join-Path $logDir "formal_stdout.log") -Append
