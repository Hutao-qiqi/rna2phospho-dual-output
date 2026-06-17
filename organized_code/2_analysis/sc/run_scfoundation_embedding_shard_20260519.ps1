param(
    [Parameter(Mandatory=$true)][int]$ShardIndex,
    [Parameter(Mandatory=$true)][string]$Device
)

$ErrorActionPreference = "Continue"

$root = "D:\data\lsy\vm_lsy_parent\lsy"
$python = "D:\data\lsy\envs\scgpt\python.exe"
$codeDir = Join-Path $root "03_code\single_cell\modeling"
$logDir = Join-Path $root "02_results\single_cell\20260519_scfoundation_ridge_gse300551_signal_v2\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:CUDA_VISIBLE_DEVICES = $Device
$env:PYTORCH_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONWARNINGS = "ignore"

$shardBase = "01_data\single_cell\intermediate\scfoundation_embeddings\cap12000_gse300551_signal_seq_v1_shards"
$outDir = "$shardBase\shard${ShardIndex}_of_2"
$log = Join-Path $logDir "embedding_shard${ShardIndex}.log"
$err = Join-Path $logDir "embedding_shard${ShardIndex}.err"

$script = Join-Path $codeDir "precompute_scfoundation_embeddings_multidomain.py"
$cmd = "`"$python`" `"$script`" --root `"$root`" --datasets gse300551_iccite_plex_kinase_2025 signal_seq_gse256403_hela_2024 signal_seq_gse256404_pdo_caf_2024 --batch-size 1 --max-gene-tokens 12000 --num-shards 2 --shard-index $ShardIndex --skip-existing --output-dir `"$outDir`" 1> `"$log`" 2> `"$err`""
& cmd.exe /c $cmd

$code = $LASTEXITCODE
Set-Content -Path (Join-Path $logDir "embedding_shard${ShardIndex}.exit_code.txt") -Value $code
exit $code
