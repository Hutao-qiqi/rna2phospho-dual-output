$ErrorActionPreference = "Stop"

$root = "D:\data\lsy\vm_lsy_parent\lsy"
$outRel = "02_results\single_cell\20260531_scp682_sc_kirc_rps6_validation_v1"
$out = Join-Path $root $outRel
$logDir = Join-Path $out "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$python = "D:\data\lsy\envs\scgpt\python.exe"
$scriptPath = Join-Path $root "03_code\single_cell\scp682_sc_kirc_rps6_validation\run_kirc_rps6_validation.py"
$stdout = Join-Path $logDir "run_kirc_rps6_validation.stdout.log"
$stderr = Join-Path $logDir "run_kirc_rps6_validation.stderr.log"
$exitFile = Join-Path $logDir "run_kirc_rps6_validation.exit_code.txt"

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONWARNINGS = "ignore"

$argList = @(
    $scriptPath,
    "--root", $root,
    "--h5ad-gz", "D:\data\lsy\GSE242299_all_cells_50236_33538.h5ad.gz",
    "--h5ad", (Join-Path $root "01_data\single_cell\intermediate\ccrcc_gse242299\GSE242299_all_cells_50236_33538.h5ad"),
    "--dataset-id", "GSE242299_ccRCC",
    "--output-dir", $outRel,
    "--model-path", (Join-Path $root "02_results\single_cell\20260522_scp682_sc11_expanded_scnet_site_gnn_v1\models\scp682_sc11_final.pt"),
    "--formal-result-dir", (Join-Path $root "02_results\single_cell\20260522_scp682_sc11_expanded_scnet_site_gnn_v1"),
    "--modeling-code-dir", (Join-Path $root "03_code\single_cell\modeling"),
    "--scfoundation-code-dir", "D:\data\lsy\repos\scFoundation_model",
    "--scfoundation-weight-path", "D:\data\lsy\models\scfoundation\models.ckpt",
    "--device", "cuda:0",
    "--embedding-batch-size", "1",
    "--prediction-batch-size", "2048",
    "--max-gene-tokens", "12000",
    "--highres", "4.0",
    "--skip-existing"
)

"started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Set-Content -Path (Join-Path $logDir "run_kirc_rps6_validation.wrapper.log")
& $python @argList 1> $stdout 2> $stderr
$code = $LASTEXITCODE
Set-Content -Path $exitFile -Value $code
"finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') exit=$code" | Add-Content -Path (Join-Path $logDir "run_kirc_rps6_validation.wrapper.log")
exit $code
