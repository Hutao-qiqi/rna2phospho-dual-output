$ErrorActionPreference = "Stop"

$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:CUDA_VISIBLE_DEVICES = "0"

$Root = "D:\data\lsy\vm_lsy_parent\lsy"
$Conda = "D:\Tools\anaconda3\Scripts\conda.exe"
$Script = Join-Path $Root "remote_scripts\train_scp682_missing_ablation_degree_rewire.py"
$Summary = Join-Path $Root "remote_scripts\summarize_scp682_missing_ablation_grid.py"
$OutRoot = Join-Path $Root "02_results\model_validation\20260602_m4_knowledge_graph_controls_e160_windows"
$PackageDir = Join-Path $Root "SCP682_MAIN"
$PriorRoot = Join-Path $Root "01_data\pathway_prior"
$Baseline = Join-Path $Root "SCP682_MAIN\inputs\general_baseline_predictions\general_baseline_internal_cptac_pdc_phosphosite.parquet"
$Rna = "D:\data\lsy\01_data\multi_omics\processed\pancancer_multi_task_locked_v2\rna_log2_tpm_paired.parquet"
$SampleManifest = "D:\data\lsy\01_data\multi_omics\processed\pancancer_multi_task_locked_v2\sample_manifest.tsv"

New-Item -ItemType Directory -Force -Path (Join-Path $OutRoot "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutRoot "tables") | Out-Null
"start $(Get-Date -Format s)" | Set-Content -Path (Join-Path $OutRoot "run_status.txt")

$Tasks = @(
    @("axis_dual_edge_rewired_all", "dual", "rewired_all"),
    @("axis_dual_edge_no_copheemap", "dual", "no_copheemap"),
    @("axis_dual_edge_no_copheeksa", "dual", "no_copheeksa"),
    @("axis_dual_edge_no_kstar", "dual", "no_kstar")
)

foreach ($Task in $Tasks) {
    $Name = $Task[0]
    $Axis = $Task[1]
    $Edge = $Task[2]
    $Out = Join-Path $OutRoot $Name
    New-Item -ItemType Directory -Force -Path $Out | Out-Null
    "[$(Get-Date -Format s)] start $Name axis=$Axis edge=$Edge" | Tee-Object -FilePath (Join-Path $OutRoot "run_status.txt") -Append

    $Stdout = Join-Path $OutRoot "logs\$Name.stdout.log"
    $Stderr = Join-Path $OutRoot "logs\$Name.stderr.log"
    & $Conda run -n scvi-env python $Script `
        --package-dir $PackageDir `
        --prior-root $PriorRoot `
        --output-dir $Out `
        --general-baseline-path $Baseline `
        --rna-path $Rna `
        --sample-manifest-path $SampleManifest `
        --group-column cancer_label `
        --device cuda:0 `
        --epochs 160 `
        --batch-size 8 `
        --hidden 96 `
        --latent 32 `
        --inter-dim 96 `
        --embd-dim 32 `
        --num-layers 1 `
        --axis-mode $Axis `
        --edge-mode $Edge `
        1> $Stdout 2> $Stderr

    if ($LASTEXITCODE -ne 0) {
        "failed $Name exit_code=$LASTEXITCODE $(Get-Date -Format s)" | Set-Content -Path (Join-Path $OutRoot "fatal.log")
        exit $LASTEXITCODE
    }
    "[$(Get-Date -Format s)] done $Name" | Tee-Object -FilePath (Join-Path $OutRoot "run_status.txt") -Append
}

& $Conda run -n scvi-env python $Summary `
    --grid-dir $OutRoot `
    --output (Join-Path $OutRoot "tables\m4_knowledge_graph_controls_summary.tsv")

if ($LASTEXITCODE -ne 0) {
    "failed summary exit_code=$LASTEXITCODE $(Get-Date -Format s)" | Set-Content -Path (Join-Path $OutRoot "fatal.log")
    exit $LASTEXITCODE
}

"done $(Get-Date -Format s)" | Set-Content -Path (Join-Path $OutRoot "done.txt")
