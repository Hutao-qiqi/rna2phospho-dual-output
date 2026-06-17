$ErrorActionPreference = "Continue"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}
$root = "D:\data\lsy\vm_lsy_parent\lsy"
$python = "D:\Tools\anaconda3\python.exe"
$script = Join-Path $root "03_code\single_cell\modeling\train_scp682_sc11_expanded_scnet_site_gnn_no_attention_ablation.py"
$out = Join-Path $root "02_results\single_cell\20260531_scp682_sc11_no_pathway_attention_ablation_v1"
$logDir = Join-Path $out "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$argsList = @(
  $script,
  "--root", $root,
  "--pathway-manifest", "02_results\single_cell\20260519_scp682_sc3_multidomain_features_v1\intermediate\pathway_gene_manifest.tsv",
  "--model-input-dir", "01_data\single_cell\intermediate\phospho_model_inputs\scfoundation_cap12000_gse300551_signal_seq_multidomain_v1",
  "--output-dir", "02_results\single_cell\20260531_scp682_sc11_no_pathway_attention_ablation_v1",
  "--train-datasets", "iccite_seq_tcell_2025,qurie_seq_bjab_2021",
  "--train-control-only-datasets", "__none__",
  "--holdout-datasets", "gse300551_iccite_plex_kinase_2025,phospho_seq_blair_2025_phospho_multi,vivo_seq_th17_2025,signal_seq_gse256403_hela_2024,signal_seq_gse256404_pdo_caf_2024",
  "--include-drug-delta-eval",
  "--target-ids", "include_in_loss",
  "--target-transform", "zscore",
  "--context-times", "6,180",
  "--epochs", "140",
  "--patience", "26",
  "--batch-size", "2048",
  "--batch-log-interval", "20",
  "--eval-batch-size", "4096",
  "--context-cells", "1024",
  "--max-eval-train-cells", "40000",
  "--hidden", "256",
  "--pathway-layers", "2",
  "--attention-heads", "4",
  "--dropout", "0.12",
  "--lr", "0.00018",
  "--weight-decay", "0.0001",
  "--recon-weight", "1.0",
  "--prior-weight", "0.03",
  "--prior-neighbors", "12",
  "--prior-temperature", "0.08",
  "--prior-min-similarity", "0.15",
  "--prior-steps-per-epoch", "12",
  "--prior-batch-size", "1024",
  "--prior-datasets", "iccite_seq_tcell_2025,qurie_seq_bjab_2021",
  "--scp68222-transfer-dir", "01_data\single_cell\intermediate\scp682_22_full_transfer_prior_v1",
  "--transfer-alpha", "0.55",
  "--transfer-attention-weight", "0.06",
  "--full-transfer-scale", "0.45",
  "--teacher-distill-weight", "0.08",
  "--site-graph-prior-root", "01_data\pathway_prior\intermediate",
  "--site-graph-weight", "0.03",
  "--site-graph-scale", "0.25",
  "--site-graph-candidate-limit", "96",
  "--site-graph-max-aux-nodes", "12000",
  "--warm-start-model", "02_results\single_cell\20260520_scp682_sc7_scfoundation_scp68222_transfer_v1\models\scp682_sc7_final.pt",
  "--huber-beta", "0.5",
  "--loss-type", "huber",
  "--warmup-epochs", "8",
  "--val-fraction", "0.12",
  "--val-cells", "30000",
  "--max-target-weight", "4.0",
  "--balance-dataset-size", "60000",
  "--grad-clip", "1.0",
  "--seed", "682522",
  "--device", "cuda:1",
  "--export-attention",
  "--attention-cells", "20000",
  "--disable-pathway-attention"
)
& $python @argsList 1> (Join-Path $logDir "formal_stdout.log") 2> (Join-Path $logDir "formal_stderr.log")
$LASTEXITCODE | Set-Content -Path (Join-Path $logDir "formal_exit_code.txt")
if ($LASTEXITCODE -eq 0) {
  Get-Date | Out-File -Encoding utf8 (Join-Path $out "done.txt")
}
