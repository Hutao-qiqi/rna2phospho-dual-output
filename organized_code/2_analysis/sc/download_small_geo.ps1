$ErrorActionPreference = "Stop"

$files = @(
    @("phospho_seq_blair_2025", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285561/suppl", "D:\lsy\01_data\single_cell\raw\phospho_seq_blair_2025", "GSE285561_BrainOrg_Features.csv.gz"),
    @("phospho_seq_blair_2025", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285561/suppl", "D:\lsy\01_data\single_cell\raw\phospho_seq_blair_2025", "GSE285561_OrgRNA_Features.csv.gz"),
    @("phospho_seq_blair_2025", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285561/suppl", "D:\lsy\01_data\single_cell\raw\phospho_seq_blair_2025", "GSE285561_PhosPilot_Features.csv.gz"),
    @("phospho_seq_blair_2025", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285561/suppl", "D:\lsy\01_data\single_cell\raw\phospho_seq_blair_2025", "GSE285561_Phos_Bench_2_Features.csv.gz"),
    @("phospho_seq_blair_2025", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285561/suppl", "D:\lsy\01_data\single_cell\raw\phospho_seq_blair_2025", "GSE285561_Phos_Bench_Features.csv.gz"),
    @("phospho_seq_blair_2025", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285561/suppl", "D:\lsy\01_data\single_cell\raw\phospho_seq_blair_2025", "GSE285561_RetOrgMulti_Features.csv.gz"),
    @("phospho_seq_blair_2025", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285561/suppl", "D:\lsy\01_data\single_cell\raw\phospho_seq_blair_2025", "GSE285561_RetOrg_Features.csv.gz"),
    @("phospho_seq_blair_2025", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285561/suppl", "D:\lsy\01_data\single_cell\raw\phospho_seq_blair_2025", "filelist.txt"),
    @("vivo_seq_th17_2025", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE297nnn/GSE297075/suppl", "D:\lsy\01_data\single_cell\raw\vivo_seq_th17_2025", "GSE297075_Vivo-seq_processed_Scanpy.h5ad"),
    @("vivo_seq_th17_2025", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE297nnn/GSE297075/suppl", "D:\lsy\01_data\single_cell\raw\vivo_seq_th17_2025", "GSE297075_barcodes.tsv.gz"),
    @("vivo_seq_th17_2025", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE297nnn/GSE297075/suppl", "D:\lsy\01_data\single_cell\raw\vivo_seq_th17_2025", "GSE297075_feature_reference.csv.gz"),
    @("vivo_seq_th17_2025", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE297nnn/GSE297075/suppl", "D:\lsy\01_data\single_cell\raw\vivo_seq_th17_2025", "GSE297075_features.tsv.gz"),
    @("vivo_seq_th17_2025", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE297nnn/GSE297075/suppl", "D:\lsy\01_data\single_cell\raw\vivo_seq_th17_2025", "GSE297075_matrix.mtx.gz")
)

$logDir = "D:\lsy\02_results\single_cell\20260509_data_inventory\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$mainLog = Join-Path $logDir "download_small_geo.log"

foreach ($row in $files) {
    $dataset, $base, $root, $file = $row
    New-Item -ItemType Directory -Force -Path $root | Out-Null
    $url = "$base/$file"
    $out = Join-Path $root $file
    Add-Content $mainLog "START $dataset $file $(Get-Date -Format s)"
    & curl.exe --ssl-no-revoke -L --fail --retry 8 --retry-delay 10 --continue-at - --output $out $url
    if ($LASTEXITCODE -ne 0) { throw "curl failed: $file" }
    Add-Content $mainLog "DONE $dataset $file bytes=$((Get-Item $out).Length) $(Get-Date -Format s)"
}

