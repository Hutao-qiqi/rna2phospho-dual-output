$ErrorActionPreference = "Stop"

$items = @(
    @{
        Dataset = "phospho_seq_blair_2025"
        Root = "D:\lsy\01_data\single_cell\raw\phospho_seq_blair_2025"
        Base = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285561/suppl"
        Files = @(
            "GSE285561_BrainOrg_Features.csv.gz",
            "GSE285561_OrgRNA_Features.csv.gz",
            "GSE285561_PhosPilot_Features.csv.gz",
            "GSE285561_Phos_Bench_2_Features.csv.gz",
            "GSE285561_Phos_Bench_Features.csv.gz",
            "GSE285561_RetOrgMulti_Features.csv.gz",
            "GSE285561_RetOrg_Features.csv.gz",
            "filelist.txt",
            "GSE285561_RAW.tar"
        )
    },
    @{
        Dataset = "vivo_seq_th17_2025"
        Root = "D:\lsy\01_data\single_cell\raw\vivo_seq_th17_2025"
        Base = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE297nnn/GSE297075/suppl"
        Files = @(
            "GSE297075_Vivo-seq_processed_Scanpy.h5ad",
            "GSE297075_barcodes.tsv.gz",
            "GSE297075_feature_reference.csv.gz",
            "GSE297075_features.tsv.gz",
            "GSE297075_matrix.mtx.gz"
        )
    }
)

$logDir = "D:\lsy\02_results\single_cell\20260509_data_inventory\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

foreach ($group in $items) {
    New-Item -ItemType Directory -Force -Path $group.Root | Out-Null
    foreach ($file in $group.Files) {
        $url = "$($group.Base)/$file"
        $out = Join-Path $group.Root $file
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Add-Content -Path (Join-Path $logDir "download_core_geo.log") -Value "[$stamp] START $($group.Dataset) $file"
        $curlLog = Join-Path $logDir "$file.curl.log"
        $ErrorActionPreference = "Continue"
        & curl.exe --ssl-no-revoke -L --fail --retry 8 --retry-delay 10 --continue-at - --output $out $url *> $curlLog
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = "Stop"
        if ($exitCode -ne 0) {
            throw "curl failed for $file with exit code $exitCode; see $curlLog"
        }
        $size = if (Test-Path $out) { (Get-Item $out).Length } else { 0 }
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Add-Content -Path (Join-Path $logDir "download_core_geo.log") -Value "[$stamp] DONE $($group.Dataset) $file bytes=$size"
    }
}
