$ErrorActionPreference = "Stop"

$jobs = @(
    @{
        Name = "vivo_seq_th17_2025"
        Root = "D:\lsy\01_data\single_cell\raw\vivo_seq_th17_2025"
        Base = "http://ftp.ncbi.nlm.nih.gov/geo/series/GSE297nnn/GSE297075/suppl"
        Files = @(
            "GSE297075_Vivo-seq_processed_Scanpy.h5ad",
            "GSE297075_barcodes.tsv.gz",
            "GSE297075_feature_reference.csv.gz",
            "GSE297075_features.tsv.gz",
            "GSE297075_matrix.mtx.gz"
        )
    },
    @{
        Name = "phospho_seq_blair_2025_raw_tar"
        Root = "D:\lsy\01_data\single_cell\raw\phospho_seq_blair_2025"
        Base = "http://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285561/suppl"
        Files = @("GSE285561_RAW.tar")
    }
)

$logDir = "D:\lsy\02_results\single_cell\20260509_data_inventory\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Import-Module BitsTransfer

foreach ($group in $jobs) {
    New-Item -ItemType Directory -Force -Path $group.Root | Out-Null
    $displayName = "lsy_$($group.Name)"
    Get-BitsTransfer -AllUsers | Where-Object DisplayName -eq $displayName | Remove-BitsTransfer -Confirm:$false -ErrorAction SilentlyContinue
    $sources = @()
    $destinations = @()
    foreach ($file in $group.Files) {
        $sources += "$($group.Base)/$file"
        $destinations += (Join-Path $group.Root $file)
    }
    Start-BitsTransfer -Source $sources -Destination $destinations -DisplayName $displayName -Asynchronous -Priority Foreground
    Add-Content -Path (Join-Path $logDir "bits_downloads.log") -Value "START $displayName $(Get-Date -Format s)"
}

Get-BitsTransfer -AllUsers | Where-Object DisplayName -like "lsy_*" |
    Select-Object DisplayName,JobState,BytesTransferred,BytesTotal,FilesTransferred,FilesTotal |
    Format-Table -AutoSize

