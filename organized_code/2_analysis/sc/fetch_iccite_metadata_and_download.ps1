$ErrorActionPreference = "Stop"

$root = "D:\lsy"
$raw = Join-Path $root "01_data\single_cell\raw\iccite_seq_tcell_2025"
$meta = Join-Path $root "01_data\shared\metadata"
$refs = Join-Path $root "refs"
$logDir = Join-Path $root "02_results\single_cell\20260510_iccite_download\logs"

New-Item -ItemType Directory -Force -Path $raw, $meta, $refs, $logDir | Out-Null

$enaUrl = "https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&query=study_accession%3D%22PRJDB16517%22&fields=study_accession,sample_accession,experiment_accession,run_accession,tax_id,scientific_name,fastq_ftp,fastq_bytes,fastq_md5,submitted_ftp,submitted_bytes,submitted_md5,library_strategy,library_source,library_selection,library_layout,instrument_platform,instrument_model&format=tsv&limit=0"
$runTable = Join-Path $meta "PRJDB16517_ena_read_run.tsv"
curl.exe -L --ssl-no-revoke --fail --retry 10 --retry-delay 10 -o $runTable $enaUrl

$xmlUrl = "https://www.ebi.ac.uk/ena/browser/api/xml/PRJDB16517"
curl.exe -L --ssl-no-revoke --fail --retry 10 --retry-delay 10 -o (Join-Path $meta "PRJDB16517_project.xml") $xmlUrl

$repo = Join-Path $refs "Perturb-icCITEseq"
if (!(Test-Path $repo)) {
    git clone https://github.com/agiguelay/Perturb-icCITEseq.git $repo
} else {
    git -C $repo pull --ff-only
}

$zenodo = "https://zenodo.org/records/16020737/files/Perturb_icCITE_seq_FOXP3_regulators.rds?download=1"
$out = Join-Path $raw "Perturb_icCITE_seq_FOXP3_regulators.rds"
$log = Join-Path $logDir "zenodo_download.log"

curl.exe -L --ssl-no-revoke --fail --retry 100 --retry-delay 30 --continue-at - -o $out $zenodo *> $log
