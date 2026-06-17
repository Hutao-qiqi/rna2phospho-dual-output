#!/usr/bin/env python3
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path("/data/lsy/Infinite_Stream")
RUN_DATE = "20260425"
RESULT_ROOT = PROJECT_ROOT / "02_results" / "shared" / f"{RUN_DATE}_data_acquisition"
MANIFEST = RESULT_ROOT / "tables" / "download_manifest.tsv"
PDC_MANIFEST = PROJECT_ROOT / "01_data" / "multi_omics" / "metadata" / "pdc_cptac_gbm_files_per_study.tsv"
PENDING = RESULT_ROOT / "tables" / "pending_or_manual_resources.tsv"


def ensure_dirs() -> None:
    for modality in [
        "multi_omics",
        "external_validation",
        "pathway_prior",
        "functional_genomics",
        "drug_response",
        "shared",
    ]:
        for leaf in ["raw", "intermediate", "plot_ready", "metadata"]:
            (PROJECT_ROOT / "01_data" / modality / leaf).mkdir(parents=True, exist_ok=True)
    for leaf in ["tables", "figures", "logs"]:
        (RESULT_ROOT / leaf).mkdir(parents=True, exist_ok=True)


def run(cmd, log_name=None, check=False):
    if log_name:
        log_path = RESULT_ROOT / "logs" / log_name
        with log_path.open("ab") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return proc


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def curl_download(url: str, out_path: Path, dataset: str, category: str, source: str, note: str = ""):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    status = "present"
    if not out_path.exists() or out_path.stat().st_size == 0:
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        cmd = [
            "curl",
            "-L",
            "--fail",
            "-C",
            "-",
            "--retry",
            "3",
            "--retry-delay",
            "5",
            "-A",
            "Mozilla/5.0",
            "-o",
            str(tmp),
            url,
        ]
        proc = run(cmd, log_name=f"{dataset}.download.log")
        if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(out_path)
            status = "downloaded"
        else:
            status = "failed"
            if tmp.exists() and tmp.stat().st_size == 0:
                tmp.unlink()
    size = out_path.stat().st_size if out_path.exists() else 0
    digest = sha256(out_path) if out_path.exists() and size > 0 else ""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "category": category,
        "dataset": dataset,
        "source": source,
        "url": url,
        "local_path": str(out_path.relative_to(PROJECT_ROOT)) if out_path.exists() else str(out_path),
        "bytes": size,
        "sha256": digest,
        "note": note,
    }


def figshare_files(article_id: str, out_json: Path):
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-L", "--fail", "-A", "Mozilla/5.0", "-o", str(out_json), url]
    run(cmd, log_name=f"figshare_{article_id}.api.log", check=True)
    with out_json.open() as fh:
        data = json.load(fh)
    return {f["name"]: f for f in data.get("files", [])}


def pdc_files_per_study(study_id: str, out_tsv: Path):
    query = (
        '{ filesPerStudy(pdc_study_id: "'
        + study_id
        + '") { pdc_study_id study_name file_id file_name file_type file_size '
        + "data_category file_format file_location md5sum } }"
    )
    payload = json.dumps({"query": query})
    raw = RESULT_ROOT / "logs" / f"pdc_{study_id}_files.json"
    cmd = [
        "curl",
        "-s",
        "https://pdc.cancer.gov/graphql",
        "-H",
        "Content-Type: application/json",
        "-d",
        payload,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    raw.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        return 0
    data = json.loads(proc.stdout)
    rows = data.get("data", {}).get("filesPerStudy", [])
    if not rows:
        return 0
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdc_study_id",
        "study_name",
        "file_id",
        "file_name",
        "file_type",
        "file_size",
        "data_category",
        "file_format",
        "file_location",
        "md5sum",
    ]
    write_header = not out_tsv.exists()
    with out_tsv.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)
    return len(rows)


def main():
    os.chdir(PROJECT_ROOT)
    ensure_dirs()
    rows = []

    geo_base = PROJECT_ROOT / "01_data" / "external_validation" / "raw" / "geo"
    geo_items = [
        ("GSE16011_series_matrix", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE16nnn/GSE16011/matrix/GSE16011_series_matrix.txt.gz", geo_base / "GSE16011" / "GSE16011_series_matrix.txt.gz"),
        ("GSE43378_series_matrix", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE43nnn/GSE43378/matrix/GSE43378_series_matrix.txt.gz", geo_base / "GSE43378" / "GSE43378_series_matrix.txt.gz"),
        ("GSE7696_series_matrix", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE7nnn/GSE7696/matrix/GSE7696_series_matrix.txt.gz", geo_base / "GSE7696" / "GSE7696_series_matrix.txt.gz"),
        ("GSE83300_series_matrix", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE83nnn/GSE83300/matrix/GSE83300_series_matrix.txt.gz", geo_base / "GSE83300" / "GSE83300_series_matrix.txt.gz"),
    ]
    for dataset, url, out in geo_items:
        rows.append(curl_download(url, out, dataset, "external_validation", "GEO"))

    cgga_base = PROJECT_ROOT / "01_data" / "external_validation" / "raw" / "cgga"
    cgga_items = [
        ("CGGA_mRNAseq_693_clinical", "https://www.cgga.org.cn/download?file=download/20200506/CGGA.mRNAseq_693_clinical.20200506.txt.zip&type=mRNAseq_693_clinical&time=20200506", cgga_base / "mRNAseq_693" / "CGGA.mRNAseq_693_clinical.20200506.txt.zip"),
        ("CGGA_mRNAseq_693_RSEM", "https://www.cgga.org.cn/download?file=download/20200506/CGGA.mRNAseq_693.RSEM-genes.20200506.txt.zip&type=mRNAseq_693&time=20200506", cgga_base / "mRNAseq_693" / "CGGA.mRNAseq_693.RSEM-genes.20200506.txt.zip"),
        ("CGGA_mRNAseq_693_counts", "https://www.cgga.org.cn/download?file=download/20220620/CGGA.mRNAseq_693.Read_Counts-genes.20220620.txt.zip&type=mRNAseq_693_counts&time=20220620", cgga_base / "mRNAseq_693" / "CGGA.mRNAseq_693.Read_Counts-genes.20220620.txt.zip"),
        ("CGGA_mRNAseq_325_clinical", "https://www.cgga.org.cn/download?file=download/20200506/CGGA.mRNAseq_325_clinical.20200506.txt.zip&type=mRNAseq_325_clinical&time=20200506", cgga_base / "mRNAseq_325" / "CGGA.mRNAseq_325_clinical.20200506.txt.zip"),
        ("CGGA_mRNAseq_325_RSEM", "https://www.cgga.org.cn/download?file=download/20200506/CGGA.mRNAseq_325.RSEM-genes.20200506.txt.zip&type=mRNAseq_325&time=20200506", cgga_base / "mRNAseq_325" / "CGGA.mRNAseq_325.RSEM-genes.20200506.txt.zip"),
        ("CGGA_mRNAseq_325_counts", "https://www.cgga.org.cn/download?file=download/20220620/CGGA.mRNAseq_325.Read_Counts-genes.20220620.txt.zip&type=mRNAseq_325_counts&time=20220620", cgga_base / "mRNAseq_325" / "CGGA.mRNAseq_325.Read_Counts-genes.20220620.txt.zip"),
        ("CGGA_mRNA_array_301_clinical", "https://www.cgga.org.cn/download?file=download/20200506/CGGA.mRNA_array_301_clinical.20200506.txt.zip&type=mRNA_array_301_clinical&time=20200506", cgga_base / "mRNA_array_301" / "CGGA.mRNA_array_301_clinical.20200506.txt.zip"),
        ("CGGA_mRNA_array_301_gene_level", "https://www.cgga.org.cn/download?file=download/20200506/CGGA.mRNA_array_301_gene_level.20200506.txt.zip&type=mRNA_array_301_gene_level&time=20200506", cgga_base / "mRNA_array_301" / "CGGA.mRNA_array_301_gene_level.20200506.txt.zip"),
        ("CGGA_mRNA_array_301_probe_level", "https://www.cgga.org.cn/download?file=download/20200506/CGGA.mRNA_array_301_probe_level.20200506.txt.zip&type=mRNA_array_301_probe_level&time=20200506", cgga_base / "mRNA_array_301" / "CGGA.mRNA_array_301_probe_level.20200506.txt.zip"),
    ]
    for dataset, url, out in cgga_items:
        rows.append(curl_download(url, out, dataset, "external_validation", "CGGA"))

    prior_base = PROJECT_ROOT / "01_data" / "pathway_prior" / "raw"
    prior_items = [
        ("KEGG_hsa_pathway_list", "https://rest.kegg.jp/list/pathway/hsa", prior_base / "kegg" / "hsa_pathway_list.txt"),
        ("KEGG_hsa_gene_to_pathway", "https://rest.kegg.jp/link/pathway/hsa", prior_base / "kegg" / "hsa_gene_to_pathway.txt"),
        ("Reactome_pathways", "https://reactome.org/download/current/ReactomePathways.txt", prior_base / "reactome" / "ReactomePathways.txt"),
        ("Reactome_uniprot_mapping", "https://reactome.org/download/current/UniProt2Reactome_All_Levels.txt", prior_base / "reactome" / "UniProt2Reactome_All_Levels.txt"),
        ("Reactome_gmt", "https://reactome.org/download/current/ReactomePathways.gmt.zip", prior_base / "reactome" / "ReactomePathways.gmt.zip"),
        ("Reactome_biopax", "https://reactome.org/download/current/biopax.zip", prior_base / "reactome" / "biopax.zip"),
        ("STRING_human_links_v12", "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz", prior_base / "string" / "9606.protein.links.v12.0.txt.gz"),
    ]
    for dataset, url, out in prior_items:
        rows.append(curl_download(url, out, dataset, "pathway_prior", dataset.split("_")[0]))

    depmap_meta = PROJECT_ROOT / "01_data" / "functional_genomics" / "metadata" / "depmap_24q4_figshare_files.json"
    depmap_files = figshare_files("27993248", depmap_meta)
    depmap_base = PROJECT_ROOT / "01_data" / "functional_genomics" / "raw" / "depmap_24q4"
    for name in ["CRISPRGeneEffect.csv", "Model.csv", "README.txt"]:
        f = depmap_files.get(name)
        if f:
            rows.append(curl_download(f["download_url"], depmap_base / name, f"DepMap_24Q4_{name}", "functional_genomics", "DepMap Figshare", "Figshare article 27993248"))

    prism_meta = PROJECT_ROOT / "01_data" / "drug_response" / "metadata" / "prism_figshare_9393293_files.json"
    prism_files = figshare_files("9393293", prism_meta)
    prism_base = PROJECT_ROOT / "01_data" / "drug_response" / "raw" / "prism"
    for name in [
        "secondary-screen-dose-response-curve-parameters.csv",
        "secondary-screen-cell-line-info.csv",
        "secondary-screen-readme.txt",
    ]:
        f = prism_files.get(name)
        if f:
            rows.append(curl_download(f["download_url"], prism_base / name, f"PRISM_{name}", "drug_response", "PRISM Figshare", "Figshare article 9393293"))

    gdsc_base = PROJECT_ROOT / "01_data" / "drug_response" / "raw" / "gdsc"
    gdsc_items = [
        ("GDSC2_fitted_dose_response", "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/GDSC2_fitted_dose_response_24Jul22.csv", gdsc_base / "GDSC2_fitted_dose_response_24Jul22.csv"),
        ("GDSC1_fitted_dose_response", "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/GDSC1_fitted_dose_response_24Jul22.csv", gdsc_base / "GDSC1_fitted_dose_response_24Jul22.csv"),
        ("GDSC_cell_line_details", "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/Cell_Lines_Details.xlsx", gdsc_base / "Cell_Lines_Details.xlsx"),
        ("GDSC_screened_compounds", "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/screened_compounds_rel_8.4.csv", gdsc_base / "screened_compounds_rel_8.4.csv"),
    ]
    for dataset, url, out in gdsc_items:
        rows.append(curl_download(url, out, dataset, "drug_response", "GDSC"))

    existing_gdc = PROJECT_ROOT / "GDCdata" / "TCGA-GBM"
    rows.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": "existing",
        "category": "multi_omics",
        "dataset": "TCGA_GBM_STAR_counts_existing",
        "source": "GDC",
        "url": "already present before this run",
        "local_path": str(existing_gdc.relative_to(PROJECT_ROOT)) if existing_gdc.exists() else str(existing_gdc),
        "bytes": sum(p.stat().st_size for p in existing_gdc.rglob("*") if p.is_file()) if existing_gdc.exists() else 0,
        "sha256": "",
        "note": "Existing project copy kept in place; not duplicated into 01_data.",
    })

    if PDC_MANIFEST.exists():
        PDC_MANIFEST.unlink()
    pdc_count = 0
    for study_id in ["PDC000204", "PDC000205", "PDC000206"]:
        try:
            pdc_count += pdc_files_per_study(study_id, PDC_MANIFEST)
        except Exception as exc:
            (RESULT_ROOT / "logs" / f"pdc_{study_id}_error.txt").write_text(str(exc))
    rows.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": "metadata_saved" if pdc_count else "failed",
        "category": "multi_omics",
        "dataset": "PDC_CPTAC_GBM_PDC000204_to_PDC000206_files",
        "source": "PDC GraphQL",
        "url": "https://pdc.cancer.gov/graphql",
        "local_path": str(PDC_MANIFEST.relative_to(PROJECT_ROOT)) if PDC_MANIFEST.exists() else str(PDC_MANIFEST),
        "bytes": PDC_MANIFEST.stat().st_size if PDC_MANIFEST.exists() else 0,
        "sha256": sha256(PDC_MANIFEST) if PDC_MANIFEST.exists() else "",
        "note": f"{pdc_count} file metadata rows saved. Bulk raw downloads are not started automatically.",
    })

    pending_rows = [
        ["PhosphoSitePlus", "pathway_prior", "kinase-substrate table", "Manual academic registration required; place downloaded files under 01_data/pathway_prior/raw/phosphositeplus/."],
        ["GBM-CoDE", "functional_genomics", "60 IDH-wildtype GBM CRISPR screen supplementary data", "Needs final supplementary URL extraction from the preprint or journal record; place under 01_data/functional_genomics/raw/gbm_code/."],
        ["GLASS", "external_validation", "paired primary-recurrent RNA-seq", "Download endpoint requires source-specific confirmation; place under 01_data/external_validation/raw/glass/."],
        ["PDC processed CPTAC GBM", "multi_omics", "selected processed proteome and phosphoproteome matrices", "PDC file metadata saved; download only selected processed matrices after choosing file names to avoid raw mzML/PSM bulk pull."],
    ]
    with PENDING.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["resource", "category", "target_content", "reason_or_next_action"])
        w.writerows(pending_rows)

    with MANIFEST.open("w", newline="") as fh:
        fieldnames = ["timestamp", "status", "category", "dataset", "source", "url", "local_path", "bytes", "sha256", "note"]
        w = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"manifest={MANIFEST}")
    print(f"rows={len(rows)}")
    print(f"pending={PENDING}")


if __name__ == "__main__":
    main()
