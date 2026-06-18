from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


APP_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("RNA2PHOSPHO_ROOT", "/data/lsy/Infinite_Stream"))
PYTHON = os.environ.get("RNA2PHOSPHO_PYTHON", "/home/USER/.local/share/mamba/envs/omicverse/bin/python")
ATLAS = ROOT / "02_results/public_bulk_phosphoproteome_atlas/20260430_fixed_v2_bulk_atlas_v1"
TCPA_ALL_TCGA = ROOT / "02_results/model_prediction/20260428_tcpa_rppa_film_vae_z_direct_residual_all_tcga_predictions_v1"
MODEL_CONTRACT = ROOT / "SCP682/SCP682_model_contract.json"
CURRENT_MODEL = ROOT / "SCP682_CURRENT.json"
FINAL_RELEASE_DIR = ROOT / "SCP682_PORTABLE"
FINAL_PACKAGE = FINAL_RELEASE_DIR.with_suffix(".tar")
FINAL_PACKAGE_SHA256 = FINAL_PACKAGE.with_suffix(".tar.sha256")
PREDICT_SCRIPT = APP_DIR / "predict_scp682.py"
JOBS_DIR = ROOT / "02_results/public_bulk_phosphoproteome_atlas/web_user_jobs"
MODEL_FULL_NAME = "Sample-centered Cross-platform Proteome and Phosphoproteome Predictor from Bulk RNA"

DOWNLOADS = {
    "tcga_cptac_total": {
        "cohort": "GDC TCGA supported",
        "layer": "CPTAC/PDC total protein",
        "targets": 11311,
        "samples": 5935,
        "path": ATLAS / "predictions/tcga_supported_predicted_cptac_total_proteome_fixed_v2_full5fold.parquet",
    },
    "tcga_cptac_phosphosite": {
        "cohort": "GDC TCGA supported",
        "layer": "CPTAC/PDC phosphosite",
        "targets": 18901,
        "samples": 5935,
        "path": ATLAS / "predictions/tcga_supported_predicted_cptac_phosphosite_fixed_v2_full5fold.parquet",
    },
    "tcga_tcpa_total": {
        "cohort": "GDC TCGA supported",
        "layer": "TCPA total antibody",
        "targets": 374,
        "samples": 5935,
        "path": ATLAS / "predictions/tcga_supported_predicted_tcpa_total_rppa.parquet",
    },
    "tcga_tcpa_phospho": {
        "cohort": "GDC TCGA supported",
        "layer": "TCPA phospho antibody",
        "targets": 73,
        "samples": 5935,
        "path": ATLAS / "predictions/tcga_supported_predicted_tcpa_phospho_rppa.parquet",
    },
    "cbio_cptac_total": {
        "cohort": "cBioPortal supported",
        "layer": "CPTAC/PDC total protein",
        "targets": 11311,
        "samples": 811,
        "path": ATLAS / "predictions/cbioportal_supported_predicted_cptac_total_proteome_fixed_v2_full5fold.parquet",
    },
    "cbio_cptac_phosphosite": {
        "cohort": "cBioPortal supported",
        "layer": "CPTAC/PDC phosphosite",
        "targets": 18901,
        "samples": 811,
        "path": ATLAS / "predictions/cbioportal_supported_predicted_cptac_phosphosite_fixed_v2_full5fold.parquet",
    },
    "cbio_tcpa_total": {
        "cohort": "cBioPortal supported",
        "layer": "TCPA total antibody",
        "targets": 374,
        "samples": 811,
        "path": ATLAS / "predictions/cbioportal_supported_predicted_tcpa_total_rppa.parquet",
    },
    "cbio_tcpa_phospho": {
        "cohort": "cBioPortal supported",
        "layer": "TCPA phospho antibody",
        "targets": 73,
        "samples": 811,
        "path": ATLAS / "predictions/cbioportal_supported_predicted_tcpa_phospho_rppa.parquet",
    },
}

CONTEXTS = [
    {"label": "BRCA_TCGA", "study": "PDC000174", "tcpa": "TCGA-BRCA"},
    {"label": "BRCA_PROSPECTIVE", "study": "PDC000121", "tcpa": "TCGA-BRCA"},
    {"label": "COAD_PROSPECTIVE", "study": "PDC000117", "tcpa": "TCGA-COAD"},
    {"label": "GBM_DISCOVERY", "study": "PDC000205", "tcpa": "TCGA-GBM"},
    {"label": "GBM_CONFIRMATORY", "study": "PDC000448", "tcpa": "TCGA-GBM"},
    {"label": "HNSCC", "study": "PDC000222", "tcpa": "TCGA-HNSC"},
    {"label": "CCRCC", "study": "PDC000128", "tcpa": "TCGA-KIRC"},
    {"label": "NON_CCRCC", "study": "PDC000465", "tcpa": "TCGA-KIRP"},
    {"label": "LUAD", "study": "PDC000149", "tcpa": "TCGA-LUAD"},
    {"label": "LUAD_CONFIRM", "study": "PDC000490", "tcpa": "TCGA-LUAD"},
    {"label": "LSCC", "study": "PDC000232", "tcpa": "TCGA-LUSC"},
    {"label": "OV_TCGA", "study": "PDC000115", "tcpa": "TCGA-OV"},
    {"label": "OV_PROSPECTIVE", "study": "PDC000119", "tcpa": "TCGA-OV"},
    {"label": "PDA", "study": "PDC000271", "tcpa": "TCGA-PAAD"},
    {"label": "STAD", "study": "PDC000615", "tcpa": "TCGA-STAD"},
    {"label": "UCEC", "study": "PDC000126", "tcpa": "TCGA-UCEC"},
    {"label": "UCEC_CONFIRM", "study": "PDC000441", "tcpa": "TCGA-UCEC"},
]

TCPA_PROJECTS = [
    "TCGA-ACC",
    "TCGA-BLCA",
    "TCGA-BRCA",
    "TCGA-CESC",
    "TCGA-CHOL",
    "TCGA-COAD",
    "TCGA-DLBC",
    "TCGA-ESCA",
    "TCGA-GBM",
    "TCGA-HNSC",
    "TCGA-KICH",
    "TCGA-KIRC",
    "TCGA-KIRP",
    "TCGA-LGG",
    "TCGA-LIHC",
    "TCGA-LUAD",
    "TCGA-LUSC",
    "TCGA-MESO",
    "TCGA-OV",
    "TCGA-PAAD",
    "TCGA-PCPG",
    "TCGA-PRAD",
    "TCGA-READ",
    "TCGA-SARC",
    "TCGA-SKCM",
    "TCGA-STAD",
    "TCGA-TGCT",
    "TCGA-THCA",
    "TCGA-THYM",
    "TCGA-UCEC",
    "TCGA-UCS",
    "TCGA-UVM",
]

app = FastAPI(title="SCP682")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


def job_path(job_id: str) -> Path:
    return JOBS_DIR / job_id


def read_job(job_id: str) -> dict:
    path = job_path(job_id) / "job.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="job not found")
    return json.loads(path.read_text(encoding="utf-8"))


def write_job(job_id: str, payload: dict) -> None:
    path = job_path(job_id)
    path.mkdir(parents=True, exist_ok=True)
    (path / "job.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_prediction(job_id: str, command: list[str]) -> None:
    payload = read_job(job_id)
    payload.update({"status": "running", "started_at": time.time(), "command": command})
    write_job(job_id, payload)
    log_path = job_path(job_id) / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    payload = read_job(job_id)
    payload["returncode"] = proc.returncode
    payload["finished_at"] = time.time()
    payload["log"] = str(log_path)
    summary_path = job_path(job_id) / "outputs/prediction_summary.json"
    if proc.returncode == 0 and summary_path.exists():
        payload["status"] = "done"
        payload["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        payload["status"] = "failed"
    write_job(job_id, payload)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (APP_DIR / "templates/index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {
        "model": "SCP682",
        "model_full_name": MODEL_FULL_NAME,
        "current_model_pointer": str(CURRENT_MODEL),
        "current_model": json.loads(CURRENT_MODEL.read_text(encoding="utf-8")) if CURRENT_MODEL.exists() else None,
        "final_release_dir": str(FINAL_RELEASE_DIR),
        "final_package": str(FINAL_PACKAGE),
        "final_package_exists": FINAL_PACKAGE.exists(),
        "upload_prediction_status": "current_main_predictor_available",
        "atlas_dir": str(ATLAS),
        "model_contract": str(MODEL_CONTRACT),
        "predict_script": str(PREDICT_SCRIPT),
        "download_files_present": sum(1 for x in DOWNLOADS.values() if x["path"].exists()),
        "download_files_total": len(DOWNLOADS),
    }


@app.get("/api/model-contract")
def model_contract() -> dict:
    if not MODEL_CONTRACT.exists():
        raise HTTPException(status_code=404, detail="model contract missing")
    return json.loads(MODEL_CONTRACT.read_text(encoding="utf-8"))


@app.get("/api/current-model")
def current_model() -> dict:
    if not CURRENT_MODEL.exists():
        raise HTTPException(status_code=404, detail="current model pointer missing")
    return json.loads(CURRENT_MODEL.read_text(encoding="utf-8"))


@app.get("/api/final-package")
def final_package():
    if not FINAL_PACKAGE.exists():
        raise HTTPException(status_code=404, detail="final package missing")
    return FileResponse(FINAL_PACKAGE, filename=FINAL_PACKAGE.name)


@app.get("/api/final-package-sha256")
def final_package_sha256():
    if not FINAL_PACKAGE_SHA256.exists():
        raise HTTPException(status_code=404, detail="final package sha256 missing")
    return FileResponse(FINAL_PACKAGE_SHA256, filename=FINAL_PACKAGE_SHA256.name, media_type="text/plain")


@app.get("/api/options")
def options() -> dict:
    return {"contexts": CONTEXTS, "tcpa_projects": TCPA_PROJECTS}


@app.get("/api/downloads")
def downloads() -> dict:
    rows = []
    for key, item in DOWNLOADS.items():
        p = item["path"]
        rows.append({
            "id": key,
            "cohort": item["cohort"],
            "layer": item["layer"],
            "targets": item["targets"],
            "samples": item["samples"],
            "exists": p.exists(),
            "bytes": p.stat().st_size if p.exists() else 0,
            "url": f"/api/downloads/{key}",
        })
    return {"items": rows}


@app.get("/api/downloads/{file_id}")
def download_file(file_id: str):
    item = DOWNLOADS.get(file_id)
    if not item:
        raise HTTPException(status_code=404, detail="unknown file id")
    path = item["path"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="file missing")
    return FileResponse(path, filename=path.name)


@app.post("/api/predict")
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    cptac_cancer_label: str = Form(...),
    cptac_study_id: str = Form(...),
    tcpa_project: str = Form(...),
    transform: str = Form("none"),
) -> dict:
    allowed = {".tsv", ".txt", ".csv", ".parquet"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="only tsv, txt, csv and parquet are supported")
    job_id = uuid.uuid4().hex[:16]
    base = job_path(job_id)
    upload_dir = base / "upload"
    out_dir = base / "outputs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / f"bulk_rna{suffix}"
    with input_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    payload = {
        "job_id": job_id,
        "status": "queued",
        "created_at": time.time(),
        "filename": file.filename,
        "parameters": {
            "cptac_cancer_label": cptac_cancer_label,
            "cptac_study_id": cptac_study_id,
            "tcpa_project": tcpa_project,
            "transform": transform,
        },
    }
    write_job(job_id, payload)
    command = [
        PYTHON,
        str(PREDICT_SCRIPT),
        "--input-rna",
        str(input_path),
        "--out-dir",
        str(out_dir),
        "--cptac-cancer-label",
        cptac_cancer_label,
        "--cptac-study-id",
        cptac_study_id,
        "--tcpa-project",
        tcpa_project,
        "--transform",
        transform,
    ]
    background_tasks.add_task(lambda: threading.Thread(target=run_prediction, args=(job_id, command), daemon=True).start())
    return {"job_id": job_id, "status": "queued", "status_url": f"/api/jobs/{job_id}"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    payload = read_job(job_id)
    if payload.get("status") == "done":
        outputs = []
        for name in [
            "predicted_cptac_pdc_total_protein.parquet",
            "predicted_cptac_pdc_phosphosite.parquet",
            "predicted_cptac_pdc_phosphosite_raw_before_sample_median_centering.parquet",
            "predicted_cptac_pdc_phosphosite_sample_medians.tsv",
            "predicted_tcpa_total_antibody.parquet",
            "predicted_tcpa_phospho_antibody.parquet",
            "input_qc_report.json",
            "prediction_summary.json",
        ]:
            p = job_path(job_id) / "outputs" / name
            if p.exists():
                outputs.append({"name": name, "bytes": p.stat().st_size, "url": f"/api/jobs/{job_id}/files/{name}"})
        payload["outputs"] = outputs
    return payload


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str):
    allowed = {
        "predicted_cptac_pdc_total_protein.parquet",
        "predicted_cptac_pdc_phosphosite.parquet",
        "predicted_cptac_pdc_phosphosite_raw_before_sample_median_centering.parquet",
        "predicted_cptac_pdc_phosphosite_sample_medians.tsv",
        "predicted_tcpa_total_antibody.parquet",
        "predicted_tcpa_phospho_antibody.parquet",
        "input_qc_report.json",
        "prediction_summary.json",
        "run.log",
    }
    if name not in allowed:
        raise HTTPException(status_code=404, detail="unknown output")
    path = job_path(job_id) / ("outputs" if name != "run.log" else "") / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="file missing")
    return FileResponse(path, filename=name)


@app.get("/api/jobs/{job_id}/zip")
def job_zip(job_id: str):
    payload = read_job(job_id)
    if payload.get("status") != "done":
        raise HTTPException(status_code=400, detail="job is not done")
    out_dir = job_path(job_id) / "outputs"
    names = [
        "predicted_cptac_pdc_total_protein.parquet",
        "predicted_cptac_pdc_phosphosite.parquet",
        "predicted_cptac_pdc_phosphosite_raw_before_sample_median_centering.parquet",
        "predicted_cptac_pdc_phosphosite_sample_medians.tsv",
        "predicted_tcpa_total_antibody.parquet",
        "predicted_tcpa_phospho_antibody.parquet",
        "input_qc_report.json",
        "prediction_summary.json",
    ]
    zip_path = job_path(job_id) / f"SCP682_outputs_{job_id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = out_dir / name
            if path.exists():
                archive.write(path, arcname=name)
    return FileResponse(zip_path, filename=zip_path.name)
