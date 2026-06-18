#!/usr/bin/env python3
from __future__ import annotations

import cgi
import json
import mimetypes
import os
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


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
    "tcga_cptac_total": ("GDC TCGA supported", "CPTAC/PDC total protein", 5935, 11311, ATLAS / "predictions/tcga_supported_predicted_cptac_total_proteome_fixed_v2_full5fold.parquet"),
    "tcga_cptac_phosphosite": ("GDC TCGA supported", "CPTAC/PDC phosphosite", 5935, 18901, ATLAS / "predictions/tcga_supported_predicted_cptac_phosphosite_fixed_v2_full5fold.parquet"),
    "tcga_tcpa_total": ("GDC TCGA supported", "TCPA total antibody", 5935, 374, ATLAS / "predictions/tcga_supported_predicted_tcpa_total_rppa.parquet"),
    "tcga_tcpa_phospho": ("GDC TCGA supported", "TCPA phospho antibody", 5935, 73, ATLAS / "predictions/tcga_supported_predicted_tcpa_phospho_rppa.parquet"),
    "cbio_cptac_total": ("cBioPortal supported", "CPTAC/PDC total protein", 811, 11311, ATLAS / "predictions/cbioportal_supported_predicted_cptac_total_proteome_fixed_v2_full5fold.parquet"),
    "cbio_cptac_phosphosite": ("cBioPortal supported", "CPTAC/PDC phosphosite", 811, 18901, ATLAS / "predictions/cbioportal_supported_predicted_cptac_phosphosite_fixed_v2_full5fold.parquet"),
    "cbio_tcpa_total": ("cBioPortal supported", "TCPA total antibody", 811, 374, ATLAS / "predictions/cbioportal_supported_predicted_tcpa_total_rppa.parquet"),
    "cbio_tcpa_phospho": ("cBioPortal supported", "TCPA phospho antibody", 811, 73, ATLAS / "predictions/cbioportal_supported_predicted_tcpa_phospho_rppa.parquet"),
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


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def read_job(job_id: str) -> dict:
    path = job_dir(job_id) / "job.json"
    if not path.exists():
        raise FileNotFoundError(job_id)
    return json.loads(path.read_text(encoding="utf-8"))


def write_job(job_id: str, payload: dict) -> None:
    path = job_dir(job_id)
    path.mkdir(parents=True, exist_ok=True)
    (path / "job.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_prediction(job_id: str, command: list[str]) -> None:
    payload = read_job(job_id)
    payload.update({"status": "running", "started_at": time.time(), "command": command})
    write_job(job_id, payload)
    log_path = job_dir(job_id) / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    payload = read_job(job_id)
    payload["returncode"] = proc.returncode
    payload["finished_at"] = time.time()
    payload["log"] = str(log_path)
    summary_path = job_dir(job_id) / "outputs/prediction_summary.json"
    if proc.returncode == 0 and summary_path.exists():
        payload["status"] = "done"
        payload["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        payload["status"] = "failed"
    write_job(job_id, payload)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_json(self, payload: dict, code: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_file(self, path: Path, download_name: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_json({"detail": "file missing"}, 404)
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(path.stat().st_size))
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self.send_file(APP_DIR / "templates/index.html")
            return
        if path.startswith("/static/"):
            rel = path.removeprefix("/static/")
            self.send_file((APP_DIR / "static" / rel).resolve())
            return
        if path == "/api/options":
            self.send_json({"contexts": CONTEXTS, "tcpa_projects": TCPA_PROJECTS})
            return
        if path == "/api/health":
            self.send_json({
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
                "download_files_present": sum(1 for item in DOWNLOADS.values() if item[4].exists()),
                "download_files_total": len(DOWNLOADS),
            })
            return
        if path == "/api/model-contract":
            if not MODEL_CONTRACT.exists():
                self.send_json({"detail": "model contract missing"}, 404)
                return
            self.send_json(json.loads(MODEL_CONTRACT.read_text(encoding="utf-8")))
            return
        if path == "/api/current-model":
            if not CURRENT_MODEL.exists():
                self.send_json({"detail": "current model pointer missing"}, 404)
                return
            self.send_json(json.loads(CURRENT_MODEL.read_text(encoding="utf-8")))
            return
        if path == "/api/final-package":
            self.send_file(FINAL_PACKAGE, FINAL_PACKAGE.name)
            return
        if path == "/api/final-package-sha256":
            self.send_file(FINAL_PACKAGE_SHA256, FINAL_PACKAGE_SHA256.name)
            return
        if path == "/api/downloads":
            rows = []
            for key, (cohort, layer, samples, targets, file_path) in DOWNLOADS.items():
                rows.append({
                    "id": key,
                    "cohort": cohort,
                    "layer": layer,
                    "samples": samples,
                    "targets": targets,
                    "exists": file_path.exists(),
                    "bytes": file_path.stat().st_size if file_path.exists() else 0,
                    "url": f"/api/downloads/{key}",
                })
            self.send_json({"items": rows})
            return
        if path.startswith("/api/downloads/"):
            key = path.rsplit("/", 1)[-1]
            item = DOWNLOADS.get(key)
            if not item:
                self.send_json({"detail": "unknown file id"}, 404)
                return
            self.send_file(item[4], item[4].name)
            return
        if path.startswith("/api/jobs/"):
            parts = path.strip("/").split("/")
            if len(parts) == 3:
                try:
                    payload = read_job(parts[2])
                except FileNotFoundError:
                    self.send_json({"detail": "job not found"}, 404)
                    return
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
                        p = job_dir(parts[2]) / "outputs" / name
                        if p.exists():
                            outputs.append({"name": name, "bytes": p.stat().st_size, "url": f"/api/jobs/{parts[2]}/files/{name}"})
                    payload["outputs"] = outputs
                self.send_json(payload)
                return
            if len(parts) == 4 and parts[3] == "zip":
                job_id = parts[2]
                try:
                    payload = read_job(job_id)
                except FileNotFoundError:
                    self.send_json({"detail": "job not found"}, 404)
                    return
                if payload.get("status") != "done":
                    self.send_json({"detail": "job is not done"}, 400)
                    return
                out_dir = job_dir(job_id) / "outputs"
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
                zip_path = job_dir(job_id) / f"SCP682_outputs_{job_id}.zip"
                with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for name in names:
                        p = out_dir / name
                        if p.exists():
                            archive.write(p, arcname=name)
                self.send_file(zip_path, zip_path.name)
                return
            if len(parts) == 5 and parts[3] == "files":
                job_id, name = parts[2], parts[4]
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
                    self.send_json({"detail": "unknown output"}, 404)
                    return
                p = job_dir(job_id) / ("outputs" if name != "run.log" else "") / name
                self.send_file(p, name)
                return
        self.send_json({"detail": "not found"}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self.send_json({"detail": "not found"}, 404)
            return
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
        upload = form["file"] if "file" in form else None
        if upload is None or not getattr(upload, "filename", ""):
            self.send_json({"detail": "file is required"}, 400)
            return
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in {".tsv", ".txt", ".csv", ".parquet"}:
            self.send_json({"detail": "only tsv, txt, csv and parquet are supported"}, 400)
            return
        job_id = uuid.uuid4().hex[:16]
        base = job_dir(job_id)
        upload_dir = base / "upload"
        out_dir = base / "outputs"
        upload_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        input_path = upload_dir / f"bulk_rna{suffix}"
        with input_path.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        cptac_label = form.getfirst("cptac_cancer_label", "")
        study_id = form.getfirst("cptac_study_id", "")
        tcpa_project = form.getfirst("tcpa_project", "")
        transform = form.getfirst("transform", "none")
        payload = {
            "job_id": job_id,
            "status": "queued",
            "created_at": time.time(),
            "filename": upload.filename,
            "parameters": {
                "cptac_cancer_label": cptac_label,
                "cptac_study_id": study_id,
                "tcpa_project": tcpa_project,
                "transform": transform,
            },
        }
        write_job(job_id, payload)
        command = [
            PYTHON,
            str(PREDICT_SCRIPT),
            "--input-rna", str(input_path),
            "--out-dir", str(out_dir),
            "--cptac-cancer-label", cptac_label,
            "--cptac-study-id", study_id,
            "--tcpa-project", tcpa_project,
            "--transform", transform,
        ]
        threading.Thread(target=run_prediction, args=(job_id, command), daemon=True).start()
        self.send_json({"job_id": job_id, "status": "queued", "status_url": f"/api/jobs/{job_id}"})


def main() -> int:
    host = os.environ.get("RNA2PHOSPHO_HOST", "0.0.0.0")
    port = int(os.environ.get("RNA2PHOSPHO_PORT", "8866"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"SCP682 serving on http://{host}:{port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
