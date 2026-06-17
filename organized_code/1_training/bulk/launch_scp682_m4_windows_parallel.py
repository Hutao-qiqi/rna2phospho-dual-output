import os
import subprocess
import time
from pathlib import Path


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
PYTHON = Path(r"D:\Tools\anaconda3\envs\scvi-env\python.exe")
SCRIPT = ROOT / "remote_scripts" / "train_scp682_missing_ablation_degree_rewire.py"
SUMMARY = ROOT / "remote_scripts" / "summarize_scp682_missing_ablation_grid.py"
OUT_ROOT = ROOT / "02_results" / "model_validation" / "20260602_m4_knowledge_graph_controls_e160_windows"
PACKAGE_DIR = ROOT / "SCP682_MAIN"
PRIOR_ROOT = ROOT / "01_data" / "pathway_prior"
BASELINE = PACKAGE_DIR / "inputs" / "general_baseline_predictions" / "general_baseline_internal_cptac_pdc_phosphosite.parquet"
RNA = Path(r"D:\data\lsy\01_data\multi_omics\processed\pancancer_multi_task_locked_v2\rna_log2_tpm_paired.parquet")
SAMPLE_MANIFEST = Path(r"D:\data\lsy\01_data\multi_omics\processed\pancancer_multi_task_locked_v2\sample_manifest.tsv")

TASKS = [
    ("axis_dual_edge_rewired_all", "dual", "rewired_all", "0"),
    ("axis_dual_edge_no_copheemap", "dual", "no_copheemap", "1"),
    ("axis_dual_edge_no_copheeksa", "dual", "no_copheeksa", "0"),
    ("axis_dual_edge_no_kstar", "dual", "no_kstar", "1"),
]


def log(message: str) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "run_status.txt").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    print(message, flush=True)


def command_for(name: str, axis: str, edge: str) -> list[str]:
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    return [
        str(PYTHON),
        str(SCRIPT),
        "--package-dir",
        str(PACKAGE_DIR),
        "--prior-root",
        str(PRIOR_ROOT),
        "--output-dir",
        str(out),
        "--general-baseline-path",
        str(BASELINE),
        "--rna-path",
        str(RNA),
        "--sample-manifest-path",
        str(SAMPLE_MANIFEST),
        "--group-column",
        "cancer_label",
        "--device",
        "cuda:0",
        "--epochs",
        "160",
        "--batch-size",
        "8",
        "--hidden",
        "96",
        "--latent",
        "32",
        "--inter-dim",
        "96",
        "--embd-dim",
        "32",
        "--num-layers",
        "1",
        "--axis-mode",
        axis,
        "--edge-mode",
        edge,
    ]


def launch(task: tuple[str, str, str, str]) -> subprocess.Popen:
    name, axis, edge, gpu = task
    logs = OUT_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{name}.stdout.log"
    stderr_path = logs / f"{name}.stderr.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    log(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] start {name} axis={axis} edge={edge} gpu={gpu}")
    process = subprocess.Popen(command_for(name, axis, edge), stdout=stdout, stderr=stderr, env=env)
    process._scp682_stdout = stdout  # type: ignore[attr-defined]
    process._scp682_stderr = stderr  # type: ignore[attr-defined]
    process._scp682_name = name  # type: ignore[attr-defined]
    return process


def close_logs(process: subprocess.Popen) -> None:
    process._scp682_stdout.close()  # type: ignore[attr-defined]
    process._scp682_stderr.close()  # type: ignore[attr-defined]


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "run_status.txt").write_text(f"start {time.strftime('%Y-%m-%dT%H:%M:%S')}\n", encoding="utf-8")
    if (OUT_ROOT / "fatal.log").exists():
        (OUT_ROOT / "fatal.log").unlink()
    if (OUT_ROOT / "done.txt").exists():
        (OUT_ROOT / "done.txt").unlink()

    for start in range(0, len(TASKS), 2):
        active = [launch(task) for task in TASKS[start : start + 2]]
        for process in active:
            code = process.wait()
            close_logs(process)
            name = process._scp682_name  # type: ignore[attr-defined]
            if code != 0:
                (OUT_ROOT / "fatal.log").write_text(
                    f"failed {name} exit_code={code} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n",
                    encoding="utf-8",
                )
                log(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] failed {name} exit_code={code}")
                return code
            log(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] done {name}")

    summary_cmd = [
        str(PYTHON),
        str(SUMMARY),
        "--grid-dir",
        str(OUT_ROOT),
        "--output",
        str(OUT_ROOT / "tables" / "m4_knowledge_graph_controls_summary.tsv"),
    ]
    log(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] start summary")
    code = subprocess.call(summary_cmd)
    if code != 0:
        (OUT_ROOT / "fatal.log").write_text(
            f"failed summary exit_code={code} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n",
            encoding="utf-8",
        )
        return code
    (OUT_ROOT / "done.txt").write_text(f"done {time.strftime('%Y-%m-%dT%H:%M:%S')}\n", encoding="utf-8")
    log(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] done all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
