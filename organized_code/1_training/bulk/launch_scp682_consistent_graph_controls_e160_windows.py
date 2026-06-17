import os
import subprocess
import time
from pathlib import Path


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
PYTHON = Path(r"D:\Tools\anaconda3\envs\scvi-env\python.exe")
SCRIPT = ROOT / "remote_scripts" / "train_scp682_consistent_graph_controls.py"
SUMMARY = ROOT / "remote_scripts" / "summarize_scp682_missing_ablation_grid.py"
OUT_ROOT = ROOT / "02_results" / "model_validation" / "20260603_consistent_graph_controls_e160_windows"
PACKAGE_DIR = ROOT / "SCP682_MAIN"
PRIOR_ROOT = ROOT / "01_data" / "pathway_prior"
BASELINE = PACKAGE_DIR / "training_set" / "v4_phosphosite_baseline.parquet"


TASKS = [
    ("axis_dual_edge_all", "dual", "all", "0"),
    ("axis_site_only_edge_all", "site_only", "all", "1"),
    ("axis_sample_only_edge_all", "sample_only", "all", "0"),
    ("axis_dual_edge_rewired_all", "dual", "rewired_all", "1"),
    ("axis_dual_edge_no_copheemap", "dual", "no_copheemap", "0"),
    ("axis_dual_edge_no_copheeksa", "dual", "no_copheeksa", "1"),
    ("axis_dual_edge_no_kstar", "dual", "no_kstar", "0"),
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
        "--v4-baseline-path",
        str(BASELINE),
        "--device",
        "cuda:0",
        "--epochs",
        "160",
        "--batch-size",
        "8",
        "--hidden",
        "160",
        "--latent",
        "64",
        "--inter-dim",
        "192",
        "--embd-dim",
        "64",
        "--num-layers",
        "2",
        "--axis-mode",
        axis,
        "--edge-mode",
        edge,
        "--shrinkage",
        "0.3",
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
    for stale in ["fatal.log", "done.txt"]:
        path = OUT_ROOT / stale
        if path.exists():
            path.unlink()

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
        str(OUT_ROOT / "tables" / "consistent_graph_controls_summary.tsv"),
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
