from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/lsy/Infinite_Stream")
SCP682_ROOT = ROOT / "SCP682"
RELEASE_ROOT = SCP682_ROOT / "frozen_release"
RELEASE_NAME = "SCP682_main_exact_scnet_gnn_20260522"
RELEASE_DIR = RELEASE_ROOT / RELEASE_NAME

CORRECT_RUN = ROOT / "SCP682-30/results/20260522_v4_exact_scnet_residual_s0p3_compact_e160"
CORRECT_SCRIPT = ROOT / "remote_scripts/train_scp682_30_v4_exact_scnet_gnn.py"
V4_RELEASE = SCP682_ROOT / "02_results/model_validation/20260503_scp682_v4_0_official_phosphosite_release"
if not V4_RELEASE.exists():
    V4_RELEASE = ROOT / "02_results/model_validation/20260503_scp682_v4_0_official_phosphosite_release"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(str(src))
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(str(src))
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def make_checksums(release_dir: Path) -> None:
    rows = []
    for path in sorted(p for p in release_dir.rglob("*") if p.is_file()):
        if path.name in {"sha256_manifest.tsv", "file_inventory.tsv"} and path.parent.name == "checksums":
            continue
        rows.append((rel(path, release_dir), path.stat().st_size, sha256_file(path)))
    ensure_dir(release_dir / "checksums")
    inv = release_dir / "checksums/file_inventory.tsv"
    man = release_dir / "checksums/sha256_manifest.tsv"
    inv.write_text("path\tsize_bytes\n" + "\n".join(f"{p}\t{s}" for p, s, _ in rows) + "\n", encoding="utf-8")
    man.write_text("path\tsize_bytes\tsha256\n" + "\n".join(f"{p}\t{s}\t{h}" for p, s, h in rows) + "\n", encoding="utf-8")


def sanitize_summary(src: Path, dst: Path) -> None:
    replacements = {
        "scp682_v4_baseline": "SCP682_V4_baseline",
        "scp682_30_v4_exact_scnet_gnn": "SCP682_main",
    }
    ensure_dir(dst.parent)
    lines = src.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        parts = line.split("\t")
        if parts and parts[0] in replacements:
            parts[0] = replacements[parts[0]]
        out.append("\t".join(parts))
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")


def sanitize_per_site(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def build_release() -> None:
    for p in [
        CORRECT_RUN / "models/scp682_30_v4_exact_scnet_gnn_best.pt",
        CORRECT_RUN / "predictions/scp682_30_v4_exact_scnet_gnn_oof_phosphosite_best.parquet",
        CORRECT_RUN / "tables/model_summary_best.tsv",
        CORRECT_RUN / "reports/final_summary.json",
        CORRECT_RUN / "reports/input_graph_summary.json",
        CORRECT_SCRIPT,
        V4_RELEASE / "predictions/SCP682_v4_0_internal_cptac_pdc_phosphosite.parquet",
    ]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    previous = None
    if RELEASE_DIR.exists():
        previous = RELEASE_ROOT / f"{RELEASE_NAME}.wrong_baseline_backup_{stamp}"
        shutil.move(str(RELEASE_DIR), str(previous))

    tar_path = RELEASE_ROOT / f"{RELEASE_NAME}.tar"
    sha_path = RELEASE_ROOT / f"{RELEASE_NAME}.tar.sha256"
    if tar_path.exists():
        shutil.move(str(tar_path), str(RELEASE_ROOT / f"{RELEASE_NAME}.wrong_baseline_backup_{stamp}.tar"))
    if sha_path.exists():
        shutil.move(str(sha_path), str(RELEASE_ROOT / f"{RELEASE_NAME}.wrong_baseline_backup_{stamp}.tar.sha256"))

    ensure_dir(RELEASE_DIR)
    ensure_dir(RELEASE_DIR / "models")
    ensure_dir(RELEASE_DIR / "predictions")
    ensure_dir(RELEASE_DIR / "performance")
    ensure_dir(RELEASE_DIR / "reports")
    ensure_dir(RELEASE_DIR / "scripts")
    ensure_dir(RELEASE_DIR / "training_set")
    ensure_dir(RELEASE_DIR / "priors")

    copy_file(CORRECT_RUN / "models/scp682_30_v4_exact_scnet_gnn_best.pt", RELEASE_DIR / "models/scp682_main_v4_exact_scnet_gnn_best.pt")
    copy_file(CORRECT_RUN / "predictions/scp682_30_v4_exact_scnet_gnn_oof_phosphosite_best.parquet", RELEASE_DIR / "predictions/scp682_main_oof_phosphosite.parquet")
    sanitize_summary(CORRECT_RUN / "tables/model_summary_best.tsv", RELEASE_DIR / "performance/model_summary.tsv")
    sanitize_per_site(CORRECT_RUN / "tables/per_site_spearman_best.tsv", RELEASE_DIR / "performance/per_site_spearman.tsv")
    for src in sorted((CORRECT_RUN / "tables").glob("sample_attention_epoch_*.tsv")):
        copy_file(src, RELEASE_DIR / "performance" / src.name)
    copy_file(CORRECT_RUN / "logs/training_history.tsv", RELEASE_DIR / "logs/training_history.tsv")
    copy_file(CORRECT_RUN / "logs/stdout.log", RELEASE_DIR / "logs/train_stdout.log")
    copy_file(CORRECT_RUN / "logs/stderr.log", RELEASE_DIR / "logs/train_stderr.log")
    copy_file(CORRECT_RUN / "reports/final_summary.json", RELEASE_DIR / "reports/final_summary.json")
    copy_file(CORRECT_RUN / "reports/input_graph_summary.json", RELEASE_DIR / "reports/input_graph_summary.json")
    copy_file(CORRECT_SCRIPT, RELEASE_DIR / "scripts/train_scp682_main_v4_exact_scnet_gnn.py")

    copy_file(V4_RELEASE / "predictions/SCP682_v4_0_internal_cptac_pdc_phosphosite.parquet", RELEASE_DIR / "training_set/v4_phosphosite_baseline.parquet")
    raw_v4 = V4_RELEASE / "predictions/SCP682_v4_0_internal_cptac_pdc_phosphosite_raw_before_sample_median_centering.parquet"
    if raw_v4.exists():
        copy_file(raw_v4, RELEASE_DIR / "training_set/v4_phosphosite_baseline_raw_before_sample_median_centering.parquet")

    if previous is not None:
        old_training = previous / "training_set"
        old_priors = previous / "priors"
    else:
        old_training = RELEASE_DIR / "_missing"
        old_priors = RELEASE_DIR / "_missing"
    for name in [
        "observed_phosphosite.parquet",
        "phosphosite_target_manifest.tsv",
        "total_protein_target_manifest.tsv",
        "rna_gene_order.tsv",
        "sample_manifest.tsv",
        "vae_latent_training_reference.tsv",
        "vae_latent_training_reference.parquet",
    ]:
        src = old_training / name
        if src.exists():
            copy_file(src, RELEASE_DIR / "training_set" / name)
    if old_priors.exists():
        copy_tree(old_priors, RELEASE_DIR / "priors")
    copy_tree(V4_RELEASE, RELEASE_DIR / "v4_baseline_release")

    current = {
        "model_id": "SCP682",
        "current_main_version": "SCP682_main_exact_scnet_gnn_20260522",
        "status": "FINAL_MAIN_MODEL",
        "decision_date": "2026-05-22",
        "corrected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "release_path": str(RELEASE_DIR),
        "primary_checkpoint": str(RELEASE_DIR / "models/scp682_main_v4_exact_scnet_gnn_best.pt"),
        "primary_oof_prediction": str(RELEASE_DIR / "predictions/scp682_main_oof_phosphosite.parquet"),
        "v4_baseline_prediction": str(RELEASE_DIR / "training_set/v4_phosphosite_baseline.parquet"),
        "primary_summary": str(RELEASE_DIR / "performance/model_summary.tsv"),
        "formula": "phosphosite_hat = SCP682_V4_phosphosite_hat + 0.3 * exact_scnet_gnn_delta",
        "previous_incorrect_release_backup": str(previous) if previous else None,
    }
    write_json(RELEASE_DIR / "CURRENT_MODEL.json", current)

    contract = {
        "model_id": "SCP682",
        "current_main_version": "SCP682_main_exact_scnet_gnn_20260522",
        "status": "FINAL_MAIN_MODEL",
        "model_principle": "冻结 SCP682 V4 磷酸化基线预测，并在其上加入 exact ScNET 双轴图残差修正。",
        "formula": {
            "v4_baseline": "SCP682_V4_phosphosite_hat = B_V4(RNA)",
            "graph_residual": "exact_scnet_gnn_delta = G_theta(SCP682_V4_phosphosite_hat, site_graph, sample_graph)",
            "phosphosite": "phosphosite_hat = SCP682_V4_phosphosite_hat + 0.3 * exact_scnet_gnn_delta",
        },
        "training_data": {
            "n_samples": 1431,
            "n_sites": 18592,
            "observed_phosphosite": "training_set/observed_phosphosite.parquet",
            "v4_baseline": "training_set/v4_phosphosite_baseline.parquet",
        },
        "graph_priors": {
            "site_edges": 420102,
            "sample_edges": 21813,
            "sources": ["CoPheeMap", "CoPheeKSA", "KSTAR"],
        },
        "performance": {
            "summary": "performance/model_summary.tsv",
            "SCP682_V4_baseline_median_spearman": 0.3053003935468414,
            "SCP682_main_median_spearman": 0.5882286248870318,
        },
        "public_interface": "外部样本先通过 SCP682 V4 得到同一位点空间的磷酸化基线预测，再进入 exact ScNET 图残差层；主接口不要求用户提供 parent_only、ridge_direct、rna_direct 三路矩阵。",
    }
    write_json(RELEASE_DIR / "SCP682_model_contract.json", contract)

    readme = """# SCP682 主模型冻结包

本包是修正后的 SCP682 主模型冻结包。主模型定义为冻结的 SCP682 V4 磷酸化基线预测加 exact ScNET 双轴图残差修正。

```text
SCP682_V4_phosphosite_hat = B_V4(RNA)
exact_scnet_gnn_delta = G_theta(SCP682_V4_phosphosite_hat, site_graph, sample_graph)
phosphosite_hat = SCP682_V4_phosphosite_hat + 0.3 * exact_scnet_gnn_delta
```

本包不再把 `parent_only`、`ridge_direct`、`rna_direct` 三路矩阵作为主模型入口。三路候选属于早期 V4 构建内部细节，冻结主模型的公开接口是 SCP682 V4 基线预测。

内部 OOF 性能见 `performance/model_summary.tsv`：

```text
model                 n_targets  median_spearman  mean_spearman  ge_0_3  ge_0_5
SCP682_V4_baseline    18413      0.305300         0.300270       9452    1165
SCP682_main           18413      0.588229         0.578465       18268   15481
```

主权重：`models/scp682_main_v4_exact_scnet_gnn_best.pt`

训练集 OOF 预测：`predictions/scp682_main_oof_phosphosite.parquet`

V4 基线预测：`training_set/v4_phosphosite_baseline.parquet`
"""
    write_text(RELEASE_DIR / "README.md", readme)

    usage = """# SCP682 主模型使用方法

## 输入和输出

SCP682 用 bulk RNA 表达预测样本级磷酸化位点丰度。主模型由两部分组成：冻结的 SCP682 V4 磷酸化基线预测和 exact ScNET 图残差层。

```text
SCP682_V4_phosphosite_hat = B_V4(RNA)
exact_scnet_gnn_delta = G_theta(SCP682_V4_phosphosite_hat, site_graph, sample_graph)
phosphosite_hat = SCP682_V4_phosphosite_hat + 0.3 * exact_scnet_gnn_delta
```

## 包内关键文件

| 文件 | 用途 |
|---|---|
| `models/scp682_main_v4_exact_scnet_gnn_best.pt` | exact ScNET 图残差层权重 |
| `training_set/v4_phosphosite_baseline.parquet` | 训练集 SCP682 V4 磷酸化基线预测 |
| `predictions/scp682_main_oof_phosphosite.parquet` | 主模型训练集 OOF 输出 |
| `training_set/observed_phosphosite.parquet` | 训练集实测磷酸化矩阵 |
| `training_set/phosphosite_target_manifest.tsv` | 位点顺序和位点信息 |
| `training_set/rna_gene_order.tsv` | RNA 基因顺序 |
| `priors/` | CoPheeMap、CoPheeKSA、KSTAR 图先验 |
| `v4_baseline_release/` | SCP682 V4 基线释放文件 |
| `performance/model_summary.tsv` | 性能摘要 |
| `checksums/sha256_manifest.tsv` | 文件完整性校验 |

## 读取训练集预测

```python
from pathlib import Path
import pandas as pd

root = Path(r"E:\\data\\gongke\\TCGA-TCPA\\SCP682_MAIN")

v4 = pd.read_parquet(root / "training_set" / "v4_phosphosite_baseline.parquet")
pred = pd.read_parquet(root / "predictions" / "scp682_main_oof_phosphosite.parquet")
obs = pd.read_parquet(root / "training_set" / "observed_phosphosite.parquet")
site_manifest = pd.read_csv(root / "training_set" / "phosphosite_target_manifest.tsv", sep="\\t")
```

## 外部样本推理

外部 RNA 样本需要先按 `training_set/rna_gene_order.tsv` 对齐基因，并通过包内 `v4_baseline_release/` 对应的 SCP682 V4 推理流程得到同一位点空间的 `SCP682_V4_phosphosite_hat`。随后把该 V4 基线预测输入 exact ScNET 图残差层，得到最终磷酸化预测。

公开接口不要求用户自己提供 `parent_only`、`ridge_direct`、`rna_direct` 三路矩阵。
"""
    write_text(RELEASE_DIR / "USAGE.md", usage)

    audit = {
        "reason": "原冻结包误把 parent_only、ridge_direct、rna_direct 三路均值作为主模型基线。该包已改为实际要求的 SCP682 V4 基线加 exact ScNET 图残差。",
        "correct_source_run": str(CORRECT_RUN),
        "correct_training_script": str(CORRECT_SCRIPT),
        "previous_backup": str(previous) if previous else None,
        "correct_formula": "phosphosite_hat = SCP682_V4_phosphosite_hat + 0.3 * exact_scnet_gnn_delta",
    }
    write_json(RELEASE_DIR / "CORRECTION_AUDIT.json", audit)

    make_checksums(RELEASE_DIR)

    subprocess.run(["tar", "-cf", str(tar_path), "-C", str(RELEASE_ROOT), RELEASE_NAME], check=True)
    tar_sha = sha256_file(tar_path)
    sha_path.write_text(f"{tar_sha}  {tar_path.name}\n", encoding="utf-8")

    write_json(SCP682_ROOT / "SCP682_CURRENT.json", current)
    write_json(SCP682_ROOT / "SCP682_model_contract.json", contract)
    write_text(SCP682_ROOT / "SCP682_TECHNICAL.md", readme)

    root_rows = []
    for path in [tar_path, sha_path, RELEASE_DIR / "CURRENT_MODEL.json", RELEASE_DIR / "README.md", RELEASE_DIR / "SCP682_model_contract.json", RELEASE_DIR / "USAGE.md"]:
        root_rows.append(f"{sha256_file(path)}  ./{path.relative_to(SCP682_ROOT).as_posix()}")
    (SCP682_ROOT / "sha256_manifest.tsv").write_text("\n".join(root_rows) + "\n", encoding="utf-8")

    print(json.dumps({
        "release_dir": str(RELEASE_DIR),
        "tar": str(tar_path),
        "tar_sha256": tar_sha,
        "previous_backup": str(previous) if previous else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build_release()
