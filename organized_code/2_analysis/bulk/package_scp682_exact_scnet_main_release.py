#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path("/data/lsy/Infinite_Stream")
SCP682 = ROOT / "SCP682"
SOURCE = ROOT / "02_results/model_validation/20260522_scp682_exact_scnet_gnn_v1_ubuntu_full"
TRAINING = ROOT / "SCP682-22/frozen_release/SCP682_22_paper_package_20260520/training_set"
SCRIPT = ROOT / "remote_scripts/train_scp682_exact_scnet_gnn_v1.py"
PRIOR_ROOT = ROOT / "01_data/pathway_prior"
RELEASE_ID = "SCP682_main_exact_scnet_gnn_20260522"
RELEASE = SCP682 / "frozen_release" / RELEASE_ID


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    for sub in ["models", "predictions", "performance", "logs", "reports", "scripts", "training_set", "checksums", "priors"]:
        (RELEASE / sub).mkdir(parents=True, exist_ok=True)

    copy_file(SOURCE / "models/scp682_exact_scnet_gnn_best.pt", RELEASE / "models/scp682_exact_scnet_gnn_best.pt")
    copy_file(SOURCE / "predictions/scp682_exact_scnet_gnn_oof_phosphosite_best.parquet", RELEASE / "predictions/scp682_exact_scnet_gnn_oof_phosphosite_best.parquet")
    copy_file(SOURCE / "tables/model_summary_best.tsv", RELEASE / "performance/model_summary.tsv")
    copy_file(SOURCE / "tables/per_site_spearman_best.tsv", RELEASE / "performance/per_site_spearman.tsv")
    copy_file(SOURCE / "tables/sample_attention_epoch_30.tsv", RELEASE / "performance/sample_attention_epoch_30.tsv")
    copy_file(SOURCE / "tables/sample_attention_epoch_60.tsv", RELEASE / "performance/sample_attention_epoch_60.tsv")
    copy_file(SOURCE / "logs/training_history.tsv", RELEASE / "logs/training_history.tsv")
    copy_file(SOURCE / "logs/stdout.log", RELEASE / "logs/train_stdout.log")
    copy_file(SOURCE / "logs/stderr.log", RELEASE / "logs/train_stderr.log")
    copy_file(SOURCE / "reports/input_graph_summary.json", RELEASE / "reports/input_graph_summary.json")
    copy_file(SOURCE / "reports/final_summary.json", RELEASE / "reports/final_summary.json")
    copy_file(SCRIPT, RELEASE / "scripts/train_scp682_exact_scnet_gnn_v1.py")

    training_files = [
        "observed_phosphosite.parquet",
        "oof_candidate_parent_only_phosphosite.parquet",
        "oof_candidate_ridge_direct_phosphosite.parquet",
        "oof_candidate_rna_direct_phosphosite.parquet",
        "phosphosite_target_manifest.tsv",
        "sample_manifest.tsv",
        "rna_gene_order.tsv",
        "total_protein_target_manifest.tsv",
        "vae_latent_training_reference.parquet",
        "vae_latent_training_reference.tsv",
        "model_level_oof_stacking_summary.tsv",
        "per_site_meta_fold_stack_weights.tsv",
    ]
    for name in training_files:
        src = TRAINING / name
        if src.exists():
            copy_file(src, RELEASE / "training_set" / name)

    prior_files = [
        PRIOR_ROOT / "processed/copheemap_v1/copheemap_site_id_to_model_gene_site.tsv",
        PRIOR_ROOT / "raw/copheemap/CoPheeMap/Supplementary_table/Table_S2_CoPheeMap.tsv.zip",
        PRIOR_ROOT / "raw/copheemap/CoPheeMap/CoPheeKSA/positive_KSA.csv",
        PRIOR_ROOT / "intermediate/kstar_20260516/kstar_default_network_edges_long.tsv",
    ]
    for src in prior_files:
        if src.exists():
            copy_file(src, RELEASE / "priors" / src.name)

    model_summary = pd.read_csv(RELEASE / "performance/model_summary.tsv", sep="\t")
    graph_summary = json.loads((RELEASE / "reports/input_graph_summary.json").read_text(encoding="utf-8"))
    rows = []
    for p in sorted(RELEASE.rglob("*")):
        if p.is_file() and p.name not in {"sha256_manifest.tsv", "file_inventory.tsv"}:
            rel = p.relative_to(RELEASE).as_posix()
            rows.append({"path": rel, "size_bytes": p.stat().st_size, "sha256": sha256(p)})
    pd.DataFrame(rows).to_csv(RELEASE / "checksums/sha256_manifest.tsv", sep="\t", index=False)
    pd.DataFrame([{"path": r["path"], "size_bytes": r["size_bytes"]} for r in rows]).to_csv(RELEASE / "checksums/file_inventory.tsv", sep="\t", index=False)

    contract = {
        "model_id": "SCP682",
        "current_main_version": RELEASE_ID,
        "decision_date": "2026-05-22",
        "status": "FINAL_MAIN_MODEL",
        "model_principle": "SCP682 main phosphosite model is the exact ScNET-style graph residual model trained on the locked CPTAC/PDC phosphosite matrix.",
        "architecture_formula": {
            "baseline": "baseline_mean = mean(parent_only, ridge_direct, rna_direct)",
            "phosphosite": "phosphosite_hat = baseline_mean + 0.3 * exact_scnet_gnn_delta",
            "graph_core": "site graph from original CoPheeMap, CoPheeKSA and KSTAR priors; sample graph from baseline-matrix KNN; exact ScNET-style mutual encoder over site and sample axes.",
            "postprocess": "OOF phosphosite prediction table is stored as released prediction matrix."
        },
        "train_objective": "masked Huber + cosine loss on phosphosite and residual targets, plus site graph reconstruction, baseline reconstruction and attention penalty",
        "scope": {
            "phosphosite": "main reported SCP682 phosphosite model",
            "total_protein": "unchanged SCP682 total protein component from previous full release when total protein output is needed"
        },
        "internal_summary": model_summary.to_dict(orient="records"),
        "graph_summary": graph_summary,
        "package_layout": {
            "models": "best exact ScNET GNN checkpoint",
            "predictions": "released OOF phosphosite predictions",
            "performance": "model-level and site-level internal evaluation tables, plus sample-attention snapshots",
            "training_set": "locked observed phosphosite matrix and baseline candidate matrices used to train the main graph model",
            "priors": "graph prior source tables used for site graph construction",
            "scripts": "training script used for the frozen run",
            "checksums": "sha256 manifest and file inventory"
        },
        "release_path": str(RELEASE),
        "source_result_path": str(SOURCE),
    }
    (RELEASE / "SCP682_model_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    (SCP682 / "SCP682_model_contract.previous_20260522.json").write_text((SCP682 / "SCP682_model_contract.json").read_text(encoding="utf-8"), encoding="utf-8")
    (SCP682 / "SCP682_model_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    current = {
        "model_id": "SCP682",
        "current_main_version": RELEASE_ID,
        "status": "FINAL_MAIN_MODEL",
        "decision_date": "2026-05-22",
        "release_path": str(RELEASE),
        "contract": str(SCP682 / "SCP682_model_contract.json"),
        "primary_checkpoint": str(RELEASE / "models/scp682_exact_scnet_gnn_best.pt"),
        "primary_oof_prediction": str(RELEASE / "predictions/scp682_exact_scnet_gnn_oof_phosphosite_best.parquet"),
        "primary_summary": str(RELEASE / "performance/model_summary.tsv"),
    }
    (RELEASE / "CURRENT_MODEL.json").write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    (SCP682 / "SCP682_CURRENT.previous_20260522.json").write_text((SCP682 / "SCP682_CURRENT.json").read_text(encoding="utf-8"), encoding="utf-8")
    (SCP682 / "SCP682_CURRENT.json").write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = f"""# SCP682 主模型冻结包：{RELEASE_ID}

本包为当前 SCP682 主模型。模型采用 exact ScNET-style phosphosite graph residual 架构，在锁定 CPTAC/PDC phosphosite 训练矩阵上训练。

主公式：

```text
baseline_mean = mean(parent_only, ridge_direct, rna_direct)
phosphosite_hat = baseline_mean + 0.3 * exact_scnet_gnn_delta
```

内部 OOF 性能见 `performance/model_summary.tsv`：

```text
{model_summary.to_string(index=False)}
```

图构建摘要见 `reports/input_graph_summary.json`。模型权重在 `models/scp682_exact_scnet_gnn_best.pt`，释放预测矩阵在 `predictions/scp682_exact_scnet_gnn_oof_phosphosite_best.parquet`。
"""
    write_text(RELEASE / "README.md", readme)
    write_text(SCP682 / "SCP682_TECHNICAL.md", readme)

    rows = []
    for p in sorted(RELEASE.rglob("*")):
        if p.is_file() and not str(p).endswith(".tar"):
            rows.append({"path": p.relative_to(RELEASE).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256(p)})
    pd.DataFrame(rows).to_csv(RELEASE / "checksums/sha256_manifest.tsv", sep="\t", index=False)
    tar_path = SCP682 / "frozen_release" / f"{RELEASE_ID}.tar"
    if tar_path.exists():
        tar_path.unlink()
    subprocess.run(["tar", "-cf", str(tar_path), "-C", str(RELEASE.parent), RELEASE.name], check=True)
    (tar_path.with_suffix(".tar.sha256")).write_text(f"{sha256(tar_path)}  {tar_path.name}\n", encoding="utf-8")
    print(json.dumps({"release": str(RELEASE), "tar": str(tar_path), "tar_sha256": str(tar_path.with_suffix('.tar.sha256'))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
