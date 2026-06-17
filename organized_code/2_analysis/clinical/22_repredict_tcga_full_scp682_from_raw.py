#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/data/lsy/Infinite_Stream")
PORTABLE_DEFAULT = ROOT / "SCP682_PORTABLE"
OUT_DEFAULT = ROOT / "02_results/model_prediction/20260529_tcga_full_scp682_main_reprediction_v1"

GDC_MAPPING = Path("/data/gongke/shared/TCGA/metadata/file_sample_mapping.tsv")
GDC_RNASEQ_DIR = Path("/data/gongke/shared/TCGA/data/rnaseq")
SURVIVAL = Path("/data/gongke/shared/TCGA/data/xena/toil/TCGA_survival_data.tsv")
TOIL_TPM = Path("/data/gongke/shared/TCGA/data/xena/toil/tcga_Kallisto_tpm.gz")
TOIL_COUNTS = Path("/data/gongke/shared/TCGA/data/xena/toil/tcga_Kallisto_est_counts.gz")
SUBTYPE = Path("/data/gongke/shared/TCGA/data/xena/pancanatlas/TCGASubtype.20170308.tsv")


def tcga_patient_id(sample: str) -> str:
    parts = str(sample).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else ""


def tcga_sample_type_id(sample: str) -> str:
    parts = str(sample).split("-")
    if len(parts) >= 4:
        return "-".join(parts[:3] + [parts[3][:2]])
    return str(sample)


def tcga_sample_type_code(sample: str) -> str:
    parts = str(sample).split("-")
    return parts[3][:2] if len(parts) >= 4 else ""


def setup_logger(outdir: Path) -> logging.Logger:
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tcga_scp682_reprediction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    fh = logging.FileHandler(outdir / "logs/run.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def ensure_dirs(outdir: Path) -> None:
    for sub in ["predictions", "tables", "logs", "intermediate", "reports"]:
        (outdir / sub).mkdir(parents=True, exist_ok=True)


def read_model_genes(portable_dir: Path) -> list[str]:
    path = portable_dir / "training_set/rna_gene_order.tsv"
    genes = pd.read_csv(path, sep="\t")["gene_symbol"].astype(str).tolist()
    if not genes:
        raise RuntimeError(f"模型基因列表为空: {path}")
    return genes


def audit_portable_package(portable_dir: Path) -> pd.DataFrame:
    rows = [
        ("portable_dir", portable_dir, "主模型可复制包"),
        ("entrypoint", portable_dir / "predict_scp682.py", "主模型外部推理入口"),
        ("v4_engine", portable_dir / "scp682_v4_engine.py", "V4 基线推理封装"),
        ("graph_runtime", portable_dir / "scp682_graph_runtime.py", "exact ScNET GNN 残差运行时"),
        ("primary_checkpoint", portable_dir / "models/scp682_main_v4_exact_scnet_gnn_best.pt", "主模型权重"),
        ("graph_state", portable_dir / "models/scp682_graph_runtime_state.pt", "样本图和位点图运行时状态"),
        ("v4_baseline", portable_dir / "training_set/v4_phosphosite_baseline.parquet", "训练集 V4 基线"),
        ("v4_gene_order", portable_dir / "training_set/rna_gene_order.tsv", "RNA 输入基因顺序"),
        ("oof_prediction", portable_dir / "predictions/scp682_main_oof_phosphosite.parquet", "训练集 OOF 主模型输出"),
    ]
    out = []
    for key, path, use in rows:
        p = Path(path)
        out.append(
            {
                "item": key,
                "path": str(p),
                "exists": p.exists(),
                "use": use,
            }
        )
    return pd.DataFrame(out)


def load_expression_manifest(logger: logging.Logger) -> pd.DataFrame:
    mapping = pd.read_csv(GDC_MAPPING, sep="\t")
    expr = mapping[mapping["data_type"].eq("Gene Expression Quantification")].copy()
    expr["rna_file"] = [
        str(GDC_RNASEQ_DIR / file_id / file_name)
        for file_id, file_name in zip(expr["file_id"].astype(str), expr["file_name"].astype(str))
    ]
    expr["rna_file_exists"] = expr["rna_file"].map(lambda x: Path(x).exists())
    expr = expr[expr["rna_file_exists"]].copy()
    expr["sample_id"] = expr["sample_barcode"].astype(str)
    expr["tcga_sample_type_id"] = expr["sample_id"].map(tcga_sample_type_id)
    expr["tcga_patient_id"] = expr["sample_id"].map(tcga_patient_id)
    expr["sample_type_code"] = expr["sample_id"].map(tcga_sample_type_code)
    expr["cancer"] = expr["project_id"].astype(str).str.replace("^TCGA-", "", regex=True)
    expr = expr.sort_values(["project_id", "sample_id", "file_id"]).drop_duplicates("sample_id", keep="first")
    logger.info("GDC STAR RNA 文件数=%d；唯一样本数=%d；癌种数=%d", len(expr), expr["sample_id"].nunique(), expr["project_id"].nunique())
    return expr


def load_survival() -> pd.DataFrame:
    surv = pd.read_csv(SURVIVAL, sep="\t")
    surv["survival_sample"] = surv["sample"].astype(str)
    surv["tcga_sample_type_id"] = surv["survival_sample"].map(tcga_sample_type_id)
    surv["survival_event"] = pd.to_numeric(surv["OS"], errors="coerce")
    surv["survival_time"] = pd.to_numeric(surv["OS.time"], errors="coerce")
    cols = ["tcga_sample_type_id", "survival_sample", "survival_time", "survival_event", "OS", "OS.time"]
    surv = surv.sort_values(["tcga_sample_type_id", "survival_sample"]).drop_duplicates("tcga_sample_type_id", keep="first")
    return surv[cols]


def write_coverage_tables(expr: pd.DataFrame, surv: pd.DataFrame, outdir: Path, logger: logging.Logger) -> pd.DataFrame:
    merged = expr.merge(surv, on="tcga_sample_type_id", how="left")
    merged["has_os_survival"] = merged["survival_time"].notna() & merged["survival_event"].notna()
    merged["survival_event"] = pd.to_numeric(merged["survival_event"], errors="coerce")
    manifest_cols = [
        "sample_id",
        "tcga_sample_type_id",
        "tcga_patient_id",
        "project_id",
        "cancer",
        "sample_type",
        "sample_type_code",
        "file_id",
        "file_name",
        "rna_file",
        "has_os_survival",
        "survival_time",
        "survival_event",
    ]
    merged[manifest_cols].to_csv(outdir / "tables/tcga_scp682_prediction_sample_manifest.tsv", sep="\t", index=False)

    overlap = merged[["sample_id", "tcga_sample_type_id", "project_id", "cancer", "has_os_survival", "survival_time", "survival_event"]].copy()
    overlap.to_csv(outdir / "tables/tcga_scp682_prediction_survival_overlap.tsv", sep="\t", index=False)

    cov = (
        merged.groupby(["project_id", "cancer"], dropna=False)
        .agg(
            rna_samples=("sample_id", "nunique"),
            rna_sample_type_ids=("tcga_sample_type_id", "nunique"),
            os_overlap_samples=("has_os_survival", "sum"),
            os_events=("survival_event", lambda x: int(pd.to_numeric(x, errors="coerce").fillna(0).sum())),
            patients=("tcga_patient_id", "nunique"),
        )
        .reset_index()
        .sort_values("project_id")
    )
    cov.to_csv(outdir / "tables/tcga_scp682_prediction_project_coverage.tsv", sep="\t", index=False)
    logger.info(
        "覆盖审计：RNA样本=%d；RNA∩OS=%d；OS事件=%d；癌种=%d",
        merged["sample_id"].nunique(),
        int(merged["has_os_survival"].sum()),
        int(merged.loc[merged["has_os_survival"], "survival_event"].sum()),
        merged["project_id"].nunique(),
    )
    return merged


def read_one_tpm(file_path: Path, genes: list[str]) -> np.ndarray:
    df = pd.read_csv(
        file_path,
        sep="\t",
        comment="#",
        usecols=["gene_name", "tpm_unstranded"],
        dtype={"gene_name": "string", "tpm_unstranded": "float32"},
    )
    df = df[df["gene_name"].notna() & df["tpm_unstranded"].notna()]
    values = df.groupby("gene_name", sort=False)["tpm_unstranded"].median()
    return values.reindex(genes).to_numpy(dtype=np.float32)


def build_rna_matrix(
    expr: pd.DataFrame,
    genes: list[str],
    out_path: Path,
    max_samples: int | None,
    force: bool,
    workers: int,
    logger: logging.Logger,
) -> Path:
    selected = expr.copy()
    if max_samples is not None:
        selected = selected.head(max_samples).copy()
    if out_path.exists() and not force:
        logger.info("复用已有 RNA 矩阵: %s", out_path)
        return out_path

    logger.info("开始拼接 GDC STAR TPM 矩阵：样本=%d；模型基因=%d", len(selected), len(genes))
    t0 = time.time()
    matrix = np.empty((len(selected), len(genes)), dtype=np.float32)
    bad_rows: list[dict[str, str]] = []
    sample_ids = selected["sample_id"].astype(str).tolist()
    file_paths = selected["rna_file"].astype(str).tolist()

    def task(idx: int, sample_id: str, file_path: str) -> tuple[int, np.ndarray, dict[str, str] | None]:
        try:
            return idx, read_one_tpm(Path(file_path), genes), None
        except Exception as exc:
            return idx, np.full(len(genes), np.nan, dtype=np.float32), {"sample_id": sample_id, "rna_file": file_path, "error": repr(exc)}

    done = 0
    if workers <= 1:
        for i, (sample_id, file_path) in enumerate(zip(sample_ids, file_paths)):
            idx, arr, err = task(i, sample_id, file_path)
            matrix[idx, :] = arr
            if err is not None:
                bad_rows.append(err)
            done += 1
            if done % 250 == 0 or done == len(selected):
                logger.info("RNA矩阵进度 %d/%d", done, len(selected))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(task, i, sample_id, file_path) for i, (sample_id, file_path) in enumerate(zip(sample_ids, file_paths))]
            for fut in as_completed(futures):
                idx, arr, err = fut.result()
                matrix[idx, :] = arr
                if err is not None:
                    bad_rows.append(err)
                done += 1
                if done % 250 == 0 or done == len(selected):
                    logger.info("RNA矩阵进度 %d/%d", done, len(selected))

    rna = pd.DataFrame(matrix, index=pd.Index(sample_ids, name="sample_id"), columns=genes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rna.to_parquet(out_path)
    if bad_rows:
        pd.DataFrame(bad_rows).to_csv(out_path.parent / "tcga_gdc_star_tpm_read_failures.tsv", sep="\t", index=False)
    logger.info("RNA矩阵写出: %s；形状=%s；耗时=%.1fs", out_path, tuple(rna.shape), time.time() - t0)
    return out_path


def run_prediction(
    portable_dir: Path,
    rna_path: Path,
    outdir: Path,
    device: str,
    v4_batch_size: int,
    graph_batch_size: int,
    knn: int,
    temperature: float,
    write_attention: bool,
    logger: logging.Logger,
) -> None:
    entrypoint = portable_dir / "predict_scp682.py"
    cmd = [
        sys.executable,
        str(entrypoint),
        "--rna",
        str(rna_path),
        "--rna-scale",
        "tpm",
        "--outdir",
        str(outdir),
        "--device",
        device,
        "--v4-batch-size",
        str(v4_batch_size),
        "--graph-batch-size",
        str(graph_batch_size),
        "--knn",
        str(knn),
        "--temperature",
        str(temperature),
    ]
    if write_attention:
        cmd.append("--write-attention")
    (outdir / "logs/prediction_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    logger.info("运行 SCP682 主模型: %s", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(portable_dir) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, cwd=str(portable_dir), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (outdir / "logs/predict_scp682_stdout.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"SCP682 主模型推理失败，返回码={proc.returncode}，日志={outdir / 'logs/predict_scp682_stdout.log'}")

    src = outdir / "predictions/scp682_main_phosphosite.parquet"
    alias = outdir / "predictions/tcga_full_scp682_predicted_phosphosite.parquet"
    if not src.exists():
        raise RuntimeError(f"主模型输出不存在: {src}")
    shutil.copy2(src, alias)
    logger.info("主模型预测别名写出: %s", alias)


def write_summary(outdir: Path, portable_dir: Path, rna_path: Path, expr: pd.DataFrame, merged: pd.DataFrame, prediction_done: bool) -> None:
    pred = outdir / "predictions/tcga_full_scp682_predicted_phosphosite.parquet"
    pred_shape = None
    if pred.exists():
        pf = pd.read_parquet(pred)
        pred_shape = list(pf.shape)
    summary = {
        "portable_dir": str(portable_dir),
        "rna_source": "GDC STAR gene quantification, tpm_unstranded",
        "rna_mapping": str(GDC_MAPPING),
        "rna_dir": str(GDC_RNASEQ_DIR),
        "survival_source": str(SURVIVAL),
        "toil_tpm_reference": str(TOIL_TPM),
        "toil_counts_reference": str(TOIL_COUNTS),
        "rna_matrix": str(rna_path),
        "rna_samples": int(expr["sample_id"].nunique()),
        "projects": int(expr["project_id"].nunique()),
        "os_overlap_samples": int(merged["has_os_survival"].sum()),
        "os_events": int(merged.loc[merged["has_os_survival"], "survival_event"].fillna(0).sum()),
        "sample_barcode_rule": "预测样本保留完整 TCGA aliquot/sample 条形码；生存匹配使用 TCGA-XX-YYYY-01 这种 sample-type 级条形码。",
        "prediction_done": bool(prediction_done and pred.exists()),
        "prediction_path": str(pred) if pred.exists() else "",
        "prediction_shape": pred_shape,
    }
    (outdir / "reports/tcga_scp682_reprediction_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="从 /data/gongke 原始 TCGA RNA 重新运行 SCP682 主模型预测")
    ap.add_argument("--portable-dir", default=str(PORTABLE_DEFAULT))
    ap.add_argument("--output-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--max-samples", type=int, default=None, help="小批量测试用；不填则全样本")
    ap.add_argument("--force-rna-matrix", action="store_true")
    ap.add_argument("--prepare-only", action="store_true", help="只生成覆盖审计和 RNA 矩阵，不运行主模型")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--v4-batch-size", type=int, default=32)
    ap.add_argument("--graph-batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--knn", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.08)
    ap.add_argument("--write-attention", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    portable_dir = Path(args.portable_dir).resolve()
    outdir = Path(args.output_dir).resolve()
    ensure_dirs(outdir)
    logger = setup_logger(outdir)

    logger.info("SCP682_PORTABLE=%s", portable_dir)
    package_audit = audit_portable_package(portable_dir)
    package_audit.to_csv(outdir / "tables/tcga_full_scp682_portable_package_audit.tsv", sep="\t", index=False)
    missing = package_audit[~package_audit["exists"]]
    blocker_path = outdir / "tables/tcga_full_scp682_reprediction_blocker_audit.tsv"
    if len(missing):
        blocker = missing.assign(blocking_function="package_audit")
        blocker.to_csv(blocker_path, sep="\t", index=False)
        raise SystemExit("SCP682_PORTABLE 缺少必要文件，已写出 blocker 审计。")
    pd.DataFrame(columns=["item", "path", "exists", "use", "blocking_function"]).to_csv(blocker_path, sep="\t", index=False)

    genes = read_model_genes(portable_dir)
    expr = load_expression_manifest(logger)
    surv = load_survival()
    merged = write_coverage_tables(expr, surv, outdir, logger)
    tag = "full" if args.max_samples is None else f"n{args.max_samples}"
    rna_path = outdir / f"intermediate/tcga_gdc_star_tpm_model_genes_{tag}.parquet"
    rna_path = build_rna_matrix(expr, genes, rna_path, args.max_samples, args.force_rna_matrix, args.workers, logger)

    prediction_done = False
    if not args.prepare_only:
        run_prediction(
            portable_dir=portable_dir,
            rna_path=rna_path,
            outdir=outdir,
            device=args.device,
            v4_batch_size=args.v4_batch_size,
            graph_batch_size=args.graph_batch_size,
            knn=args.knn,
            temperature=args.temperature,
            write_attention=args.write_attention,
            logger=logger,
        )
        prediction_done = True

    write_summary(outdir, portable_dir, rna_path, expr, merged, prediction_done)
    logger.info("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
