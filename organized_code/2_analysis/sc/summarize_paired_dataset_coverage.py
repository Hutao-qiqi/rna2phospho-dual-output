import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


DATASETS = {
    "phospho_seq_blair_2025_phospho_multi": {
        "kind": "blair",
        "rna_matrix": "rna_counts.tsv",
        "feature_table": "adt_counts.tsv",
        "feature_kind": "adt",
    },
    "vivo_seq_th17_2025": {
        "kind": "vivo",
        "rna_matrix": "rna_counts.mtx",
        "feature_table": "phospho_counts.tsv",
        "feature_kind": "phospho",
    },
    "iccite_seq_tcell_2025": {
        "kind": "iccite",
        "rna_matrix": "rna_full_counts/rna_full_counts.mtx",
        "feature_table": "phospho_counts/phospho_counts.mtx",
        "feature_kind": "phospho",
    },
    "qurie_seq_bjab_2021": {
        "kind": "qurie",
        "rna_matrix": "rna_counts.mtx.gz",
        "feature_table": "phospho_counts.tsv.gz",
        "feature_kind": "phospho",
    },
}


FALLBACK_TARGETS = {
    "pRPS6": ("RPS6_pSitePending", "RPS6", "pending_antibody_clone"),
    "p-S6": ("RPS6_pSitePending", "RPS6", "pending_antibody_clone"),
    "p-Erk1/2": ("MAPK1_MAPK3_pSitePending", "MAPK1;MAPK3", "pending_antibody_clone"),
    "p-STAT3": ("STAT3_pSitePending", "STAT3", "pending_antibody_clone"),
    "p-STAT1": ("STAT1_pSitePending", "STAT1", "pending_antibody_clone"),
    "p-p38": ("MAPK14_pSitePending", "MAPK14", "pending_antibody_clone"),
    "p-Akt": ("AKT_pSitePending", "AKT1;AKT2;AKT3", "pending_antibody_clone"),
    "p-Btk": ("BTK_pSitePending", "BTK", "pending_antibody_clone"),
    "p-Syk": ("SYK_pSitePending", "SYK", "pending_antibody_clone"),
    "p-PLC-y2": ("PLCG2_pSitePending", "PLCG2", "pending_antibody_clone"),
    "p-PLC-y2Y759": ("PLCG2_Y759", "PLCG2", "Y759"),
}


def count_lines(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def read_lines(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in handle]


def mtx_shape_fast(path: Path) -> tuple[int, int, int]:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8", errors="replace") as handle:
        first = handle.readline()
        if not first.startswith("%%MatrixMarket"):
            raise ValueError(f"Not a MatrixMarket file: {path}")
        for line in handle:
            if line.startswith("%"):
                continue
            rows, cols, nnz = [int(x) for x in line.strip().split()[:3]]
            return rows, cols, nnz
    raise ValueError(f"Could not read MatrixMarket shape: {path}")


def dense_tsv_shape(path: Path) -> tuple[int, int]:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    with opener(path, mode, encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        rows = sum(1 for _ in handle)
    return rows, max(0, len(header) - 1)


def load_schema(root: Path) -> dict[str, dict[str, str]]:
    path = root / "01_data" / "shared" / "metadata" / "phospho_target_schema.tsv"
    if not path.exists():
        return {}
    schema = pd.read_csv(path, sep="\t")
    out = {}
    for _, row in schema.iterrows():
        info = {
            "target_id": str(row["target_id"]),
            "protein_symbol": str(row["protein_symbol"]),
            "residue": str(row["residue"]),
            "canonical_label": str(row["canonical_label"]),
        }
        out[str(row["source_label"])] = info
        out[str(row["canonical_label"])] = info
    return out


def normalize_feature(feature: str, schema: dict[str, dict[str, str]]) -> dict[str, str]:
    if feature in schema:
        return schema[feature]
    if feature in FALLBACK_TARGETS:
        target_id, protein_symbol, residue = FALLBACK_TARGETS[feature]
        return {
            "target_id": target_id,
            "protein_symbol": protein_symbol,
            "residue": residue,
            "canonical_label": f"{protein_symbol}_{residue}",
        }
    return {
        "target_id": feature,
        "protein_symbol": feature,
        "residue": "unknown",
        "canonical_label": feature,
    }


def is_blair_phospho(feature: str) -> bool:
    return feature.startswith("p") or "phospho" in feature.lower()


def stats_from_dense_table(path: Path, dataset_id: str, schema: dict[str, dict[str, str]], feature_filter=None) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    if feature_filter is not None:
        keep = [c for c in df.columns if feature_filter(c)]
        df = df.loc[:, keep]

    rows = []
    for feature in df.columns:
        values = df[feature]
        info = normalize_feature(str(feature), schema)
        non_missing = int(values.notna().sum())
        nonzero = int((values.fillna(0) != 0).sum())
        rows.append(
            {
                "dataset_id": dataset_id,
                "feature_name": feature,
                **info,
                "n_cells": int(df.shape[0]),
                "n_non_missing": non_missing,
                "n_nonzero": nonzero,
                "non_missing_rate": non_missing / max(1, int(df.shape[0])),
                "nonzero_rate": nonzero / max(1, int(df.shape[0])),
            }
        )
    summary = {
        "n_feature_cells": int(df.shape[0]),
        "n_features": int(df.shape[1]),
        "feature_total_values": int(df.shape[0] * df.shape[1]),
        "feature_non_missing": int(df.notna().sum().sum()),
        "feature_nonzero": int((df.fillna(0) != 0).sum().sum()),
    }
    return pd.DataFrame(rows), summary


def stats_from_mtx(path: Path, features_path: Path, barcodes_path: Path, dataset_id: str, schema: dict[str, dict[str, str]]) -> tuple[pd.DataFrame, dict]:
    features = read_lines(features_path)
    barcodes = read_lines(barcodes_path)
    matrix = scipy.io.mmread(path).tocsc()
    if matrix.shape[0] == len(features):
        matrix = matrix.T.tocsc()
    if matrix.shape[1] != len(features):
        raise ValueError(f"Feature count mismatch for {dataset_id}: {matrix.shape}, {len(features)}")
    rows = []
    for j, feature in enumerate(features):
        col = matrix.getcol(j)
        info = normalize_feature(feature, schema)
        rows.append(
            {
                "dataset_id": dataset_id,
                "feature_name": feature,
                **info,
                "n_cells": len(barcodes),
                "n_non_missing": len(barcodes),
                "n_nonzero": int(col.nnz),
                "non_missing_rate": 1.0,
                "nonzero_rate": int(col.nnz) / max(1, len(barcodes)),
            }
        )
    summary = {
        "n_feature_cells": len(barcodes),
        "n_features": len(features),
        "feature_total_values": len(barcodes) * len(features),
        "feature_non_missing": len(barcodes) * len(features),
        "feature_nonzero": int(matrix.nnz),
    }
    return pd.DataFrame(rows), summary


def raw_inventory(root: Path) -> pd.DataFrame:
    raw_roots = [
        root / "01_data" / "single_cell" / "raw",
        root / "01_data" / "shared" / "raw",
    ]
    rows = []
    for base in raw_roots:
        if not base.exists():
            continue
        for dataset_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
            files = [p for p in dataset_dir.rglob("*") if p.is_file()]
            rows.append(
                {
                    "modality": base.parent.name,
                    "dataset_id": dataset_dir.name,
                    "n_files": len(files),
                    "total_bytes": int(sum(p.stat().st_size for p in files)),
                    "path": str(dataset_dir),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    parser.add_argument("--out-name", default="20260511_data_processing_inventory")
    args = parser.parse_args()

    root = Path(args.root)
    paired = root / "01_data" / "single_cell" / "intermediate" / "paired_matrices"
    out = root / "02_results" / "single_cell" / args.out_name
    tables = out / "tables"
    reports = out / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    schema = load_schema(root)
    coverage_rows = []
    feature_tables = []

    for dataset_id, spec in DATASETS.items():
        dataset_dir = paired / dataset_id
        if not dataset_dir.exists():
            continue

        if dataset_id == "iccite_seq_tcell_2025":
            n_cells = count_lines(dataset_dir / "rna_full_counts" / "rna_full_counts_barcodes.tsv")
            n_genes = count_lines(dataset_dir / "rna_full_counts" / "rna_full_counts_features.tsv")
            rna_rows, rna_cols, rna_nnz = mtx_shape_fast(dataset_dir / spec["rna_matrix"])
            feature_df, feature_summary = stats_from_mtx(
                dataset_dir / spec["feature_table"],
                dataset_dir / "phospho_counts" / "phospho_counts_features.tsv",
                dataset_dir / "phospho_counts" / "phospho_counts_barcodes.tsv",
                dataset_id,
                schema,
            )
        else:
            n_cells = count_lines(dataset_dir / "barcodes.tsv") if (dataset_dir / "barcodes.tsv").exists() else 0
            n_genes = count_lines(dataset_dir / "genes.tsv") if (dataset_dir / "genes.tsv").exists() else 0
            rna_path = dataset_dir / spec["rna_matrix"]
            if rna_path.suffix in {".mtx", ".gz"} and ".mtx" in rna_path.name:
                rna_rows, rna_cols, rna_nnz = mtx_shape_fast(rna_path)
            else:
                rna_rows, rna_cols = dense_tsv_shape(rna_path)
                rna_nnz = np.nan

            feature_path = dataset_dir / spec["feature_table"]
            feature_filter = is_blair_phospho if dataset_id == "phospho_seq_blair_2025_phospho_multi" else None
            feature_df, feature_summary = stats_from_dense_table(feature_path, dataset_id, schema, feature_filter)

        feature_tables.append(feature_df)
        total = max(1, feature_summary["feature_total_values"])
        coverage_rows.append(
            {
                "dataset_id": dataset_id,
                "n_cells": int(n_cells),
                "n_rna_genes": int(n_genes),
                "rna_matrix_rows": int(rna_rows),
                "rna_matrix_cols": int(rna_cols),
                "rna_nnz": rna_nnz,
                "n_phospho_features": int(feature_summary["n_features"]),
                "phospho_total_values": int(feature_summary["feature_total_values"]),
                "phospho_non_missing": int(feature_summary["feature_non_missing"]),
                "phospho_nonzero": int(feature_summary["feature_nonzero"]),
                "phospho_non_missing_rate": feature_summary["feature_non_missing"] / total,
                "phospho_nonzero_rate": feature_summary["feature_nonzero"] / total,
            }
        )

    coverage = pd.DataFrame(coverage_rows).sort_values("dataset_id")
    features = pd.concat(feature_tables, ignore_index=True) if feature_tables else pd.DataFrame()
    target_coverage = (
        features.groupby(["target_id", "protein_symbol", "residue"], dropna=False)
        .agg(
            n_datasets=("dataset_id", "nunique"),
            datasets=("dataset_id", lambda x: ";".join(sorted(set(map(str, x))))),
            features=("feature_name", lambda x: ";".join(sorted(set(map(str, x))))),
            total_nonzero_cells=("n_nonzero", "sum"),
        )
        .reset_index()
        .sort_values(["n_datasets", "target_id"], ascending=[False, True])
    )

    coverage.to_csv(tables / "dataset_matrix_coverage.tsv", sep="\t", index=False)
    features.to_csv(tables / "phospho_feature_long.tsv", sep="\t", index=False)
    target_coverage.to_csv(tables / "phospho_target_dataset_coverage.tsv", sep="\t", index=False)
    raw_inventory(root).to_csv(tables / "raw_dataset_inventory.tsv", sep="\t", index=False)

    manifest = {
        "project_root": str(root),
        "output_dir": str(out),
        "datasets": coverage.to_dict(orient="records"),
        "n_phospho_feature_rows": int(features.shape[0]),
        "n_targets": int(target_coverage.shape[0]),
    }
    (reports / "data_processing_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_lines = [
        "# 数据处理记录",
        "",
        f"项目目录：`{root}`",
        f"结果目录：`{out}`",
        "",
        "已生成四个同细胞矩阵的数据覆盖表。QuRIE-seq 已转成标准中间矩阵；iPhosChip 当前是质谱结果文件，不进入 RNA 配对训练矩阵。",
        "",
        "跨数据集可直接比较的目标仍然以 RPS6、STAT3、MAPK/ERK、p38、RELA、JUN 这类抗体层级重叠为主。QuRIE 给 BCR 通路补了 B 细胞动力学场景，但多数残基需要后续查抗体说明书才能定死。",
    ]
    (reports / "data_processing_summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(coverage.to_string(index=False))
    print()
    print(target_coverage.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
