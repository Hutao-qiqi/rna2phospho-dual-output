#!/usr/bin/env python3
"""Build focused GSEA pathway auxiliary labels from RNA expression."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path("/data/lsy/Infinite_Stream")

DEFAULT_REGEX = (
    r"RTK|RECEPTOR_TYROSINE_KINASE|EGFR|ERBB|HER2|MET|PDGF|FGF|FGFR|KIT|AXL|"
    r"MAPK|ERK|JNK|P38|RAS|RAF|MEK|PI3K|AKT|MTOR|PTEN|JAK|STAT|SRC|FAK|"
    r"TGF|WNT|NOTCH|HIPPO|P53|MYC|E2F|G2M|CELL_CYCLE|CHECKPOINT|"
    r"DNA_REPAIR|DAMAGE|APOPTOSIS|HYPOXIA|EMT|INFLAMMATORY|INTERFERON|TNFA|IL6"
)


def read_symbol_matrix(path: Path) -> pd.DataFrame:
    df = pq.read_table(path).to_pandas()
    gene_col = df.columns[0]
    df = df.set_index(gene_col)
    df.index = df.index.astype(str)
    x = df.T
    x.index = x.index.astype(str)
    return x.apply(pd.to_numeric, errors="coerce")


def parse_gmt(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            genes = sorted({g.strip() for g in parts[2:] if g.strip()})
            if name and genes:
                rows.append({"pathway": name, "source_file": path.name, "genes": genes})
    return rows


def select_gene_sets(gmt_paths: list[Path], available_genes: set[str], pattern: str, min_size: int, max_size: int) -> pd.DataFrame:
    rx = re.compile(pattern, flags=re.IGNORECASE)
    records = []
    seen = set()
    for gmt in gmt_paths:
        for row in parse_gmt(gmt):
            name = row["pathway"]
            is_hallmark = name.startswith("HALLMARK_")
            is_focused = bool(rx.search(name))
            if not (is_hallmark or is_focused):
                continue
            genes = sorted(set(row["genes"]) & available_genes)
            if len(genes) < min_size or len(genes) > max_size:
                continue
            if name in seen:
                continue
            seen.add(name)
            records.append(
                {
                    "pathway": name,
                    "source_file": row["source_file"],
                    "selection_group": "hallmark" if is_hallmark else "focused_canonical",
                    "n_genes": len(genes),
                    "genes": ",".join(genes),
                }
            )
    out = pd.DataFrame(records).sort_values(["selection_group", "pathway"]).reset_index(drop=True)
    return out


def score_pathways(x: pd.DataFrame, sets: pd.DataFrame) -> pd.DataFrame:
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1).replace(0, np.nan)
    z = (x - mu) / sd
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rows = []
    for _, row in sets.iterrows():
        genes = [g for g in str(row["genes"]).split(",") if g in z.columns]
        if not genes:
            continue
        vals = z[genes].mean(axis=1)
        vals = (vals - vals.mean()) / vals.std(ddof=1)
        rows.append(pd.Series(vals, name=row["pathway"]))
    scores = pd.DataFrame(rows)
    scores.index.name = "pathway"
    return scores.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--x-symbols", type=Path, default=Path("data/processed/X_all.symbols.parquet"))
    parser.add_argument(
        "--gmt",
        type=Path,
        nargs="+",
        default=[
            Path("resources/msigdb/h.all.v2025.1.Hs.symbols.gmt"),
            Path("resources/msigdb/c2.cp.v2025.1.Hs.symbols.gmt"),
        ],
    )
    parser.add_argument("--pattern", default=DEFAULT_REGEX)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--max-size", type=int, default=300)
    parser.add_argument("--out-labels", type=Path, default=Path("data/processed/Y_pathway_aux_gsea_relevant.parquet"))
    parser.add_argument("--out-sets", type=Path, default=Path("01_data/pathway_prior/processed/gsea_auxiliary_pathway_sets_v1.tsv"))
    parser.add_argument("--out-log", type=Path, default=Path("02_results/model_validation/20260426_gsea_auxiliary_labels/logs/prepare_gsea_pathway_aux_labels.json"))
    args = parser.parse_args()

    root = args.project_root.resolve()
    x = read_symbol_matrix(root / args.x_symbols)
    sets = select_gene_sets(
        [root / p for p in args.gmt],
        available_genes=set(x.columns.astype(str)),
        pattern=args.pattern,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    scores = score_pathways(x, sets)

    (root / args.out_labels).parent.mkdir(parents=True, exist_ok=True)
    (root / args.out_sets).parent.mkdir(parents=True, exist_ok=True)
    (root / args.out_log).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(scores), root / args.out_labels)
    sets.to_csv(root / args.out_sets, sep="\t", index=False)

    log = {
        "scope": "GSEA pathway auxiliary labels for phosphorylation-oriented model training.",
        "x_symbols": str(root / args.x_symbols),
        "gmt": [str(root / p) for p in args.gmt],
        "out_labels": str(root / args.out_labels),
        "out_sets": str(root / args.out_sets),
        "n_samples": int(x.shape[0]),
        "n_genes": int(x.shape[1]),
        "n_pathways": int(scores.shape[0]),
        "selection_group_counts": sets["selection_group"].value_counts().to_dict() if not sets.empty else {},
        "caveat": "Pathway labels are RNA-derived auxiliary targets, not independent validation labels.",
    }
    (root / args.out_log).write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
