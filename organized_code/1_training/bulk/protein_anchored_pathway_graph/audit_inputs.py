from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .data import (
        build_pathway_tensors,
        discover_input_paths,
        load_site_kinases,
        select_pathways,
    )
    from .train import read_inputs
except ImportError:
    from data import (  # type: ignore
        build_pathway_tensors,
        discover_input_paths,
        load_site_kinases,
        select_pathways,
    )
    from train import read_inputs  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit SCP682 protein-anchored pathway-graph inputs")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--max-pathways", type=int, default=72)
    args = parser.parse_args()

    paths = discover_input_paths(args.project_root.resolve())
    inputs = read_inputs(paths)
    rna = inputs["rna"]
    protein = inputs["protein"]
    phosphosite = inputs["phosphosite"]
    targets = inputs["targets"]
    parent_genes = inputs["parent_genes"]
    protein_genes = inputs["protein_genes"]

    variance = np.nanvar(rna.to_numpy(dtype=np.float32), axis=0)
    variance_order = rna.columns[np.argsort(-variance)].astype(str).tolist()
    pathways = select_pathways(
        paths.hallmark_gmt,
        paths.c2_gmt,
        rna.columns.astype(str),
        parent_genes,
        variance_order,
        max_pathways=args.max_pathways,
    )
    site_kinases = load_site_kinases(paths.kinase_edges, targets)
    tensors = build_pathway_tensors(
        pathways,
        rna.columns.astype(str),
        protein_genes,
        targets,
        parent_genes,
        site_kinases,
    )

    non_global = tensors.site_pathway_mask.copy()
    non_global &= tensors.site_pathway_index != 0
    report = {
        "n_samples": int(rna.shape[0]),
        "n_rna_genes": int(rna.shape[1]),
        "n_total_proteins": int(protein.shape[1]),
        "n_phosphosites": int(phosphosite.shape[1]),
        "n_pathways": len(tensors.names),
        "n_kinases": len(tensors.kinase_names),
        "n_sites_with_parent_protein": int(tensors.parent_protein_mask.sum()),
        "n_sites_with_kinase_annotation": int(tensors.site_kinase_mask.any(axis=1).sum()),
        "n_sites_with_non_global_pathway": int(non_global.any(axis=1).sum()),
        "paths": {name: str(value) if value is not None else None for name, value in vars(paths).items()},
    }
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
