import argparse
import json
import pickle
from pathlib import Path

import anndata as ad
import h5py
import pandas as pd


def load_vocab_ids(vocab_h5: Path) -> set[str]:
    with h5py.File(vocab_h5, "r") as handle:
        return {x.decode() if isinstance(x, bytes) else str(x) for x in handle["keys"][()]}


def map_symbol(symbol: str, mapping: dict, vocab_ids: set[str]) -> str:
    s = str(symbol)
    candidates = [s, s.upper(), s.replace("-", "_"), s.upper().replace("-", "_")]
    for key in candidates:
        value = mapping.get(key)
        if value in vocab_ids:
            return str(value)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\data\lsy\vm_lsy_parent\lsy")
    parser.add_argument(
        "--input-dir",
        default=r"01_data\single_cell\intermediate\foundation_model_h5ad_inputs_v1",
    )
    parser.add_argument(
        "--output-dir",
        default=r"01_data\single_cell\intermediate\foundation_model_h5ad_inputs_transcriptformer_geneid_v1",
    )
    parser.add_argument(
        "--mapping-pkl",
        default=r"D:\data\lsy\models\geneformer_v1_10m_repo\geneformer\ensembl_mapping_dict_gc104M.pkl",
    )
    parser.add_argument(
        "--tf-vocab-h5",
        default=r"D:\data\lsy\models\transcriptformer\tf_sapiens\vocabs\homo_sapiens_gene.h5",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--assay", default="10x 3' transcription profiling")
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    src_dir = root / args.input_dir
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.mapping_pkl, "rb") as fh:
        mapping = pickle.load(fh)
    vocab_ids = load_vocab_ids(Path(args.tf_vocab_h5))
    manifest = pd.read_csv(src_dir / "foundation_h5ad_manifest.tsv", sep="\t")
    wanted = set(args.datasets) if args.datasets else None
    rows = []
    for row in manifest.to_dict("records"):
        dataset_id = str(row["dataset_id"])
        if wanted is not None and dataset_id not in wanted:
            continue
        src = Path(str(row["h5ad"]))
        dst = out_dir / src.name
        if args.skip_existing and dst.exists():
            obj = ad.read_h5ad(dst, backed="r")
            mapped = int((obj.var["ensembl_id"].astype(str) != "").sum()) if "ensembl_id" in obj.var else 0
            has_assay = "assay" in obj.obs
            obj.file.close()
            if mapped > 0 and has_assay:
                rows.append({**row, "h5ad": str(dst), "status": "skipped_existing", "n_mapped_ensembl": mapped})
                continue

        print(f"prepare {dataset_id}: {src}", flush=True)
        obj = ad.read_h5ad(src)
        obj.obs["assay"] = args.assay
        symbols = obj.var["gene_symbol"].astype(str).tolist() if "gene_symbol" in obj.var else obj.var_names.astype(str).tolist()
        mapped_ids = [map_symbol(x, mapping, vocab_ids) for x in symbols]
        obj.var["ensembl_id"] = mapped_ids
        mapped_count = int(sum(bool(x) for x in mapped_ids))
        obj.uns["transcriptformer_gene_id_mapping"] = {
            "mapping_pkl": str(Path(args.mapping_pkl)),
            "tf_vocab_h5": str(Path(args.tf_vocab_h5)),
            "n_mapped_ensembl": mapped_count,
            "n_genes": int(obj.n_vars),
        }
        obj.write_h5ad(dst)
        rows.append({**row, "h5ad": str(dst), "status": "completed", "n_mapped_ensembl": mapped_count})
        print(f"done {dataset_id}: mapped={mapped_count}/{obj.n_vars}", flush=True)

    out_manifest = pd.DataFrame(rows)
    out_manifest.to_csv(out_dir / "foundation_h5ad_manifest.tsv", sep="\t", index=False)
    (out_dir / "transcriptformer_h5ad_manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
