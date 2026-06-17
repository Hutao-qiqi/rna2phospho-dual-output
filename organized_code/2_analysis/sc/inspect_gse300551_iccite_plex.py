import gzip
import json
import tarfile
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
RAW_DIR = ROOT / r"01_data\single_cell\raw\external_single_cell_phospho_validation_v1\icCITE-plex_GSE300551"
OUT_DIR = ROOT / r"02_results\single_cell\20260518_gse300551_iccite_plex_inspect"


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text_preview(path, n=4000):
    try:
        if str(path).endswith(".gz"):
            with gzip.open(path, "rt", errors="replace") as handle:
                return handle.read(n)
        return path.read_text(encoding="utf-8", errors="replace")[:n]
    except Exception as exc:
        return f"READ_FAILED: {exc}"


def gzip_member_text(tar, member_name, n=None):
    with tar.extractfile(member_name) as raw:
        with gzip.GzipFile(fileobj=raw) as gz:
            if n is None:
                return gz.read().decode("utf-8", errors="replace")
            return gz.read(n).decode("utf-8", errors="replace")


def mtx_shape_from_member(tar, member_name):
    text = gzip_member_text(tar, member_name, 4096)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        if len(parts) >= 3 and all(p.lstrip("-").isdigit() for p in parts[:3]):
            return int(parts[0]), int(parts[1]), int(parts[2])
    return None, None, None


def count_gzip_lines_in_member(tar, member_name):
    with tar.extractfile(member_name) as raw:
        with gzip.GzipFile(fileobj=raw) as gz:
            return sum(1 for _ in gz)


def read_gzip_lines_in_member(tar, member_name, max_lines=60):
    text = gzip_member_text(tar, member_name)
    return text.splitlines()[:max_lines]


def inspect_10x_h5_from_tar(tar, member, temp_dir):
    out_path = Path(temp_dir) / Path(member.name).name
    with tar.extractfile(member) as src, out_path.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    try:
        import h5py
    except Exception as exc:
        return {"file": member.name, "inspect_error": f"h5py_not_available: {exc}", "bytes": member.size}
    out = {"file": member.name, "bytes": member.size}
    with h5py.File(out_path, "r") as h5:
        def visit(name, obj):
            pass
        if "matrix" in h5:
            mat = h5["matrix"]
            if "shape" in mat:
                out["shape"] = [int(x) for x in mat["shape"][()]]
            if "barcodes" in mat:
                out["n_barcodes"] = int(len(mat["barcodes"]))
            if "features" in mat:
                feat = mat["features"]
                if "name" in feat:
                    names = [x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x) for x in feat["name"][:40]]
                    out["feature_name_preview"] = names
                if "feature_type" in feat:
                    types = [x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x) for x in feat["feature_type"][:]]
                    out["feature_type_counts"] = dict(pd.Series(types).value_counts().to_dict())
        else:
            out["root_keys"] = list(h5.keys())
    return out


def main():
    ensure_dir(OUT_DIR / "tables")
    ensure_dir(OUT_DIR / "reports")
    raw_tar = RAW_DIR / "GSE300551_RAW.tar"
    soft_path = RAW_DIR / "GSE300551_family.soft.gz"
    xml_path = RAW_DIR / "GSE300551_family.xml.tgz"
    sra_path = RAW_DIR / "PRJNA1346700_SraRunInfo.csv"
    geo_page = RAW_DIR / "GSE300551_geo_page.html"

    previews = {
        "soft_preview": read_text_preview(soft_path, 8000),
        "sra_preview": read_text_preview(sra_path, 2000),
        "geo_page_preview": read_text_preview(geo_page, 2000),
    }
    try:
        with tarfile.open(xml_path, "r:gz") as xtar:
            names = xtar.getnames()
            previews["xml_members"] = names
            if names:
                previews["xml_preview"] = xtar.extractfile(names[0]).read(8000).decode("utf-8", errors="replace")
    except Exception as exc:
        previews["xml_error"] = str(exc)

    with tarfile.open(raw_tar, "r") as tar:
        members = tar.getmembers()
        member_rows = []
        for m in members:
            name = m.name
            if name.endswith("_filtered_feature_bc_matrix.h5"):
                assay = "rna_h5"
            elif name.endswith("_TSB_count_filtered_feature_bc_matrix.h5"):
                assay = "tsb_h5"
            elif "_ADT_" in name:
                assay = "adt"
            elif "_HTO_" in name:
                assay = "hto"
            elif "_Celltags_" in name:
                assay = "celltag"
            else:
                assay = "other"
            sample = ""
            for token in ("D9", "E9", "F9", "G9", "H9"):
                if f"_{token}" in name or name.endswith(f"{token}_filtered_feature_bc_matrix.h5"):
                    sample = token
                    break
            species = ""
            if "_Human_" in name:
                species = "Human"
            elif "_Mouse_" in name:
                species = "Mouse"
            member_rows.append({"member": name, "bytes": int(m.size), "assay": assay, "sample": sample, "species": species})
        member_df = pd.DataFrame(member_rows)

        matrix_rows = []
        feature_rows = []
        feature_keywords = ("phosph", "p-", "perk", "pakt", "psyk", "pstat", "p38", "plc", "btk", "kinase")
        for m in members:
            name = m.name
            if name.endswith(".mtx.gz"):
                rows, cols, nnz = mtx_shape_from_member(tar, name)
                matrix_rows.append({"member": name, "rows": rows, "cols": cols, "nnz": nnz})
            if name.endswith(".genes.txt.gz"):
                lines = read_gzip_lines_in_member(tar, name, 500)
                for i, line in enumerate(lines):
                    text = line.strip()
                    low = text.lower()
                    feature_rows.append(
                        {
                            "member": name,
                            "feature_order": i,
                            "feature": text,
                            "is_phospho_like": any(k in low for k in feature_keywords),
                        }
                    )
        matrix_df = pd.DataFrame(matrix_rows)
        feature_df = pd.DataFrame(feature_rows)

        h5_members = [m for m in members if m.name.endswith("_filtered_feature_bc_matrix.h5")]
        h5_inspect = []
        with tempfile.TemporaryDirectory() as tmp:
            for m in h5_members:
                h5_inspect.append(inspect_10x_h5_from_tar(tar, m, tmp))

    member_df.to_csv(OUT_DIR / "tables" / "gse300551_raw_members.tsv", sep="\t", index=False)
    matrix_df.to_csv(OUT_DIR / "tables" / "gse300551_mtx_shapes.tsv", sep="\t", index=False)
    feature_df.to_csv(OUT_DIR / "tables" / "gse300551_feature_previews.tsv", sep="\t", index=False)
    pd.DataFrame(h5_inspect).to_csv(OUT_DIR / "tables" / "gse300551_h5_inspect.tsv", sep="\t", index=False)
    (OUT_DIR / "reports" / "gse300551_previews.json").write_text(json.dumps(previews, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "raw_tar": str(raw_tar),
        "raw_tar_bytes": raw_tar.stat().st_size,
        "n_members": int(len(member_df)),
        "assay_counts": member_df["assay"].value_counts().to_dict(),
        "samples": sorted([x for x in member_df["sample"].dropna().unique().tolist() if x]),
        "h5_inspect": h5_inspect,
        "phospho_like_feature_count_preview": int(feature_df["is_phospho_like"].sum()) if len(feature_df) else 0,
    }
    (OUT_DIR / "reports" / "gse300551_inspect_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if len(feature_df):
        print("\nPHOSPHO_LIKE_FEATURES_PREVIEW")
        print(feature_df[feature_df["is_phospho_like"]].head(80).to_string(index=False), flush=True)
    print("\nMATRIX_SHAPES")
    print(matrix_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
