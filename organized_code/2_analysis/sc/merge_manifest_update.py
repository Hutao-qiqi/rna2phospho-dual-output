from pathlib import Path

root = Path(r"D:\lsy\01_data\shared\metadata")
manifest = root / "dataset_manifest.tsv"
update = root / "dataset_manifest_update.tsv"

rows = {}
header = None

for path in (manifest, update):
    with path.open("r", encoding="utf-8", newline="") as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    if not lines:
        continue
    file_header = lines[0].split("\t")
    if header is None:
        header = file_header
    if file_header != header:
        raise RuntimeError(f"header mismatch in {path}")
    for line in lines[1:]:
        parts = line.split("\t")
        rows[parts[0]] = parts

if "hao_pbmc_citeseq" in rows:
    rows["hao_pbmc_citeseq"][2] = "raw_downloaded"
    rows["hao_pbmc_citeseq"][-1] = rows["hao_pbmc_citeseq"][-1] + " Local file: D:\\lsy\\01_data\\single_cell\\raw\\hao_pbmc_citeseq\\pbmc_multimodal.h5seurat."

order = [
    "phospho_seq_blair_2025",
    "vivo_seq_th17_2025",
    "iccite_seq_tcell_2025",
    "hao_pbmc_citeseq",
    "bendall_2011_cytof",
    "levine_2015_aml_cytof",
    "covid_immune_cytof",
]

with manifest.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write("\t".join(header) + "\n")
    for dataset_id in order:
        if dataset_id in rows:
            handle.write("\t".join(rows[dataset_id]) + "\n")
    for dataset_id in sorted(set(rows) - set(order)):
        handle.write("\t".join(rows[dataset_id]) + "\n")

print(f"merged {len(rows)} manifest rows into {manifest}")
