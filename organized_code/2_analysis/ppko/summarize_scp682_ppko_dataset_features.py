import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
INPUT = ROOT / r"01_data\single_cell\intermediate\phospho_perturb\decryptm_p100_joint_v1"
EXTERNAL_RAW = ROOT / r"01_data\single_cell\raw\external_bulk_phospho_validation_v1"
EXTERNAL_RESULT = ROOT / r"02_results\single_cell\20260519_scp682_ppko_1_external_bulk_validation_v51_kstar_sparse"
OUT = ROOT / r"02_results\single_cell\20260519_scp682_ppko_dataset_feature_audit"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_unique(series):
    if series is None:
        return []
    values = []
    for value in series.dropna().astype(str):
        if value and value.lower() != "nan":
            values.append(value)
    return sorted(set(values))


def top_values(series, n=12):
    if series is None or len(series) == 0:
        return ""
    vc = series.fillna("").astype(str)
    vc = vc[vc.ne("")]
    out = []
    for key, value in vc.value_counts().head(n).items():
        out.append(f"{key}:{int(value)}")
    return ";".join(out)


def range_text(values):
    vals = pd.to_numeric(values, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return ""
    return f"{float(vals.min()):.4g}-{float(vals.max()):.4g}"


def summarize_main_training():
    x = np.load(INPUT / "phospho_values.npy", mmap_mode="r")
    mask = np.load(INPUT / "target_mask.npy", mmap_mode="r").astype(bool)
    meta = pd.read_csv(INPUT / "cell_metadata.tsv", sep="\t")
    cond = pd.read_csv(INPUT / "condition_table.tsv", sep="\t")
    targets = pd.read_csv(INPUT / "phospho_target_table.tsv", sep="\t")
    manifest = pd.read_csv(INPUT / "pathway_target_manifest.tsv", sep="\t")
    priors = pd.read_csv(INPUT / "condition_pathway_prior.tsv", sep="\t")

    dataset_rows = []
    for dataset, sub_meta in meta.groupby("dataset_id"):
        sub_cond = cond[cond["dataset_id"].astype(str).eq(str(dataset))]
        row_idx = sub_meta.index.to_numpy(dtype=np.int64)
        sub_mask = np.asarray(mask[row_idx], dtype=bool)
        union_sites = int(sub_mask.any(axis=0).sum())
        finite_fraction = float(sub_mask.mean()) if sub_mask.size else float("nan")
        noncontrol = sub_cond[~sub_cond["perturbation_type"].astype(str).str.lower().isin(["control"])]
        target_gene_tokens = []
        for text in noncontrol.get("target_genes", pd.Series(dtype=str)).fillna("").astype(str):
            for token in text.replace("/", ";").replace(",", ";").split(";"):
                token = token.strip()
                if token and token.lower() != "nan":
                    target_gene_tokens.append(token)
        dataset_rows.append(
            {
                "dataset_id": dataset,
                "n_sample_rows": int(len(sub_meta)),
                "n_conditions": int(len(sub_cond)),
                "n_control_conditions": int(sub_cond["perturbation_type"].astype(str).str.lower().eq("control").sum()) if len(sub_cond) else 0,
                "n_noncontrol_conditions": int(len(noncontrol)),
                "n_cell_types": int(sub_cond["cell_type"].nunique()) if "cell_type" in sub_cond else 0,
                "cell_types": ";".join(safe_unique(sub_cond.get("cell_type", pd.Series(dtype=str)))[:30]),
                "n_perturbations_noncontrol": int(noncontrol["perturbation"].nunique()) if len(noncontrol) else 0,
                "top_perturbations": top_values(noncontrol.get("perturbation", pd.Series(dtype=str))),
                "perturbation_type_counts": top_values(sub_cond.get("perturbation_type", pd.Series(dtype=str))),
                "n_target_genes_noncontrol": int(len(set(target_gene_tokens))),
                "top_target_genes": ";".join(pd.Series(target_gene_tokens).value_counts().head(20).index.tolist()) if target_gene_tokens else "",
                "time_hours_range": range_text(sub_cond.get("time_hours", pd.Series(dtype=float))),
                "n_time_values": int(pd.to_numeric(sub_cond.get("time_hours", pd.Series(dtype=float)), errors="coerce").dropna().nunique()),
                "dose_molar_range_nonzero": range_text(pd.to_numeric(noncontrol.get("dose_molar", pd.Series(dtype=float)), errors="coerce").replace(0, np.nan)),
                "n_dose_values_nonzero": int(pd.to_numeric(noncontrol.get("dose_molar", pd.Series(dtype=float)), errors="coerce").replace(0, np.nan).dropna().nunique()),
                "observed_site_union": union_sites,
                "finite_fraction": finite_fraction,
            }
        )

    cell_rows = []
    for (dataset, cell), sub in cond.groupby(["dataset_id", "cell_type"], dropna=False):
        noncontrol = sub[~sub["perturbation_type"].astype(str).str.lower().isin(["control"])]
        cell_rows.append(
            {
                "dataset_id": dataset,
                "cell_type": cell,
                "n_conditions": int(len(sub)),
                "n_noncontrol_conditions": int(len(noncontrol)),
                "n_perturbations_noncontrol": int(noncontrol["perturbation"].nunique()) if len(noncontrol) else 0,
                "perturbations": ";".join(safe_unique(noncontrol.get("perturbation", pd.Series(dtype=str)))[:20]),
                "target_genes": ";".join(safe_unique(noncontrol.get("target_genes", pd.Series(dtype=str)))[:20]),
                "time_hours_range": range_text(sub.get("time_hours", pd.Series(dtype=float))),
                "dose_molar_range_nonzero": range_text(pd.to_numeric(noncontrol.get("dose_molar", pd.Series(dtype=float)), errors="coerce").replace(0, np.nan)),
            }
        )

    source_rows = []
    for source, sub in targets.groupby("source_datasets"):
        source_rows.append(
            {
                "source_datasets": source,
                "n_targets": int(len(sub)),
                "n_molecules": int(sub["molecule"].replace("", np.nan).dropna().nunique()),
                "pathway_counts": top_values(sub["primary_pathway"], n=20),
            }
        )

    overview = {
        "n_sample_rows": int(x.shape[0]),
        "n_sites": int(x.shape[1]),
        "finite_fraction": float(mask.mean()),
        "n_conditions": int(len(cond)),
        "n_datasets": int(cond["dataset_id"].nunique()),
        "n_targets": int(len(targets)),
        "n_pathways": int(manifest["pathway"].nunique()),
        "n_condition_pathway_priors": int(len(priors)),
    }

    return overview, pd.DataFrame(dataset_rows), pd.DataFrame(cell_rows), pd.DataFrame(source_rows)


def folder_size(path):
    total = 0
    n = 0
    if not Path(path).exists():
        return 0, 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            n += 1
            total += f.stat().st_size
    return n, total


def summarize_external():
    rows = []
    if EXTERNAL_RAW.exists():
        for folder in sorted(EXTERNAL_RAW.iterdir()):
            if not folder.is_dir():
                continue
            n_files, total = folder_size(folder)
            ext_counts = {}
            for f in folder.rglob("*"):
                if f.is_file():
                    ext = f.suffix.lower() or "no_ext"
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
            rows.append(
                {
                    "dataset": folder.name,
                    "n_files": int(n_files),
                    "size_gb": round(total / 1024**3, 4),
                    "extension_counts": ";".join(f"{k}:{v}" for k, v in sorted(ext_counts.items(), key=lambda x: -x[1])[:12]),
                }
            )
    raw_df = pd.DataFrame(rows)

    report_rows = []
    report_dir = EXTERNAL_RESULT / "reports"
    if report_dir.exists():
        for path in sorted(report_dir.glob("*_validation_summary.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            report_rows.append(
                {
                    "report": path.name,
                    "analysis": data.get("analysis", ""),
                    "n_external_observed_rows": data.get("n_external_observed_rows", ""),
                    "n_external_model_overlap_rows": data.get("n_external_model_overlap_rows", ""),
                    "baseline_source": data.get("baseline_source", ""),
                    "action_type": data.get("action_type", ""),
                    "mean_site_cosine": data.get("mean_site_cosine", data.get("mean_site_cosine_all", "")),
                    "mean_spearman": data.get("mean_spearman", data.get("mean_spearman_all", "")),
                    "mean_direction_accuracy": data.get("mean_direction_accuracy", data.get("mean_direction_accuracy_all", "")),
                    "mean_site_cosine_regulated": data.get("mean_site_cosine_regulated", ""),
                }
            )
    return raw_df, pd.DataFrame(report_rows)


def main():
    for sub in ("tables", "reports", "logs"):
        ensure_dir(OUT / sub)
    overview, dataset_df, cell_df, source_df = summarize_main_training()
    raw_df, report_df = summarize_external()

    dataset_df.to_csv(OUT / "tables" / "main_training_dataset_summary.tsv", sep="\t", index=False)
    cell_df.to_csv(OUT / "tables" / "main_training_cell_context_summary.tsv", sep="\t", index=False)
    source_df.to_csv(OUT / "tables" / "main_training_target_source_summary.tsv", sep="\t", index=False)
    raw_df.to_csv(OUT / "tables" / "external_raw_dataset_inventory.tsv", sep="\t", index=False)
    report_df.to_csv(OUT / "tables" / "external_validation_summary.tsv", sep="\t", index=False)
    with (OUT / "reports" / "dataset_feature_overview.json").open("w", encoding="utf-8") as handle:
        json.dump(overview, handle, indent=2)
    print(json.dumps(overview, indent=2))
    print("\nmain_training_dataset_summary")
    print(dataset_df.to_string(index=False))
    print("\nexternal_validation_summary")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
