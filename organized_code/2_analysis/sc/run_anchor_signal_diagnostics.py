from __future__ import annotations

from pathlib import Path
import gzip
import csv
import math
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.stats import spearmanr


ROOT = Path(r"D:\lsy")
PAIRED = ROOT / "01_data" / "single_cell" / "intermediate" / "paired_matrices"
OUT = ROOT / "02_results" / "single_cell" / "20260510_anchor_signal_diagnostics"
for sub in ["tables", "reports"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)


ANCHORS = {
    "RPS6_S235_S236": {
        "genes": ["RPS6"],
        "datasets": {
            "phospho_seq_blair_2025_phospho_multi": "pRPS6",
            "iccite_seq_tcell_2025": "intra-B1126-RPS6Pho",
        },
        "pathway": "MTOR_SIGNALING",
    },
    "STAT3_Y705": {
        "genes": ["STAT3"],
        "datasets": {
            "vivo_seq_th17_2025": "STAT3_P40763_Y705",
            "iccite_seq_tcell_2025": "intra-phospho-Stat3-Tyr705-D3A7",
        },
        "pathway": "JAK_STAT_SIGNALING",
    },
    "MAPK1_MAPK3_T202_Y204": {
        "genes": ["MAPK1", "MAPK3"],
        "mouse_genes": ["Mapk1", "Mapk3"],
        "datasets": {
            "vivo_seq_th17_2025": "MAPK1_P28482_T202_Y204__MAPK3_P27361_T185_Y187",
            "iccite_seq_tcell_2025": "intra-HM0025-phospho-p44-42-197G2",
        },
        "pathway": "ERK_MAPK_SIGNALING",
    },
}


GENE_SETS = {
    "MTOR_SIGNALING": [
        "RPS6", "RPS6KB1", "RPS6KB2", "EIF4EBP1", "EIF4E", "MTOR", "RPTOR", "RICTOR",
        "AKT1", "AKT2", "TSC1", "TSC2", "RHEB", "RRAGA", "RRAGB", "RRAGC", "RRAGD",
        "MLST8", "DEPTOR", "ULK1", "ULK2", "SLC7A5", "SLC3A2", "LAMTOR1", "LAMTOR2",
        "LAMTOR3", "LAMTOR4", "LAMTOR5", "EIF4B", "PDCD4",
    ],
    "JAK_STAT_SIGNALING": [
        "STAT1", "STAT2", "STAT3", "STAT4", "STAT5A", "STAT5B", "STAT6", "JAK1",
        "JAK2", "JAK3", "TYK2", "IL2RA", "IL2RB", "IL2RG", "IL6R", "IL6ST", "IL7R",
        "SOCS1", "SOCS2", "SOCS3", "CISH", "PIAS1", "PIAS3", "IRF1", "BCL3", "MYC",
        "BCL2L1", "PIM1", "IFNGR1", "IFNGR2",
    ],
    "ERK_MAPK_SIGNALING": [
        "MAPK1", "MAPK3", "MAP2K1", "MAP2K2", "RAF1", "BRAF", "ARAF", "KRAS", "NRAS",
        "HRAS", "DUSP1", "DUSP2", "DUSP4", "DUSP5", "DUSP6", "FOS", "FOSB", "JUN",
        "JUNB", "JUND", "ELK1", "ELK3", "ETV1", "ETV4", "ETV5", "MYC", "SPRY1",
        "SPRY2", "SPRY4", "RSK1", "RPS6KA1", "RPS6KA2", "RPS6KA3", "RPS6KA4",
    ],
}

MOUSE_MAP = {
    "RPS6": "Rps6", "RPS6KB1": "Rps6kb1", "RPS6KB2": "Rps6kb2", "EIF4EBP1": "Eif4ebp1",
    "EIF4E": "Eif4e", "MTOR": "Mtor", "RPTOR": "Rptor", "RICTOR": "Rictor", "AKT1": "Akt1",
    "AKT2": "Akt2", "TSC1": "Tsc1", "TSC2": "Tsc2", "RHEB": "Rheb", "ULK1": "Ulk1",
    "ULK2": "Ulk2", "STAT1": "Stat1", "STAT2": "Stat2", "STAT3": "Stat3", "STAT4": "Stat4",
    "STAT5A": "Stat5a", "STAT5B": "Stat5b", "STAT6": "Stat6", "JAK1": "Jak1", "JAK2": "Jak2",
    "JAK3": "Jak3", "TYK2": "Tyk2", "IL2RA": "Il2ra", "IL2RB": "Il2rb", "IL2RG": "Il2rg",
    "IL6R": "Il6r", "IL6ST": "Il6st", "IL7R": "Il7r", "SOCS1": "Socs1", "SOCS2": "Socs2",
    "SOCS3": "Socs3", "CISH": "Cish", "IRF1": "Irf1", "MYC": "Myc", "BCL2L1": "Bcl2l1",
    "PIM1": "Pim1", "MAPK1": "Mapk1", "MAPK3": "Mapk3", "MAP2K1": "Map2k1", "MAP2K2": "Map2k2",
    "RAF1": "Raf1", "BRAF": "Braf", "ARAF": "Araf", "KRAS": "Kras", "NRAS": "Nras",
    "HRAS": "Hras", "DUSP1": "Dusp1", "DUSP2": "Dusp2", "DUSP4": "Dusp4", "DUSP5": "Dusp5",
    "DUSP6": "Dusp6", "FOS": "Fos", "FOSB": "Fosb", "JUN": "Jun", "JUNB": "Junb",
    "JUND": "Jund", "ELK1": "Elk1", "MYC": "Myc", "SPRY1": "Spry1", "SPRY2": "Spry2",
    "SPRY4": "Spry4", "RPS6KA1": "Rps6ka1", "RPS6KA2": "Rps6ka2", "RPS6KA3": "Rps6ka3",
    "RPS6KA4": "Rps6ka4",
}


def corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return np.nan, int(mask.sum())
    return float(spearmanr(x[mask], y[mask]).statistic), int(mask.sum())


def clr_matrix(mat):
    x = mat.astype(float)
    x.data = np.log1p(x.data)
    if sparse.issparse(x):
        dense_mean = np.asarray(x.mean(axis=0)).ravel()
        return x, dense_mean
    return np.log1p(x), np.log1p(x).mean(axis=0)


def zscore(v):
    v = np.asarray(v, dtype=float)
    sd = np.nanstd(v)
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(v)
    return (v - np.nanmean(v)) / sd


def load_mtx_bundle(mtx_path, features_path, barcodes_path):
    mat = mmread(mtx_path).tocsr()
    features = pd.read_csv(features_path, sep="\t", header=None)[0].astype(str).tolist()
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None)[0].astype(str).tolist()
    return mat, features, barcodes


def get_rows(mat, features, genes):
    idx = [features.index(g) for g in genes if g in features]
    if not idx:
        return None, []
    if mat.shape[0] == len(features):
        sub = mat[idx, :]
        vals = np.asarray(sub.mean(axis=0)).ravel()
    elif mat.shape[1] == len(features):
        sub = mat[:, idx]
        vals = np.asarray(sub.mean(axis=1)).ravel()
    else:
        raise ValueError(f"matrix shape {mat.shape} does not match feature length {len(features)}")
    return vals, [features[i] for i in idx]


def module_score(mat, features, genes):
    x, used = get_rows(mat, features, genes)
    if x is None:
        return None, []
    return np.log1p(x), used


def load_vivo():
    d = PAIRED / "vivo_seq_th17_2025"
    rna = mmread(d / "rna_counts.mtx").tocsr()
    genes = pd.read_csv(d / "genes.tsv", sep="\t", header=None)[0].astype(str).tolist()
    meta = pd.read_csv(d / "cell_metadata.tsv", sep="\t")
    phos = pd.read_csv(d / "phospho_counts.tsv", sep="\t", index_col=0)
    return {"rna": rna, "genes": genes, "meta": meta, "phos": phos}


def load_blair():
    d = PAIRED / "phospho_seq_blair_2025_phospho_multi"
    adt = pd.read_csv(d / "adt_counts.tsv", sep="\t", index_col=0)
    meta = pd.read_csv(d / "cell_metadata.tsv", sep="\t")
    raw = ROOT / "01_data" / "single_cell" / "raw" / "phospho_seq_blair_2025" / "extracted" / "GSM8726041_Phospho-Multi_RNA.csv.gz"
    wanted = sorted(set(["RPS6"] + GENE_SETS["MTOR_SIGNALING"]))
    rows = {}
    with gzip.open(raw, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        barcodes = header[1:]
        for row in reader:
            if row and row[0] in wanted:
                rows[row[0]] = np.asarray(row[1:], dtype=float)
    return {"rna_rows": rows, "barcodes": barcodes, "meta": meta, "adt": adt}


def load_iccite():
    d = PAIRED / "iccite_seq_tcell_2025"
    rna, genes, barcodes = load_mtx_bundle(
        d / "rna_full_counts" / "rna_full_counts.mtx",
        d / "rna_full_counts" / "rna_full_counts_features.tsv",
        d / "rna_full_counts" / "rna_full_counts_barcodes.tsv",
    )
    corrected = d / "iccite_background_corrected_strict"
    phos, phos_features, phos_barcodes = load_mtx_bundle(
        corrected / "phospho_counts_control_mean_subtracted.mtx",
        corrected / "phospho_counts_control_mean_subtracted_features.tsv",
        corrected / "phospho_counts_control_mean_subtracted_barcodes.tsv",
    )
    raw_phos, raw_features, _ = load_mtx_bundle(
        d / "phospho_counts" / "phospho_counts.mtx",
        d / "phospho_counts" / "phospho_counts_features.tsv",
        d / "phospho_counts" / "phospho_counts_barcodes.tsv",
    )
    meta = pd.read_csv(d / "cell_metadata.tsv", sep="\t")
    return {
        "rna": rna,
        "genes": genes,
        "barcodes": barcodes,
        "phos": phos,
        "phos_features": phos_features,
        "raw_phos": raw_phos,
        "raw_phos_features": raw_features,
        "meta": meta,
    }


def phospho_vector_iccite(data, feature, normalized):
    idx = data["phos_features"].index(feature)
    raw = np.asarray(data["phos"].getrow(idx).toarray()).ravel()
    if normalized == "corrected_counts":
        return raw
    if normalized == "corrected_log1p":
        return np.log1p(raw)
    if normalized == "corrected_clr":
        mat = data["phos"].astype(float).copy()
        mat.data = np.log1p(mat.data)
        col_mean = np.asarray(mat.mean(axis=0)).ravel()
        return np.log1p(raw) - col_mean
    raise ValueError(normalized)


def phospho_vector_vivo(data, feature, normalized):
    raw = pd.to_numeric(data["phos"][feature], errors="coerce").to_numpy(float)
    if normalized == "corrected_counts":
        return raw
    if normalized == "corrected_log1p":
        return np.log1p(raw)
    if normalized == "corrected_clr":
        panel_cols = [c for c in data["phos"].columns if c != "cell_id"]
        arr = data["phos"].apply(pd.to_numeric, errors="coerce")
        logged = np.log1p(arr)
        return (logged[feature] - logged.mean(axis=1, skipna=True)).to_numpy(float)
    raise ValueError(normalized)


def phospho_vector_blair(data, feature, normalized):
    raw = pd.to_numeric(data["adt"][feature], errors="coerce").to_numpy(float)
    if normalized == "corrected_counts":
        return raw
    if normalized == "corrected_log1p":
        return np.log1p(raw)
    if normalized == "corrected_clr":
        logged = np.log1p(data["adt"].apply(pd.to_numeric, errors="coerce"))
        return (logged[feature] - logged.mean(axis=1)).to_numpy(float)
    raise ValueError(normalized)


def run():
    vivo = load_vivo()
    blair = load_blair()
    icc = load_iccite()
    datasets = {
        "vivo_seq_th17_2025": vivo,
        "phospho_seq_blair_2025_phospho_multi": blair,
        "iccite_seq_tcell_2025": icc,
    }
    rows = []
    variance_rows = []
    strat_rows = []
    normalizations = ["corrected_counts", "corrected_log1p", "corrected_clr"]

    # isotype-vs-phospho comparison for icCITE
    bg = pd.read_csv(PAIRED / "iccite_seq_tcell_2025" / "iccite_background_corrected_strict" / "cell_control_background_summary.tsv", sep="\t")
    bg_vec = bg["control_mean"].to_numpy(float)

    for anchor, cfg in ANCHORS.items():
        for ds, phos_feature in cfg["datasets"].items():
            data = datasets[ds]
            if ds == "vivo_seq_th17_2025":
                genes = cfg.get("mouse_genes", [MOUSE_MAP.get(g, g.title()) for g in cfg["genes"]])
                self_rna, used_genes = get_rows(data["rna"], data["genes"], genes)
                path_genes = [MOUSE_MAP.get(g, g.title()) for g in GENE_SETS[cfg["pathway"]]]
                pathway, used_path = module_score(data["rna"], data["genes"], path_genes)
                phos_raw = phospho_vector_vivo(data, phos_feature, "corrected_counts")
                meta = data["meta"]
                groupings = ["culture", "panel", "stim", "culture_panel_stim"]
            elif ds == "phospho_seq_blair_2025_phospho_multi":
                self_vals = [blair["rna_rows"][g] for g in cfg["genes"] if g in blair["rna_rows"]]
                self_rna = np.mean(self_vals, axis=0) if self_vals else None
                used_genes = [g for g in cfg["genes"] if g in blair["rna_rows"]]
                path_vals = [blair["rna_rows"][g] for g in GENE_SETS[cfg["pathway"]] if g in blair["rna_rows"]]
                pathway = np.log1p(np.mean(path_vals, axis=0)) if path_vals else None
                used_path = [g for g in GENE_SETS[cfg["pathway"]] if g in blair["rna_rows"]]
                phos_raw = phospho_vector_blair(data, phos_feature, "corrected_counts")
                meta = data["meta"]
                groupings = ["assay_subset", "cell_type_label"]
            else:
                self_rna, used_genes = get_rows(data["rna"], data["genes"], cfg["genes"])
                pathway, used_path = module_score(data["rna"], data["genes"], GENE_SETS[cfg["pathway"]])
                phos_raw = phospho_vector_iccite(data, phos_feature, "corrected_counts")
                meta = data["meta"]
                groupings = ["Donor", "orig.ident"]
                iso_r, iso_n = corr(bg_vec, phos_raw)
                rows.append({
                    "analysis": "isotype_vs_phospho",
                    "anchor": anchor, "dataset_id": ds, "normalization": "corrected_counts",
                    "feature": phos_feature, "rna_or_pathway": "isotype_control_mean",
                    "n": iso_n, "spearman": iso_r, "used_genes": "intra-B0092-mIgG2b",
                })

            for norm in normalizations:
                if ds == "vivo_seq_th17_2025":
                    phos = phospho_vector_vivo(data, phos_feature, norm)
                elif ds == "phospho_seq_blair_2025_phospho_multi":
                    phos = phospho_vector_blair(data, phos_feature, norm)
                else:
                    phos = phospho_vector_iccite(data, phos_feature, norm)
                variance_rows.append({
                    "anchor": anchor, "dataset_id": ds, "normalization": norm,
                    "phospho_mean": float(np.nanmean(phos)),
                    "phospho_var": float(np.nanvar(phos)),
                    "phospho_nonzero_rate": float(np.nanmean(np.nan_to_num(phos) > 0)),
                })
                if self_rna is not None:
                    r, n = corr(np.log1p(self_rna), phos)
                    rows.append({
                        "analysis": "self_rna",
                        "anchor": anchor, "dataset_id": ds, "normalization": norm,
                        "feature": phos_feature, "rna_or_pathway": "+".join(cfg["genes"]),
                        "n": n, "spearman": r, "used_genes": ";".join(used_genes),
                    })
                if pathway is not None:
                    r, n = corr(pathway, phos)
                    rows.append({
                        "analysis": "pathway_score",
                        "anchor": anchor, "dataset_id": ds, "normalization": norm,
                        "feature": phos_feature, "rna_or_pathway": cfg["pathway"],
                        "n": n, "spearman": r, "used_genes": ";".join(used_path),
                    })

                if self_rna is not None and norm == "corrected_log1p":
                    x = np.log1p(self_rna)
                    for grouping in groupings:
                        if grouping not in meta.columns:
                            continue
                        for level, idx in meta.groupby(grouping).indices.items():
                            if len(idx) < 30:
                                continue
                            idx = np.asarray(list(idx), dtype=int)
                            r, n = corr(x[idx], phos[idx])
                            strat_rows.append({
                                "anchor": anchor, "dataset_id": ds, "grouping": grouping,
                                "level": str(level), "n": n, "spearman_self_rna": r,
                                "normalization": norm,
                            })

    diag = pd.DataFrame(rows)
    diag.to_csv(OUT / "tables" / "anchor_self_rna_pathway_spearman.tsv", sep="\t", index=False)
    pd.DataFrame(variance_rows).to_csv(OUT / "tables" / "phospho_variance_by_normalization.tsv", sep="\t", index=False)
    pd.DataFrame(strat_rows).to_csv(OUT / "tables" / "cell_group_stratified_self_rna_spearman.tsv", sep="\t", index=False)

    main = diag[(diag["normalization"] == "corrected_log1p") & (diag["analysis"].isin(["self_rna", "pathway_score"]))]
    go_self = main[(main["analysis"] == "self_rna") & (main["spearman"] > 0.10)]["anchor"].nunique()
    go_path = main[(main["analysis"] == "pathway_score") & (main["spearman"] > 0.20)]["anchor"].nunique()
    best_strat = pd.DataFrame(strat_rows)["spearman_self_rna"].max() if strat_rows else np.nan
    verdict = "GO" if (go_self >= 2 and go_path >= 2 and best_strat > 0.30) else "WEAK_OR_NO_GO"
    lines = [
        "# Anchor 信号诊断",
        "",
        f"判定：{verdict}",
        "",
        f"self-RNA spearman > 0.10 的 anchor 数：{go_self}",
        f"pathway score spearman > 0.20 的 anchor 数：{go_path}",
        f"最佳分层 self-RNA spearman：{best_strat}",
        "",
        "主表使用 corrected_log1p；corrected_counts 和 corrected_clr 也已输出。",
    ]
    (OUT / "reports" / "anchor_signal_diagnostics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(main.sort_values(["analysis", "anchor", "dataset_id"]).to_string(index=False))


if __name__ == "__main__":
    run()
