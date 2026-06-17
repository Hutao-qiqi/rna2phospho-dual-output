#!/usr/bin/env python3
"""Compute EGFR / pEGFR-like (EGFRPY1068) deltas from an external pairing manifest.

This script:
- reads a pairing manifest (baseline vs perturbed sample titles + expression matrix path)
- extracts the minimal set of genes required by the specified protein models
- applies the same feature scaling saved in models/<protein>/model.pkl
- predicts protein values for baseline/perturbed, and computes deltas.

It is designed for GEO supplementary processed matrices (TSV/CSV, optionally .gz)
where rows are genes and columns are sample titles.

Outputs:
  reports/external_validation/external_cellline_predicted_deltas.tsv

Notes:
- Many external matrices are TPM or counts; --transform=auto tries to pick a sane transform.
- Missing genes are filled with 0.0 (consistent with existing fill_tcpas_missing.py behavior).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def open_text(path: Path):
    return (
        gzip.open(path, "rt", encoding="utf-8", errors="replace")
        if str(path).lower().endswith(".gz")
        else open(path, "rt", encoding="utf-8", errors="replace")
    )


def guess_delimiter(line: str) -> str:
    tabs = line.count("\t")
    commas = line.count(",")
    if tabs == 0 and commas == 0:
        return "whitespace"
    return "\t" if tabs >= commas else ","


def strip_quotes(s: str) -> str:
    s2 = str(s)
    if len(s2) >= 2 and ((s2[0] == '"' and s2[-1] == '"') or (s2[0] == "'" and s2[-1] == "'")):
        return s2[1:-1]
    return s2


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def strip_ensembl_version(gene_id: str) -> str:
    return re.sub(r"\.[0-9]+$", "", str(gene_id).strip())


def looks_like_ensembl(gene_id: str) -> bool:
    return bool(re.match(r"^ENSG[0-9]{6,}", str(gene_id).strip(), flags=re.I))


def ensure_ensembl_symbol_map(ensembl_ids: list[str], cache_tsv: Path) -> dict[str, str]:
    """Build/load an Ensembl->HGNC symbol mapping.

    Uses an R helper script backed by org.Hs.eg.db (offline Bioconductor annotation).
    """

    cache_tsv.parent.mkdir(parents=True, exist_ok=True)

    if not cache_tsv.exists() or cache_tsv.stat().st_size == 0:
        ids_txt = cache_tsv.with_suffix(cache_tsv.suffix + ".ids.txt")
        ids = [strip_ensembl_version(x) for x in ensembl_ids]
        ids = [x for x in ids if x]
        ids = sorted(set(ids))
        ids_txt.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

        r_script = Path("scripts/external_validation/map_ensembl_to_symbol.R")
        if not r_script.exists():
            raise FileNotFoundError(f"Missing R mapping helper: {r_script}")

        cmd = [
            "Rscript",
            str(r_script),
            "--in",
            str(ids_txt),
            "--out",
            str(cache_tsv),
        ]
        subprocess.run(cmd, check=True)

    # Load mapping (keep first symbol per ensembl in the cache file).
    try:
        mp = pd.read_csv(cache_tsv, sep="\t", dtype=str).fillna("")
    except Exception as e:
        raise RuntimeError(f"Failed to read Ensembl map TSV: {cache_tsv}: {e}")

    out: dict[str, str] = {}
    if not mp.empty and "ENSEMBL" in mp.columns and "SYMBOL" in mp.columns:
        for _, r in mp.iterrows():
            ens = strip_ensembl_version(r["ENSEMBL"])
            sym = str(r["SYMBOL"]).strip()
            if ens and sym and ens not in out:
                out[ens] = sym
    return out


def iter_matrix_rows(matrix_path: Path, delim: str):
    """Yield parsed rows (as lists of strings) after skipping the header line."""
    with open_text(matrix_path) as fh:
        _ = fh.readline()  # header
        if delim == "whitespace":
            for line in fh:
                if line.strip():
                    yield re.split(r"\s+", line.strip())
        else:
            reader = csv.reader(fh, delimiter=delim)
            for row in reader:
                if row:
                    yield row


def extract_rep_from_title(title: str) -> int | None:
    m = re.search(r"\brep\s*([0-9]+)\b", str(title), flags=re.I)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def guess_condition_letter(title: str) -> str | None:
    t = str(title).lower()
    # Common control-like
    if any(k in t for k in ["vehicle", "control", "untreated", "dmso"]):
        return "V"
    # Common treated-like / states
    if "persister" in t:
        return "P"
    # Common EGFR-TKIs
    if "osimertinib" in t:
        return "O"
    if "gefitinib" in t:
        return "G"
    if "erlotinib" in t:
        return "E"
    if "afatinib" in t:
        return "A"
    return None


def load_model_pack(models_dir: Path, protein: str) -> dict:
    pdir = models_dir / protein
    if not (pdir / "model.pkl").exists():
        raise FileNotFoundError(f"Missing model.pkl for {protein}: {pdir}")
    pack = joblib.load(pdir / "model.pkl")
    if not isinstance(pack, dict) or "model" not in pack:
        raise ValueError(f"Unexpected model pack format for {protein}: {pdir / 'model.pkl'}")
    if "features" not in pack:
        raise ValueError(f"Model pack missing features list for {protein}: {pdir / 'model.pkl'}")
    return pack


def choose_transform_auto(values: np.ndarray) -> str:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return "none"
    # Heuristic: log-scale expression is typically ~0-15; counts/TPM can be 50-1e5.
    q95 = float(np.quantile(v, 0.95))
    vmax = float(np.max(v))
    if vmax > 500 or q95 > 50:
        return "log2p1"
    return "none"


def apply_transform(x: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "none":
        return x
    if mode == "log2p1":
        arr = x.to_numpy(dtype=float)
        # only apply to non-negative; keep negatives as-is
        m = np.isfinite(arr) & (arr >= 0)
        arr2 = arr.copy()
        arr2[m] = np.log2(arr2[m] + 1.0)
        return pd.DataFrame(arr2, index=x.index, columns=x.columns)
    raise ValueError(f"Unknown transform: {mode}")


def read_gene_matrix_subset(
    matrix_path: Path,
    wanted_titles: list[str],
    wanted_genes_upper: set[str],
) -> tuple[pd.DataFrame, dict[str, str], str]:
    """Read a gene x sample matrix, returning sample x gene dataframe for needed titles/genes.

    Returns:
      X_raw: DataFrame indexed by actual matrix column names (samples), columns are UPPERCASE gene symbols.
      title_to_col: mapping from requested title -> actual column name
      delim: inferred delimiter
    """

    with open_text(matrix_path) as fh:
        first = fh.readline()
        if not first:
            raise RuntimeError(f"Empty matrix: {matrix_path}")
        delim = guess_delimiter(first)

        if delim == "whitespace":
            header = re.split(r"\s+", first.strip())
            row_iter = (re.split(r"\s+", line.strip()) for line in fh)
        else:
            reader = csv.reader(fh, delimiter=delim)
            header = [strip_quotes(c).strip() for c in (first.rstrip("\n").split(delim))]
            # If we guessed delimiter from the first line, re-parse the remainder with csv for robustness.
            row_iter = reader

        if not header or len(header) < 2:
            raise RuntimeError(f"Matrix has no header / too few columns: {matrix_path}")

        header = [strip_quotes(c).strip() for c in header]

        # Detect which column contains gene symbols.
        header_lower = [h.lower() for h in header]
        gene_idx = 0
        # IMPORTANT: prefer explicit gene symbol columns over generic "gene".
        for key in ("genesymbol", "gene_symbol", "symbol", "genename", "gene_name", "gene"):
            if key in header_lower:
                gene_idx = header_lower.index(key)
                break

        # Assume sample columns are to the right of the gene column, but skip any
        # extra annotation columns (e.g., locus/chr/start/end) that sometimes appear
        # between the gene symbol and the first sample.
        anno_after_gene = {
            "locus",
            "chr",
            "chrom",
            "chromosome",
            "start",
            "end",
            "strand",
            "entrezid",
            "entrez",
            "geneid",
            "gene_id",
            "ensembl",
            "ensemblid",
            "ensembl_gene_id",
            "genetype",
            "gene_type",
        }
        sample_start = gene_idx + 1
        while sample_start < len(header_lower) and header_lower[sample_start] in anno_after_gene:
            sample_start += 1

        sample_cols = header[sample_start:]
        sample_offset = sample_start
        if not sample_cols:
            return (
                pd.DataFrame(index=pd.Index([], name="sample"), columns=sorted(wanted_genes_upper), dtype=float),
                {},
                delim,
            )

        col_set = set(sample_cols)
        col_norm = {norm_key(c): c for c in sample_cols if c}

        title_to_col: dict[str, str] = {}
        for t in wanted_titles:
            t = str(t).strip()
            if not t:
                continue
            if t in col_set:
                title_to_col[t] = t
                continue
            nk = norm_key(t)
            if nk in col_norm:
                title_to_col[t] = col_norm[nk]
                continue

            # Common GEO pattern: title like "HCC827_DMSO1" but matrix uses
            # suffix letters like "Hcc827_DMSO_A".
            m = re.match(r"^([A-Za-z0-9]+)_([A-Za-z]+)([0-9]+)$", t)
            if m:
                cell, cond, rep_s = m.group(1), m.group(2), m.group(3)
                try:
                    rep_n = int(rep_s)
                except Exception:
                    rep_n = None
                if rep_n is not None and 1 <= rep_n <= 26:
                    rep_letter = chr(ord("A") + rep_n - 1)
                    cand2 = f"{cell}_{cond}_{rep_letter}"
                    nk2 = norm_key(cand2)
                    if nk2 in col_norm:
                        title_to_col[t] = col_norm[nk2]
                        continue

            # Pattern fallback: titles like "HCC827EV, osimertinib, rep1" may map to
            # columns like "HCC827EV_O1" in some processed matrices.
            cell = re.split(r"\s*,\s*", t, maxsplit=1)[0]
            cell = re.sub(r"\s+", "", cell)
            rep = extract_rep_from_title(t)
            letter = guess_condition_letter(t)
            if cell and rep is not None and letter:
                cand_name = f"{cell}_{letter}{rep}"
                if cand_name in col_set:
                    title_to_col[t] = cand_name
                    continue

            # Token-based fallback: match by (cell line, treatment keywords) against
            # column names. This helps cases like "Vehicle-treated HCC827" where
            # matrix columns are like "HCC827...Vehicle".
            tokens: list[str] = []
            # cell line tokens (best-effort)
            for c in re.findall(r"\b[A-Za-z]{2,}[0-9]{2,}\b", t):
                tokens.append(norm_key(c))
            tl = t.lower()
            for kw in ("vehicle", "dmso", "control", "untreated", "erlotinib", "gefitinib", "osimertinib", "afatinib", "combination"):
                if kw in tl:
                    tokens.append(norm_key(kw))
            if tokens:
                hits = []
                for c in sample_cols:
                    ck = norm_key(c)
                    if ck and all(tok in ck for tok in tokens):
                        hits.append(c)
                if len(hits) == 1:
                    title_to_col[t] = hits[0]
                    continue

            # Fallback: some matrices use shortened sample IDs (e.g. "P1") while
            # the GEO series_matrix title is longer (e.g. "Bulk_P1_Parental...").
            # If a column key appears as a substring of the normalized title, pick
            # the longest unique match.
            cand = []
            for c in sample_cols:
                ck = norm_key(c)
                if ck and ck in nk:
                    cand.append((len(ck), c))
            if cand:
                cand.sort(reverse=True)
                best_len = cand[0][0]
                best = [c for L, c in cand if L == best_len]
                if len(best) == 1:
                    title_to_col[t] = best[0]

        used_cols = sorted(set(title_to_col.values()))
        if not used_cols:
            # Still return an empty frame with correct column set
            return (
                pd.DataFrame(index=pd.Index([], name="sample"), columns=sorted(wanted_genes_upper), dtype=float),
                title_to_col,
                delim,
            )

        col_idx = {c: i for i, c in enumerate(sample_cols)}
        used_indices = [col_idx[c] for c in used_cols if c in col_idx]

        data: dict[str, list[float]] = {g: [] for g in sorted(wanted_genes_upper)}
        # We'll build row-wise values then transpose at the end.
        # Keep a parallel list of samples (columns).
        samples = used_cols

        # Temporary storage for extracted genes
        extracted: dict[str, list[float]] = {}

        # Detect Ensembl-ID matrices (e.g., ENSG00000...) so we can map to gene symbols.
        first_gene = ""
        for row in iter_matrix_rows(matrix_path, delim):
            if row and gene_idx < len(row):
                first_gene = strip_quotes(row[gene_idx]).strip()
                if first_gene:
                    break

        ensembl_mode = looks_like_ensembl(first_gene)
        wanted_ensembl_upper = {g for g in wanted_genes_upper if looks_like_ensembl(g)}
        wanted_symbol_upper = set(wanted_genes_upper) - set(wanted_ensembl_upper)

        ensembl_map: dict[str, str] = {}
        if ensembl_mode and wanted_symbol_upper:
            # Build mapping cache per matrix file (only needed when we want symbols).
            cache_dir = Path("reports/external_validation/ensembl_symbol_cache")
            cache_tsv = cache_dir / f"{matrix_path.name}.orgHsEgDb.tsv"

            ensembl_ids: list[str] = []
            for row in iter_matrix_rows(matrix_path, delim):
                if not row or gene_idx >= len(row):
                    continue
                gene = strip_quotes(row[gene_idx]).strip()
                if not gene:
                    continue
                ens = strip_ensembl_version(gene)
                if looks_like_ensembl(ens):
                    ensembl_ids.append(ens)

            ensembl_map = ensure_ensembl_symbol_map(ensembl_ids, cache_tsv)

        for row in iter_matrix_rows(matrix_path, delim):
            if not row:
                continue
            if gene_idx >= len(row):
                continue
            gene = strip_quotes(row[gene_idx]).strip()
            if not gene:
                continue
            keys_to_take: list[str] = []
            if ensembl_mode:
                ens = strip_ensembl_version(gene)
                if looks_like_ensembl(ens):
                    ens_up = ens.upper()
                    if ens_up in wanted_ensembl_upper:
                        keys_to_take.append(ens_up)
                    if wanted_symbol_upper:
                        sym = ensembl_map.get(ens, "")
                        if sym:
                            sym_up = sym.upper()
                            if sym_up in wanted_symbol_upper:
                                keys_to_take.append(sym_up)
            else:
                gup = gene.upper()
                if gup in wanted_genes_upper:
                    keys_to_take.append(gup)

            if not keys_to_take:
                continue

            vals: list[float] = []
            for j in used_indices:
                cell_idx = sample_offset + j
                cell = strip_quotes(row[cell_idx]).strip() if cell_idx < len(row) else ""
                if cell == "" or cell.upper() == "NA":
                    vals.append(0.0)
                    continue
                try:
                    vals.append(float(cell))
                except Exception:
                    vals.append(0.0)

            for gup in keys_to_take:
                # Only aggregate duplicates for symbol-mapped keys (multiple ENSG -> same symbol).
                if ensembl_mode and (gup in wanted_symbol_upper) and (gup in extracted):
                    prev = extracted[gup]
                    extracted[gup] = [a + b for a, b in zip(prev, vals)]
                else:
                    extracted[gup] = vals

        # Build sample x gene with missing genes filled 0.0 (fast, non-fragmenting)
        genes_sorted = sorted(wanted_genes_upper)
        X = pd.DataFrame(extracted, index=samples, dtype=float)
        X = X.reindex(columns=genes_sorted, fill_value=0.0)
        # Attach lightweight QC info (e.g., detect Ensembl-ID matrices where no gene symbols match).
        X.attrs["n_genes_found"] = len(extracted)
        X.attrs["n_genes_wanted"] = len(wanted_genes_upper)
        X.attrs["has_egfr_gene"] = bool("EGFR" in extracted)

        return X, title_to_col, delim


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("metadata/external_cellline_manifest.tsv"))
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--proteins", type=str, default="EGFR,EGFRPY1068")
    ap.add_argument("--transform", choices=("auto", "log2p1", "none"), default="auto")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("reports/external_validation/external_cellline_predicted_deltas.tsv"),
    )
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    proteins = [p.strip() for p in str(args.proteins).split(",") if p.strip()]
    if not proteins:
        raise SystemExit("No proteins specified")

    packs = {p: load_model_pack(args.models_dir, p) for p in proteins}
    wanted_genes_upper: set[str] = set()
    for p, pack in packs.items():
        feats = [str(f) for f in pack.get("features", [])]
        wanted_genes_upper |= {f.upper() for f in feats}

    # also extract EGFR mRNA for reference
    wanted_genes_upper.add("EGFR")

    man = pd.read_csv(args.manifest, sep="\t", dtype=str).fillna("")
    if man.empty:
        raise SystemExit("Manifest is empty")

    # Normalize column names (older manifests might use baseline_title)
    if "baseline_label" not in man.columns and "baseline_title" in man.columns:
        man = man.rename(columns={"baseline_title": "baseline_label"})
    if "perturbed_label" not in man.columns and "perturbed_title" in man.columns:
        man = man.rename(columns={"perturbed_title": "perturbed_label"})

    needed_cols = {"gse", "expression_matrix", "baseline_label", "perturbed_label", "gsm_baseline", "gsm_perturbed"}
    missing = sorted([c for c in needed_cols if c not in man.columns])
    if missing:
        raise SystemExit(f"Manifest missing required columns: {missing}")

    out_rows: list[dict[str, object]] = []

    # Cache per matrix: predictions and raw expression
    cache: dict[str, dict[str, object]] = {}

    for _, r in man.iterrows():
        expr_raw = str(r.get("expression_matrix", "") or "").strip()
        if not expr_raw:
            out_rows.append(
                {
                    "gse": r.get("gse", ""),
                    "pairing_type": r.get("pairing_type", ""),
                    "cell_line": r.get("cell_line", ""),
                    "gsm_baseline": r.get("gsm_baseline", ""),
                    "gsm_perturbed": r.get("gsm_perturbed", ""),
                    "baseline_label": r.get("baseline_label", ""),
                    "perturbed_label": r.get("perturbed_label", ""),
                    "expression_matrix": "",
                    "error": "missing_expression_matrix_path",
                }
            )
            continue

        matrix_path = Path(expr_raw)
        if matrix_path.exists() and matrix_path.is_dir():
            out_rows.append(
                {
                    "gse": r.get("gse", ""),
                    "pairing_type": r.get("pairing_type", ""),
                    "cell_line": r.get("cell_line", ""),
                    "gsm_baseline": r.get("gsm_baseline", ""),
                    "gsm_perturbed": r.get("gsm_perturbed", ""),
                    "baseline_label": r.get("baseline_label", ""),
                    "perturbed_label": r.get("perturbed_label", ""),
                    "expression_matrix": str(matrix_path),
                    "error": "expression_matrix_is_directory",
                }
            )
            continue
        if not matrix_path.exists():
            # Keep row but mark missing matrix
            out_rows.append(
                {
                    "gse": r.get("gse", ""),
                    "pairing_type": r.get("pairing_type", ""),
                    "cell_line": r.get("cell_line", ""),
                    "gsm_baseline": r.get("gsm_baseline", ""),
                    "gsm_perturbed": r.get("gsm_perturbed", ""),
                    "baseline_label": r.get("baseline_label", ""),
                    "perturbed_label": r.get("perturbed_label", ""),
                    "expression_matrix": str(matrix_path),
                    "error": "missing_expression_matrix",
                }
            )
            continue

        key = str(matrix_path)
        if key not in cache:
            # Gather all titles needed for this matrix
            sub = man[man["expression_matrix"] == str(matrix_path)]
            titles = sorted({t for t in sub["baseline_label"].tolist() + sub["perturbed_label"].tolist() if str(t).strip()})

            X_raw, title_to_col, _delim = read_gene_matrix_subset(
                matrix_path=matrix_path,
                wanted_titles=titles,
                wanted_genes_upper=wanted_genes_upper,
            )

            n_genes_found = int(X_raw.attrs.get("n_genes_found", 0) or 0)
            n_genes_wanted = int(X_raw.attrs.get("n_genes_wanted", len(wanted_genes_upper)) or len(wanted_genes_upper))
            gene_coverage_frac = (n_genes_found / n_genes_wanted) if n_genes_wanted else np.nan

            # Decide transform
            transform_mode = args.transform
            if transform_mode == "auto":
                transform_mode = choose_transform_auto(X_raw.to_numpy(dtype=float))
            X_model = apply_transform(X_raw, transform_mode)

            preds_by_protein: dict[str, pd.Series] = {}
            if X_model.shape[0] == 0 or (np.isfinite(gene_coverage_frac) and gene_coverage_frac < 0.05):
                # No matched sample titles in this matrix. We'll mark per-row errors later.
                preds_by_protein = {p: pd.Series(dtype=float) for p in packs.keys()}
            else:
                for p, pack in packs.items():
                    model = pack["model"]
                    scaler = pack.get("scaler")
                    # Important: sklearn transformers fitted on a pandas DataFrame validate
                    # feature *names*, including case (e.g., C16orf89 != C16ORF89).
                    # Our matrix reader normalizes genes to UPPERCASE for robust matching,
                    # so here we reindex by UPPERCASE but then rename columns back to the
                    # exact names seen during scaler.fit.
                    expected_features = [str(f) for f in getattr(scaler, "feature_names_in_", pack.get("features", []))]
                    expected_upper = [f.upper() for f in expected_features]
                    Xm = X_model.reindex(columns=expected_upper, fill_value=0.0).copy()
                    Xm.columns = expected_features
                    Xm = Xm.fillna(0.0)
                    Xs = scaler.transform(Xm) if scaler is not None else Xm.to_numpy(dtype=float)
                    pred = model.predict(Xs)
                    preds_by_protein[p] = pd.Series(pred, index=Xm.index, dtype=float)

            cache[key] = {
                "transform": transform_mode,
                "title_to_col": title_to_col,
                "X_raw": X_raw,
                "preds": preds_by_protein,
                "gene_coverage_frac": gene_coverage_frac,
                "n_genes_found": n_genes_found,
                "n_genes_wanted": n_genes_wanted,
                "has_egfr_gene": bool(getattr(X_raw, "attrs", {}).get("has_egfr_gene", False)),
            }

        entry = cache[key]
        title_to_col = entry["title_to_col"]
        X_raw: pd.DataFrame = entry["X_raw"]
        preds: dict[str, pd.Series] = entry["preds"]
        transform_mode = entry["transform"]
        gene_coverage_frac = float(entry.get("gene_coverage_frac", np.nan))
        n_genes_found = int(entry.get("n_genes_found", 0) or 0)
        n_genes_wanted = int(entry.get("n_genes_wanted", 0) or 0)
        has_egfr_gene = bool(entry.get("has_egfr_gene", False))

        t0 = str(r.get("baseline_label", "")).strip()
        t1 = str(r.get("perturbed_label", "")).strip()
        c0 = title_to_col.get(t0, "")
        c1 = title_to_col.get(t1, "")

        rec: dict[str, object] = {
            "gse": r.get("gse", ""),
            "pairing_type": r.get("pairing_type", ""),
            "cell_line": r.get("cell_line", ""),
            "gsm_baseline": r.get("gsm_baseline", ""),
            "gsm_perturbed": r.get("gsm_perturbed", ""),
            "baseline_label": t0,
            "perturbed_label": t1,
            "baseline_col": c0,
            "perturbed_col": c1,
            "expression_matrix": str(matrix_path),
            "transform": transform_mode,
            "gene_coverage_frac": gene_coverage_frac,
            "n_genes_found": n_genes_found,
            "n_genes_wanted": n_genes_wanted,
            "has_egfr_gene": has_egfr_gene,
            "error": "",
        }

        if np.isfinite(gene_coverage_frac) and gene_coverage_frac < 0.05:
            rec["error"] = "low_gene_coverage"
            out_rows.append(rec)
            continue

        if not c0 or not c1 or c0 not in X_raw.index or c1 not in X_raw.index:
            rec["error"] = "missing_title_mapping"
            out_rows.append(rec)
            continue

        # EGFR mRNA raw (pre-transform)
        egfr0 = float(X_raw.at[c0, "EGFR"]) if "EGFR" in X_raw.columns else np.nan
        egfr1 = float(X_raw.at[c1, "EGFR"]) if "EGFR" in X_raw.columns else np.nan
        rec.update(
            {
                "egfr_mrna_baseline": egfr0,
                "egfr_mrna_perturbed": egfr1,
                "egfr_mrna_delta": (egfr1 - egfr0) if np.isfinite(egfr0) and np.isfinite(egfr1) else np.nan,
            }
        )

        for p, s in preds.items():
            v0 = float(s.get(c0, np.nan))
            v1 = float(s.get(c1, np.nan))
            rec[f"pred_{p}_baseline"] = v0
            rec[f"pred_{p}_perturbed"] = v1
            rec[f"pred_{p}_delta"] = (v1 - v0) if np.isfinite(v0) and np.isfinite(v1) else np.nan

        out_rows.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(args.out, sep="\t", index=False)
    print("WROTE", args.out, "rows=", len(out_rows))


if __name__ == "__main__":
    main()
