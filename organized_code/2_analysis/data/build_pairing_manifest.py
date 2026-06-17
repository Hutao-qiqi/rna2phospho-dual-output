#!/usr/bin/env python3
"""Build an initial external cell-line pairing manifest from GEO series_matrix metadata.

Inputs:
  reports/external_validation/series_matrix_characteristics_long.tsv
  reports/external_validation/series_matrix_samples.tsv

Outputs:
  metadata/external_cellline_manifest.tsv

Heuristics (best-effort):
- Prefer pairing using key 'tki response' (Sensitive vs Resistant) grouped by 'cell line'.
- Fallback: infer cell line from sample title prefix and pair control/nontreated vs treated/selected.

This manifest is intended for human review/editing.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()


def infer_cell_line_from_title(title: str) -> str:
    t = str(title).strip()
    if not t:
        return ""
    # Common patterns: HCC827-GR, H1975-OR, PC9GR, PC9OR
    t = re.sub(r"[\s_]+", "", t)
    t = re.sub(r"(-?GR|-?OR|-?RESISTANT|-?SENSITIVE)$", "", t, flags=re.I)
    return t


def extract_rep(title: str) -> int | None:
    """Extract replicate number from a sample title (e.g. 'rep1', 'Rep 2')."""
    m = re.search(r"\brep\s*([0-9]+)\b", str(title), flags=re.I)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def pairing_key_from_title(title: str) -> str:
    """Derive a pairing key from a sample title.

    Goal: pair within the same subline/background while ignoring treatment tokens.
    Example: 'PC9_GR3_WZ.rep1' and 'PC9_GR3_VEH.rep1' -> 'pc9_gr3'
    """

    t = str(title).strip().lower()
    if not t:
        return ""

    # Remove replicate suffix.
    t = re.sub(r"\brep\s*[0-9]+\b", "", t, flags=re.I)

    # Normalize: turn any non-alphanumeric into spaces so token boundaries work
    # with \b (note: '_' counts as a word char).
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # Remove common treatment/control tokens.
    # Keep this list conservative; it can be expanded as needed.
    t = re.sub(
        r"\b(vehicle|veh|dmso|control|ctrl|untreated|nontreated|baseline|mock)\b",
        "",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"\b(gefitinib|gef|erlotinib|erl|osimertinib|osi|afatinib|afa|wz4002|wz)\b",
        "",
        t,
        flags=re.I,
    )

    # Collapse leftover separators.
    t = re.sub(r"\s+", "_", t).strip("_")
    return t


def is_control_like(title: str, treatment: str) -> bool:
    t = norm(title)
    tr = norm(treatment)
    if re.search(r"\b(control|ctrl|untreated|nontreated|vehicle|veh|dmso|parental|baseline|mock)\b", t) or re.search(
        r"\b(control|ctrl|untreated|nontreated|vehicle|veh|dmso|parental|baseline|mock)\b", tr
    ):
        if re.search(r"\b(resistan|selected|gefitinib|gef\b|erlotinib|erl\b|osimertinib|osi\b|afatinib|tki|dtc|toleran|wz\b|wz4002)\b", t) or re.search(
            r"\b(resistan|selected|gefitinib|gef\b|erlotinib|erl\b|osimertinib|osi\b|afatinib|tki|dtc|toleran|wz\b|wz4002)\b", tr
        ):
            return False
        return True
    return False


def is_treated_like(title: str, treatment: str) -> bool:
    t = norm(title)
    tr = norm(treatment)
    return bool(
        re.search(r"\b(resistan|selected|gefitinib|gef\b|erlotinib|erl\b|osimertinib|osi\b|afatinib|tki|dtc|toleran|wz\b|wz4002)\b", t)
        or re.search(r"\b(resistan|selected|gefitinib|gef\b|erlotinib|erl\b|osimertinib|osi\b|afatinib|tki|dtc|toleran|wz\b|wz4002)\b", tr)
    )


def _iter_expression_candidates(gse_dir: Path) -> list[Path]:
    cand: list[Path] = []
    for p in gse_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if "series_matrix" in name:
            continue
        # Some GEO processed matrices are mislabeled as .xls/.xlsx but are actually
        # plain text tables, so we include these extensions as candidates too.
        if name.endswith(
            (
                ".txt",
                ".tsv",
                ".csv",
                ".xls",
                ".xlsx",
                ".txt.gz",
                ".tsv.gz",
                ".csv.gz",
                ".xls.gz",
                ".xlsx.gz",
            )
        ):
            cand.append(p)
    return cand


def choose_expression_matrix(gse_dir: Path, cell_line: str = "") -> str:
    """Pick the best processed expression matrix file in a GEO supplementary folder.

    Some GSE folders contain multiple matrices (e.g., one per cell line). In that
    case we try to select a file whose name matches the cell line.
    """

    cand = _iter_expression_candidates(gse_dir)
    if not cand:
        return ""

    cell_key = re.sub(r"[^a-z0-9]+", "", str(cell_line).lower())

    def score(p: Path) -> tuple[int, int]:
        n = p.name.lower()
        n_key = re.sub(r"[^a-z0-9]+", "", n)
        s = 0

        # Strong preference when filename contains the cell line.
        if cell_key and cell_key in n_key:
            s += 100

        # Prefer likely expression/count matrices.
        for key in [
            "tpm",
            "fpkm",
            "rpkm",
            "counts",
            "countmatrix",
            "matrix",
            "expr",
            "expression",
            "normalized",
            "raw",
            "bulk",
            "gene",
        ]:
            if key in n:
                s += 10

        # Minor extension preferences.
        if n.endswith(".csv.gz"):
            s += 1
        if n.endswith(".txt.gz"):
            s += 2

        return (s, -len(n))

    cand.sort(key=score, reverse=True)
    return str(cand[0].as_posix())


def main() -> None:
    samples_path = Path("reports/external_validation/series_matrix_samples.tsv")
    long_path = Path("reports/external_validation/series_matrix_characteristics_long.tsv")
    if not samples_path.exists() or not long_path.exists():
        raise SystemExit("Run parse_geo_series_matrix.py first")

    samples = pd.read_csv(samples_path, sep="\t", dtype=str).fillna("")
    long_df = pd.read_csv(long_path, sep="\t", dtype=str).fillna("")

    # pivot characteristics to dict per sample (keep last value if repeated)
    long_df["key_norm"] = long_df["key"].map(norm)
    kv = (
        long_df.sort_values(["gse", "gsm"])
        .groupby(["gse", "gsm"])
        .apply(lambda d: {k: " | ".join(d.loc[d["key_norm"] == k, "value"].tolist()) for k in sorted(set(d["key_norm"]))})
    )
    kv = kv.to_dict()

    rows = []
    for gse, group in samples.groupby("gse"):
        gse_dir = Path("data/raw/geo_supplementary") / gse
        gse_dir_exists = gse_dir.exists()

        # Build per-sample record with helpful fields
        recs = []
        for _, r in group.iterrows():
            gsm = r["gsm"]
            title = r["title"]
            d = kv.get((gse, gsm), {})
            cell_line = d.get("cell line", "") or infer_cell_line_from_title(title)
            tki_response = d.get("tki response", "")
            treatment = d.get("treatment", "") or d.get("drug treatment", "")
            recs.append(
                {
                    "gse": gse,
                    "gsm": gsm,
                    "title": title,
                    "cell_line": cell_line,
                    "tki_response": tki_response,
                    "treatment": treatment,
                }
            )

        df = pd.DataFrame(recs)
        if df.empty:
            continue

        # Pairing strategy A: tki response sensitive vs resistant by cell line
        paired_any = False
        if (df["tki_response"].str.len() > 0).any() and (df["cell_line"].str.len() > 0).any():
            for cell_line, d2 in df.groupby("cell_line"):
                if not cell_line:
                    continue
                sens = d2[d2["tki_response"].str.contains("sensitive", case=False, na=False)]
                res = d2[d2["tki_response"].str.contains("resist", case=False, na=False)]
                if len(sens) >= 1 and len(res) >= 1:
                    paired_any = True
                    base = sens.iloc[0]
                    for _, rr in res.iterrows():
                        rows.append(
                            {
                                "gse": gse,
                                "pairing_type": "tki_response",
                                "cell_line": cell_line,
                                "gsm_baseline": base["gsm"],
                                "gsm_perturbed": rr["gsm"],
                                "baseline_label": base["title"],
                                "perturbed_label": rr["title"],
                                "expression_matrix": choose_expression_matrix(gse_dir, cell_line) if gse_dir_exists else "",
                                "notes": "auto: sensitive vs resistant",
                            }
                        )

        # Pairing strategy B: infer control vs treated from treatment field/title
        if not paired_any and (df["cell_line"].str.len() > 0).any():
            for cell_line, d2 in df.groupby("cell_line"):
                if not cell_line or len(d2) < 2:
                    continue
                ctrl = d2[d2.apply(lambda r: is_control_like(r.get("title", ""), r.get("treatment", "")), axis=1)]
                treated = d2[d2.apply(lambda r: is_treated_like(r.get("title", ""), r.get("treatment", "")), axis=1)]
                if ctrl.empty or treated.empty:
                    continue

                # Build pairs. Prefer matching by rep number, else by order.
                ctrl2 = ctrl.copy()
                treated2 = treated.copy()
                ctrl2["rep"] = ctrl2["title"].map(extract_rep)
                treated2["rep"] = treated2["title"].map(extract_rep)
                ctrl2["pair_key"] = ctrl2["title"].map(pairing_key_from_title)
                treated2["pair_key"] = treated2["title"].map(pairing_key_from_title)

                pairs: list[tuple[pd.Series, pd.Series]] = []
                if ctrl2["rep"].notna().any() and treated2["rep"].notna().any():
                    # Prefer matching by (pair_key, rep) to avoid mixing sublines.
                    repkey_to_ctrl = {}
                    for _, rrr in ctrl2.dropna(subset=["rep"]).iterrows():
                        rk = (str(rrr.get("pair_key", "") or ""), int(rrr["rep"]))
                        if rk not in repkey_to_ctrl:
                            repkey_to_ctrl[rk] = rrr

                    for _, rr in treated2.dropna(subset=["rep"]).iterrows():
                        rk = (str(rr.get("pair_key", "") or ""), int(rr["rep"]))
                        c = repkey_to_ctrl.get(rk)
                        if c is not None:
                            pairs.append((c, rr))

                    # Fallback: if no pair_key matches, match by rep only.
                    if not pairs:
                        rep_to_ctrl = {int(r["rep"]): r for _, r in ctrl2.dropna(subset=["rep"]).iterrows()}
                        for _, rr in treated2.dropna(subset=["rep"]).iterrows():
                            c = rep_to_ctrl.get(int(rr["rep"]))
                            if c is not None:
                                pairs.append((c, rr))

                if not pairs:
                    base = ctrl2.iloc[0]
                    for _, rr in treated2.iterrows():
                        pairs.append((base, rr))

                for base, rr in pairs:
                    rows.append(
                        {
                            "gse": gse,
                            "pairing_type": "treatment",
                            "cell_line": cell_line,
                            "gsm_baseline": base["gsm"],
                            "gsm_perturbed": rr["gsm"],
                            "baseline_label": base["title"],
                            "perturbed_label": rr["title"],
                            "expression_matrix": choose_expression_matrix(gse_dir, cell_line) if gse_dir_exists else "",
                            "notes": "auto: control-like vs treated-like",
                        }
                    )

    out = Path("metadata/external_cellline_manifest.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        out_df = pd.DataFrame(
            columns=[
                "gse",
                "pairing_type",
                "cell_line",
                "gsm_baseline",
                "gsm_perturbed",
                "baseline_label",
                "perturbed_label",
                "expression_matrix",
                "notes",
            ]
        )
    out_df.to_csv(out, sep="\t", index=False)
    print("WROTE", out, "rows=", len(out_df))


if __name__ == "__main__":
    main()
