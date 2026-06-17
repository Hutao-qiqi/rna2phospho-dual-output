from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.io import mmread, mmwrite
from scipy import sparse

root = Path("/data/lsy/phospho_project")
base = root / "export" / "iccite_seq_tcell_2025"
out = root / "export" / "iccite_background_corrected"
out.mkdir(parents=True, exist_ok=True)

intra_dir = base / "intra_counts"
mtx = mmread(intra_dir / "intra_counts.mtx").tocsr()
features = pd.read_csv(intra_dir / "intra_counts_features.tsv", sep="\t", header=None)[0].astype(str).tolist()
barcodes = pd.read_csv(intra_dir / "intra_counts_barcodes.tsv", sep="\t", header=None)[0].astype(str).tolist()

feature_table = pd.DataFrame({"assay": "intra", "feature": features})
feature_table["is_control_like"] = feature_table["feature"].str.contains(r"isotype|control|(^|-)mIgG", case=False, regex=True)
feature_table["is_phospho_like"] = feature_table["feature"].str.contains("phospho|Pho|pSTAT|RPS6Pho", case=False, regex=True)
feature_table.to_csv(out / "intra_feature_control_phospho_flags.tsv", sep="\t", index=False)

control_idx = np.where(feature_table["is_control_like"].values)[0]
phospho_idx = np.where(feature_table["is_phospho_like"].values)[0]

if len(control_idx) == 0:
    raise SystemExit("no control-like intra features found")
if len(phospho_idx) == 0:
    raise SystemExit("no phospho-like intra features found")

control = mtx[control_idx, :].toarray()
control_mean = control.mean(axis=0)
control_median = np.median(control, axis=0)
control_max = control.max(axis=0)

pd.DataFrame(
    {
        "cell_id": barcodes,
        "control_mean": control_mean,
        "control_median": control_median,
        "control_max": control_max,
    }
).to_csv(out / "cell_control_background_summary.tsv", sep="\t", index=False)

rows = []
corrected_rows = []
for idx in phospho_idx:
    label = features[idx]
    x = mtx[idx, :].toarray().ravel().astype(float)
    z = np.maximum(x - control_mean, 0.0)
    corrected_rows.append(sparse.csr_matrix(z))
    rows.append(
        {
            "feature": label,
            "raw_mean": float(np.mean(x)),
            "raw_median": float(np.median(x)),
            "raw_nonzero_rate": float(np.mean(x > 0)),
            "control_mean_mean": float(np.mean(control_mean)),
            "spearman_control_mean": float(pd.Series(x).corr(pd.Series(control_mean), method="spearman")),
            "corrected_mean": float(np.mean(z)),
            "corrected_median": float(np.median(z)),
            "corrected_nonzero_rate": float(np.mean(z > 0)),
            "fraction_removed_by_mean_subtraction": float(1.0 - (z.sum() / x.sum())) if x.sum() > 0 else np.nan,
        }
    )

qc = pd.DataFrame(rows)
qc.to_csv(out / "phospho_background_qc.tsv", sep="\t", index=False)

corrected = sparse.vstack(corrected_rows).tocsr()
mmwrite(out / "phospho_counts_control_mean_subtracted.mtx", corrected)
pd.Series([features[i] for i in phospho_idx]).to_csv(out / "phospho_counts_control_mean_subtracted_features.tsv", sep="\t", index=False, header=False)
pd.Series(barcodes).to_csv(out / "phospho_counts_control_mean_subtracted_barcodes.tsv", sep="\t", index=False, header=False)

pd.DataFrame(
    {
        "item": ["n_cells", "n_intra_features", "n_control_features", "n_phospho_features"],
        "value": [len(barcodes), len(features), len(control_idx), len(phospho_idx)],
    }
).to_csv(out / "background_correction_summary.tsv", sep="\t", index=False)

print("control features")
print("\n".join(feature_table.loc[control_idx, "feature"].tolist()))
print("phospho qc")
print(qc.to_string(index=False))
