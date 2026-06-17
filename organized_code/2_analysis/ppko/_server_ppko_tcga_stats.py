from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
OUT = ROOT / "02_results" / "clinical_validation" / "20260527_tcga_tcpa_ppko_patient_response_v1"
pred = pd.read_csv(OUT / "tables" / "all_model_patient_predictions.tsv", sep="\t")


def auc_from_scores(y_true, scores):
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg)
        wins += 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


rng = np.random.default_rng(20260527)
rows = []
score_cols = [
    "ppko_abs_delta_mean",
    "ppko_abs_delta_top200_mean",
    "ppko_target_prior_abs_mean",
    "ppko_observed_site_abs_mean",
]
for model_name, sub in pred.groupby("model_name"):
    y = sub["response_binary"].to_numpy(float)
    for score_col in score_cols:
        score = sub[score_col].to_numpy(float)
        auc = auc_from_scores(y, score)
        boot = []
        for _ in range(2000):
            idx = rng.integers(0, len(sub), len(sub))
            val = auc_from_scores(y[idx], score[idx])
            if np.isfinite(val):
                boot.append(val)
        perm = []
        for _ in range(2000):
            yp = rng.permutation(y)
            val = auc_from_scores(yp, score)
            if np.isfinite(val):
                perm.append(val)
        rows.append({
            "model_name": model_name,
            "score": score_col,
            "n": int(len(sub)),
            "n_responder": int(np.sum(y == 1)),
            "n_non_responder": int(np.sum(y == 0)),
            "auc": auc,
            "bootstrap_ci_low": float(np.quantile(boot, 0.025)) if boot else np.nan,
            "bootstrap_ci_high": float(np.quantile(boot, 0.975)) if boot else np.nan,
            "permutation_p_right": float((np.sum(np.asarray(perm) >= auc) + 1) / (len(perm) + 1)) if perm else np.nan,
        })

stats = pd.DataFrame(rows)
stats.to_csv(OUT / "tables" / "model_score_auc_ci_permutation.tsv", sep="\t", index=False)
print(stats.to_string(index=False))
