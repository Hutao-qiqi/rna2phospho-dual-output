from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/data/lsy/Infinite_Stream")
TRAIN_SCRIPT = BASE / "SCP682-22/scripts/train_scp682_22_cancer_group_pathway_residual.py"
OUT = BASE / "02_results/model_validation/20260520_scp682_reviewer_strict_cancer_holdout_extra"


def load_train_module():
    spec = importlib.util.spec_from_file_location("scp68222_train", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.MODEL_ID = "SCP682_reviewer_strict_cancer_holdout_extra"
    return module


def summarize_delta(per: pd.DataFrame, holdout_name: str) -> dict:
    wide = per.pivot(index="target", columns="model", values="spearman")
    pair = wide[["v4_official", "scp682_strict_holdout"]].dropna()
    diff = pair["scp682_strict_holdout"] - pair["v4_official"]
    return {
        "holdout": holdout_name,
        "n_targets": int(pair.shape[0]),
        "v4_median": float(pair["v4_official"].median()),
        "scp682_median": float(pair["scp682_strict_holdout"].median()),
        "delta_median": float(pair["scp682_strict_holdout"].median() - pair["v4_official"].median()),
        "target_level_delta_median": float(diff.median()),
        "scp682_win_targets": int((diff > 0).sum()),
        "v4_win_targets": int((diff < 0).sum()),
    }


def main() -> int:
    mod = load_train_module()
    mod.mkdirs()
    device = mod.torch.device("cuda:0" if mod.torch.cuda.is_available() else "cpu")
    print("device", device, mod.torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", flush=True)
    mod.seed_all(mod.SEED + 77)
    data = mod.load_inputs()
    data["parent"] = mod.parent_matrix(list(data["y"].columns), data["parent_hat"], data["y"].index)
    pathways, site_weight, _ = mod.build_pathways(data["rna"], list(data["y"].columns))
    manifest = data["manifest"].loc[data["y"].index].copy()
    cancer = manifest["cancer_label"].astype(str)

    holdouts = {
        "HNSCC": cancer.eq("HNSCC").to_numpy(),
        "STAD": cancer.eq("STAD").to_numpy(),
    }
    all_per = []
    summaries = []
    for fold, (name, mask) in enumerate(holdouts.items(), start=1):
        val_idx = np.where(mask)[0]
        train_idx = np.where(~mask)[0]
        print(f"holdout {name}: train={len(train_idx)} val={len(val_idx)}", flush=True)
        pred, row = mod.train_fold(fold, train_idx, val_idx, data, pathways, site_weight, device)
        pred.to_parquet(OUT / "predictions" / f"scp682_strict_holdout_{name}_phosphosite.parquet")
        y = data["y"].loc[pred.index, pred.columns]
        v4 = data["v4"].loc[pred.index, pred.columns]
        per = pd.concat(
            [mod.per_site(y, v4, "v4_official"), mod.per_site(y, pred, "scp682_strict_holdout")],
            ignore_index=True,
        )
        per["holdout"] = name
        all_per.append(per)
        summary = summarize_delta(per, name)
        summary.update({"n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "best_val_loss": float(row["best_val_loss"]), "epochs": int(row["epochs"])})
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(OUT / "tables/scp682_reviewer_strict_holdout_summary.tsv", sep="\t", index=False)
        pd.concat(all_per, ignore_index=True).to_csv(OUT / "tables/scp682_reviewer_strict_holdout_per_site.tsv", sep="\t", index=False)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    (OUT / "logs/run_metadata.json").write_text(json.dumps({"holdouts": summaries}, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "done.txt").write_text("done\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
