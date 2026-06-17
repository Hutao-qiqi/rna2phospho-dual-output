# 模型: SCP682-SC
# 作用: 导出五个外部验证队列的逐细胞预测值、实测值和逐靶点 Spearman。
# 输入: ./data_root 下的统一模型输入、./results/SCP682_SC/models/scp682_sc11_final.pt。
# 输出: ./results/SCP682_SC/tables/scp682_sc11_predicted_observed_<cohort>.tsv 与汇总表。
# 依赖: Python, numpy, pandas, torch, scipy, SCP682-SC 训练脚本。
# 原始路径: D:\data\lsy\vm_lsy_parent\lsy\03_code\single_cell\modeling\export_sc11_external_predicted_observed_all.py
# 原始版本: 2026-05-23 补算外部逐细胞预测/实测明细。

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path("./data_root")
CODE_DIR = ROOT / "03_code/single_cell/modeling"
RESULT_DIR = Path("./results/SCP682_SC")
DEVICE = "cuda:0"
BATCH_SIZE = 4096

sys.path.insert(0, str(CODE_DIR))
import train_scp682_sc11_expanded_scnet_site_gnn as sc


def spearman_pvalue(x, y):
    try:
        from scipy.stats import spearmanr

        r, p = spearmanr(x, y)
        return float(r), float(p)
    except Exception:
        r = pd.Series(x).corr(pd.Series(y), method="spearman")
        return float(r), np.nan


def inverse_one(values, stat):
    mode = str(stat.get("mode", ""))
    if mode == "zscore":
        mean = float(stat.get("mean", 0.0))
        sd = float(stat.get("sd", 1.0))
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        return values * sd + mean
    return values


def main():
    ckpt = torch.load(RESULT_DIR / r"models\scp682_sc11_final.pt", map_location="cpu", weights_only=False)
    args = dict(ckpt["args"])
    input_dir = ROOT / args["model_input_dir"]

    features = np.load(input_dir / "embeddings.npy", mmap_mode="r")
    meta = pd.read_csv(input_dir / "cell_metadata.tsv", sep="\t", low_memory=False)
    y_all = np.load(input_dir / "targets.npy", mmap_mode="r")
    mask_all = np.load(input_dir / "target_mask.npy", mmap_mode="r")

    target_rows = ckpt["target_rows"]
    pathway_names = ckpt["pathway_names"]
    target_indices = [int(r["target_index"]) for r in target_rows]
    y_raw = np.asarray(y_all[:, target_indices], dtype=np.float32)
    obs_mask = np.asarray(mask_all[:, target_indices], dtype=bool) & np.isfinite(y_raw)

    args_obj = type("Args", (), args)
    train_idx = sc.build_train_idx(meta, obs_mask, args_obj)
    y, transform_stats = sc.transform_targets(
        y_raw,
        obs_mask,
        train_idx,
        args["target_transform"],
        target_rows,
    )
    if isinstance(transform_stats, pd.DataFrame):
        stats_by_target = {int(row["target_order"]): row for row in transform_stats.to_dict("records")}
    else:
        stats_by_target = {int(row["target_order"]): row for row in transform_stats}

    model = sc.ScFoundationPathwayPredictor(
        len(pathway_names),
        y.shape[1],
        features.shape[1],
        ckpt["target_pathway_prior"],
        hidden=ckpt["model_config"]["hidden"],
        n_layers=ckpt["model_config"]["pathway_layers"],
        n_heads=ckpt["model_config"]["attention_heads"],
        dropout=ckpt["model_config"]["dropout"],
        bulk_pathway_embedding=ckpt.get("full_pathway_embedding", ckpt.get("scp682_main_full_pathway_embedding", ckpt.get("scp68222_full_pathway_embedding"))),
        bulk_site_embedding=ckpt.get("full_site_embedding", ckpt.get("scp682_main_full_site_embedding", ckpt.get("scp68222_full_site_embedding"))),
        bulk_site_mask=ckpt.get("full_site_mask", ckpt.get("scp682_main_full_site_mask", ckpt.get("scp68222_full_site_mask"))),
        full_transfer_scale=args["full_transfer_scale"],
        site_graph_edge_index=ckpt.get("scnet_site_graph_edge_index"),
        site_graph_edge_weight=ckpt.get("scnet_site_graph_edge_weight"),
        n_graph_nodes=ckpt.get("scnet_site_graph_summary", {}).get("n_graph_nodes", y.shape[1]),
        site_graph_scale=args["site_graph_scale"],
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    present = np.ones((len(meta), len(pathway_names)), dtype=np.float32)
    holdouts = [x for x in str(args["holdout_datasets"]).split(",") if x]
    table_dir = RESULT_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    summary_rows = []
    cell_col = "cell_id" if "cell_id" in meta.columns else None

    with torch.inference_mode():
        for cohort in holdouts:
            cohort_idx = np.flatnonzero((meta["dataset_id"].astype(str).to_numpy() == cohort) & obs_mask.any(axis=1))
            if len(cohort_idx) == 0:
                manifest_rows.append(
                    {
                        "cohort_id": cohort,
                        "n_cells": 0,
                        "n_observed_pairs": 0,
                        "n_targets": 0,
                        "output_file": "NA",
                    }
                )
                continue

            pred = np.zeros((len(cohort_idx), y.shape[1]), dtype=np.float32)
            for start in range(0, len(cohort_idx), BATCH_SIZE):
                b = cohort_idx[start : start + BATCH_SIZE]
                xb = torch.as_tensor(np.asarray(features[b]), dtype=torch.float32, device=device)
                pb = torch.as_tensor(present[b], dtype=torch.float32, device=device)
                out, _ = model(xb, pb)
                pred[start : start + len(b)] = out.detach().cpu().numpy()

            long_rows = []
            cohort_mask = obs_mask[cohort_idx]
            for j, row in enumerate(target_rows):
                local = np.flatnonzero(cohort_mask[:, j])
                if len(local) == 0:
                    continue
                pred_t = pred[local, j].astype(np.float64)
                obs_t = y[cohort_idx[local], j].astype(np.float64)
                pred_raw = inverse_one(pred_t, stats_by_target.get(j, {}))
                obs_raw = y_raw[cohort_idx[local], j].astype(np.float64)
                sp, pv = spearman_pvalue(pred_t, obs_t)
                summary_rows.append(
                    {
                        "cohort_id": cohort,
                        "target_id": row["target_id"],
                        "target_order": j,
                        "sample_size_used": int(len(local)),
                        "spearman": sp,
                        "spearman_pvalue": pv,
                    }
                )
                for li, p, o, pr, oraw in zip(local, pred_t, obs_t, pred_raw, obs_raw):
                    global_i = int(cohort_idx[li])
                    long_rows.append(
                        {
                            "cohort_id": cohort,
                            "cell_id": str(meta.iloc[global_i][cell_col]) if cell_col else str(global_i),
                            "row_index": global_i,
                            "target_id": row["target_id"],
                            "target_order": j,
                            "predicted": float(p),
                            "observed": float(o),
                            "predicted_raw_scale": float(pr),
                            "observed_raw_scale": float(oraw),
                        }
                    )

            out_file = table_dir / f"scp682_sc11_predicted_observed_{cohort}.tsv"
            pd.DataFrame(long_rows).to_csv(out_file, sep="\t", index=False)
            manifest_rows.append(
                {
                    "cohort_id": cohort,
                    "n_cells": int(len(cohort_idx)),
                    "n_observed_pairs": int(len(long_rows)),
                    "n_targets": int(pd.DataFrame(long_rows)["target_id"].nunique()) if long_rows else 0,
                    "output_file": out_file.name,
                }
            )
            print(cohort, len(cohort_idx), len(long_rows), out_file.name, flush=True)

    pd.DataFrame(manifest_rows).to_csv(
        table_dir / "scp682_sc11_external_predicted_observed_manifest.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(summary_rows).to_csv(
        table_dir / "scp682_sc11_external_per_target_from_predicted_observed.tsv",
        sep="\t",
        index=False,
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
