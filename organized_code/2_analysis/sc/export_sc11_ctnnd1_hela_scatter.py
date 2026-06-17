import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

root = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
code = root / r"03_code\single_cell\modeling"
sys.path.insert(0, str(code))
import train_scp682_sc11_expanded_scnet_site_gnn as sc

out = root / r"02_results\single_cell\20260522_scp682_sc11_expanded_scnet_site_gnn_v1"
ckpt = torch.load(out / r"models\scp682_sc11_final.pt", map_location="cpu", weights_only=False)
args = ckpt["args"]
input_dir = root / args["model_input_dir"]
features = np.load(input_dir / "embeddings.npy", mmap_mode="r")
meta = pd.read_csv(input_dir / "cell_metadata.tsv", sep="\t", low_memory=False)
target_table = pd.read_csv(input_dir / "phospho_target_table.tsv", sep="\t")
y_all = np.load(input_dir / "targets.npy", mmap_mode="r")
mask_all = np.load(input_dir / "target_mask.npy", mmap_mode="r")

target_rows = ckpt["target_rows"]
pathway_names = ckpt["pathway_names"]
target_indices = [int(r["target_index"]) for r in target_rows]
y_raw = np.asarray(y_all[:, target_indices], dtype=np.float32)
obs_mask = np.asarray(mask_all[:, target_indices], dtype=bool) & np.isfinite(y_raw)
train_idx = sc.build_train_idx(meta, obs_mask, type("A", (), args))
y, transform_stats = sc.transform_targets(y_raw, obs_mask, train_idx, args["target_transform"], target_rows)

model = sc.ScFoundationPathwayPredictor(
    len(pathway_names), y.shape[1], features.shape[1], ckpt["target_pathway_prior"],
    hidden=ckpt["model_config"]["hidden"],
    n_layers=ckpt["model_config"]["pathway_layers"],
    n_heads=ckpt["model_config"]["attention_heads"],
    dropout=ckpt["model_config"]["dropout"],
    bulk_pathway_embedding=ckpt.get("scp68222_full_pathway_embedding"),
    bulk_site_embedding=ckpt.get("scp68222_full_site_embedding"),
    bulk_site_mask=ckpt.get("scp68222_full_site_mask"),
    full_transfer_scale=args["full_transfer_scale"],
    site_graph_edge_index=ckpt.get("scnet_site_graph_edge_index"),
    site_graph_edge_weight=ckpt.get("scnet_site_graph_edge_weight"),
    n_graph_nodes=ckpt.get("scnet_site_graph_summary", {}).get("n_graph_nodes", y.shape[1]),
    site_graph_scale=args["site_graph_scale"],
)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device); model.eval()
present = np.ones((len(meta), len(pathway_names)), dtype=np.float32)
idx_target = {r["target_id"]: i for i,r in enumerate(target_rows)}["CTNND1_T310"]
idx = np.flatnonzero((meta["dataset_id"].astype(str).to_numpy()=="signal_seq_gse256403_hela_2024") & obs_mask[:, idx_target])
rows=[]
with torch.inference_mode():
    for start in range(0, len(idx), 2048):
        b=idx[start:start+2048]
        xb=torch.as_tensor(np.asarray(features[b]), dtype=torch.float32, device=device)
        pb=torch.as_tensor(present[b], dtype=torch.float32, device=device)
        pred,_=model(xb,pb)
        pred=pred[:, idx_target].detach().cpu().numpy()
        obs=y[b, idx_target]
        for cell_id, p, o in zip(meta.iloc[b]["cell_id"].astype(str), pred, obs):
            rows.append({"cell_id": cell_id, "predicted": float(p), "observed": float(o), "target_id": "CTNND1_T310", "cohort_id": "signal_seq_gse256403_hela_2024"})
res=pd.DataFrame(rows)
figdir=out / "tables"
res.to_csv(figdir / "scp682_sc11_hela_ctnnd1_t310_predicted_observed.tsv", sep="\t", index=False)
print(len(res), res["predicted"].corr(res["observed"], method="spearman"), res["predicted"].corr(res["observed"], method="pearson"))
