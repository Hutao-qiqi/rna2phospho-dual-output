from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
RESULT = ROOT / r"02_results\single_cell\20260522_scp682_sc11_expanded_scnet_site_gnn_v1"


def main():
    import torch

    ckpt = torch.load(RESULT / r"models\scp682_sc11_final.pt", map_location="cpu", weights_only=False)
    input_dir = ROOT / ckpt["args"]["model_input_dir"]
    meta = pd.read_csv(input_dir / "cell_metadata.tsv", sep="\t", low_memory=False)
    features = np.load(input_dir / "embeddings.npy", mmap_mode="r")
    ds = meta["dataset_id"].astype(str).to_numpy()
    idx = np.flatnonzero(ds == "signal_seq_gse256403_hela_2024")
    x = np.asarray(features[idx], dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        import umap
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.20, metric="cosine", random_state=682)
        emb = reducer.fit_transform(x)
        method = "umap"
    except Exception:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2, random_state=682).fit_transform(x)
        method = "pca_fallback"

    out = pd.DataFrame(
        {
            "row_index": idx.astype(int),
            "cell_id": meta.iloc[idx]["cell_id"].astype(str).to_numpy() if "cell_id" in meta.columns else idx.astype(str),
            "umap1": emb[:, 0],
            "umap2": emb[:, 1],
            "method": method,
        }
    )
    out_file = RESULT / "tables" / "scp682_sc11_hela_scfoundation_umap.tsv"
    out.to_csv(out_file, sep="\t", index=False)
    print(out_file)
    print(out.shape)
    print(method)


if __name__ == "__main__":
    main()
