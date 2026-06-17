import anndata as ad

p = r"D:\lsy\01_data\single_cell\raw\vivo_seq_th17_2025\GSE297075_Vivo-seq_processed_Scanpy.h5ad"
a = ad.read_h5ad(p, backed="r")
print(a)
print("obs_columns", list(a.obs.columns)[:120])
print("var_columns", list(a.var.columns)[:120])
print("layers", list(a.layers.keys()))
print("obsm", list(a.obsm.keys()))
print("uns", list(a.uns.keys())[:80])
print("var_head")
print(a.var.head(30).to_string())
print("obs_head")
print(a.obs.head(8).to_string())
