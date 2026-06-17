import numpy as np, pandas as pd, json
from pathlib import Path
root=Path(r'D:\data\lsy\vm_lsy_parent\lsy')
transfer=root/r'01_data\single_cell\intermediate\scp682_main_sc_transfer_prior_v1\tables\scp682_main_sc_transfer_arrays.npz'
arr=np.load(transfer, allow_pickle=True)
for k in arr.files:
    v=arr[k]
    if hasattr(v,'dtype') and np.issubdtype(v.dtype, np.number):
        print('ARRAY',k,v.shape,v.dtype,'finite',np.isfinite(v).all(),'nan',np.isnan(v).sum(),'min',np.nanmin(v),'max',np.nanmax(v))
    else:
        print('ARRAY',k,v.shape,v.dtype)
smoke=root/r'02_results\single_cell\20260529_scp682_sc11_current_main_transfer_smoke_v1\tables'
for name in ['scp682_sc11_scp682_main_full_pathway_embedding.tsv','scp682_sc11_scp682_main_full_site_embedding.tsv','scp682_sc11_scp682_main_transfer_prior.tsv','scp682_sc11_target_transform.tsv']:
    p=smoke/name
    print('FILE',name,p.exists())
    if p.exists():
        df=pd.read_csv(p,sep='\t')
        nums=df.select_dtypes(include=['number']).to_numpy()
        print(df.shape,'numeric_finite',np.isfinite(nums).all() if nums.size else True,'nan',np.isnan(nums).sum() if nums.size else 0)
        print(df.head(3).to_string())
