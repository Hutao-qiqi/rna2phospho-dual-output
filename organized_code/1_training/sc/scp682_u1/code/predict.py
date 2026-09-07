"""Frozen SCP682 prediction from RNA counts and row-aligned scFoundation embeddings."""
import argparse
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from model import SCP682,normalize_log1p

def main():
    p=argparse.ArgumentParser();p.add_argument('--release-dir',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--rna-h5ad',type=Path,required=True);p.add_argument('--embeddings',type=Path,required=True)
    p.add_argument('--embedding-cell-ids',type=Path,required=True,help='Headerless cell ID file in embedding row order')
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--domain',type=int,choices=range(4),default=0)
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=128);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    prep=np.load(a.release_dir/'preprocessing.npz');readouts=pd.read_csv(a.release_dir/'readouts.tsv',sep='\t')
    x=ad.read_h5ad(a.rna_h5ad);e=np.load(a.embeddings,mmap_mode='r')
    ids=pd.read_csv(a.embedding_cell_ids,sep='\t',header=None)[0].astype(str)
    assert x.obs_names.is_unique and x.var_names.is_unique
    assert np.array_equal(ids.to_numpy(),x.obs_names.to_numpy(str)),'Embedding cell order mismatch'
    assert e.shape==(x.n_obs,3072)
    values=x.X.data if sparse.issparse(x.X) else np.asarray(x.X)
    assert np.isfinite(values).all() and (values>=0).all(),'X must contain finite nonnegative RNA counts'
    assert np.allclose(values,np.round(values),atol=1e-5),'Supply untransformed counts in X'
    pos=x.var_names.get_indexer(prep['genes']);good=pos>=0
    mapper=sparse.csr_matrix((np.ones(good.sum()),(pos[good],np.flatnonzero(good))),shape=(x.n_vars,len(pos)))
    xx=(normalize_log1p(sparse.csr_matrix(x.X))@mapper).tocsr()
    model=SCP682(ns=len(readouts)).to(a.device)
    ck=torch.load(a.release_dir/'SCP682_U1.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ck['state_dict'],strict=True);model.eval();pieces=[]
    with torch.inference_mode():
        for start in range(0,len(ids),a.batch_size):
            h=(xx[start:start+a.batch_size].toarray()-prep['gene_mean'])/prep['gene_std']
            ee=np.asarray(e[start:start+a.batch_size],np.float32);assert np.isfinite(ee).all()
            pred,_=model(torch.tensor(h,dtype=torch.float32,device=a.device),torch.tensor(ee,device=a.device),torch.full((len(h),),a.domain,dtype=torch.long,device=a.device))
            pieces.append(pred.cpu().numpy())
    prediction=np.vstack(pieces)
    if a.domain:
        norm=np.load(a.release_dir/'domain_normalization.npz')
        prediction=prediction*norm[f'std_{a.domain}']+norm[f'mean_{a.domain}']
        use=norm[f'mask_{a.domain}'].astype(bool)
    else:use=readouts.supervised.astype(bool).to_numpy()
    result=pd.DataFrame(prediction[:,use],index=ids.to_numpy(),columns=readouts.loc[use,'target_id'])
    result.index.name='cell_id';result.to_csv(a.output_dir/'predictions.tsv',sep='\t')
    pd.DataFrame({'gene':prep['genes'],'present':good}).to_csv(a.output_dir/'gene_coverage.tsv',sep='\t',index=False)
    (a.output_dir/'inference.json').write_text(json.dumps({'model':'SCP682','configuration':'U1','domain':a.domain,
       'cells':len(ids),'readouts':int(use.sum()),'present_hvg':int(good.sum()),'unit':'standardized_phosphorylation' if not a.domain else 'training_domain_native_scale',
       'missing_genes':'zero_counts_before_frozen_zscore','gene_identity':'exact case and species namespace'},indent=2))
if __name__=='__main__':main()
