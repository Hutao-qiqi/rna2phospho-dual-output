import argparse
import gzip
import json
import math
import sys
import tarfile
import time
import subprocess
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.stats import spearmanr, pearsonr
import torch
from torch import nn
import torch.nn.functional as F

from model import SCP682 as Unified, normalize_log1p

DATASETS = ['gse300551_iccite_plex_kinase_2025', 'iccite_seq_tcell_2025', 'qurie_seq_bjab_2021']

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')

def raw_inputs(root, meta):
    ds = DATASETS[1]
    dest = root/'data/h5ad'/f'{ds}.h5ad'
    if not dest.exists():
        with tarfile.open(root/'data/unified_raw/iccite_export_full_rna.tar.gz') as archive:
            members = archive.getnames()
            def get(suffix):
                return archive.extractfile(next(n for n in members if n.endswith(suffix)))
            cells = pd.Index(pd.read_csv(get('rna_full_counts_barcodes.tsv'), header=None, sep='\t')[0].astype(str))
            genes = pd.Index(pd.read_csv(get('rna_full_counts_features.tsv'), header=None, sep='\t')[0].astype(str))
            x = mmread(get('rna_full_counts.mtx')).tocsr()
        if x.shape == (len(genes),len(cells)):
            x = x.T.tocsr()
        wanted = meta.loc[meta.dataset_id.eq(ds),'cell_id'].astype(str)
        idx = cells.get_indexer(wanted)
        assert (idx >= 0).all(), 'icCITE missing RNA cells'
        temporary = dest.with_suffix('.writing.h5ad')
        ad.AnnData(x[idx].astype('float32'), obs=pd.DataFrame(index=pd.Index(wanted, name='cell_id')), var=pd.DataFrame(index=genes.rename('gene'))).write_h5ad(temporary, compression='gzip')
        temporary.replace(dest)
    ds = DATASETS[2]
    dest = root/'data/h5ad'/f'{ds}.h5ad'
    if not dest.exists():
        parts = []
        current = meta[meta.dataset_id.eq(ds)]
        with tarfile.open(root/'data/unified_raw/GSE162461_RAW.tar') as archive:
            for gsm, subset in current.groupby('rna_gsm',sort=False):
                name = next(n for n in archive.getnames() if n.startswith(gsm) and 'mrna.counts' in n.lower())
                with gzip.open(archive.extractfile(name),'rt') as f:
                    table = pd.read_csv(f,sep='\t',index_col=0)
                table = table.loc[subset.raw_barcode.astype(str)]
                parts.append(ad.AnnData(sparse.csr_matrix(table.to_numpy('float32')),obs=pd.DataFrame(index=pd.Index(subset.cell_id.astype(str),name='cell_id')),var=pd.DataFrame(index=table.columns.astype(str).rename('gene'))))
        combined = ad.concat(parts,join='outer',fill_value=0)
        temporary = dest.with_suffix('.writing.h5ad')
        combined[current.cell_id.astype(str)].write_h5ad(temporary,compression='gzip')
        temporary.replace(dest)

def prepare(root, out):
    meta = pd.read_csv(root/'data/model_input/cell_metadata.tsv',sep='\t',low_memory=False)
    raw_inputs(root,meta)
    y = np.load(root/'data/model_input/targets.npy',mmap_mode='r')
    mask = np.load(root/'data/model_input/target_mask.npy',mmap_mode='r')
    panel = pd.read_csv(root/'data/model_input/phospho_target_table.tsv',sep='\t')
    frames, ranks, arrays = [], [], []
    for di, ds in enumerate(DATASETS):
        m = meta[meta.dataset_id.eq(ds)].copy()
        a = ad.read_h5ad(root/'data/h5ad'/f'{ds}.h5ad')
        assert a.obs_names.is_unique and a.var_names.is_unique
        idx = a.obs_names.get_indexer(m.cell_id.astype(str))
        assert (idx >= 0).all()
        x = normalize_log1p(sparse.csr_matrix(a.X)[idx])
        assert np.isfinite(x.data).all()
        split = np.full(len(m),'optimization',dtype='<U12')
        manifest = pd.read_csv(Path(__file__).resolve().parents[1]/'splits.tsv.gz',sep='\t')
        manifest = manifest[manifest.dataset.eq(ds)].set_index('cell_id')
        split = manifest.loc[m.cell_id.astype(str),'split'].to_numpy(str)
        opt = np.flatnonzero(split=='optimization')
        mu = np.asarray(x[opt].mean(0)).ravel()
        var = np.maximum(np.asarray(x[opt].power(2).mean(0)).ravel()-mu**2,0)
        ranks.append(a.var_names[np.argsort(-var,kind='stable')].tolist())
        arrays.append((x,a.var_names,m,split))
        frames.append(pd.DataFrame({'dataset':ds,'cell_id':m.cell_id.to_numpy(),'global_index':m.index,'split':split}))
    genes=[]; seen=set()
    for items in zip(*ranks):
        for g in items:
            if g not in seen:
                seen.add(g); genes.append(g)
                if len(genes)==4000: break
        if len(genes)==4000: break
    target_map=panel[panel.dataset_id.isin(DATASETS)][['target_index','target_id']].drop_duplicates().sort_values('target_index')
    assert target_map.target_index.is_unique
    cols=target_map.target_index.to_numpy(int)
    all_means=[]; all_second=[]; mapped=[]
    for x, names, m, split in arrays:
        positions=names.get_indexer(genes); good=positions>=0
        mapper=sparse.csr_matrix((np.ones(good.sum()),(positions[good],np.flatnonzero(good))),shape=(len(names),len(genes)))
        xx=(x@mapper).astype('float32').tocsr(); mapped.append(xx)
        opt=split=='optimization'
        all_means.append(np.asarray(xx[opt].mean(0)).ravel()); all_second.append(np.asarray(xx[opt].power(2).mean(0)).ravel())
    gm=np.mean(all_means,axis=0); gs=np.sqrt(np.maximum(np.mean(all_second,axis=0)-gm**2,1e-4))
    counts=[]
    for di,((_,names,m,split),xx) in enumerate(zip(arrays,mapped)):
        yy=np.asarray(y[np.ix_(m.index,cols)],dtype='float32'); mm=np.asarray(mask[np.ix_(m.index,cols)],bool)&np.isfinite(yy)
        if di in (1, 2):
            eligible = panel.loc[panel.dataset_id.eq(DATASETS[di]) & panel.include_in_loss.astype(str).str.lower().eq('true'), 'target_index'].to_numpy(int)
            mm[:, ~np.isin(cols, eligible)] = False
        expected = [20, 12, 28][di]
        if int(mm.any(0).sum()) != expected:
            raise ValueError(f'{DATASETS[di]}: expected {expected} supervised readouts, got {int(mm.any(0).sum())}')
        opt=split=='optimization'; train_mask=mm & opt[:,None]
        tm=np.zeros(len(cols),np.float32); ts=np.ones(len(cols),np.float32); tau=np.zeros(len(cols),np.float32); rate=np.zeros(len(cols),np.float32)
        for j in range(len(cols)):
            vals=yy[train_mask[:,j],j]
            if not len(vals): mm[:,j]=False; continue
            tm[j]=vals.mean(); ts[j]=max(vals.std(),1e-4)
            positive=vals[vals>0]; tau[j]=np.quantile(positive,.1) if len(positive) else 0
            rate[j]=np.mean(vals<=tau[j]*.5)
        sparse.save_npz(out/f'data_{di}.npz',xx)
        np.savez(out/f'labels_{di}.npz',y=yy,mask=mm,split=split,global_rows=m.index.to_numpy(),mean=tm,std=ts,tau=tau,rate=rate)
        counts.append({'dataset':DATASETS[di],'cells':len(m),'readouts':int(mm.any(0).sum()),'split_counts':pd.Series(split).value_counts().to_dict()})
    pd.concat(frames).to_csv(out/'splits.tsv',sep='\t',index=False)
    target_map.to_csv(out/'readouts.tsv',sep='\t',index=False)
    np.savez(out/'preprocessing.npz',genes=np.asarray(genes),gene_mean=gm,gene_std=gs)
    write_json(out/'contract.json',{'seed':682,'datasets':counts,'hvg':'equal_dataset_round_robin_training_variance','gene_identity':'exact_case_species_namespaced; no_uppercase_orthology_substitution','checkpoint':'mean_dataset_median_validation_spearman','epochs':40,'steps_per_epoch':300,'batch_per_dataset':64})
    (out/'DATA_READY').touch()
    print(json.dumps(counts),flush=True)

def objective(pred, logits, raw, mask, mean, std, tau, rate, variant):
    safe=torch.where(mask,raw,mean)
    target=(safe-mean)/std
    active=safe>tau; neutral=safe<=tau*.5
    weight=torch.ones_like(safe)
    if variant!='U0':
        nw=torch.where(rate>=.3,.2,torch.where(rate>=.1,.5,1.))
        weight=torch.where(neutral,nw,torch.where(active,1.,torch.where(rate>=.1,.05,1.)))
    weight=weight*mask
    observed=mask.any(0)
    per_site=(F.smooth_l1_loss(pred,target,reduction='none')*weight).sum(0)/weight.sum(0).clamp_min(1)
    loss=per_site[observed].mean()
    if variant=='U2':
        valid=mask&(active|neutral)&(rate>=.3)
        use=valid.any(0)
        if use.any():
            aux=(F.binary_cross_entropy_with_logits(logits,active.float(),reduction='none')*valid).sum(0)/valid.sum(0).clamp_min(1)
            loss=loss+.1*aux[use].mean()
    return loss

def train(root,out,variant,smoke=False,device='cuda:0'):
    torch.manual_seed(682); np.random.seed(682); torch.set_num_threads(4)
    dest=out/('smoke' if smoke else 'formal')/variant; dest.mkdir(parents=True,exist_ok=True)
    if (dest/'SUCCESS').exists(): return
    prep=np.load(out/'preprocessing.npz'); readouts=pd.read_csv(out/'readouts.tsv',sep='\t')
    emb=np.load(root/'data/model_input/embeddings.npy',mmap_mode='r')
    data=[]
    for i in range(3):
        labels=dict(np.load(out/f'labels_{i}.npz'))
        data.append((sparse.load_npz(out/f'data_{i}.npz'),labels))
    model=Unified(4000,emb.shape[1],len(readouts)).to(device)
    if device.startswith('cuda'): torch.cuda.set_per_process_memory_fraction(.68)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    rng=np.random.default_rng(682); best=-np.inf; stale=0; history=[]
    def batch(i,rows):
        x,l=data[i]
        gx=(x[rows].toarray()-prep['gene_mean'])/prep['gene_std']
        ee=np.asarray(emb[l['global_rows'][rows]],dtype='float32')
        assert np.isfinite(gx).all() and np.isfinite(ee).all()
        return torch.tensor(gx,device=device),torch.tensor(ee,device=device)
    for epoch in range(1,(1 if smoke else 40)+1):
        model.train(); losses=[]
        for step in range(1 if smoke else 300):
            if device.startswith('cuda') and step%10==0:
                while True:
                    status=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]
                    temp,used,total=map(float,status.split(','))
                    if temp<76 and used/total<.68: break
                    print('WAIT_GPU',status,flush=True); time.sleep(20)
            optimizer.zero_grad(); total_loss=0
            for i,(x,l) in enumerate(data):
                opt=np.flatnonzero(l['split']=='optimization'); rows=rng.choice(opt,size=4 if smoke else 64)
                xx,ee=batch(i,rows); domain=torch.full((len(rows),),i+1,device=device,dtype=torch.long)
                domain[torch.rand(len(rows),device=device)<.2]=0
                pred,logits=model(xx,ee,domain)
                vals={k:torch.tensor(l[k],device=device) for k in ['mean','std','tau','rate']}
                loss=objective(pred,logits,torch.tensor(l['y'][rows],device=device),torch.tensor(l['mask'][rows],device=device),**vals,variant=variant)/3
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                loss.backward(); total_loss+=loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); losses.append(total_loss)
        model.eval(); metrics=[]
        with torch.inference_mode():
            for i,(x,l) in enumerate(data):
                rows=np.flatnonzero(l['split']=='validation')
                if smoke: rows=rows[:12]
                pieces=[]
                for start in range(0,len(rows),128):
                    take=rows[start:start+128]; xx,ee=batch(i,take)
                    p,_=model(xx,ee,torch.full((len(take),),i+1,device=device,dtype=torch.long))
                    pieces.append(p.cpu().numpy()*l['std']+l['mean'])
                pp=np.vstack(pieces)
                for j,t in enumerate(readouts.target_id):
                    good=l['mask'][rows,j]&np.isfinite(l['y'][rows,j])
                    y=l['y'][rows,j][good]; p=pp[good,j]
                    if len(y)<3 or np.std(y)==0 or np.std(p)==0: continue
                    metrics.append({'dataset':DATASETS[i],'target':t,'spearman':spearmanr(y,p).statistic,'pearson':pearsonr(y,p).statistic})
        table=pd.DataFrame(metrics); score=table.groupby('dataset').spearman.median().mean()
        history.append({'epoch':epoch,'loss':np.mean(losses),'validation_score':score})
        pd.DataFrame(history).to_csv(dest/'history.tsv',sep='\t',index=False)
        print(variant,history[-1],flush=True)
        if score>best:
            best=score; stale=0
            torch.save({'state_dict':model.state_dict(),'variant':variant,'epoch':epoch},dest/'best.pt')
            table.to_csv(dest/'validation_per_site.tsv',sep='\t',index=False)
        else: stale+=1
        if epoch>=10 and stale>=8: break
    write_json(dest/'summary.json',{'variant':variant,'validation_score':best,'epochs':epoch,'test_evaluated':False})
    (dest/'SUCCESS').touch()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['prepare','train']); p.add_argument('--root',type=Path,required=True); p.add_argument('--variant',default='U1',choices=['U0','U1','U2']); p.add_argument('--smoke',action='store_true'); p.add_argument('--device',default='cuda:0'); args=p.parse_args()
    out=args.root/'results/scp682_u1'; out.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare': prepare(args.root,out)
    else: train(args.root,out,args.variant,args.smoke,args.device)
