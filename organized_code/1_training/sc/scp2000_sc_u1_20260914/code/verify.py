"""Verify the released checkpoint against fixed synthetic reference predictions."""
from pathlib import Path
import argparse
import numpy as np
import torch
from model import SCP2000sc

p=argparse.ArgumentParser();p.add_argument('--release-dir',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
torch.set_num_threads(4)
z=np.load(a.release_dir/'verification_inputs.npz')
model=SCP2000sc(ns=z['expected'].shape[1])
model.load_state_dict(torch.load(a.release_dir/'SCP2000_sc_U1.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
model.eval()
with torch.inference_mode():
    pred=model(torch.tensor(z['x']),torch.tensor(z['e']),torch.tensor(z['domain']))[0].numpy()
assert np.allclose(pred,z['expected'],rtol=1e-5,atol=1e-5)
print('PASS; max absolute error:',float(np.max(abs(pred-z['expected']))))
