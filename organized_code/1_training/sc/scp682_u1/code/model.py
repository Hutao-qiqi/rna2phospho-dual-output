"""SCP682 unified single-cell architecture (U1); checkpoint-compatible module names."""
import numpy as np
from scipy import sparse
import torch
from torch import nn

def normalize_log1p(matrix):
    matrix=matrix.tocsr().astype(np.float32)
    totals=np.asarray(matrix.sum(1)).ravel()
    matrix=sparse.diags((1e4/np.maximum(totals,1)).astype(np.float32))@matrix
    matrix.data=np.log1p(matrix.data)
    return matrix.tocsr()

class ResidualBlock(nn.Module):
    def __init__(self,width,dropout=.1):
        super().__init__();self.norm=nn.LayerNorm(width)
        self.net=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Dropout(dropout),nn.Linear(width*2,width),nn.Dropout(dropout))
    def forward(self,x):return x+self.net(self.norm(x))

class DualInputEncoder(nn.Module):
    def __init__(self,ng,ne,ns):
        super().__init__()
        self.hvg=nn.Sequential(nn.Linear(ng,2048),nn.LayerNorm(2048),nn.GELU(),ResidualBlock(2048),
            nn.Linear(2048,1024),nn.LayerNorm(1024),nn.GELU(),ResidualBlock(1024),
            nn.Linear(1024,512),nn.LayerNorm(512),nn.GELU(),ResidualBlock(512))
        self.embedding=nn.Sequential(nn.Linear(ne,512),nn.LayerNorm(512),nn.GELU(),ResidualBlock(512))
        self.fusion=nn.Sequential(nn.Linear(1024,512),nn.LayerNorm(512),nn.GELU(),ResidualBlock(512))
        self.shared_head=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,ns))
    def forward(self,x,e):
        h=self.fusion(torch.cat([self.hvg(x),self.embedding(e)],dim=1))
        return self.shared_head(h),h

class SCP682(nn.Module):
    def __init__(self,ng=4000,ne=3072,ns=52):
        super().__init__();self.encoder=DualInputEncoder(ng,ne,ns)
        self.site=nn.Embedding(ns,32);self.domain=nn.Embedding(4,32,padding_idx=0)
        self.condition=nn.Linear(32,512)
        nn.init.zeros_(self.condition.weight);nn.init.zeros_(self.condition.bias)
        self.query=nn.Sequential(nn.Linear(544,128),nn.GELU(),nn.Linear(128,1))
        self.detect=nn.Linear(512,ns)
        self.scale=nn.Embedding(4,ns,padding_idx=0);self.shift=nn.Embedding(4,ns,padding_idx=0)
        nn.init.zeros_(self.scale.weight);nn.init.zeros_(self.shift.weight)
    def forward(self,x,e,d):
        base,h=self.encoder(x,e);h=h+self.condition(self.domain(d))
        q=self.site.weight[None].expand(len(h),-1,-1)
        z=torch.cat([h[:,None].expand(-1,len(self.site.weight),-1),q],dim=-1)
        pred=base+self.query(z).squeeze(-1)
        return pred*torch.exp(.2*torch.tanh(self.scale(d)))+self.shift(d),self.detect(h)
