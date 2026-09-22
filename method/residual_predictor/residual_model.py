"""Frozen original predictor + zero-initialized gated residue correction."""
import sys as _sys, pathlib as _pl
for _c in _pl.Path(__file__).resolve().parents:
    if (_c / 'pamp_paths.py').exists():
        _sys.path.insert(0, str(_c)); break
from pamp_paths import (add_module_paths, RC_TABM_SCRIPTS, TABM_VENDOR,
                        ORIGINAL_PREDICTOR_DIR)
import sys
from pathlib import Path

add_module_paths(RC_TABM_SCRIPTS, TABM_VENDOR, ORIGINAL_PREDICTOR_DIR)
from common import ROOT,BASE,sha,dump,load_data,metrics
import torch
from torch import nn
from tabm import TabM
from improved_kcat_ensemble import ImprovedKcatEnsemble


class LocalCorrection(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=dict(config)
        for name,dim in [('protein',1280),('substrate',1024)]:
            self.register_buffer(name+'_mean',torch.zeros(dim))
            self.register_buffer(name+'_scale',torch.ones(dim))
        self.register_buffer('target_scale',torch.tensor(1.))
        self.gamma=nn.Parameter(torch.tensor(float(config['gamma_init'])))
        d,a=config['d_model'],config['attention_hidden_dim']
        self.protein_projection=nn.Linear(1280,d)
        self.substrate_projection=nn.Linear(1024,d)
        self.score_protein=nn.Linear(d,a,bias=False)
        self.score_substrate=nn.Linear(d,a)
        self.score_out=nn.Linear(a,1,bias=False)
        self.dropout=nn.Dropout(config['dropout'])
        self.tabm=TabM.make(n_num_features=2*d,d_out=1,k=config['k'],n_blocks=config['n_blocks'],
            d_block=config['d_block'],dropout=config['dropout'],arch_type='tabm')

    def set_scalers(self,p,s,y,train_ids):
        with torch.no_grad():
            for name,value in [('protein',p),('substrate',s)]:
                getattr(self,name+'_mean').copy_(torch.as_tensor(value[train_ids].mean(0)))
                getattr(self,name+'_scale').copy_(torch.as_tensor(value[train_ids].std(0).clip(1e-6)))
            self.target_scale.fill_(float(y[train_ids].std()))

    def forward(self,H,s,mask,return_attention=False):
        if mask.dtype!=torch.bool or mask.shape!=H.shape[:2] or not mask.any(1).all():
            raise ValueError('Each sample requires nonempty valid residue mask')
        p=self.protein_projection((H.float()-self.protein_mean)/self.protein_scale)
        q=self.substrate_projection((s.float()-self.substrate_mean)/self.substrate_scale)
        score=self.score_out(torch.tanh(self.score_protein(p)+self.score_substrate(q)[:,None])).squeeze(-1).float()
        alpha=score.masked_fill(~mask,-torch.inf).softmax(1)
        local=(alpha[:,:,None]*p.float()).sum(1)
        # No replacement of the original global representation; only local+substrate.
        members=self.tabm(self.dropout(torch.cat([local,q.float()],dim=-1))).squeeze(-1).float()
        correction=self.gamma*members*self.target_scale
        return (correction,alpha) if return_attention else correction


class ResidualPredictor(nn.Module):
    def __init__(self,config,global_manifest=None):
        super().__init__()
        self.global_manifest=Path(global_manifest) if global_manifest else BASE/'model_manifest.json'
        self.global_branch=ImprovedKcatEnsemble.from_manifest(self.global_manifest,device='cpu').requires_grad_(False)
        self.local_branch=LocalCorrection(config)
        self.global_branch.eval()

    def train(self,mode=True):
        super().train(mode)
        self.global_branch.eval()
        return self

    def forward(self,h_mean,H,s,mask):
        # Keep the mean encoder's existing float32 features as a distinct input.
        with torch.autocast(device_type=h_mean.device.type,enabled=False):
            global_y=self.global_branch(h_mean.float(),s.float()).flatten()
        return global_y[:,None]+self.local_branch(H,s,mask)

    def predict_log2(self,h_mean,H,s,mask):
        return self(h_mean,H,s,mask).mean(1)


def load_predictor(checkpoint,device='cpu'):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert sha(ck['global_manifest'])==ck['global_manifest_sha256']
    torch.backends.mha.set_fastpath_enabled(False)
    model=ResidualPredictor(ck['config'],ck['global_manifest'])
    model.local_branch.load_state_dict(ck['state_dict'])
    return model.to(device).eval()
