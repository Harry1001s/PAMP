"""Differentiable, fixed-weight ensemble of existing mean kcat predictors.

forward() returns unstandardized log2(kcat), shape [B,1]. All model parameters
are frozen; gradients with respect to mean protein inputs remain available.
"""
import hashlib,json
from pathlib import Path
import torch
from torch import nn
from kcat_models import build_predictor


class MeanKcatEnsemble(nn.Module):
    def __init__(self, checkpoints, weights=(.5,.5), device='cpu'):
        super().__init__()
        if len(checkpoints)!=len(weights) or len(checkpoints)==0:
            raise ValueError('One weight is required per checkpoint')
        weights=torch.tensor(weights,dtype=torch.float32)
        if not torch.isfinite(weights).all() or (weights<0).any() or not torch.isclose(weights.sum(),torch.tensor(1.)):
            raise ValueError('Finite nonnegative weights must sum to one')
        models=[]; means=[]; scales=[]; splits=[]; sources=[]
        for path in checkpoints:
            ck=torch.load(path,map_location='cpu',weights_only=False)
            if ck['model_config']['architecture'] not in ['mean_cls_transformer','mean_interaction_mlp']:
                raise ValueError('Unexpected architecture')
            m=build_predictor(**ck['model_config']); m.load_state_dict(ck['state_dict'],strict=True)
            m.requires_grad_(False); models.append(m)
            means.append(ck['mu']); scales.append(ck['sigma']); splits.append(ck['split_sha256']); sources.append(ck['data_sha256'])
        if len(set(splits))!=1 or len(set(sources))!=1:
            raise ValueError('Ensemble checkpoints must share source data and split')
        self.models=nn.ModuleList(models)
        self.register_buffer('weights',weights)
        self.register_buffer('label_means',torch.tensor(means,dtype=torch.float32))
        self.register_buffer('label_scales',torch.tensor(scales,dtype=torch.float32))
        self.to(device); self.eval()

    @classmethod
    def from_manifest(cls,path,device='cpu'):
        path=Path(path); spec=json.loads(path.read_text()); checkpoints=[]
        for record in spec['checkpoints']:
            p=Path(record['path']); p=p if p.is_absolute() else path.parent/p
            if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError(f'Checkpoint fingerprint mismatch: {p}')
            checkpoints.append(p)
        return cls(checkpoints,spec['weights'],device)

    def forward(self, protein_mean, substrate):
        parts=torch.stack([m(protein_mean,substrate).float() for m in self.models],dim=-1)
        return ((parts*self.label_scales+self.label_means)*self.weights).sum(-1)
