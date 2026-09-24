"""Compact differentiable mean-embedding predictors with in-graph scaling."""
from __future__ import annotations
import torch
from torch import nn


class CompactKcatPredictor(nn.Module):
    def __init__(self, *, prot_dim, smiles_dim=1024, fingerprint_dim=0, width=256,
                 dropout=.1, input_normalization='standard'):
        super().__init__()
        if input_normalization not in ['standard','layernorm']:
            raise ValueError('Unknown input normalization')
        self._config=dict(prot_dim=prot_dim,smiles_dim=smiles_dim,fingerprint_dim=fingerprint_dim,
                          width=width,dropout=dropout,input_normalization=input_normalization)
        self.register_buffer('protein_mean',torch.zeros(prot_dim))
        self.register_buffer('protein_scale',torch.ones(prot_dim))
        self.register_buffer('smiles_mean',torch.zeros(smiles_dim))
        self.register_buffer('smiles_scale',torch.ones(smiles_dim))
        self.register_buffer('label_mean',torch.tensor(0.))
        self.register_buffer('label_scale',torch.tensor(1.))
        self.protein_norm=nn.LayerNorm(prot_dim) if input_normalization=='layernorm' else nn.Identity()
        self.smiles_norm=nn.LayerNorm(smiles_dim) if input_normalization=='layernorm' else nn.Identity()
        self.regressor=nn.Sequential(nn.Linear(prot_dim+smiles_dim+fingerprint_dim,width),
            nn.BatchNorm1d(width),nn.ReLU(),nn.Dropout(dropout),nn.Linear(width,1))

    def config_dict(self):
        return dict(self._config)

    def set_scalers(self, protein_mean,protein_scale,smiles_mean,smiles_scale,label_mean,label_scale):
        with torch.no_grad():
            for name,value in dict(protein_mean=protein_mean,protein_scale=protein_scale,
                    smiles_mean=smiles_mean,smiles_scale=smiles_scale,
                    label_mean=label_mean,label_scale=label_scale).items():
                target=getattr(self,name); target.copy_(torch.as_tensor(value,device=target.device,dtype=target.dtype))
        assert (self.protein_scale>0).all() and (self.smiles_scale>0).all() and self.label_scale>0

    def forward(self, protein, smiles, fingerprint=None):
        c=self._config
        if protein.ndim!=2 or protein.shape[1]!=c['prot_dim'] or protein.shape[0]==0:
            raise ValueError('Protein input must be nonempty [B,Dp] mean embeddings')
        if smiles.shape!=(len(protein),c['smiles_dim']) or smiles.device!=protein.device:
            raise ValueError('Substrate dimensions/device do not match')
        if not protein.is_floating_point() or not smiles.is_floating_point():
            raise TypeError('Embedding inputs must be floating point')
        p=protein.to(self.protein_mean.dtype); s=smiles.to(self.smiles_mean.dtype)
        if c['input_normalization']=='standard':
            p=(p-self.protein_mean)/self.protein_scale; s=(s-self.smiles_mean)/self.smiles_scale
        else:
            p=self.protein_norm(p); s=self.smiles_norm(s)
        parts=[p,s]
        if c['fingerprint_dim']:
            if fingerprint is None or fingerprint.shape!=(len(p),c['fingerprint_dim']) or fingerprint.device!=p.device:
                raise ValueError('Aligned fingerprint input required')
            parts.append(fingerprint.to(p.dtype))  # Binary bits are not standardized.
        elif fingerprint is not None:
            raise ValueError('This model was configured without fingerprints')
        return self.regressor(torch.cat(parts,dim=-1))

    def predict_log2(self, protein, smiles, fingerprint=None):
        return self(protein,smiles,fingerprint).float()*self.label_scale+self.label_mean


def build_compact_predictor(config):
    return CompactKcatPredictor(**config)


def load_compact_checkpoint(path, device='cpu', freeze=True):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=build_compact_predictor(ck['model_config'])
    model.load_state_dict(ck['state_dict'],strict=True); model.to(device).eval()
    if freeze: model.requires_grad_(False)
    return model,ck
