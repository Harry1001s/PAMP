"""Frozen CLS, interaction, and compact-head ensemble."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from kcat_models import build_predictor
from compact_kcat_models import load_compact_checkpoint

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

class ImprovedKcatEnsemble(nn.Module):
    def __init__(self, old_manifest, compact_checkpoints, *, source_sha256, split_sha256, device='cpu'):
        super().__init__()
        if len(compact_checkpoints) != 3:
            raise ValueError('Exactly the three prespecified seeds are required')
        self.old = MeanKcatEnsemble.from_manifest(old_manifest)
        old_spec = json.loads(Path(old_manifest).read_text())
        old_ck = torch.load(old_spec['checkpoints'][0]['path'], map_location='cpu', weights_only=False)
        if old_ck['data_sha256'] != source_sha256 or old_ck['split_sha256'] != split_sha256:
            raise ValueError('Old predictors have a different data source or split')
        heads, seeds = [], []
        for path in compact_checkpoints:
            head, ck = load_compact_checkpoint(path)
            p = ck['contract']['provenance']
            if p['source_sha256'] != source_sha256 or p['split_sha256'] != split_sha256:
                raise ValueError('All components must use the same source and split')
            if head.config_dict()['prot_dim'] != 1280:
                raise ValueError('This ensemble preserves the ESM2 1280-dimensional space')
            seeds.append(ck['contract']['training']['seed'])
            heads.append(head)
        if sorted(seeds) != [42, 2024, 3407]:
            raise ValueError('Wrong seeds')
        if any(h.config_dict() != heads[0].config_dict() for h in heads[1:]):
            raise ValueError('The new heads must share an architecture')
        for h in heads[1:]:
            torch.testing.assert_close(h.protein_scale, heads[0].protein_scale, rtol=0, atol=0)
        self.heads = nn.ModuleList(heads)
        self.fingerprint_dim = heads[0].config_dict()['fingerprint_dim']
        self.register_buffer('protein_scale', heads[0].protein_scale.clone())
        self.requires_grad_(False)
        self.to(device).eval()

    @classmethod
    def from_manifest(cls, manifest_path, device='cpu'):
        spec = json.loads(Path(manifest_path).read_text())
        if spec['family_weights'] != [1/3, 1/3, 1/3] or spec['new_seed_weights'] != [1/3, 1/3, 1/3]:
            raise ValueError('Only the prespecified equal weighting is supported')
        paths = []
        for record in [spec['old_manifest']] + spec['new_checkpoints']:
            path = Path(record['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('Manifest/checkpoint integrity mismatch: ' + str(path))
            paths.append(path)
        return cls(paths[0], paths[1:], source_sha256=spec['source_sha256'],
                   split_sha256=spec['split_sha256'], device=device)

    def forward(self, protein_mean, substrate, fingerprint=None):
        if not self.fingerprint_dim and fingerprint is not None:
            raise ValueError('This selected ensemble uses no MACCS features')
        new = torch.stack([m.predict_log2(protein_mean, substrate, fingerprint) for m in self.heads]).mean(0)
        # Old branch is already the 0.5/0.5 average of CLS and interaction MLP.
        return (2 * self.old(protein_mean, substrate) + new) / 3

    def predict_log2(self, protein_mean, substrate, fingerprint=None):
        return self(protein_mean, substrate, fingerprint)
