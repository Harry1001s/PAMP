"""Frozen differentiable output average, retaining the existing ESM2 space."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from compact_kcat_models import load_compact_checkpoint
from kcat_prediction_ensemble import MeanKcatEnsemble


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


class ImprovedKcatAttackAdapter:
    """Single-protein pred/input_gradient interface for the existing ESM2 attacks.

    H is [L,1280] in the original ESM2 space. Pooling is differentiable; the
    prediction head gives identical gradients to each valid residue, scaled 1/L.
    The substrate embedding and optional MACCS fingerprint are held fixed.
    """
    def __init__(self, manifest_path, device='cuda', fingerprint=None):
        self.device = torch.device(device)
        self.model = ImprovedKcatEnsemble.from_manifest(manifest_path, device=device)
        self.fingerprint = None
        if self.model.fingerprint_dim:
            if fingerprint is None:
                raise ValueError('Supply the fixed substrate MACCS167 fingerprint for this model')
            bits = torch.as_tensor(fingerprint, device=self.device, dtype=torch.float32).reshape(1, -1)
            if bits.shape != (1, self.model.fingerprint_dim) or not ((bits == 0) | (bits == 1)).all():
                raise ValueError('Expected a binary MACCS167 vector')
            self.fingerprint = bits.detach().clone()
        elif fingerprint is not None:
            raise ValueError('This model does not use MACCS')

    def pred(self, H, smiles_embedding, mask=None):
        H = H.to(self.device, dtype=torch.float32)
        if H.ndim != 2 or H.shape[1] != 1280:
            raise ValueError('Expected original ESM2 residues [L,1280]')
        if mask is None:
            mask = torch.ones(len(H), dtype=torch.bool, device=self.device)
        else:
            mask = mask.to(self.device)
        if mask.shape != (len(H),) or mask.dtype != torch.bool or not mask.any():
            raise ValueError('Expected a nonempty boolean residue mask')
        mean = H.masked_fill(~mask[:, None], 0).sum(0, keepdim=True) / mask.sum()
        s = smiles_embedding.to(self.device, dtype=torch.float32).reshape(1, -1)
        return self.model(mean, s, self.fingerprint).reshape(())

    def input_gradient(self, H0, smiles_embedding, mask=None):
        H = H0.detach().to(self.device, dtype=torch.float32).clone().requires_grad_(True)
        prediction = self.pred(H, smiles_embedding, mask)
        gradient = torch.autograd.grad(prediction, H)[0]
        return prediction.detach(), gradient.detach()
