"""Mean protein CLS fusion + explicit protein/substrate interaction MLP."""
from __future__ import annotations
import torch
from torch import nn
from kcat_models import MeanCLSTransformer, MeanInteractionMLP


class MeanCLSHybrid(nn.Module):
    """Concat normalized final CLS (256) and [p,s,p*s,abs(p-s)] (2048).

    Inputs are protein mean [B,Dp] and UniKP substrate [B,Ds]. This is a
    jointly trained feature-fusion model, not an average of two predictions.
    Frozen pretrained embeddings are shared; predictor branch weights are
    independent and trained from scratch. No full-residue inputs are used.
    """
    def __init__(self, *, prot_dim: int, smiles_dim: int = 1024, dropout: float = .1):
        super().__init__()
        self.cls_encoder=MeanCLSTransformer(prot_dim=prot_dim,smiles_dim=smiles_dim,
            d_model=256,num_heads=4,num_layers=4,dim_feedforward=1024,dropout=dropout)
        # Keep a normalized CLS feature, removing the old scalar regression head.
        self.cls_encoder.regressor=nn.LayerNorm(256)
        self.interaction_encoder=MeanInteractionMLP(prot_dim=prot_dim,smiles_dim=smiles_dim,
                                                   global_dim=512,dropout=dropout)
        self.interaction_encoder.regressor=nn.Identity()
        self.regressor=nn.Sequential(nn.LayerNorm(2304),nn.Linear(2304,512),
            nn.SiLU(),nn.Dropout(dropout),nn.Linear(512,128),nn.SiLU(),
            nn.LayerNorm(128),nn.Dropout(dropout),nn.Linear(128,1))
        self._config=dict(architecture='mean_cls_mlp_hybrid',prot_dim=prot_dim,
                          smiles_dim=smiles_dim,dropout=dropout)

    def config_dict(self):
        return dict(self._config)

    def forward(self, protein_embs, smiles_embs, *, return_aux=False):
        cls=self.cls_encoder(protein_embs,smiles_embs)
        interactions=self.interaction_encoder(protein_embs,smiles_embs)
        joint=torch.cat([cls,interactions],dim=-1)
        prediction=self.regressor(joint)
        if return_aux:
            return prediction,dict(cls_state=cls,interaction_features=interactions,joint=joint)
        return prediction


def smoke_test():
    from kcat_models import build_predictor
    torch.set_num_threads(1); torch.manual_seed(42)
    model=MeanCLSHybrid(prot_dim=1280).eval()
    h=torch.randn(4,1280,requires_grad=True); s=torch.randn(4,1024,requires_grad=True)
    y,aux=model(h,s,return_aux=True)
    assert y.shape==(4,1) and torch.isfinite(y).all()
    assert aux['cls_state'].shape==(4,256) and aux['interaction_features'].shape==(4,2048)
    assert aux['joint'].shape==(4,2304)
    assert not any(isinstance(m,nn.Linear) for m in model.cls_encoder.regressor.modules())
    assert isinstance(model.interaction_encoder.regressor,nn.Identity)
    perm=torch.tensor([3,1,0,2])
    torch.testing.assert_close(y[perm],model(h[perm],s[perm]),atol=2e-5,rtol=2e-5)
    for i in range(4):
        torch.testing.assert_close(y[i:i+1],model(h[i:i+1],s[i:i+1]),atol=2e-5,rtol=2e-5)
    aux['cls_state'].retain_grad(); aux['interaction_features'].retain_grad()
    y.square().mean().backward()
    for x in [h,s,aux['cls_state'],aux['interaction_features']]:
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert model.cls_encoder.cls_token.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    restored=build_predictor(**model.config_dict()).eval(); restored.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(y,restored(h,s))
    return dict(tests='passed',trainable_parameters=sum(p.numel() for p in model.parameters()),
                cls_dim=256,interaction_dim=2048,fused_dim=2304,device='cpu',torch=torch.__version__)


if __name__=='__main__':
    import json
    print(json.dumps(smoke_test(),indent=2))
