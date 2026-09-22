"""Official TabM heads, with scalers inside the differentiable predictor."""
from common import *
import torch
from torch import nn
from tabm import TabM


class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.kind = config['kind']
        for name, dim in [('protein',1280), ('substrate',1024)]:
            self.register_buffer(name+'_mean',torch.zeros(dim))
            self.register_buffer(name+'_scale',torch.ones(dim))
        self.register_buffer('target_mean',torch.tensor(0.))
        self.register_buffer('target_scale',torch.tensor(1.))
        if self.kind == 'condpool':
            d, a = config['d_model'], config['attention_hidden_dim']
            self.protein_projection = nn.Linear(1280,d)
            self.substrate_projection = nn.Linear(1024,d)
            self.score_protein = nn.Linear(d,a, bias=False)
            self.score_substrate = nn.Linear(d,a)
            self.score_out = nn.Linear(a,1,bias=False)
            self.dropout = nn.Dropout(config['dropout'])
            dim = d * 3
        else:
            assert self.kind == 'mean'
            dim = 2304
        self.head = TabM.make(n_num_features=dim, d_out=1, k=config['k'],
            n_blocks=config['n_blocks'],d_block=config['d_block'],dropout=config['dropout'],arch_type='tabm')

    def set_scalers(self, protein, substrate, y, ids):
        with torch.no_grad():
            for name, x in [('protein',protein), ('substrate',substrate)]:
                getattr(self,name+'_mean').copy_(torch.as_tensor(x[ids].mean(0)))
                getattr(self,name+'_scale').copy_(torch.as_tensor(x[ids].std(0).clip(1e-6)))
            self.target_mean.fill_(float(y[ids].mean()))
            self.target_scale.fill_(float(y[ids].std()))

    def forward(self, protein, substrate, mask=None, return_attention=False):
        protein = (protein.float()-self.protein_mean)/self.protein_scale
        substrate = (substrate.float()-self.substrate_mean)/self.substrate_scale
        alpha = None
        if self.kind == 'mean':
            z = torch.cat([protein,substrate],dim=-1)
        else:
            if mask is None or mask.shape != protein.shape[:2] or not mask.any(1).all():
                raise ValueError('Every sequence requires a nonempty valid residue mask')
            p = self.protein_projection(protein)
            q = self.substrate_projection(substrate)
            score = self.score_out(torch.tanh(self.score_protein(p)+self.score_substrate(q)[:,None])).squeeze(-1).float()
            alpha = score.masked_fill(~mask, -torch.inf).softmax(1)
            local = (alpha[:,:,None]*p.float()).sum(1)
            global_mean = (p.float()*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
            z = self.dropout(torch.cat([global_mean,local,q.float()],dim=-1))
        members = self.head(z).squeeze(-1)
        assert members.ndim == 2 and members.shape[1] == self.config['k']
        return (members,alpha) if return_attention else members

    def predict_log2(self, protein, substrate, mask=None):
        return self(protein,substrate,mask).float().mean(1)*self.target_scale+self.target_mean


def load_predictor(checkpoint, device='cpu'):
    ck = torch.load(checkpoint,map_location='cpu',weights_only=False)
    model = Predictor(ck['config'])
    model.load_state_dict(ck['state_dict'])
    return model.to(device).eval()
