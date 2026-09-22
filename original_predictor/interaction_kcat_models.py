"""Small residual low-rank protein/substrate interactions on fixed mean embeddings."""
import math
import torch
from torch import nn
from compact_kcat_models import CompactKcatPredictor


class LowRankResidualKcat(CompactKcatPredictor):
    def __init__(self, *, prot_dim=1280, smiles_dim=1024, fingerprint_dim=0, width=128,
                 dropout=.1, input_normalization='standard', rank=64, pair_dropout=.1,
                 initial_pair_gate=.1, architecture='low_rank_residual'):
        if rank < 2 or not 0 < initial_pair_gate < 1 or architecture != 'low_rank_residual':
            raise ValueError('Invalid low-rank configuration')
        super().__init__(prot_dim=prot_dim, smiles_dim=smiles_dim, fingerprint_dim=fingerprint_dim,
                         width=width, dropout=dropout, input_normalization=input_normalization)
        self._config.update(rank=rank, pair_dropout=pair_dropout, initial_pair_gate=initial_pair_gate,
                            architecture=architecture)
        self.pair_protein = nn.Sequential(nn.Linear(prot_dim, rank, bias=False), nn.LayerNorm(rank))
        self.pair_substrate = nn.Sequential(nn.Linear(smiles_dim + fingerprint_dim, rank, bias=False), nn.LayerNorm(rank))
        self.pair_head = nn.Sequential(nn.Linear(rank, 64), nn.SiLU(), nn.Dropout(pair_dropout), nn.Linear(64, 1))
        self.pair_gate_logit = nn.Parameter(torch.tensor(math.log(initial_pair_gate / (1 - initial_pair_gate))))

    def forward(self, protein, smiles, fingerprint=None):
        base = super().forward(protein, smiles, fingerprint)
        p, s = protein.to(self.protein_mean.dtype), smiles.to(self.smiles_mean.dtype)
        if self._config['input_normalization'] == 'standard':
            p = (p - self.protein_mean) / self.protein_scale
            s = (s - self.smiles_mean) / self.smiles_scale
        else:
            p, s = self.protein_norm(p), self.smiles_norm(s)
        if fingerprint is not None:
            s = torch.cat([s, fingerprint.to(s.dtype)], dim=1)
        interaction = self.pair_protein(p) * self.pair_substrate(s)
        return base + self.pair_gate_logit.sigmoid() * self.pair_head(interaction)


def smoke_test():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    from compact_kcat_models import build_compact_predictor
    for fdim in [0, 167]:
        model = LowRankResidualKcat(fingerprint_dim=fdim, rank=32).double().eval()
        h, s = torch.randn(3, 1280, dtype=torch.float64, requires_grad=True), torch.randn(3, 1024, dtype=torch.float64)
        fp = torch.randint(2, (3, fdim)).double() if fdim else None
        pred = model(h, s, fp)
        assert pred.shape == (3, 1) and torch.isfinite(pred).all()
        grad = torch.autograd.grad(pred.sum(), h, retain_graph=True)[0]
        gate_grad = torch.autograd.grad(pred.sum(), model.pair_gate_logit, retain_graph=True)[0]
        branch_grad = torch.autograd.grad(pred.sum(), model.pair_protein[0].weight)[0]
        assert grad.abs().sum() > 0 and gate_grad.abs() > 0 and branch_grad.abs().sum() > 0
        direction = torch.randn_like(h)
        direction /= direction.norm()
        numeric = (model(h + 1e-5 * direction, s, fp).sum() - model(h - 1e-5 * direction, s, fp).sum()) / 2e-5
        torch.testing.assert_close(numeric, (grad * direction).sum(), atol=1e-6, rtol=1e-3)
        for i in range(3):
            torch.testing.assert_close(pred[i:i+1], model(h[i:i+1], s[i:i+1], fp[i:i+1] if fp is not None else None))
        restored = build_compact_predictor(model.config_dict()).double().eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(pred, restored(h, s, fp))
        with torch.no_grad():
            model.pair_head[-1].weight.zero_()
            model.pair_head[-1].bias.zero_()
        torch.testing.assert_close(model(h, s, fp), CompactKcatPredictor.forward(model, h, s, fp))
    return 'shape, batch invariance, checkpoint config, branch/input gradients, finite difference, zero residual: passed'


if __name__ == '__main__':
    print(smoke_test())
