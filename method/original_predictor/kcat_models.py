"""CLS and interaction heads for mean protein and substrate embeddings."""
from __future__ import annotations
import torch
from torch import Tensor, nn

class ShallowProjection(nn.Module):
    """Feature-wise normalization and a small trainable projection."""
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim),
            nn.SiLU(), nn.LayerNorm(output_dim), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _validate_mean(h, s, prot_dim, smiles_dim):
    if h.ndim != 2 or h.shape[1] != prot_dim or h.shape[0] == 0:
        raise ValueError("Expected nonempty protein mean embeddings [B,Dp].")
    if s.shape != (h.shape[0], smiles_dim) or s.device != h.device:
        raise ValueError("Expected aligned substrate embeddings [B,Ds] on the same device.")
    if not h.is_floating_point() or not s.is_floating_point():
        raise TypeError("Embeddings must be floating point.")
    return h.float(), s.float()


class MeanCLSTransformer(nn.Module):
    """Three tokens [CLS, SUB, PROTEIN_MEAN]; no synthetic residue tokens."""
    def __init__(self, *, prot_dim, smiles_dim=1024, d_model=256, num_heads=4,
                 num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.prot_dim, self.smiles_dim = prot_dim, smiles_dim
        self._config = dict(architecture="mean_cls_transformer", prot_dim=prot_dim,
                            smiles_dim=smiles_dim, d_model=d_model, num_heads=num_heads,
                            num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.protein_projection = ShallowProjection(prot_dim, d_model, 0.0)
        self.substrate_projection = ShallowProjection(smiles_dim, d_model, 0.0)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.type_embeddings = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, activation="gelu",
            batch_first=True, norm_first=True) for _ in range(num_layers)])
        self.regressor = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 128),
                                       nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def config_dict(self):
        return dict(self._config)

    def forward(self, protein_embs, smiles_embs):
        h, s = _validate_mean(protein_embs, smiles_embs, self.prot_dim, self.smiles_dim)
        tokens = torch.cat([self.cls_token.expand(len(h), -1, -1),
                            self.substrate_projection(s).unsqueeze(1),
                            self.protein_projection(h).unsqueeze(1)], dim=1)
        tokens = self.input_dropout(tokens + self.type_embeddings)
        for layer in self.layers:
            tokens = layer(tokens)
        return self.regressor(tokens[:, 0])


class MeanInteractionMLP(nn.Module):
    """Original global backbone and four-way interaction, without residue pooling."""
    def __init__(self, *, prot_dim, smiles_dim=1024, global_dim=512, dropout=0.1):
        super().__init__()
        self.prot_dim, self.smiles_dim = prot_dim, smiles_dim
        self._config = dict(architecture="mean_interaction_mlp", prot_dim=prot_dim,
                            smiles_dim=smiles_dim, global_dim=global_dim, dropout=dropout)
        self.protein_global = ShallowProjection(prot_dim, global_dim, dropout)
        self.substrate_global = ShallowProjection(smiles_dim, global_dim, dropout)
        self.protein_fusion_norm = nn.LayerNorm(global_dim)
        self.regressor = nn.Sequential(nn.LayerNorm(4 * global_dim),
            nn.Linear(4 * global_dim, global_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(global_dim, 128), nn.SiLU(), nn.LayerNorm(128),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def config_dict(self):
        return dict(self._config)

    def forward(self, protein_embs, smiles_embs):
        h, s = _validate_mean(protein_embs, smiles_embs, self.prot_dim, self.smiles_dim)
        p = self.protein_fusion_norm(self.protein_global(h))
        s = self.substrate_global(s)
        return self.regressor(torch.cat([p, s, p*s, (p-s).abs()], dim=-1))


def build_predictor(architecture, **kwargs):
    models = {"mean_cls_transformer": MeanCLSTransformer, "mean_interaction_mlp": MeanInteractionMLP}
    return models[architecture](**kwargs)
