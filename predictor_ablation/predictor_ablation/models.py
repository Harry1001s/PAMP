from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


def make_norm(name: str, dim: int) -> nn.Module:
    if name == "none":
        return nn.Identity()
    if name == "layernorm":
        return nn.LayerNorm(dim, eps=1e-5, elementwise_affine=True)
    if name == "batchnorm":
        return nn.BatchNorm1d(dim, eps=1e-5, momentum=0.1, affine=True)
    raise ValueError(name)


class DenseBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: str, norm: str, dropout: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = make_norm(norm, out_dim)
        self.activation = ACTIVATIONS[activation]()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.norm(self.linear(x))))


class ConcatPredictor(nn.Module):
    """Two-modality predictor with explicit projection and concat/fusion choices."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        dp, ds = int(config["protein_dim"]), int(config["smiles_dim"])
        projection_mode = config.get("projection_mode", "none")
        q = int(config.get("projection_dim", 256))
        activation = config.get("activation", "gelu")
        projected_p = projection_mode in {"protein_only", "both"}
        projected_s = projection_mode in {"smiles_only", "both"}
        self.protein_projection = nn.Sequential(nn.Linear(dp, q), ACTIVATIONS[activation]()) if projected_p else nn.Identity()
        self.smiles_projection = nn.Sequential(nn.Linear(ds, q), ACTIVATIONS[activation]()) if projected_s else nn.Identity()
        pdim, sdim = (q if projected_p else dp), (q if projected_s else ds)
        fusion = config.get("fusion", "concat")
        self.fusion_name = fusion
        self.gate = None
        self.bilinear_a = self.bilinear_b = None
        if fusion == "concat":
            head_in = pdim + sdim
        elif fusion == "sum":
            if pdim != sdim:
                raise ValueError("sum fusion requires equal branch dimensions")
            head_in = pdim
        elif fusion == "gated":
            if pdim != sdim:
                raise ValueError("gated fusion requires equal branch dimensions")
            self.gate = nn.Linear(pdim + sdim, pdim)
            head_in = pdim
        elif fusion == "bilinear":
            if pdim != sdim:
                raise ValueError("bilinear fusion requires equal branch dimensions")
            rank = int(config.get("bilinear_rank", 32))
            self.bilinear_a = nn.Linear(pdim, rank, bias=False)
            self.bilinear_b = nn.Linear(sdim, rank, bias=False)
            head_in = pdim + sdim + rank
        else:
            raise ValueError(fusion)
        dims = [int(v) for v in config["hidden_dims"]]
        norm, dropout = config.get("norm", "none"), float(config.get("dropout", 0.1))
        self.blocks = nn.ModuleList()
        self.residual = bool(config.get("residual", False))
        previous = head_in
        for width in dims:
            self.blocks.append(DenseBlock(previous, width, activation, norm, dropout))
            previous = width
        self.output = nn.Linear(previous, 1)

    def fuse(self, p: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.fusion_name == "concat":
            return torch.cat([p, s], dim=-1)
        if self.fusion_name == "sum":
            return (p + s) / math.sqrt(2.0)
        if self.fusion_name == "gated":
            assert self.gate is not None
            gate = torch.sigmoid(self.gate(torch.cat([p, s], dim=-1)))
            return gate * p + (1.0 - gate) * s
        assert self.bilinear_a is not None and self.bilinear_b is not None
        interaction = self.bilinear_a(p) * self.bilinear_b(s)
        return torch.cat([p, s, interaction], dim=-1)

    def forward(self, protein: torch.Tensor, smiles: torch.Tensor) -> torch.Tensor:
        x = self.fuse(self.protein_projection(protein), self.smiles_projection(smiles))
        for index, block in enumerate(self.blocks):
            old = x
            x = block(x)
            if self.residual and index > 0 and old.shape == x.shape:
                x = x + old
        return self.output(x).squeeze(-1)


def build_model(config: dict[str, Any]) -> nn.Module:
    family = config.get("family", "concat_mlp")
    if family != "concat_mlp":
        raise ValueError(f"Unsupported family: {family}")
    return ConcatPredictor(config)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "buffer_elements": sum(b.numel() for b in model.buffers()),
    }
