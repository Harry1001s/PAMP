"""Mean-embedding kcat heads, with legacy full-residue heads retained explicitly.

Python >= 3.9; tested with PyTorch 2.7.1. No pretrained model download is performed.
Mean heads: protein_embs [B,Dp], smiles_embs [B,Ds], no mask.
Legacy residue heads: protein_embs [B,L,Dp], mask [B,L], True=real residue.
All heads output [B,1].

The output is a standardized log2(kcat) value only when the training targets
were standardized log2(kcat). This module does not infer the target scale.
Use the TRAIN-set mean/std for inverse transformation outside the model.
Ragged storage is handled by the caller; only a minibatch may be padded.

This is a fresh implementation of the proposed designs, not a byte-for-byte
reproduction of the user's existing Global+Local checkpoint implementation.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn


def _validate_and_mask(
    protein_embs: Tensor, smiles_embs: Tensor, mask: Tensor | None,
    prot_dim: int, smiles_dim: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if protein_embs.ndim != 3 or protein_embs.shape[-1] != prot_dim:
        raise ValueError(f"Expected protein_embs [B,L,{prot_dim}], got {tuple(protein_embs.shape)}")
    if protein_embs.shape[0] == 0 or protein_embs.shape[1] == 0:
        raise ValueError("Empty batches and zero-length sequences are unsupported.")
    if smiles_embs.shape != (protein_embs.shape[0], smiles_dim):
        raise ValueError(f"Expected smiles_embs [B,{smiles_dim}], got {tuple(smiles_embs.shape)}")
    if protein_embs.device != smiles_embs.device:
        raise ValueError("Protein and substrate features must be on the same device.")
    if not protein_embs.is_floating_point() or not smiles_embs.is_floating_point():
        raise TypeError("Embeddings must be floating point tensors.")
    if mask is None:
        mask = torch.ones(protein_embs.shape[:2], dtype=torch.bool, device=protein_embs.device)
    if mask.dtype != torch.bool or mask.shape != protein_embs.shape[:2]:
        raise ValueError("mask must be boolean [B,L], with True for actual residues.")
    if mask.device != protein_embs.device:
        raise ValueError("The residue mask must be on the feature device.")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Each sequence must contain at least one real residue.")
    # Upcast only the current minibatch, not the entire cached dataset.
    # torch.where also neutralizes arbitrary/NaN values at padded positions.
    h = torch.where(mask.unsqueeze(-1), protein_embs.float(), 0.0)
    return h, smiles_embs.float(), mask


def _masked_mean(h: Tensor, mask: Tensor) -> Tensor:
    counts = mask.sum(dim=1, keepdim=True).to(h.dtype)
    return (h * mask.unsqueeze(-1)).sum(dim=1) / counts


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


def _sinusoidal_positions(length: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if width % 2:
        raise ValueError("Sinusoidal positions require an even d_model in this implementation.")
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    scales = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32)
                       * (-math.log(10000.0) / width))
    angles = positions * scales.unsqueeze(0)
    output = torch.empty(length, width, device=device, dtype=torch.float32)
    output[:, 0::2], output[:, 1::2] = angles.sin(), angles.cos()
    return output.to(dtype=dtype)


class CLSResidueTransformer(nn.Module):
    """[CLS, SUB, GLOBAL, residues...] -> bidirectional Transformer -> CLS -> scalar.

    No 2-D image conversion; patch size is one residue. The global mean enters
    as an INPUT token, never as a prediction bypass. CLS is a newly learned
    regression-summary token, not the pretrained protein model's BOS token.
    """
    def __init__(
        self, *, prot_dim: int, smiles_dim: int = 1024, d_model: int = 256,
        num_heads: int = 4, num_layers: int = 4, dim_feedforward: int = 1024,
        dropout: float = 0.1, add_global_token: bool = True,
    ):
        super().__init__()
        if min(prot_dim, smiles_dim, d_model, num_heads, num_layers, dim_feedforward) <= 0:
            raise ValueError("Dimensions, heads, and layer count must be positive.")
        if d_model % num_heads or d_model % 2:
            raise ValueError("d_model must be even and divisible by num_heads.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0,1).")
        self.prot_dim, self.smiles_dim = prot_dim, smiles_dim
        self.d_model, self.add_global_token = d_model, add_global_token
        self._config = dict(
            architecture="cls_transformer", prot_dim=prot_dim, smiles_dim=smiles_dim,
            d_model=d_model, num_heads=num_heads, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, add_global_token=add_global_token,
        )
        # Each block has independent parameters and initialization.
        self.residue_projection = ShallowProjection(prot_dim, d_model, 0.0)
        self.substrate_projection = ShallowProjection(smiles_dim, d_model, 0.0)
        self.global_projection = ShallowProjection(prot_dim, d_model, 0.0) if add_global_token else None
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        # Types: CLS, substrate, global protein, protein residue.
        self.type_embeddings = nn.Parameter(torch.empty(4, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.type_embeddings, std=0.02)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def config_dict(self) -> dict[str, Any]:
        return dict(self._config)

    def forward(
        self, protein_embs: Tensor, smiles_embs: Tensor, mask: Tensor | None = None,
        *, return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        h, s, valid = _validate_and_mask(protein_embs, smiles_embs, mask, self.prot_dim, self.smiles_dim)
        b, length, _ = h.shape
        residues = self.residue_projection(h)
        residue_positions = _sinusoidal_positions(length, self.d_model, h.device, residues.dtype)
        residues = residues + residue_positions.unsqueeze(0) + self.type_embeddings[3]
        extras = [
            self.cls_token.expand(b, -1, -1) + self.type_embeddings[0],
            self.substrate_projection(s).unsqueeze(1) + self.type_embeddings[1],
        ]
        if self.global_projection is not None:
            extras.append(self.global_projection(_masked_mean(h, valid)).unsqueeze(1) + self.type_embeddings[2])
        tokens = self.input_dropout(torch.cat([*extras, residues], dim=1))
        # PyTorch TransformerEncoderLayer: True in key_padding_mask = IGNORE.
        extra_valid = torch.ones(b, len(extras), dtype=torch.bool, device=h.device)
        token_valid = torch.cat([extra_valid, valid], dim=1)
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=~token_valid, is_causal=False)
        cls_state = tokens[:, 0, :]
        result = self.regressor(cls_state)  # The only prediction readout.
        if return_aux:
            return result, {"cls_state": cls_state, "token_valid": token_valid}
        return result


class GlobalLocalMLP(nn.Module):
    """Previous-style global mean backbone plus substrate-conditioned local pooling.

    This contains one cross-attention POOL, not a stack of residue self-attention
    encoders. The scalar gate is learned globally and initialized at 0.20.
    """
    def __init__(
        self, *, prot_dim: int, smiles_dim: int = 1024, global_dim: int = 512,
        local_dim: int = 256, num_heads: int = 4, dropout: float = 0.1,
        local_gate_init: float = 0.2,
    ):
        super().__init__()
        if min(prot_dim, smiles_dim, global_dim, local_dim, num_heads) <= 0:
            raise ValueError("Dimensions and number of heads must be positive.")
        if local_dim % num_heads:
            raise ValueError("local_dim must be divisible by num_heads.")
        if not 0 < local_gate_init < 1 or not 0 <= dropout < 1:
            raise ValueError("gate must be in (0,1); dropout must be in [0,1).")
        self.prot_dim, self.smiles_dim = prot_dim, smiles_dim
        self._config = dict(
            architecture="global_local_mlp", prot_dim=prot_dim, smiles_dim=smiles_dim,
            global_dim=global_dim, local_dim=local_dim, num_heads=num_heads,
            dropout=dropout, local_gate_init=local_gate_init,
        )
        self.protein_global = ShallowProjection(prot_dim, global_dim, dropout)
        self.substrate_global = ShallowProjection(smiles_dim, global_dim, dropout)
        self.residue_local = ShallowProjection(prot_dim, local_dim, dropout)
        self.substrate_to_local = ShallowProjection(global_dim, local_dim, 0.0)
        self.local_attention = nn.MultiheadAttention(local_dim, num_heads, dropout=0.0, batch_first=True)
        self.local_to_global = ShallowProjection(local_dim, global_dim, 0.0)
        self.local_gate_logit = nn.Parameter(torch.tensor(math.log(local_gate_init / (1 - local_gate_init))))
        self.protein_fusion_norm = nn.LayerNorm(global_dim)
        self.regressor = nn.Sequential(
            nn.LayerNorm(4 * global_dim), nn.Linear(4 * global_dim, global_dim),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(global_dim, 128),
            nn.SiLU(), nn.LayerNorm(128), nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def config_dict(self) -> dict[str, Any]:
        return dict(self._config)

    def forward(
        self, protein_embs: Tensor, smiles_embs: Tensor, mask: Tensor | None = None,
        *, return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        h, s, valid = _validate_and_mask(protein_embs, smiles_embs, mask, self.prot_dim, self.smiles_dim)
        raw_mean = _masked_mean(h, valid)
        global_protein = self.protein_global(raw_mean)
        substrate = self.substrate_global(s)
        residues = self.residue_local(h)
        query = self.substrate_to_local(substrate).unsqueeze(1)
        local, weights = self.local_attention(
            query, residues, residues, key_padding_mask=~valid,
            need_weights=return_aux, average_attn_weights=False,
        )
        local = self.local_to_global(local[:, 0, :])
        gate = self.local_gate_logit.sigmoid()
        protein = self.protein_fusion_norm(global_protein + gate * local)
        interaction = torch.cat([protein, substrate, protein * substrate, (protein - substrate).abs()], dim=-1)
        result = self.regressor(interaction)
        if return_aux:
            aux = {"protein_vector": protein, "substrate_vector": substrate,
                   "local_gate": gate, "raw_mean": raw_mean}
            if weights is not None:
                aux["attention_weights"] = weights.squeeze(2)
            return result, aux
        return result


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


def build_predictor(architecture: str, **kwargs: Any) -> nn.Module:
    if architecture == "mean_cls_mlp_hybrid":
        from kcat_cls_mlp_hybrid import MeanCLSHybrid
        return MeanCLSHybrid(**kwargs)
    if architecture == "mean_cls_transformer":
        return MeanCLSTransformer(**kwargs)
    if architecture == "mean_interaction_mlp":
        return MeanInteractionMLP(**kwargs)
    if architecture == "cls_transformer":
        return CLSResidueTransformer(**kwargs)
    if architecture == "global_local_mlp":
        return GlobalLocalMLP(**kwargs)
    raise ValueError(f"Unknown architecture: {architecture!r}")


def smoke_test() -> dict[str, Any]:
    """Small CPU tests only; not a training/performance or CUDA-memory benchmark."""
    torch.set_num_threads(1)
    torch.manual_seed(42)
    report: dict[str, Any] = {"torch_version": torch.__version__, "device": "cpu", "models": {}}
    b, length, dp, ds = 3, 7, 1280, 1024
    h, s = torch.randn(b, length, dp), torch.randn(b, ds)
    valid = torch.arange(length).unsqueeze(0) < torch.tensor([7, 3, 5]).unsqueeze(1)
    for architecture in ["cls_transformer", "global_local_mlp"]:
        model = build_predictor(architecture, prot_dim=dp, smiles_dim=ds).eval()
        with torch.no_grad():
            expected = model(h, s, valid)
            assert expected.shape == (b, 1) and torch.isfinite(expected).all()
            modified = h.clone()
            modified[~valid] = float("nan")
            torch.testing.assert_close(expected, model(modified, s, valid), atol=2e-5, rtol=2e-5)
            padded_h = torch.cat([h, torch.randn(b, 3, dp)], dim=1)
            padded_mask = torch.cat([valid, torch.zeros(b, 3, dtype=torch.bool)], dim=1)
            torch.testing.assert_close(expected, model(padded_h, s, padded_mask), atol=2e-5, rtol=2e-5)
            permutation = torch.tensor([2, 0, 1])
            torch.testing.assert_close(expected[permutation], model(h[permutation], s[permutation], valid[permutation]), atol=2e-5, rtol=2e-5)
            for i in range(b):
                n = int(valid[i].sum())
                individual = model(h[i:i+1, :n], s[i:i+1])
                torch.testing.assert_close(expected[i:i+1], individual, atol=2e-5, rtol=2e-5)
        model.train()
        h_grad, s_grad = h.clone().requires_grad_(True), s.clone().requires_grad_(True)
        prediction = model(h_grad, s_grad, valid)
        prediction.square().mean().backward()
        assert h_grad.grad is not None and s_grad.grad is not None
        assert torch.isfinite(h_grad.grad).all() and torch.isfinite(s_grad.grad).all()
        assert h_grad.grad[valid].abs().sum() > 0 and s_grad.grad.abs().sum() > 0
        assert torch.equal(h_grad.grad[~valid], torch.zeros_like(h_grad.grad[~valid]))
        if architecture == "cls_transformer":
            assert model.cls_token.grad is not None and model.cls_token.grad.abs().sum() > 0
        # Configuration round-trip; old checkpoints need their original loader.
        cfg = model.config_dict()
        rebuilt = build_predictor(**cfg)
        rebuilt.load_state_dict(model.state_dict(), strict=True)
        model.eval(), rebuilt.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(h, s, valid), rebuilt(h, s, valid))
        report["models"][architecture] = {
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "output_shape": list(prediction.shape),
            "shape_mask_padding_batch_invariance_gradient_checkpoint_tests": "passed",
        }
    for architecture in ["mean_cls_transformer", "mean_interaction_mlp"]:
        model = build_predictor(architecture, prot_dim=dp, smiles_dim=ds).eval()
        mean_h = torch.randn(b, dp, requires_grad=True)
        mean_s = s.clone().requires_grad_(True)
        expected = model(mean_h, mean_s)
        assert expected.shape == (b, 1) and torch.isfinite(expected).all()
        permutation = torch.tensor([2, 0, 1])
        torch.testing.assert_close(expected[permutation], model(mean_h[permutation], mean_s[permutation]), atol=2e-5, rtol=2e-5)
        for i in range(b):
            torch.testing.assert_close(expected[i:i+1], model(mean_h[i:i+1], mean_s[i:i+1]), atol=2e-5, rtol=2e-5)
        expected.square().mean().backward()
        for feature in (mean_h, mean_s):
            assert torch.isfinite(feature.grad).all() and feature.grad.abs().sum() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if architecture == "mean_cls_transformer":
            assert model.cls_token.grad.abs().sum() > 0
        rebuilt = build_predictor(**model.config_dict()).eval()
        rebuilt.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(expected, rebuilt(mean_h, mean_s))
        try:
            model(mean_h.unsqueeze(1), mean_s)
        except ValueError:
            pass
        else:
            raise AssertionError("Mean heads must reject residue-shaped input")
        report["models"][architecture] = {
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "shape_batch_gradient_checkpoint_input_contract_tests": "passed",
        }

    return report


if __name__ == "__main__":
    import json
    print(json.dumps(smoke_test(), indent=2))
