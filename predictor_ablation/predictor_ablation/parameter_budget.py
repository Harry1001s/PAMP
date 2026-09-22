from __future__ import annotations

from copy import deepcopy
from typing import Any

from .models import build_model, parameter_counts
import torch


BUDGETS = {"small": 300_000, "medium": 1_000_000, "large": 4_000_000, "xl": 8_000_000}


def hidden_shape(width: int, depth: int, shape: str = "uniform") -> list[int]:
    if shape == "uniform":
        return [width] * depth
    if shape == "tapered":
        return [max(16, int(round((width / (2**i)) / 8) * 8)) for i in range(depth)]
    raise ValueError(shape)


def match_budget(base: dict[str, Any], budget_id: str, depth: int, shape: str = "uniform") -> dict[str, Any]:
    target = BUDGETS[budget_id]
    best: tuple[int, dict[str, Any], int] | None = None
    for width in range(16, 8193, 8):
        cfg = deepcopy(base)
        cfg.update({"budget_id": budget_id, "target_params": target, "depth": depth,
                    "shape": shape, "hidden_dims": hidden_shape(width, depth, shape)})
        with torch.device("meta"):
            count = parameter_counts(build_model(cfg))["trainable_params"]
        error = abs(count - target)
        if best is None or error < best[0] or (error == best[0] and count < best[2]):
            best = (error, cfg, count)
        if count > target and width > 64:
            break
    assert best is not None
    cfg, count = best[1], best[2]
    cfg["trainable_params"] = count
    cfg["parameter_error_fraction"] = abs(count - target) / target
    cfg["budget_match_status"] = "matched" if cfg["parameter_error_fraction"] <= 0.05 else (
        "relaxed" if cfg["parameter_error_fraction"] <= 0.10 else "infeasible")
    return cfg
