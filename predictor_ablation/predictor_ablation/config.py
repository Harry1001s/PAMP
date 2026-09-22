from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .parameter_budget import match_budget
from .registry import config_id


def deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict): result[key] = deep_merge(result[key], value)
        else: result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def base_config(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    reference = load_yaml(root / "configs/reference.yaml")
    protocol = load_yaml(root / "configs/protocol.yaml")
    return deep_merge(reference, protocol["training"])


def finalize(cfg: dict[str, Any], stage: str, group: str, anchor: str | None = None) -> dict[str, Any]:
    cfg = deepcopy(cfg); cfg.update({"stage": stage, "comparison_group": group, "anchor_config_id": anchor})
    cfg["config_id"] = config_id(cfg)
    return cfg


def a0(root: str | Path) -> dict[str, Any]:
    cfg = match_budget(base_config(root), "medium", 2)
    return finalize(cfg, "baseline", "a0")


def capacity_grid(root: str | Path) -> list[dict[str, Any]]:
    anchor = a0(root)["config_id"]
    return [finalize(match_budget(base_config(root), budget, depth), "capacity_depth", "capacity_depth", anchor)
            for budget in ("small", "medium", "large", "xl") for depth in (1, 2, 3, 4)]


def block_grid(root: str | Path, anchor_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows, aid = [], anchor_cfg["config_id"]
    variants = []
    variants += [("activation", {"activation": x}) for x in ("relu", "silu")]
    variants += [("dropout", {"dropout": x}) for x in (0.0, 0.2, 0.3, 0.5)]
    variants += [("norm", {"norm": x}) for x in ("layernorm", "batchnorm")]
    variants += [("residual", {"residual": True})]
    variants += [("input_normalization", {"input_normalization": {"protein": x, "smiles": x}}) for x in ("l2", "standard")]
    for group, changes in variants:
        base = deep_merge(anchor_cfg, changes)
        cfg = match_budget(base, anchor_cfg["budget_id"], anchor_cfg["depth"], anchor_cfg.get("shape", "uniform"))
        rows.append(finalize(cfg, "block_ablation", group, aid))
    return rows


def concat_projection_grid(root: str | Path, anchor_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows, aid = [], anchor_cfg["config_id"]
    # The user requested explicit attention to concat. Keep fusion fixed first.
    for mode in ("protein_only", "smiles_only", "both"):
        cfg = deep_merge(anchor_cfg, {"projection_mode": mode, "projection_dim": 256, "fusion": "concat"})
        cfg = match_budget(cfg, anchor_cfg["budget_id"], anchor_cfg["depth"])
        rows.append(finalize(cfg, "projection_fusion", "concat_projection", aid))
    for q in (128, 512):
        cfg = deep_merge(anchor_cfg, {"projection_mode": "both", "projection_dim": q, "fusion": "concat"})
        cfg = match_budget(cfg, anchor_cfg["budget_id"], anchor_cfg["depth"])
        rows.append(finalize(cfg, "projection_fusion", "concat_projection_dim", aid))
    for fusion in ("sum", "gated", "bilinear"):
        cfg = deep_merge(anchor_cfg, {"projection_mode": "both", "projection_dim": 256, "fusion": fusion})
        cfg = match_budget(cfg, anchor_cfg["budget_id"], anchor_cfg["depth"])
        rows.append(finalize(cfg, "projection_fusion", "fusion", aid))
    return rows
