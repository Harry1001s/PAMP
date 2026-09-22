from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .reproducibility import atomic_json, sha256_json


CONFIG_FINGERPRINT_FIELDS = (
    "family", "protein_dim", "smiles_dim", "pooling", "input_normalization",
    "projection_mode", "projection_dim", "fusion", "bilinear_rank", "depth",
    "hidden_dims", "shape", "activation", "dropout", "norm", "residual",
    "loss", "huber_delta", "optimizer", "lr", "weight_decay", "scheduler",
    "effective_batch_size", "micro_batch_size", "max_epochs", "minimum_epochs_before_early_stop",
    "early_stop_patience", "target_standardization", "precision",
)


def config_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    return {k: config.get(k) for k in CONFIG_FINGERPRINT_FIELDS}


def config_id(config: dict[str, Any]) -> str:
    return f"cfg_{sha256_json(config_fingerprint(config))[:12]}"


class RunRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"schema_version": 1, "runs": {}}

    def update(self, run_id: str, **fields: Any) -> None:
        row = self.data["runs"].setdefault(run_id, {})
        row.update(fields)
        atomic_json(self.path, self.data)

    def get(self, run_id: str) -> dict[str, Any] | None:
        return self.data["runs"].get(run_id)
