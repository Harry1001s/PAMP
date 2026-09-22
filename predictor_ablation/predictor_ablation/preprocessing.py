from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class TargetScaler:
    mean: float
    scale: float

    @classmethod
    def fit(cls, y: np.ndarray) -> "TargetScaler":
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        mean, scale = float(y.mean()), float(y.std(ddof=0))
        if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Training target mean/std is invalid or constant")
        return cls(mean, scale)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((np.asarray(y, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=np.float64) * self.scale + self.mean).astype(np.float64)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class FeatureScaler:
    mode: str
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    @classmethod
    def fit(cls, x: np.ndarray, mode: str) -> "FeatureScaler":
        if mode not in {"none", "standard", "l2"}:
            raise ValueError(f"Unknown feature scaling mode: {mode}")
        if mode != "standard":
            return cls(mode)
        mean = np.asarray(x, dtype=np.float64).mean(axis=0)
        scale = np.asarray(x, dtype=np.float64).std(axis=0)
        scale[scale == 0] = 1.0
        return cls(mode, mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.mode == "none":
            return x
        if self.mode == "standard":
            assert self.mean is not None and self.scale is not None
            return ((x - self.mean) / self.scale).astype(np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return (x / np.maximum(norms, 1e-12)).astype(np.float32)
