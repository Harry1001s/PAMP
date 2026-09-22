from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.stats import pearsonr, rankdata


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or y.size == 0:
        raise ValueError(f"Metric inputs must be nonempty and shape matched: {y.shape} vs {p.shape}")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metric inputs contain NaN/Inf")
    residual = p - y
    mae = float(np.mean(np.abs(residual), dtype=np.float64))
    rmse = float(math.sqrt(np.mean(residual * residual, dtype=np.float64)))
    denom = float(np.sum((y - y.mean()) ** 2, dtype=np.float64))
    r2 = None if y.size < 2 or denom == 0 else float(1.0 - np.sum(residual * residual) / denom)
    if y.size < 2 or np.std(y) == 0 or np.std(p) == 0:
        pearson = spearman = None
    else:
        pearson = float(pearsonr(y, p).statistic)
        spearman = float(pearsonr(rankdata(y), rankdata(p)).statistic)
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson, "spearman": spearman}
