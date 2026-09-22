from __future__ import annotations

import csv
import json
import math
import os
import resource
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn

from .data import load_manifest, load_vector_data
from .metrics import regression_metrics
from .models import build_model, parameter_counts
from .preprocessing import FeatureScaler, TargetScaler
from .reproducibility import atomic_json, canonical_json, seed_everything, sha256_file, sha256_json


def epoch_batches(order, batch_size):
    if batch_size < 2:
        raise ValueError("Batch size must be at least two")
    batches = list(order.split(batch_size))
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2] = torch.cat([batches[-2], batches[-1]])
        batches.pop()
    return batches


def _loss(config: dict[str, Any]) -> nn.Module:
    name = config.get("loss", "mse")
    if name == "mse": return nn.MSELoss()
    if name == "mae": return nn.L1Loss()
    if name == "huber": return nn.HuberLoss(delta=float(config.get("huber_delta", 1.0)))
    raise ValueError(name)


def _parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        (no_decay if parameter.ndim == 1 or name.endswith("bias") else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def _optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    lr, wd = float(config["lr"]), float(config["weight_decay"])
    groups = _parameter_groups(model, wd)
    name = config["optimizer"]
    if name == "adamw": return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    if name == "adam": return torch.optim.Adam(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    if name == "sgd": return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)
    raise ValueError(name)


class EpochScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, config: dict[str, Any]):
        self.optimizer, self.config = optimizer, config
        self.name, self.base_lr = config["scheduler"], float(config["lr"])
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, threshold=1e-4,
            threshold_mode="rel", min_lr=1e-6) if self.name == "plateau" else None

    def step(self, epoch: int, val_rmse: float) -> None:
        if self.name == "plateau":
            assert self.plateau is not None; self.plateau.step(val_rmse); return
        if self.name == "none": return
        warmup, total = 5, int(self.config["max_epochs"])
        if epoch < warmup: factor = 0.1 + 0.9 * epoch / warmup
        else:
            progress = (epoch - warmup) / max(1, total - warmup)
            factor = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups: group["lr"] = self.base_lr * factor

    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name, "plateau": self.plateau.state_dict() if self.plateau else None}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.plateau and state.get("plateau"): self.plateau.load_state_dict(state["plateau"])


def _predict(model: nn.Module, p: torch.Tensor, s: torch.Tensor, indices: torch.Tensor,
             target: TargetScaler, batch_size: int) -> tuple[np.ndarray, dict[str, float | None]]:
    model.eval(); chunks = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            chunks.append(model(p[idx], s[idx]).float().cpu().numpy())
    scaled = np.concatenate(chunks)
    return target.inverse(scaled), {}


def train_one(manifest_path: str | Path, config: dict[str, Any], out_dir: str | Path,
              seed: int, resume: bool = False, pilot_epochs: int | None = None,
              stop_after_epoch: int | None = None) -> dict[str, Any]:
    seed_everything(seed)
    manifest = load_manifest(manifest_path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    config = dict(config); config["seed"] = int(seed)
    if pilot_epochs is not None:
        config["max_epochs"] = int(pilot_epochs)
    elif not manifest.get("formal_training_authorized_by_audit", False):
        raise RuntimeError("Formal training requires resolved provenance and a passed, locked audit")
    if int(config["effective_batch_size"]) != int(config["micro_batch_size"]):
        raise NotImplementedError("Gradient accumulation is not validated yet")
    resolved_path = out / "config.resolved.json"
    if resolved_path.exists():
        if not resume:
            raise FileExistsError(f"Existing trial: {out}; pass resume or use a new attempt")
        if json.loads(resolved_path.read_text()) != config:
            raise ValueError("Resume config mismatch before writing any artifacts")
        if (out / "metrics.json").exists():
            return json.loads((out / "metrics.json").read_text())
    atomic_json(out / "config.resolved.json", config)
    atomic_json(out / "status.json", {"state": "running", "seed": seed, "started_at": time.time()})
    p_raw, s_raw, y_all, splits = load_vector_data(manifest, config.get("pooling", "mean"))
    train_idx_np, val_idx_np = splits["train"], splits["val"]
    if pilot_epochs is not None:
        subset = np.sort(train_idx_np)[:160]
        train_idx_np, val_idx_np = subset[:128], subset[128:]
    p_scaler = FeatureScaler.fit(p_raw[train_idx_np], config.get("input_normalization", {}).get("protein", "none"))
    s_scaler = FeatureScaler.fit(s_raw[train_idx_np], config.get("input_normalization", {}).get("smiles", "none"))
    target = TargetScaler.fit(y_all[train_idx_np])
    joblib.dump({"protein": p_scaler, "smiles": s_scaler, "target": target}, out / "preprocessing.joblib")
    device = torch.device(config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    p = torch.from_numpy(np.array(p_scaler.transform(p_raw), copy=True)).to(device)
    s = torch.from_numpy(np.array(s_scaler.transform(s_raw), copy=True)).to(device)
    y_scaled = torch.from_numpy(target.transform(y_all)).to(device)
    train_idx = torch.as_tensor(train_idx_np, dtype=torch.long, device=device)
    val_idx = torch.as_tensor(val_idx_np, dtype=torch.long, device=device)
    model = build_model(config).to(device)
    counts = parameter_counts(model)
    optimizer = _optimizer(model, config); scheduler = EpochScheduler(optimizer, config); criterion = _loss(config)
    start_epoch, best_rmse, best_epoch, patience_count = 1, float("inf"), 0, 0
    history: list[dict[str, Any]] = []
    last_path = out / "last.pt"
    if resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        if ckpt["config_hash"] != sha256_json(config): raise ValueError("Resume config hash mismatch")
        model.load_state_dict(ckpt["model_state_dict"]); optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch, best_rmse, best_epoch, patience_count = ckpt["epoch"] + 1, ckpt["best_rmse"], ckpt["best_epoch"], ckpt["patience_count"]
        history = ckpt["history"]
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in ckpt["cuda_rng_state"]])
        np.random.set_state(ckpt["numpy_rng_state"])
        random.setstate(ckpt["python_rng_state"])
    max_epochs, batch_size = int(config["max_epochs"]), int(config["micro_batch_size"])
    minimum_epochs, patience = int(config["minimum_epochs_before_early_stop"]), int(config["early_stop_patience"])
    min_rel = float(config.get("min_relative_rmse_improvement", 1e-4))
    started_fit = time.perf_counter(); peak_gpu = 0
    for epoch in range(start_epoch, max_epochs + 1):
        epoch_started = time.perf_counter(); model.train()
        generator = torch.Generator(device=device); generator.manual_seed(seed * 1_000_003 + epoch)
        order = train_idx[torch.randperm(len(train_idx), generator=generator, device=device)]
        weighted_loss, seen, optimizer_steps = 0.0, 0, 0
        if scheduler.name == "cosine":
            scheduler.step(epoch - 1, best_rmse)
        for idx in epoch_batches(order, batch_size):
            optimizer.zero_grad(set_to_none=True)
            pred = model(p[idx], s[idx])
            if pred.shape != y_scaled[idx].shape: raise RuntimeError(f"Prediction/target shape mismatch: {pred.shape}/{y_scaled[idx].shape}")
            loss = criterion(pred, y_scaled[idx]); loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"])))
            if not torch.isfinite(loss) or not math.isfinite(grad_norm): raise FloatingPointError("Non-finite loss/gradient")
            optimizer.step(); optimizer_steps += 1
            weighted_loss += float(loss.detach()) * len(idx); seen += len(idx)
        val_pred, _ = _predict(model, p, s, val_idx, target, batch_size)
        val_metrics = regression_metrics(y_all[val_idx_np], val_pred)
        val_rmse = float(val_metrics["rmse"])
        if scheduler.name == "plateau": scheduler.step(epoch, val_rmse)
        numeric_best = val_rmse < best_rmse
        relative_improved = val_rmse < best_rmse * (1.0 - min_rel)
        if numeric_best:
            best_rmse, best_epoch = val_rmse, epoch
            torch.save({"model_state_dict": model.state_dict(), "resolved_config": config, "epoch": epoch,
                        "best_metric": val_rmse, "target_scaler": target.to_dict(), "data_hash": manifest["data_hash"],
                        "split_hash": manifest["split_hash"], "config_hash": sha256_json(config)}, out / "best.tmp.pt")
            os.replace(out / "best.tmp.pt", out / "best.pt")
        patience_count = 0 if relative_improved else patience_count + 1
        if torch.cuda.is_available(): peak_gpu = max(peak_gpu, torch.cuda.max_memory_allocated(device) / 1024**2)
        history.append({"epoch": epoch, "train_loss_scaled": weighted_loss / seen, **{f"val_{k}": v for k, v in val_metrics.items()},
                        "lr": optimizer.param_groups[0]["lr"], "optimizer_steps": optimizer_steps,
                        "samples_seen": seen, "epoch_seconds": time.perf_counter() - epoch_started, "grad_norm_last": grad_norm})
        checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "best_rmse": best_rmse,
                      "best_epoch": best_epoch, "patience_count": patience_count, "history": history,
                      "config_hash": sha256_json(config), "torch_rng_state": torch.get_rng_state(),
                      "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                      "numpy_rng_state": np.random.get_state()}
        checkpoint["python_rng_state"] = random.getstate()
        torch.save(checkpoint, out / "last.tmp.pt")
        os.replace(out / "last.tmp.pt", last_path)
        with (out / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            atomic_json(out / "status.json", {"state": "interrupted_for_resume_test", "epoch": epoch})
            return {"status": "interrupted_for_resume_test"}
        if epoch >= minimum_epochs and patience_count >= patience: break
    fit_seconds = time.perf_counter() - started_fit
    best = torch.load(out / "best.pt", map_location=device, weights_only=False); model.load_state_dict(best["model_state_dict"])
    train_pred, _ = _predict(model, p, s, train_idx, target, batch_size)
    val_pred, _ = _predict(model, p, s, val_idx, target, batch_size)
    train_metrics, val_metrics = regression_metrics(y_all[train_idx_np], train_pred), regression_metrics(y_all[val_idx_np], val_pred)
    pred_path = out / "val_predictions.csv"
    with pred_path.open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["sample_id", "source_row_id", "split", "y_true", "y_pred", "residual"])
        for idx, true, pred in zip(val_idx_np, y_all[val_idx_np], val_pred): writer.writerow([f"row_{int(idx):06d}", int(idx), "val", float(true), float(pred), float(pred-true)])
    metrics = {"train": train_metrics, "val": val_metrics, "best_epoch": best_epoch, "epochs_run": history[-1]["epoch"],
               "fit_seconds": fit_seconds, "peak_gpu_memory_mb": peak_gpu,
               "peak_cpu_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               **counts, "checkpoint_sha256": sha256_file(out / "best.pt"), "status": "completed"}
    atomic_json(out / "metrics.json", metrics)
    atomic_json(out / "status.json", {"state": "completed", "seed": seed, "completed_at": time.time()})
    return metrics
