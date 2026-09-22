from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .reproducibility import sha256_file, sha256_json


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_splits(path: str | Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    result = {name: np.asarray(z[f"{name}_idx"], dtype=np.int64) for name in ("train", "val", "test")}
    sets = [set(v.tolist()) for v in result.values()]
    for name, values in result.items():
        if values.ndim != 1 or not len(values) or len(set(values.tolist())) != len(values):
            raise ValueError(f"Empty or duplicate indices in {name}")
        if (values < 0).any():
            raise ValueError(f"Negative indices in {name}")
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Split indices overlap")
    return result


def derive_protein_pooling(manifest: dict[str, Any], output: str | Path, modes: tuple[str, ...] = ("mean", "max")) -> dict[str, Any]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    index_path = Path(manifest["protein_cache_index"])
    data_path = Path(manifest["protein_cache_data"])
    with index_path.open("rb") as handle:
        index = pickle.load(handle)
    n, dim = int(index["n_sequences"]), int(index["embedding_dim"])
    arrays = {mode: np.lib.format.open_memmap(output.with_name(f"protein_{mode}.npy"), mode="w+", dtype=np.float32, shape=(n, dim)) for mode in modes}
    with data_path.open("rb") as handle:
        for row, offset in enumerate(index["offsets"]):
            handle.seek(int(offset))
            emb = pickle.load(handle)
            if tuple(emb.shape) != (int(index["lengths"][row]), dim):
                raise ValueError(f"Protein cache shape mismatch at row {row}: {emb.shape}")
            if not np.isfinite(emb).all():
                raise ValueError(f"Nonfinite protein embedding at row {row}")
            if "mean" in arrays:
                arrays["mean"][row] = np.asarray(emb, dtype=np.float32).mean(axis=0)
            if "max" in arrays:
                arrays["max"][row] = np.asarray(emb, dtype=np.float32).max(axis=0)
    result: dict[str, Any] = {}
    for mode, arr in arrays.items():
        arr.flush()
        path = output.with_name(f"protein_{mode}.npy")
        result[mode] = {"path": str(path.resolve()), "shape": [n, dim], "sha256": sha256_file(path)}
    return result


def load_vector_data(manifest: dict[str, Any], pooling: str = "mean", include_test_labels: bool = False):
    if include_test_labels:
        raise PermissionError("Training data API cannot expose test labels; use the locked final evaluator")
    protein_path = Path(manifest["derived_protein_features"][pooling]["path"])
    p = np.load(protein_path, mmap_mode="r", allow_pickle=False)
    s = np.load(manifest["smiles_embeddings"], mmap_mode="r", allow_pickle=False)
    y = np.load(manifest["labels"], mmap_mode="r", allow_pickle=False).reshape(-1)
    splits = load_splits(manifest["split_indices"])
    if not (len(p) == len(s) == len(y)):
        raise ValueError("Feature and label row counts differ")
    # Read only train/validation label elements; test entries remain inaccessible NaNs.
    visible_y = np.full(len(y), np.nan, dtype=np.float32)
    for name in ("train", "val"):
        visible_y[splits[name]] = y[splits[name]]
    return p, s, visible_y, splits


def audit_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    table = pd.read_excel(manifest["source_table"])
    with Path(manifest["protein_cache_index"]).open("rb") as handle:
        index = pickle.load(handle)
    s = np.load(manifest["smiles_embeddings"], mmap_mode="r")
    y = np.load(manifest["labels"], mmap_mode="r").reshape(-1)
    splits = load_splits(manifest["split_indices"])
    n = len(table)
    hard: list[str] = []
    if not (n == len(index["sequences"]) == len(s) == len(y)):
        hard.append("row_count_mismatch")
    if index["original_rows"] != list(range(n)):
        hard.append("cache_original_rows_not_identity")
    sequence_match = bool(np.all(table[manifest["columns"]["protein"]].astype(str).to_numpy() == np.asarray(index["sequences"], dtype=object)))
    smiles_match = bool(np.all(table[manifest["columns"]["smiles"]].astype(str).to_numpy() == np.load(manifest["smiles_strings"], allow_pickle=False)))
    label_match = bool(np.allclose(table[manifest["columns"]["label"]].to_numpy(float), y, rtol=1e-6, atol=1e-6))
    if not sequence_match: hard.append("sequence_cache_mapping_mismatch")
    if not smiles_match: hard.append("smiles_cache_mapping_mismatch")
    if not label_match: hard.append("label_mapping_mismatch")
    seq = table[manifest["columns"]["protein"]].astype(str)
    smi = table[manifest["columns"]["smiles"]].astype(str)
    pair = seq + "\x1f" + smi
    overlap: dict[str, Any] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        li, ri = splits[left], splits[right]
        overlap[f"{left}_{right}"] = {
            "row_id": int(len(set(li.tolist()) & set(ri.tolist()))),
            "sequence": int(len(set(seq.iloc[li]) & set(seq.iloc[ri]))),
            "smiles_raw": int(len(set(smi.iloc[li]) & set(smi.iloc[ri]))),
            "exact_pair": int(len(set(pair.iloc[li]) & set(pair.iloc[ri]))),
        }
    protein_col, smiles_col, label_col = (manifest["columns"][k] for k in ("protein", "smiles", "label"))
    organisms = table[manifest["columns"]["organism"]].fillna("").astype(str).to_numpy()
    proteins = table[protein_col].astype(str).to_numpy()
    smiles_strings = table[smiles_col].astype(str).to_numpy()
    labels = table[label_col].to_numpy(float)
    pair_keys = np.asarray([sha256_json([a, b]) for a, b in zip(proteins, smiles_strings)], dtype=object)
    full_keys = np.asarray([sha256_json([a, b, c, float(d)]) for a, b, c, d in zip(proteins, smiles_strings, organisms, labels)], dtype=object)
    overlap: dict[str, Any] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        li, ri = splits[left], splits[right]
        def count(values: np.ndarray) -> int:
            return len(set(values[li].tolist()) & set(values[ri].tolist()))
        overlap[f"{left}_{right}"] = {
            "source_row_id": len(set(li.tolist()) & set(ri.tolist())),
            "exact_sequence": count(proteins), "exact_smiles": count(smiles_strings),
            "exact_pair": count(pair_keys), "complete_record": count(full_keys),
        }
    finite_smiles = bool(np.isfinite(np.asarray(s)).all())
    finite_labels = bool(np.isfinite(np.asarray(y)).all())
    if not finite_smiles: hard.append("nonfinite_smiles_embedding")
    if not finite_labels: hard.append("nonfinite_label")
    sample_rows = np.linspace(0, max(0, n - 1), 12, dtype=np.int64)
    sampled_cache_shapes = []
    data_path = Path(manifest["protein_cache_data"])
    with data_path.open("rb") as handle:
        for row in sample_rows:
            handle.seek(int(index["offsets"][row])); emb = pickle.load(handle)
            expected = (int(index["lengths"][row]), int(index["embedding_dim"]))
            sampled_cache_shapes.append({"row": int(row), "shape": list(emb.shape), "expected": list(expected), "finite": bool(np.isfinite(emb).all())})
            if emb.shape != expected or not np.isfinite(emb).all(): hard.append(f"bad_protein_cache_sample_{row}")
    return {
        "n_rows": n,
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "feature_shapes": {"protein_ragged_dim": int(index["embedding_dim"]), "smiles": list(s.shape), "labels": list(y.shape)},
        "mapping_checks": {"sequence": sequence_match, "smiles": smiles_match, "label": label_match},
        "finite": {"smiles": bool(np.isfinite(s).all()), "labels": bool(np.isfinite(y).all())},
        "measurement_id_available": False,
        "homology_audit_completed": False,
        "overlap": overlap,
        "smiles_embedding_finite": finite_smiles,
        "labels_finite": finite_labels,
        "sampled_protein_cache": sampled_cache_shapes,
        "hard_errors": hard,
        "status": "failed_protocol" if hard else (
            "requires_provenance_review" if any(v["complete_record"] for v in overlap.values()) else "passed"),
        "split_hash": sha256_json({k + "_idx": v.tolist() for k, v in sorted(splits.items())}),
    }
