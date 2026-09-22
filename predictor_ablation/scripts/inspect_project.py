#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from src.reproducibility import atomic_json, sha256_file, sha256_json


def args():
    p = argparse.ArgumentParser(description="Inspect the existing predictor inputs and write an explicit data manifest.")
    p.add_argument("--project-root", default="../..")
    p.add_argument("--output", default="configs/data_manifest.json")
    p.add_argument("--experiment", default="predictor_v1")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    a = args(); root = Path(a.project_root).resolve(); output = Path(a.output).resolve()
    paths = {
        "source_table": root / "filtered_sequences.xlsx",
        "labels": root / "all_labels.npy",
        "smiles_strings": root / "all_smiles.npy",
        "smiles_embeddings": root / "all_smiles_embs_unikp1024.npy",
        "protein_cache_data": root / "full_esm2_cache_pkl/protein_full_embs.pkl",
        "protein_cache_index": root / "full_esm2_cache_pkl/protein_full_embs_index.pkl",
        "split_indices": root / "predictor_unikp1024_global_local/split_indices.npz",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing: raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    table = pd.read_excel(paths["source_table"])
    labels = np.load(paths["labels"], mmap_mode="r")
    smiles = np.load(paths["smiles_embeddings"], mmap_mode="r")
    with paths["protein_cache_index"].open("rb") as handle: index = pickle.load(handle)
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    split = np.load(paths["split_indices"])
    split_hash = sha256_json({k: np.asarray(split[k], dtype=np.int64).tolist() for k in sorted(split.files)})
    manifest = {"schema_version": 1, "experiment": a.experiment, "project_root": str(root),
                **{k: str(v) for k, v in paths.items()}, "columns": {"protein": "enzyme", "smiles": "smiles", "label": "log2Kcat", "organism": "organism"},
                "target_scale": "log2Kcat", "target_unit": None, "target_unit_verified": False,
                "n_samples": len(table), "protein_embedding_dim": int(index["embedding_dim"]),
                "smiles_embedding_dim": int(smiles.shape[1]), "hashes": hashes, "data_hash": sha256_json(hashes),
                "split_hash": split_hash, "split_sizes": {k.replace("_idx", ""): int(len(split[k])) for k in split.files},
                "derived_protein_features": {}, "test_previously_observed": True,
                "excluded_models": ["ExtraTrees"], "excluded_representations": ["ESMC_600M"]}
    atomic_json(output, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
