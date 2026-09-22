"""Read-only access to the existing CataPro experiment. Never creates splits."""
import sys as _sys, pathlib as _pl
for _c in _pl.Path(__file__).resolve().parents:
    if (_c / 'pamp_paths.py').exists():
        _sys.path.insert(0, str(_c)); break
from pamp_paths import (KCAT_CSV, PROTEIN_SUBSTRATE_V1, DATA_ROOT,
                        add_module_paths, ORIGINAL_PREDICTOR_DIR, TABM_VENDOR)
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = DATA_ROOT
add_module_paths(TABM_VENDOR, ORIGINAL_PREDICTOR_DIR)
SOURCE = KCAT_CSV
SPLIT = ROOT / 'experiment_mean/split_indices.npz'
PROTEIN = ROOT / 'catpro_esm2_mean_pooling/protein_mean_embs.pkl'
SUBSTRATE = ROOT / 'experiment_mean/smiles_unikp1024.npy'
BASE = PROTEIN_SUBSTRATE_V1


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def load_data():
    df = pd.read_csv(SOURCE)
    manifest = pd.read_csv(ROOT / 'experiment_mean/data_manifest.csv')
    assert np.array_equal(df.iloc[:, 0], manifest.iloc[:, 0])
    assert np.array_equal(df.Sequence, manifest.sequence_clean)
    assert np.array_equal(df.Smiles, manifest.Smiles)
    y = np.log2(df['kcat(s^-1)'].to_numpy())
    np.testing.assert_allclose(y, manifest.y_log2, atol=1e-12)
    split = dict(np.load(SPLIT))
    split = {k: split[k + '_idx'] for k in ('train', 'val', 'test')}
    assert [len(split[k]) for k in ('train', 'val', 'test')] == [22126, 2766, 2766]
    assert np.array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(len(df)))
    with PROTEIN.open('rb') as f:
        protein = pickle.load(f)
    substrate = np.load(SUBSTRATE)
    assert protein.shape == (len(df), 1280) and substrate.shape == (len(df), 1024)
    assert all(np.isfinite(a).all() for a in (protein, substrate, y))
    return df, manifest, split, protein, substrate, y


def metrics(y, pred):
    from run_mean_kcat_comparison import metrics as existing_metrics
    return existing_metrics(y, pred)
