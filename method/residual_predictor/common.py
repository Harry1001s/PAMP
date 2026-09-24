"""Data, residue batches, and checkpoint utilities for RC-TabM."""
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

OUT = DATA_ROOT / 'experiments/catapro_tabm_small'
SCREEN = OUT
EXPERIMENT = DATA_ROOT / 'experiments/catapro_residual_condpool'
CONFIG = _pl.Path(__file__).resolve().parents[2] / 'data/outputs/predictor_config.json'
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

import random
import torch

class CataproResidueCache:
    def __init__(self, directory=None):
        self.directory=Path(directory) if directory else OUT/'cache/esm2_residue'
        self.index=json.loads((self.directory/'index.json').read_text())
        self.row_mapping=np.asarray(self.index['row_mapping'],dtype=np.int64)

    def __len__(self):
        return len(self.row_mapping)

    def by_sequence(self, unique_sequence_index):
        i=int(unique_sequence_index)
        if not 0<=i<len(self.index['lengths']): raise IndexError(i)
        value=np.load(self.directory/f'{i:05d}.npy',mmap_mode='r',allow_pickle=False)
        if value.shape!=(self.index['lengths'][i],1280): raise ValueError(f'Invalid residue shape for sequence {i}')
        return value

    def __getitem__(self, dataset_row):
        i=int(dataset_row)
        if not 0<=i<len(self.row_mapping): raise IndexError(i)
        return self.by_sequence(self.row_mapping[i])

class Batches:
    def __init__(self, config, protein, substrate, y, mapping, device):
        self.config = config
        self.device = device
        self.protein = torch.tensor(protein,device=device)
        self.substrate = torch.tensor(substrate,device=device)
        self.y = torch.tensor(y,dtype=torch.float32,device=device)
        self.mapping = mapping
        self.cache = {}
        if config['kind'] == 'condpool':
            assert (OUT/'cache/esm2_residue/completion.json').exists(), 'Complete residue extraction first'
            # CPU caching avoids repeated disk I/O. Float16 ragged arrays, no global padding.
            for uid in np.unique(mapping):
                self.cache[int(uid)] = torch.from_numpy(np.load(OUT/f'cache/esm2_residue/{uid:05d}.npy'))
            self.lengths = np.array([len(self.cache[int(i)]) for i in mapping])

    def order(self, ids, training=False):
        ids = np.asarray(ids).copy()
        if training:
            np.random.shuffle(ids)
        if self.config['kind'] == 'condpool':
            # Sort small random pools, then shuffle batches; every row occurs exactly once.
            pools = [ids[i:i+1024] for i in range(0,len(ids),1024)]
            ids = np.concatenate([p[np.argsort(self.lengths[p],kind='stable')] for p in pools])
        batches = [ids[i:i+self.config['batch_size']] for i in range(0,len(ids),self.config['batch_size'])]
        if training:
            random.shuffle(batches)
        return batches

    def get(self, ids):
        ix = torch.as_tensor(ids,device=self.device)
        mask = None
        if self.config['kind'] == 'mean':
            h = self.protein[ix]
        else:
            lengths = self.lengths[ids]
            h = torch.zeros((len(ids),int(max(lengths)),1280),dtype=torch.float16)
            for j,i in enumerate(ids):
                h[j,:lengths[j]] = self.cache[int(self.mapping[i])]
            h = h.to(self.device)
            mask = torch.arange(h.shape[1],device=self.device)[None,:] < torch.tensor(lengths,device=self.device)[:,None]
        return h, self.substrate[ix], mask, self.y[ix]


def atomic_checkpoint(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value,tmp)
    tmp.replace(path)
