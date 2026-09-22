#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from src.data import load_splits
from src.reproducibility import atomic_json, sha256_json
from src.config import a0, capacity_grid, concat_projection_grid


def main():
    p = argparse.ArgumentParser(description='Export source-row duplicate evidence and planned concat comparisons')
    p.add_argument('--experiment', required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    manifest_path = HERE / 'configs/data_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    run = HERE / 'runs' / a.experiment
    audit = run / 'audit'
    table = pd.read_excel(manifest['source_table'])
    splits = load_splits(manifest['split_indices'])
    table['source_row_id'] = np.arange(len(table))
    table['sample_id'] = [f'row_{i:06d}' for i in range(len(table))]
    table['split'] = ''
    for split, rows in splits.items(): table.loc[rows, 'split'] = split
    keys = ['organism', 'enzyme', 'smiles', 'log2Kcat']
    table['record_key'] = [sha256_json(list(row)) for row in table[keys].itertuples(index=False, name=None)]
    across = table.groupby('record_key')['split'].nunique()
    duplicate = table[table.record_key.isin(across[across > 1].index)].copy()
    duplicate.to_csv(audit / 'cross_split_complete_records.csv', index=False)
    canonical = {}
    invalid = []
    for raw in table.smiles.unique():
        molecule = Chem.MolFromSmiles(raw)
        canonical[raw] = Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule is not None else None
        if molecule is None: invalid.append(raw)
    table['canonical_smiles'] = table.smiles.map(canonical)
    table['pair_key'] = [sha256_json([sequence, smi if smi is not None else raw])
                         for sequence, smi, raw in zip(table.enzyme, table.canonical_smiles, table.smiles)]
    canonical_overlap = {}
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        lv, rv = table[table.split == left], table[table.split == right]
        canonical_overlap[left + '_' + right] = {
            'canonical_smiles': len(set(lv.canonical_smiles.dropna()) & set(rv.canonical_smiles.dropna())),
            'sequence_canonical_pair': len(set(lv.pair_key) & set(rv.pair_key))}
    atomic_json(audit / 'canonical_overlap.json', {'overlap': canonical_overlap, 'invalid_smiles': invalid})
    table[['sample_id','source_row_id','split','record_key','pair_key']].to_csv(audit / 'sample_manifest.csv', index=False)
    # The spreadsheet and checkpoint establish the log base, but not the physical unit.
    manifest.update(target_scale='log2Kcat', target_unit=None, target_unit_verified=False,
                    formal_training_authorized_by_audit=False)
    atomic_json(manifest_path, manifest)
    planned = {'status': 'draft_not_locked_not_trained', 'capacity': capacity_grid(HERE),
               'projection_fusion_illustrated_at_a0': concat_projection_grid(HERE, a0(HERE)),
               'note': 'S4 anchor must be selected using completed S2 validation results before its first fit'}
    atomic_json(run / 'manifests/planned_configurations.json', planned)
    blockers = {'status': 'blocked_inputs', 'formal_fits_completed': 0,
                'unit_verified': False, 'duplicate_record_groups_across_splits': int((across > 1).sum()),
                'duplicate_record_rows_across_splits': len(duplicate),
                'required_input': 'Original kcat unit and measurement provenance, or direction to use a conservative pair-grouped derived track',
                'remaining_stages': ['S0 protocol lock/pilot', 'S1 baselines', 'S2 capacity', 'S3 blocks',
                    'S4 projection/fusion', 'S5 training', 'S6 search', 'S7 confirmation', 'S8 ensemble', 'S9 evaluation/report']}
    atomic_json(run / 'completion_status.json', blockers)
    print(json.dumps(blockers, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
