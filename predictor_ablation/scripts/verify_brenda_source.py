#!/usr/bin/env python
"""Verify the user-identified BRENDA export without changing training inputs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from predictor_ablation.reproducibility import atomic_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    source = Path(args.source).resolve()
    manifest_path = HERE / 'configs/data_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    original = pd.read_csv(source, skiprows=1)
    filtered = pd.read_excel(manifest['source_table'])
    keys = ['organism', 'smiles', 'enzyme']
    kept = original[original.enzyme.str.len() <= 1021]
    if len(kept) != len(filtered):
        raise ValueError('Length-filter reconstruction row count mismatch')
    for column in keys:
        if not np.array_equal(kept[column].to_numpy(), filtered[column].to_numpy()):
            raise ValueError(f'Ordered source reconstruction mismatch: {column}')
    delta = np.abs(kept.log2Kcat.to_numpy() - filtered.log2Kcat.to_numpy())
    if not np.all(delta <= 1e-10):
        raise ValueError('Source labels differ beyond CSV round-off tolerance')
    lookup = defaultdict(list)
    for row, record in original.iterrows():
        lookup[tuple(record[k] for k in keys)].append((int(row), float(record.log2Kcat)))
    mappings = []
    for row, record in filtered.iterrows():
        matches = [j for j, y in lookup[tuple(record[k] for k in keys)]
                   if abs(y - float(record.log2Kcat)) <= 1e-10]
        mappings.append({'sample_id': f'row_{row:06d}', 'cache_row': row,
                         'brenda_row_reconstructed_by_order': int(kept.index[row]),
                         'equivalent_brenda_rows': json.dumps(matches),
                         'n_equivalent_source_rows': len(matches)})
    audit = HERE / 'runs' / args.experiment / 'audit'
    pd.DataFrame(mappings).to_csv(audit / 'brenda_source_row_mapping.csv', index=False)
    report = {
        'source': str(source), 'source_sha256': sha256_file(source),
        'source_rows': len(original), 'cached_rows': len(filtered),
        'excluded_by_reconstructed_length_rule': len(original) - len(kept),
        'reconstruction_rule': 'enzyme length <= 1021, original row order, no label transformation',
        'reconstruction_status': 'exact text columns and labels within absolute 1e-10',
        'label_max_abs_csv_roundoff': float(delta.max()),
        'source_label_column': 'log2Kcat',
        'unit': 's^-1',
        'unit_evidence': 'Inferred from user-confirmed BRENDA provenance and official turnover-number field definition; export has no explicit physical-unit column',
        'unit_reference': 'https://www.brenda-enzymes.org/datafields.php',
        'measurement_id_present': False, 'literature_or_assay_conditions_present': False,
        'duplicate_measurement_identity': 'unresolved: distinct CSV rows do not establish distinct biological measurements',
    }
    atomic_json(audit / 'brenda_source_verification.json', report)
    manifest.update(original_source=str(source), original_source_sha256=report['source_sha256'],
                    target_unit='s^-1', target_unit_evidence=report['unit_evidence'],
                    target_unit_reference=report['unit_reference'],
                    target_unit_verified=False, target_unit_status='source_convention_supported')
    atomic_json(manifest_path, manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
