#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from predictor_ablation.config import base_config
from predictor_ablation.reproducibility import atomic_json
from predictor_ablation.training import train_one


def main():
    parser = argparse.ArgumentParser(description='Correctness and interrupted/resumed training on a train-only subset')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--device', default='cpu')
    a = parser.parse_args()
    if a.dry_run:
        print('Smoke: one seed, 4-epoch uninterrupted fit + 2+2-epoch resume fit; 160 train rows only')
        return
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.discover(str(HERE / 'tests'))
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    config = base_config(HERE)
    config.update(hidden_dims=[32, 32], device=a.device, micro_batch_size=32, effective_batch_size=32)
    directory = HERE / 'runs' / a.experiment / 'smoke'
    manifest = HERE / 'configs/data_manifest.json'
    train_one(manifest, config, directory / 'uninterrupted', 42, resume=a.resume, pilot_epochs=4)
    if not (directory / 'resumed' / 'last.pt').exists():
        train_one(manifest, config, directory / 'resumed', 42, pilot_epochs=4, stop_after_epoch=2)
    train_one(manifest, config, directory / 'resumed', 42, resume=True, pilot_epochs=4)
    x = torch.load(directory / 'uninterrupted/last.pt', map_location='cpu', weights_only=False)
    y = torch.load(directory / 'resumed/last.pt', map_location='cpu', weights_only=False)
    for key in x['model_state_dict']:
        torch.testing.assert_close(x['model_state_dict'][key], y['model_state_dict'][key], rtol=0, atol=0)
    if x['best_epoch'] != y['best_epoch'] or x['patience_count'] != y['patience_count']:
        raise AssertionError('Resume early-stop state mismatch')
    report = {'status': 'passed', 'device': a.device, 'test_labels_accessed': False,
              'subset': '128 training + 32 holdout rows taken only from historical train',
              'resume_weights_bitwise_equal': True, 'formal_fits_completed': 0}
    atomic_json(directory / 'smoke_results.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
