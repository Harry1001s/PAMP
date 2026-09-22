#!/usr/bin/env python
"""Cross-predictor transfer check: re-score the FROZEN 2754 attack outputs with Extra Trees.

Nothing is re-selected. For every record and method we take the already-chosen round-1 edit
(Top1 and best-of-5) from results_no_esm_lm_2754/endpoint_results.csv, re-encode the mutant with
the same FP32 ESM2 path as run.py, and score it with the frozen Extra Trees (2304 = ESM2 mean
1280 + UniKP 1024) that never saw these held-out test pairs. The Residual value is recomputed
too and must reproduce the stored one (integrity gate).
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from attack_adapter import Engine, AttackRow, mutate

# attack_adapter puts another directory's run.py on sys.path; load this folder's explicitly.
import importlib.util
_spec = importlib.util.spec_from_file_location('multi_run', Path(__file__).resolve().parent / 'run.py')
_multi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_multi)
load_data, ET, sha = _multi.load_data, _multi.ET, _multi.sha

HERE = Path(__file__).resolve().parent
SRC = HERE / 'results_no_esm_lm_2754'
OUT = HERE / 'cross_predictor_transfer_20260920'
ENDPOINTS = ('round1_top1', 'round1_best_of_top5')


def parse_edit(sequence, edits):
    (edit,) = json.loads(edits)
    wt, pos, aa = edit[0], int(edit[1:-1]) - 1, edit[-1]
    assert sequence[pos] == wt
    return mutate(sequence, pos, aa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    cohort = SRC / 'cohort.csv'
    frame, p, s, _ = load_data('catapro', cohort_path=cohort)
    end = pd.read_csv(SRC / 'endpoint_results.csv')
    end = end[end.endpoint.isin(ENDPOINTS)]
    assert set(end.row_id) == set(frame.row_id) and len(frame) == 2754
    tree = joblib.load(ET)
    tree.n_jobs = 4
    engine = Engine()
    out_path = OUT / 'transfer_rows.jsonl'
    done = set()
    if out_path.exists():
        done = {json.loads(l)['row_id'] for l in out_path.read_text().splitlines() if l}
    groups = {rid: g for rid, g in end.groupby('row_id')}
    t0, n = time.time(), 0
    with out_path.open('a') as fh:
        for i, rec in enumerate(frame.itertuples()):
            if rec.row_id in done:
                continue
            if args.limit and n >= args.limit:
                break
            seq, rid = rec.sequence_clean, int(rec.row_id)
            row = AttackRow(engine, seq, p[i], s[i])
            cache = {}

            def score(sequence):
                if sequence not in cache:
                    H = row.live0 if sequence == seq else engine.encode(sequence)[0]
                    values, mean = row.values(H)
                    et = float(tree.predict(np.concatenate([mean, s[i]])[None])[0])
                    cache[sequence] = (values['residual'], et)
                return cache[sequence]

            wt_res, wt_et = score(seq)
            recs = []
            for r in groups[rid].itertuples():
                mut = parse_edit(seq, r.edits)
                m_res, m_et = score(mut)
                recs.append(dict(method=r.method, endpoint=r.endpoint, edits=r.edits,
                                 stored_res_delta=float(r.delta_log2),
                                 res_delta=m_res - wt_res, et_delta=m_et - wt_et,
                                 wt_stored=float(r.wt_log2), wt_res=wt_res, wt_et=wt_et))
            fh.write(json.dumps(dict(row_id=rid, L=len(seq), records=recs)) + '\n')
            fh.flush()
            n += 1
            if n == 1 or n % 50 == 0:
                el = time.time() - t0
                print(f'{n} rows  {el:.0f}s  eta {(el / n) * (len(frame) - len(done) - n) / 60:.1f} min',
                      flush=True)
    (OUT / 'provenance.json').write_text(json.dumps(dict(
        extra_trees=str(ET), extra_trees_sha256=sha(ET), source=str(SRC),
        source_endpoint_sha256=sha(SRC / 'endpoint_results.csv'),
        note='Frozen selections re-scored; no re-selection; ET is a transfer evaluator only'), indent=2))
    print('DONE', n, flush=True)


if __name__ == '__main__':
    main()
