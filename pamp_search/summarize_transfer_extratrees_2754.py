#!/usr/bin/env python
"""Summarize the cross-predictor (Residual -> Extra Trees) transfer check."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
OUT = HERE / 'cross_predictor_transfer_20260920'
cohort = pd.read_csv(HERE / 'results_no_esm_lm_2754/cohort.csv')
seq = dict(zip(cohort.row_id, cohort.sequence_clean))

rows = []
for line in (OUT / 'transfer_rows.jsonl').read_text().splitlines():
    r = json.loads(line)
    for x in r['records']:
        rows.append(dict(row_id=r['row_id'], L=r['L'], **x))
df = pd.DataFrame(rows)
df['cluster'] = df.row_id.map(seq)
assert np.abs(df.res_delta - df.stored_res_delta).max() < 1e-6, 'Residual values no longer reproduce'
n_rows = df.row_id.nunique()
df.to_csv(OUT / 'transfer_records.csv', index=False)


def cluster_boot(frame, col, B=4000, seed=0):
    """95% CI of the mean, resampling whole protein sequences."""
    g = frame.groupby('cluster')[col].agg(['sum', 'count'])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(B, len(g)))
    means = g['sum'].to_numpy()[idx].sum(1) / g['count'].to_numpy()[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_boot(frame_a, frame_b, col, B=4000, seed=0):
    m = frame_a[['row_id', 'cluster', col]].merge(frame_b[['row_id', col]], on='row_id', suffixes=('_a', '_b'))
    m['d'] = m[col + '_a'] - m[col + '_b']
    return float(m.d.mean()), *cluster_boot(m, 'd', B, seed)


order = ['A3_fw_avg_pamp', 'A0_pamp', 'A2_fw_avg', 'A1_hotflip', 'B2_random']
table = []
for ep in ('round1_top1', 'round1_best_of_top5'):
    for m in order:
        f = df[(df.endpoint == ep) & (df.method == m)]
        if f.empty:
            continue
        lo, hi = cluster_boot(f, 'et_delta')
        rlo, rhi = cluster_boot(f, 'res_delta')
        table.append(dict(endpoint=ep, method=m, N=len(f),
                          res_mean=f.res_delta.mean(), res_ci=f'[{rlo:.3f},{rhi:.3f}]',
                          res_pos=(f.res_delta > 0).mean(),
                          et_mean=f.et_delta.mean(), et_ci=f'[{lo:.3f},{hi:.3f}]',
                          et_geo_lift_pct=100 * (2 ** f.et_delta.mean() - 1),
                          et_pos=(f.et_delta > 0).mean(), et_zero=(f.et_delta == 0).mean(),
                          retained=f.et_delta.mean() / f.res_delta.mean(),
                          spearman_res_et=spearmanr(f.res_delta, f.et_delta)[0],
                          sign_agree=(np.sign(f.res_delta) == np.sign(f.et_delta)).mean()))
t = pd.DataFrame(table)
t.to_csv(OUT / 'transfer_summary.csv', index=False)

paired = []
for ep in ('round1_top1', 'round1_best_of_top5'):
    rnd = df[(df.endpoint == ep) & (df.method == 'B2_random')]
    for m in order[:-1]:
        f = df[(df.endpoint == ep) & (df.method == m)]
        if f.empty or rnd.empty:
            continue
        d, lo, hi = paired_boot(f, rnd, 'et_delta')
        paired.append(dict(endpoint=ep, contrast=f'{m} - B2_random (ET)', diff=d, ci_lo=lo, ci_hi=hi))
    a = df[(df.endpoint == ep) & (df.method == 'A3_fw_avg_pamp')]
    for m in ('A1_hotflip',):
        f = df[(df.endpoint == ep) & (df.method == m)]
        if len(f) and len(a):
            d, lo, hi = paired_boot(a, f, 'et_delta')
            paired.append(dict(endpoint=ep, contrast='A3_fw_avg_pamp - A1_hotflip (ET)', diff=d, ci_lo=lo, ci_hi=hi))
p = pd.DataFrame(paired)
p.to_csv(OUT / 'transfer_paired.csv', index=False)

pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.float_format', '{:.4f}'.format)
print('rows scored:', n_rows)
print(t.to_string(index=False))
print(p.to_string(index=False))
