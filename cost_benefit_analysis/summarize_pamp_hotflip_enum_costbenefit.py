#!/usr/bin/env python
"""Aggregate per-row results of run_pamp_hotflip_enum_costbenefit.py into a cost-benefit table."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / 'experiment_pamp_hotflip_enum_costbenefit_v1'
rows = [json.loads(p.read_text()) for p in sorted((OUT / 'rows').glob('row_*.json'))]
n = len(rows)
L = np.array([r['L'] for r in rows])
t_grad = np.array([r['t_grad_s'] for r in rows])
t_enum = np.array([r['t_enum_s'] for r in rows])
t_conf = np.array([r['t_clean_confirm_s'] for r in rows])
oracle = np.array([r['oracle_max'] for r in rows])
mean_all = np.array([r['mean_all'] for r in rows])
n_cand = np.array([r['n_candidates'] for r in rows])
per_mut = t_enum / n_cand                       # measured seconds per clean mutant in the batch
deltas = {r['row_id']: np.load(OUT / 'rows' / f"delta_{r['row_id']}.npy") for r in rows}


def picks(r, name, k):
    return [d for _, _, d in r[name][str(k)]]


def percentile_top(r, value):
    """Fraction (0-1) of the 19L candidates whose delta is >= value (smaller = better)."""
    d = deltas[r['row_id']]
    d = d[~np.isnan(d)]
    return float((d >= value - 1e-9).mean())


def boot(x, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(np.asarray(x), size=(B, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


records = []


def add(label, gain, seconds, forwards, extra=None):
    gain, seconds = np.asarray(gain), np.asarray(seconds)
    regret = oracle - gain
    lo, hi = boot(gain)
    rec = {'method': label, 'mean_delta_log2': gain.mean(), 'ci_lo': lo, 'ci_hi': hi,
           'geo_lift_pct': 100 * (2 ** gain.mean() - 1),
           'frac_of_oracle_gain': gain.mean() / oracle.mean(),
           'mean_regret': regret.mean(), 'seconds_per_seq': seconds.mean(),
           'model_forwards_per_seq': forwards, 'gain_per_second': gain.mean() / seconds.mean()}
    if extra:
        rec.update(extra)
    records.append(rec)


def rank_stats(name, k=1):
    top = np.array([percentile_top(r, picks(r, name, 1)[0]) for r in rows])
    return {'median_top_pct': 100 * np.median(top), 'hit_top1pct': (top <= 0.01).mean(),
            'hit_top5pct': (top <= 0.05).mean(), 'hit_top10pct': (top <= 0.10).mean()}


# clean confirmation = one extra FP32 forward pair (WT + mutant) for the single chosen mutant
for name in ('pamp', 'hotflip'):
    add(f'{name.upper()} top-1', [picks(r, name, 1)[0] for r in rows], t_grad + t_conf, '1 fwd+bwd(fp16) + 1',
        rank_stats(name))
for name in ('pamp', 'hotflip'):
    for k in (5, 20, 50):
        gain = [max(picks(r, name, k)) for r in rows]
        # gradient pass + k exact queries priced at the measured per-mutant batched cost
        add(f'{name.upper()} top-{k} + exact rerank', gain, t_grad + k * per_mut, f'1 fwd+bwd(fp16) + {k}')
add('Batched full enumeration (oracle)', oracle, t_enum, '19L', {
    'median_top_pct': 0.0, 'hit_top1pct': 1.0, 'hit_top5pct': 1.0, 'hit_top10pct': 1.0})
# expectation for k uniformly random distinct candidates: E[max of k draws] from the oracle table
rng = np.random.default_rng(1)
for k in (1, 5, 20):
    g = []
    for r in rows:
        d = deltas[r['row_id']]
        d = d[~np.isnan(d)]
        g.append(np.mean([rng.choice(d, size=k, replace=False).max() for _ in range(400)]))
    add(f'Random top-{k} (expected)', g, k * per_mut, f'{k}')

table = pd.DataFrame(records)
table['speedup_vs_enum'] = t_enum.mean() / table.seconds_per_seq
table.to_csv(OUT / 'costbenefit_table.csv', index=False)

per_row = pd.DataFrame({'row_id': [r['row_id'] for r in rows], 'L': L, 'n_candidates': n_cand,
                        't_grad_s': t_grad, 't_enum_s': t_enum, 'enum_over_grad': t_enum / t_grad,
                        'oracle_max': oracle, 'mean_all_candidates': mean_all,
                        'pamp1': [picks(r, 'pamp', 1)[0] for r in rows],
                        'hotflip1': [picks(r, 'hotflip', 1)[0] for r in rows]})
per_row.to_csv(OUT / 'per_row.csv', index=False)

fp16 = [r for r in rows if 't_enum_fp16_s' in r]
summary = {'n_sequences': n, 'mean_L': float(L.mean()), 'mean_candidates': float(n_cand.mean()),
           'enum_ms_per_mutant_fp32': 1000 * float(per_mut.mean()),
           'grad_pass_s_mean': float(t_grad.mean()), 'enum_s_mean': float(t_enum.mean()),
           'enum_over_grad_ratio': float(t_enum.mean() / t_grad.mean()),
           'oracle_mean_delta_log2': float(oracle.mean()),
           'random_single_mutant_mean_delta_log2': float(mean_all.mean()),
           'pamp_vs_hotflip_top1_paired_diff': float(np.mean(
               [picks(r, 'pamp', 1)[0] - picks(r, 'hotflip', 1)[0] for r in rows]))}
if fp16:
    summary['fp16_enumeration'] = {
        'n': len(fp16),
        'speedup_vs_fp32': float(np.mean([r['t_enum_s'] / r['t_enum_fp16_s'] for r in fp16])),
        'mean_spearman': float(np.mean([r['fp16_spearman'] for r in fp16])),
        'argmax_same_frac': float(np.mean([r['fp16_argmax_same'] for r in fp16])),
        'max_abs_err': float(max(r['fp16_max_abs_err'] for r in fp16))}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.float_format', '{:.4f}'.format)
print(table[['method', 'mean_delta_log2', 'ci_lo', 'ci_hi', 'frac_of_oracle_gain', 'mean_regret',
             'seconds_per_seq', 'speedup_vs_enum', 'gain_per_second', 'median_top_pct',
             'hit_top5pct']].to_string(index=False))
print(json.dumps(summary, indent=2))

# Pareto plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.5, 4.2))
style = {'PAMP': ('#1f77b4', 'o'), 'HOTFLIP': ('#d62728', 's'), 'Batched': ('#2ca02c', '*'),
         'Random': ('#7f7f7f', 'x')}
for _, r in table.iterrows():
    key = next(k for k in style if r.method.upper().startswith(k.upper()))
    c, m = style[key]
    ax.scatter(r.seconds_per_seq, r.mean_delta_log2, c=c, marker=m, s=70 if m != '*' else 160)
    ax.annotate(r.method.replace(' + exact rerank', '').replace(' (expected)', ''),
                (r.seconds_per_seq, r.mean_delta_log2), fontsize=7, xytext=(4, -9),
                textcoords='offset points')
ax.set_xscale('log')
ax.set_xlabel('wall-clock per sequence (s, log scale)')
ax.set_ylabel('mean clean Δlog2 kcat (frozen ensemble)')
ax.set_title(f'Cost-benefit on {n} test sequences (mean L={L.mean():.0f})', fontsize=10)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT / 'costbenefit.png', dpi=200)
