"""Speed test: A1 HotFlip (1 gradient) vs A3 fw_avg_pamp (3-gradient path) on 10 sequences."""
import importlib.util, json, time
from pathlib import Path
import numpy as np, torch
from attack_adapter import Engine, AttackRow, mutate
spec = importlib.util.spec_from_file_location('multi_run', Path(__file__).resolve().parent / 'run.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
HERE = Path(__file__).resolve().parent
frame, p, s, _ = m.load_data('catapro', cohort_path=HERE / 'results_no_esm_lm_2754/cohort.csv')
ok = np.where(frame.sequence_clean.str.len().between(150, 500))[0]
idx = np.sort(np.random.default_rng(20260920).choice(ok, 11, replace=False))
warm, idx = idx[0], idx[1:]
engine = Engine()
sync = torch.cuda.synchronize

def run(i, method):
    seq = frame.sequence_clean.iloc[i]
    row = AttackRow(engine, seq, p[i], s[i])
    sync(); t = time.perf_counter()
    choices, _, _ = row.proposals(seq, 'residual', 1, 1, methods=[method])
    sync(); dt = time.perf_counter() - t
    pos, aa = choices[method][0]
    return dt, f'{seq[pos]}{pos+1}{aa}', len(seq)

run(warm, 'A1_hotflip'); run(warm, 'A3_fw_avg_pamp')          # warm-up, discarded
res = []
for k, i in enumerate(idx):
    order = ['A1_hotflip', 'A3_fw_avg_pamp'] if k % 2 == 0 else ['A3_fw_avg_pamp', 'A1_hotflip']
    r = {'row_id': int(frame.row_id.iloc[i])}
    for meth in order:
        dt, mut, L = run(i, meth)
        r[meth] = dt; r[meth + '_mut'] = mut; r['L'] = L
    res.append(r)
    print(r, flush=True)
a1 = np.array([r['A1_hotflip'] for r in res]); a3 = np.array([r['A3_fw_avg_pamp'] for r in res])
summ = dict(n=len(res), mean_L=float(np.mean([r['L'] for r in res])),
            A1_mean_s=float(a1.mean()), A1_sd=float(a1.std(ddof=1)),
            A3_mean_s=float(a3.mean()), A3_sd=float(a3.std(ddof=1)),
            ratio_A3_over_A1=float(a3.mean() / a1.mean()),
            same_mutation_count=int(sum(r['A1_hotflip_mut'] == r['A3_fw_avg_pamp_mut'] for r in res)))
(HERE / 'speed_a1_a3.json').write_text(json.dumps(dict(summary=summ, rows=res), indent=2))
print(json.dumps(summ, indent=2))
