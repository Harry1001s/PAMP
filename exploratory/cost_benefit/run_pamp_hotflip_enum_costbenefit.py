#!/usr/bin/env python
"""Time-vs-benefit comparison of PAMP, HotFlip and batched exhaustive enumeration.

On a small, pre-drawn subset of the frozen test cohort we measure, per sequence:
  * PAMP / HotFlip : one fp16 fwd+bwd through ESM2 input embeddings (the existing
                     compact-attack gradient code) and closed-form [L,20] scoring;
  * enumeration    : every one of the 19*L single substitutions re-encoded with FP32 ESM2
                     in same-length, token-budgeted batches and scored by the frozen
                     ensemble (mutant-minus-WT mean update on the cached WT mean, i.e. the
                     exact contract of run_compact_topk_adversarial_attack.py).
The enumeration table is the clean oracle: PAMP/HotFlip picks (and their top-K shortlists)
are looked up in it, so gain/regret/percentile are exact, while their wall-clock is
measured separately. Labels are never used. Evaluation only; not a generation method.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_compact_topk_adversarial_attack as A

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiment_pamp_hotflip_enum_costbenefit_v1'
AAS = A.CANONICAL_AAS
SEED = 20260920


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def pick_subset(frame, n, lo, hi):
    pool = frame[frame.attack_eligible & frame.sequence_clean.str.len().between(lo, hi)]
    pool = pool.drop_duplicates('sequence_clean')
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return pool.iloc[np.sort(idx)].reset_index(drop=True)


# --------------------------------------------------------------------------- gradient
def gradient_scores(row, source_index, arrays, esm, converter, ensemble, aa_emb,
                    pair_dist2, median_distance, device):
    """Return (pamp[L,20], hotflip[L,20]) with the WT column masked to -inf."""
    sequence = row.sequence_clean
    L = len(sequence)
    hcache = torch.tensor(arrays[0][source_index], device=device).reshape(1, -1)
    substrate = torch.tensor(np.asarray(arrays[1][source_index]), device=device).reshape(1, -1)
    fp = torch.tensor(np.asarray(arrays[2][source_index]), device=device,
                      dtype=torch.float32).reshape(1, -1)
    _, _, tokens = converter([('protein', sequence)])
    tokens = tokens.to(device)
    with A.CaptureInputEmbedding(esm.embed_tokens) as cap:
        out = esm(tokens, repr_layers=[33], return_contacts=False)
        fresh = out['representations'][33][0, 1:L + 1].float().mean(0, keepdim=True)
        value = ensemble.predict_log2(hcache + (fresh - fresh.detach()), substrate, fp).reshape(())
        grad = torch.autograd.grad(value, cap.tensor)[0][0, 1:L + 1].float()
        wt_emb = cap.tensor.detach()[0, 1:L + 1].float()
    direction = grad / grad.flatten().norm().clamp_min(1e-12)
    lin = direction @ aa_emb.T - (direction * wt_emb).sum(1, keepdim=True)
    raw = grad @ aa_emb.T - (grad * wt_emb).sum(1, keepdim=True)
    wt_idx = torch.tensor([AAS.index(a) for a in sequence], device=device)
    eps = A.RADIUS_SCALE * median_distance * math.sqrt(L)
    pamp = 2 * eps * lin - pair_dist2[wt_idx]
    pamp, raw = pamp.clone(), raw.clone()
    ar = torch.arange(L, device=device)
    pamp[ar, wt_idx] = -float('inf')
    raw[ar, wt_idx] = -float('inf')
    return pamp.cpu().numpy(), raw.cpu().numpy(), wt_idx.cpu().numpy()


def shortlist(score, k):
    """Top-k distinct sites by best alternative AA -> list of (pos, aa_index)."""
    best_aa = np.argmax(score, axis=1)
    best = score[np.arange(len(score)), best_aa]
    order = np.lexsort((np.arange(len(score)), -best))[:k]
    return [(int(p), int(best_aa[p])) for p in order]


# --------------------------------------------------------------------------- enumeration
@torch.inference_mode()
def enumerate_all(esm, alphabet, converter, ensemble, row, source_index, arrays, wt_idx,
                  device, max_tokens, aa_tok):
    """Delta log2 kcat for all L x 20 substitutions (WT column = nan). FP or fp16 esm."""
    sequence = row.sequence_clean
    L = len(sequence)
    _, _, tokens = converter([('protein', sequence)])
    tokens = tokens.to(device)
    dtype = next(esm.parameters()).dtype
    cached = torch.tensor(arrays[0][source_index], device=device, dtype=torch.float32).reshape(1, -1)
    substrate = torch.tensor(np.asarray(arrays[1][source_index]), device=device).reshape(1, -1)
    fp = torch.tensor(np.asarray(arrays[2][source_index]), device=device,
                      dtype=torch.float32).reshape(1, -1)

    def mean_repr(tok):
        rep = esm(tok, repr_layers=[33], return_contacts=False)['representations'][33]
        return rep[:, 1:L + 1].float().mean(1)

    wt_mean = mean_repr(tokens)                         # [1,1280] same numerics as mutants
    pos_list = [(p, a) for p in range(L) for a in range(20) if a != wt_idx[p]]
    delta = np.full((L, 20), np.nan, dtype=np.float64)
    batch = max(1, max_tokens // (L + 2))
    for s in range(0, len(pos_list), batch):
        chunk = pos_list[s:s + batch]
        tok = tokens.repeat(len(chunk), 1)
        rows = torch.arange(len(chunk), device=device)
        cols = torch.tensor([p + 1 for p, _ in chunk], device=device)
        tok[rows, cols] = aa_tok[[a for _, a in chunk]]
        means = cached + (mean_repr(tok) - wt_mean)
        pred = ensemble.predict_log2(means, substrate.expand(len(chunk), -1),
                                     fp.expand(len(chunk), -1)).reshape(-1).double().cpu().numpy()
        for (p, a), v in zip(chunk, pred):
            delta[p, a] = v
    return delta   # absolute predictions; caller subtracts cached WT prediction


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--len-min', type=int, default=150)
    ap.add_argument('--len-max', type=int, default=350)
    ap.add_argument('--max-tokens', type=int, default=16384)
    ap.add_argument('--tune', action='store_true', help='only tune batch token budget')
    ap.add_argument('--fp16-check', type=int, default=5)
    ap.add_argument('--limit', type=int, default=0, help='debug: stop after this many rows')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(exist_ok=True)
    (OUT / 'rows').mkdir(exist_ok=True)

    frame, test, protein, smiles, maccs = A.load_inputs()
    cohort = pd.read_csv(A.ROOT / 'experiment_compact_topk_attack_v1/test_cohort.csv',
                         keep_default_na=False)
    frame['cached_wt_prediction_log2'] = cohort.cached_wt_prediction_log2.to_numpy()
    subset = pick_subset(frame, args.n, args.len_min, args.len_max)
    if args.limit:
        subset = subset.iloc[:args.limit]
    arrays = (protein, smiles, maccs)
    print('SUBSET', len(subset), 'mean L', subset.sequence_clean.str.len().mean(), flush=True)

    ensemble = A.ImprovedKcatEnsemble.from_manifest(A.MANIFEST, device=device)
    esm16, alphabet, converter = A.load_esm(torch.float16, device)
    aa_ids = torch.tensor([alphabet.get_idx(a) for a in AAS], device=device)
    aa_emb = esm16.embed_tokens.weight.index_select(0, aa_ids).detach().float()
    pair_dist2 = torch.cdist(aa_emb, aa_emb).square()
    up = torch.triu_indices(20, 20, offset=1, device=device)
    median_distance = float(pair_dist2[up[0], up[1]].sqrt().median().cpu())
    esm32, _, _ = A.load_esm(torch.float32, device)

    if args.tune:
        row = next(subset.itertuples())
        si = int(test[int(row.test_offset)])
        _, _, wt_idx = gradient_scores(row, si, arrays, esm16, converter, ensemble, aa_emb,
                                       pair_dist2, median_distance, device)
        L = len(row.sequence_clean)
        print('tune on L =', L)
        for mt in (2048, 4096, 8192, 16384, 32768, 65536):
            for name, m in (('fp32', esm32), ('fp16', esm16)):
                try:
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    sync(); t = time.perf_counter()
                    enumerate_all(m, alphabet, converter, ensemble, row, si, arrays, wt_idx,
                                  device, mt, aa_ids)
                    sync(); dt = time.perf_counter() - t
                    print(f'{name} max_tokens={mt:6d} batch={mt // (L + 2):4d} '
                          f'{19 * L / dt:8.1f} mut/s  peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB',
                          flush=True)
                except torch.cuda.OutOfMemoryError:
                    print(f'{name} max_tokens={mt} OOM', flush=True)
        return

    # warm-up (kernels, cudnn, allocator) on the first row, results discarded
    w = next(subset.itertuples())
    wsi = int(test[int(w.test_offset)])
    gradient_scores(w, wsi, arrays, esm16, converter, ensemble, aa_emb, pair_dist2,
                    median_distance, device)
    enumerate_all(esm32, alphabet, converter, ensemble,
                  w._replace(sequence_clean=w.sequence_clean[:40]), wsi, arrays,
                  np.array([AAS.index(a) for a in w.sequence_clean[:40]]), device,
                  args.max_tokens, aa_ids)
    sync()

    for r in subset.itertuples():
        path = OUT / 'rows' / f'row_{int(r.row_id)}.json'
        if path.exists():
            continue
        si = int(test[int(r.test_offset)])
        L = len(r.sequence_clean)
        sync(); t0 = time.perf_counter()
        pamp, hot, wt_idx = gradient_scores(r, si, arrays, esm16, converter, ensemble, aa_emb,
                                            pair_dist2, median_distance, device)
        sync(); t_grad = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = enumerate_all(esm32, alphabet, converter, ensemble, r, si, arrays, wt_idx,
                             device, args.max_tokens, aa_ids)
        sync(); t_enum = time.perf_counter() - t0
        wt_pred = float(r.cached_wt_prediction_log2)
        delta = pred - wt_pred
        np.save(OUT / 'rows' / f'delta_{int(r.row_id)}.npy', delta.astype(np.float32))
        # cost of one clean re-encode of a single chosen mutant (batch of 1 + WT mean)
        sync(); t0 = time.perf_counter()
        p0, a0 = shortlist(pamp, 1)[0]
        with torch.inference_mode():
            _, _, tok = converter([('protein', r.sequence_clean)])
            tok = tok.to(device).repeat(2, 1)
            tok[1, p0 + 1] = aa_ids[a0]
            rep = esm32(tok, repr_layers=[33], return_contacts=False)['representations'][33]
            m = rep[:, 1:L + 1].float().mean(1)
            cached = torch.tensor(protein[si], device=device).reshape(1, -1)
            ensemble.predict_log2(cached + (m[1:] - m[:1]),
                                  torch.tensor(np.asarray(smiles[si]), device=device).reshape(1, -1),
                                  torch.tensor(np.asarray(maccs[si]), device=device,
                                               dtype=torch.float32).reshape(1, -1)).cpu()
        sync(); t_clean1 = time.perf_counter() - t0

        rec = {'row_id': int(r.row_id), 'L': L, 'n_candidates': 19 * L,
               't_grad_s': t_grad, 't_enum_s': t_enum, 't_clean_confirm_s': t_clean1,
               'wt_pred': wt_pred}
        valid = delta[~np.isnan(delta)]
        rec['oracle_max'] = float(valid.max())
        rec['mean_all'] = float(valid.mean())
        for name, score in (('pamp', pamp), ('hotflip', hot)):
            rec[name] = {str(k): [(p, a, float(delta[p, a])) for p, a in shortlist(score, k)]
                         for k in (1, 5, 20, 50)}
        # descending sorted values, for percentile ranks
        rec['sorted_desc_head'] = np.sort(valid)[::-1][:int(0.10 * len(valid)) + 1].tolist()
        rec['n_valid'] = int(len(valid))
        if args.fp16_check and len([f for f in (OUT / 'rows').glob('row_*.json')]) < args.fp16_check:
            sync(); t0 = time.perf_counter()
            p16 = enumerate_all(esm16, alphabet, converter, ensemble, r, si, arrays, wt_idx,
                                device, args.max_tokens * 2, aa_ids)
            sync(); rec['t_enum_fp16_s'] = time.perf_counter() - t0
            d16 = p16 - wt_pred
            mask = ~np.isnan(delta)
            rec['fp16_spearman'] = spearman(delta[mask], d16[mask])
            rec['fp16_argmax_same'] = bool(np.nanargmax(delta) == np.nanargmax(d16))
            rec['fp16_max_abs_err'] = float(np.abs(delta[mask] - d16[mask]).max())
        (path).write_text(json.dumps(rec))
        print(f"ROW {int(r.row_id)} L={L} grad={t_grad:.2f}s enum={t_enum:.1f}s "
              f"oracle={rec['oracle_max']:.3f} pamp1={rec['pamp']['1'][0][2]:.3f} "
              f"hot1={rec['hotflip']['1'][0][2]:.3f}", flush=True)


if __name__ == '__main__':
    main()
