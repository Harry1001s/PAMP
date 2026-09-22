#!/usr/bin/env python
"""Discrete Top-1/Top-5 attacks on the frozen R2=0.618754 ensemble.

The attack never changes model weights and never uses observed kcat labels to
choose mutations. Candidate generation differentiates through frozen ESM2 input
embeddings. Every reported mutation is then re-encoded as an actual sequence;
Top-5 chooses the largest clean, re-encoded prediction among five distinct sites.

The four predeclared methods are:
  pamp     : normalized activity gradient with the fixed proximal distance term;
  hotflip  : raw first-order activity gradient (PAMP distance-term ablation);
  esm_lm   : ESM2 alternative-amino-acid logit margin (no kcat gradient);
  random   : deterministic random positions and alternative amino acids.

All methods receive the same one or five exact predictor queries. PAMP/HotFlip
also use a white-box gradient, so equal query count is not equal wall-clock cost.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl
for _c in _pl.Path(__file__).resolve().parents:
    if (_c / 'pamp_paths.py').exists():
        _sys.path.insert(0, str(_c)); break
from pamp_paths import (ESM2_CHECKPOINT, KCAT_CSV, add_module_paths,
                        ORIGINAL_PREDICTOR_DIR)
add_module_paths(ORIGINAL_PREDICTOR_DIR)

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from generate_full_esm2_embeddings_pkl import clean_sequence
from improved_kcat_ensemble import ImprovedKcatEnsemble


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiment_compact_topk_attack_v1'
SOURCE = KCAT_CSV
MANIFEST = ROOT / 'experiment_compact_improvements/improved_ensemble_manifest.json'
SPLIT = ROOT / 'experiment_mean/split_indices.npz'
DATA_MANIFEST = ROOT / 'experiment_mean/data_manifest.csv'
PROTEIN = ROOT / 'catpro_esm2_mean_pooling/protein_mean_embs.pkl'
SMILES = ROOT / 'experiment_mean/smiles_unikp1024.npy'
MACCS = ROOT / 'experiment_compact_improvements/maccs.npy'
WEIGHTS = ESM2_CHECKPOINT
METHODS = ('pamp', 'hotflip', 'esm_lm', 'random')
CANONICAL_AAS = tuple('ACDEFGHIKLMNPQRSTVWY')
CANONICAL_SET = set(CANONICAL_AAS)
TOP_K = 5
RADIUS_SCALE = 2.0
SEED = 20260909
MAX_CHUNK = 1022
EXPECTED_R2 = 0.6187542060917297


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n')
    os.replace(temporary, path)


def apply_mutation(sequence, position, mutant):
    if mutant == sequence[position] or mutant not in CANONICAL_SET:
        raise ValueError('A point mutation must replace the WT canonical residue')
    return sequence[:position] + mutant + sequence[position + 1:]


def r2(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    return float(1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2))


def load_inputs():
    source = pd.read_csv(SOURCE)
    frame = pd.read_csv(DATA_MANIFEST, keep_default_na=False)
    split = np.load(SPLIT)
    test = split['test_idx'].astype(np.int64)
    if not np.array_equal(frame.row_id, np.arange(len(source))):
        raise ValueError('Data manifest row order changed')
    if not np.array_equal(frame.Sequence, source.Sequence):
        raise ValueError('Data manifest sequence mapping changed')
    if not np.array_equal(frame.Smiles, source.Smiles):
        raise ValueError('Data manifest substrate mapping changed')
    with PROTEIN.open('rb') as handle:
        protein = np.asarray(pickle.load(handle), dtype=np.float32)
    smiles = np.load(SMILES, mmap_mode='r')
    maccs = np.load(MACCS, mmap_mode='r')
    if protein.shape != (len(frame), 1280) or smiles.shape != (len(frame), 1024):
        raise ValueError('Feature shapes changed')
    if maccs.shape != (len(frame), 167):
        raise ValueError('MACCS shape changed')
    if not np.isfinite(protein).all() or not np.isfinite(smiles).all():
        raise ValueError('Nonfinite frozen features')
    selected = frame.iloc[test].copy().reset_index(drop=True)
    selected['test_offset'] = np.arange(len(selected))
    selected['attack_eligible'] = selected.sequence_clean.map(
        lambda sequence: bool(sequence) and set(sequence) <= CANONICAL_SET)
    selected['exclusion_reason'] = np.where(selected.attack_eligible, '',
                                             'sequence_contains_U_or_X; cached cleaning is not reversible')
    selected['sequence_sha256'] = selected.sequence_clean.map(
        lambda value: hashlib.sha256(value.encode()).hexdigest())
    return selected, test, protein, smiles, maccs


@torch.inference_mode()
def cached_predictions(model, protein, smiles, maccs, ids, device):
    output = []
    for start in range(0, len(ids), 256):
        batch = ids[start:start + 256]
        p = torch.tensor(protein[batch], device=device)
        s = torch.tensor(np.asarray(smiles[batch]), device=device)
        f = torch.tensor(np.asarray(maccs[batch]), device=device, dtype=torch.float32)
        output.append(model.predict_log2(p, s, f).reshape(-1).cpu())
    return torch.cat(output).numpy()


def prepare_contract():
    OUT.mkdir(exist_ok=True)
    selected, test, protein, smiles, maccs = load_inputs()
    model = ImprovedKcatEnsemble.from_manifest(MANIFEST, device='cpu')
    prediction = cached_predictions(model, protein, smiles, maccs, test, 'cpu')
    baseline = r2(selected.y_log2.to_numpy(), prediction)
    if not np.isclose(baseline, EXPECTED_R2, atol=1e-10, rtol=0):
        raise ValueError(f'Frozen baseline changed: R2={baseline}')
    selected['cached_wt_prediction_log2'] = prediction
    selected.to_csv(OUT / 'test_cohort.csv', index=False)
    del model
    files = [SOURCE, MANIFEST, SPLIT, DATA_MANIFEST, PROTEIN, SMILES, MACCS,
             WEIGHTS, WEIGHTS.with_name(WEIGHTS.stem + '-contact-regression.pt'),
             Path(__file__).resolve()]
    contract = {
        'version': 'compact_topk_discrete_attack_v1',
        'frozen_model': 'old CLS/interaction families plus three ESM+MACCS seeds; fixed weights',
        'frozen_test_rows': len(selected), 'eligible_rows': int(selected.attack_eligible.sum()),
        'excluded_rows': int((~selected.attack_eligible).sum()),
        'baseline_test_R2': baseline, 'baseline_test_RMSE_log2': float(np.sqrt(np.mean(
            (selected.y_log2.to_numpy() - prediction) ** 2))),
        'methods': list(METHODS), 'top_k_distinct_positions': TOP_K,
        'budgets': {'top1': 1, 'top5_rerank': 5},
        'pamp_radius_scale': RADIUS_SCALE,
        'pamp_parameter_source': 'Frozen value from the pre-existing PAMP implementation; not tuned here',
        'candidate_encoder': 'ESM2 input-token gradient in float16; full-sequence objective uses cached FP32 mean',
        'exact_evaluation': 'Actual mutant sequence; FP32 ESM2; cached WT mean plus mutant-minus-WT chunk delta',
        'long_sequence_policy': 'Non-overlapping <=1022 residue chunks; gradients and exact deltas weighted by chunk length',
        'ranking': 'One best amino acid per site, then five distinct sites; top5 exact reranking uses frozen ensemble only',
        'fairness': 'All methods get identical exact-query budgets; gradient methods have additional white-box cost',
        'labels_in_candidate_selection': False,
        'negative_attack_outcomes_filtered': False,
        'test_set_is_previously_observed': True,
        'seed': SEED,
        'canonical_amino_acid_order': ''.join(CANONICAL_AAS),
        'files_sha256': {str(path): sha(path) for path in files},
    }
    path = OUT / 'attack_contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise RuntimeError('Attack contract changed; use a new versioned output directory')
    if not path.exists():
        dump(path, contract)
    snapshot = OUT / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    target = snapshot / Path(__file__).name
    if not target.exists():
        shutil.copy2(Path(__file__), target)
    print('CONTRACT_FROZEN', sha(path), 'test', len(selected), 'eligible',
          int(selected.attack_eligible.sum()), 'R2', baseline, flush=True)
    return selected, test, protein, smiles, maccs, contract


def load_esm(dtype, device):
    import esm
    with torch.serialization.safe_globals([argparse.Namespace]):
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(WEIGHTS))
    model = model.eval().to(device=device, dtype=dtype)
    model.requires_grad_(False)
    return model, alphabet, alphabet.get_batch_converter()


class CaptureInputEmbedding:
    def __init__(self, module):
        self.module, self.handle, self.tensor = module, None, None

    def __enter__(self):
        def hook(_module, _inputs, output):
            self.tensor = output.detach().requires_grad_(True)
            return self.tensor
        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is not None:
            self.handle.remove()


def rank_distinct_sites(score, sequence, method):
    """Return five stable (site, best-AA) candidates from [L,20] scores."""
    score = np.asarray(score, dtype=np.float64)
    if score.shape != (len(sequence), 20) or not np.isfinite(score).any():
        raise ValueError(f'Invalid {method} score matrix')
    for position, wt in enumerate(sequence):
        score[position, CANONICAL_AAS.index(wt)] = -np.inf
    best_aa = np.argmax(score, axis=1)
    best_score = score[np.arange(len(sequence)), best_aa]
    # Descending score; ascending position is the deterministic tie break.
    order = np.lexsort((np.arange(len(sequence)), -best_score))[:min(TOP_K, len(sequence))]
    result = []
    for rank, position in enumerate(order, start=1):
        aa_index = int(best_aa[position])
        result.append({'method': method, 'candidate_rank': rank,
                       'position_0based': int(position), 'position_1based': int(position + 1),
                       'wt_aa': sequence[position], 'mut_aa': CANONICAL_AAS[aa_index],
                       'selection_score': float(best_score[position]),
                       'aa_index': aa_index})
    return result


def random_candidates(sequence, row_id):
    digest = hashlib.sha256(f'{SEED}:{row_id}:{sequence}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    positions = rng.choice(len(sequence), size=min(TOP_K, len(sequence)), replace=False)
    result = []
    for rank, position in enumerate(positions, start=1):
        alternatives = [aa for aa in CANONICAL_AAS if aa != sequence[position]]
        mutant = alternatives[int(rng.integers(len(alternatives)))]
        result.append({'method': 'random', 'candidate_rank': rank,
                       'position_0based': int(position), 'position_1based': int(position + 1),
                       'wt_aa': sequence[position], 'mut_aa': mutant,
                       'selection_score': None, 'aa_index': CANONICAL_AAS.index(mutant)})
    return result


def select_one(row, source_index, arrays, esm_model, alphabet, converter,
               ensemble, aa_embeddings, pair_dist2, median_distance, device):
    sequence = row.sequence_clean
    hcache = torch.tensor(arrays[0][source_index], device=device).reshape(1, -1)
    substrate = torch.tensor(np.asarray(arrays[1][source_index]), device=device).reshape(1, -1)
    fingerprint = torch.tensor(np.asarray(arrays[2][source_index]), device=device,
                               dtype=torch.float32).reshape(1, -1)
    all_gradients, all_wt_embeddings, all_lm = [], [], []
    for start in range(0, len(sequence), MAX_CHUNK):
        chunk = sequence[start:start + MAX_CHUNK]
        _, _, tokens = converter([('protein', chunk)])
        tokens = tokens.to(device)
        with CaptureInputEmbedding(esm_model.embed_tokens) as capture:
            output = esm_model(tokens, repr_layers=[33], return_contacts=False)
            residues = output['representations'][33][0, 1:len(chunk) + 1].float()
            fresh_mean = residues.mean(0, keepdim=True)
            weight = len(chunk) / len(sequence)
            straight_through = hcache + weight * (fresh_mean - fresh_mean.detach())
            value = ensemble.predict_log2(straight_through, substrate, fingerprint).reshape(())
            gradient = torch.autograd.grad(value, capture.tensor)[0][0, 1:len(chunk) + 1]
            wt_embedding = capture.tensor.detach()[0, 1:len(chunk) + 1]
        lm = output['logits'][0, 1:len(chunk) + 1].index_select(
            -1, torch.tensor([alphabet.get_idx(aa) for aa in CANONICAL_AAS], device=device))
        all_gradients.append(gradient.detach().float().cpu())
        all_wt_embeddings.append(wt_embedding.detach().float().cpu())
        all_lm.append(lm.detach().float().cpu())
        del output, residues, fresh_mean, straight_through, value, gradient, wt_embedding, tokens
    gradients = torch.cat(all_gradients).to(device)
    wt_embeddings = torch.cat(all_wt_embeddings).to(device)
    lm_logits = torch.cat(all_lm).numpy()
    norm = gradients.flatten().norm().clamp_min(1e-12)
    direction = gradients / norm
    linear_direction = direction @ aa_embeddings.T - (direction * wt_embeddings).sum(1, keepdim=True)
    raw_linear = gradients @ aa_embeddings.T - (gradients * wt_embeddings).sum(1, keepdim=True)
    wt_indices = torch.tensor([CANONICAL_AAS.index(aa) for aa in sequence], device=device)
    distance2 = pair_dist2[wt_indices]
    epsilon = RADIUS_SCALE * median_distance * math.sqrt(len(sequence))
    pamp = 2 * epsilon * linear_direction - distance2
    wt_lm_logits = lm_logits[np.arange(len(sequence)), wt_indices.cpu().numpy()]
    lm_margin = lm_logits - wt_lm_logits[:, None]
    records = []
    for method, score in [('pamp', pamp.cpu().numpy()),
                          ('hotflip', raw_linear.cpu().numpy()),
                          ('esm_lm', lm_margin)]:
        for item in rank_distinct_sites(score, sequence, method):
            pos, aa_index = item['position_0based'], item.pop('aa_index')
            item.update({
                'row_id': int(row.row_id), 'test_offset': int(row.test_offset),
                'sequence_sha256': row.sequence_sha256, 'sequence_length': len(sequence),
                'mutation': f'{sequence[pos]}{pos + 1}{CANONICAL_AAS[aa_index]}',
                'first_order_delta_log2': float(raw_linear[pos, aa_index].cpu()),
                'embedding_distance2': float(distance2[pos, aa_index].cpu()),
                'lm_logit_margin': float(lm_margin[pos, aa_index]),
                'pamp_radius_scale': RADIUS_SCALE,
            })
            records.append(item)
    for item in random_candidates(sequence, row.row_id):
        pos, aa_index = item['position_0based'], item.pop('aa_index')
        item.update({
            'row_id': int(row.row_id), 'test_offset': int(row.test_offset),
            'sequence_sha256': row.sequence_sha256, 'sequence_length': len(sequence),
            'mutation': f'{sequence[pos]}{pos + 1}{CANONICAL_AAS[aa_index]}',
            'first_order_delta_log2': float(raw_linear[pos, aa_index].cpu()),
            'embedding_distance2': float(distance2[pos, aa_index].cpu()),
            'lm_logit_margin': float(lm_margin[pos, aa_index]),
            'pamp_radius_scale': RADIUS_SCALE,
        })
        records.append(item)
    if len(records) != len(METHODS) * min(TOP_K, len(sequence)):
        raise AssertionError('Each method must produce five distinct-site candidates')
    for item in records:
        mutant = apply_mutation(sequence, item['position_0based'], item['mut_aa'])
        if sum(a != b for a, b in zip(sequence, mutant)) != 1:
            raise AssertionError('Candidate is not one substitution')
    return records


def run_selection(prepared, device):
    frame, test, protein, smiles, maccs, contract = prepared
    target = OUT / 'selection_rows'
    target.mkdir(exist_ok=True)
    ensemble = ImprovedKcatEnsemble.from_manifest(MANIFEST, device=device)
    esm_model, alphabet, converter = load_esm(torch.float16, device)
    aa_ids = torch.tensor([alphabet.get_idx(aa) for aa in CANONICAL_AAS], device=device)
    aa_embeddings = esm_model.embed_tokens.weight.index_select(0, aa_ids).detach().float()
    pair_dist2 = torch.cdist(aa_embeddings, aa_embeddings).square()
    upper = torch.triu_indices(20, 20, offset=1, device=device)
    median_distance = float(pair_dist2[upper[0], upper[1]].sqrt().median().cpu())
    arrays = (protein, smiles, maccs)
    eligible = frame[frame.attack_eligible]
    completed = 0
    started = time.monotonic()
    for row in eligible.itertuples():
        path = target / f'row_{int(row.row_id)}.json'
        if path.exists():
            completed += 1
            continue
        source_index = int(test[int(row.test_offset)])
        records = select_one(row, source_index, arrays, esm_model, alphabet, converter,
                             ensemble, aa_embeddings, pair_dist2, median_distance, device)
        dump(path, records)
        completed += 1
        if completed == 1 or completed % 25 == 0:
            dump(OUT / 'STATUS.json', {'stage': 'candidate_selection', 'completed': completed,
                 'total': len(eligible), 'elapsed_seconds': time.monotonic() - started})
            print('SELECTED', completed, '/', len(eligible), flush=True)
            torch.cuda.empty_cache()
    records = []
    for row in eligible.itertuples():
        records.extend(json.loads((target / f'row_{int(row.row_id)}.json').read_text()))
    candidates = pd.DataFrame(records).sort_values(['test_offset', 'method', 'candidate_rank'])
    candidates.to_csv(OUT / 'candidates.csv', index=False)
    if len(candidates) != len(eligible) * len(METHODS) * TOP_K:
        raise AssertionError('Incomplete candidate table')
    dump(OUT / 'selection_metadata.json', {
        'rows': len(eligible), 'candidates': len(candidates), 'methods': list(METHODS),
        'median_canonical_aa_embedding_distance': median_distance,
        'candidate_file_sha256': sha(OUT / 'candidates.csv'),
        'esm_parameter_dtype': str(next(esm_model.parameters()).dtype),
        'objective_value_uses_cached_FP32_WT_embedding': True,
    })
    del esm_model, ensemble
    gc.collect()
    torch.cuda.empty_cache()
    print('SELECTION_COMPLETE', len(candidates), flush=True)


@torch.inference_mode()
def encode_chunks(model, converter, chunks, device, max_tokens=3500):
    """Encode chunk means in FP32 with dynamic batches and single-item fallback."""
    result = []
    start = 0
    while start < len(chunks):
        longest, end = 0, start
        while end < len(chunks):
            proposed = max(longest, len(chunks[end])) * (end - start + 1)
            if end > start and proposed > max_tokens:
                break
            longest = max(longest, len(chunks[end]))
            end += 1
        subset = chunks[start:end]
        _, _, tokens = converter([(str(i), seq) for i, seq in enumerate(subset)])
        try:
            output = model(tokens.to(device), repr_layers=[33], return_contacts=False)
        except torch.cuda.OutOfMemoryError:
            if len(subset) == 1:
                raise
            torch.cuda.empty_cache()
            middle = start + max(1, len(subset) // 2)
            first = encode_chunks(model, converter, chunks[start:middle], device, max_tokens=max_tokens // 2)
            second = encode_chunks(model, converter, chunks[middle:end], device, max_tokens=max_tokens // 2)
            result.extend(first + second)
            start = end
            continue
        representations = output['representations'][33]
        for i, sequence in enumerate(subset):
            result.append(representations[i, 1:len(sequence) + 1].float().mean(0).cpu().numpy())
        del output, representations, tokens
        start = end
    return result


def run_exact(prepared, device):
    frame, test, protein, smiles, maccs, contract = prepared
    candidates = pd.read_csv(OUT / 'candidates.csv', keep_default_na=False)
    selection_meta = json.loads((OUT / 'selection_metadata.json').read_text())
    if sha(OUT / 'candidates.csv') != selection_meta['candidate_file_sha256']:
        raise ValueError('Candidate selection changed')
    target = OUT / 'exact_rows'
    target.mkdir(exist_ok=True)
    ensemble = ImprovedKcatEnsemble.from_manifest(MANIFEST, device=device)
    esm_model, alphabet, converter = load_esm(torch.float32, device)
    baseline_chunk_cache = {}
    mutant_chunk_cache = {}
    eligible = frame[frame.attack_eligible]
    completed = 0
    started = time.monotonic()
    for row in eligible.itertuples():
        path = target / f'row_{int(row.row_id)}.json'
        if path.exists():
            completed += 1
            continue
        sequence = row.sequence_clean
        source_index = int(test[int(row.test_offset)])
        subset = candidates[candidates.row_id.eq(row.row_id)].copy()
        if len(subset) != len(METHODS) * TOP_K:
            raise ValueError('Candidate count changed')
        unique_mutations = subset.drop_duplicates('mutation')
        keys, chunks = [], []
        for candidate in unique_mutations.itertuples():
            pos = int(candidate.position_0based)
            chunk_index = pos // MAX_CHUNK
            start = chunk_index * MAX_CHUNK
            chunk = sequence[start:start + MAX_CHUNK]
            local = pos - start
            mutant_chunk = apply_mutation(chunk, local, candidate.mut_aa)
            key = (row.sequence_sha256, chunk_index, candidate.mutation)
            keys.append((key, row.sequence_sha256, chunk_index, chunk, mutant_chunk))
            if key not in mutant_chunk_cache:
                chunks.append(mutant_chunk)
        missing_keys = [entry[0] for entry in keys if entry[0] not in mutant_chunk_cache]
        if chunks:
            encoded = encode_chunks(esm_model, converter, chunks, device)
            if len(encoded) != len(missing_keys):
                raise AssertionError('Mutant chunk encoding mismatch')
            mutant_chunk_cache.update(zip(missing_keys, encoded))
        baseline_missing = []
        baseline_keys = []
        for _, seqhash, chunk_index, chunk, _ in keys:
            key = (seqhash, chunk_index)
            if key not in baseline_chunk_cache and key not in baseline_keys:
                baseline_keys.append(key)
                baseline_missing.append(chunk)
        if baseline_missing:
            baseline_chunk_cache.update(zip(baseline_keys,
                                             encode_chunks(esm_model, converter, baseline_missing, device)))
        mutation_to_mean = {}
        cached = protein[source_index].astype(np.float64)
        for key, seqhash, chunk_index, chunk, _ in keys:
            delta = (mutant_chunk_cache[key].astype(np.float64)
                     - baseline_chunk_cache[(seqhash, chunk_index)].astype(np.float64))
            mutation_to_mean[key[2]] = (cached + len(chunk) / len(sequence) * delta).astype(np.float32)
        ordered = subset.mutation.tolist()
        means = np.stack([mutation_to_mean[mutation] for mutation in ordered])
        p = torch.tensor(means, device=device)
        s = torch.tensor(np.repeat(np.asarray(smiles[source_index])[None, :], len(subset), axis=0),
                         device=device)
        f = torch.tensor(np.repeat(np.asarray(maccs[source_index])[None, :], len(subset), axis=0),
                         device=device, dtype=torch.float32)
        pred = ensemble.predict_log2(p, s, f).reshape(-1).cpu().numpy()
        wt = float(row.cached_wt_prediction_log2)
        records = subset.to_dict('records')
        for record, value in zip(records, pred):
            mutant = apply_mutation(sequence, int(record['position_0based']), record['mut_aa'])
            record.update({'mutant_sequence': mutant, 'wt_prediction_log2': wt,
                           'mutant_prediction_log2': float(value),
                           'delta_prediction_log2': float(value - wt),
                           'predicted_fold_change': float(np.exp2(np.clip(value - wt, -100, 100))),
                           'wt_prediction_kcat_s_inverse': float(np.exp2(np.clip(wt, -100, 100))),
                           'mutant_prediction_kcat_s_inverse': float(np.exp2(np.clip(value, -100, 100))),
                           'exact_encoder': 'ESM2 FP32 mutant-minus-WT chunk update to frozen cached mean'})
        dump(path, records)
        completed += 1
        if completed == 1 or completed % 25 == 0:
            # Bound memory without affecting deterministic results.
            if len(mutant_chunk_cache) > 2500:
                mutant_chunk_cache.clear()
            dump(OUT / 'STATUS.json', {'stage': 'exact_reencoding', 'completed': completed,
                 'total': len(eligible), 'elapsed_seconds': time.monotonic() - started})
            print('EXACT', completed, '/', len(eligible), flush=True)
            torch.cuda.empty_cache()
    records = []
    for row in eligible.itertuples():
        records.extend(json.loads((target / f'row_{int(row.row_id)}.json').read_text()))
    exact = pd.DataFrame(records).sort_values(['test_offset', 'method', 'candidate_rank'])
    exact.to_csv(OUT / 'candidate_exact_predictions.csv', index=False)
    dump(OUT / 'exact_metadata.json', {
        'rows': len(eligible), 'candidate_rows': len(exact),
        'prediction_file_sha256': sha(OUT / 'candidate_exact_predictions.csv'),
        'esm_parameter_dtype': str(next(esm_model.parameters()).dtype),
        'FP32_TF32_disabled': True, 'negative_deltas_retained': True,
    })
    del esm_model, ensemble
    gc.collect()
    torch.cuda.empty_cache()
    print('EXACT_COMPLETE', len(exact), flush=True)


def cluster_bootstrap_difference(frame, a, b, seed=SEED, n_boot=4000):
    pivot = frame.pivot(index='row_id', columns='method', values='delta_prediction_log2')
    metadata = frame.drop_duplicates('row_id').set_index('row_id')
    values = (pivot[a] - pivot[b]).rename('difference').to_frame().join(metadata.sequence_sha256)
    groups = [part.difference.to_numpy() for _, part in values.groupby('sequence_sha256')]
    observed = float(values.difference.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.integers(0, len(groups), len(groups))
        sample = np.concatenate([groups[j] for j in chosen])
        boot[i] = sample.mean()
    low, high = np.quantile(boot, [.025, .975])
    return {'mean_paired_delta_difference_log2': observed,
            'cluster_bootstrap_CI95_low': float(low), 'cluster_bootstrap_CI95_high': float(high),
            'cluster_unit': 'exact sequence', 'clusters': len(groups), 'bootstrap_replicates': n_boot}


def summarize(prepared):
    frame, test, protein, smiles, maccs, contract = prepared
    exact = pd.read_csv(OUT / 'candidate_exact_predictions.csv', keep_default_na=False)
    if sha(OUT / 'candidate_exact_predictions.csv') != json.loads(
            (OUT / 'exact_metadata.json').read_text())['prediction_file_sha256']:
        raise ValueError('Exact result file changed')
    top1 = exact[exact.candidate_rank.eq(1)].copy()
    best_indices = exact.groupby(['row_id', 'method']).mutant_prediction_log2.idxmax()
    top5 = exact.loc[best_indices].copy()
    top5['selected_from_rank'] = top5.candidate_rank
    top5['candidate_rank'] = 0
    top1.to_csv(OUT / 'top1_results.csv', index=False)
    top5.to_csv(OUT / 'top5_best_results.csv', index=False)
    summaries, comparisons = [], []
    budget_frames = {'top1': top1, 'top5_rerank': top5}
    for budget, table in budget_frames.items():
        for method in METHODS:
            part = table[table.method.eq(method)]
            delta = part.delta_prediction_log2.to_numpy()
            summaries.append({
                'budget': budget, 'method': method, 'N': len(part),
                'mean_delta_log2': float(delta.mean()), 'median_delta_log2': float(np.median(delta)),
                'q05_delta_log2': float(np.quantile(delta, .05)),
                'q95_delta_log2': float(np.quantile(delta, .95)),
                'positive_rate': float(np.mean(delta > 0)),
                'negative_rate': float(np.mean(delta < 0)),
                'at_least_1p2x_rate': float(np.mean(delta >= math.log2(1.2))),
                'at_least_2x_rate': float(np.mean(delta >= 1)),
                'geometric_mean_fold_change': float(2 ** delta.mean()),
                'mean_selected_rank': float(part.get('selected_from_rank', pd.Series([1] * len(part))).mean()),
            })
        for baseline in ('hotflip', 'esm_lm', 'random'):
            stats = cluster_bootstrap_difference(table, 'pamp', baseline,
                                                 seed=SEED + 17 * (METHODS.index(baseline) + (budget == 'top5_rerank')))
            pivot = table.pivot(index='row_id', columns='method', values='delta_prediction_log2')
            difference = pivot.pamp - pivot[baseline]
            comparisons.append({'budget': budget, 'method_a': 'pamp', 'method_b': baseline,
                                'fraction_a_better': float(np.mean(difference > 0)),
                                'fraction_tied': float(np.mean(np.isclose(difference, 0, atol=1e-8))),
                                **stats})
    stats = cluster_bootstrap_difference(pd.concat([
        top5[top5.method.eq('pamp')].assign(method='pamp_top5'),
        top1[top1.method.eq('pamp')].assign(method='pamp_top1')]), 'pamp_top5', 'pamp_top1')
    comparisons.append({'budget': 'top5_vs_top1', 'method_a': 'pamp_top5',
                        'method_b': 'pamp_top1', **stats})
    summary = pd.DataFrame(summaries)
    comparison = pd.DataFrame(comparisons)
    summary.to_csv(OUT / 'attack_summary.csv', index=False)
    comparison.to_csv(OUT / 'paired_comparisons.csv', index=False)
    exclusions = frame.loc[~frame.attack_eligible,
                           ['row_id', 'test_offset', 'sequence_length', 'sequence_sha256',
                            'exclusion_reason']]
    exclusions.to_csv(OUT / 'excluded_rows.csv', index=False)
    conclusion = {}
    for budget in ['top1', 'top5_rerank']:
        rows = comparison[comparison.budget.eq(budget)]
        conclusion[budget] = {
            row.method_b: bool(row.cluster_bootstrap_CI95_low > 0)
            for row in rows.itertuples()
        }
    dump(OUT / 'results.json', {
        'status': 'complete', 'contract_sha256': sha(OUT / 'attack_contract.json'),
        'eligible_rows': int(frame.attack_eligible.sum()), 'excluded_rows': len(exclusions),
        'summary': summaries, 'paired_comparisons': comparisons,
        'pamp_stronger_with_CI_above_zero': conclusion,
        'interpretation': 'Strength means larger prediction increase on the attacked frozen model after exact sequence re-encoding; it is not measured biochemical improvement.',
    })
    dump(OUT / 'STATUS.json', {'stage': 'complete', 'candidate_rows': len(exact),
                              'top1_rows': len(top1), 'top5_rows': len(top5)})
    print('SUMMARY\n', summary.to_string(index=False), flush=True)
    print('COMPARISONS\n', comparison.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'select', 'exact', 'summarize', 'all'], default='all')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    prepared = prepare_contract()
    if args.stage == 'prepare':
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if args.stage in ('select', 'all'):
        run_selection(prepared, device)
    if args.stage in ('exact', 'all'):
        run_exact(prepared, device)
    if args.stage in ('summarize', 'all'):
        summarize(prepared)


if __name__ == '__main__':
    main()
