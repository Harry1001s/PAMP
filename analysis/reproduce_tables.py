#!/usr/bin/env python3
"""Recompute manuscript tables and validate the released record-level data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'data' / 'outputs'
METHODS = {
    'A0_pamp': 'Single-gradient', 'A1_hotflip': 'HotFlip',
    'A2_fw_avg': 'PAMP (no penalty)', 'A3_fw_avg_pamp': 'PAMP',
    'B2_random': 'Random',
}


def read(root, name):
    return pd.read_csv(root / name, float_precision='round_trip')


def prediction_metrics(y, prediction):
    assert np.isfinite(y).all() and np.isfinite(prediction).all()
    return dict(n=len(y), rmse=float(np.sqrt(np.mean((prediction-y)**2))),
                pearson_r=float(np.corrcoef(y, prediction)[0, 1]))


def design_metrics(delta):
    assert np.isfinite(delta).all()
    return dict(n=len(delta), mean=float(delta.mean()), median=float(delta.median()),
                positive_fraction=float((delta > 0).mean()),
                negative_fraction=float((delta < 0).mean()))


def compare_saved(actual, expected):
    """Allow floating-point reduction differences between NumPy versions."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare_saved(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            compare_saved(left, right)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    else:
        assert actual == expected


def cluster_interval(frame, column, replicates=4000, seed=0):
    """Resample exact sequences; retain record weighting within each draw."""
    grouped = frame.groupby('cluster')[column].agg(['sum', 'count'])
    rng = np.random.default_rng(seed)
    draws = []
    # Chunk the random matrix to bound memory without changing its draw order.
    for start in range(0, replicates, 100):
        indices = rng.integers(0, len(grouped),
                              size=(min(100, replicates-start), len(grouped)))
        draws.extend(grouped['sum'].to_numpy()[indices].sum(1)
                     / grouped['count'].to_numpy()[indices].sum(1))
    return np.percentile(draws, [2.5, 97.5]).tolist()


def verify_sequences(frame, reference):
    for row in frame.itertuples():
        original = reference.loc[row.row_id]
        chars = list(original)
        sites = set()
        edits = json.loads(row.edits)
        for edit in edits:
            site = int(edit[1:-1])-1
            assert 0 <= site < len(chars) and site not in sites
            assert chars[site] == edit[0] and edit[0] != edit[-1]
            assert edit[-1] in 'ACDEFGHIKLMNPQRSTVWY'
            sites.add(site)
            chars[site] = edit[-1]
        assert ''.join(chars) == row.mutant_sequence
        assert len(sites) == row.net_substitutions
        if hasattr(row, 'reference_sequence'):
            assert row.reference_sequence == original
    np.testing.assert_allclose(frame.mutant_log2-frame.wt_log2,
                               frame.delta_log2, atol=1e-12, rtol=0)


def reproduce(root, verify=False, bootstrap=False):
    metadata = json.loads((root / 'paper_results.json').read_text())
    splits = read(root, 'split_manifest.csv.gz')
    predictions = read(root, 'predictor_test_predictions.csv')
    external = read(root, 'brenda_sequence_unseen.csv.gz')
    endpoints = read(root, 'mutation_endpoints.csv.gz')
    rounds = read(root, 'pamp_five_rounds.csv.gz')
    transfer = read(root, 'extra_trees_transfer.csv.gz')
    reference = splits.set_index('row_id').reference_sequence
    cohort = splits[splits.design_cohort]
    lengths = cohort.reference_sequence.str.len()
    results = {'cohort': dict(
        split_counts={k: int(v) for k, v in splits.split.value_counts().items()},
        unique_sequences=int(reference.nunique()),
        unique_substrates=int(splits.canonical_smiles.nunique()),
        design_records=len(cohort), design_unique_sequences=int(cohort.reference_sequence.nunique()),
        length_min=int(lengths.min()), length_max=int(lengths.max()),
        length_q25_median_q75=lengths.quantile([.25, .5, .75]).tolist())}
    columns = {'Extra Trees': 'y_extra_trees', 'CLS head': 'y_cls',
               'Interaction MLP': 'y_interaction', 'Two-head average': 'y_two_head',
               'Original Predictor (ESM2)': 'y_global',
               'Residual Predictor (RC-TabM)': 'y_pred'}
    results['table_I_internal'] = {
        name: prediction_metrics(predictions.y_true, predictions[col])
        for name, col in columns.items()}
    results['table_I_external'] = {
        name: prediction_metrics(external.y_true, external[col])
        for name, col in {'Extra Trees': 'y_extra_trees', 'Residual Predictor': 'y_pred'}.items()}
    results['table_II'] = [dict(method=m, endpoint=e, **design_metrics(f.delta_log2))
                           for (m, e), f in endpoints.groupby(['method', 'endpoint'])]
    results['five_rounds'] = [dict(round=int(r), cumulative=design_metrics(f.delta_log2),
                                  increment=design_metrics(f.increment_log2))
                              for r, f in rounds.groupby('round')]
    results['transfer'] = [dict(method=m, endpoint=e, residual_mean=float(f.res_delta.mean()),
                               extra_trees=design_metrics(f.et_delta),
                               retained_ratio=float(f.et_delta.mean()/f.res_delta.mean()))
                           for (m, e), f in transfer.groupby(['method', 'endpoint'])]
    paired = []
    for endpoint in ['round1_top1', 'round1_best_of_top5']:
        group = endpoints[endpoints.endpoint == endpoint]
        a = group[group.method == 'A3_fw_avg_pamp'].set_index('row_id').sort_index()
        for method in ['A0_pamp', 'A1_hotflip', 'A2_fw_avg']:
            b = group[group.method == method].set_index('row_id').loc[a.index]
            difference = a.delta_log2-b.delta_log2
            same = a.mutant_sequence.eq(b.mutant_sequence)
            wins, losses = int((difference > 0).sum()), int((difference < 0).sum())
            assert (difference[same] == 0).all()
            assert wins+losses+int(same.sum()) == len(a)
            paired.append(dict(endpoint=endpoint, baseline=method,
                               mean_difference=float(difference.mean()),
                               positive_fraction_difference=float((a.delta_log2 > 0).mean()-(b.delta_log2 > 0).mean()),
                               wins=wins, ties=int(same.sum()), losses=losses))
    results['search_paired'] = paired
    top1 = endpoints[(endpoints.method == 'A3_fw_avg_pamp') & (endpoints.endpoint == 'round1_top1')]
    replacements = top1.edits.map(lambda s: json.loads(s)[0]).map(lambda e: e[0]+e[-1])
    results['substitutions_top1'] = {k: int(v) for k, v in replacements.value_counts().items()}

    if verify:
        files = {p.name for p in root.iterdir() if p.is_file()}
        assert len(files) <= 20 and files == set(metadata['files']) | {'paper_results.json'}
        for name, info in metadata['files'].items():
            path = root / name
            assert path.stat().st_size == info['bytes'], name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256'], name
        assert splits.row_id.is_unique and splits.pair_key.is_unique
        assert results['cohort']['split_counts'] == dict(train=22126, val=2766, test=2766)
        assert splits.design_cohort.eq((splits.split == 'test') & (splits.reference_sequence.str.len() > 80)).all()
        assert set(predictions.row_id) == set(splits.loc[splits.split == 'test', 'row_id'])
        aligned = splits.set_index('row_id').loc[predictions.row_id]
        assert np.array_equal(predictions.pair_key, aligned.pair_key)
        np.testing.assert_allclose(predictions.y_true, aligned.y_log2, atol=1e-12, rtol=0)
        assert not set(external.reference_sequence) & set(reference)
        assert len(external) == 3794 and external.eligible_sequence_novel.all()
        assert len(endpoints) == 41310 and len(rounds) == 13770 and len(transfer) == 27540
        assert set(endpoints.method) == set(METHODS)
        assert not endpoints.duplicated(['row_id', 'method', 'endpoint']).any()
        assert not rounds.duplicated(['row_id', 'round']).any()
        for _, f in endpoints.groupby(['method', 'endpoint']):
            assert set(f.row_id) == set(cohort.row_id)
        for _, f in rounds.groupby('round'):
            assert set(f.row_id) == set(cohort.row_id)
        candidates = read(root, 'mutation_candidates.csv.gz')
        assert len(candidates) == 82620
        assert not candidates.duplicated(['row_id', 'method', 'regime', 'candidate_rank']).any()
        first = candidates[candidates.regime == 'round1_top5']
        assert first.groupby(['row_id', 'method']).candidate_rank.apply(lambda x: sorted(x) == [1, 2, 3, 4, 5]).all()
        assert first.net_substitutions.eq(1).all()
        for frame in [candidates, endpoints, rounds]:
            verify_sequences(frame, reference)
        joined = endpoints.merge(candidates, on=['row_id', 'method', 'regime', 'candidate_rank'],
                                 suffixes=('_endpoint', '_candidate'), validate='many_to_one')
        assert len(joined) == len(endpoints)
        assert joined.mutant_sequence_endpoint.eq(joined.mutant_sequence_candidate).all()
        np.testing.assert_allclose(joined.delta_log2_endpoint, joined.delta_log2_candidate, atol=1e-12, rtol=0)
        best = first.groupby(['row_id', 'method']).delta_log2.max().sort_index()
        recorded = endpoints[endpoints.endpoint == 'round1_best_of_top5'].set_index(['row_id', 'method']).delta_log2.sort_index()
        np.testing.assert_allclose(best, recorded, atol=1e-12, rtol=0)
        ordered = rounds.sort_values(['row_id', 'round'])
        previous = ordered.groupby('row_id').mutant_log2.shift().fillna(ordered.wt_log2)
        np.testing.assert_allclose(ordered.mutant_log2-previous, ordered.increment_log2, atol=1e-12, rtol=0)
        assert (ordered.net_substitutions == ordered['round']).all()
        for r in [1, 2]:
            a = rounds[rounds['round'] == r].set_index('row_id').sort_index()
            b = endpoints[(endpoints.method == 'A3_fw_avg_pamp') & (endpoints.endpoint == f'round{r}_top1')].set_index('row_id').loc[a.index]
            assert a.mutant_sequence.eq(b.mutant_sequence).all()
            np.testing.assert_allclose(a.delta_log2, b.delta_log2, atol=1e-12, rtol=0)
        joined = transfer.merge(endpoints, on=['row_id', 'method', 'endpoint'], validate='one_to_one', suffixes=('_transfer', '_endpoint'))
        assert len(joined) == len(transfer) and joined.edits_transfer.eq(joined.edits_endpoint).all()
        np.testing.assert_allclose(joined.res_delta, joined.delta_log2, atol=1e-12, rtol=0)
        archived = metadata['archived_statistics']['search_paired']
        for row in paired:
            endpoint = 'Top1' if row['endpoint'] == 'round1_top1' else 'best-of-Top5'
            entries = {x['stat']: x for x in archived if x['endpoint'] == endpoint and x['comparison'] == 'A3 - '+row['baseline']}
            np.testing.assert_allclose([row['mean_difference'], row['positive_fraction_difference']],
                                       [float(entries['mean_diff']['estimate']), float(entries['positive_coverage_diff']['estimate'])], atol=1e-12, rtol=0)
            assert entries['n_win/n_tie/n_loss']['estimate'] == f"{row['wins']}/{row['ties']}/{row['losses']}"
        if 'computed' in metadata:
            compare_saved(results, metadata['computed'])

    intervals = []
    if bootstrap:
        transfer['cluster'] = transfer.row_id.map(reference)
        for (method, endpoint), frame in transfer.groupby(['method', 'endpoint']):
            lo, hi = cluster_interval(frame, 'et_delta')
            archived = next(x for x in metadata['archived_statistics']['transfer_summary']
                            if x['method'] == method and x['endpoint'] == endpoint)
            assert f'[{lo:.3f},{hi:.3f}]' == archived['et_ci']
            intervals.append(dict(method=method, endpoint=endpoint, ci95=[lo, hi]))
        for archived in metadata['archived_statistics']['transfer_paired']:
            left, right = archived['contrast'].removesuffix(' (ET)').split(' - ')
            frame = transfer[transfer.endpoint == archived['endpoint']]
            a = frame[frame.method == left].set_index('row_id').sort_index().copy()
            b = frame[frame.method == right].set_index('row_id').loc[a.index]
            a['difference'] = a.et_delta-b.et_delta
            lo, hi = cluster_interval(a, 'difference')
            np.testing.assert_allclose([a.difference.mean(), lo, hi],
                                       [archived['diff'], archived['ci_lo'], archived['ci_hi']], atol=1e-12, rtol=0)
    return results, intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--check', action='store_true', help='Validate hashes, cohorts, edits, rankings and archived estimates')
    parser.add_argument('--bootstrap', action='store_true', help='Reproduce Extra Trees sequence-cluster intervals (4,000 draws, seed 0)')
    parser.add_argument('--json', action='store_true', help='Print computed tables as JSON')
    args = parser.parse_args()
    results, intervals = reproduce(args.data, args.check, args.bootstrap)
    if args.json:
        print(json.dumps(dict(computed=results, transfer_intervals=intervals), indent=2, allow_nan=False))
    else:
        print(f"Computed Tables I–III and five-round summaries from {results['cohort']['design_records']:,} design records.")
        if args.check:
            print('PASS: file hashes, partitions, candidate sequences, endpoint selection and paired estimates.')
        if args.bootstrap:
            print('PASS: archived Extra Trees cluster-bootstrap intervals and paired contrasts.')


if __name__ == '__main__':
    main()
