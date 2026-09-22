import inspect
import platform
import subprocess
import importlib.metadata

from common import *
import torch
from tabm import TabM


def main():
    df, manifest, splits, protein, substrate, y = load_data()
    meta = json.loads((ROOT / 'catpro_esm2_mean_pooling/embedding_metadata.json').read_text())
    assert sha(SOURCE) == meta['source_sha256']
    reference = json.loads((BASE / 'model_manifest.json').read_text())
    assert reference['split_sha256'] == sha(SPLIT)
    contract = json.loads((BASE / 'evaluation_contract.json').read_text())
    for path in (PROTEIN, SUBSTRATE, ROOT / 'experiment_mean/data_manifest.csv'):
        assert sha(path) == contract['input_sha256'][str(path)]
    mapping = pd.read_csv(ROOT / 'catpro_esm2_mean_pooling/row_mapping.csv')
    unique = list(dict.fromkeys(df.Sequence))
    uv = np.load(ROOT / 'catpro_esm2_mean_pooling/unique_mean.partial.npy', mmap_mode='r')
    assert np.array_equal(mapping.sample_id, df.iloc[:, 0])
    assert all(unique[i] == s for i, s in zip(mapping.unique_sequence_index, df.Sequence))
    np.testing.assert_array_equal(protein, uv[mapping.unique_sequence_index])
    ids = np.random.default_rng(42).choice(len(df), 20, replace=False)
    alignment = dict(status='mean/substrate/label verified; residue pending extraction',
        samples=[dict(row_id=int(i), sample_id=df.iloc[i, 0],
                      sequence_sha256=hashlib.sha256(df.Sequence[i].encode()).hexdigest(),
                      sequence_length=len(df.Sequence[i]), unique_sequence_index=int(mapping.unique_sequence_index[i]),
                      canonical_smiles=manifest.canonical_smiles[i], label=float(y[i]),
                      substrate_provenance='exact baseline cache hash; canonical SMILES order checked') for i in ids])
    from rdkit import Chem
    canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True) for s in df.Smiles]
    assert canonical == manifest.canonical_smiles.tolist()
    fm = json.loads((ROOT / 'experiment_mean/feature_manifest.json').read_text())
    assert hashlib.sha256(json.dumps(canonical).encode()).hexdigest() == fm['canonical_smiles_sha256']
    baselines = []
    for split in ('val', 'test'):
        pred = pd.read_csv(BASE / f'{split}_predictions.csv')
        np.testing.assert_array_equal(pred.row_id, splits[split])
        np.testing.assert_allclose(pred.y_log2, y[splits[split]])
        for name in ('PAMP predictor', 'Two-head average', 'Extra Trees'):
            baselines.append(dict(model=name, split=split, **metrics(pred.y_log2, pred[name])))
    dump(OUT / 'reports/baselines.json', baselines)
    dump(OUT / 'reports/cache_alignment_check.json', alignment)
    git = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    run = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU requires sandbox escalation',
        tabm=importlib.metadata.version('tabm'), rtdl_num_embeddings=importlib.metadata.version('rtdl_num_embeddings'),
        git_commit=git.stdout.strip() if git.returncode == 0 else None,
        files={str(p): sha(p) for p in (SOURCE, SPLIT, PROTEIN, SUBSTRATE)},
        split_sizes={k: len(v) for k, v in splits.items()}, protein=meta,
        substrate=dict(path=str(SUBSTRATE), dimension=1024, encoder='UniKP trfm_12_23000.pkl',
                       positional_encoding='legacy sample-axis; batch size 64, canonical unique SMILES first occurrence order'),
        screening_label='SCREENING ONLY — legacy substrate cache', seeds=[42],
        stage2_rule='Run seeds 3407,2026 for both models if CondPool val RMSE improves Mean-TabM, or Mean-TabM val RMSE improves baseline by >=0.02',
        tabm_api=str(inspect.signature(TabM)), tabm_constructor=dict(k=16,n_blocks=3,d_block=512,dropout=.1,arch_type='tabm',d_out=1),
        baseline_manifest=str(BASE / 'model_manifest.json'))
    dump(OUT / 'reports/run_manifest.json', run)
    content = '# Environment audit\n\nSCREENING ONLY — legacy substrate cache\n\n'
    content += f'- Dataset: `{SOURCE}`; labels = log2(kcat).\n- Existing split: `{SPLIT}`; train 22126 / val 2766 / test 2766. User confirmed this split.\n'
    content += f'- Protein: `{PROTEIN}`; ESM2 t33 650M UR50D, layer 33, 1280-D, BOS/EOS excluded, non-overlapping <=1022 residue chunks.\n- Substrate: `{SUBSTRATE}`; UniKP 1024-D. No corrected full CataPro cache found. Both models use identical legacy features.\n'
    content += '- Residue cache: old full_esm2_cache_pkl directory is empty; no reusable CataPro residue cache found. Generate frozen ESM2 float16 per-sequence files in this experiment; approximately 13.23 GiB.\n'
    content += '- Preprocessing: train-only per-feature population mean/std, std floor 1e-6; train-only target mean/std. CondPool applies the same mean-feature scaler to individual residues, so masked mean matches standardized mean features.\n'
    content += '- Existing scripts: run_mean_kcat_comparison.py, train_compact_kcat.py; reference predictor is the fixed CLS/MLP/compact ensemble listed in the baseline manifest. Extra Trees: paper_revision extra_trees_quick_v1/features_2304/model_seed_42.joblib.\n'
    content += '- Package: official https://github.com/yandex-research/tabm; installed locally under vendor, versions and actual API below. TabM.make defaults: 3 blocks, width 512, dropout 0.1; K overridden to 16.\n'
    content += '- Validation RMSE selects checkpoints; test once after freezing each run. First seed 42; no architecture/learning-rate search.\n\n```json\n' + json.dumps(run, indent=2) + '\n```\n'
    (OUT / 'reports/environment_audit.md').write_text(content)
    for kind in ('mean', 'condpool'):
        dump(OUT / f'configs/tabm_{kind}.json', dict(kind=kind,k=16,n_blocks=3,d_block=512,dropout=.1,d_model=256,attention_hidden_dim=128,
            lr=.0005,weight_decay=.001,batch_size=64,max_epochs=200,patience=20,grad_clip=1.,seed=42,amp='bfloat16',monitor='val RMSE',feature_scaler='train only'))
    print(json.dumps(run, indent=2))


if __name__ == '__main__':
    main()
