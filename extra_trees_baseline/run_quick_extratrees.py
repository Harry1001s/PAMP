#!/usr/bin/env python
"""Quick, fixed-parameter comparison. ExtraTreesRegressor on the immutable current random split.

2304: identical ESM2/UniKP information to the two individual heads.
2471: additionally MACCS, matching the feature union of the selected ensemble.
No test-based selection. Existing caches are used as-is, including the known
UniKP batch-slot defect; corrected-cache experiments require a new protocol.
"""
import argparse, hashlib, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

REV=Path('/root/paper_revision/20260910T034935Z')
PROJ=Path('/root/rivermind-data')

def sha(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):d.update(b)
    return d.hexdigest()

def save(p,x):
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def scores(y,p):
    return {'N':len(y),'R2':float(r2_score(y,p)),
        'RMSE':float(np.sqrt(mean_squared_error(y,p))), 'MAE':float(mean_absolute_error(y,p)),
        'PCC':float(pearsonr(y,p)[0]),'Spearman':float(spearmanr(y,p)[0])}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=int,choices=[2304,2471],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--confirm-training',action='store_true',help='Explicit execution guard; training has NOT been performed during revision.')
    p.add_argument('--n-jobs',type=int,default=8)
    args=p.parse_args()
    if not args.confirm_training:p.error('Training is disabled without --confirm-training.')
    dest=args.out.resolve()
    if REV not in dest.parents or dest.exists():p.error('Choose a NEW output directory inside this revision root.')
    contract=json.loads((PROJ/'experiment_compact_topk_attack_v1/attack_contract.json').read_text())
    paths={'source':Path('/root/kcat-data_0.4simi-10fold.csv'),
        'split':PROJ/'experiment_mean/split_indices.npz',
        'protein':PROJ/'catpro_esm2_mean_pooling/protein_mean_embs.pkl',
        'smiles':PROJ/'experiment_mean/smiles_unikp1024.npy',
        'maccs':PROJ/'experiment_compact_improvements/maccs.npy',
        'manifest':PROJ/'experiment_mean/data_manifest.csv'}
    hashes={k:sha(v) for k,v in paths.items()}
    for k,v in paths.items():
        if hashes[k]!=contract['files_sha256'][str(v)]:raise ValueError('Frozen input changed: '+str(v))
    frame=pd.read_csv(paths['source']);manifest=pd.read_csv(paths['manifest'])
    assert np.array_equal(frame.Sequence,manifest.Sequence)
    assert np.array_equal(frame.Smiles,manifest.Smiles)
    with paths['protein'].open('rb') as f:h=np.asarray(pickle.load(f),dtype=np.float32)
    s=np.load(paths['smiles']);parts=[h,s]
    if args.features==2471:parts.append(np.load(paths['maccs']).astype(np.float32))
    x=np.concatenate(parts,axis=1);assert x.shape==(len(frame),args.features) and np.isfinite(x).all()
    raw=pd.to_numeric(frame['kcat(s^-1)']).to_numpy();assert np.isfinite(raw).all() and (raw>0).all()
    y=np.log2(raw)
    splits=np.load(paths['split']);tr,va,te=[splits[k+'_idx'] for k in ['train','val','test']]
    assert np.array_equal(np.sort(np.r_[tr,va,te]),np.arange(len(frame)))
    for a,b in [(tr,va),(tr,te),(va,te)]:
        assert not (set(manifest.iloc[a].pair_key)&set(manifest.iloc[b].pair_key))
    dest.mkdir(parents=True,exist_ok=False)
    seeds=[42]
    cfg={'task':'regression','features':args.features,'target':'log2(kcat / (1 s^-1))',
        'n_estimators':100,'max_features':0.3,'min_samples_leaf':1,'bootstrap':False,'seeds':seeds,'source_hashes':hashes,
        'split':'saved random pair-disjoint 80/10/10; not ten-fold',
        'selection':'fixed quick configuration before fitting; no hyperparameter search; train only; validation descriptive',
        'known_limitation':'existing UniKP cached batch-slot positional-encoding behavior retained',
        'status':'RUNNING','script_sha256':sha(__file__)}
    save(dest/'run_config.json',cfg)
    best={'max_features':0.3,'min_samples_leaf':1}
    save(dest/'selected_parameters_locked.json',best)
    fitted=[]
    for seed in seeds:
        model=ExtraTreesRegressor(n_estimators=100,max_features=best['max_features'],
            min_samples_leaf=best['min_samples_leaf'],random_state=seed,bootstrap=False,n_jobs=args.n_jobs)
        save(dest/'STATUS.json',{'stage':'training','trees':100,'seed':seed})
        print('TRAINING',args.features,seed,flush=True)
        t=time.perf_counter();model.fit(x[tr],y[tr]);seconds=time.perf_counter()-t
        joblib.dump(model,dest/f'model_seed_{seed}.joblib',compress=3);fitted.append((seed,seconds))
        del model
    # First read of test labels/predictions for model evaluation occurs after all model choices are frozen.
    results=[]
    for seed,seconds in fitted:
        model=joblib.load(dest/f'model_seed_{seed}.joblib');pred=model.predict(x[te])
        val_pred=model.predict(x[va]);save(dest/'validation_metrics.json',scores(y[va],val_pred))
        result=dict(seed=seed,train_seconds=seconds,**scores(y[te],pred));results.append(result)
        print('TEST_RESULT',args.features,json.dumps(result),flush=True)
        pd.DataFrame({'row_id':te,'pair_key':manifest.iloc[te].pair_key.to_numpy(),
            'y_true_log2':y[te],'y_pred_log2':pred}).to_csv(dest/f'test_predictions_seed_{seed}.csv',index=False)
    pd.DataFrame(results).to_csv(dest/'test_metrics.csv',index=False)
    cfg['status']='COMPLETE';save(dest/'run_config.json',cfg)
    save(dest/'STATUS.json',{'status':'complete','warning':'Quick fixed-parameter single-seed baseline; not a tuned result or ten-fold estimate.'})

if __name__=='__main__':main()
