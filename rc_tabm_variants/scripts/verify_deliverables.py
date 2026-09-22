"""Validate saved predictions and checkpoint selection without re-running test inference."""
import shutil
from common import *
import torch


def main():
    df,manifest,splits,p,s,y=load_data()
    run_manifest=json.loads((OUT/'reports/run_manifest.json').read_text())
    for path,digest in run_manifest['files'].items(): assert sha(path)==digest
    reports=sorted((OUT/'reports').glob('tabm_*_metrics.json'))
    results=[]
    for path in reports:
        r=json.loads(path.read_text()); name=r['model']; seed=r['seed']
        ckdir=OUT/'checkpoints'/name
        ck=torch.load(ckdir/'best.pt',map_location='cpu',weights_only=False)
        selection=json.loads((ckdir/'selection.json').read_text())
        assert selection['checkpoint_sha256']==sha(ckdir/'best.pt')
        assert selection['test_used_for_selection'] is False
        history=pd.read_csv(OUT/f'logs/{name}_seed{seed}.csv')
        best=history.loc[history.val_rmse.idxmin()]
        assert int(best.epoch)==ck['epoch']==r['best_epoch']
        np.testing.assert_allclose(best.val_rmse,r['metrics']['val']['RMSE'],atol=1e-6)
        for split,ids in splits.items():
            pred=pd.read_csv(OUT/f'predictions/{name}_{split}.csv')
            np.testing.assert_array_equal(pred.row_id,ids)
            np.testing.assert_array_equal(pred.sample_id,df.iloc[ids,0])
            np.testing.assert_allclose(pred.y_true,y[ids],atol=1e-12)
            assert pred.split.eq(split).all() and np.isfinite(pred[['y_pred','pred_std_members']]).all().all()
            np.testing.assert_allclose(pred.error,pred.y_pred-pred.y_true,atol=1e-6)
            computed=metrics(y[ids],pred.y_pred)
            for key,value in computed.items(): np.testing.assert_allclose(value,r['metrics'][split][key],atol=1e-6)
        snapshot=ckdir/'source'; snapshot.mkdir(exist_ok=True)
        for file,digest in ck['provenance']['code_sha256'].items():
            source=Path(file)
            if sha(source)==digest and not (snapshot/source.name).exists(): shutil.copy2(source,snapshot/source.name)
        for filename in ('common.py','predictor.py','train.py'):
            assert (snapshot/filename).exists(), f'Missing original training source {filename}'
            assert sha(snapshot/filename)==ck['provenance']['code_sha256'][str(OUT/'scripts'/filename)]
        results.append(dict(model=name,seed=seed,status='PASS',checkpoint_selection='minimum validation RMSE',
                            prediction_counts={k:len(v) for k,v in splits.items()},test_inference_repeated=False))
    attention=OUT/'reports/attention_diagnostics.csv'
    if attention.exists():
        a=pd.read_csv(attention); assert len(a)>=20 and a.all_finite.all()
        for row in a.itertuples():
            w=np.asarray(json.loads(row.attention_weights))
            assert len(w)==row.length and (w>=0).all()
            np.testing.assert_allclose(w.sum(),1.,atol=1e-5)
    dump(OUT/'reports/deliverable_verification.json',dict(status='PASS',source_and_split_hashes_unchanged=True,runs=results))
    print('DELIVERABLE CHECKS PASSED',len(results),'runs',flush=True)


if __name__=='__main__': main()
