"""Run the 58 long test records and integrate five-method results without duplicate IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import run as runner

HERE = Path(__file__).resolve().parent
OUT = HERE/'catapro_test_long58_residual_distinct'
MERGED = HERE/'results_no_esm_lm_2754'
OLD = [HERE/'catapro_test_2697_residual_a3_distinct', HERE/'catapro_test_2697_residual_remaining_distinct']
METHODS = ['A3_fw_avg_pamp','A0_pamp','A2_fw_avg','A1_hotflip','B2_random']

def prepare():
    OUT.mkdir(exist_ok=True)
    full = pd.read_csv(HERE/'catapro_test_a3/cohort.csv')
    assert len(full)==2766 and full.row_id.is_unique
    cohort = full[full.sequence_clean.str.len()>1022].copy()
    assert len(cohort)==58 and cohort.sequence_clean.nunique()==47
    cohort = cohort.iloc[np.argsort(-cohort.sequence_clean.str.len().to_numpy(),kind='stable')]
    path = OUT/'input_cohort.csv'
    if path.exists(): pd.testing.assert_frame_equal(pd.read_csv(path),cohort.reset_index(drop=True))
    else: cohort.to_csv(path,index=False)
    snap=OUT/'source';snap.mkdir(exist_ok=True)
    for name in ['run.py','attack_adapter.py','run_long58.py']:
        dst=snap/name
        if dst.exists(): assert dst.read_bytes()==(HERE/name).read_bytes()
        else: shutil.copy2(HERE/name,dst)
    return path

def integrate():
    runs=OLD+[OUT]
    for run in runs:
        assert json.loads((run/'status.json').read_text())['stage']=='complete'
        assert json.loads((run/'verification.json').read_text())['status']=='PASS'
    contracts=[json.loads((r/'contract.json').read_text()) for r in runs]
    for key in ['sources','evaluators','site_policy','rounds','first_round_budget','second_round_budget','endpoint','precision']:
        assert all(c[key]==contracts[0][key] for c in contracts),key
    for path,digest in contracts[0]['input_hashes'].items():
        if path.endswith('run.py'): continue
        assert all(c['input_hashes'].get(path)==digest for c in contracts),path
    old=pd.read_csv(OLD[0]/'cohort.csv'); long=pd.read_csv(OUT/'cohort.csv')
    pd.testing.assert_frame_equal(old,pd.read_csv(OLD[1]/'cohort.csv'))
    overlap=set(old.row_id)&set(long.row_id)
    assert overlap=={10055}
    cohort=pd.concat([old[~old.row_id.isin(long.row_id)],long],ignore_index=True).sort_values('row_id')
    assert len(cohort)==2754 and cohort.row_id.is_unique
    full=pd.read_csv(HERE/'catapro_test_a3/cohort.csv')
    assert set(cohort.row_id)<=set(full.row_id)
    MERGED.mkdir(exist_ok=True)
    cohort.to_csv(MERGED/'cohort.csv',index=False)
    for filename,keys,count in [('candidate_results.csv',['row_id','source','evaluator','method','regime','candidate_rank'],6),('endpoint_results.csv',['row_id','source','evaluator','method','endpoint'],3)]:
        frames=[]
        for r in runs:
            f=pd.read_csv(r/filename,float_precision='round_trip')
            f=f[f.method.isin(METHODS)]
            if r!=OUT:f=f[~f.row_id.isin(long.row_id)]
            frames.append(f)
        f=pd.concat(frames,ignore_index=True)
        assert not f.duplicated(keys).any()
        assert set(f.row_id)==set(cohort.row_id) and set(f.method)==set(METHODS)
        assert len(f)==2754*5*count
        assert f.groupby(['row_id','method']).size().eq(count).all()
        assert np.isfinite(f[['wt_log2','mutant_log2','delta_log2']]).all().all()
        np.testing.assert_allclose(f.mutant_log2-f.wt_log2,f.delta_log2,rtol=1e-12,atol=1e-12)
        assert f.loc[f.regime=='round2_top1','net_substitutions'].eq(2).all()
        f.to_csv(MERGED/filename,index=False,float_format='%.17g')
    summary=[]
    for keys,g in f.groupby(['source','evaluator','method','endpoint']):
        d=g.delta_log2.to_numpy()
        summary.append(dict(zip(['source','evaluator','method','endpoint'],keys),N=len(d),mean_delta_log2=float(d.mean()),geometric_mean_fold=float(2**d.mean()),median_delta_log2=float(np.median(d)),positive_fraction=float((d>1e-6).mean()),negative_fraction=float((d < -1e-6).mean())))
    pd.DataFrame(summary).to_csv(MERGED/'summary.csv',index=False,float_format='%.17g')
    runner.write(MERGED/'provenance.json',dict(source_runs=[str(r) for r in runs],overlap_ids=sorted(overlap),overlap_policy='Use newly tested long-cohort results for all five methods',methods=METHODS,input_hashes={str(r/n):runner.sha(r/n) for r in runs for n in ['contract.json','cohort.csv','candidate_results.csv','endpoint_results.csv']}))
    (MERGED/'README.md').write_text('# 扩展对抗攻击实验\n\n原 2697 条与长序列 58 条按 row_id 合并，共 2754 条；重叠的 10055 使用新长序列实验结果。包含五种方法，不含 ESM-LM。58 条长序列的独立统计见 ../catapro_test_long58_residual_distinct/summary.csv。原 2766 条中另有 12 条不在此次合并队列中。结果表示预测变化，不是实测活性。\n')
    runner.write(MERGED/'verification.json',dict(status='PASS',records=2754,long_records=58,overlap_ids=sorted(overlap),methods=METHODS,distinct_sites_verified=True))
    print('Integrated 2754 records:',MERGED,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--integrate-only',action='store_true');args=parser.parse_args()
    if args.integrate_only:integrate()
    else:
        cohort=prepare()
        if not args.prepare_only:
            try:
                runner.METHODS=METHODS
                runner.main(argparse.Namespace(dataset='catapro',out=str(OUT),cohort=str(cohort),min_length=1023,max_length=None,limit=0,methods='all',sources='residual',site_policy='distinct',extra_trees=False,sample_order='longest'))
                integrate()
            except Exception:
                import traceback
                runner.write(OUT/'pipeline_error.json',dict(error=traceback.format_exc()))
                raise
