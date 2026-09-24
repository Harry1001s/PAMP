"""Continue PAMP Top1 trajectories to five distinct substitutions."""
import argparse
import json
import re
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
from run import load_data, seed, sha, write
from attack_adapter import Engine, AttackRow, RES, mutate

HERE=Path(__file__).resolve().parent
BASE=HERE/'results_no_esm_lm_2754'
OUT=HERE/'catapro_test_2754_residual_a3_round5_distinct'
METHOD='A3_fw_avg_pamp'

def reconstruct(sequence,edits):
    positions=[]
    for edit in edits:
        match=re.fullmatch(r'([A-Z])(\d+)([A-Z])',edit)
        assert match,edit
        old,pos,new=match.groups();pos=int(pos)-1
        assert sequence[pos]==old and pos not in positions
        sequence=mutate(sequence,pos,new);positions.append(pos)
    return sequence,positions

def inputs():
    assert json.loads((BASE/'verification.json').read_text())['status']=='PASS'
    cohort,p,s,paths=load_data('catapro',str(BASE/'cohort.csv'))
    f=pd.read_csv(BASE/'endpoint_results.csv',float_precision='round_trip')
    f=f[f.method.eq(METHOD)]
    previous={ep:f[f.endpoint.eq(ep)].set_index('row_id') for ep in ['round1_top1','round2_top1']}
    for frame in previous.values():
        assert frame.index.is_unique and set(frame.index)==set(cohort.row_id)
        assert frame.source.eq('residual').all() and frame.evaluator.eq('residual').all()
    return cohort,p,s,previous,paths

def prepare(cohort,paths):
    OUT.mkdir(parents=True,exist_ok=True)
    contract=dict(n=len(cohort),method=METHOD,source='residual',rounds=5,additional_rounds=3,budget_per_round=1,site_policy='five distinct sites',
        continuation='Start from saved round2 Top1; retain original WT anchors; recompute full A3 gradient each round on current mutant; block all prior sites in path updates and final ranking; always accept even negative changes',
        stability='Round3/4/5-minus-previous mean/median/positive fraction; strict improvement at every step; paired sequence-cluster bootstrap of record-weighted mean',
        input_hashes={str(q):sha(q) for q in paths+[BASE/'endpoint_results.csv',BASE/'verification.json',HERE/'attack_adapter.py',Path(__file__),RES/'checkpoints/best.pt']})
    if (OUT/'contract.json').exists():assert json.loads((OUT/'contract.json').read_text())==contract
    else:write(OUT/'contract.json',contract)
    cohort.to_csv(OUT/'cohort.csv',index=False)
    snap=OUT/'source';snap.mkdir(exist_ok=True)
    for name in ['run_a3_round5.py','attack_adapter.py','run.py']:
        dst=snap/name
        if dst.exists():assert dst.read_bytes()==(HERE/name).read_bytes()
        else:shutil.copy2(HERE/name,dst)

def bootstrap_mean(values,sequence_ids):
    # Resample exact protein sequences, retaining all substrate records in each cluster.
    groups=pd.DataFrame({'v':values,'seq':sequence_ids}).groupby('seq').v.agg(['sum','count'])
    sums=groups['sum'].to_numpy();counts=groups['count'].to_numpy();rng=np.random.default_rng(20260917)
    means=[]
    for _ in range(2000):
        ix=rng.integers(len(groups),size=len(groups));means.append(sums[ix].sum()/counts[ix].sum())
    return np.quantile(means,[.025,.975]).tolist()

def report(cohort):
    records=[json.loads((OUT/'rows'/f'{rid}.json').read_text()) for rid in cohort.row_id]
    f=pd.DataFrame(records)
    assert len(f)==len(cohort) and f.row_id.is_unique
    seqmap=cohort.set_index('row_id').sequence_clean
    for r in f.itertuples():
        final,positions=reconstruct(seqmap[r.row_id],r.edits)
        assert len(positions)==5 and sum(a!=b for a,b in zip(final,seqmap[r.row_id]))==5
        assert final==r.mutant_sequence
    for k in ['wt_log2']+[f'round{r}_log2' for r in range(1,6)]:
        assert np.isfinite(f[k]).all()
    for r in [1,2,3,4,5]:
        f[f'round{r}_delta_log2']=f[f'round{r}_log2']-f.wt_log2
        f[f'round{r}_increment_log2']=f[f'round{r}_log2']-f['wt_log2' if r==1 else f'round{r-1}_log2']
    f['strictly_improves_each_round']=f[[f'round{r}_increment_log2' for r in [1,2,3,4,5]]].gt(1e-6).all(axis=1)
    f['edits']=f.edits.map(json.dumps)
    f.to_csv(OUT/'trajectory_results.csv',index=False,float_format='%.17g')
    summaries=[]
    for label,g in [('all',f),('length_le_1022',f[f.length<=1022]),('length_gt_1022',f[f.length>1022])]:
        for round_number in [1,2,3,4,5]:
            for metric in ['delta','increment']:
                x=g[f'round{round_number}_{metric}_log2'].to_numpy()
                low,high=bootstrap_mean(x,g.row_id.map(seqmap))
                summaries.append(dict(cohort=label,round=round_number,metric=metric,N=len(g),unique_sequences=g.row_id.map(seqmap).nunique(),mean_delta_log2=x.mean(),median_delta_log2=np.median(x),geometric_mean_fold=2**x.mean(),positive_fraction=(x>1e-6).mean(),negative_fraction=(x < -1e-6).mean(),mean_ci95_low=low,mean_ci95_high=high))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'summary.csv',index=False,float_format='%.17g')
    verification=dict(status='PASS',rows=len(f),five_distinct_sites_verified=True,round2_predictions_reproduced=True,all_steps_positive_fraction=float(f.strictly_improves_each_round.mean()),bootstrap_replicates=2000,bootstrap_unit='exact protein sequence; not homology clusters',max_round2_prediction_abs_error=float(f.round2_reproduction_error.abs().max()))
    write(OUT/'verification.json',verification)
    with pd.ExcelWriter(OUT/'a3_five_round_results.xlsx') as writer:
        summary.to_excel(writer,sheet_name='summary',index=False)
        f.to_excel(writer,sheet_name='trajectories',index=False)
    text=f'# A3 五轮不同位点 Top1 攻击\n\n{len(cohort)} 条配对记录；沿用前两轮结果，第 3–5 轮每轮始终接受一个新位点突变。delta 表示相对 WT 的累计变化，increment 表示相对上一轮的变化。\n\n'
    text+='| 轮次 | 平均累计 Δlog2 | 累计提升比例 | 平均本轮增量 | 本轮提升比例 | 本轮均值 95% CI |\n|---|---:|---:|---:|---:|---|\n'
    for r in [1,2,3,4,5]:
        d=summary.query("cohort == 'all' and round == @r and metric == 'delta'").iloc[0]
        inc=summary.query("cohort == 'all' and round == @r and metric == 'increment'").iloc[0]
        text+=f'| {r} | {d.mean_delta_log2:.6f} | {d.positive_fraction:.2%} | {inc.mean_delta_log2:.6f} | {inc.positive_fraction:.2%} | [{inc.mean_ci95_low:.6f}, {inc.mean_ci95_high:.6f}] |\n'
    text+=f'\n五个步骤均提升的记录比例：{verification["all_steps_positive_fraction"]:.2%}。置信区间按唯一原蛋白序列进行 2000 次簇 bootstrap，保留同蛋白不同底物的相关性；未控制同源序列间相关性。结果仅表示冻结预测器的变化，不代表实测活性或更多轮次必然提升。长序列分组统计见 summary.csv。\n'
    (OUT/'RESULTS.md').write_text(text)
    print(text,flush=True)

def main(args):
    cohort,p,s,previous,paths=inputs();prepare(cohort,paths)
    if args.prepare_only:return
    if args.report_only:report(cohort);return
    engine=Engine();start=time.monotonic();done=0
    # Test longest first; all records are retained.
    order=np.argsort(-cohort.sequence_clean.str.len().to_numpy(),kind='stable')
    for i in order:
        r=cohort.iloc[i];rid=int(r.row_id);destination=OUT/'rows'/f'{rid}.json'
        if destination.exists():done+=1;continue
        seq=r.sequence_clean;old=previous['round2_top1'].loc[rid];first=previous['round1_top1'].loc[rid]
        edits=json.loads(old.edits);second,blocked=reconstruct(seq,edits)
        assert len(blocked)==2
        assert edits[0]==json.loads(first.edits)[0]
        row=AttackRow(engine,seq,p[i],s[i]);wt=row.values(row.live0)[0]['residual']
        assert abs(wt-old.wt_log2)<1e-5
        current=second; all_edits=list(edits)
        values={}; diagnostics=[]
        for round_number in [3,4,5]:
            choices,value,trace=row.proposals(current,'residual',seed(rid,f'round{round_number}'),1,[METHOD],blocked_positions=blocked)
            expected=float(old.mutant_log2) if round_number==3 else values[f'round{round_number-1}_log2']
            assert abs(value-expected)<1e-5,(rid,round_number,value,expected)
            if round_number==3:reproduction_error=value-float(old.mutant_log2)
            position,aa=choices[METHOD][0]
            assert position not in blocked and all(t['position']-1 not in blocked for t in trace)
            final=mutate(current,position,aa)
            y=row.values(engine.encode(final)[0])[0]['residual']
            all_edits.append(f'{current[position]}{position+1}{aa}')
            assert sum(a!=b for a,b in zip(seq,final))==round_number
            diagnostics.append(dict(round=round_number,choices=choices,trace=trace,blocked_positions_1based=[b+1 for b in blocked],previous_prediction=value))
            values[f'round{round_number}_log2']=y
            blocked.append(position);current=final
        write(OUT/'selections'/f'{rid}_rounds3_to5.json',diagnostics)
        write(destination,dict(row_id=rid,length=len(seq),method=METHOD,wt_log2=float(old.wt_log2),round1_log2=float(first.mutant_log2),round2_log2=float(old.mutant_log2),**values,round2_reproduction_error=reproduction_error,edits=all_edits,mutant_sequence=current))
        done+=1
        state=dict(stage='running',completed=done,total=len(cohort),elapsed_seconds=time.monotonic()-start)
        write(OUT/'status.json',state)
        if done%50==0 or done<=3:print(json.dumps(state),flush=True)
    report(cohort)
    write(OUT/'status.json',dict(stage='complete',completed=len(cohort),total=len(cohort),elapsed_seconds=time.monotonic()-start))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--base',type=Path,required=True,help='Two-round search output directory')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    BASE=args.base
    OUT=args.out
    try:main(args)
    except Exception:
        import traceback
        write(OUT/'status.json',dict(stage='failed',error=traceback.format_exc()));raise
