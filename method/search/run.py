"""Two frozen white-box sources, three evaluators, Top5 and reranked Top1x2.

Extra Trees is explicitly a transfer evaluator, never a white-box target.
"""
import sys as _sys, pathlib as _pl
for _c in _pl.Path(__file__).resolve().parents:
    if (_c / 'pamp_paths.py').exists():
        _sys.path.insert(0, str(_c)); break
from pamp_paths import EXTRA_TREES_MODEL, KCAT_CSV
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from attack_adapter import Engine,AttackRow,ROOT,RES,METHODS,mutate

HERE=Path(__file__).resolve().parent
ET=EXTRA_TREES_MODEL


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False));temp.replace(path)


def seed(row_id,phase):
    return int(hashlib.sha256(f'multi-pamp-v1:{row_id}:{phase}'.encode()).hexdigest()[:8],16)


def load_data(dataset,cohort_path=None,min_length=None,max_length=None):
    if dataset=='brenda':
        d=ROOT/'experiment_brenda_external'
        frame=pd.read_csv(d/'evaluation_rows.csv').rename(columns={'brenda_row':'row_id'})
        sequences=list(dict.fromkeys(frame.sequence_clean));smiles=list(dict.fromkeys(frame.canonical_smiles))
        seq_ids=frame.sequence_clean.map({s:i for i,s in enumerate(sequences)}).to_numpy()
        sub_ids=frame.canonical_smiles.map({s:i for i,s in enumerate(smiles)}).to_numpy()
        p=np.load(d/'protein_unique.npy')[seq_ids];s=np.load(d/'smiles_unique.npy')[sub_ids]
        assert len(frame)==4835 and frame.eligible_pair_novel.all()
        paths=[d/n for n in ('evaluation_rows.csv','protein_unique.npy','smiles_unique.npy')]
    else:
        d=ROOT/'experiment_mean'
        split=np.load(d/'split_indices.npz')['test_idx']
        data=pd.read_csv(str(KCAT_CSV))
        if cohort_path:
            frame=pd.read_csv(cohort_path)
            split=frame.row_id.to_numpy(dtype=int)
            np.testing.assert_array_equal(frame.sequence_clean,data.Sequence.iloc[split].to_numpy())
        else:
            frame=pd.DataFrame({'row_id':split,'sequence_clean':data.Sequence.iloc[split].to_numpy()})
        with (ROOT/'catpro_esm2_mean_pooling/protein_mean_embs.pkl').open('rb') as f:p=np.asarray(pickle.load(f))[split]
        s=np.load(d/'smiles_unikp1024.npy')[split]
        paths=[KCAT_CSV,d/'split_indices.npz',d/'smiles_unikp1024.npy',ROOT/'catpro_esm2_mean_pooling/protein_mean_embs.pkl']
        if cohort_path:paths.append(Path(cohort_path))
    if dataset!='catapro' and cohort_path:raise ValueError('Explicit cohort currently supports CataPro only')
    lengths=frame.sequence_clean.str.len().to_numpy()
    keep=np.ones(len(frame),dtype=bool)
    if min_length is not None:keep &= lengths>=min_length
    if max_length is not None:keep &= lengths<=max_length
    frame=frame.loc[keep].copy();p=p[keep];s=s[keep]
    assert frame.row_id.is_unique and np.isfinite(p).all() and np.isfinite(s).all()
    return frame,p,s,paths


def run_row(engine,tree,record,p,s,methods,out,sources=('original','residual'),site_policy='allow'):
    rid=int(record.row_id);seq=record.sequence_clean
    row=AttackRow(engine,seq,p,s)
    exact={}
    def evaluate(sequence):
        if sequence not in exact:
            H=row.live0 if sequence==seq else engine.encode(sequence)[0]
            values,mean=row.values(H)
            values={name:values[name] for name in sources}
            if tree is not None:
                values['extra_trees']=float(tree.predict(np.concatenate([mean,s])[None])[0])
            exact[sequence]=values
        return exact[sequence]
    wt=evaluate(seq)
    # Freeze all first-round rankings before exact mutant queries.
    selections={}
    for source in sources:
        choices,value,trace=row.proposals(seq,source,seed(rid,'round1'),5,methods)
        assert abs(value-wt[source])<1e-5
        selections[source]={'methods':choices,'trace':trace,'wt_prediction':value}
    write(out/'selections'/f'{rid}_round1.json',selections)
    results=[]
    def export(source,method,regime,candidate_rank,sequence,edits):
        values=evaluate(sequence)
        for evaluator,value in values.items():
            results.append(dict(row_id=rid,source=source,evaluator=evaluator,method=method,regime=regime,
                candidate_rank=candidate_rank,edits=edits,wt_log2=wt[evaluator],mutant_log2=value,
                delta_log2=value-wt[evaluator],net_substitutions=sum(a!=b for a,b in zip(seq,sequence))))
    for source in sources:
        grouped={}
        for method,choices in selections[source]['methods'].items():
            for k,(position,aa) in enumerate(choices,1):
                mutant=mutate(seq,position,aa)
                export(source,method,'round1_top5',k,mutant,[f'{seq[position]}{position+1}{aa}'])
            position,aa=choices[0];first=mutate(seq,position,aa)
            grouped.setdefault(first,[]).append(method)
        for first,current_methods in grouped.items():
            # Shared exact first mutant permits shared gradients, never best-of-5.
            blocked=[i for i,(a,b) in enumerate(zip(seq,first)) if a!=b] if site_policy=='distinct' else []
            choices,value,trace=row.proposals(first,source,seed(rid,'round2'),1,current_methods,blocked_positions=blocked)
            assert abs(value-evaluate(first)[source])<1e-5
            key=hashlib.sha256(first.encode()).hexdigest()[:16]
            write(out/'selections'/f'{rid}_{source}_{key}_round2.json',dict(methods=choices,trace=trace,current_prediction=value,blocked_positions_1based=[i+1 for i in blocked]))
            for method,[(position,aa)] in choices.items():
                p1,a1=selections[source]['methods'][method][0]
                final=mutate(first,position,aa)
                if site_policy=='distinct':
                    assert position!=p1 and final[p1]==a1
                    assert sum(a!=b for a,b in zip(seq,final))==2
                    assert all(t['position']!=p1+1 for t in trace)
                export(source,method,'round2_top1',1,final,[f'{seq[p1]}{p1+1}{a1}',f'{first[position]}{position+1}{aa}'])
    # Retain negative increments; distinct policy guarantees two net substitutions.
    write(out/'rows'/f'{rid}.json',dict(row_id=rid,length=len(seq),records=results,unique_exact_sequences=len(exact)))


def report(out,cohort):
    records=[]
    for rid in cohort.row_id:records.extend(json.loads((out/'rows'/f'{rid}.json').read_text())['records'])
    f=pd.DataFrame(records)
    contract=json.loads((out/'contract.json').read_text())
    if contract.get('site_policy')=='distinct':
        assert f.loc[f.regime=='round2_top1','net_substitutions'].eq(2).all()
    f['edits']=f.edits.map(json.dumps)
    f.to_csv(out/'candidate_results.csv',index=False,float_format='%.17g')
    endpoints=[]
    first=f[f.regime=='round1_top5']
    for (rid,source,method),group in first.groupby(['row_id','source','method']):
        own=group[group.evaluator==source].sort_values('candidate_rank')
        # Candidate rank breaks ties; selection uses only the source predictor.
        best=int(own.loc[own.mutant_log2.idxmax(),'candidate_rank'])
        for name,k in [('round1_top1',1),('round1_best_of_top5',best)]:
            x=group[group.candidate_rank==k].copy();x['endpoint']=name;endpoints.append(x)
    x=f[f.regime=='round2_top1'].copy();x['endpoint']='round2_top1';endpoints.append(x)
    endpoints=pd.concat(endpoints,ignore_index=True)
    endpoints.to_csv(out/'endpoint_results.csv',index=False,float_format='%.17g')
    rows=[]
    for keys,g in endpoints.groupby(['source','evaluator','method','endpoint']):
        d=g.delta_log2.to_numpy()
        rows.append(dict(zip(['source','evaluator','method','endpoint'],keys),N=len(d),mean_delta_log2=float(d.mean()),
            geometric_mean_fold=float(2**d.mean()),median_delta_log2=float(np.median(d)),positive_fraction=float((d>1e-6).mean()),
            negative_fraction=float((d < -1e-6).mean())))
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False,float_format='%.17g')
    write(out/'verification.json',dict(status='PASS',rows=len(cohort),candidate_records=len(f),
        frozen_rank1_for_round2=True,top5_transfer_selected_by_source_only=True,negative_results_retained=True,
        site_policy=contract.get('site_policy','allow'),two_distinct_sites_verified=contract.get('site_policy')=='distinct'))


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    frame,p,s,paths=load_data(args.dataset,args.cohort,args.min_length,args.max_length)
    if args.limit:
        # Quick smoke: shortest rows; full run uses every row, original order.
        lengths=frame.sequence_clean.str.len().to_numpy()
        ix=np.argsort(-lengths if args.sample_order=='longest' else lengths,kind='stable')[:args.limit]
        frame=frame.iloc[ix].copy();p=p[ix];s=s[ix]
    methods=METHODS if args.methods=='all' else [m for m in METHODS if m!='A3_fw_avg_pamp'] if args.methods=='remaining' else ['A3_fw_avg_pamp']
    sources=['residual'] if args.sources=='residual' else ['original','residual']
    evaluators=sources+(['extra_trees'] if args.extra_trees else [])
    split_counts=None
    if args.dataset=='catapro':
        splits=np.load(ROOT/'experiment_mean/split_indices.npz')
        split_counts={k:len(set(frame.row_id)&set(splits[k+'_idx'])) for k in ('train','val','test')}
    contract=dict(dataset=args.dataset,n=len(frame),sources=sources,evaluators=evaluators,
        residual_attack_objective='Full final output: frozen global + gamma * local; differentiate both branches through all residue representations to ESM2 input embeddings',
        cohort_source=str(Path(args.cohort).resolve()) if args.cohort else 'default dataset cohort',
        requested_length_bounds=[args.min_length,args.max_length],actual_length_bounds=[int(frame.sequence_clean.str.len().min()),int(frame.sequence_clean.str.len().max())],
        source_split_counts=split_counts,
        extra_trees_mode='transfer only; no Extra Trees candidate reranking' if args.extra_trees else 'disabled by user',methods=methods,rounds=2,first_round_budget=5,second_round_budget=1,
        site_policy=args.site_policy,
        second_round='Recompute scores from each method first-ranked mutant; '+('exclude first edited site in path vertices and final ranking; exactly two distinct net substitutions' if args.site_policy=='distinct' else 'allow same-site edits and reversions')+'; always accept',
        top5='Five first-round proposals, per-site cap2. Report all and best-of-five by source only. Top1 is fixed rank1.',
        endpoint='Mean0 + mean(Hlive-mutant)-mean(Hlive-WT); H0 + Hlive-mutant-Hlive-WT; substrate fixed',
        precision='FP32; TF32 off; MHA fastpath off. H0 is fp16 residue cache-equivalent. Attack predictions can differ slightly from bf16 evaluation.',
        interpretation='Computational predictor changes, not measured activity; legacy batch-dependent substrate cache retained',
        smoke=bool(args.limit),sample_order=args.sample_order if args.limit else 'original saved test order',
        noncanonical_policy='Keep all rows and full sequences; only canonical positions can be mutated',
        input_hashes={str(q):sha(q) for q in paths+[HERE/'run.py',HERE/'attack_adapter.py',RES/'checkpoints/best.pt']+([ET] if args.extra_trees else [])})
    if (out/'contract.json').exists():assert json.loads((out/'contract.json').read_text())==contract
    else:write(out/'contract.json',contract)
    frame.to_csv(out/'cohort.csv',index=False)
    engine=Engine();tree=joblib.load(ET) if args.extra_trees else None
    if tree is not None:tree.n_jobs=4
    start=time.monotonic()
    for i,record in enumerate(frame.itertuples()):
        if (out/'rows'/f'{record.row_id}.json').exists():continue
        run_row(engine,tree,record,p[i],s[i],methods,out,sources,args.site_policy)
        state=dict(stage='running',completed=i+1,total=len(frame),elapsed_seconds=time.monotonic()-start)
        write(out/'status.json',state);print(json.dumps(state),flush=True)
    report(out,frame)
    write(out/'status.json',dict(stage='complete',completed=len(frame),elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['brenda','catapro'],required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--methods',choices=['all','a3','remaining'],default='all')
    parser.add_argument('--extra-trees',action='store_true',help='Opt in to Extra Trees transfer evaluation')
    parser.add_argument('--sample-order',choices=['shortest','longest'],default='shortest')
    parser.add_argument('--cohort',help='Explicit existing CataPro cohort CSV, with row_id and sequence_clean')
    parser.add_argument('--min-length',type=int)
    parser.add_argument('--max-length',type=int)
    parser.add_argument('--sources',choices=['both','residual'],default='residual')
    parser.add_argument('--site-policy',choices=['allow','distinct'],default='distinct')
    args=parser.parse_args()
    try:main(args)
    except Exception:
        import traceback
        write(Path(args.out)/'status.json',dict(stage='failed',error=traceback.format_exc()))
        raise
