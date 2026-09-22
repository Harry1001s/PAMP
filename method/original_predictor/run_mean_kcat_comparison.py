"""Paired mean-embedding comparison; test is evaluated only after all fits finish."""
import sys as _sys, pathlib as _pl
for _c in _pl.Path(__file__).resolve().parents:
    if (_c / 'pamp_paths.py').exists():
        _sys.path.insert(0, str(_c)); break
from pamp_paths import KCAT_CSV
import argparse, hashlib, json, math, pickle, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from kcat_models import build_predictor

ROOT = Path(__file__).resolve().parent
SOURCE = KCAT_CSV

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False))

def metrics(y, p):
    return dict(N=len(y), R2=float(r2_score(y,p)), RMSE=float(np.sqrt(mean_squared_error(y,p))),
                MAE=float(mean_absolute_error(y,p)), PCC=float(pearsonr(y,p)[0]),
                Spearman=float(spearmanr(y,p)[0]))

def prepare(out, device):
    df = pd.read_csv(SOURCE)
    meta = json.loads((ROOT/'catpro_esm2_mean_pooling/embedding_metadata.json').read_text())
    assert sha(SOURCE) == meta['source_sha256']
    mapping = pd.read_csv(ROOT/'catpro_esm2_mean_pooling/row_mapping.csv')
    assert np.array_equal(mapping.row_index, np.arange(len(df)))
    assert np.array_equal(mapping.sample_id, df.iloc[:,0])
    from generate_full_esm2_embeddings_pkl import clean_sequence
    sequences = df.Sequence.str.replace(r'\s+', '', regex=True).str.upper()
    changed = [i for i,s in enumerate(sequences) if clean_sequence(s) != s]
    if changed:
        raise ValueError(f'Cached sequence cleaning modifies residues in rows {changed[:10]}')
    with (ROOT/'catpro_esm2_mean_pooling/protein_mean_embs.pkl').open('rb') as f:
        protein = pickle.load(f)
    assert protein.shape == (len(df),1280) and np.isfinite(protein).all()
    unique_seq = list(dict.fromkeys(sequences))
    assert all(unique_seq[int(j)] == s for j,s in zip(mapping.unique_sequence_index,sequences))
    unique_vectors = np.load(ROOT/'catpro_esm2_mean_pooling/unique_mean.partial.npy', mmap_mode='r')
    assert np.array_equal(protein, unique_vectors[mapping.unique_sequence_index.to_numpy()])
    canonical = []
    for sm in df.Smiles:
        mol = Chem.MolFromSmiles(sm) if isinstance(sm,str) else None
        canonical.append(Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None)
    raw = pd.to_numeric(df['kcat(s^-1)'], errors='coerce').to_numpy()
    valid = np.isfinite(raw) & (raw>0) & sequences.notna().to_numpy() & (sequences.str.len().to_numpy()>0) & np.array([s is not None for s in canonical])
    df['row_id'] = np.arange(len(df)); df['sequence_clean'] = sequences
    df['canonical_smiles'] = canonical
    df['pair_key'] = [hashlib.sha256((s+'\n'+str(m)).encode()).hexdigest() for s,m in zip(sequences,canonical)]
    df['y_log2'] = np.log2(np.where(raw>0,raw,np.nan))
    # Deduplicate identical pair+measurement, retaining first original row and provenance.
    duplicates = df.duplicated(['pair_key','kcat(s^-1)']).to_numpy() & valid
    df['exclusion_reason'] = np.where(~valid,'invalid_sequence_smiles_or_label',np.where(duplicates,'duplicate_pair_and_label',''))
    df.to_csv(out/'data_manifest.csv',index=False)
    ids = np.flatnonzero(valid & ~duplicates)
    groups = df.iloc[ids].groupby('pair_key',sort=True).indices
    keys = np.array(list(groups)); np.random.default_rng(42).shuffle(keys)
    a,b=int(.8*len(keys)),int(.9*len(keys))
    splits = {name: np.sort(np.concatenate([ids[groups[k]] for k in subset])) for name,subset in zip(['train','val','test'],[keys[:a],keys[a:b],keys[b:]])}
    split_path=out/'split_indices.npz'
    if split_path.exists():
        old=np.load(split_path)
        assert all(np.array_equal(old[k+'_idx'],v) for k,v in splits.items())
    else:
        np.savez(split_path,**{k+'_idx':v for k,v in splits.items()})
    for x,y in [('train','val'),('train','test'),('val','test')]:
        assert not set(df.iloc[splits[x]].pair_key)&set(df.iloc[splits[y]].pair_key)
        assert not set(splits[x])&set(splits[y])
    manifest=df[['row_id','pair_key','fold']].copy(); manifest['split']='excluded'
    for k,v in splits.items(): manifest.loc[v,'split']=k
    manifest.to_csv(out/'split_manifest.csv',index=False)
    audit=dict(raw_rows=len(df),valid_rows=int(valid.sum()),excluded_invalid=int((~valid).sum()),
               duplicate_pair_label_rows=int(duplicates.sum()),retained_rows=len(ids),unique_pairs=len(keys),
               unique_sequences=int(df.iloc[ids].sequence_clean.nunique()),unique_substrates=int(df.iloc[ids].canonical_smiles.nunique()),
               conflicting_label_pairs=int(sum(len(v)>1 for v in groups.values())),
               split_rows={k:len(v) for k,v in splits.items()},split_seed=42,pair_overlap=0,
               exact_sequence_overlap={x+'_'+y:len(set(sequences.iloc[splits[x]])&set(sequences.iloc[splits[y]])) for x,y in [('train','val'),('train','test'),('val','test')]},
               source_sha256=sha(SOURCE),split_sha256=sha(split_path),rdkit=rdBase.rdkitVersion,
               label_source='CataPro raw kcat(s^-1); see local compare_kcat_brenda.py and official dataset',
               duplicate_rule='same canonical pair + raw label keep first; conflicting labels retained and group-bound; unweighted row MSE')
    dump(out/'data_audit.json',audit); dump(out/'split_audit.json',audit)
    smiles_path=out/'smiles_unikp1024.npy'
    feature_manifest=dict(protein=meta,source_sha256=sha(SOURCE),canonical_smiles_sha256=hashlib.sha256(json.dumps(canonical).encode()).hexdigest(),
                          substrate_model='UniKP trfm_12_23000.pkl',substrate_weights_sha256=sha(ROOT/'UniKP/trfm_12_23000.pkl'),
                          smiles_dim=1024,seq_len=220,pooling='UniKP mean/max/first(last)/first(penultimate), including padding as original encoder')
    if smiles_path.exists():
        assert json.loads((out/'feature_manifest.json').read_text()) == feature_manifest
        smiles=np.load(smiles_path)
    else:
        from generate_unikp_smiles1024_fixed import load_unikp_modules, load_vocab_compat, get_ids, encode_batch_unikp
        bv, Trfm, split_fn=load_unikp_modules(ROOT/'UniKP')
        vocab=load_vocab_compat(ROOT/'UniKP/vocab.pkl',bv)
        model=Trfm(len(vocab),256,len(vocab),4).to(device).eval()
        model.load_state_dict(torch.load(ROOT/'UniKP/trfm_12_23000.pkl',map_location=device,weights_only=True))
        unique=list(dict.fromkeys(s for s in canonical if s is not None)); lookup={s:i for i,s in enumerate(unique)}
        encoded=[]
        for start in range(0,len(unique),64):
            src=torch.tensor([get_ids(s,split_fn,vocab) for s in unique[start:start+64]],device=device).T.contiguous()
            encoded.append(encode_batch_unikp(model,src).cpu().numpy())
        vectors=np.concatenate(encoded)
        smiles=np.stack([vectors[lookup[s]] if s is not None else np.zeros(1024,np.float32) for s in canonical])
        np.save(smiles_path,smiles); dump(out/'feature_manifest.json',feature_manifest)
        long_count=sum(len(split_fn(s).split())>218 for s in unique)
        dump(out/'smiles_audit.json',dict(unique_smiles=len(unique),over_218_tokens=long_count,truncation='first109+last109',invalid_rows=int(sum(s is None for s in canonical))))
        del model
    assert smiles.shape==(len(df),1024) and np.isfinite(smiles).all()
    mu=float(df.y_log2.iloc[splits['train']].mean()); sigma=float(df.y_log2.iloc[splits['train']].std(ddof=0))
    assert sigma>0 and np.isfinite(sigma)
    dump(out/'label_scaler.json',dict(mu=mu,sigma=sigma,label='log2(kcat)',ddof=0))
    print('DATA',json.dumps(audit),flush=True)
    return df,splits,torch.tensor(protein,device=device),torch.tensor(smiles,device=device),torch.tensor((df.y_log2.to_numpy()-mu)/sigma,dtype=torch.float32,device=device),mu,sigma


def predict(model,h,s,idx):
    model.eval(); outputs=[]
    with torch.no_grad():
        for ids in idx.split(512):
            with torch.autocast('cuda',dtype=torch.bfloat16): outputs.append(model(h[ids],s[ids]).flatten().float())
    return torch.cat(outputs)


def fit(args,out,arch,seed,data):
    df,splits,h,s,y,mu,sigma=data
    run=out/arch/f'seed_{seed}'; run.mkdir(parents=True,exist_ok=True)
    if args.resume and (run/'fit_summary.json').exists():
        ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        assert ck['data_sha256']==sha(SOURCE) and ck['split_sha256']==sha(out/'split_indices.npz')
        assert ck['seed']==seed and ck['model_config']['architecture']==arch
        print('REUSE_COMPLETED',str(run),flush=True)
        return run
    if not args.resume and (run/'best.pt').exists():
        raise RuntimeError('Existing checkpoint; use --resume or a new output directory')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=build_predictor(arch,prot_dim=h.shape[1],smiles_dim=s.shape[1]).cuda()
    decay=[]; nodecay=[]
    for name,p in model.named_parameters():
        (nodecay if p.ndim<2 or 'cls_token' in name or 'type_embeddings' in name else decay).append(p)
    lr=getattr(args,'learning_rate',None) or (2e-4 if arch=='mean_cls_transformer' else 5e-4)
    opt=torch.optim.AdamW([dict(params=decay,weight_decay=1e-3),dict(params=nodecay,weight_decay=0)],lr=lr)
    idx={k:torch.tensor(v,device='cuda') for k,v in splits.items()}
    best=float('inf'); stale=0; history=[]; start=time.monotonic()
    first_epoch=1; elapsed_offset=0.0
    if args.resume and (run/'best.pt').exists():
        ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        assert ck['model_config']==model.config_dict() and ck['seed']==seed
        assert ck['data_sha256']==sha(SOURCE) and ck['split_sha256']==sha(out/'split_indices.npz')
        assert ck['feature_manifest']==json.loads((out/'feature_manifest.json').read_text())
        assert ck['mu']==mu and ck['sigma']==sigma and ck['lr_schedule']['max_epochs']==args.epochs
        model.load_state_dict(ck['state_dict']); opt.load_state_dict(ck['optimizer'])
        old_history=pd.read_csv(run/'training_history.csv')
        history=old_history[old_history.epoch<=ck['epoch']].to_dict('records')
        assert history[-1]['epoch']==ck['epoch']
        elapsed_offset=float(history[-1]['elapsed_seconds'])
        best=ck['val_rmse_log2']; first_epoch=ck['epoch']+1
        torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
        np.random.set_state(ck['numpy_rng']); random.setstate(ck['python_rng'])
        dump(run/'resume_info.json',dict(checkpoint_epoch=ck['epoch'],first_resumed_epoch=first_epoch,
            restored=['model','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng','lr_schedule_position'],
            previous_elapsed_seconds=elapsed_offset))
        print('RESUME',arch,'epoch',first_epoch,flush=True)
    start=time.monotonic()-elapsed_offset
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(first_epoch,args.epochs+1):
        rate=lr*epoch/5 if epoch<=5 else 1e-6+(lr-1e-6)*.5*(1+math.cos(math.pi*(epoch-5)/max(1,args.epochs-5)))
        for group in opt.param_groups: group['lr']=rate
        model.train(); order=idx['train'][torch.randperm(len(idx['train']),device='cuda')]; total=torch.zeros((),device='cuda')
        for batch in order.split(64):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(h[batch],s[batch]).flatten(); loss=(pred.float()-y[batch]).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step(); total+=loss.detach()*len(batch)
        vp=predict(model,h,s,idx['val']); rmse=float((vp-y[idx['val']]).square().mean().sqrt())*sigma
        assert math.isfinite(rmse)
        history.append(dict(epoch=epoch,train_mse_z=float(total)/len(order),val_rmse_log2=rmse,lr=rate,elapsed_seconds=time.monotonic()-start))
        pd.DataFrame(history).to_csv(run/'training_history.csv',index=False)
        if rmse<best:
            best=rmse; stale=0
            torch.save(dict(model_config=model.config_dict(),state_dict=model.state_dict(),mu=mu,sigma=sigma,seed=seed,epoch=epoch,val_rmse_log2=best,
                optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                lr_schedule=dict(epoch=epoch,max_epochs=args.epochs,warmup=5,min_lr=1e-6),
                feature_manifest=json.loads((out/'feature_manifest.json').read_text()),data_sha256=sha(SOURCE),split_sha256=sha(out/'split_indices.npz'),torch_version=torch.__version__),run/'best.pt')
        else: stale+=1
        if epoch==1 or epoch%5==0: print(arch,seed,'epoch',epoch,'val_RMSE',round(rmse,4),'best',round(best,4),'seconds',round(time.monotonic()-start),flush=True)
        if stale>=args.patience: break
    info=dict(architecture=arch,seed=seed,epochs=epoch,best_val_rmse_log2=best,trainable_parameters=sum(p.numel() for p in model.parameters()),
              train_seconds=time.monotonic()-start,cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,feature_resident_mib=(h.numel()*h.element_size()+s.numel()*s.element_size()+y.numel()*y.element_size())/2**20)
    dump(run/'fit_summary.json',info)
    print('FIT_DONE',json.dumps(info),flush=True)
    return run


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='experiment_mean'); p.add_argument('--epochs',type=int,default=200); p.add_argument('--patience',type=int,default=25); p.add_argument('--seeds',type=int,nargs='+',default=[42]); p.add_argument('--resume',action='store_true'); args=p.parse_args()
    assert torch.cuda.is_available(), 'CUDA required'
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    config=dict(epochs=args.epochs,patience=args.patience,seeds=args.seeds,batch_size=64,architectures=['mean_cls_transformer','mean_interaction_mlp'],split_seed=42,amp='bf16',torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
    if (out/'run_config.json').exists(): assert json.loads((out/'run_config.json').read_text())==config
    dump(out/'run_config.json',config)
    data=prepare(out,'cuda'); runs=[]
    for seed in args.seeds:
        for arch in config['architectures']: runs.append(fit(args,out,arch,seed,data))
    df,splits,h,s,y,mu,sigma=data; results=[]
    for run in runs:
        ck=torch.load(run/'best.pt',map_location='cuda',weights_only=False)
        model=build_predictor(**ck['model_config']).cuda(); model.load_state_dict(ck['state_dict']); model.eval()
        ids=torch.tensor(splits['test'],device='cuda'); predict(model,h,s,ids[:64]); torch.cuda.synchronize(); start=time.monotonic()
        pred=predict(model,h,s,ids).cpu().numpy()*sigma+mu; elapsed=time.monotonic()-start
        truth=df.y_log2.iloc[splits['test']].to_numpy(); info=json.loads((run/'fit_summary.json').read_text())
        info.update(metrics(truth,pred)); info['inference_samples_per_second']=len(ids)/elapsed; info['best_epoch']=ck['epoch']; results.append(info)
        frame=df.iloc[splits['test']][['row_id','pair_key']].copy(); frame['y_true_log2']=truth; frame['y_pred_log2']=pred; frame['architecture']=info['architecture']; frame['seed']=info['seed']; frame.to_csv(run/'test_predictions.csv',index=False)
        dump(run/'test_metrics.json',info)
    pd.DataFrame(results).to_csv(out/'comparison_metrics.csv',index=False)
    dump(out/'STATUS.json',dict(status='complete',test_comparison_complete=True,seeds=args.seeds))
    print('COMPARISON',json.dumps(results),flush=True)

if __name__=='__main__': main()
