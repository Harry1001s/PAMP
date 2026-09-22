import argparse
import random
import time

from common import *
import torch
from predictor import Predictor


class Batches:
    def __init__(self, config, protein, substrate, y, mapping, device):
        self.config = config
        self.device = device
        self.protein = torch.tensor(protein,device=device)
        self.substrate = torch.tensor(substrate,device=device)
        self.y = torch.tensor(y,dtype=torch.float32,device=device)
        self.mapping = mapping
        self.cache = {}
        if config['kind'] == 'condpool':
            assert (OUT/'cache/esm2_residue/completion.json').exists(), 'Complete residue extraction first'
            # CPU caching avoids repeated disk I/O. Float16 ragged arrays, no global padding.
            for uid in np.unique(mapping):
                self.cache[int(uid)] = torch.from_numpy(np.load(OUT/f'cache/esm2_residue/{uid:05d}.npy'))
            self.lengths = np.array([len(self.cache[int(i)]) for i in mapping])

    def order(self, ids, training=False):
        ids = np.asarray(ids).copy()
        if training:
            np.random.shuffle(ids)
        if self.config['kind'] == 'condpool':
            # Sort small random pools, then shuffle batches; every row occurs exactly once.
            pools = [ids[i:i+1024] for i in range(0,len(ids),1024)]
            ids = np.concatenate([p[np.argsort(self.lengths[p],kind='stable')] for p in pools])
        batches = [ids[i:i+self.config['batch_size']] for i in range(0,len(ids),self.config['batch_size'])]
        if training:
            random.shuffle(batches)
        return batches

    def get(self, ids):
        ix = torch.as_tensor(ids,device=self.device)
        mask = None
        if self.config['kind'] == 'mean':
            h = self.protein[ix]
        else:
            lengths = self.lengths[ids]
            h = torch.zeros((len(ids),int(max(lengths)),1280),dtype=torch.float16)
            for j,i in enumerate(ids):
                h[j,:lengths[j]] = self.cache[int(self.mapping[i])]
            h = h.to(self.device)
            mask = torch.arange(h.shape[1],device=self.device)[None,:] < torch.tensor(lengths,device=self.device)[:,None]
        return h, self.substrate[ix], mask, self.y[ix]


@torch.no_grad()
def evaluate(model, batches, ids):
    model.eval()
    output = {}
    member_sse = 0.
    for ix in batches.order(ids):
        h,s,mask,y = batches.get(ix)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            members = model(h,s,mask).float()
        target = (y-model.target_mean)/model.target_scale
        member_sse += float((members-target[:,None]).square().mean(1).sum())
        raw = members*model.target_scale+model.target_mean
        for i,p,sd in zip(ix,raw.mean(1).cpu().numpy(),raw.std(1,unbiased=False).cpu().numpy()):
            output[int(i)] = (p,sd)
    result = np.asarray([output[int(i)] for i in ids])
    return result[:,0],result[:,1],member_sse/len(ids)


def atomic_checkpoint(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value,tmp)
    tmp.replace(path)


def train(kind, seed, k=None):
    config = json.loads((OUT/f'configs/tabm_{kind}.json').read_text())
    config['seed'] = seed
    if k is not None:
        config['k']=k
    base=f'tabm_{kind}' + (f'_k{config["k"]}' if config['k']!=16 else '')
    name = base if seed == 42 else f'{base}_seed{seed}'
    report = OUT/f'reports/{name}_metrics.json'
    if report.exists():
        return json.loads(report.read_text())
    ckdir = OUT/f'checkpoints/{name}'
    ckdir.mkdir(parents=True,exist_ok=True)
    if (ckdir/'best.pt').exists():
        raise RuntimeError(f'Incomplete existing run {name}; refuses to overwrite. Diagnose before resuming.')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    df,manifest,splits,p,s,y = load_data()
    model = Predictor(config)
    model.set_scalers(p,s,y,splits['train'])
    model = model.cuda()
    mapping = pd.read_csv(ROOT/'catpro_esm2_mean_pooling/row_mapping.csv').unique_sequence_index.to_numpy()
    batches = Batches(config,p,s,y,mapping,'cuda')
    opt = torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    provenance = json.loads((OUT/'reports/run_manifest.json').read_text())
    for path,digest in provenance['files'].items():
        assert sha(path) == digest
    provenance['gpu'] = torch.cuda.get_device_name(0)
    provenance['code_sha256'] = {str(p):sha(p) for p in (OUT/'scripts').glob('*.py')}
    dump(ckdir/'run_manifest.json',dict(config=config,provenance=provenance))
    history=[]; best=float('inf'); stale=0; start=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,config['max_epochs']+1):
        model.train(); total=0.
        for ix in batches.order(splits['train'],training=True):
            h,sub,mask,target = batches.get(ix)
            target = (target-model.target_mean)/model.target_scale
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                members = model(h,sub,mask)
                # Crucial: each member has its own MSE; average only after squaring.
                loss = (members.float()-target[:,None]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'],error_if_nonfinite=True)
            opt.step()
            total += float(loss.detach())*len(ix)
        vp,vs,vl = evaluate(model,batches,splits['val'])
        tp,ts,tl = evaluate(model,batches,splits['train'])
        vr=float(np.sqrt(np.mean((vp-y[splits['val']])**2)))
        tr=float(np.sqrt(np.mean((tp-y[splits['train']])**2)))
        assert np.isfinite(vr)
        history.append(dict(epoch=epoch,train_loss=total/len(splits['train']),train_eval_loss=tl,val_loss=vl,
                            train_rmse=tr,val_rmse=vr,generalization_gap=vr-tr,seconds=time.monotonic()-start))
        pd.DataFrame(history).to_csv(OUT/f'logs/{name}_seed{seed}.csv',index=False)
        if vr < best:
            best=vr; stale=0
            atomic_checkpoint(ckdir/'best.pt',dict(config=config,state_dict=model.state_dict(),epoch=epoch,
                val_rmse=vr,provenance=provenance,optimizer=opt.state_dict()))
        else:
            stale += 1
        print(name,seed,'epoch',epoch,'train_RMSE',round(tr,5),'val_RMSE',round(vr,5),'best',round(best,5),'sec',round(time.monotonic()-start),flush=True)
        if stale >= config['patience']:
            break
    ck = torch.load(ckdir/'best.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(ck['state_dict'])
    # Freeze and record the selected checkpoint BEFORE touching test outcomes.
    dump(ckdir/'selection.json',dict(epoch=ck['epoch'],validation_rmse=ck['val_rmse'],checkpoint_sha256=sha(ckdir/'best.pt'),test_used_for_selection=False))
    result=dict(model=name,seed=seed,best_epoch=ck['epoch'],epochs=epoch,parameters=sum(p.numel() for p in model.parameters()),
                training_seconds=time.monotonic()-start,peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/2**20,checkpoint=str(ckdir/'best.pt'),metrics={})
    for split in ('train','val','test'):
        ix=splits[split]
        pred,std,loss=evaluate(model,batches,ix)
        result['metrics'][split]=dict(**metrics(y[ix],pred),loss_standardized_member_mse=loss,mean_member_prediction_std=float(std.mean()))
        pd.DataFrame(dict(sample_id=df.iloc[ix,0].to_numpy(),row_id=ix,split=split,y_true=y[ix],y_pred=pred,error=pred-y[ix],pred_std_members=std)).to_csv(OUT/f'predictions/{name}_{split}.csv',index=False)
    result['generalization_gap']=result['metrics']['val']['RMSE']-result['metrics']['train']['RMSE']
    dump(report,result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hist=pd.DataFrame(history)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for field in ('train_loss','val_loss'): axes[0].plot(hist.epoch,hist[field],label=field)
    for field in ('train_rmse','val_rmse'): axes[1].plot(hist.epoch,hist[field],label=field)
    for ax in axes: ax.set_xlabel('Epoch'); ax.legend(); ax.axvline(ck['epoch'],color='gray',ls='--')
    axes[0].set_ylabel('Standardized per-member MSE'); axes[1].set_ylabel('RMSE, log2(kcat)')
    fig.tight_layout(); fig.savefig(OUT/f'reports/{name}_training_curve.png',dpi=160); plt.close(fig)
    if kind=='condpool':
        attention_diagnostics(model,batches,df,manifest,splits['test'],name)
    print('DONE',json.dumps(result),flush=True)
    return result


@torch.no_grad()
def attention_diagnostics(model,batches,df,manifest,ids,name):
    model.eval()
    selected=list(np.random.default_rng(42).choice(ids,20,replace=False))
    grouped={}
    for i in ids: grouped.setdefault(df.Sequence[i],[]).append(i)
    pairs=[]
    for group in grouped.values():
        if len(set(manifest.canonical_smiles[group]))>1:
            a=group[0]; b=next(i for i in group if manifest.canonical_smiles[i]!=manifest.canonical_smiles[a])
            pairs.append((int(a),int(b))); selected.extend([a,b])
            if len(pairs)==5: break
    records=[]; weights={}
    for i in dict.fromkeys(selected):
        h,s,mask,y=batches.get(np.array([i]))
        with torch.autocast('cuda',dtype=torch.bfloat16): members,alpha=model(h,s,mask,return_attention=True)
        w=alpha[0,:len(df.Sequence[i])].cpu().numpy(); weights[int(i)]=w
        top=np.argsort(-w)[:10]
        records.append(dict(sample_id=df.iloc[i,0],row_id=int(i),sequence_id=hashlib.sha256(df.Sequence[i].encode()).hexdigest(),
            substrate_id=manifest.canonical_smiles[i],length=len(w),top_10_attention_positions=json.dumps((top+1).tolist()),
            attention_weights=json.dumps(w.tolist()),prediction=float(members.float().mean()*model.target_scale+model.target_mean),label=float(y),
            max_attention=float(w.max()),normalized_entropy=float(-(w*np.log(np.maximum(w,1e-30))).sum()/np.log(len(w))),
            all_finite=bool(np.isfinite(w).all()),nearly_uniform=bool(np.ptp(w)<1e-6),collapsed=bool(w.max()>.99)))
    pd.DataFrame(records).to_csv(OUT/f'reports/{"attention_diagnostics" if name=="tabm_condpool" else name+"_attention"}.csv',index=False)
    dump(OUT/f'reports/{name}_attention_pairs.json',[dict(row_a=a,row_b=b,mean_abs_weight_difference=float(np.abs(weights[a]-weights[b]).mean())) for a,b in pairs])


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--kind',choices=['mean','condpool'],required=True); parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--k',type=int,choices=[16,32])
    args=parser.parse_args(); train(args.kind,args.seed,args.k)
