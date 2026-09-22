"""Reusable training and validation-only selection for compact kcat heads."""
import json,math,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from compact_kcat_models import build_compact_predictor,load_compact_checkpoint
from run_mean_kcat_comparison import dump,sha,metrics


def predict(model,h,s,fp,idx):
    model.eval(); result=[]
    with torch.no_grad():
        for batch in idx.split(512):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result.append(model.predict_log2(h[batch],s[batch],fp[batch] if fp is not None else None).flatten().float())
    return torch.cat(result)


def fit_run(out,h,s,fp,y,split,model_config,train_config,provenance):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    contract=dict(model=model_config,training=train_config,provenance=provenance)
    if (out/'contract.json').exists(): assert json.loads((out/'contract.json').read_text())==contract
    else: dump(out/'contract.json',contract)
    if (out/'summary.json').exists():
        print('REUSE',str(out),flush=True); return json.loads((out/'summary.json').read_text())
    seed=train_config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=build_compact_predictor(model_config).cuda()
    train=split['train']; val=split['val']
    hp=h[train].double(); sp=s[train].double()
    pmean=hp.mean(0).float(); pscale=hp.std(0,unbiased=False).float(); pscale=torch.where(pscale<1e-6,1.,pscale)
    smean=sp.mean(0).float(); sscale=sp.std(0,unbiased=False).float(); sscale=torch.where(sscale<1e-6,1.,sscale)
    del hp,sp
    mu=y[train].double().mean().float(); sigma=y[train].double().std(unbiased=False).float()
    model.set_scalers(pmean,pscale,smean,sscale,mu,sigma)
    z=(y-mu)/sigma
    decay=[p for p in model.parameters() if p.ndim>=2]; nodecay=[p for p in model.parameters() if p.ndim<2]
    optimizer=torch.optim.AdamW([dict(params=decay,weight_decay=train_config['weight_decay']),dict(params=nodecay,weight_decay=0.)],lr=train_config['lr'])
    best=float('inf'); stale=0; history=[]; first=1; offset=0.
    if (out/'latest.pt').exists():
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        assert ck['contract']==contract
        model.load_state_dict(ck['state_dict']); optimizer.load_state_dict(ck['optimizer'])
        best=ck['best']; stale=ck['stale']; history=ck['history']; first=ck['epoch']+1; offset=ck['elapsed']
        torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
        np.random.set_state(ck['numpy_rng']); random.setstate(ck['python_rng'])
    torch.cuda.reset_peak_memory_stats(); start=time.monotonic()-offset
    maximum=train_config['epochs']; lr=train_config['lr']
    epoch=first-1
    remaining_epochs=range(first,maximum+1) if stale<train_config['patience'] else []
    for epoch in remaining_epochs:
        rate=lr*epoch/5 if epoch<=5 else 1e-6+(lr-1e-6)*.5*(1+math.cos(math.pi*(epoch-5)/max(1,maximum-5)))
        for group in optimizer.param_groups: group['lr']=rate
        model.train(); order=train[torch.randperm(len(train),device='cuda')]; total=torch.zeros((),device='cuda')
        batches=list(order.split(train_config['batch_size']))
        if len(batches[-1])==1: batches[-2]=torch.cat([batches[-2],batches[-1]]); batches.pop()
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model(h[batch],s[batch],fp[batch] if fp is not None else None).flatten()
                loss=(prediction.float()-z[batch]).square().mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
            total+=loss.detach()*len(batch)
        vp=predict(model,h,s,fp,val)
        rmse=float((vp-y[val]).square().mean().sqrt()); assert math.isfinite(rmse)
        history.append(dict(epoch=epoch,train_mse_z=float(total)/len(train),val_rmse_log2=rmse,lr=rate,elapsed_seconds=time.monotonic()-start))
        improved=rmse<best
        if improved:
            best=rmse; stale=0
            tmp=out/'best.tmp'
            torch.save(dict(model_config=model.config_dict(),state_dict=model.state_dict(),epoch=epoch,val_rmse_log2=best,contract=contract),tmp)
            tmp.replace(out/'best.pt')
        else: stale+=1
        pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
        if epoch%5==0 or stale>=train_config['patience'] or epoch==maximum:
            latest=dict(model_config=model.config_dict(),state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                best=best,stale=stale,history=history,elapsed=time.monotonic()-start,contract=contract,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
            tmp=out/'latest.tmp'; torch.save(latest,tmp); tmp.replace(out/'latest.pt')
        if epoch==1 or epoch%25==0: print(out.name,'epoch',epoch,'val',round(rmse,4),'best',round(best,4),flush=True)
        if stale>=train_config['patience']: break
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    result=dict(best_epoch=ck['epoch'],epochs=epoch,val_rmse_log2=best,seed=seed,train_seconds=time.monotonic()-start,
                parameters=sum(p.numel() for p in model.parameters()),cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                model_config=model_config,train_config=train_config,path=str(out.resolve()))
    dump(out/'summary.json',result); print('FINISHED',json.dumps(result),flush=True)
    return result
