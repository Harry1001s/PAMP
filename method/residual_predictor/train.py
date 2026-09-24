"""Seed-42 residual screening; validation-selected baseline anchor and automatic reporting."""
import json
import random
import signal
import time
import traceback
import shutil
import numpy as np
import pandas as pd
from residual_model import *
from common import Batches,atomic_checkpoint,CataproResidueCache

STOP_REQUESTED=False


def stop_signal(signum,frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True


def should_stop():
    return STOP_REQUESTED or (EXPERIMENT/'STOP').exists()


def globals_from_cache(splits,y):
    result=np.full(len(y),np.nan,dtype=np.float64)
    paths={}
    for split in ('train','val','test'):
        path=SCREEN/'predictions/baseline_train.csv' if split=='train' else BASE/f'{split}_predictions.csv'
        f=pd.read_csv(path);ids=splits[split]
        np.testing.assert_array_equal(f.row_id,ids)
        np.testing.assert_allclose(f.y_true if split=='train' else f.y_log2,y[ids],atol=1e-12)
        result[ids]=f.y_pred if split=='train' else f['PAMP predictor']
        paths[str(path)]=sha(path)
    assert np.isfinite(result).all()
    return result,paths


@torch.no_grad()
def evaluate(local,batches,ids,global_y):
    local.eval();outputs={}
    for ix in batches.order(ids):
        H,s,mask,y=batches.get(ix)
        with torch.autocast('cuda',dtype=torch.bfloat16):delta=local(H,s,mask)
        d=delta.float().cpu().numpy()
        for i,v,sd in zip(ix,d.mean(1),d.std(1)): outputs[int(i)]=(v,sd)
    a=np.asarray([outputs[int(i)] for i in ids])
    return global_y[ids]+a[:,0],a[:,0],a[:,1]


def save_best(local,config,epoch,val_rmse,provenance):
    atomic_checkpoint(EXPERIMENT/'checkpoints/best.pt',dict(config=config,state_dict=local.state_dict(),epoch=epoch,
        val_rmse=val_rmse,gamma=float(local.gamma.detach()),global_manifest=str(BASE/'model_manifest.json'),
        global_manifest_sha256=sha(BASE/'model_manifest.json'),provenance=provenance))


def check_model():
    (EXPERIMENT/'reports').mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.mha.set_fastpath_enabled(False)
    config=json.loads(CONFIG.read_text())
    df,manifest,splits,p,s,y=load_data();ids=splits['train'][:3]
    cache=CataproResidueCache();lengths=[len(cache[i]) for i in ids];L=max(lengths)+3
    H=torch.zeros(3,L,1280);mask=torch.zeros(3,L,dtype=torch.bool)
    for j,i in enumerate(ids): H[j,:lengths[j]]=torch.tensor(np.asarray(cache[i]).copy());mask[j,:lengths[j]]=True
    model=ResidualPredictor(config)
    model.local_branch.set_scalers(p,s,y,splits['train']);model.train()
    assert not model.global_branch.training and all(not v.requires_grad for v in model.global_branch.parameters())
    h=torch.tensor(p[ids],requires_grad=True);sub=torch.tensor(s[ids]);H.requires_grad_()
    g=model.global_branch(h,sub).flatten();out=model(h,H,sub,mask)
    cached=pd.read_csv(SCREEN/'predictions/baseline_train.csv').set_index('row_id').loc[ids,'y_pred'].to_numpy()
    np.testing.assert_allclose(g.detach().numpy(),cached,atol=2e-5,rtol=1e-6)
    torch.testing.assert_close(out,g[:,None].expand_as(out),rtol=0,atol=0)
    loss=((out-torch.tensor(y[ids],dtype=torch.float32)[:,None])/model.local_branch.target_scale).square().mean()
    loss.backward()
    assert model.local_branch.gamma.grad is not None and abs(model.local_branch.gamma.grad.item())>1e-10
    assert all(v.grad is None for v in model.global_branch.parameters())
    assert h.grad.abs().sum()>0
    assert H.grad.abs().sum()==0
    initial_gamma_grad=model.local_branch.gamma.grad.item()
    opt=torch.optim.AdamW(model.local_branch.parameters(),lr=config['lr'])
    opt.step();opt.zero_grad();H.grad=None;h.grad=None
    out=model(h,H,sub,mask);out.square().mean().backward()
    assert H.grad.abs().sum()>0 and torch.isfinite(H.grad).all()
    assert (H.grad[~mask]==0).all()
    assert model.local_branch.protein_projection.weight.grad.abs().sum()>0
    model.eval();changed=H.detach().clone();changed[~mask]=1234
    torch.testing.assert_close(model(h,H,sub,mask),model(h,changed,sub,mask),atol=1e-6,rtol=1e-6)
    temporary=EXPERIMENT/'reports/sanity_checkpoint.pt'
    torch.save(dict(config=config,state_dict=model.local_branch.state_dict(),global_manifest=str(BASE/'model_manifest.json'),
                    global_manifest_sha256=sha(BASE/'model_manifest.json')),temporary)
    restored=load_predictor(temporary)
    torch.testing.assert_close(model(h,H,sub,mask),restored(h,H,sub,mask),atol=0,rtol=0)
    result=dict(status='PASS',zero_gamma_exactly_preserves_original_prediction=True,global_weights_frozen=True,
        global_eval_mode_preserved=True,initial_gamma_gradient=initial_gamma_grad,
        local_gradients_activate_after_gamma_step=True,padding_excluded=True,cached_global_predictions_match=True,
        checkpoint_roundtrip=True,training_rows=ids.tolist())
    dump(EXPERIMENT/'reports/model_checks.json',result);print(result)


def main():
    for name in ('reports','logs','predictions','checkpoints'):
        (EXPERIMENT/name).mkdir(parents=True,exist_ok=True)
    status=EXPERIMENT/'reports/status.json'
    if (EXPERIMENT/'reports/metrics.json').exists():
        print('Already complete; test inference is not repeated');return
    assert not (EXPERIMENT/'checkpoints/best.pt').exists(), 'Existing partial run: do not overwrite'
    if not (EXPERIMENT/'reports/model_checks.json').exists():
        check_model()
    assert json.loads((EXPERIMENT/'reports/model_checks.json').read_text())['status']=='PASS'
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    config=json.loads(CONFIG.read_text())
    seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    df,manifest,splits,p,s,y=load_data()
    global_y,global_files=globals_from_cache(splits,y)
    parent=json.loads((SCREEN/'reports/run_manifest.json').read_text())
    for file,digest in parent['files'].items():assert sha(file)==digest
    assert sha(SCREEN/'cache/esm2_residue/index.json')==parent['residue_cache']['index_sha256']
    assert sha(SCREEN/'cache/esm2_residue/file_sha256.json')==parent['residue_cache']['file_hash_manifest_sha256']
    provenance=dict(source_files=parent['files'],residue_cache=parent['residue_cache'],global_prediction_files=global_files,
        global_manifest=str(BASE/'model_manifest.json'),global_manifest_sha256=sha(BASE/'model_manifest.json'),
        config=config,python=parent['python'],tabm=parent['tabm'],rtdl_num_embeddings=parent['rtdl_num_embeddings'],
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        cached_global_arithmetic='Existing frozen original predictor, CPU float32; training loss uses float32; evaluation adds float32 correction to original saved predictions',
        local_amp='bfloat16',screening='SCREENING ONLY — legacy substrate cache',
        selection='minimum validation RMSE including epoch 0; no test selection',code_sha256={p.name:sha(p) for p in (EXPERIMENT/'scripts').glob('*.py')})
    dump(EXPERIMENT/'reports/run_manifest.json',provenance)
    snapshot=EXPERIMENT/'checkpoints/source';snapshot.mkdir(exist_ok=True)
    for source in (EXPERIMENT/'scripts').glob('*.py'):shutil.copy2(source,snapshot/source.name)
    local=LocalCorrection(config);local.set_scalers(p,s,y,splits['train']);local=local.cuda()
    gamma_params=[local.gamma];other=[v for n,v in local.named_parameters() if n!='gamma']
    opt=torch.optim.AdamW([dict(params=other,weight_decay=config['weight_decay']),dict(params=gamma_params,weight_decay=0.)],lr=config['lr'])
    mapping=pd.read_csv(ROOT/'catpro_esm2_mean_pooling/row_mapping.csv').unique_sequence_index.to_numpy()
    batches=Batches(config,p,s,y,mapping,'cuda')
    g=torch.tensor(global_y,dtype=torch.float32,device='cuda')
    base_train=metrics(y[splits['train']],global_y[splits['train']]);base_val=metrics(y[splits['val']],global_y[splits['val']])
    best=base_val['RMSE'];stale=0
    save_best(local,config,0,best,provenance)
    history=[dict(epoch=0,train_loss=float(np.mean(((global_y[splits['train']]-y[splits['train']])/float(local.target_scale))**2)),
                  train_rmse=base_train['RMSE'],val_rmse=best,val_R2=base_val['R2'],val_PCC=base_val['PCC'],val_Spearman=base_val['Spearman'],
                  gamma=0.,generalization_gap=best-base_train['RMSE'],seconds=0.)]
    pd.DataFrame(history).to_csv(EXPERIMENT/'logs/training.csv',index=False)
    dump(status,dict(stage='training',epoch=0,best_epoch=0,best_val_rmse=best,gamma=0.))
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();last_completed=0
    print('BASELINE_ANCHOR',base_val,flush=True)
    for epoch in range(1,config['max_epochs']+1):
        if should_stop():break
        local.train();total=0.;count=0
        for ix in batches.order(splits['train'],training=True):
            if should_stop():break
            H,sub,mask,target=batches.get(ix);ix_gpu=torch.as_tensor(ix,device='cuda')
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                correction=local(H,sub,mask)
                # Separate loss for each TabM member, in standardized log2 units.
                error=(g[ix_gpu,None]+correction-target[:,None])/local.target_scale
                loss=error.float().square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite residual loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(local.parameters(),config['grad_clip'],error_if_nonfinite=True);opt.step()
            total+=float(loss.detach())*len(ix);count+=len(ix)
        if should_stop():break
        vp,vd,vs=evaluate(local,batches,splits['val'],global_y)
        tp,td,ts=evaluate(local,batches,splits['train'],global_y)
        vm=metrics(y[splits['val']],vp);tm=metrics(y[splits['train']],tp)
        last_completed=epoch
        history.append(dict(epoch=epoch,train_loss=total/count,train_rmse=tm['RMSE'],val_rmse=vm['RMSE'],
            val_R2=vm['R2'],val_PCC=vm['PCC'],val_Spearman=vm['Spearman'],gamma=float(local.gamma.detach()),
            generalization_gap=vm['RMSE']-tm['RMSE'],train_mean_abs_correction=float(np.abs(td).mean()),
            val_mean_abs_correction=float(np.abs(vd).mean()),seconds=time.monotonic()-start))
        if vm['RMSE']<best:
            best=vm['RMSE'];stale=0;save_best(local,config,epoch,best,provenance)
        else:stale+=1
        pd.DataFrame(history).to_csv(EXPERIMENT/'logs/training.csv',index=False)
        dump(status,dict(stage='training',epoch=epoch,best_val_rmse=best,stale_epochs=stale,gamma=float(local.gamma.detach())))
        print('EPOCH',epoch,'val_RMSE',round(vm['RMSE'],6),'best',round(best,6),'gamma',round(float(local.gamma.detach()),6),'seconds',round(time.monotonic()-start),flush=True)
        if stale>=config['patience']:break
    train_seconds=time.monotonic()-start;peak=torch.cuda.max_memory_allocated()/2**20
    ck=torch.load(EXPERIMENT/'checkpoints/best.pt',map_location='cuda',weights_only=False);local.load_state_dict(ck['state_dict'])
    dump(EXPERIMENT/'checkpoints/selection.json',dict(epoch=ck['epoch'],gamma=ck['gamma'],validation_rmse=ck['val_rmse'],
        checkpoint_sha256=sha(EXPERIMENT/'checkpoints/best.pt'),test_used_for_selection=False))
    dump(status,dict(stage='evaluating',best_epoch=ck['epoch']))
    result=dict(seed=seed,best_epoch=ck['epoch'],gamma=ck['gamma'],last_completed_epoch=last_completed,
        stop_reason='user_requested' if should_stop() else 'patience_or_epoch_limit',training_seconds=train_seconds,
        peak_gpu_memory_mib=peak,local_parameters=sum(v.numel() for v in local.parameters()),global_parameters_frozen=6983429,
        checkpoint=str(EXPERIMENT/'checkpoints/best.pt'),metrics={},baseline_metrics={})
    for split in ('train','val','test'):
        ids=splits[split];pred,delta,std=evaluate(local,batches,ids,global_y)
        result['metrics'][split]=dict(**metrics(y[ids],pred),mean_abs_correction=float(np.abs(delta).mean()),mean_member_std=float(std.mean()))
        result['baseline_metrics'][split]=metrics(y[ids],global_y[ids])
        pd.DataFrame(dict(sample_id=df.iloc[ids,0].to_numpy(),row_id=ids,split=split,y_true=y[ids],y_global=global_y[ids],
            delta_local=delta,gamma=ck['gamma'],y_pred=pred,error=pred-y[ids],pred_std_members=std)).to_csv(EXPERIMENT/f'predictions/{split}.csv',index=False)
    result['generalization_gap']=result['metrics']['val']['RMSE']-result['metrics']['train']['RMSE']
    assert result['metrics']['val']['RMSE']<=base_val['RMSE']+1e-8
    dump(EXPERIMENT/'reports/metrics.json',result)
    make_report(result,history,local,batches,df,manifest,splits,global_y)
    dump(status,dict(stage='complete',best_epoch=ck['epoch'],gamma=ck['gamma'],report=str(EXPERIMENT/'reports/final_report.md')))
    print('COMPLETE',json.dumps(result),flush=True)


def make_report(result,history,local,batches,df,manifest,splits,global_y):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h=pd.DataFrame(history);fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(h.epoch,h.train_rmse,label='Train');axes[0].plot(h.epoch,h.val_rmse,label='Validation')
    axes[0].axhline(result['baseline_metrics']['val']['RMSE'],ls='--',color='gray',label='Frozen global validation')
    axes[0].set_ylabel('RMSE, log2(kcat)');axes[0].legend();axes[1].plot(h.epoch,h.gamma);axes[1].set_ylabel('Gamma')
    for ax in axes:ax.set_xlabel('Epoch');ax.axvline(result['best_epoch'],ls=':',color='gray')
    fig.tight_layout();fig.savefig(EXPERIMENT/'reports/training_curve.png',dpi=160);plt.close(fig)
    rows=[]
    for split in ('val','test'):
        for name,m in [('Original predictor',result['baseline_metrics'][split]),('Residual CondPool',result['metrics'][split])]:
            rows.append(dict(model=name,split=split,**{k:m[k] for k in ('R2','RMSE','MAE','PCC','Spearman')}))
        ref=next(r for r in json.loads((SCREEN/'reports/baselines.json').read_text()) if r['model']=='Extra Trees' and r['split']==split)
        rows.append(dict(model='Extra Trees',split=split,**{k:ref[k] for k in ('R2','RMSE','MAE','PCC','Spearman')}))
    pd.DataFrame(rows).to_csv(EXPERIMENT/'reports/comparison.csv',index=False)
    text='# Residual CondPool 实验结果\n\nSCREENING ONLY — legacy substrate cache\n\n'
    text+='结构：冻结原 predictor(mean, substrate) + γ × Local-TabM(residues, substrate)。γ 初始为 0，原模型作为 epoch 0 候选。所有 scaler 只在训练集拟合，验证集选择 checkpoint，测试集仅在权重固定后评估。\n\n'
    text+=f'最佳 epoch：{result["best_epoch"]}；γ：{result["gamma"]:.6f}。\n\n'
    text+='| 模型 | Split | R² | RMSE | MAE | Pearson/PCC | Spearman |\n|---|---|---:|---:|---:|---:|---:|\n'
    for r in rows:text+=f'| {r["model"]} | {r["split"]} | {r["R2"]:.4f} | {r["RMSE"]:.4f} | {r["MAE"]:.4f} | {r["PCC"]:.4f} | {r["Spearman"]:.4f} |\n'
    b=result['baseline_metrics'];m=result['metrics']
    text+=f'\n验证 RMSE 变化：{m["val"]["RMSE"]-b["val"]["RMSE"]:+.4f}；测试 RMSE 变化：{m["test"]["RMSE"]-b["test"]["RMSE"]:+.4f}。\n'
    text+=f'\nTrain–validation gap：{result["generalization_gap"]:.4f}；原模型 gap：{b["val"]["RMSE"]-b["train"]["RMSE"]:.4f}。\n'
    text+='\n结论：'+('未选中任何残差更新，保留原模型。' if result['best_epoch']==0 else '验证和测试 RMSE 均下降，值得进一步独立重复；单 seed 尚不能证明稳定提升。' if m['test']['RMSE']<b['test']['RMSE'] else '验证集改善未转化为测试集改善，当前不建议替换原模型。')
    text+='\n\n全局分支训练输出使用既有冻结预测缓存，避免重复计算和更新全局权重。残差拟合使用原模型的训练集内误差，仍可能过拟合；γ 小并不自动保证泛化。PCC 与 Pearson 为同一指标。\n'
    (EXPERIMENT/'reports/final_report.md').write_text(text)
    checks=[]
    for split,ids in splits.items():
        frame=pd.read_csv(EXPERIMENT/f'predictions/{split}.csv');np.testing.assert_array_equal(frame.row_id,ids)
        np.testing.assert_allclose(frame.y_pred,frame.y_global+frame.delta_local,atol=1e-7)
        for k,v in metrics(frame.y_true,frame.y_pred).items():np.testing.assert_allclose(v,result['metrics'][split][k],atol=1e-6)
        checks.append(split)
    selection=json.loads((EXPERIMENT/'checkpoints/selection.json').read_text())
    assert sha(EXPERIMENT/'checkpoints/best.pt')==selection['checkpoint_sha256']
    assert int(h.loc[h.val_rmse.idxmin(),'epoch'])==result['best_epoch']
    dump(EXPERIMENT/'reports/verification.json',dict(status='PASS',prediction_splits=checks,additive_identity=True,baseline_anchor_retained=True,test_inference_repeated=False))
    attention=[];selected=np.random.default_rng(42).choice(splits['test'],20,replace=False)
    saved=pd.read_csv(EXPERIMENT/'predictions/test.csv').set_index('row_id')
    local.eval()
    with torch.no_grad():
        for i in selected:
            H,s,mask,_=batches.get(np.array([i]))
            with torch.autocast('cuda',dtype=torch.bfloat16):_,alpha=local(H,s,mask,return_attention=True)
            w=alpha[0,:len(df.Sequence[i])].cpu().numpy();assert np.isfinite(w).all()
            np.testing.assert_allclose(w.sum(),1.,atol=1e-5);top=np.argsort(-w)[:10]
            attention.append(dict(row_id=int(i),sample_id=df.iloc[i,0],length=len(w),substrate=manifest.canonical_smiles[i],
                top10_positions=json.dumps((top+1).tolist()),weights=json.dumps(w.tolist()),
                normalized_entropy=float(-(w*np.log(np.maximum(w,1e-30))).sum()/np.log(len(w))),
                y_global=float(saved.loc[i,'y_global']),delta_local=float(saved.loc[i,'delta_local']),y_pred=float(saved.loc[i,'y_pred'])))
    pd.DataFrame(attention).to_csv(EXPERIMENT/'reports/attention_diagnostics.csv',index=False)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Train the residual correction branch.')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    try:
        check_model() if args.check_only else main()
    except Exception:
        dump(EXPERIMENT/'reports/status.json',dict(stage='failed',error=traceback.format_exc()))
        raise
