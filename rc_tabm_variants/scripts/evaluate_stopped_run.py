"""Evaluate the preserved best checkpoint after user-requested early stop. No training."""
from train import *


def main():
    kind='condpool';seed=42;name='tabm_condpool'
    report=OUT/'reports/tabm_condpool_metrics.json'
    if report.exists():
        print('Existing final metrics found; test inference will not repeat',flush=True)
        return
    assert not (OUT/'predictions/tabm_condpool_test.csv').exists(), 'Existing test predictions require recovery without repeating inference'
    ckdir=OUT/'checkpoints/tabm_condpool'
    ck=torch.load(ckdir/'best.pt',map_location='cpu',weights_only=False)
    config=ck['config']
    hist=pd.read_csv(OUT/'logs/tabm_condpool_seed42.csv')
    epoch=int(hist.epoch.iloc[-1]);history=hist.to_dict('records')
    assert ck['epoch']==int(hist.loc[hist.val_rmse.idxmin(),'epoch'])
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    df,manifest,splits,p,s,y=load_data()
    for path,digest in ck['provenance']['files'].items():assert sha(path)==digest
    model=Predictor(config).cuda()
    mapping=pd.read_csv(ROOT/'catpro_esm2_mean_pooling/row_mapping.csv').unique_sequence_index.to_numpy()
    batches=Batches(config,p,s,y,mapping,'cuda')
    start=time.monotonic()
    dump(OUT/'reports/manual_early_stop.json',dict(reason='User explicitly requested immediate early stop',last_completed_epoch=epoch,best_epoch=ck['epoch'],best_val_rmse=ck['val_rmse'],checkpoint_sha256=sha(ckdir/'best.pt'),additional_training_disabled=True,training_peak_gpu_memory='not recorded before process termination'))
    ck = torch.load(ckdir/'best.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(ck['state_dict'])
    # Freeze and record the selected checkpoint BEFORE touching test outcomes.
    dump(ckdir/'selection.json',dict(epoch=ck['epoch'],validation_rmse=ck['val_rmse'],checkpoint_sha256=sha(ckdir/'best.pt'),test_used_for_selection=False))
    result=dict(model=name,seed=seed,best_epoch=ck['epoch'],epochs=epoch,parameters=sum(p.numel() for p in model.parameters()),
                training_seconds=float(hist.seconds.iloc[-1]),peak_gpu_memory_mib=None,stop_reason="user_requested_early_stop",checkpoint=str(ckdir/'best.pt'),metrics={})
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


if __name__=='__main__': main()
