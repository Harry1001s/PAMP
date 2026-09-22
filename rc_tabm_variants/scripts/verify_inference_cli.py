"""Exercise the public inference CLI on training rows only."""
import subprocess
from common import *


def main():
    df,manifest,splits,p,s,y=load_data()
    ids=splits['train'][:3]
    folder=OUT/'cache/inference_checks';folder.mkdir(exist_ok=True)
    if not (folder/'mean.npy').exists(): np.save(folder/'mean.npy',p[ids])
    if not (folder/'substrate.npy').exists(): np.save(folder/'substrate.npy',s[ids])
    results=[]
    for path in sorted((OUT/'reports').glob('tabm_*_metrics.json')):
        run=json.loads(path.read_text());name=run['model']
        inp=folder/'mean.npy'
        if name.startswith('tabm_condpool'):
            from cache_reader import CataproResidueCache
            inp=folder/'residues.npz'
            if not inp.exists():
                cache=CataproResidueCache()
                np.savez(inp,**{str(j):np.asarray(cache[i]) for j,i in enumerate(ids)})
        output=folder/f'{name}.csv'
        if not output.exists():
            subprocess.run([sys.executable,str(OUT/'scripts/predict.py'),'--checkpoint',run['checkpoint'],
                            '--protein',str(inp),'--substrate',str(folder/'substrate.npy'),'--output',str(output)],check=True)
        pred=pd.read_csv(output)
        assert len(pred)==len(ids) and np.isfinite(pred[['pred_log2_kcat','pred_std_members']]).all().all()
        ref=pd.read_csv(OUT/f'predictions/{name}_train.csv').set_index('row_id').loc[ids]
        # CLI defaults to CPU float32; training metrics used CUDA bf16 autocast.
        delta=float(np.abs(pred.pred_log2_kcat.to_numpy()-ref.y_pred.to_numpy()).max())
        assert delta<.1, f'Unexpected inference precision discrepancy: {delta}'
        results.append(dict(model=name,training_rows=ids.tolist(),status='PASS',
                            maximum_cpu_fp32_vs_cuda_bf16_difference_log2=delta,test_rows_used=False))
    dump(OUT/'reports/inference_cli_verification.json',results)
    print(results)


if __name__=='__main__': main()
