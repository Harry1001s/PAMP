"""Measure the frozen baseline training gap; reuse its existing val/test outputs."""
from common import *
import torch
from improved_kcat_ensemble import ImprovedKcatEnsemble


def main():
    torch.set_num_threads(4); torch.backends.mha.set_fastpath_enabled(False)
    df,manifest,splits,p,s,y=load_data()
    model=ImprovedKcatEnsemble.from_manifest(BASE/'model_manifest.json',device='cpu')
    predictions=[]; ids=splits['train']
    with torch.inference_mode():
        for start in range(0,len(ids),256):
            ix=ids[start:start+256]
            predictions.append(model(torch.tensor(p[ix]),torch.tensor(s[ix])).flatten().numpy())
    pred=np.concatenate(predictions)
    value=metrics(y[ids],pred)
    value['parameters']=sum(p.numel() for p in model.parameters())
    value['manifest_sha256']=sha(BASE/'model_manifest.json')
    dump(OUT/'reports/baseline_train.json',value)
    pd.DataFrame(dict(row_id=ids,y_true=y[ids],y_pred=pred)).to_csv(OUT/'predictions/baseline_train.csv',index=False)
    print(value)


if __name__=='__main__': main()
