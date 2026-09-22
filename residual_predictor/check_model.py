import json
import numpy as np
import pandas as pd
from residual_model import *
from cache_reader import CataproResidueCache


def main():
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.mha.set_fastpath_enabled(False)
    config=json.loads((EXPERIMENT/'config.json').read_text())
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


if __name__=='__main__': main()
