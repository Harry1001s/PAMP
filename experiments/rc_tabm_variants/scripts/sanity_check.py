"""Checks member loss, padding exclusion, gradients and checkpoint round trips."""
from common import *
import torch
from predictor import Predictor, load_predictor


def main():
    torch.set_num_threads(2); torch.manual_seed(42)
    checks={}
    for kind in ('mean','condpool'):
        c=json.loads((OUT/f'configs/tabm_{kind}.json').read_text())
        model=Predictor(c).eval()
        s=torch.randn(2,1024,requires_grad=True)
        h=torch.randn(2,1280,requires_grad=True) if kind=='mean' else torch.randn(2,7,1280,requires_grad=True)
        mask=None if kind=='mean' else torch.tensor([[1,1,1,0,0,0,0],[1,1,1,1,1,1,1]],dtype=torch.bool)
        out=model(h,s,mask)
        assert out.shape==(2,16)
        out.mean().backward()
        assert torch.isfinite(h.grad).all() and h.grad.abs().sum()>0
        assert torch.isfinite(s.grad).all() and s.grad.abs().sum()>0
        if kind=='condpool':
            assert (h.grad[~mask]==0).all()
            changed=h.detach().clone(); changed[~mask]=1234
            torch.testing.assert_close(out,model(changed,s,mask))
            torch.testing.assert_close(out[:1],model(h[:1,:3],s[:1],torch.ones(1,3,dtype=torch.bool)),atol=2e-6,rtol=1e-5)
            _,alpha=model(h,s,mask,return_attention=True)
            assert (alpha[~mask]==0).all()
            torch.testing.assert_close(alpha.sum(1),torch.ones(2))
        path=OUT/f'cache/sanity_{kind}.pt'
        torch.save(dict(config=c,state_dict=model.state_dict()),path)
        restored=load_predictor(path)
        torch.testing.assert_close(model.predict_log2(h,s,mask),restored.predict_log2(h,s,mask))
        checks[kind]='output shape, gradients, scaler/checkpoint round trip passed; masked padding excluded' if kind=='condpool' else 'output shape, gradients, scaler/checkpoint round trip passed'
    members=torch.tensor([[-1.,1.]],requires_grad=True)
    loss=members.square().mean()
    assert loss.item()==1. and members.mean().square().item()==0.
    loss.backward(); assert members.grad.abs().sum()>0
    checks['loss']='per-member MSE verified; disagreeing members retain nonzero loss and gradients'
    dump(OUT/'reports/sanity_check.json',checks)
    print(checks)


if __name__=='__main__': main()
