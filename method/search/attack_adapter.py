"""Frozen original/residual predictors with full-residue anchored ESM2 gradients.

The chunkwise vector-Jacobian product is the chain rule for the full predictor;
it does not detach the local branch or replace full residues by their mean.
"""
import sys as _sys, pathlib as _pl
for _c in _pl.Path(__file__).resolve().parents:
    if (_c / 'pamp_paths.py').exists():
        _sys.path.insert(0, str(_c)); break
from pamp_paths import DATA_ROOT, ESM2_CHECKPOINT, add_module_paths, RESIDUAL_DIR
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = DATA_ROOT
RES = ROOT/'experiments/catapro_residual_condpool'
add_module_paths(RESIDUAL_DIR)
from residual_model import load_predictor

AA = 'ACDEFGHIKLMNPQRSTVWY'
METHODS = ['A3_fw_avg_pamp','A0_pamp','A2_fw_avg','A1_hotflip','B2_random']


def rank(scores, sequence, count=5, per_site=2, blocked_positions=()):
    scores=np.asarray(scores,dtype=np.float64).copy()
    assert scores.shape==(len(sequence),20)
    for position in blocked_positions:
        assert 0<=position<len(sequence)
        scores[position,:]=-np.inf
    for i,a in enumerate(sequence):
        if a in AA:scores[i,AA.index(a)]=-np.inf
        else:scores[i,:]=-np.inf
    candidates=[];used={}
    for flat in np.argsort(-scores.ravel(),kind='stable'):
        i,a=divmod(int(flat),20)
        if not np.isfinite(scores[i,a]) or used.get(i,0)>=per_site:continue
        candidates.append((i,AA[a]));used[i]=used.get(i,0)+1
        if len(candidates)==count:return candidates
    raise ValueError('Insufficient legal substitutions')


def mutate(sequence,position,aa):
    assert aa in AA and sequence[position] in AA and aa!=sequence[position]
    return sequence[:position]+aa+sequence[position+1:]


class Engine:
    def __init__(self,device='cuda'):
        self.device=device
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.mha.set_fastpath_enabled(False)
        self.model=load_predictor(RES/'checkpoints/best.pt',device=device).requires_grad_(False)
        import esm
        with torch.serialization.safe_globals([argparse.Namespace]):
            self.esm,self.alphabet=esm.pretrained.load_model_and_alphabet_local(str(ESM2_CHECKPOINT))
        self.esm=self.esm.to(device).eval().requires_grad_(False)
        self.convert=self.alphabet.get_batch_converter()
        self.aa_ids=torch.tensor([self.alphabet.get_idx(a) for a in AA],device=device)
        self.aa=self.esm.embed_tokens.weight[self.aa_ids].detach()
        self.dist=torch.cdist(self.aa,self.aa).square()
        indices=torch.triu_indices(20,20,1,device=device)
        self.median=float(self.dist[indices[0],indices[1]].sqrt().median())

    def chunk(self,sequence,delta=None,gradient=False):
        _,_,tokens=self.convert([('protein',sequence)])
        tokens=tokens.to(self.device);captured=[]
        def hook(_module,_args,value):
            value=value.detach().clone()
            if delta is not None:
                value[:,1:len(sequence)+1]+=delta[None]
            value.requires_grad_(gradient);captured.append(value)
            return value
        handle=self.esm.embed_tokens.register_forward_hook(hook)
        try:
            with torch.set_grad_enabled(gradient):
                result=self.esm(tokens,repr_layers=[33],return_contacts=False)
        finally:handle.remove()
        return result['representations'][33][0,1:len(sequence)+1],result['logits'][0,1:len(sequence)+1,self.aa_ids],captured[0]

    def encode(self,sequence,delta=None):
        hs=[];ls=[]
        for start in range(0,len(sequence),1022):
            stop=min(start+1022,len(sequence))
            h,logits,_=self.chunk(sequence[start:stop],None if delta is None else delta[start:stop])
            hs.append(h.detach());ls.append(logits.detach())
        return torch.cat(hs),torch.cat(ls)


class AttackRow:
    def __init__(self,engine,sequence,protein_mean,substrate,residue_anchor=None):
        self.e=engine;self.sequence=str(sequence)
        self.mean0=torch.as_tensor(protein_mean,dtype=torch.float32,device=engine.device).reshape(1,-1)
        self.s=torch.as_tensor(substrate,dtype=torch.float32,device=engine.device).reshape(1,-1)
        self.live0,self.logits0=engine.encode(sequence)
        self.h0=self.live0.half().float() if residue_anchor is None else torch.as_tensor(residue_anchor,dtype=torch.float32,device=engine.device)
        assert self.h0.shape==self.live0.shape==(len(sequence),1280)
        assert torch.isfinite(self.h0).all()
        self.meanlive0=self.live0.mean(0,keepdim=True)
        self.mask=torch.ones(1,len(sequence),dtype=torch.bool,device=engine.device)

    def head(self,live,target):
        mean=self.mean0+(live.mean(0,keepdim=True)-self.meanlive0)
        global_y=self.e.model.global_branch(mean,self.s).reshape(())
        if target=='original':return global_y
        if target!='residual':raise ValueError(target)
        H=self.h0+(live-self.live0)
        correction=self.e.model.local_branch(H[None],self.s,self.mask).mean()
        return global_y+correction

    def values(self,live):
        with torch.no_grad():
            mean=self.mean0+(live.mean(0,keepdim=True)-self.meanlive0)
            return dict(original=float(self.head(live,'original')),residual=float(self.head(live,'residual'))),mean.cpu().numpy().reshape(-1)

    def state(self,sequence,target,delta=None,gradient=True):
        assert len(sequence)==len(self.sequence)
        H,logits=self.e.encode(sequence,delta)
        H=H.detach().requires_grad_(gradient)
        with torch.set_grad_enabled(gradient):
            value=self.head(H,target)
        result=dict(value=float(value.detach()),logits=logits)
        if gradient:
            # Predictor depends jointly on all residues. Compute its full VJP
            # first, then propagate each chunk through the frozen ESM2 encoder.
            dH=torch.autograd.grad(value,H)[0].detach()
            grads=[]
            for start in range(0,len(sequence),1022):
                stop=min(start+1022,len(sequence))
                rep,_,inputs=self.e.chunk(sequence[start:stop],None if delta is None else delta[start:stop],True)
                g=torch.autograd.grad(rep,inputs,grad_outputs=dH[start:stop])[0][0,1:stop-start+1]
                grads.append(g.detach())
                del rep,inputs,g
            result['gradient']=torch.cat(grads)
            assert torch.isfinite(result['gradient']).all() and result['gradient'].norm()>0
        return result

    def scores(self,sequence,gradient,penalized=True):
        ids=torch.tensor([self.e.alphabet.get_idx(a) for a in sequence],device=self.e.device)
        residues=self.e.esm.embed_tokens.weight[ids].detach()
        raw=gradient@self.e.aa.T-(gradient*residues).sum(1,keepdim=True)
        if penalized:
            epsilon=2*self.e.median*np.sqrt(len(sequence))
            distance=torch.cdist(residues,self.e.aa).square()
            raw=2*epsilon*raw/gradient.norm().clamp_min(1e-12)-distance
        return raw.detach().cpu().numpy()

    def path(self,sequence,target,initial,blocked_positions=()):
        gradient=initial['gradient'];gradients=[gradient];trace=[]
        delta=torch.zeros_like(gradient)
        ids=torch.tensor([self.e.alphabet.get_idx(a) for a in sequence],device=self.e.device)
        residues=self.e.esm.embed_tokens.weight[ids].detach()
        for step in range(2):
            position,aa=rank(self.scores(sequence,gradient,False),sequence,1,blocked_positions=blocked_positions)[0]
            vertex=torch.zeros_like(delta)
            vertex[position]=self.e.aa[AA.index(aa)]-residues[position]
            delta=.5*delta+.5*vertex
            current=self.state(sequence,target,delta)
            gradient=current['gradient'];gradients.append(gradient)
            trace.append(dict(step=step+1,position=position+1,aa=aa,value=current['value']))
        return torch.stack(gradients).mean(0),trace

    def proposals(self,sequence,target,seed,count=5,methods=METHODS,blocked_positions=()):
        needs_gradient=any(m != 'B2_random' for m in methods)
        current=self.state(sequence,target,gradient=needs_gradient)
        scores={};trace=[]
        if needs_gradient:
            scores['A0_pamp']=self.scores(sequence,current['gradient'])
            scores['A1_hotflip']=self.scores(sequence,current['gradient'],False)
        if any(m in methods for m in ('A3_fw_avg_pamp','A2_fw_avg')):
            avg,trace=self.path(sequence,target,current,blocked_positions)
            scores['A3_fw_avg_pamp']=self.scores(sequence,avg)
            scores['A2_fw_avg']=self.scores(sequence,avg,False)
        scores['B2_random']=np.random.default_rng(seed).random((len(sequence),20))
        return {m:rank(scores[m],sequence,count,blocked_positions=blocked_positions) for m in methods},current['value'],trace
