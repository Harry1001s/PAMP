"""Predict from precomputed embeddings using the scaler stored in a checkpoint."""
import argparse
from common import *
import torch
from predictor import load_predictor


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--protein',required=True,help='Mean: numeric NPY [N,1280]. CondPool: NPZ with one [Li,1280] array per sample, keys 0..N-1.')
    p.add_argument('--substrate',required=True,help='Numeric NPY [N,1024], same feature contract as training')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    target=Path(args.output)
    if target.exists(): raise FileExistsError(target)
    torch.set_num_threads(4)
    model=load_predictor(args.checkpoint,args.device)
    s=np.load(args.substrate,allow_pickle=False)
    h=np.load(args.protein,allow_pickle=False)
    assert s.ndim==2 and s.shape[1]==1024 and np.isfinite(s).all()
    if model.kind=='mean': assert h.shape==(len(s),1280) and np.isfinite(h).all()
    rows=[]
    with torch.inference_mode():
        for i in range(len(s)):
            protein=h[i:i+1] if model.kind=='mean' else h[str(i)][None]
            assert np.isfinite(protein).all()
            mask=None if model.kind=='mean' else torch.ones(protein.shape[:2],dtype=torch.bool,device=args.device)
            members=model(torch.tensor(protein,device=args.device),torch.tensor(s[i:i+1],device=args.device),mask).float()
            members=members*model.target_scale+model.target_mean
            rows.append(dict(sample_id=i,pred_log2_kcat=float(members.mean()),pred_std_members=float(members.std(unbiased=False))))
    pd.DataFrame(rows).to_csv(target,index=False)


if __name__=='__main__': main()
