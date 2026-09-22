"""Re-encode audit samples with the original legacy batch slots, on CPU."""
import argparse
from common import *
import torch
from generate_unikp_smiles1024_fixed import load_unikp_modules,load_vocab_compat,get_ids,encode_batch_unikp


def main(device='cpu'):
    torch.set_num_threads(4)
    # Original run_mean_kcat_comparison enables TF32 before prepare() encodes substrates.
    torch.backends.cuda.matmul.allow_tf32=True
    df,manifest,splits,p,s,y=load_data()
    audit=json.loads((OUT/'reports/cache_alignment_check.json').read_text())
    unique=list(dict.fromkeys(manifest.canonical_smiles))
    lookup={v:i for i,v in enumerate(unique)}
    modules,Trfm,split_fn=load_unikp_modules(ROOT/'UniKP')
    vocab=load_vocab_compat(ROOT/'UniKP/vocab.pkl',modules)
    model=Trfm(len(vocab),256,len(vocab),4).to(device).eval()
    model.load_state_dict(torch.load(ROOT/'UniKP/trfm_12_23000.pkl',map_location='cpu',weights_only=True))
    groups=sorted({lookup[sample['canonical_smiles']]//64 for sample in audit['samples']})
    encoded={}
    for group in groups:
        start=group*64
        src=torch.tensor([get_ids(sm,split_fn,vocab) for sm in unique[start:start+64]],device=device).T.contiguous()
        values=encode_batch_unikp(model,src).cpu().numpy()
        for j,value in enumerate(values): encoded[start+j]=value
    result=[]
    for sample in audit['samples']:
        i=sample['row_id']; uid=lookup[sample['canonical_smiles']]
        delta=float(np.abs(encoded[uid]-s[i]).max())
        result.append(dict(row_id=i,unique_smiles_index=uid,legacy_batch_slot=uid%64,max_abs_error=delta))
    passed=all(r['max_abs_error']<3e-5 for r in result)
    dump(OUT/f'reports/substrate_reencoding_{device}.json',dict(status='PASS' if passed else 'MISMATCH',device=device,tf32=True,batch_size=64,samples=result))
    assert passed, f'Substrate re-encoding mismatch: maximum {max(r["max_abs_error"] for r in result)}'
    print('PASS: 20 substrate vectors numerically reproduced with original batch slots',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args();main(a.device)
