"""Frozen ESM2 extraction matching the existing mean pipeline exactly."""
import argparse
import shutil
import time

from common import *
import torch


def main(limit=None):
    cache=OUT/'cache/esm2_residue'; cache.mkdir(parents=True,exist_ok=True)
    df,manifest,splits,protein,substrate,y=load_data()
    mapping=pd.read_csv(ROOT/'catpro_esm2_mean_pooling/row_mapping.csv').unique_sequence_index.to_numpy()
    sequences=list(dict.fromkeys(df.Sequence))
    ref=np.load(ROOT/'catpro_esm2_mean_pooling/unique_mean.partial.npy')
    paths=[cache/f'{i:05d}.npy' for i in range(len(sequences))]
    contract=dict(source_sha256=sha(SOURCE),split_sha256=sha(SPLIT),model='esm2_t33_650M_UR50D',layer=33,
        dtype='float16',lengths=[len(s) for s in sequences],sequence_sha256=[hashlib.sha256(s.encode()).hexdigest() for s in sequences],
        row_mapping=mapping.tolist(),chunking='non-overlapping <=1022; BOS/EOS excluded; full float32 frozen ESM2 inference',
        checkpoint_sha256=sha(Path('/root/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt')))
    if (cache/'index.json').exists():
        assert json.loads((cache/'index.json').read_text())==contract
    else:
        dump(cache/'index.json',contract)
    required=sum(len(s)*1280*2+128 for s,p in zip(sequences,paths) if not p.exists())
    assert shutil.disk_usage(cache).free > required+512*1024**2, 'Insufficient free space for complete ragged cache'
    missing=[i for i,p in enumerate(paths) if not p.exists()]
    if limit: missing=missing[:limit]
    torch.set_num_threads(4)
    # Match generate_catpro_mean_pooling.py: no autocast, default float32 matmul.
    torch.backends.cuda.matmul.allow_tf32=False
    import esm
    model,alphabet=esm.pretrained.esm2_t33_650M_UR50D()
    model=model.cuda().eval().requires_grad_(False)
    convert=alphabet.get_batch_converter(); start=time.monotonic()
    diagnostics=[]
    with torch.inference_mode():
        for n,i in enumerate(missing):
            seq=sequences[i]; parts=[]
            for j in range(0,len(seq),1022):
                chunk=seq[j:j+1022]
                _,_,tokens=convert([('protein',chunk)])
                result=model(tokens.cuda(),repr_layers=[33],return_contacts=False)
                parts.append(result['representations'][33][0,1:len(chunk)+1].float().cpu().numpy())
            raw=np.concatenate(parts)
            mean=raw.mean(0,dtype=np.float64)
            # Tolerance accounts for float32 reduction/kernel variation, before storage quantization.
            max_error=float(np.abs(mean-ref[i]).max())
            if max_error>2e-4:
                dump(OUT/'reports/residue_alignment_failure.json',dict(unique_sequence_index=i,length=len(seq),max_abs_difference=max_error))
                raise RuntimeError(f'ESM2 mean mismatch at {i}: {max_error}; refusing to continue silently')
            value=raw.astype(np.float16)
            assert value.shape==(len(seq),1280) and np.isfinite(value).all()
            temp=paths[i].with_suffix('.tmp')
            with temp.open('wb') as f: np.save(f,value)
            temp.replace(paths[i])
            diagnostics.append(dict(unique_sequence_index=i,length=len(seq),fp32_mean_max_abs_error=max_error,
                                    stored_mean_max_abs_error=float(np.abs(value.mean(0,dtype=np.float32)-ref[i]).max())))
            if (n+1)%25==0 or n+1==len(missing):
                print('RESIDUES',n+1,'/',len(missing),'unique',i,'seconds',round(time.monotonic()-start),flush=True)
                pd.DataFrame(diagnostics).to_csv(cache/'latest_extraction_alignment.csv',index=False)
    del model
    if limit and any(not p.exists() for p in paths):
        print('SMOKE EXTRACTION COMPLETE',flush=True); return
    lengths=[]; errors=[]
    for i,p in enumerate(paths):
        a=np.load(p,mmap_mode='r')
        assert a.shape==(len(sequences[i]),1280) and np.isfinite(a).all()
        errors.append(float(np.abs(a.mean(0,dtype=np.float32)-ref[i]).max()))
        lengths.append(len(a))
    assert max(errors)<.002, 'Stored fp16 means disagree with existing mean cache'
    alignment=json.loads((OUT/'reports/cache_alignment_check.json').read_text())
    alignment['status']='PASS: dataset/mean/residue/substrate/label aligned'
    alignment['residue']=dict(all_unique_sequences_checked=len(sequences),length_match=True,
        maximum_stored_mean_absolute_error=max(errors),mean_of_max_absolute_errors=float(np.mean(errors)),
        reason_for_small_difference='Float16 residue storage quantization and float32 reductions; extraction itself matches original float32 ESM2 pipeline',
        tolerance=.002,cache_index_sha256=sha(cache/'index.json'))
    for sample in alignment['samples']:
        uid=sample['unique_sequence_index']
        sample['residue_length']=lengths[uid]; sample['residue_mean_max_abs_error']=errors[uid]
    dump(OUT/'reports/cache_alignment_check.json',alignment)
    dump(cache/'completion.json',dict(unique_sequences=len(sequences),residues=sum(lengths),storage_bytes=sum(p.stat().st_size for p in paths),max_mean_error=max(errors)))
    print('RESIDUE CACHE COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args();main(a.limit)
