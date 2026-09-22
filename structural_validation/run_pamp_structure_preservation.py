#!/usr/bin/env python3
"""Resumable ESMFold-v1 structural evaluation for PAMP mutations."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, re, subprocess, sys, time
import gc, io, types, fcntl
from functools import lru_cache
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

AA = set("ACDEFGHIKLMNPQRSTVWY")

def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]

def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)

def atomic_text(path: Path, text: str) -> None:
    atomic_write(path, text.encode())

def clean(s) -> str:
    return re.sub(r"\s+", "", str(s).strip().upper())

def mutation_info(wt: str, mut: str):
    if len(wt) != len(mut): raise ValueError("length mismatch")
    d = [(i,a,b) for i,(a,b) in enumerate(zip(wt,mut)) if a != b]
    if len(d) != 1: raise ValueError(f"expected one substitution, got {len(d)}")
    return d[0]

def load_rows(inp: Path) -> pd.DataFrame:
    df = pd.read_csv(inp, dtype=str, keep_default_na=False)
    required = {"status","orig_seq","mut_seq","mutation","cache_row","sample_id","delta_log2_pred",
                "position_0based","position_1based","wt_aa","mut_aa","n_mutations","seq_len"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"missing columns: {sorted(missing)}")
    if not df.status.str.lower().eq("success").all():
        raise ValueError("input contains non-success rows; expected the successful-row CSV")
    for i,r in df.iterrows():
        wt, mut = clean(r.orig_seq), clean(r.mut_seq)
        if wt != r.orig_seq or mut != r.mut_seq:
            raise ValueError(f'Sequence would require normalization at row {i}; refusing to alter input')
        if not wt or not mut or not set(wt) <= AA or not set(mut) <= AA:
            raise ValueError(f"noncanonical sequence at source row {i}")
        p, a, b = mutation_info(wt, mut)
        if (r.mutation != f"{a}{p+1}{b}" or int(r.position_0based) != p
            or int(r.position_1based) != p+1 or r.wt_aa != a or r.mut_aa != b
            or int(r.n_mutations) != 1 or int(r.seq_len) != len(wt)):
            raise ValueError(f"mutation metadata mismatch at source row {i}")
    if df.cache_row.duplicated().any(): raise ValueError("cache_row is not unique")
    if not np.isfinite(pd.to_numeric(df.delta_log2_pred,errors='coerce')).all():
        raise ValueError('Invalid activity delta')
    return df

def save_json(path: Path, obj) -> None:
    def sanitize(x):
        if isinstance(x, dict): return {k:sanitize(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return [sanitize(v) for v in x]
        if isinstance(x, (float,np.floating)) and not math.isfinite(x): return None
        if isinstance(x, np.generic): return x.item()
        return x
    atomic_text(path, json.dumps(sanitize(obj), indent=2, sort_keys=True, allow_nan=False, default=str))

def load_model(model_dir: str, chunk: int):
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    # Avoid requiring Accelerate in the existing kcat environment; the host has
    # ample RAM and the checkpoint is loaded conventionally before moving to GPU.
    model = EsmForProteinFolding.from_pretrained(model_dir, local_files_only=True, low_cpu_mem_usage=False)
    model.eval()
    if model.config.esmfold_config.trunk.max_recycles != 4:
        raise RuntimeError('Unexpected checkpoint recycle configuration')
    if hasattr(model, "esm"): model.esm = model.esm.half()
    if hasattr(model, "trunk") and hasattr(model.trunk, "set_chunk_size"): model.trunk.set_chunk_size(chunk)
    model = model.cuda()
    return tok, model

def infer_sequence(seq: str, tok, model, chunk: int):
    import torch
    if hasattr(model, "trunk") and hasattr(model.trunk, "set_chunk_size"): model.trunk.set_chunk_size(chunk)
    # The native infer() path constructs the exact AlphaFold residue indices
    # expected by the Transformers ESMFold implementation.
    with torch.inference_mode(): out = model.infer(seq)
    # This pinned implementation's categorical_lddt outputs probabilities 0..1.
    # Scale explicitly, including the PDB B-factor field, rather than guessing
    # the scale from a sequence's observed confidence values.
    out['plddt'] = out['plddt'] * 100.0
    pdb = model.output_to_pdb(out)[0]
    p = out["plddt"].detach().float().cpu()[0]
    exists = out["atom37_atom_exists"].detach().float().cpu()[0]
    if p.ndim == 2:
        # ESMFold v1 emits atom37 pLDDT; use C-alpha, with atom-weighted fallback.
        ca = p[:,1] if p.shape[1] > 1 else (p*exists).sum(-1)/exists.sum(-1).clamp_min(1)
    else: ca = p
    ca = ca.numpy().astype(np.float32)
    if len(ca) != len(seq): raise RuntimeError(f"pLDDT length {len(ca)} != sequence {len(seq)}")
    if not np.isfinite(ca).all() or np.min(ca)<0 or np.max(ca)>100:
        raise RuntimeError('Invalid confidence values')
    ptm = out.get("ptm")
    ptm = float(ptm.detach().float().cpu().reshape(-1)[0]) if ptm is not None else float("nan")
    return pdb, ca, ptm

def parse_atoms(pdb: Path):
    atoms = {}
    for line in pdb.read_text().splitlines():
        if not line.startswith("ATOM"): continue
        atom = line[12:16].strip()
        if atom not in {"N","CA","C","O"}: continue
        alt = line[16:17]
        if alt not in (" ","A"): continue
        key = (line[21:22], int(line[22:26]), line[26:27].strip(), atom)
        if key in atoms: continue
        atoms[key] = np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])], dtype=float)
    return atoms

def backbone_rmsd(wt_pdb: Path, mut_pdb: Path, pos0: int, window: int = 5):
    a,b = parse_atoms(wt_pdb), parse_atoms(mut_pdb)
    residues = sorted({k[:3] for k in a}&{k[:3] for k in b}, key=lambda x:(x[0],x[1],x[2]))
    if set(a)!=set(b) or len(a)!=4*len(residues):
        raise RuntimeError('Incomplete or mismatched backbone residue correspondence')
    names = ["N","CA","C","O"]
    pairs = [(a[r+(n,)], b[r+(n,)]) for r in residues for n in names if r+(n,) in a and r+(n,) in b]
    if not pairs: raise RuntimeError("no paired backbone atoms")
    ref=np.array([x for x,y in pairs]); mob=np.array([y for x,y in pairs])
    rc,mc=ref.mean(0),mob.mean(0); x,y=ref-rc,mob-mc
    u,_,vt=np.linalg.svd(y.T@x); rot=u@vt
    if np.linalg.det(rot)<0: u[:,-1]*=-1; rot=u@vt
    aligned=y@rot+rc
    d=np.linalg.norm(aligned-ref,axis=1)
    # C-alpha displacement and local backbone RMSD use residue order.
    ca_d={r:float(np.linalg.norm((b[r+("CA",)]-mc)@rot+rc-a[r+("CA",)])) for r in residues if r+("CA",) in a and r+("CA",) in b}
    all_rms=float(np.sqrt(np.mean(d*d)))
    if not 0 <= pos0 < len(residues): raise RuntimeError('Mutation position absent in structure')
    site_r=pos0; site=float(ca_d[residues[site_r]])
    lo=max(0,site_r-window); hi=min(len(residues),site_r+window+1)
    local_pairs=[(a[r+(n,)],b[r+(n,)]) for r in residues[lo:hi] for n in names if r+(n,) in a and r+(n,) in b]
    ld=np.array([np.linalg.norm(((y0-mc)@rot+rc)-x0) for x0,y0 in local_pairs])
    return {"backbone_rmsd_angstrom":all_rms,"mutation_site_ca_displacement_angstrom":site,"local_backbone_rmsd_pm5_angstrom":float(np.sqrt(np.mean(ld*ld)))}

TM_RE=re.compile(r"TM-score=\s*([0-9.]+).*?normalized by length of Structure_(\d)",re.I)
ALIGN_RE=re.compile(r"Aligned length=\s*(\d+),\s*RMSD=\s*([0-9.]+)",re.I)
def usalign(wt: Path, mut: Path, binary: str):
    run=subprocess.run([binary,str(mut),str(wt),"-TMscore","1"],capture_output=True,text=True,timeout=120)
    text=(run.stdout or "")+"\n"+(run.stderr or "")
    if run.returncode: raise RuntimeError(f"US-align return code {run.returncode}: {text[:500]}")
    tm={int(i):float(s) for s,i in TM_RE.findall(text)}; m=ALIGN_RE.search(text)
    if not {1,2} <= set(tm) or not m: raise RuntimeError("could not parse US-align scores")
    return {"tm_score_mutant_norm":tm.get(1,float("nan")),"tm_score_wt_norm":tm.get(2,float("nan")),"tm_score":tm.get(2,tm.get(1,float("nan"))),"usalign_aligned_length":int(m.group(1)) if m else np.nan,"usalign_rmsd_angstrom":float(m.group(2)) if m else np.nan}

def local_mean(a,pos,r): return float(np.mean(a[max(0,pos-r):min(len(a),pos+r+1)]))

MODEL_SHA = '2ee07356b125d1e3e57503c204111fd7323347fc4735d41d3caac57c2a78e116'
MODEL_REV = '75a3841ee059df2bf4d56688166c8fb459ddd97a'
SETTINGS = {'schema':4, 'model_revision':MODEL_REV, 'model_sha256':MODEL_SHA,
            'backend':'transformers-4.57.6', 'recycles_total_passes':4,
            'esm_precision':'float16', 'trunk_precision':'float32',
            'plddt_definition':'C-alpha, 0-100', 'seed':2026,
            'chunk_candidates':[64,32,16], 'allow_tf32':False}

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def write_csv(path, df):
    atomic_text(path, df.to_csv(index=False))

@lru_cache(maxsize=4)
def binary_digest(path):
    return file_sha(path)

def cache_paths(out,seq):
    key=sha(json.dumps(SETTINGS,sort_keys=True)+'\n'+seq)
    return key, out/'structures'/f'{key}.pdb', out/'plddt'/f'{key}.npy', out/'sequence_metrics'/f'{key}.json'

def validate_pdb(path, seq):
    from Bio.SeqUtils import seq1
    atoms=parse_atoms(path)
    if len(atoms)!=4*len(seq) or not np.isfinite(list(atoms.values())).all():
        raise ValueError('Missing or invalid backbone atoms')
    ca=[line for line in path.read_text().splitlines() if line.startswith('ATOM') and line[12:16].strip()=='CA']
    if len(ca)!=len(seq) or ''.join(seq1(line[17:20]) for line in ca)!=seq:
        raise ValueError('PDB sequence mismatch')
    if [int(line[22:26]) for line in ca]!=list(range(1,len(seq)+1)):
        raise ValueError('PDB numbering mismatch')

def cached_structure(out,seq):
    key,pp,npy,js=cache_paths(out,seq)
    if not all(p.exists() for p in (pp,npy,js)): return None
    try:
        meta=json.loads(js.read_text())
        if meta['settings']!=SETTINGS or meta['sequence']!=seq: return None
        if meta['pdb_sha256']!=file_sha(pp) or meta['plddt_sha256']!=file_sha(npy): return None
        p=np.load(npy,allow_pickle=False)
        if p.shape!=(len(seq),) or not np.isfinite(p).all() or p.min()<0 or p.max()>100: return None
        validate_pdb(pp,seq)
        return {**meta,'pdb_path':str(pp),'plddt_path':str(npy),'status':'success'}
    except (ValueError,KeyError,OSError): return None

def fold_with_cache(out,seq,tok,model):
    import torch
    from transformers.models.esm.modeling_esmfold import EsmForProteinFolding
    existing=cached_structure(out,seq)
    if existing: return existing
    key,pp,npy,js=cache_paths(out,seq)
    errors=[]
    start=time.monotonic()
    # Offload just the language-model weights between its pass and the folding
    # trunk, avoiding any change to sequence length, precision, or recycles.
    original=EsmForProteinFolding.compute_language_model_representations
    def offloaded(this,esmaa):
        this.esm.cuda()
        try: return original(this,esmaa)
        finally:
            this.esm.cpu()
            torch.cuda.empty_cache()
    for chunk,offload in [(64,False),(32,False),(16,False),(16,True)]:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            model.compute_language_model_representations=types.MethodType(offloaded if offload else original,model)
            model.esm.cuda()
            torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
            pdb,p,ptm=infer_sequence(seq,tok,model,chunk)
            peak=torch.cuda.max_memory_allocated()/1024**3
            buf=io.BytesIO(); np.save(buf,p,allow_pickle=False)
            atomic_text(pp,pdb); atomic_write(npy,buf.getvalue()); validate_pdb(pp,seq)
            meta={'sequence':seq,'sequence_hash':sha(seq),'settings':SETTINGS,'length':len(seq),
                  'mean_plddt':float(p.mean(dtype=np.float64)),'median_plddt':float(np.median(p)),
                  'ptm':ptm,'runtime_sec':time.monotonic()-start,'chunk_size':chunk,
                  'cpu_offload':offload,'peak_gpu_gib':peak,'pdb_sha256':file_sha(pp),
                  'plddt_sha256':file_sha(npy),'retry_errors':errors}
            save_json(js,meta)
            return {**meta,'status':'success','pdb_path':str(pp),'plddt_path':str(npy)}
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            errors.append({'chunk':chunk,'offload':offload,'error':error})
            print(f'length={len(seq)} chunk={chunk} offload={offload}: {error[:180]}',flush=True)
            is_oom='out of memory' in error.lower()
        # Leave the exception scope before releasing CUDA memory (traceback can
        # retain large tensors and defeat an OOM retry).
        gc.collect(); torch.cuda.empty_cache()
        if not is_oom: break
    result={'status':'prediction_error','sequence_hash':sha(seq),'length':len(seq),
            'error':errors[-1]['error'],'attempts':errors,'runtime_sec':time.monotonic()-start}
    save_json(out/'logs'/f'failed_{key}.json',result)
    return result

def pair_metrics(out,wt,mut,cache,binary):
    wm,mm=cache[wt],cache[mut]
    if any(m['status']!='success' for m in [wm,mm]):
        metrics={'structure_status':'prediction_error','structure_error':'; '.join(
            side+': '+m['error'] for side,m in [('wt',wm),('mut',mm)] if m['status']!='success')}
        pos,_,_=mutation_info(wt,mut)
        for side,m in [('wt',wm),('mut',mm)]:
            if m['status']=='success':
                p=np.load(m['plddt_path'],allow_pickle=False)
                metrics[side+'_pdb_path']=m['pdb_path']
                metrics[side+'_mean_plddt']=float(np.mean(p,dtype=np.float64))
                metrics[side+'_mutation_site_plddt']=float(p[pos])
                metrics[side+'_local_plddt_pm5']=local_mean(p,pos,5)
        return metrics
    pair_key=sha(json.dumps(SETTINGS,sort_keys=True)+wt+'|'+mut+binary_digest(binary)
                 +wm.get('pdb_sha256','')+mm.get('pdb_sha256','')
                 +wm.get('plddt_sha256','')+mm.get('plddt_sha256',''))
    pairpath=out/'pair_metrics'/f'{pair_key}.json'
    if pairpath.exists():
        saved=json.loads(pairpath.read_text())
        if saved.get('structure_status')=='success': return saved
    pos,_,_=mutation_info(wt,mut)
    wa=np.load(wm['plddt_path']); ma=np.load(mm['plddt_path'])
    metrics={'structure_status':'success','structure_error':'','wt_pdb_path':wm['pdb_path'],'mut_pdb_path':mm['pdb_path']}
    for name,get in [('mean',lambda a:float(np.mean(a,dtype=np.float64))),
                     ('mutation_site',lambda a:float(a[pos])),
                     ('local_pm5',lambda a:local_mean(a,pos,5))]:
        suffix={'mean':'mean_plddt','mutation_site':'mutation_site_plddt','local_pm5':'local_plddt_pm5'}[name]
        metrics['wt_'+suffix]=get(wa); metrics['mut_'+suffix]=get(ma)
        metrics['delta_'+suffix]=get(ma)-get(wa)
    metrics['local_start_1based']=max(1,pos+1-5); metrics['local_end_1based']=min(len(wt),pos+1+5)
    try:
        metrics.update(usalign(Path(wm['pdb_path']),Path(mm['pdb_path']),binary))
        metrics.update(backbone_rmsd(Path(wm['pdb_path']),Path(mm['pdb_path']),pos))
        if not all(np.isfinite(metrics[k]) for k in ['tm_score','backbone_rmsd_angstrom']): raise ValueError('nonfinite comparison')
    except Exception as exc:
        metrics.update(structure_status='metric_error',structure_error=f'{type(exc).__name__}: {exc}')
    save_json(pairpath,metrics)
    return metrics

def make_results(df,out,cache,binary):
    pairs={}; records=[]
    for _,r in df.iterrows():
        pair=(r.orig_seq,r.mut_seq)
        if pair not in pairs:
            pairs[pair]=pair_metrics(out,*pair,cache,binary) if all(s in cache for s in pair) else {'structure_status':'pending','structure_error':''}
        rec=r.to_dict(); rec.update(pairs[pair]); records.append(rec)
    return pd.DataFrame(records)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--model-dir',required=True); ap.add_argument('--usalign',default='/root/tools/USalign')
    ap.add_argument('--pilot',type=int,default=0); ap.add_argument('--bootstrap',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--summarize-only',action='store_true')
    args=ap.parse_args(); out=Path(args.out).resolve(); inp=Path(args.input).resolve()
    if out==inp.parent or out in inp.parents: raise ValueError('Output must be separate from PAMP input directory')
    for name in ['structures','plddt','sequence_metrics','pair_metrics','logs']: (out/name).mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    df=load_rows(inp); input_sha=file_sha(inp)
    manifest={'input':str(inp),'input_sha256':input_sha,'settings':SETTINGS,
              'n_rows':len(df),'n_unique_sequences':len(set(df.orig_seq)|set(df.mut_seq)),
              'n_unique_pairs':len(df.drop_duplicates(['orig_seq','mut_seq'])),
              'runner_sha256':file_sha(__file__),
              'statistics_sha256':file_sha(Path(__file__).with_name('pamp_structure_statistics.py')),
              'usalign_sha256':file_sha(args.usalign),'bootstrap_repeats':args.bootstrap,
              'bootstrap_seed':args.seed,'bootstrap_unit':'exact WT sequence',
              'model_directory':str(Path(args.model_dir).resolve())}
    old=out/'run_manifest.json'
    if old.exists() and json.loads(old.read_text()).get('input_sha256')!=input_sha:
        raise ValueError('Output directory belongs to a different input')
    save_json(old,manifest)
    if args.summarize_only:
        from pamp_structure_statistics import finalize
        finalize(pd.read_csv(out/'per_sample_structural_results.csv'),out,args.bootstrap,args.seed)
        return
    checkpoint=Path(args.model_dir)/'pytorch_model.bin'
    if checkpoint.stat().st_size!=8442062570 or file_sha(checkpoint)!=MODEL_SHA:
        raise RuntimeError('ESMFold checkpoint incomplete or wrong checksum')
    work=df
    if args.pilot:
        unique=df.drop_duplicates(['orig_seq','mut_seq']).copy()
        unique['sort_len']=unique.orig_seq.str.len()
        unique=unique.sort_values('sort_len',kind='stable')
        # Includes shortest, median, upper tail, and longest pair.
        quantiles=[0,.5,.95,1] if args.pilot==4 else np.linspace(0,1,args.pilot)
        work=unique.iloc[np.unique([round(q*(len(unique)-1)) for q in quantiles])].drop(columns=['sort_len'])
        write_csv(out/'pilot_selection.csv',work)
    cache={}; seqs=sorted(set(work.orig_seq)|set(work.mut_seq),key=lambda s:(len(s),s))
    for s in seqs:
        m=cached_structure(out,s)
        if m: cache[s]=m
    tok=model=None
    remaining=[s for s in seqs if s not in cache]
    if remaining:
        import torch, transformers
        if transformers.__version__!='4.57.6': raise RuntimeError('Unexpected Transformers version')
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        print(f'GPU={torch.cuda.get_device_name(0)} loading checkpoint; remaining={len(remaining)}',flush=True)
        manifest.update(torch_version=torch.__version__,transformers_version=transformers.__version__,
                        gpu=torch.cuda.get_device_name(0),cuda_runtime=torch.version.cuda)
        save_json(old,manifest)
        tok,model=load_model(args.model_dir,64)
    output_name='pilot_results.csv' if args.pilot else 'per_sample_structural_results.csv'
    write_csv(out/output_name,make_results(work,out,cache,args.usalign))
    for i,s in enumerate(remaining):
        cache[s]=fold_with_cache(out,s,tok,model)
        print(f'{i+1}/{len(remaining)} length={len(s)} status={cache[s]["status"]} seconds={cache[s]["runtime_sec"]:.1f}',flush=True)
        if (i+1)%10==0 or args.pilot or i==len(remaining)-1:
            result=make_results(work,out,cache,args.usalign); write_csv(out/output_name,result)
            save_json(out/'progress.json',{'mode':'pilot' if args.pilot else 'full',
                'structures_done':len(cache),'structures_total':len(seqs),
                'row_status_counts':result.structure_status.value_counts().to_dict()})
    result=make_results(work,out,cache,args.usalign); write_csv(out/output_name,result)
    save_json(out/('pilot_sequence_summary.json' if args.pilot else 'sequence_failure_summary.json'),
              {'n_sequences':len(seqs),'n_success':sum(m['status']=='success' for m in cache.values()),
               'n_failed':sum(m['status']!='success' for m in cache.values()),
               'failure_rate':sum(m['status']!='success' for m in cache.values())/len(seqs),
               'failures':[m for m in cache.values() if m['status']!='success']})
    if file_sha(inp)!=input_sha: raise RuntimeError('Input changed during analysis')
    if args.pilot:
        if not result.structure_status.eq('success').all(): raise RuntimeError('Pilot failed; inspect pilot_results.csv')
        print('PILOT PASSED',flush=True)
    else:
        from pamp_structure_statistics import finalize
        finalize(result,out,args.bootstrap,args.seed)
        print('FULL EVALUATION COMPLETE',flush=True)

if __name__=="__main__": main()
