"""Supervise download -> verified length-stratified pilot -> full evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'pamp_structure_preservation_esmfold_v1'

def status(stage,**kwargs):
    obj={'stage':stage,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs}
    tmp=OUT/'pipeline_status.tmp'; tmp.write_text(json.dumps(obj,indent=2)); os.replace(tmp,OUT/'pipeline_status.json')
    print(json.dumps(obj),flush=True)

def command(args,logname):
    with (OUT/'logs'/logname).open('a',buffering=1) as log:
        child=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        status(logname,child_pid=child.pid)
        code=child.wait()
        if code: raise RuntimeError(f'{logname} exited {code}; see {OUT/"logs"/logname}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--download-pid',type=int); args=ap.parse_args()
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    status('waiting_for_download',pid=os.getpid())
    if args.download_pid:
        proc=Path(f'/proc/{args.download_pid}/cmdline')
        while proc.exists():
            try:
                if b'download_esmfold_checkpoint.py' not in proc.read_bytes(): break
            except FileNotFoundError: break
            time.sleep(15)
    try:
        command([sys.executable,'-u','download_esmfold_checkpoint.py'],'download.log')
        base=[sys.executable,'-u','run_pamp_structure_preservation.py',
              '--input',str(ROOT/'pamp_ensemble_outputs/pamp_ensemble_full_results.csv'),
              '--out',str(OUT),'--model-dir',str(ROOT/'.models/esmfold_v1'),
              '--bootstrap','10000']
        command(base+['--pilot','4'],'pilot.log')
        command(base,'full.log')
        status('complete')
    except Exception as exc:
        status('error',error=str(exc)); raise

if __name__=='__main__': main()
