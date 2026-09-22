"""Launch the user-selected fixed CataPro/A3 run in a detached process."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
OUT=HERE/'catapro_test_2697_residual_a3_distinct'


def main():
    assert json.loads((HERE/'adapter_checks.json').read_text())['status']=='PASS'
    assert json.loads((HERE/'smoke_catapro_long/verification.json').read_text())['status']=='PASS'
    assert json.loads((HERE/'smoke_distinct/verification.json').read_text())['two_distinct_sites_verified']
    OUT.mkdir(exist_ok=True)
    pidpath=OUT/'run.pid'
    if pidpath.exists():
        pid=int(pidpath.read_text())
        path=Path('/proc')/str(pid)/'cmdline'
        if path.exists() and str(OUT).encode() in path.read_bytes():
            print(f'Already running PID {pid}');return
    status=OUT/'status.json'
    if status.exists() and json.loads(status.read_text()).get('stage')=='complete':
        print('Already complete');return
    snapshot=OUT/'source';snapshot.mkdir(exist_ok=True)
    for name in ['run.py','attack_adapter.py','test_adapter.py','launch_catapro_a3.py','README.md']:
        destination=snapshot/name
        if destination.exists():assert destination.read_bytes()==(HERE/name).read_bytes()
        else:shutil.copy2(HERE/name,destination)
    command=[sys.executable,'-u',str(HERE/'run.py'),'--dataset','catapro','--out',str(OUT),'--methods','a3','--min-length','80','--max-length','1024','--sources','residual','--site-policy','distinct']
    with (OUT/'run.log').open('ab',buffering=0) as log:
        process=subprocess.Popen(command,cwd=HERE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pidpath.write_text(str(process.pid)+'\n')
    (OUT/'launch.json').write_text(json.dumps(dict(pid=process.pid,command=command,extra_trees=False,sources=['residual'],methods=['A3_fw_avg_pamp'],test_rows=2697,length_bounds=[80,1024],rounds=2,first_round_topk=5,site_policy='distinct'),indent=2))
    print(json.dumps(dict(pid=process.pid,out=str(OUT),log=str(OUT/'run.log'))))


if __name__=='__main__':main()
