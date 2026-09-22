"""Launch the five remaining methods on the fixed 2697-row test cohort."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
OUT=HERE/'catapro_test_2697_residual_remaining_distinct'


def main():
    check=json.loads((HERE/'benchmark_remaining_distinct/verification.json').read_text())
    assert check['status']=='PASS' and check['two_distinct_sites_verified']
    OUT.mkdir(exist_ok=True)
    pidfile=OUT/'run.pid'
    if pidfile.exists():
        pid=int(pidfile.read_text());cmd=Path('/proc')/str(pid)/'cmdline'
        if cmd.exists() and str(OUT).encode() in cmd.read_bytes():
            print('Already running',pid);return
    status=OUT/'status.json'
    if status.exists() and json.loads(status.read_text()).get('stage')=='complete':
        print('Already complete');return
    snapshot=OUT/'source';snapshot.mkdir(exist_ok=True)
    for name in ['run.py','attack_adapter.py','launch_remaining.py']:
        target=snapshot/name
        if target.exists():assert target.read_bytes()==(HERE/name).read_bytes()
        else:shutil.copy2(HERE/name,target)
    command=[sys.executable,'-u',str(HERE/'run.py'),'--dataset','catapro','--out',str(OUT),
        '--methods','remaining','--sources','residual','--site-policy','distinct','--min-length','80','--max-length','1024']
    with (OUT/'run.log').open('ab',buffering=0) as log:
        process=subprocess.Popen(command,cwd=HERE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pidfile.write_text(str(process.pid)+'\n')
    (OUT/'launch.json').write_text(json.dumps(dict(pid=process.pid,command=command,methods=['A0_pamp','A2_fw_avg','A1_hotflip','B0_esm_lm','B2_random'],records=2697,extra_trees=False),indent=2))
    print(json.dumps(dict(pid=process.pid,out=str(OUT),log=str(OUT/'run.log'))))


if __name__=='__main__':main()
