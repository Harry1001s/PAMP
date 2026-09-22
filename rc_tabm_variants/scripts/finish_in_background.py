"""Finish the authorized experiment after the currently running CondPool process."""
import argparse
import fcntl
import os
import subprocess
import time
import traceback

from common import *


def run_script(filename, *args):
    command=[sys.executable,'-u',str(OUT/'scripts'/filename),*map(str,args)]
    print('RUN',command,flush=True)
    subprocess.run(command,check=True)


def process_identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().split()
        return None if stat[2]=='Z' else stat[21]
    except FileNotFoundError:
        return None


def main(pid):
    lock=(OUT/'logs/background_finish.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=OUT/'reports/status.json'
    try:
        identity=process_identity(pid)
        if identity is not None:
            command=Path(f'/proc/{pid}/cmdline').read_bytes()
            assert str(OUT/'scripts/train.py').encode() in command and b'condpool' in command
            dump(status,dict(stage='condpool_seed42_training',training_pid=pid,supervisor_pid=os.getpid(),
                next='freeze best, evaluate, diagnostics, conditional seed confirmation, final report'))
            while process_identity(pid)==identity:
                time.sleep(10)
        cond=json.loads((OUT/'reports/tabm_condpool_metrics.json').read_text())
        mean=json.loads((OUT/'reports/tabm_mean_metrics.json').read_text())
        assert (OUT/'reports/attention_diagnostics.csv').exists(), 'CondPool diagnostics missing; investigate training exit'
        baseline=json.loads((OUT/'reports/baselines.json').read_text())
        baseline_val=next(r['RMSE'] for r in baseline if r['model']=='PAMP predictor' and r['split']=='val')
        gate=cond['metrics']['val']['RMSE']<mean['metrics']['val']['RMSE'] or mean['metrics']['val']['RMSE']<baseline_val-.02
        dump(OUT/'reports/stage2_decision.json',dict(run_three_seed_confirmation=gate,
            criterion='CondPool val RMSE < Mean K16 val RMSE OR Mean K16 val RMSE < baseline val RMSE - 0.02',
            condpool_val_rmse=cond['metrics']['val']['RMSE'],mean_val_rmse=mean['metrics']['val']['RMSE'],
            baseline_val_rmse=baseline_val,test_used_for_gate=False))
        run_script('compare_results.py')
        run_script('verify_deliverables.py')
        if gate:
            for seed in (3407,2026):
                for kind in ('mean','condpool'):
                    dump(status,dict(stage='three_seed_confirmation',model=kind,seed=seed,supervisor_pid=os.getpid()))
                    run_script('train.py','--kind',kind,'--seed',seed)
                    run_script('compare_results.py')
        dump(status,dict(stage='final_verification',supervisor_pid=os.getpid()))
        run_script('compare_results.py')
        run_script('verify_deliverables.py')
        run_script('verify_inference_cli.py')
        dump(status,dict(stage='complete',three_seed_confirmation=gate,
            final_report=str(OUT/'reports/final_report.md'),completed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        print('EXPERIMENT COMPLETE',flush=True)
    except Exception:
        error=traceback.format_exc()
        dump(status,dict(stage='failed',error=error,supervisor_pid=os.getpid()))
        print(error,flush=True)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int,required=True);a=p.parse_args();main(a.wait_pid)
