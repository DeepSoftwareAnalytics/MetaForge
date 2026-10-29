#!/usr/bin/env python3
"""Fresh isolated evolution, followed by matched 120-request held-out arms."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from rho.agent.request_budget import write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=ROOT/'runs/swebench-v1-clean-deps')
    parser.add_argument('--resume-initial',action='store_true')
    args=parser.parse_args();run=args.run_root.resolve()
    lock=(run/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight=json.loads((run/'preflight_passed.json').read_text())
    assert preflight['passed'] is True and preflight['network_policy']=='dependencies-only-v1'
    selection=json.loads((run/'selection.json').read_text());manifest=json.loads((run/'split_manifest.json').read_text())
    core=selection['selected_task_ids'];evaluation=manifest['groups']['evaluation_100']
    assert len(core)==10 and len(set(core))==10 and len(evaluation)==100 and not set(core)&set(evaluation)
    def status(phase,**extra):
        write_json(run/'pipeline_status.json',{'phase':phase,'updated_at':time.time(),
            'solve_workers':8,'grade_workers':8,'evaluation_model_requests':120,
            'evaluation_wall_timeout':None,**extra})
        print(phase,extra,flush=True)
    def wait_images(ids):
        while True:
            images=json.loads((run/'images.json').read_text());missing=[i for i in ids if i not in images]
            if not missing:return
            status('waiting_for_images',ready=len(ids)-len(missing),required=len(ids),missing=missing)
            service=subprocess.run(['systemctl','--user','is-active','--quiet','rho-clean-restore-images.service'])
            if service.returncode:raise RuntimeError('Image restoration stopped before required images were ready')
            time.sleep(15)
    evolution=run/'evolution'
    if not (evolution/'reports/summary.json').exists():
        if (evolution/'config.json').exists() and not args.resume_initial:raise RuntimeError('Interrupted evolution requires checkpoint recovery; refusing to discard it')
        wait_images(core)
        for task_id in core:
            check=json.loads((run/'environment_preflight/results'/(hashlib.sha256(task_id.encode()).hexdigest()+'.json')).read_text())
            assert check['passed'] and check['network_checks']['network_policy']=='dependencies-only-v1'
            assert check['base_comparison']['passed']
        status('evolution',core_tasks=10,initial_rollouts=30,optimize_samples=3)
        command=[sys.executable,str(ROOT/'scripts/run-swebench-v1-evolution.py'),
            '--selection-json',str(run/'selection.json'),'--run-dir',str(evolution),
            '--container-images',str(run/'images.json')]
        if args.resume_initial:command.append('--resume-initial')
        with (run/'evolution.log').open('a') as out:
            result=subprocess.run(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
        if result.returncode:
            status('evolution_failed',exit_code=result.returncode)
            raise RuntimeError('Evolution failed; completed checkpoints retained, see evolution.log')
    wait_images(core+evaluation)
    policy={
        'source_run':str(evolution.relative_to(ROOT)),
        'pairwise_results':'reports/summary.json','pairwise_results_format':'rho_summary',
        'split_manifest':str((run/'split_manifest.json').relative_to(ROOT)),
        'output_root':str((run/'heldout').relative_to(ROOT)),
        'container_images':str((run/'images.json').relative_to(ROOT)),
        'arms':['baseline','final'],'max_model_requests_per_task':120,
        'request_counting':'outbound_model_requests_including_retries_and_compaction',
        'solve_timeout_seconds':None,'solve_workers':8,'grade_workers':8,
        'max_grading_tasks':100,'solve_attempts':1,'model':'deepseek-v4-flash-0731',
        'reasoning_effort':'high','codex_version':'0.163.0','cache':'off'}
    write_json(run/'heldout_policy.json',policy)
    for arm in policy['arms']:
        if (run/'heldout'/arm/'summary.json').exists():continue
        status('heldout_'+arm)
        command=[sys.executable,str(ROOT/'scripts/run-swebench-v1-evaluation.py'),
                 '--policy',str(run/'heldout_policy.json'),'--arm',arm,'--execute']
        for attempt in range(1,4):
            with (run/'heldout.log').open('a') as out:
                result=subprocess.run(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
            if result.returncode==0:break
            if attempt==3:raise RuntimeError(f'{arm} failed after 3 resumable attempts; inspect heldout.log')
            status('retrying_'+arm,attempt=attempt);time.sleep(30)
    status('complete')


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True)
        raise
