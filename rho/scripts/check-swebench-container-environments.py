#!/usr/bin/env python3
"""Validate the selected 10 + held-out 100 environments without model calls."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import subprocess
from rho.agent.container_codex import ContainerCodexAgent
from rho.agent.codex_pool import configure_global_codex_pool
from rho.agent.request_budget import write_json
from rho.local_settings import codex_config, configure_network

ROOT=Path(__file__).resolve().parents[1]
SHIM_HEADER = "#!/bin/sh\nset -eu\ncd /app\n. /etc/profile.d/rho-image-path.sh\n. /etc/profile.d/rho-dependencies.sh\npython3 - <<'CHECK'\n"
SHIM = SHIM_HEADER + (ROOT/'scripts/container-runtime/check_dependencies.py').read_text() + "\nCHECK\n"



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=ROOT/'runs/swebench-v1-clean-deps')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--core-only',action='store_true')
    parser.add_argument('--task-id',action='append')
    args=parser.parse_args();run=args.run_root.resolve();out=run/'environment_preflight';out.mkdir(exist_ok=True)
    manifest=json.loads((run/'split_manifest.json').read_text());core=json.loads((run/'selection.json').read_text())['selected_task_ids'];evaluation=manifest['groups']['evaluation_100'];ids=args.task_id or (core if args.core_only else core+evaluation)
    shim=out/'offline-check';shim.write_text(SHIM);shim.chmod(0o755)
    os.environ['DASHSCOPE_API_KEY']='unused-offline-preflight'
    configure_network()
    configure_global_codex_pool(args.workers)
    agent=ContainerCodexAgent(codex_config(ROOT),images_path=run/'images.json',audit_root=out/'runtime',binary=shim,maximum=1,default_timeout_s=600)
    lock=threading.Lock();active={};passed={};failed={}
    for path in (out/'results').glob('*.json'):
        value=json.loads(path.read_text())
        if value['passed'] and value.get('base_comparison',{}).get('passed'):passed[value['task_id']]=value
        elif not value['passed']:failed[value['task_id']]=value
    def status():
        with lock:
            value={'phase':'running','core_passed':sum(i in passed for i in core),'core_total':10,
                'evaluation_passed':sum(i in passed for i in evaluation),'evaluation_total':100,
                'active':dict(active),'failed':dict(failed),'workers':args.workers,
                'model_requests':0,'updated_at':time.time()}
            if all(i in passed for i in ids):value['phase']='complete'
            write_json(out/'status.json',value)
    def check(task_id):
        if task_id in passed:return
        with lock:active[task_id]='waiting_for_image';failed.pop(task_id,None)
        status()
        try:
            while task_id not in json.loads((run/'images.json').read_text()):
                if subprocess.run(['systemctl','--user','is-active','--quiet','rho-clean-restore-images.service']).returncode:
                    raise RuntimeError('Required local image missing and restore service is not active')
                time.sleep(15)
            with lock:active[task_id]='checking'
            status()
            with tempfile.TemporaryDirectory(prefix='rho-environment-check-') as tmp:
                ws=Path(tmp);(ws/'harness').mkdir();(ws/'task').mkdir()
                (ws/'task/prompt.md').write_text('Offline environment validation only. No benchmark statement or grading material is supplied.')
                tr=agent.run(ws,'Offline environment validation',task_id=task_id,harness_id='environment-check',kind='solve')
            budget=next(e for e in tr.events if e.get('type')=='model_request_budget')
            assert budget['requests_sent']==0
            if tr.exit_code!=0 or tr.timed_out:raise RuntimeError(tr.stderr[-2200:] or 'preflight timeout/failure')
            assert not tr.workspace_diff and not tr.workspace_deletions
            detail=json.loads(tr.final_message);assert detail['passed']
            value={'task_id':task_id,'passed':True,'image_id':json.loads((run/'images.json').read_text())[task_id]['image_id'],
                   'trajectory_id':tr.id,'tools':detail['tools'],'model_requests':0,'network_checks':detail,
                   'dependency_policy':json.loads(Path(next(e['audit'] for e in tr.events if e.get('type')=='container_isolation')).read_text())['dependency_policy'],
                   'base_comparison':json.loads(Path(next(e['audit'] for e in tr.events if e.get('type')=='container_isolation')).read_text())['base_comparison']}
            with lock:passed[task_id]=value
        except Exception as exc:
            value={'task_id':task_id,'passed':False,'error':type(exc).__name__+': '+str(exc)}
            with lock:failed[task_id]=value
        write_json(out/'results'/(hashlib.sha256(task_id.encode()).hexdigest()+'.json'),value)
        with lock:active.pop(task_id,None)
        status();print(json.dumps({'task_id':task_id,'passed':value['passed'],
            'error':value.get('error'),'download':value.get('network_checks',{}).get('dependency_download'),
            'source_changes':value.get('base_comparison',{}).get('source',{}).get('changed_paths')}),flush=True)
    status()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in as_completed([pool.submit(check,i) for i in ids]):result.result()
    status()
    if failed:raise RuntimeError(f'{len(failed)} environment checks failed; see status.json')


if __name__=='__main__':main()
