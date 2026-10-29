"""Compare trusted runtime preparation against its immutable source image."""
from collections import Counter
import json
from pathlib import PurePosixPath
import subprocess


def change_category(path):
    p=PurePosixPath(path)
    if path in {'/sbin/docker-init','/usr/sbin/docker-init'}:return 'container_init_process'
    if '.git' in p.parts or '/pkg/mod/cache/vcs' in path:return 'git_history_cleanup'
    if path in {'/build.sh','/preprocess.sh','/root/.bash_history','/root/.python_history','/root/.gitconfig','/root/.git-credentials'}:
        return 'build_recipe_or_history_cleanup'
    if path.startswith(('/workspace','/rho-home','/rho-codex','/rho-runtime','/rho-model','/rho-dependency')):
        return 'isolated_runtime_files'
    if path in {'/etc/profile.d/rho-image-path.sh','/etc/profile.d/rho-dependencies.sh','/etc/ssl/certs/ca-certificates.crt'}:
        return 'toolchain_path_or_dependency_proxy_certificate'
    if '__pycache__' in p.parts and (p.suffix=='.pyc' or p.name=='__pycache__'):return 'generated_python_bytecode'
    return None


def inspect_image_diff(container,audit,source_comparison):
    result=subprocess.run(['docker','diff',container],capture_output=True,text=True,check=True,timeout=120)
    (audit/'base_image_diff.txt').write_text(result.stdout)
    entries=[(line[0],line[2:]) for line in result.stdout.splitlines() if len(line)>2]
    uncertain=[path for status,path in entries if change_category(path) is None and status=='C']
    check=subprocess.run(['docker','exec','-i',container,'python3','-c',
        'import json,sys,os; print(json.dumps([p for p in json.load(sys.stdin) if os.path.isdir(p)]))'],
        input=json.dumps(uncertain),capture_output=True,text=True,check=True,timeout=60)
    directories=set(json.loads(check.stdout));counts=Counter();unexpected=[]
    verified={'/app/'+p for p in source_comparison['verified_unchanged_paths']}
    for status,path in entries:
        category=change_category(path)
        if category is None and status=='C' and path in verified:
            category='source_metadata_only_content_and_mode_verified'
        if category is None and path in directories: category='parent_directory_metadata'
        if category is None: unexpected.append({'status':status,'path':path});category='unexpected'
        counts[category]+=1
    report={'original_image_unchanged':True,'source':source_comparison,'change_counts':dict(counts),
            'unexpected_changes':unexpected,'passed':not unexpected and not source_comparison['changed_paths']}
    (audit/'base_image_comparison.json').write_text(json.dumps(report,indent=2))
    if not report['passed']:raise RuntimeError('Unexpected base-image changes: '+repr(unexpected[:20]))
    return report
