"""Run inside the task container. No model requests or benchmark tests."""
import json
import base64
import os
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import tempfile
import urllib.request
import urllib.error

assert not Path('/mnt/sdc/home/huangy956').exists()
assert not Path('/home/huangy956').exists()
assert not Path('/var/run/docker.sock').exists()
assert subprocess.check_output(['git','-C','/app','rev-list','--all','--count']).strip()==b'1'
context=ssl.create_default_context()
def read(url):
    with urllib.request.urlopen(url,context=context,timeout=90) as response:
        return response.read()
def denied(url):
    try: read(url)
    except (urllib.error.HTTPError,urllib.error.URLError) as exc:
        assert '403' in str(exc), (url,str(exc))
    else:raise AssertionError('Forbidden URL succeeded: '+url)

# A direct outbound socket cannot bypass the dependency gateway.
try: socket.create_connection(('1.1.1.1',443),timeout=2)
except OSError: pass
else:raise AssertionError('Direct network bypass is possible')
for url in ['https://github.com/ansible/ansible/commits/devel',
            'https://raw.githubusercontent.com/ansible/ansible/devel/README.md',
            'https://www.google.com/search?q=benchmark+answer',
            'http://169.254.169.254/latest/meta-data/']:
    denied(url)
policy=json.loads(Path('/rho-dependency-policy.json').read_text())
if policy['blocked_modules']:
    module=policy['blocked_modules'][0]
    denied('https://proxy.golang.org/'+module+'/@latest')
elif policy['blocked_projects']:
    package=policy['blocked_projects'][0]
    denied('https://registry.npmjs.org/'+urllib.parse.quote(package,safe='@'))
versions={}
def run(cmd,**kwargs):
    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=180,**kwargs)
    if proc.returncode:raise RuntimeError(str(cmd)+': '+(proc.stderr+proc.stdout)[-1800:])
    return proc.stdout.strip()
versions['python3']=run(['python3','--version'])
read('https://pypi.org/simple/idna/')
with tempfile.TemporaryDirectory(prefix='dependency-check-') as scratch:
    if Path('/app/go.mod').exists():
        versions['go']=run(['go','version'])
        env=dict(os.environ,GOMODCACHE=scratch+'/go',GOPATH=scratch+'/gopath')
        result=json.loads(run(['go','mod','download','-json','github.com/google/uuid@v1.6.0'],cwd=scratch,env=env))
        assert Path(result['Zip']).is_file()
        downloaded='github.com/google/uuid@v1.6.0'
    elif Path('/app/package.json').exists():
        versions['node']=run(['node','--version']);versions['npm']=run(['npm','--version'])
        run(['npm','pack','is-number@7.0.0','--cache',scratch+'/cache'],cwd=scratch)
        assert list(Path(scratch).glob('*.tgz'))
        downloaded='is-number@7.0.0'
    else:
        run(['python3','-m','pip','download','--no-cache-dir','--no-deps','--disable-pip-version-check','idna==3.10','-d',scratch])
        assert list(Path(scratch).glob('*.whl'))
        downloaded='idna==3.10'
# Exercise nonempty patch export under this image's own Python version.
# Remove the synthetic file and index entry before the real final export.
fd, probe_name = tempfile.mkstemp(prefix='rho-export-probe-', suffix='.py', dir='/app')
os.close(fd)
probe = Path(probe_name)
try:
    probe.write_bytes(b'# export compatibility probe\n')
    run(['git','-C','/app','add','-f','--',probe.name])
    exported = json.loads(run(['python3','/rho-runtime.py','export','HEAD']))
    assert base64.b64decode(exported['changes']['task/repo/'+probe.name]) == probe.read_bytes()
finally:
    probe.unlink()
    run(['git','-C','/app','reset','-q','HEAD','--',probe.name])
value={'passed':True,'tools':versions,'dependency_download':downloaded,'network_policy':policy['version'],
       'answer_sites_blocked':True,'current_project_download_blocked':True,'direct_network_blocked':True,
       'nonempty_patch_export':True}
Path('/workspace/.rho/last_message.txt').write_text(json.dumps(value))
print(json.dumps(value))
