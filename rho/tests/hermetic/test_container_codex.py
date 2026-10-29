import io
import json
import tarfile
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import httpx
import pytest

from rho.agent.container_codex import workspace_archive, restore_harness, budget_identity
from rho.agent.request_budget import ModelRequestBudget


def test_workspace_excludes_host_repository_and_history(tmp_path):
    (tmp_path/'task/repo/.git').mkdir(parents=True)
    (tmp_path/'task/repo/.git/secret').write_text('future answer')
    (tmp_path/'task/repo/code.py').write_text('host checkout')
    (tmp_path/'task/prompt.md').write_text('allowed question')
    archive=tarfile.open(fileobj=io.BytesIO(workspace_archive(tmp_path)))
    assert archive.getnames()==['task','task/prompt.md']


def test_evolution_samples_have_independent_persistent_budgets(tmp_path):
    for sample in range(3):
        key = budget_identity('task', 'h_empty', sample_index=sample, per_sample=True)
        ledger = tmp_path / (key + '.json')
        with ModelRequestBudget('http://127.0.0.1:1', 170, ledger) as budget:
            assert budget.state['requests_sent'] == 0
            # Simulate a fully consumed sample without real model calls.
            budget.state['requests_sent'] = 170
            ledger.write_text(json.dumps(budget.state))
        with ModelRequestBudget('http://127.0.0.1:1', 170, ledger) as resumed:
            assert resumed.state['requests_sent'] == 170
    assert len(list(tmp_path.glob('*.json'))) == 3
    candidate = budget_identity('task', 'h_new', per_sample=True)
    assert not (tmp_path / (candidate + '.json')).exists()


def test_harness_export_rejects_path_escape_and_links(tmp_path):
    target=tmp_path/'harness';target.mkdir();(target/'keep').write_text('original')
    for name, link in [('../outside',False),('link',True)]:
        buf=io.BytesIO()
        with tarfile.open(fileobj=buf,mode='w') as archive:
            member=tarfile.TarInfo(name)
            if link: member.type=tarfile.SYMTYPE;member.linkname='/etc/passwd'
            archive.addfile(member)
        with pytest.raises(ValueError): restore_harness(buf.getvalue(),target)
        assert (target/'keep').read_text()=='original'


def test_unix_model_gateway_injects_host_key_and_limits_requests(tmp_path,monkeypatch):
    monkeypatch.setenv('NO_PROXY','127.0.0.1');monkeypatch.setenv('no_proxy','127.0.0.1')
    received=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length','0')))
            received.append(self.headers.get('Authorization'))
            self.send_response(200);self.end_headers();self.wfile.write(b'{}')
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        socket=tmp_path/'api.sock'
        with ModelRequestBudget(f'http://127.0.0.1:{server.server_port}',2,tmp_path/'ledger.json',unix_socket=socket,authorization='Bearer host-only') as gateway:
            with httpx.Client(transport=httpx.HTTPTransport(uds=str(socket)),base_url='http://localhost') as client:
                route=f'/{gateway.token}/responses'
                assert client.post(route,headers={'Authorization':'Bearer dummy'},json={}).status_code==200
                assert client.post(f'/{gateway.token}/arbitrary',json={}).status_code==400
                assert client.post(route,json={}).status_code==200
                assert client.post(route,json={}).status_code==400
                assert received==['Bearer host-only','Bearer host-only']
                assert gateway.state['requests_sent']==2
        assert not socket.exists()
    finally:
        server.shutdown();server.server_close();thread.join()
