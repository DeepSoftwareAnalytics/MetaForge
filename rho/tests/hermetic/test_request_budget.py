from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import httpx
import pytest

from rho.agent.request_budget import ModelRequestBudget, RequestLimitedCodexAgent


@pytest.fixture
def upstream(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data": []}')
        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            calls.append((self.path, body))
            self.send_response(429 if b'force_rate_limit' in body else 200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            self.wfile.write(b'data: {"type":"response.completed"}\n\n')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}/v1', calls
    server.shutdown(); server.server_close(); thread.join()


def test_exact_120_requests_with_concurrent_calls_and_restart(tmp_path, upstream):
    url, calls = upstream
    ledger = tmp_path / 'ledger.json'
    with ModelRequestBudget(url, 120, ledger) as gateway, httpx.Client(timeout=30) as client:
        assert client.get(gateway.base_url + '/models').status_code == 200
        def request(_):
            return client.post(gateway.base_url + '/responses', json={'model': 'test'}).status_code
        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses = list(pool.map(request, range(125)))
        assert statuses.count(200) == 120 and statuses.count(400) == 5
        assert len(calls) == 120
    with ModelRequestBudget(url, 120, ledger) as gateway, httpx.Client() as client:
        response = client.post(gateway.base_url + '/responses', json={'model': 'test'})
        assert response.status_code == 400 and 'model_request_budget_exhausted' in response.text
    assert len(calls) == 120
    assert json.loads(ledger.read_text())['requests_sent'] == 120


def test_rate_limit_attempts_and_compaction_consume_budget(tmp_path, upstream):
    url, calls = upstream
    with ModelRequestBudget(url, 2, tmp_path / 'ledger.json') as gateway, httpx.Client() as client:
        assert client.post(gateway.base_url + '/responses', content=b'force_rate_limit').status_code == 429
        response = client.post(gateway.base_url + '/responses/compact', json={})
        assert response.status_code == 200 and 'response.completed' in response.text
        assert client.post(gateway.base_url + '/responses', json={}).status_code == 400
    assert len(calls) == 2


def test_budget_exhaustion_preserves_patch_and_records_counts(tmp_path, upstream):
    url, calls = upstream
    config = tmp_path / 'config.toml'
    config.write_text(f'model_provider="mock"\n[model_providers.mock]\nname="mock"\nbase_url="{url}"\nwire_api="responses"\n')
    binary = tmp_path / 'codex-mock'
    binary.write_text('''#!/usr/bin/env python3
import json,sys,urllib.request,urllib.error
from pathlib import Path
args=sys.argv[1:]
base=json.loads(next(x.split('=',1)[1] for x in args if x.startswith('model_providers.mock.base_url=')))
ws=Path(args[args.index('--cd')+1])
for i in range(121):
    try:
        urllib.request.urlopen(urllib.request.Request(base+'/responses',data=b'{}',headers={'Content-Type':'application/json'})).read()
    except urllib.error.HTTPError as exc:
        print(json.dumps({'type':'error','message':exc.read().decode()}),flush=True)
        sys.exit(1)
    (ws/'fix.txt').write_text('fixed after request '+str(i+1))
''')
    binary.chmod(0o755)
    ws = tmp_path / 'workspace'
    ws.mkdir()
    agent = RequestLimitedCodexAgent(config, maximum=120, budgets_root=tmp_path / 'budgets',
        binary=str(binary), sandbox='danger-full-access', codex_auth_home=tmp_path/'empty-auth')
    tr = agent.run(ws, 'solve', task_id='task1', harness_id='h_empty')
    assert tr.exit_code == 1 and not tr.timed_out
    assert tr.workspace_diff['fix.txt'] == b'fixed after request 120'
    assert tr.events[-1]['requests_sent'] == 120 and tr.events[-1]['blocked_requests'] == 1
    assert len(calls) == 120
    # Another attempt for the same task/harness cannot reset the budget.
    tr2 = agent.run(ws, 'retry', task_id='task1', harness_id='h_empty')
    assert tr2.events[-1]['requests_sent'] == 120 and len(calls) == 120
