"""Exercise the locally pinned Codex against a fake Responses server (no paid API)."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest

from rho.agent.request_budget import RequestLimitedCodexAgent


def test_pinned_codex_stops_before_third_model_request(tmp_path, monkeypatch):
    binary = Path('/mnt/sdc/home/huangy956/.codex/packages/standalone/releases/0.163.0-x86_64-unknown-linux-musl/bin/codex')
    if not binary.exists():
        pytest.skip('Pinned experimental binary is not installed')
    monkeypatch.setenv('NO_PROXY', 'localhost,127.0.0.1')
    monkeypatch.setenv('no_proxy', 'localhost,127.0.0.1')
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            n = len(requests)
            rid = f'resp_{n}'
            events = [
                {'type': 'response.created', 'response': {'id': rid}},
                {'type': 'response.output_item.done', 'item': {'type': 'function_call',
                 'call_id': f'call_{n}', 'name': 'exec_command',
                 'arguments': json.dumps({'cmd': 'printf fixed > fix.txt'})}},
                {'type': 'response.completed', 'response': {'id': rid,
                 'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2}}},
            ]
            body = ''.join(f'data: {json.dumps(e)}\n\n' for e in events).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = tmp_path/'config.toml'
    config.write_text(f'''model_provider = "local_mock"
approval_policy = "never"
[model_providers.local_mock]
name = "Local Mock"
base_url = "http://127.0.0.1:{server.server_port}/v1"
wire_api = "responses"
requires_openai_auth = false
request_max_retries = 0
stream_max_retries = 0
''')
    ws = tmp_path/'workspace'
    ws.mkdir()
    try:
        agent = RequestLimitedCodexAgent(config, maximum=2, budgets_root=tmp_path/'budgets',
            binary=str(binary), model='deepseek-v4-flash-0731', sandbox='danger-full-access',
            codex_auth_home=tmp_path/'empty-auth', default_timeout_s=60, stream_logs=True)
        tr = agent.run(ws, 'Use exec_command to write fix.txt.', task_id='test', harness_id='h_empty')
        assert not tr.timed_out, tr.stderr[-1500:]
        assert len(requests) == 2, tr.stderr[-1500:]
        assert tr.events[-1]['requests_sent'] == 2
        assert tr.events[-1]['blocked_requests'] >= 1
        assert tr.workspace_diff['fix.txt'] == b'fixed'
    finally:
        server.shutdown(); server.server_close(); thread.join()
