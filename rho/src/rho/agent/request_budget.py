"""Per-task HTTP model-request budget for isolated evaluation agents."""
from __future__ import annotations

from contextlib import AbstractContextManager
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import socket
import threading
import time
import tomllib
from urllib.parse import urlsplit

import httpx

from rho.agent.codex import CodexAgent


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


class ModelRequestBudget(AbstractContextManager):
    """Count actual outbound inference attempts, including retries and compaction.

    Requests are reserved durably *before* forwarding. A restart does not refund
    attempts. This controls the configured provider transport, not arbitrary
    network access from shell tools (the experiment has no OS network sandbox).
    """

    def __init__(self, upstream_url: str, maximum: int, ledger: Path, *,
                 unix_socket: Path | None = None, authorization: str | None = None):
        if maximum < 1:
            raise ValueError("maximum must be positive")
        self.upstream_url = upstream_url.rstrip("/")
        self.maximum = maximum
        self.ledger = ledger
        self.lock = threading.Lock()
        self.closing = False
        self.token = secrets.token_urlsafe(24)
        self.unix_socket = unix_socket
        self.authorization = authorization

    def __enter__(self):
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.file_lock = self.ledger.with_suffix(".lock").open("a")
        fcntl.flock(self.file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if self.ledger.exists():
                self.state = json.loads(self.ledger.read_text())
                if self.state["maximum"] != self.maximum:
                    raise ValueError("Cannot change an existing task's request budget")
            else:
                self.state = {"maximum": self.maximum, "requests_sent": 0, "blocked_requests": 0,
                              "counting": "outbound_model_requests_including_retries_and_compaction", "requests": []}
                write_json(self.ledger, self.state)
            self.client = httpx.Client(timeout=httpx.Timeout(3600, connect=60), follow_redirects=False)
            owner = self

            class Handler(BaseHTTPRequestHandler):
                protocol_version = "HTTP/1.1"

                def log_message(self, *_args):
                    pass  # Never log headers, prompts, or provider credentials.

                def reply_error(self, code, message):
                    body = json.dumps({"error": {"type": "invalid_request_error", "message": message}}).encode()
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(body)
                    self.close_connection = True

                def do_POST(self):
                    self.forward()

                def do_GET(self):
                    self.forward()

                def forward(self):
                    prefix = f"/{owner.token}"
                    parsed = urlsplit(self.path)
                    if not parsed.path.startswith(prefix + "/"):
                        self.reply_error(403, "Unknown evaluation session")
                        return
                    route = parsed.path[len(prefix):]
                    inference = self.command == "POST" and route in ("/responses", "/responses/compact", "/chat/completions")
                    metadata = self.command == "GET" and route == "/models" and not self.headers.get("Upgrade")
                    if not (inference or metadata):
                        self.reply_error(400, "Unsupported evaluation transport route")
                        return
                    if self.headers.get("Transfer-Encoding"):
                        self.reply_error(400, "Evaluation transport requires Content-Length")
                        return
                    body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                    index = None
                    if inference:
                        with owner.lock:
                            if owner.closing or owner.state["requests_sent"] >= owner.maximum:
                                owner.state["blocked_requests"] += 1
                                write_json(owner.ledger, owner.state)
                                self.reply_error(400, "model_request_budget_exhausted: maximum model requests reached")
                                return
                            index = owner.state["requests_sent"]
                            owner.state["requests_sent"] += 1
                            owner.state["requests"].append({"number": index + 1, "path": route,
                                                           "started_at": time.time(), "status": "reserved"})
                            write_json(owner.ledger, owner.state)
                    headers = {k: v for k, v in self.headers.items()
                               if k.lower() not in ("host", "connection", "content-length", "transfer-encoding", "accept-encoding")}
                    headers["Accept-Encoding"] = "identity"
                    if owner.authorization is not None:
                        headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
                        headers["Authorization"] = owner.authorization
                    target = owner.upstream_url + route + ("?" + parsed.query if parsed.query else "")
                    response_started = False
                    try:
                        with owner.client.stream(self.command, target, headers=headers, content=body) as response:
                            if index is not None:
                                owner.record_status(index, response.status_code)
                            self.send_response(response.status_code)
                            for key, value in response.headers.items():
                                if key.lower() not in ("connection", "transfer-encoding", "content-length"):
                                    self.send_header(key, value)
                            self.send_header("Connection", "close")
                            self.end_headers()
                            response_started = True
                            for chunk in response.iter_raw():
                                self.wfile.write(chunk)
                                self.wfile.flush()
                    except (httpx.HTTPError, OSError) as exc:
                        if index is not None:
                            owner.record_status(index, type(exc).__name__)
                        if not response_started:
                            try:
                                self.reply_error(502, "Evaluation model transport failed")
                            except OSError:
                                pass
                    finally:
                        self.close_connection = True

            if self.unix_socket is not None:
                class UnixHTTPServer(ThreadingHTTPServer):
                    address_family = socket.AF_UNIX

                    def server_bind(self):
                        self.socket.bind(self.server_address)
                        self.server_name = "localhost"
                        self.server_port = 0

                self.server = UnixHTTPServer(str(self.unix_socket), Handler)
                self.unix_socket.chmod(0o666)
            else:
                self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            self.base_url = f"http://127.0.0.1:{self.server.server_port}/{self.token}"
            return self
        except BaseException:
            self.file_lock.close()
            raise

    def record_status(self, index, status):
        with self.lock:
            if self.closing:
                return
            self.state["requests"][index]["status"] = status
            write_json(self.ledger, self.state)

    def __exit__(self, *exc):
        with self.lock:
            self.closing = True
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.client.close()
        self.file_lock.close()
        if self.unix_socket is not None:
            self.unix_socket.unlink(missing_ok=True)


class RequestLimitedCodexAgent:
    """Evaluation-only wrapper: separate ledger for each task and harness arm."""

    def __init__(self, config_path: Path, *, maximum: int, budgets_root: Path, **agent_kwargs):
        self.config_path = Path(config_path)
        config = tomllib.loads(self.config_path.read_text())
        self.provider = config["model_provider"]
        provider = config["model_providers"][self.provider]
        if provider.get("wire_api") != "responses":
            raise ValueError("Budgeted evaluation requires the Responses HTTP transport")
        self.upstream_url = provider["base_url"]
        self.maximum = maximum
        self.budgets_root = budgets_root
        self.agent_kwargs = agent_kwargs
        self.model = agent_kwargs.get("model", config.get("model"))
        self.reasoning_effort = agent_kwargs.get("reasoning_effort", config.get("model_reasoning_effort"))
        self.cache_mode = "off"

    def run(self, workspace, instructions, **kwargs):
        task_id = kwargs.get("task_id", "")
        harness_id = kwargs.get("harness_id", "")
        if not task_id or not harness_id:
            raise ValueError("Request budgets require task_id and harness_id")
        key = hashlib.sha256(json.dumps([task_id, harness_id]).encode()).hexdigest()
        with ModelRequestBudget(self.upstream_url, self.maximum, self.budgets_root / f"{key}.json") as budget:
            options = dict(self.agent_kwargs)
            options["extra_flags"] = tuple(options.get("extra_flags", ())) + (
                "-c", f"model_providers.{self.provider}.base_url={json.dumps(budget.base_url)}",
                "-c", f"model_providers.{self.provider}.supports_websockets=false",
            )
            agent = CodexAgent(self.config_path, **options)
            extra_env = dict(kwargs.pop("env", None) or {})
            # Loopback transport must not travel through the user's HTTP proxy.
            extra_env.update(NO_PROXY="localhost,127.0.0.1,127.0.0.2", no_proxy="localhost,127.0.0.1,127.0.0.2")
            trajectory = agent.run(workspace, instructions, env=extra_env, **kwargs)
            with budget.lock:
                accounting = {"type": "model_request_budget", "maximum": self.maximum,
                              "requests_sent": budget.state["requests_sent"],
                              "blocked_requests": budget.state["blocked_requests"], "task_id": task_id,
                              "harness_id": harness_id, "ledger": str(budget.ledger)}
            trajectory.events.append(accounting)
            write_json(Path(workspace) / ".rho/model_request_budget.json", accounting)
            return trajectory
