from __future__ import annotations

import textwrap


def render_dispatcher() -> str:
    return textwrap.dedent(
        '''\
        #!/usr/bin/env python3
        from __future__ import annotations

        import argparse
        import json
        import socket
        import struct
        import sys
        from pathlib import Path


        def _handle_path() -> Path:
            return Path(__file__).resolve().parents[1] / ".gaia2" / "handle.json"


        def _socket_path() -> str:
            handle_path = _handle_path()
            try:
                handle = json.loads(handle_path.read_text(encoding="utf-8"))
            except OSError as exc:
                raise SystemExit(f"cannot read GAIA-2 handle at {handle_path}: {exc}") from exc
            try:
                return str(handle["socket_path"])
            except KeyError as exc:
                raise SystemExit(f"GAIA-2 handle at {handle_path} is missing socket_path") from exc


        def _rpc(payload: dict) -> dict:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect(_socket_path())
                sock.sendall(struct.pack(">I", len(raw)) + raw)
                header = _recv_exact(sock, 4)
                size = struct.unpack(">I", header)[0]
                body = _recv_exact(sock, size)
            response = json.loads(body.decode("utf-8"))
            if not response.get("ok", False):
                message = response.get("error") or response
                raise SystemExit(str(message))
            return response


        def _recv_exact(sock: socket.socket, size: int) -> bytes:
            chunks = []
            remaining = size
            while remaining:
                chunk = sock.recv(remaining)
                if not chunk:
                    raise SystemExit("sidecar closed connection early")
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)


        def _print(payload: dict) -> None:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


        def main(argv: list[str] | None = None) -> int:
            parser = argparse.ArgumentParser(description="Call GAIA-2 ARE tools through the task sidecar.")
            sub = parser.add_subparsers(dest="command", required=True)
            sub.add_parser("list", help="List available ARE tools")

            schema = sub.add_parser("schema", help="Show one tool schema")
            schema.add_argument("app")
            schema.add_argument("function")

            call = sub.add_parser("call", help="Call one ARE tool with a JSON object")
            call.add_argument("app")
            call.add_argument("function")
            call.add_argument("--json", required=True, help="JSON object containing named args")

            state = sub.add_parser("state", help="Dump mirrored app state")
            state.add_argument("app", nargs="?")

            sub.add_parser("poll", help="Poll notifications and refresh state")

            wait = sub.add_parser("wait", help="Wait for a notification")
            wait.add_argument("--timeout-seconds", type=float, default=30.0)

            args = parser.parse_args(argv)
            if args.command == "list":
                _print(_rpc({"method": "list_tools"}))
            elif args.command == "schema":
                _print(_rpc({"method": "schema", "app": args.app, "function": args.function}))
            elif args.command == "call":
                try:
                    call_args = json.loads(args.json)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"--json must be a JSON object: {exc}") from exc
                if not isinstance(call_args, dict):
                    raise SystemExit("--json must be a JSON object")
                _print(_rpc({"method": "call_tool", "app": args.app, "function": args.function, "args": call_args}))
            elif args.command == "state":
                _print(_rpc({"method": "dump_state", "app": args.app}))
            elif args.command == "poll":
                _print(_rpc({"method": "poll_notifications"}))
            elif args.command == "wait":
                timeout = min(max(args.timeout_seconds, 0.0), 60.0)
                _print(_rpc({"method": "wait_for_notification", "timeout_seconds": timeout}))
            return 0


        if __name__ == "__main__":
            raise SystemExit(main(sys.argv[1:]))
        '''
    )
