"""Stdlib-only HTTP server for the LifeBench pipeline viewer.

Endpoints:
    GET  /api/health          → readiness info
    GET  /api/data            → viewer payload for current output dir
    POST /api/run             → start a pipeline run (JSON body)
    GET  /api/run/{run_id}    → status + logs + summary + error for one run
    GET  /api/runs            → recent run list
    GET  /                    → index.html
    GET  /{anything else}     → served from the repository root (explorer.html, etc.)

FastAPI is not used on purpose: the feature only needs a handful of
endpoints and avoiding an extra dependency keeps the setup story simple
(see README). The handler is framework-agnostic enough that swapping it
for FastAPI later would be mechanical.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse


def _safe_output_dir(path_value: str | None, default_root: Path) -> Path:
    if not path_value:
        return default_root
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = (default_root.parent / candidate).resolve()
    else:
        candidate = candidate.resolve()
    root = default_root.parent.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output must stay under {root}") from exc
    return candidate


class _FastThreadingHTTPServer(ThreadingHTTPServer):
    """HTTP server that skips the reverse-DNS lookup in ``server_bind``.

    Python's ``HTTPServer.server_bind`` calls ``socket.getfqdn(host)`` to
    populate ``server_name``. On macOS this can take 30+ seconds against
    127.0.0.1, which kills test suites and makes dev startup painful.
    The value is cosmetic for our needs, so we skip it.
    """

    def server_bind(self) -> None:  # type: ignore[override]
        import socketserver
        socketserver.TCPServer.server_bind(self)  # type: ignore[attr-defined]
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port

from server.data_loader import load_metadata, load_viewer_payload
from server.run_manager import RunManager

DEFAULT_OUTPUT_DIR = "output_web"
REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_STATIC_FILES = {
    "index.html",
    "explorer.html",
    "pipeline_data.json",
}


def _detect_python_executable() -> str:
    candidates = [
        os.environ.get("LIFEBENCH_PYTHON"),
        sys.executable,
        shutil.which("python3.12"),
        shutil.which("python3.11"),
        shutil.which("python3.10"),
        shutil.which("python3"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            probe = subprocess.run(
                [candidate, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            major, minor = map(int, probe.stdout.strip().split("."))
            if (major, minor) >= (3, 10):
                return candidate
        except Exception:
            continue
    return sys.executable


def _json_bytes(payload: Any, status: int = 200) -> Tuple[int, bytes, str]:
    return status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8"


class AppState:
    """Bundled state shared with the request handler."""

    def __init__(
        self,
        output_dir: Path,
        run_manager: RunManager,
        static_root: Path,
        python_executable: str,
    ) -> None:
        self.output_dir = output_dir
        self.run_manager = run_manager
        self.static_root = static_root
        self.active_output_dir = output_dir
        self.python_executable = python_executable


def _make_handler(state: AppState):
    run_path_re = re.compile(r"^/api/run/(?P<run_id>[A-Za-z0-9_-]+)$")

    class Handler(BaseHTTPRequestHandler):
        server_version = "LifeBenchViewer/1.0"

        # ── logging ──────────────────────────────────────────────────────
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            # Silence stderr access log during tests; still override so
            # users can turn it on with env var if needed.
            if os.environ.get("LIFEBENCH_HTTP_LOG") == "1":
                super().log_message(format, *args)

        # ── shared helpers ───────────────────────────────────────────────
        def _send(self, status: int, body: bytes, content_type: str,
                  extra_headers: Optional[Dict[str, str]] = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_json(self, payload: Any, status: int = 200) -> None:
            status_, body, ct = _json_bytes(payload, status=status)
            self._send(status_, body, ct)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        # ── routing ──────────────────────────────────────────────────────
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/health":
                return self._handle_health()
            if path == "/api/data":
                return self._handle_data()
            if path == "/api/runs":
                return self._handle_list_runs()
            m = run_path_re.match(path)
            if m:
                return self._handle_get_run(m.group("run_id"))
            if path.startswith("/api/"):
                return self._send_json({"error": "not found"}, status=404)
            return self._serve_static(path)

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/run":
                return self._handle_start_run()
            return self._send_json({"error": "not found"}, status=404)

        # ── handlers ─────────────────────────────────────────────────────
        def _handle_health(self) -> None:
            self._send_json({
                "status": "ok",
                "output_dir": str(state.active_output_dir),
                "default_output_dir": str(state.output_dir),
                "output_exists": state.active_output_dir.exists(),
            })

        def _handle_data(self) -> None:
            entries = load_viewer_payload(state.active_output_dir)
            metadata = load_metadata(state.active_output_dir)
            self._send_json({
                "personas": entries,
                "metadata": metadata,
                "output_dir": str(state.active_output_dir),
                "default_output_dir": str(state.output_dir),
            })

        def _handle_start_run(self) -> None:
            try:
                body = self._read_json_body()
            except json.JSONDecodeError:
                return self._send_json({"error": "invalid json"}, status=400)

            try:
                count = int(body.get("count", 1))
            except (TypeError, ValueError):
                return self._send_json({"error": "count must be int"}, status=400)
            if count < 1:
                return self._send_json({"error": "count must be >= 1"}, status=400)
            provider = str(body.get("provider", "claude"))
            if provider not in ("claude", "gpt", "glm"):
                return self._send_json({"error": "invalid provider"}, status=400)
            test_mode = bool(body.get("test_mode", True))
            output_override = body.get("output")
            try:
                output_dir = _safe_output_dir(output_override, state.output_dir)
            except ValueError as e:
                return self._send_json({"error": str(e)}, status=400)
            previous_output_dir = state.active_output_dir
            run_id = state.run_manager.start(
                count=count,
                provider=provider,
                test_mode=test_mode,
                output_dir=output_dir,
            )
            state.run_manager.attach_metadata(run_id, previous_output_dir=str(previous_output_dir))
            state.active_output_dir = output_dir
            self._send_json({"run_id": run_id, "status": "running", "output_dir": str(output_dir)}, status=202)

        def _handle_get_run(self, run_id: str) -> None:
            record = state.run_manager.get(run_id)
            if record is None:
                return self._send_json({"error": "run not found"}, status=404)
            if record.get("status") == "failed" and record.get("previous_output_dir"):
                state.active_output_dir = Path(record["previous_output_dir"])
            self._send_json(record)

        def _handle_list_runs(self) -> None:
            runs = state.run_manager.list()
            self._send_json({"runs": runs})

        # ── static ───────────────────────────────────────────────────────
        def _serve_static(self, path: str) -> None:
            rel = "index.html" if path in ("", "/") else path.lstrip("/")
            if rel not in ALLOWED_STATIC_FILES:
                return self._send_json({"error": "not found"}, status=404)
            full = (state.static_root / rel).resolve()
            try:
                full.relative_to(state.static_root.resolve())
            except ValueError:
                return self._send_json({"error": "forbidden"}, status=403)
            if not full.exists() or not full.is_file():
                return self._send_json({"error": "not found"}, status=404)
            ctype, _ = mimetypes.guess_type(str(full))
            ctype = ctype or "application/octet-stream"
            try:
                data = full.read_bytes()
            except OSError:
                return self._send_json({"error": "read error"}, status=500)
            self._send(200, data, ctype)

    return Handler


def create_app(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    static_root: Path | str = REPO_ROOT,
    run_manager: Optional[RunManager] = None,
) -> Tuple[AppState, type]:
    """Return ``(state, handler_cls)`` for embedding in a server."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    python_executable = _detect_python_executable()
    manager = run_manager or RunManager(default_output_dir=out, runner_kwargs={"python_executable": python_executable})
    state = AppState(
        output_dir=out,
        run_manager=manager,
        static_root=Path(static_root),
        python_executable=python_executable,
    )
    return state, _make_handler(state)


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    static_root: Path | str = REPO_ROOT,
) -> None:
    """Blocking run of the HTTP server (used by ``python -m server.app``)."""
    state, handler_cls = create_app(output_dir=output_dir, static_root=static_root)
    httpd = _FastThreadingHTTPServer((host, port), handler_cls)
    print(f"LifeBench web server → http://{host}:{port}/  (output: {state.output_dir})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def start_background_server(
    host: str = "127.0.0.1",
    port: int = 0,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    static_root: Path | str = REPO_ROOT,
    run_manager: Optional[RunManager] = None,
) -> Tuple["_FastThreadingHTTPServer", threading.Thread, int, AppState]:
    """Start the server on a background thread (used by tests and CLIs).

    When ``port`` is 0 the OS picks a free port and the effective port is
    returned, which makes parallel test execution painless.
    """
    state, handler_cls = create_app(
        output_dir=output_dir,
        static_root=static_root,
        run_manager=run_manager,
    )
    if port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            port = s.getsockname()[1]
    httpd = _FastThreadingHTTPServer((host, port), handler_cls)
    thread = threading.Thread(
        target=httpd.serve_forever, name="lifebench-http", daemon=True
    )
    thread.start()
    return httpd, thread, port, state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LifeBench viewer + pipeline web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory served by /api/data and used by /api/run")
    args = parser.parse_args()
    serve(host=args.host, port=args.port, output_dir=args.output)
