"""In-process pipeline-run tracker.

The web app drops a :class:`RunManager` instance alongside the HTTP handler.
Each ``start`` call spins up a daemon thread that executes the runner
function (defaults to :func:`server.pipeline_runner.run_pipeline`) and
stores status + logs + summary + error in memory.

This module deliberately has no FastAPI / framework dependencies so it can
be unit-tested with stdlib ``unittest``.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

RunnerFn = Callable[..., Dict[str, Any]]


def _default_runner_fn(**kwargs):
    # Imported lazily so importing ``server.run_manager`` never forces the
    # heavy LangGraph / LLM dependency graph to load.
    from server.pipeline_runner import run_pipeline
    return run_pipeline(**kwargs)


class RunManager:
    def __init__(
        self,
        default_output_dir: Path | str,
        runner_fn: Optional[RunnerFn] = None,
        history_limit: int = 20,
        runner_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._default_output_dir = Path(default_output_dir)
        self._runner_fn: RunnerFn = runner_fn or _default_runner_fn
        self._history_limit = history_limit
        self._runner_kwargs = dict(runner_kwargs or {})
        self._lock = threading.Lock()
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []

    # ── public API ───────────────────────────────────────────────────────

    def start(
        self,
        *,
        count: int,
        provider: str,
        test_mode: bool,
        output_dir: Optional[Path | str] = None,
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        out_dir = Path(output_dir) if output_dir else self._default_output_dir
        record = {
            "run_id": run_id,
            "status": "running",
            "count": count,
            "provider": provider,
            "test_mode": test_mode,
            "output_dir": str(out_dir),
            "started_at": time.time(),
            "finished_at": None,
            "logs": [],
            "summary": None,
            "error": None,
            "previous_output_dir": None,
        }
        with self._lock:
            self._runs[run_id] = record
            self._order.append(run_id)
            self._trim_history_locked()

        thread = threading.Thread(
            target=self._execute,
            args=(run_id, count, provider, test_mode, out_dir),
            name=f"run-{run_id}",
            daemon=True,
        )
        thread.start()
        return run_id

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._runs.get(run_id)
            return _snapshot(record) if record else None

    def attach_metadata(self, run_id: str, **fields: Any) -> None:
        with self._lock:
            record = self._runs.get(run_id)
            if record is not None:
                record.update(fields)

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [_snapshot(self._runs[i]) for i in self._order if i in self._runs]

    # ── internals ────────────────────────────────────────────────────────

    def _execute(
        self,
        run_id: str,
        count: int,
        provider: str,
        test_mode: bool,
        output_dir: Path,
    ) -> None:
        def log(msg: str) -> None:
            with self._lock:
                record = self._runs.get(run_id)
                if record is not None:
                    record["logs"].append(str(msg))

        try:
            summary = self._runner_fn(
                count=count,
                provider=provider,
                output_dir=output_dir,
                test_mode=test_mode,
                log=log,
                **self._runner_kwargs,
            )
            with self._lock:
                record = self._runs.get(run_id)
                if record is not None:
                    record["summary"] = summary
                    record["status"] = "success"
                    record["finished_at"] = time.time()
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            with self._lock:
                record = self._runs.get(run_id)
                if record is not None:
                    record["status"] = "failed"
                    record["error"] = f"{exc}\n{tb}"
                    record["finished_at"] = time.time()

    def _trim_history_locked(self) -> None:
        while len(self._order) > self._history_limit:
            oldest = self._order.pop(0)
            self._runs.pop(oldest, None)


def _snapshot(record: Dict[str, Any]) -> Dict[str, Any]:
    copy = dict(record)
    copy["logs"] = list(record.get("logs", []))
    return copy
