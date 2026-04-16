"""Shared pipeline execution helpers.

Used by both the CLI (``run_full.py``) and the web server so the two never
duplicate pipeline-state construction or output-serialization logic.

The base LangGraph graph is intentionally imported lazily inside
``run_pipeline`` — that keeps this module importable from tests without
requiring the heavy LangGraph / LLM dependency chain.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_initial_state(count: int, provider: str, test_mode: bool) -> Dict[str, Any]:
    """Build the full pipeline's initial state dict.

    Mirrors the seed values used by ``run_full.py`` so any consumer (CLI or
    web server) starts from the same baseline.
    """
    return {
        "count": count,
        "provider": provider,
        "test_mode": test_mode,
        "persons": [],
        "personas": [],
        "daily_drafts": {},
        "daily_events_map": {},
        "raw_calls": [],
        "raw_sms": [],
        "raw_push": [],
        "raw_emails": [],
        "persona_event_id_map": {},
        "generated_calls": [],
        "generated_sms": [],
        "generated_noti": [],
        "generated_gmail": [],
        "behavior_events": [],
        "behavior_events_map": {},
        "metadata": {"start_time": time.time(), "provider": provider},
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _run_pipeline_subprocess(
    *,
    count: int,
    provider: str,
    output_dir: Path | str,
    test_mode: bool,
    log,
    python_executable: str,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_executable,
        "run_full.py",
        "--count", str(count),
        "--provider", provider,
        "--output", str(output_root),
    ]
    if test_mode:
        cmd.append("--test")

    env = os.environ.copy()
    env["LIFEBENCH_FORCE_LOCAL_GRAPH"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    lines: List[str] = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        log(line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Pipeline subprocess failed with exit code {rc}")

    metadata = {}
    meta_path = output_root / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    personas = []
    for child in sorted(output_root.iterdir() if output_root.exists() else []):
        if not child.is_dir():
            continue
        counts = {}
        for type_name in ("call", "sms", "noti", "gmail"):
            p = child / f"behavior_events_{type_name}.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    counts[type_name] = len(json.load(f))
            else:
                counts[type_name] = 0
        total_path = child / "behavior_events.json"
        total = 0
        if total_path.exists():
            with open(total_path, "r", encoding="utf-8") as f:
                total = len(json.load(f))
        personas.append({
            "name": child.name,
            "event_count": total,
            "counts": counts,
            "output_dir": str(child),
        })

    return {
        "output_dir": str(output_root),
        "elapsed_seconds": metadata.get("elapsed_seconds"),
        "personas": personas,
        "metadata": metadata,
        "count": count,
        "provider": provider,
        "test_mode": test_mode,
    }


def save_pipeline_output(
    result: Dict[str, Any],
    output_dir: Path | str,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """Persist per-persona results and metadata to ``output_dir``.

    Returns a structured summary suitable for API consumers.
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    personas: List[Dict[str, Any]] = result.get("personas", []) or []
    behavior_events_map: Dict[str, List[Dict[str, Any]]] = dict(
        result.get("behavior_events_map") or {}
    )
    behavior_events_by_type: Dict[str, Dict[str, List[Dict[str, Any]]]] = dict(
        result.get("behavior_events_by_type") or {}
    )

    if not behavior_events_map:
        # Preserve run_full.py's fallback behaviour: when the graph failed to
        # split events per persona, dump the combined list under "all" but
        # continue emitting per-persona directories so callers can still iterate.
        behavior_events_map = {"all": result.get("behavior_events", [])}

    persona_summaries: List[Dict[str, Any]] = []
    for persona in personas:
        name = persona.get("name", "unknown")
        events = behavior_events_map.get(name, [])
        by_type = behavior_events_by_type.get(name, {})
        persona_dir = output_root / name
        persona_dir.mkdir(parents=True, exist_ok=True)

        _write_json(persona_dir / "behavior_events.json", events)
        for type_name, type_events in by_type.items():
            _write_json(persona_dir / f"behavior_events_{type_name}.json", type_events)
        _write_json(persona_dir / "persona.json", persona)

        counts = {t: len(v) for t, v in by_type.items()}
        persona_summaries.append({
            "name": name,
            "event_count": len(events),
            "counts": {
                "call":  counts.get("call", 0),
                "sms":   counts.get("sms", 0),
                "noti":  counts.get("noti", 0),
                "gmail": counts.get("gmail", 0),
            },
            "output_dir": str(persona_dir),
        })

    metadata = dict(result.get("metadata") or {})
    metadata["elapsed_seconds"] = elapsed_seconds
    _write_json(output_root / "metadata.json", metadata)

    return {
        "output_dir": str(output_root),
        "elapsed_seconds": elapsed_seconds,
        "personas": persona_summaries,
        "metadata": metadata,
    }


def run_pipeline(
    count: int,
    provider: str,
    output_dir: Path | str,
    test_mode: bool = False,
    graph: Optional[Any] = None,
    log: Optional[callable] = None,
    python_executable: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the full pipeline end-to-end and persist results.

    ``graph`` may be injected (tests, alternate variants). When omitted the
    base full graph is built lazily — this keeps imports cheap for consumers
    that only need the helper functions.
    """
    say = log or (lambda msg: None)

    if graph is None:
        if os.environ.get("LIFEBENCH_FORCE_LOCAL_GRAPH") != "1" and python_executable and Path(python_executable).resolve() != Path(sys.executable).resolve():
            return _run_pipeline_subprocess(
                count=count,
                provider=provider,
                output_dir=output_dir,
                test_mode=test_mode,
                log=say,
                python_executable=python_executable,
            )
        from pipeline.full_graph import build_full_graph
        graph = build_full_graph()

    initial_state = build_initial_state(count=count, provider=provider, test_mode=test_mode)

    say(f"Running pipeline (count={count}, provider={provider}, test_mode={test_mode})")
    t0 = time.time()
    result = graph.invoke(initial_state)
    elapsed = time.time() - t0
    say(f"Pipeline completed in {elapsed:.1f}s")

    summary = save_pipeline_output(result, output_dir, elapsed_seconds=elapsed)
    summary["count"] = count
    summary["provider"] = provider
    summary["test_mode"] = test_mode
    return summary
