"""Viewer payload builder.

Reads per-persona output folders produced by ``save_pipeline_output`` and
shapes them into the JSON list that ``explorer.html`` expects
(``{persona, stats, samples}`` per entry).

Only the filesystem is inspected — no pipeline state is required, so the
data loader is safe to call from the web server before/without any run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

SAMPLE_LIMIT = 10


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _persona_entry(persona_dir: Path) -> Dict[str, Any] | None:
    persona = _read_json(persona_dir / "persona.json", None)
    if not persona:
        return None

    calls = _read_json(persona_dir / "behavior_events_call.json", [])
    sms   = _read_json(persona_dir / "behavior_events_sms.json", [])
    noti  = _read_json(persona_dir / "behavior_events_noti.json", [])
    gmail = _read_json(persona_dir / "behavior_events_gmail.json", [])

    total = _read_json(persona_dir / "behavior_events.json", None)
    if total is None:
        total_count = len(calls) + len(sms) + len(noti) + len(gmail)
    else:
        total_count = len(total)

    transcript_count = sum(
        1 for e in calls
        if (e.get("payload") or {}).get("transcriptDialog")
    )

    return {
        "persona": persona,
        "stats": {
            "total": total_count,
            "calls": len(calls),
            "sms": len(sms),
            "noti": len(noti),
            "gmail": len(gmail),
            "transcript_count": transcript_count,
        },
        "samples": {
            "calls": calls[:SAMPLE_LIMIT],
            "sms":   sms[:SAMPLE_LIMIT],
            "noti":  noti[:SAMPLE_LIMIT],
            "gmail": gmail[:SAMPLE_LIMIT],
        },
    }


def load_viewer_payload(output_dir: Path | str) -> List[Dict[str, Any]]:
    """Return viewer-shaped entries for every persona under ``output_dir``."""
    root = Path(output_dir)
    if not root.exists() or not root.is_dir():
        return []

    entries: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        entry = _persona_entry(child)
        if entry is not None:
            entries.append(entry)
    return entries


def load_metadata(output_dir: Path | str) -> Dict[str, Any]:
    """Return the metadata.json payload for ``output_dir`` (empty dict if none)."""
    meta = _read_json(Path(output_dir) / "metadata.json", {})
    return meta if isinstance(meta, dict) else {}
