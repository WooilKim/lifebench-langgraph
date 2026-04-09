"""loader node — reads raw JSON files from the user's data directory."""
import json
import time
from pathlib import Path

from pipeline.state import PipelineState


def load_data(state: PipelineState) -> PipelineState:
    """Load persona, daily_events, and phone_data JSON files into state."""
    data_dir = Path(state["data_dir"])
    phone_dir = data_dir / "phone_data"

    def _load(path: Path):
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    persona = _load(data_dir / "persona.json")
    daily_events = _load(data_dir / "daily_event.json")
    raw_calls = _load(phone_dir / "call.json")
    raw_sms = _load(phone_dir / "sms.json")
    raw_push = _load(phone_dir / "push.json")

    metadata = state.get("metadata", {})
    metadata["load_time"] = time.time()
    metadata["user_id"] = state["user_id"]
    metadata["counts"] = {
        "raw_calls": len(raw_calls),
        "raw_sms": len(raw_sms),
        "raw_push": len(raw_push),
        "daily_events": len(daily_events) if isinstance(daily_events, list) else 1,
    }

    return {
        **state,
        "persona": persona,
        "daily_events": daily_events if isinstance(daily_events, list) else [daily_events],
        "raw_calls": raw_calls,
        "raw_sms": raw_sms,
        "raw_push": raw_push,
        "metadata": metadata,
    }
