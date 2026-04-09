"""call_gen node — rule-based conversion of raw calls to BehaviorEventEntity."""
import json
from datetime import datetime
from typing import Any

from pipeline.state import PipelineState


def _datetime_to_unix_ms(dt_str: str) -> int:
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp() * 1000)


def _convert_call(call):  # type: (dict) -> dict
    timestamp = _datetime_to_unix_ms(call["datetime"])

    if call.get("datetime_end"):
        ts_end = _datetime_to_unix_ms(call["datetime_end"])
        duration = (ts_end - timestamp) // 1000
    else:
        duration = 0

    call_result = call.get("call_result", "")
    direction = call.get("direction", 0)

    if "missed" in call_result.lower():
        call_type = 2  # Missed
    elif direction == 0:
        call_type = 0  # Outgoing
    else:
        call_type = 1  # Incoming

    payload = {
        "number": call.get("phoneNumber", ""),
        "name": call.get("contactName", ""),
        "date": timestamp,
        "type": call_type,
        "geocodedLocation": "",
        "duration": duration,
        "missedReason": "" if call_type != 2 else "No answer",
        "transcriptDialog": "",
    }

    return {
        "identifier": f"call_{call['event_id']}_{timestamp}",
        "timestamp": timestamp,
        "event_source": "CALL_LOG",
        "payload": json.dumps(payload, ensure_ascii=False),
        "contextGroupId": None,
    }


def generate_calls(state: PipelineState) -> PipelineState:
    """Convert raw_calls → generated_calls."""
    results = []
    for call in state.get("raw_calls", []):
        try:
            results.append(_convert_call(call))
        except Exception as e:
            print(f"[call_gen] skipping {call.get('event_id')}: {e}")

    return {**state, "generated_calls": results}
