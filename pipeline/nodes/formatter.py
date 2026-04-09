"""formatter node — merges all events, sorts by timestamp, assigns contextGroupId."""
import time
from datetime import datetime

from pipeline.state import PipelineState


def _get_date(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d")


def _assign_context_group_ids(events):
    date_to_group: dict[str, int] = {}
    counter = 1
    for event in events:
        date = _get_date(event["timestamp"])
        if date not in date_to_group:
            date_to_group[date] = counter
            counter += 1
        event["contextGroupId"] = date_to_group[date]
    return events


def merge_and_sort(state: PipelineState) -> PipelineState:
    """Merge generated_calls + generated_sms + generated_noti, sort by timestamp."""
    all_events = (
        list(state.get("generated_calls", []))
        + list(state.get("generated_sms", []))
        + list(state.get("generated_noti", []))
    )
    all_events.sort(key=lambda e: e["timestamp"])
    all_events = _assign_context_group_ids(all_events)

    metadata = state.get("metadata", {})
    metadata["finish_time"] = time.time()
    metadata["total_events"] = len(all_events)
    metadata["event_counts"] = {
        "calls": len(state.get("generated_calls", [])),
        "sms": len(state.get("generated_sms", [])),
        "notifications": len(state.get("generated_noti", [])),
    }

    return {**state, "behavior_events": all_events, "metadata": metadata}
