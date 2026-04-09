"""Full LangGraph pipeline: person_gen → persona_gen → draft_gen → simulator → phone_data_gen
→ call_gen → sms_gen → noti_gen → multi_formatter → END
"""
import time
from datetime import datetime

from langgraph.graph import StateGraph, END

from pipeline.full_state import FullPipelineState
from pipeline.nodes.person_gen import generate_persons
from pipeline.nodes.persona_gen import generate_personas
from pipeline.nodes.draft_gen import generate_drafts
from pipeline.nodes.simulator import simulate_daily_events
from pipeline.nodes.phone_data_gen import generate_phone_data
from pipeline.nodes.call_gen import generate_calls
from pipeline.nodes.sms_gen import generate_sms
from pipeline.nodes.noti_gen import generate_notifications
from pipeline.nodes.gmail_gen import generate_gmail


# ── multi_formatter ────────────────────────────────────────────────────────────

def _get_date(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d")


def _assign_context_group_ids(events: list) -> list:
    date_to_group: dict = {}
    counter = 1
    for event in events:
        date = _get_date(event["timestamp"])
        if date not in date_to_group:
            date_to_group[date] = counter
            counter += 1
        event["contextGroupId"] = date_to_group[date]
    return events


def multi_formatter(state: FullPipelineState) -> FullPipelineState:
    """Merge + sort all events, then split into per-persona behavior_events_map."""
    all_events = (
        list(state.get("generated_calls", []))
        + list(state.get("generated_sms", []))
        + list(state.get("generated_noti", []))
        + list(state.get("generated_gmail", []))
    )
    all_events.sort(key=lambda e: e["timestamp"])
    all_events = _assign_context_group_ids(all_events)

    # Split per persona using pre-computed identifier sets
    persona_event_id_map: dict = state.get("persona_event_id_map", {})
    personas = state.get("personas", [])
    behavior_events_map: dict = {}
    behavior_events_by_type: dict = {}

    for persona in personas:
        name = persona.get("name", "")
        id_sets = persona_event_id_map.get(name, {})
        call_ids  = id_sets.get("call_ids", set())
        sms_ids   = id_sets.get("sms_ids", set())
        noti_ids  = id_sets.get("noti_ids", set())
        gmail_ids = id_sets.get("gmail_ids", set())
        all_ids = call_ids | sms_ids | noti_ids | gmail_ids

        persona_events = [e for e in all_events if e["identifier"] in all_ids]
        behavior_events_map[name] = persona_events

        # Per-type split
        behavior_events_by_type[name] = {
            "call":  [e for e in persona_events if e["identifier"] in call_ids],
            "sms":   [e for e in persona_events if e["identifier"] in sms_ids],
            "noti":  [e for e in persona_events if e["identifier"] in noti_ids],
            "gmail": [e for e in persona_events if e["identifier"] in gmail_ids],
        }
        print(f"[multi_formatter] {name}: {len(persona_events)} events "
              f"(call={len(behavior_events_by_type[name]['call'])}, "
              f"sms={len(behavior_events_by_type[name]['sms'])}, "
              f"noti={len(behavior_events_by_type[name]['noti'])}, "
              f"gmail={len(behavior_events_by_type[name]['gmail'])})")

    metadata = state.get("metadata", {})
    metadata["finish_time"] = time.time()
    metadata["total_events"] = len(all_events)
    metadata["event_counts"] = {
        "calls": len(state.get("generated_calls", [])),
        "sms": len(state.get("generated_sms", [])),
        "notifications": len(state.get("generated_noti", [])),
    }
    metadata["personas"] = [p.get("name") for p in personas]

    return {
        **state,
        "behavior_events": all_events,
        "behavior_events_map": behavior_events_map,
        "behavior_events_by_type": behavior_events_by_type,
        "metadata": metadata,
    }


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_full_graph():
    """Build and return the compiled full pipeline graph."""
    g = StateGraph(FullPipelineState)

    g.add_node("person_gen",     generate_persons)
    g.add_node("persona_gen",    generate_personas)
    g.add_node("draft_gen",      generate_drafts)
    g.add_node("simulator",      simulate_daily_events)
    g.add_node("phone_data_gen", generate_phone_data)
    g.add_node("call_gen",       generate_calls)
    g.add_node("sms_gen",        generate_sms)
    g.add_node("noti_gen",       generate_notifications)
    g.add_node("gmail_gen",      generate_gmail)
    g.add_node("formatter",      multi_formatter)

    g.set_entry_point("person_gen")
    g.add_edge("person_gen",     "persona_gen")
    g.add_edge("persona_gen",    "draft_gen")
    g.add_edge("draft_gen",      "simulator")
    g.add_edge("simulator",      "phone_data_gen")
    g.add_edge("phone_data_gen", "call_gen")
    g.add_edge("call_gen",       "sms_gen")
    g.add_edge("sms_gen",        "noti_gen")
    g.add_edge("noti_gen",       "gmail_gen")
    g.add_edge("gmail_gen",      "formatter")
    g.add_edge("formatter",      END)

    return g.compile()


def export_full_graph_structure() -> dict:
    return {
        "name": "full",
        "nodes": [
            "persona_gen", "draft_gen", "simulator", "phone_data_gen",
            "call_gen", "sms_gen", "noti_gen", "formatter",
        ],
        "edges": [
            {"from": "persona_gen",    "to": "draft_gen"},
            {"from": "draft_gen",      "to": "simulator"},
            {"from": "simulator",      "to": "phone_data_gen"},
            {"from": "phone_data_gen", "to": "call_gen"},
            {"from": "call_gen",       "to": "sms_gen"},
            {"from": "sms_gen",        "to": "noti_gen"},
            {"from": "noti_gen",       "to": "formatter"},
            {"from": "formatter",      "to": "END"},
        ],
        "replaced_nodes": ["loader"],
        "new_nodes": ["persona_gen", "draft_gen", "simulator", "phone_data_gen"],
    }
