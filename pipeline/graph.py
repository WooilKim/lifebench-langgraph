"""Base LangGraph pipeline graph."""
from langgraph.graph import StateGraph, END

from pipeline.state import PipelineState
from pipeline.nodes import (
    load_data,
    generate_calls,
    generate_sms,
    generate_notifications,
    merge_and_sort,
)


def build_base_graph() -> StateGraph:
    """Build and return the compiled base graph."""
    g = StateGraph(PipelineState)

    g.add_node("loader", load_data)
    g.add_node("call_gen", generate_calls)
    g.add_node("sms_gen", generate_sms)
    g.add_node("noti_gen", generate_notifications)
    g.add_node("formatter", merge_and_sort)

    g.set_entry_point("loader")
    g.add_edge("loader", "call_gen")
    g.add_edge("call_gen", "sms_gen")
    g.add_edge("sms_gen", "noti_gen")
    g.add_edge("noti_gen", "formatter")
    g.add_edge("formatter", END)

    return g.compile()


def export_graph_structure() -> dict:
    """Return a JSON-serialisable description of the base graph structure."""
    return {
        "name": "base",
        "nodes": ["loader", "call_gen", "sms_gen", "noti_gen", "formatter"],
        "edges": [
            {"from": "loader",    "to": "call_gen"},
            {"from": "call_gen",  "to": "sms_gen"},
            {"from": "sms_gen",   "to": "noti_gen"},
            {"from": "noti_gen",  "to": "formatter"},
            {"from": "formatter", "to": "END"},
        ],
        "replaced_nodes": [],
    }
