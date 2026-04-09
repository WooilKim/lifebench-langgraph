"""llm_enriched variant — replaces call_gen and sms_gen with LLM-augmented versions."""
import json
from datetime import datetime

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from pipeline.state import PipelineState
from pipeline.nodes import load_data, generate_notifications, merge_and_sort
from pipeline.nodes.call_gen import _datetime_to_unix_ms, _convert_call
from pipeline.nodes.sms_gen import _convert_sms
from llm.client import get_llm_client


def _enrich_transcript(llm, call: dict) -> str:
    """Ask the LLM to produce a short call transcript summary."""
    contact = call.get("contactName", "Unknown")
    duration_s = 0
    if call.get("datetime_end") and call.get("datetime"):
        ts = _datetime_to_unix_ms(call["datetime"])
        ts_end = _datetime_to_unix_ms(call["datetime_end"])
        duration_s = (ts_end - ts) // 1000
    direction = "outgoing" if call.get("direction", 0) == 0 else "incoming"
    date_str = call.get("datetime", "")[:10]
    call_event_id = call.get("event_id", "")
    prompt = f"""다음 이벤트 정보와 생성 지침에 따라 통화 데이터를 생성하세요.

## 입력 정보
- 날짜: {date_str}
- 이벤트 정보: call with {contact} ({direction}, {duration_s}s)
- 생성 지침: transcriptDialog 요약 생성
- 이벤트 ID: {call_event_id}

## 생성 요구사항
1. 이벤트와 연관된 통화 생성
2. 통화 방향(발신/수신), 시간, 결과 포함
3. 참여자 관계 반영 (가족/동료/친구)
4. 현실적인 통화 시간
통화 요약(1~2문장)만 출력하세요."""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"[llm_enriched] transcript error: {e}")
        return ""


def _enrich_sms_body(llm, sms: dict) -> str:
    """Ask the LLM to rewrite the SMS body more naturally."""
    original = sms.get("message_content", "")
    if not original:
        return original
    date_str = sms.get("datetime", "")[:10]
    sms_event_id = sms.get("event_id", "")
    prompt = f"""다음 이벤트 정보와 생성 지침에 따라 문자(SMS) 데이터를 생성하세요.

## 입력 정보
- 날짜: {date_str}
- 이벤트 정보: SMS content: {original}
- 생성 지침: 더 자연스럽고 구어체적으로 재작성
- 이벤트 ID: {sms_event_id}

## 생성 요구사항
1. 이벤트와 자연스럽게 연결되는 문자 대화 생성
2. 참여자의 관계와 성격 반영
3. 시간적 논리 유지
4. 현실적인 문자 내용과 길이
재작성된 메시지만 출력하세요."""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"[llm_enriched] sms enrich error: {e}")
        return original


def llm_enriched_call_gen(state: PipelineState) -> PipelineState:
    """LLM-augmented call_gen: adds transcriptDialog summaries."""
    llm = get_llm_client()
    results = []
    for call in state.get("raw_calls", []):
        try:
            event = _convert_call(call)
            payload = json.loads(event["payload"])
            payload["transcriptDialog"] = _enrich_transcript(llm, call)
            event["payload"] = json.dumps(payload, ensure_ascii=False)
            results.append(event)
        except Exception as e:
            print(f"[llm_enriched call_gen] skipping {call.get('event_id')}: {e}")

    return {**state, "generated_calls": results}


def llm_enriched_sms_gen(state: PipelineState) -> PipelineState:
    """LLM-augmented sms_gen: rewrites SMS body to sound more natural."""
    llm = get_llm_client()
    thread_id_map = {}
    results = []
    for sms in state.get("raw_sms", []):
        try:
            # Enrich the message body before conversion
            enriched = dict(sms)
            enriched["message_content"] = _enrich_sms_body(llm, sms)
            results.append(_convert_sms(enriched, thread_id_map))
        except Exception as e:
            print(f"[llm_enriched sms_gen] skipping {sms.get('event_id')}: {e}")

    return {**state, "generated_sms": results}


def build_llm_enriched_graph():
    """Build the LLM-enriched variant graph (replaces call_gen + sms_gen)."""
    g = StateGraph(PipelineState)

    g.add_node("loader", load_data)
    g.add_node("call_gen", llm_enriched_call_gen)
    g.add_node("sms_gen", llm_enriched_sms_gen)
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
    return {
        "name": "llm_enriched",
        "nodes": ["loader", "call_gen", "sms_gen", "noti_gen", "formatter"],
        "edges": [
            {"from": "loader",    "to": "call_gen"},
            {"from": "call_gen",  "to": "sms_gen"},
            {"from": "sms_gen",   "to": "noti_gen"},
            {"from": "noti_gen",  "to": "formatter"},
            {"from": "formatter", "to": "END"},
        ],
        "replaced_nodes": ["call_gen", "sms_gen"],
    }
