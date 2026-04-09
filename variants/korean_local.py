"""korean_local variant — localises notifications and names to Korean."""
import json
import re

from langgraph.graph import StateGraph, END

from pipeline.state import PipelineState
from pipeline.nodes import load_data, generate_calls, generate_sms, merge_and_sort
from pipeline.nodes.noti_gen import _convert_notification, _get_package_name, _get_category
from pipeline.nodes.noti_gen import _datetime_to_unix_ms

# ── App name translations ──────────────────────────────────────────────────────
APP_NAME_KO = {
    "NetEase Cloud Music": "넷이즈 클라우드 뮤직",
    "Keep": "킵",
    "WeChat": "위챗",
    "Alipay": "알리페이",
    "Taobao": "타오바오",
    "JD": "징둥",
    "Meituan": "메이퇀",
    "Didi": "디디",
    "Douyin": "더우인",
    "Xiaohongshu": "샤오홍수",
    "Bilibili": "비리비리",
    "iQIYI": "아이치이",
    "Tencent Video": "텐센트 비디오",
    "QQ Music": "QQ뮤직",
    "Baidu Map": "바이두 지도",
    "Amap": "어거지 지도",
    "12306": "12306 철도",
    "China Mobile": "차이나 모바일",
    "China Unicom": "차이나 유니콤",
    "Bank of China": "중국은행",
    "ICBC": "공상은행",
    "CCB": "건설은행",
    "ABC": "농업은행",
    "Pinduoduo": "핀둬둬",
    "Ele.me": "어러머",
    "Ctrip": "씨트립",
    "Qunar": "취날",
    "Fliggy": "페이주",
    "Zhihu": "즈후",
    "Toutiao": "터우탸오",
    "Weibo": "웨이보",
}

# ── Simple keyword translations for notification text ─────────────────────────
_KEYWORD_MAP = [
    ("Daily Recommendations", "오늘의 추천"),
    ("Morning Run", "아침 달리기"),
    ("Exercise Plan", "운동 계획"),
    ("Running Plan", "러닝 플랜"),
    ("playlist", "플레이리스트"),
    ("recommended", "추천"),
    ("Click to", "눌러서"),
    ("Tap to", "눌러서"),
    ("Today's", "오늘의"),
    ("reminder", "알림"),
    ("Reminder", "알림"),
    ("Update", "업데이트"),
]


def _localise_text(text: str, app_name_ko: str) -> str:
    """Apply keyword substitutions and replace app name."""
    for en, ko in _KEYWORD_MAP:
        text = text.replace(en, ko)
    return text


def _convert_notification_ko(noti: dict) -> dict:
    """Convert notification entry with Korean app names and text."""
    event = _convert_notification(noti)
    payload = json.loads(event["payload"])

    source = noti.get("source", "")
    ko_name = APP_NAME_KO.get(source, source)

    payload["appName"] = ko_name
    payload["title"] = _localise_text(noti.get("title", ""), ko_name)
    payload["text"] = _localise_text(noti.get("content", ""), ko_name)

    event["payload"] = json.dumps(payload, ensure_ascii=False)
    return event


def korean_local_noti_gen(state: PipelineState) -> PipelineState:
    """Korean-localised notification generation."""
    results = []
    for noti in state.get("raw_push", []):
        try:
            results.append(_convert_notification_ko(noti))
        except Exception as e:
            print(f"[korean_local noti_gen] skipping {noti.get('event_id')}: {e}")

    return {**state, "generated_noti": results}


def build_korean_local_graph():
    """Build the Korean-localised variant (replaces noti_gen)."""
    g = StateGraph(PipelineState)

    g.add_node("loader", load_data)
    g.add_node("call_gen", generate_calls)
    g.add_node("sms_gen", generate_sms)
    g.add_node("noti_gen", korean_local_noti_gen)
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
        "name": "korean_local",
        "nodes": ["loader", "call_gen", "sms_gen", "noti_gen", "formatter"],
        "edges": [
            {"from": "loader",    "to": "call_gen"},
            {"from": "call_gen",  "to": "sms_gen"},
            {"from": "sms_gen",   "to": "noti_gen"},
            {"from": "noti_gen",  "to": "formatter"},
            {"from": "formatter", "to": "END"},
        ],
        "replaced_nodes": ["noti_gen"],
    }
