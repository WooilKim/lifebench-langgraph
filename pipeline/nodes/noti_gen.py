"""noti_gen node — rule-based conversion of raw push notifications."""
import json
from datetime import datetime
from typing import Any

from pipeline.state import PipelineState

APP_PACKAGE_MAP = {
    # Korean apps
    "커카오톡": "com.kakao.talk",
    "네이버": "com.naver.search",
    "유튜브": "com.google.android.youtube",
    "인스타그램": "com.instagram.android",
    "토스": "com.viva.republica.toss",
    "커카오페이": "com.kakao.pay",
    "배달의민족": "com.devsisters.cj",
    "쿠팡": "com.coupang.mobile",
    "네이버페이": "com.nhn.android.naverpay",
    "커카오맵": "net.daum.android.map",
    "네이버지도": "com.nhn.android.nmap",
    "멜론": "com.iloen.melon",
    "스포티파이": "com.spotify.music",
    "왓차": "com.watcha.watchaplay",
    "넷플릭스": "com.netflix.mediaclient",
    "당근마켓": "com.daangn.karrot",
    "무신사": "com.musinsa.app",
    "클래스101": "com.class101.app",
    "삼성페이": "com.samsung.android.spay",
    "국민은행": "com.kbstar.kbbank",
    "신한은행": "com.shinhan.sbanking",
    "하나은행": "com.kebhana.hanapay",
    "우리은행": "com.wooribank.smart.gbmb",
    "롯데ON": "com.lotteon.lotteon",
    "11번가": "com.sk11st.app",
    "네이버 블로그": "com.nhn.android.blog",
    "트위터": "com.twitter.android",
    "페이스북": "com.facebook.katana",
    # Legacy Chinese apps (keep for data compatibility)
    "NetEase Cloud Music": "com.netease.cloudmusic",
    "Keep": "com.gotokeep.keep",
    "WeChat": "com.tencent.mm",
    "Alipay": "com.eg.android.AlipayGphone",
    "Taobao": "com.taobao.taobao",
    "Meituan": "com.sankuai.meituan",
    "Didi": "com.sidu.didi.psnger",
    "Douyin": "com.ss.android.ugc.aweme",
    "Bilibili": "tv.danmaku.bili",
    "Amap": "com.autonavi.minimap",
}

_CATEGORY_MAP = {
    # Korean apps
    "커카오톡": "social",
    "네이버": "news",
    "유튜브": "entertainment",
    "인스타그램": "social",
    "토스": "finance",
    "커카오페이": "finance",
    "배달의민족": "food",
    "쿠팡": "shopping",
    "네이버페이": "finance",
    "커카오맵": "navigation",
    "네이버지도": "navigation",
    "멜론": "music",
    "스포티파이": "music",
    "왓차": "entertainment",
    "넷플릭스": "entertainment",
    "당근마켓": "shopping",
    "무신사": "shopping",
    "클래스101": "education",
    "삼성페이": "finance",
    "국민은행": "finance",
    "신한은행": "finance",
    "하나은행": "finance",
    "우리은행": "finance",
    "롯데ON": "shopping",
    "11번가": "shopping",
    "네이버 블로그": "social",
    "트위터": "social",
    "페이스북": "social",
    # Legacy Chinese
    "NetEase Cloud Music": "music",
    "Keep": "health",
    "WeChat": "social",
    "Alipay": "finance",
    "Taobao": "shopping",
    "Meituan": "food",
    "Didi": "transport",
    "Douyin": "entertainment",
    "Bilibili": "entertainment",
    "Amap": "navigation",
}


def _get_package_name(source: str) -> str:
    return APP_PACKAGE_MAP.get(
        source,
        f"com.app.{source.lower().replace(' ', '').replace('.', '')}",
    )


def _get_category(source: str) -> str:
    return _CATEGORY_MAP.get(source, "other")


def _datetime_to_unix_ms(dt_str: str) -> int:
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp() * 1000)


def _convert_notification(noti):  # type: (dict) -> dict
    timestamp = _datetime_to_unix_ms(noti["datetime"])
    source = noti.get("source", "")

    event_id_str = str(noti["event_id"])
    try:
        event_id = int(event_id_str)
    except ValueError:
        event_id = hash(event_id_str) % (10**9)

    is_clearable = noti.get("push_status") == "Read"

    payload = {
        "packageName": _get_package_name(source),
        "postTime": timestamp,
        "key": f"noti_{event_id}",
        "id": event_id,
        "isOngoing": False,
        "isClearable": is_clearable,
        "title": noti.get("title", ""),
        "text": noti.get("content", ""),
        "subText": "",
        "category": _get_category(source),
        "appName": source,
        "analysis": {},
    }

    return {
        "identifier": f"noti_{event_id}_{timestamp}",
        "timestamp": timestamp,
        "event_source": "NOTIFICATION",
        "payload": payload,
        "contextGroupId": None,
    }


def generate_notifications(state: PipelineState) -> PipelineState:
    """Convert raw_push → generated_noti."""
    results = []
    for noti in state.get("raw_push", []):
        try:
            results.append(_convert_notification(noti))
        except Exception as e:
            print(f"[noti_gen] skipping {noti.get('event_id')}: {e}")

    return {**state, "generated_noti": results}
