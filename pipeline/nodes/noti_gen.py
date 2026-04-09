"""noti_gen node — rule-based conversion of raw push notifications."""
import json
from datetime import datetime
from typing import Any

from pipeline.state import PipelineState

APP_PACKAGE_MAP = {
    "NetEase Cloud Music": "com.netease.cloudmusic",
    "Keep": "com.gotokeep.keep",
    "WeChat": "com.tencent.mm",
    "Alipay": "com.eg.android.AlipayGphone",
    "Taobao": "com.taobao.taobao",
    "JD": "com.jingdong.app.mall",
    "Meituan": "com.sankuai.meituan",
    "Didi": "com.sidu.didi.psnger",
    "Douyin": "com.ss.android.ugc.aweme",
    "Xiaohongshu": "com.xingin.xhs",
    "Bilibili": "tv.danmaku.bili",
    "iQIYI": "com.qiyi.video",
    "Tencent Video": "com.tencent.qqlive",
    "QQ Music": "com.tencent.qqmusic",
    "Baidu Map": "com.baidu.BaiduMap",
    "Amap": "com.autonavi.minimap",
    "12306": "com.MobileTicket",
    "China Mobile": "com.greenpoint.android.mc10086.activity",
    "China Unicom": "com.sinovatech.unicom.ui",
    "Bank of China": "com.chinamworld.bocmbci",
    "ICBC": "com.icbc",
    "CCB": "com.ccb.android.mbank",
    "ABC": "com.android.bankabc",
    "Pinduoduo": "com.xunmeng.pinduoduo",
    "Ele.me": "me.ele",
    "Ctrip": "ctrip.android.view",
    "Qunar": "com.Qunar",
    "Fliggy": "com.taobao.trip",
    "Zhihu": "com.zhihu.android",
    "Toutiao": "com.ss.android.article.news",
    "Weibo": "com.sina.weibo",
}

_CATEGORY_MAP = {
    "NetEase Cloud Music": "music",
    "QQ Music": "music",
    "Keep": "health",
    "WeChat": "social",
    "Alipay": "finance",
    "Bank of China": "finance",
    "ICBC": "finance",
    "CCB": "finance",
    "ABC": "finance",
    "Taobao": "shopping",
    "JD": "shopping",
    "Pinduoduo": "shopping",
    "Meituan": "food",
    "Ele.me": "food",
    "Didi": "transport",
    "Douyin": "entertainment",
    "Xiaohongshu": "entertainment",
    "Bilibili": "entertainment",
    "iQIYI": "entertainment",
    "Tencent Video": "entertainment",
    "Baidu Map": "navigation",
    "Amap": "navigation",
    "12306": "travel",
    "Ctrip": "travel",
    "Qunar": "travel",
    "Fliggy": "travel",
    "Zhihu": "news",
    "Toutiao": "news",
    "Weibo": "news",
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
        "payload": json.dumps(payload, ensure_ascii=False),
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
