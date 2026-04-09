"""sms_gen node — rule-based conversion of raw SMS to BehaviorEventEntity."""
import json
from datetime import datetime
from typing import Any

from pipeline.state import PipelineState


def _datetime_to_unix_ms(dt_str: str) -> int:
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp() * 1000)


def _convert_sms(sms, thread_id_map):  # type: (dict, dict) -> dict
    timestamp = _datetime_to_unix_ms(sms["datetime"])
    contact_name = sms.get("contactName", "")

    if contact_name not in thread_id_map:
        thread_id_map[contact_name] = len(thread_id_map) + 1
    thread_id = thread_id_map[contact_name]

    message_type = 1 if sms.get("message_type") == "Send" else 0
    is_incoming = 0 if message_type == 1 else 1

    event_id_str = str(sms["event_id"])
    try:
        message_id = int(event_id_str)
    except ValueError:
        message_id = hash(event_id_str) % (10**9)

    payload = {
        "messageTypeAndId": f"sms_{message_id}",
        "messageType": message_type,
        "messageId": message_id,
        "threadId": thread_id,
        "date": timestamp,
        "address": sms.get("phoneNumber", ""),
        "title": contact_name if contact_name else None,
        "body": sms.get("message_content", ""),
        "isIncoming": is_incoming,
        "creator": "com.android.mms",
    }

    return {
        "identifier": f"sms_{message_id}_{timestamp}",
        "timestamp": timestamp,
        "event_source": "SMS",
        "payload": json.dumps(payload, ensure_ascii=False),
        "contextGroupId": None,
    }


def generate_sms(state: PipelineState) -> PipelineState:
    """Convert raw_sms → generated_sms."""
    thread_id_map = {}
    results = []
    for sms in state.get("raw_sms", []):
        try:
            results.append(_convert_sms(sms, thread_id_map))
        except Exception as e:
            print(f"[sms_gen] skipping {sms.get('event_id')}: {e}")

    return {**state, "generated_sms": results}
