"""gmail_gen node — converts raw emails to BehaviorEventEntity (GMAIL event_source).

format.txt Gmail payload fields:
  id: String          — unique email ID
  uid: Long           — numeric UID
  from: String        — sender address
  subject: String     — email subject
  snippet: String     — short preview text
  receivedEpochMs: Long — reception timestamp in ms
"""
import json
from datetime import datetime

from pipeline.full_state import FullPipelineState


def _datetime_to_unix_ms(dt_str: str) -> int:
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp() * 1000)


def _convert_email(email: dict) -> dict:
    """Convert raw email dict to BehaviorEventEntity format."""
    timestamp = _datetime_to_unix_ms(email["datetime"])
    event_id = email["event_id"]

    payload = {
        "id": f"gmail_{event_id}",
        "uid": event_id,
        "from": email.get("from_address", ""),
        "subject": email.get("subject", ""),
        "snippet": email.get("snippet", ""),
        "receivedEpochMs": timestamp,
    }

    return {
        "identifier": f"gmail_{event_id}_{timestamp}",
        "timestamp": timestamp,
        "event_source": "GMAIL",
        "payload": json.dumps(payload, ensure_ascii=False),
        "contextGroupId": None,
    }


def generate_gmail(state: FullPipelineState) -> FullPipelineState:
    """Convert raw_emails → generated_gmail."""
    results = []
    for email in state.get("raw_emails", []):
        try:
            results.append(_convert_email(email))
        except Exception as e:
            print(f"[gmail_gen] skipping {email.get('event_id')}: {e}")

    return {**state, "generated_gmail": results}
