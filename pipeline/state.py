"""PipelineState definition shared by all graph nodes."""
from typing import Any
from typing_extensions import TypedDict


class PipelineState(TypedDict):
    user_id: str          # e.g. "fenghaoran"
    data_dir: str         # absolute path to user's LifeBench directory
    persona: dict         # parsed persona.json
    daily_events: list    # parsed daily_event.json
    raw_calls: list       # parsed call.json
    raw_sms: list         # parsed sms.json
    raw_push: list        # parsed push.json
    generated_calls: list      # BehaviorEventEntity dicts for calls
    generated_sms: list        # BehaviorEventEntity dicts for SMS
    generated_noti: list       # BehaviorEventEntity dicts for notifications
    behavior_events: list      # merged + sorted final list
    metadata: dict             # run metadata (timestamps, model, etc.)
