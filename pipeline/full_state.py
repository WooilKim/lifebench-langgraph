"""FullPipelineState — covers the entire persona-to-behavior-events pipeline."""
from typing_extensions import TypedDict


class FullPipelineState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────
    count: int      # number of personas to generate
    provider: str   # LLM provider: claude | gpt | glm
    test_mode: bool # if True, limit draft events and simulator LLM calls

    # ── Step -1: person_gen ───────────────────────────────────────────────
    persons: list   # [{name, age, gender, job}] — basic skeleton before persona expansion

    # ── Step 0: persona_gen ───────────────────────────────────────────────
    personas: list  # list of persona dicts

    # ── Step 1: draft_gen ─────────────────────────────────────────────────
    daily_drafts: dict  # persona_name → list of {date, title, category, description}

    # ── Step 2: simulator ─────────────────────────────────────────────────
    daily_events_map: dict  # persona_name → list of {datetime, event_type, description, location}

    # ── Step 3: phone_data_gen ────────────────────────────────────────────
    raw_calls: list    # flat list across all personas; each record has persona_name field
    raw_sms: list
    raw_push: list
    raw_emails: list   # email data (for GMAIL event_source)
    # pre-computed identifier sets for splitting after conversion
    persona_event_id_map: dict  # persona_name → {call_ids, sms_ids, noti_ids, gmail_ids}

    # ── Steps 4–7: call_gen / sms_gen / noti_gen / gmail_gen ─────────────
    generated_calls: list
    generated_sms: list
    generated_noti: list
    generated_gmail: list

    # ── Step 8: formatter ─────────────────────────────────────────────────
    behavior_events: list        # all personas combined, sorted by timestamp

    # ── Final per-persona split ───────────────────────────────────────────
    behavior_events_map: dict       # persona_name → list of BehaviorEventEntity dicts
    behavior_events_by_type: dict   # persona_name → {call, sms, noti, gmail}

    metadata: dict
