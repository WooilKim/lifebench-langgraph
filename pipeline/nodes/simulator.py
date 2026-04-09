"""simulator node — expands draft events into detailed daily event sequences (Step 2)."""
import json
import re
from datetime import date, timedelta

from langchain.schema import HumanMessage

from pipeline.full_state import FullPipelineState


# ── LLM helper ────────────────────────────────────────────────────────────────

def _get_llm(provider: str, max_tokens: int = 8192):
    p = (provider or "claude").lower()
    if p == "claude":
        from langchain_anthropic import ChatAnthropic
        from llm.config import CLAUDE_MODEL, ANTHROPIC_API_KEY
        return ChatAnthropic(model=CLAUDE_MODEL, api_key=ANTHROPIC_API_KEY, max_tokens=max_tokens)
    if p == "gpt":
        from langchain_openai import ChatOpenAI
        from llm.config import GPT_MODEL, OPENAI_API_KEY
        return ChatOpenAI(model=GPT_MODEL, api_key=OPENAI_API_KEY, max_tokens=max_tokens)
    if p == "glm":
        from langchain_openai import ChatOpenAI
        from llm.config import GLM_MODEL, GLM_API_KEY, GLM_BASE_URL
        return ChatOpenAI(model=GLM_MODEL, api_key=GLM_API_KEY, base_url=GLM_BASE_URL, max_tokens=max_tokens)
    raise ValueError(f"Unknown provider: {p!r}")


def _parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


# ── Routine day template ───────────────────────────────────────────────────────

def _routine_day_events(dt: str, job: str, home_city: str) -> list:
    """Return a template set of events for an ordinary work day."""
    work_locations = {
        "서울": "서울 강남 사무실",
        "부산": "부산 서면 사무실",
        "대구": "대구 중구 사무실",
        "인천": "인천 부평 사무실",
        "광주": "광주 동구 사무실",
        "대전": "대전 유성구 사무실",
    }
    work_loc = work_locations.get(home_city, "사무실")

    return [
        {"datetime": f"{dt} 07:00:00", "event_type": "기상", "description": "기상 및 아침 준비", "location": "집"},
        {"datetime": f"{dt} 08:30:00", "event_type": "출근", "description": "대중교통 이용 출근", "location": "이동 중"},
        {"datetime": f"{dt} 09:00:00", "event_type": "업무 시작", "description": f"{job} 업무 시작", "location": work_loc},
        {"datetime": f"{dt} 12:30:00", "event_type": "점심 식사", "description": "동료와 근처 식당 점심", "location": f"{home_city} 식당"},
        {"datetime": f"{dt} 18:00:00", "event_type": "퇴근", "description": "업무 마무리 후 퇴근", "location": work_loc},
        {"datetime": f"{dt} 19:30:00", "event_type": "저녁 식사", "description": "집에서 저녁 식사", "location": "집"},
        {"datetime": f"{dt} 23:00:00", "event_type": "취침", "description": "하루 마무리 및 취침", "location": "집"},
    ]


def _weekend_day_events(dt: str, hobbies: list, home_city: str) -> list:
    hobby = hobbies[0] if hobbies else "휴식"
    return [
        {"datetime": f"{dt} 09:00:00", "event_type": "기상", "description": "주말 늦게 기상", "location": "집"},
        {"datetime": f"{dt} 10:30:00", "event_type": "취미 활동", "description": f"{hobby} 즐기기", "location": home_city},
        {"datetime": f"{dt} 13:00:00", "event_type": "점심 식사", "description": "가족이나 친구와 점심", "location": f"{home_city} 식당"},
        {"datetime": f"{dt} 15:00:00", "event_type": "휴식", "description": "오후 자유 시간", "location": "집"},
        {"datetime": f"{dt} 19:00:00", "event_type": "저녁 식사", "description": "저녁 식사", "location": "집"},
        {"datetime": f"{dt} 23:30:00", "event_type": "취침", "description": "취침", "location": "집"},
    ]


def _is_weekend(dt_str: str) -> bool:
    d = date.fromisoformat(dt_str)
    return d.weekday() >= 5  # 5=Sat, 6=Sun


# ── Node ──────────────────────────────────────────────────────────────────────

def simulate_daily_events(state: FullPipelineState) -> FullPipelineState:
    """Step 2: Expand draft events into detailed daily event sequences."""
    provider = state.get("provider", "claude")
    personas = state.get("personas", [])
    daily_drafts = state.get("daily_drafts", {})
    test_mode = state.get("test_mode", False)

    llm = _get_llm(provider, max_tokens=8192)
    daily_events_map: dict = {}

    for persona in personas:
        name = persona.get("name", "이름없음")
        job = persona.get("job", "직장인")
        city = persona.get("home_city", "서울")
        hobbies = persona.get("hobbies", [])
        drafts = daily_drafts.get(name, [])

        print(f"[simulator] Simulating daily events for {name} ({len(drafts)} draft days, {'test:template-only' if test_mode else 'LLM+template'})...")

        # Index drafts by date
        draft_by_date: dict = {d["date"]: d for d in drafts}

        all_events: list = []

        # --- Generate routine (template) events for ALL 365 days ---
        for i in range(365):
            dt = (date(2025, 1, 1) + timedelta(days=i)).isoformat()
            if dt not in draft_by_date:
                if _is_weekend(dt):
                    all_events.extend(_weekend_day_events(dt, hobbies, city))
                else:
                    all_events.extend(_routine_day_events(dt, job, city))

        # --- Generate LLM-detailed events for draft days in batches of 5 ---
        draft_dates = sorted(draft_by_date.keys())
        # test_mode: skip LLM, use templates for all draft days too
        if test_mode:
            for dt in draft_dates:
                if _is_weekend(dt):
                    all_events.extend(_weekend_day_events(dt, hobbies, city))
                else:
                    all_events.extend(_routine_day_events(dt, job, city))
            all_events.sort(key=lambda e: e.get("datetime", ""))
            daily_events_map[name] = all_events
            print(f"[simulator] {name}: {len(all_events)} events (test/template-only)")
            continue

        batch_size = 5

        for batch_start in range(0, len(draft_dates), batch_size):
            batch = draft_dates[batch_start: batch_start + batch_size]
            events_desc = "\n".join(
                f"- {d}: [{draft_by_date[d]['category']}] {draft_by_date[d]['title']} — {draft_by_date[d]['description']}"
                for d in batch
            )

            prompt = f"""{name}({job}, {city})의 아래 날짜들에 대한 상세 이벤트를 JSON으로 생성하세요.

날짜별 주요 이벤트:
{events_desc}

각 날짜당 5~8개의 상세 이벤트를 생성하세요.
출력 형식:
{{
  "YYYY-MM-DD": [
    {{"datetime": "YYYY-MM-DD HH:MM:SS", "event_type": "...", "description": "...", "location": "..."}}
  ]
}}

JSON만 출력하세요. 마크다운, 설명 없이."""

            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                parsed = _parse_json(response.content)
                if isinstance(parsed, dict):
                    for dt, evs in parsed.items():
                        if isinstance(evs, list):
                            all_events.extend(evs)
            except Exception as e:
                print(f"[simulator] {name} batch {batch_start}: parse error {e}. Using templates.")
                for dt in batch:
                    if _is_weekend(dt):
                        all_events.extend(_weekend_day_events(dt, hobbies, city))
                    else:
                        all_events.extend(_routine_day_events(dt, job, city))

        # Sort by datetime
        all_events.sort(key=lambda e: e.get("datetime", ""))
        daily_events_map[name] = all_events
        print(f"[simulator] {name}: {len(all_events)} detailed event(s)")

    return {**state, "daily_events_map": daily_events_map}
