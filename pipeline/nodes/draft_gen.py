"""draft_gen node — generates ~50 yearly life events per persona via LLM (Step 1)."""
import json
import re

from langchain_core.messages import HumanMessage

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


# ── Fallback defaults ─────────────────────────────────────────────────────────

_DEFAULT_EVENTS = [
    {"date": "2025-01-01", "title": "신년", "category": "family", "description": "새해 첫날 가족과 함께"},
    {"date": "2025-01-29", "title": "설날", "category": "family", "description": "설날 가족 모임, 세배"},
    {"date": "2025-02-14", "title": "발렌타인데이", "category": "social", "description": "지인과 초콜릿 교환"},
    {"date": "2025-03-01", "title": "삼일절 연휴", "category": "hobby", "description": "독립 기념일 연휴 휴식"},
    {"date": "2025-04-05", "title": "식목일", "category": "hobby", "description": "공원 나들이"},
    {"date": "2025-05-05", "title": "어린이날 연휴", "category": "family", "description": "가족 나들이"},
    {"date": "2025-05-08", "title": "어버이날", "category": "family", "description": "부모님께 카네이션 선물"},
    {"date": "2025-06-06", "title": "현충일", "category": "hobby", "description": "현충일 연휴 휴식"},
    {"date": "2025-07-15", "title": "여름 휴가", "category": "hobby", "description": "여름 휴가 여행"},
    {"date": "2025-08-15", "title": "광복절", "category": "hobby", "description": "광복절 연휴"},
    {"date": "2025-09-15", "title": "추석 명절", "category": "family", "description": "추석 가족 모임"},
    {"date": "2025-10-03", "title": "개천절", "category": "hobby", "description": "개천절 연휴"},
    {"date": "2025-10-06", "title": "추석", "category": "family", "description": "추석 차례"},
    {"date": "2025-11-11", "title": "빼빼로데이", "category": "social", "description": "동료에게 빼빼로 선물"},
    {"date": "2025-12-25", "title": "크리스마스", "category": "social", "description": "크리스마스 모임"},
    {"date": "2025-12-31", "title": "연말", "category": "social", "description": "연말 송년회"},
]


# ── Node ──────────────────────────────────────────────────────────────────────

def generate_drafts(state: FullPipelineState) -> FullPipelineState:
    """Step 1: For each persona, generate ~50 yearly life events."""
    provider = state.get("provider", "claude")
    personas = state.get("personas", [])
    test_mode = state.get("test_mode", False)
    event_count = 10 if test_mode else 40

    llm = _get_llm(provider, max_tokens=16384)
    daily_drafts: dict = {}

    for persona in personas:
        name = persona.get("name", "이름없음")
        age = persona.get("age", 30)
        job = persona.get("job", "직장인")
        hobbies = ", ".join(persona.get("hobbies", []))
        city = persona.get("home_city", "서울")

        print(f"[draft_gen] Generating draft events for {name} ({'test:10' if test_mode else '40'} events)...")

        prompt = f"""{name}({age}세, {job}, {city} 거주)의 2025년 주요 생활 이벤트 {event_count}개를 JSON 배열로 생성하세요.

취미: {hobbies}

조건:
- 날짜는 2025-01-01 ~ 2025-12-31 사이에 고르게 분포
- 한국 공휴일 반영: 설날(2025-01-29), 삼일절(2025-03-01), 어린이날(2025-05-05), 현충일(2025-06-06), 광복절(2025-08-15), 추석(2025-10-06), 개천절(2025-10-03), 한글날(2025-10-09), 크리스마스(2025-12-25)
- 한국 직장 문화 반영 (야근, 회식, 성과 평가 등)
- 개인 취미 활동, 가족 이벤트 포함
- description은 30자 이내로 간결하게

출력 형식 (JSON 배열):
[{{"date": "YYYY-MM-DD", "title": "이벤트명", "category": "work|family|health|social|hobby|finance", "description": "간단설명"}}]

JSON 배열만 출력하세요. 마크다운, 설명 없이."""

        events = []
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            parsed = _parse_json(response.content)
            if isinstance(parsed, list):
                events = parsed
        except Exception as e:
            print(f"[draft_gen] {name}: parse error {e}. Using defaults.")

        if not events:
            events = list(_DEFAULT_EVENTS)

        daily_drafts[name] = events
        print(f"[draft_gen] {name}: {len(events)} draft event(s)")

    return {**state, "daily_drafts": daily_drafts}
