"""simulator node — expands draft events into detailed daily event sequences (Step 2).

Seven sub-steps mirroring Original LifeBench Mind/Refiner/Formatter pipeline:
  Sub-step 1 (fuzzy_memory_monthly):              monthly fuzzy memory build ×12
  Sub-step 2 (get_plan4):                         rule-based draft event extraction per day
  Sub-step 3 (subjective_plan):                   1st-person subjective planning per event-day
  Sub-step 4 (objective_optimize):                objective optimization + time-fill per event-day
  Sub-step 5 (traffic_adjust):                    location + commute time adjustment per event-day
  Sub-step 6 (reflection + long_term_memory):     daily reflection + long-term memory update
  Sub-step 7 (impact_events_analysis ×12):        monthly health/state analysis
  Sub-step 8 (event_format_sequence):             final event formatting per event-day

test_mode: skips LLM, uses rule-based templates for all days.
"""
import json
import re
from datetime import date, timedelta

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


def _parse_json_safe(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


# ── Rule-based templates ───────────────────────────────────────────────────────

def _routine_day_events(dt: str, job: str, home_city: str) -> list:
    work_locations = {
        "서울": "서울 강남 사무실", "부산": "부산 서면 사무실",
        "대구": "대구 중구 사무실", "인천": "인천 부평 사무실",
        "광주": "광주 동구 사무실", "대전": "대전 유성구 사무실",
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
    return date.fromisoformat(dt_str).weekday() >= 5


# ── Sub-step 1: fuzzy_memory_monthly ──────────────────────────────────────────

def _build_fuzzy_memory(llm, persona: dict, drafts: list) -> dict:
    """Build monthly fuzzy memory summaries (×12)."""
    name = persona.get("name", "")
    persona_summary = f"{name}({persona.get('age')}세, {persona.get('job')}, {persona.get('home_city')} 거주)"

    monthly_memory: dict = {}

    for month in range(1, 13):
        month_str = f"2025-{month:02d}"
        month_events = [d for d in drafts if d.get("date", "").startswith(month_str)]
        if not month_events:
            monthly_memory[month_str] = ""
            continue

        events_desc = "\n".join(
            f"  {e['date']}: [{e['category']}] {e['title']} — {e['description']}"
            for e in month_events
        )

        prompt = f"""당신은 기억 전문가입니다. 다음 인물 프로파일과 2025년 {month}월의 이벤트를 기반으로,
아래 기준에 따라 중요 이벤트를 엄격하게 선별하여 요약하세요:

세 가지 유형의 이벤트만 포함:
1. 개인에게 현저한 영향을 미치고 미래 행동을 변화시킬 수 있는 이벤트
2. 인물 프로파일 업데이트가 필요한 변화성 이벤트
3. 일정 기간 생활에 지속적 영향을 미치는 이벤트

인물 프로파일: {persona_summary}

2025년 {month}월의 이벤트:
{events_desc}

출력 요구사항:
- 1인칭으로 작성
- 이벤트+영향의 간결한 요약 (200자 이내)
- 해당 없으면 빈 문자열 출력"""

        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            monthly_memory[month_str] = resp.content.strip()
        except Exception as e:
            print(f"[simulator] {name} fuzzy_memory {month_str} error: {e}")
            monthly_memory[month_str] = ""

    return monthly_memory


# ── Sub-step 2: get_plan4 (rule-based) ────────────────────────────────────────

def _get_plan4(dt: str, draft_by_date: dict) -> list:
    """Extract draft events for a given date (rule-based)."""
    return draft_by_date.get(dt, [])


# ── Sub-step 3: subjective_plan ───────────────────────────────────────────────

def _generate_subjective_plan(llm, persona: dict, dt: str, plan_events: list,
                               monthly_memory: dict) -> dict:
    """1st-person subjective daily planning."""
    month_str = dt[:7]
    memory = monthly_memory.get(month_str, "")
    cognition = f"{persona.get('mbti', '')} {persona.get('job', '')}, {persona.get('home_city', '')} 거주"
    plan_desc = "\n".join(
        f"  [{e.get('category','')}] {e.get('title','')} — {e.get('description','')}"
        for e in plan_events
    ) or "(특별 이벤트 없음)"

    prompt = f"""역할에 완전히 몰입하여, **1인칭 "나"** 의 시각으로 깊이 생각하고, 아래 정보를 종합 활용하여 오늘의 일정을 분석·계획하세요:
- 자아인식 (성격 특성, 행동 습관, 개인 선호)
- 기억 데이터 (과거 활동 궤적, 반복 이벤트 패턴, 장기 생활 습관)
- 현재 생각 (최근 의향, 관심사)
- 현재 상태 (아침에 깨어난 장소, 신체/정신 상태)
- 오늘 날짜 속성 (평일/휴일 명확히 구분)
- **오늘 초기 계획:**
{plan_desc}

## 사고 프레임워크
1. **현재 상태 및 기억 회고**
   - 기억에서 핵심 정보 추출, 아침에 깨어난 구체적 장소 명확화
   - 현재 신체 상태 분석

2. **오늘 이벤트 배치 아이디어**
   - 초기 계획의 핵심 이벤트 우선 배치
   - 성격 선호에 맞는 활동 보완

인지 상태: {cognition}
퍼지 기억: {memory or '(없음)'}
날짜: {dt}

출력 (JSON):
{{"thought": "1인칭 오늘 생각...", "plan": ["HH:MM 이벤트1", "HH:MM 이벤트2", ...]}}
JSON만 출력."""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        return _parse_json_safe(resp.content)
    except Exception as e:
        print(f"[simulator] subjective_plan {dt} error: {e}")
        return {"thought": "", "plan": []}


# ── Sub-step 4: objective_optimize ────────────────────────────────────────────

def _generate_objective_events(llm, persona: dict, dt: str, subjective: dict) -> str:
    """Objective optimization of the subjective plan — full day event table."""
    plan_text = "\n".join(subjective.get("plan", []))
    thought = subjective.get("thought", "")

    prompt = f"""의인화된 사고 과정(개인의 당일 주관적 계획, 반드시 이 계획대로 실행되지 않을 수 있음)을 기반으로, 객관적 시각에서 당일 일정·사고 과정·객관적 합리성 원칙을 결합하여 하루 전체 이벤트를 조정·보완·최적화합니다.

## 핵심 요구사항
1. 모든 이벤트의 발생 시간대 명확화, 공백 시간대 채워서 내용 공백 방지.
2. 주관적 사고에서 언급되지 않은 일상화된 활동 패턴 보완 (평일 근무, 식사 등).
3. 이벤트 기술 상세화: 발생 및 경과 전 과정, 관련 인물/장소/시간 포함.

주관적 계획:
{thought}
{plan_text}

날짜: {dt}
인물: {persona.get('name')}({persona.get('age')}세, {persona.get('job')}, {persona.get('home_city')})

출력: "시간 구간, 이벤트, 장소, 설명" 형식으로 전일 이벤트 텍스트."""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content.strip()
    except Exception as e:
        print(f"[simulator] objective_optimize {dt} error: {e}")
        return ""


# ── Sub-step 5: traffic_adjust ────────────────────────────────────────────────

def _adjust_traffic(llm, persona: dict, dt: str, objective_text: str) -> str:
    """Add real addresses and commute times to events."""
    city = persona.get("home_city", "서울")

    prompt = f"""원래 이벤트 배치와 지도 검색 결과를 바탕으로, 해당 일의 이벤트에 실제 주소와 이동 시간을 배분하여 개인의 이동 궤적이 실제 시간 논리에 부합하도록 보장합니다.

## 핵심 요구사항
1. 이벤트 합리성 검증 후 필요 시 수정.
2. 원래 이벤트 배치를 최대한 따름. 세부 사항만 보충, 시간 조정, 실제 주소 배분.
3. 장소 전환 논리 명확화. 구체적 이동 시간(분 단위), 이동 방식(지하철/버스/자가용/도보) 포함.
4. 하루 이벤트를 풍부하게 하여 생동감 있고 현실적으로.

해당 일 이벤트 원래 배치:
{objective_text}

날짜: {dt}
도시: {city}
인물: {persona.get('name')}({persona.get('age')}세, {persona.get('job')})

출력: "시간 구간, 이벤트, 상세 장소(시/구/도로/건물 포함), 설명" 형식으로 전일 이벤트."""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content.strip()
    except Exception as e:
        print(f"[simulator] traffic_adjust {dt} error: {e}")
        return objective_text


# ── Sub-step 6a: daily_reflection ────────────────────────────────────────────

def _generate_reflection(llm, persona: dict, dt: str, events_text: str,
                          plan_events: list) -> dict:
    """Generate daily reflection."""
    plan_desc = "\n".join(
        f"  {e.get('title','')}" for e in plan_events
    ) or "(특별 이벤트 없음)"

    prompt = f""""나"의 1인칭 시각으로 서술 전개. 먼저 오늘 발생한 모든 이벤트를 완전히 정리하여 몰입한 뒤 아래 차원들을 자연스럽게 통합 표현하세요:

- **핵심 감정**: 오늘 경험에서 생겨난 진실한 감정, 구체적 이벤트로 감정 유발점 설명
- **자아 통찰**: 자아인식·과거 기억 연결, 오늘 이벤트가 자신 상태에 미친 영향 분석
- **이벤트 기억**: 오늘 이벤트에 후속 이벤트 언급 여부 확인, 해당 이벤트 기록
- **생각 반성**: 가장 인상 깊은 일 1-2가지 간략 코멘트

이벤트:
{events_text[:1000]}

초기 계획: {plan_desc}
날짜: {dt}
인물: {persona.get('name')}

출력 (JSON):
{{"date": "{dt}", "thought": "오늘...", "future_plans": [{{"date": "2025-XX-XX", "plan": "..."}}]}}
JSON만 출력."""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        return _parse_json_safe(resp.content)
    except Exception as e:
        print(f"[simulator] reflection {dt} error: {e}")
        return {"date": dt, "thought": "", "future_plans": []}


# ── Sub-step 6b: update_long_term_memory ─────────────────────────────────────

def _update_long_term_memory(llm, persona: dict, dt: str, events_text: str,
                              current_memory: str) -> str:
    """Update long-term memory with today's events."""
    prompt = f"""오늘의 생활 데이터를 기반으로 개인 장기기억을 업데이트하세요.

### 장기기억 업데이트 핵심 규칙
1. 고가치 정보만 보존: 미래 계획(날짜 명시), 객관적 사실, 인상 깊은 핵심 이벤트, 반복 이벤트 요약
2. 가치 없는 내용 삭제, 이미 발생한 미래 이벤트·시효 지난 구 기억 제거
3. 모든 기억 이벤트에 날짜 명시 필수 (형식: YYYY-MM-DD)

<life>{events_text[:800]}</life>
<memory>{current_memory or '(없음)'}</memory>

날짜: {dt}
인물: {persona.get('name')}

업데이트된 장기기억 텍스트만 출력 (200자 이내)."""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content.strip()
    except Exception as e:
        print(f"[simulator] long_term_memory {dt} error: {e}")
        return current_memory


# ── Sub-step 7: impact_events_analysis ×12 ───────────────────────────────────

def _analyze_monthly_impact(llm, persona: dict, month: int,
                             monthly_events_text: str, initial_health: str) -> dict:
    """Monthly health/state impact analysis."""
    month_str = f"2025-{month:02d}"
    persona_summary = f"{persona.get('name')}({persona.get('age')}세, {persona.get('job')}, {persona.get('home_city')} 거주)"

    prompt = f"""다음 인물 프로파일, 초기 건강 상태, 지정 기간 내 이벤트 목록을 기반으로, 해당 월 15일 중간 상태, 말일 건강 상태 기록 및 완전한 변화 노드를 분석·생성해주세요.

## 인물 프로파일
{persona_summary}

## 초기 건강 상태
{initial_health or '(정보 없음)'}

## 입력 데이터
{monthly_events_text[:1500] or '(이 달 이벤트 없음)'}

## 분석 요구사항
객관적 환경 변화, 사회 상식적 변화 및 이벤트 영향을 결합하여:
1. 운동/수면/식이/건강 상태 관련 트렌드와 변화 파악
2. 합리적 변화 이벤트 추가 (최소 3개)
3. 15일 중간 상태 생성
4. 다양성 있는 최종 상태 생성

출력 (JSON):
{{"mid_month_state": {{"health_status": {{"summary": "..."}},"habits_summary": {{"lifestyle_summary": "..."}}}}, "end_of_month_state": {{"health_status": {{"summary": "..."}}, "habits_summary": {{"lifestyle_summary": "..."}}}}, "change_nodes": [{{"date": "2025-MM-DD", "event": "...", "impact": "..."}}]}}
JSON만 출력."""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        return _parse_json_safe(resp.content)
    except Exception as e:
        print(f"[simulator] impact_analysis {month_str} error: {e}")
        return {}


# ── Sub-step 8: event_format_sequence ────────────────────────────────────────

def _format_event_sequence(llm, persona: dict, dt: str, events_text: str) -> list:
    """Format raw event text into structured event list (≤20 events)."""
    prompt = f""""이전에 생성된 하루 전체 이벤트 목록"을 기반으로, 새로운 형식화된 이벤트 시퀀스를 추출·생성하세요.

1) 이벤트 추출 규칙: "시간 구간 + 설명"에서 이벤트 설명과 제목을 최적화 조정.
2) 이벤트 수량 제한: 최종 출력 이벤트 수는 20개 이하. 시간 인접/장소 기준으로 합리적 병합 가능.

오늘 날짜: {dt}
인물: {persona.get('name')}({persona.get('age')}세, {persona.get('job')}, {persona.get('home_city')})

## 출력 형식 (JSON 배열)
- type: Career / Education / Relationships / Family&Living Situation / Personal Life / Finance / Health / Unexpected Events / Other
- name: 이벤트 이름
- description: 상세 설명
- location: 상세 주소
- date: "YYYY-MM-DD HH:MM:SS至YYYY-MM-DD HH:MM:SS"
- participant: [{{"name": "이름", "relation": "관계"}}]

이벤트 목록:
{events_text[:2000]}

JSON 배열만 출력."""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        result = _parse_json_safe(resp.content)
        if isinstance(result, list):
            return result
    except Exception as e:
        print(f"[simulator] format_sequence {dt} error: {e}")

    # fallback: parse text into basic events
    lines = [l.strip() for l in events_text.split("\n") if l.strip()]
    events = []
    for line in lines[:15]:
        m = re.match(r"^(\d{1,2}:\d{2})[^\d](.+)", line)
        if m:
            events.append({
                "datetime": f"{dt} {m.group(1)}:00",
                "event_type": m.group(2)[:30],
                "description": m.group(2),
                "location": persona.get("home_city", "서울"),
            })
    return events


# ── Node ──────────────────────────────────────────────────────────────────────

def simulate_daily_events(state: FullPipelineState) -> FullPipelineState:
    """Step 2: Expand draft events into detailed daily event sequences via 8 sub-steps."""
    provider = state.get("provider", "claude")
    personas = state.get("personas", [])
    daily_drafts = state.get("daily_drafts", {})
    test_mode = state.get("test_mode", False)

    daily_events_map: dict = {}

    if test_mode:
        # test_mode: template-only for all days
        for persona in personas:
            name = persona.get("name", "")
            job = persona.get("job", "직장인")
            city = persona.get("home_city", "서울")
            hobbies = persona.get("hobbies", [])
            all_events: list = []
            for i in range(365):
                dt = (date(2025, 1, 1) + timedelta(days=i)).isoformat()
                if _is_weekend(dt):
                    all_events.extend(_weekend_day_events(dt, hobbies, city))
                else:
                    all_events.extend(_routine_day_events(dt, job, city))
            all_events.sort(key=lambda e: e.get("datetime", ""))
            daily_events_map[name] = all_events
            print(f"[simulator] {name}: {len(all_events)} events (test/template-only)")
        return {**state, "daily_events_map": daily_events_map}

    llm = _get_llm(provider, max_tokens=8192)

    for persona in personas:
        name = persona.get("name", "")
        job = persona.get("job", "직장인")
        city = persona.get("home_city", "서울")
        hobbies = persona.get("hobbies", [])
        drafts = daily_drafts.get(name, [])
        draft_by_date: dict = {d["date"]: [d] for d in drafts}

        print(f"[simulator] {name}: starting 8-sub-step simulation ({len(drafts)} draft events)...")

        # ── Sub-step 1: fuzzy_memory_monthly ×12 ──────────────────────────────
        print(f"[simulator] {name}: Sub-step 1 — fuzzy_memory_monthly ×12...")
        monthly_memory = _build_fuzzy_memory(llm, persona, drafts)

        # ── Sub-steps 2-6, 8: per event-day (draft dates) ─────────────────────
        all_events: list = []

        # Rule-based events for non-draft days
        for i in range(365):
            dt = (date(2025, 1, 1) + timedelta(days=i)).isoformat()
            if dt not in draft_by_date:
                if _is_weekend(dt):
                    all_events.extend(_weekend_day_events(dt, hobbies, city))
                else:
                    all_events.extend(_routine_day_events(dt, job, city))

        # LLM pipeline for draft days
        long_term_memory = ""
        draft_dates = sorted(draft_by_date.keys())

        for dt in draft_dates:
            plan_events = _get_plan4(dt, draft_by_date)  # Sub-step 2

            # Sub-step 3: subjective plan
            subjective = _generate_subjective_plan(llm, persona, dt, plan_events, monthly_memory)

            # Sub-step 4: objective optimize
            objective_text = _generate_objective_events(llm, persona, dt, subjective)

            # Sub-step 5: traffic adjust
            adjusted_text = _adjust_traffic(llm, persona, dt, objective_text)

            # Sub-step 6a: reflection
            reflection = _generate_reflection(llm, persona, dt, adjusted_text, plan_events)

            # Sub-step 6b: update long-term memory
            long_term_memory = _update_long_term_memory(
                llm, persona, dt, adjusted_text, long_term_memory
            )

            # Sub-step 8: format sequence
            formatted = _format_event_sequence(llm, persona, dt, adjusted_text)
            if formatted:
                # Convert to common format
                for ev in formatted:
                    if isinstance(ev, dict):
                        # Normalize to our standard format
                        date_str = ev.get("date", "")
                        # Extract start datetime
                        dt_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", date_str)
                        all_events.append({
                            "datetime": dt_match.group(1) if dt_match else f"{dt} 09:00:00",
                            "event_type": ev.get("name", ev.get("type", "이벤트")),
                            "description": ev.get("description", ""),
                            "location": ev.get("location", city),
                        })
            else:
                # fallback to routine
                if _is_weekend(dt):
                    all_events.extend(_weekend_day_events(dt, hobbies, city))
                else:
                    all_events.extend(_routine_day_events(dt, job, city))

            print(f"[simulator] {name}: {dt} done")

        # ── Sub-step 7: impact_events_analysis ×12 ────────────────────────────
        print(f"[simulator] {name}: Sub-step 7 — impact_events_analysis ×12...")
        initial_health = persona.get("health_desc", "")
        for month in range(1, 13):
            month_str = f"2025-{month:02d}"
            month_events_text = "\n".join(
                f"  {e.get('datetime','')}: {e.get('event_type','')} — {e.get('description','')}"
                for e in all_events
                if e.get("datetime", "").startswith(month_str)
            )
            result = _analyze_monthly_impact(llm, persona, month, month_events_text, initial_health)
            # Update initial_health for next month
            end_state = result.get("end_of_month_state", {})
            if end_state:
                health = end_state.get("health_status", {})
                initial_health = health.get("summary", initial_health)

        all_events.sort(key=lambda e: e.get("datetime", ""))
        daily_events_map[name] = all_events
        print(f"[simulator] {name}: {len(all_events)} total events")

    return {**state, "daily_events_map": daily_events_map}
