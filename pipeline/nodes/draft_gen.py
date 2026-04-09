"""draft_gen node — generates yearly life events per persona via LLM (Step 1).

Five sub-steps mirroring Original LifeBench Scheduler:
  Round 1 (yearterm_eventgen):            ~100 events (9 categories)
  Round 2 (yearterm_complete):            ~90  more events (complement)
  Round 3 (extract_impact_events):        extract impact / habit-change events
  Round 4 (monthly_events_analysis):      monthly analysis × 12
  Round 5 (timeline_conflict_resolution): conflict resolution + annual overview
"""
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


def _parse_json_safe(text: str):
    from json_repair import repair_json
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(repair_json(text.strip()))


# ── Event text parsers ─────────────────────────────────────────────────────────

def _parse_event_line(line: str, category: str) -> dict | None:
    """Parse a single event line from yearterm output.

    Expected formats:
      이벤트명 (2025-MM-DD): 설명
      이벤트명 (2025-MM-DD 至 2025-MM-DD): 설명
    """
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("<"):
        return None
    m = re.match(
        r"^(.+?)\s*[（(]\s*(\d{4}-\d{2}-\d{2})(?:\s*[至~]\s*\d{4}-\d{2}-\d{2})?\s*[）)]\s*[:：]\s*(.+)$",
        line,
    )
    if m:
        return {
            "date": m.group(2),
            "title": m.group(1).strip(),
            "category": category,
            "description": m.group(3).strip(),
        }
    return None


def _parse_yearterm_output(text: str) -> list[dict]:
    """Parse category-grouped event text from yearterm_eventgen / yearterm_complete.

    Handles <type>Career（N개）</type> section headers.
    """
    events: list[dict] = []
    current_category = "Other"

    for line in text.split("\n"):
        # Section header: <type>Career（N개）</type>
        cat_m = re.search(r"<type>\s*([^（(）)\n<]+?)[\s（(]", line)
        if not cat_m:
            # Fallback: **Career（N개）** style
            cat_m = re.search(r"\*\*\s*([A-Za-z&\s]+?)[\s（(]", line)
        if cat_m:
            current_category = cat_m.group(1).strip()
            continue

        event = _parse_event_line(line, current_category)
        if event:
            events.append(event)

    return events


# ── Persona summary builder ────────────────────────────────────────────────────

def _build_summary(persona: dict) -> str:
    """Build a compact persona summary string for prompts."""
    parts = [
        f"{persona.get('name', '')}({persona.get('age', '')}세, "
        f"{persona.get('job', '')}, {persona.get('home_city', '서울')} 거주)"
    ]
    for key, label in [
        ("mbti", "MBTI"), ("education", "학력"), ("marital_status", "혼인"),
    ]:
        val = persona.get(key)
        if val:
            parts.append(f"{label}: {val}")
    hobbies = persona.get("hobbies", [])
    if hobbies:
        parts.append(f"취미: {', '.join(hobbies)}")
    for key in ("work_desc", "lifestyle_desc", "health_desc", "description"):
        val = persona.get(key)
        if val:
            parts.append(val)
            break
    return "\n".join(parts)


def _build_category_summaries(events: list[dict]) -> str:
    """Group events by category for the monthly analysis prompt."""
    by_cat: dict[str, list[str]] = {}
    for e in events:
        cat = e.get("category", "Other")
        by_cat.setdefault(cat, []).append(
            f"  {e['date']}: {e['title']} — {e['description']}"
        )
    lines: list[str] = []
    for cat, evs in by_cat.items():
        lines.append(f"[{cat}] ({len(evs)}개)")
        lines.extend(evs[:5])  # cap per category
    return "\n".join(lines)


def _extract_events_from_monthly(monthly_results: list[dict]) -> list[dict]:
    """Convert monthly analysis key_events to a flat event list."""
    events: list[dict] = []
    for mr in monthly_results:
        if not isinstance(mr, dict):
            continue
        for ke in mr.get("key_events", []):
            if not isinstance(ke, dict):
                continue
            tp = ke.get("time_point", "")
            ev = ke.get("event", "")
            if tp and ev:
                if ":" in ev:
                    title, desc = ev.split(":", 1)
                else:
                    title, desc = ev[:40], ev
                events.append({
                    "date": tp,
                    "title": title.strip(),
                    "category": "general",
                    "description": desc.strip(),
                })
    return events


# ── Fallback defaults ─────────────────────────────────────────────────────────

_DEFAULT_EVENTS = [
    {"date": "2025-01-01", "title": "신년", "category": "family", "description": "새해 첫날 가족과 함께"},
    {"date": "2025-01-29", "title": "설날", "category": "family", "description": "설날 가족 모임, 세배"},
    {"date": "2025-03-01", "title": "삼일절 연휴", "category": "hobby", "description": "독립 기념일 연휴 휴식"},
    {"date": "2025-05-05", "title": "어린이날 연휴", "category": "family", "description": "가족 나들이"},
    {"date": "2025-07-15", "title": "여름 휴가", "category": "hobby", "description": "여름 휴가 여행"},
    {"date": "2025-09-15", "title": "추석 명절", "category": "family", "description": "추석 가족 모임"},
    {"date": "2025-12-25", "title": "크리스마스", "category": "social", "description": "크리스마스 모임"},
    {"date": "2025-12-31", "title": "연말", "category": "social", "description": "연말 송년회"},
]


# ── Node ──────────────────────────────────────────────────────────────────────

def generate_drafts(state: FullPipelineState) -> FullPipelineState:
    """Step 1: Generate yearly life events per persona via 5 sub-steps.

    test_mode: Round 1 only (20 events), skip Rounds 2-5.
    full mode: All 5 rounds (100 + 90 events → impact extract → monthly × 12 → conflict resolve).
    """
    provider = state.get("provider", "claude")
    personas = state.get("personas", [])
    test_mode = state.get("test_mode", False)

    llm = _get_llm(provider, max_tokens=16384)
    daily_drafts: dict = {}

    for persona in personas:
        name    = persona.get("name", "이름없음")
        summary = _build_summary(persona)

        print(f"[draft_gen] {name}: starting {'test(Round 1 only)' if test_mode else '5-round'} generation...")

        # ── Round 1: template_yearterm_eventgen ────────────────────────────────
        event_count = 20 if test_mode else 100
        print(f"[draft_gen] {name}: Round 1 — yearterm_eventgen ({event_count} events)...")

        r1_prompt = f"""인물 분석을 기반으로 2025년 내 발생 가능한 {event_count}개의 생활 이벤트를 생성합니다.

## 一、핵심 창작 원칙

### 1. 진실성과 부합성
- 인물 프로파일 심층 반영: 직업 배경, 가족 관계, 생활 습관, 성격 특성, 경제 상황, 소셜 서클과 강하게 연결.
- 현실 논리 부합: 시간 규율(예: "겨울 스키", "설 귀성"), 생활 상식(예: "인테리어는 1-2개월 소요"), 사회 규칙.
- 자연스러운 인과 관계: 일부 이벤트 간 합리적 인과 관계 (예: "새 집 구매" → "인테리어 설계" → "이사 축하").

### 2. 다양성과 균형성
- 생활 차원 전면 포함: 직업, 가족, 사교, 건강, 재무, 개인 성장 등
- 이벤트 유형 다양화: 단기(1-3일), 중기(1-4주), 장기(1-3개월), 일회성/주기성
- 감정과 강도 층차: 긍정적/중립적/경미한 도전적 이벤트 포함

### 3. 비일상성과 기억 포인트
- 기록할 가치 있고, 사전 준비가 필요하고, 개인 특성을 보여주는 장면 우선 선택.
- 각 카테고리의 이벤트는 독특성을 가지며, 동일 카테고리 내 동질화 금지.

## 二、이벤트 세부 요구사항

### 1. 시간 설정 규범
- 형식 통일: 모든 이벤트에 구체적 시간 표시 (형식: 2025-XX-XX 또는 2025-XX-XX 至 2025-XX-XX).
- 주기성 이벤트: 여러 번 발생하는 이벤트는 모든 시간 노드 표시.
- 시간 분포: 이벤트는 전년도 각 월에 균등하게 분포, 과도한 집중이나 희소화 금지.
- 시간 중복: 합리적 시간 중복 허용 (예: "여행" 기간 중 "원격 업무 처리").

### 2. 카테고리 포함 요구사항 (9개 카테고리)
- Career: 직업/업무 관련 (예: "업계 포럼 참석", "신제품 R&D 프로젝트 참여")
- Education: 교육/학습 관련 (예: "수업", "전문 자격증 취득")
- Relationships: 인간 관계 (예: "부모님 생일 파티 준비", "친구 여행 조직")
- Family&Living Situation: 가족 생활 및 거주 환경 (예: "주택 인테리어", "스마트홈 설치")
- Personal Life: 자기 관리/오락/생활방식 (예: "정기 스파", "단기 여행")
- Finance: 개인 자산/재무 (예: "금융 상품 구매", "자동차 유지 보수")
- Health: 건강 관리 (예: "한방 치료", "요가 정기 수련")
- Unexpected Events: 돌발 대응 (예: "차량 소사고 처리", "임시 초과 근무")
- Other: 기타 미포함 카테고리

카테고리 배분 원칙:
- 인물의 직업/생활 상태/성격 특성과 높게 관련된 카테고리 우선 생성
- 인물 프로파일과 관련성이 매우 낮은 카테고리는 이벤트를 적게 또는 생성하지 않을 수 있음

### 3. 이벤트 설명 요구사항
- 구체적이고 명확하게: 이벤트 설명에 who(참여자)/what(내용)/why(목적/의미) 포함.
- 간결하게: 각 이벤트 설명은 15-50자.

## 三、출력 형식 요구사항

### 1. 구조화 출력
- 엄격하게 「카테고리 + 이벤트 수」로 그룹화하여 출력, 각 카테고리 독립 단락.
- 각 카테고리 아래 이벤트는 시간 순서로 정렬.
- 카테고리 이름은 위의 9개 카테고리와 완전히 일치.

### 2. 형식 예시
<type>Career（N개）</type>
    업계 고봉 포럼 참가 (2025-03-18 至 2025-03-19): 업계 트렌드 학습 및 공급업체 자원 확장.
    신제품 출시 기획 (2025-04-01 至 2025-04-15): 여름 신제품 홍보 방안 주도 수립.

### 3. 핵심 제약
- 출력 내용에는 카테고리 제목과 이벤트 목록만 포함, 추가 분석/설명 텍스트 금지.
- 각 이벤트는 반드시 「이벤트명（시간）: 설명」 형식으로 제시.

---

인물 분석: {summary}"""

        events_r1: list[dict] = []
        try:
            resp = llm.invoke([HumanMessage(content=r1_prompt)])
            events_r1 = _parse_yearterm_output(resp.content)
            print(f"[draft_gen] {name}: Round 1 → {len(events_r1)} events parsed")
        except Exception as e:
            print(f"[draft_gen] {name}: Round 1 error: {e}")

        # test_mode: stop here
        if test_mode:
            events = events_r1 if events_r1 else list(_DEFAULT_EVENTS)
            print(f"[draft_gen] {name}: test mode done — {len(events)} events")
            daily_drafts[name] = events
            continue

        # ── Round 2: template_yearterm_complete ────────────────────────────────
        print(f"[draft_gen] {name}: Round 2 — yearterm_complete (~90 more events)...")

        r1_text = "\n".join(
            f"  {e['date']}: [{e['category']}] {e['title']} — {e['description']}"
            for e in events_r1[:60]
        )

        r2_prompt = f"""인물 프로파일과 1라운드에서 생성한 이벤트를 기반으로, 요구사항에 엄격히 부합하는 새로운 이벤트 약 90개를 추가 창작하세요. 중복을 철저히 피하고 정확한 보완을 실현하세요.

## 一、핵심 창작 원칙

### 1. 절대적 중복 제거와 정확한 보완
- 내용 완전 차별화: 새 이벤트는 1라운드 이벤트와 이벤트 주제, 구체적 장면, 참여 인물, 시간 배치 등 모든 면에서 명확히 달라야 하며, 어떤 형태의 유사성이나 변형 반복도 금지.
- 1라운드 공백 채우기: 1라운드에서 다루지 않은 인물 프로파일 정보, 생활 영역, 이벤트 유형 우선 포함.

### 2. 보완 방향 가이드

#### (1) 충분히 활용되지 않은 프로파일 정보
- 1라운드에서 다루지 않은 프로파일 세부 사항 심층 탐구 (특정 기술, 취미, 사교 관계, 가족 배경, 직업 목표 등)

#### (2) 공휴일 및 계절 특색
- 법정 공휴일: 설날, 삼일절, 어린이날, 추석, 광복절, 개천절, 크리스마스 등 특색 이벤트
- 민간 기념일: 발렌타인데이, 화이트데이, 빼빼로데이 등
- 계절 특성: 봄 나들이, 여름 피서, 가을 단풍 구경, 겨울 관련 이벤트

#### (3) 카테고리 다양성 심화
- Career/Education: 프로젝트 단계별 작업, 직업 인증 등
- Relationships: 다양한 사교 서클 (동료, 친구, 친척, 이웃 등)
- Family&Living Situation: 가정 유지 관리, 거주 환경 개선
- Health: 예방 건강 관리, 건강 검진

## 二、이벤트 세부 요구사항

- 형식 규범: 모든 이벤트에 구체적 시간 표시 (2025-XX-XX 또는 2025-XX-XX 至 2025-XX-XX)
- 균등 분포: 이벤트는 전년도 각 월에 균등하게 분포
- 합리적 중복: 1라운드 이벤트와 시간 중복 허용, 단 현실 논리 부합 필수
- 표준 카테고리 엄격 사용: Career, Education, Relationships, Family&Living Situation, Personal Life, Finance, Health, Unexpected Events, Other
- 총 수량: 목표 약 90개

## 三、출력 형식

<type>Career（N개）</type>
    이벤트명 (2025-MM-DD): 설명.

(카테고리 제목과 이벤트 목록만 출력, 추가 설명 금지)

---

인물 프로파일: {summary}

1라운드 이벤트 목록:
{r1_text}"""

        events_r2: list[dict] = []
        try:
            resp = llm.invoke([HumanMessage(content=r2_prompt)])
            events_r2 = _parse_yearterm_output(resp.content)
            print(f"[draft_gen] {name}: Round 2 → {len(events_r2)} events parsed")
        except Exception as e:
            print(f"[draft_gen] {name}: Round 2 error: {e}")

        all_events = events_r1 + events_r2
        if not all_events:
            all_events = list(_DEFAULT_EVENTS)
            print(f"[draft_gen] {name}: using fallback events")

        # ── Round 3: template_extract_impact_events ────────────────────────────
        print(f"[draft_gen] {name}: Round 3 — extract_impact_events...")

        events_text = "\n".join(
            f"  {e['date']}: [{e['category']}] {e['title']} — {e['description']}"
            for e in all_events[:80]
        )

        r3_prompt = f"""인물 프로파일과 이벤트 목록을 기반으로, **두 가지 유형의 중요 이벤트만 추출**하세요:
1. **프로파일 정보에 현저한 변화를 가져오는 이벤트**: 직업/생활의 큰 변동, 직업/교육 이정표, 재무 변화, 건강 변화, 인간 관계 변화, 취미/생활 습관 변화.
2. **최근 생활 습관에 영향을 미치는 이벤트**: 단기 출장/여행, 단기 질병, 연속 야근, 단기 훈련/학습, 가족 단기 변고.

이벤트의 후속 발전을 합리적으로 추론하여 영향을 분석하고, 타임라인 형식으로 모든 핵심 이벤트 노드와 구체적 영향을 출력하세요.

**관련 없는 이벤트는 완전히 무시** (단순 감정 증진, 실질적 변화 없는 사교 활동 등).

## 이벤트 선별 기준

**첫 번째 유형: 프로파일 정보에 현저한 변화를 가져오는 이벤트**
- 직업/생활의 큰 변동 / 직업·교육 이정표 / 재무 변화 / 건강 변화
- 인간 관계 변화 (새 친구, 관계 악화, 큰 변동만) / 취미·생활 습관 변화

**두 번째 유형: 최근 생활 습관에 영향을 미치는 이벤트**
- 단기 출장·여행 / 단기 질병 / 연속 야근
- 단기 훈련·학습 / 가족 단기 변고 (생활 배치 1주 이상 조정 필요)

**엄격히 제외**: 감정만 증진하는 사교 활동, 24시간 내 회복 가능한 일시적 불편, 실질적 영향 없는 일상 활동.

---

인물 프로파일: {summary}

이벤트 목록:
{events_text}"""

        impact_text = ""
        try:
            resp = llm.invoke([HumanMessage(content=r3_prompt)])
            impact_text = resp.content
            print(f"[draft_gen] {name}: Round 3 → impact events extracted ({len(impact_text)} chars)")
        except Exception as e:
            print(f"[draft_gen] {name}: Round 3 error: {e}")

        # ── Round 4: template_monthly_events_analysis × 12 ───────────────────
        print(f"[draft_gen] {name}: Round 4 — monthly_events_analysis × 12...")

        category_summaries_str = _build_category_summaries(all_events)
        monthly_results: list[dict] = []

        for month in range(1, 13):
            month_str = f"2025-{month:02d}"
            monthly_events = [e for e in all_events if e["date"].startswith(month_str)]

            monthly_events_str = "\n".join(
                f"  {e['date']}: [{e['category']}] {e['title']} — {e['description']}"
                for e in monthly_events
            ) or "(이 달에 예정된 이벤트 없음)"

            r4_prompt = f"""핵심 과제: 지정 월의 중요 이벤트와 변화 기록 분석, 이벤트의 영향 유형 식별 (고정 변화와 근접 영향)

## 과제 요구사항:

### 1. 월간 이벤트 분석
- 지정 월의 중요 이벤트 선별·분석 (직업, 생활, 학습, 건강, 사교 등 각 카테고리)
- 월간 관련 카테고리 요약과 결합하여 이벤트 최적화 분석 (이벤트 추가·삭제·수정 가능)
  - 이벤트 발생 날짜 등 속성 조정으로 불합리한 충돌 이벤트 삭제
  - 다양성·연속성 이벤트 추가로 개인의 한 달 중요 이벤트와 변화를 기록
- 이벤트 설명 정확하고 완전하며 타임라인 명확하게 보장

### 2. 영향 분석
- **고정 변화**: 이벤트가 초래하는 장기적 정보 변화 식별. 이후 모든 이벤트가 이와 충돌하는지 고려해야 함.
- **근접 영향**: 이벤트가 초래하는 단기적 생활 상태 변화 식별 (특정 기간)

### 3. 출력 형식 (엄격한 JSON)

{{"month": "{month_str}", "month_overview": "월 전체 개요", "key_events": [{{"time_point": "2025-MM-DD", "event": "이벤트명: 설명", "fixed_changes": ["고정 변화 설명"], "short_term_impacts": [{{"start_date": "2025-MM-DD", "end_date": "2025-MM-DD", "impact_description": "근접 영향 설명"}}]}}]}}

JSON만 출력하세요. 마크다운 없이.

---

인물 프로파일: {summary}

지정 월: {month_str}

해당 월의 이벤트 목록:
{monthly_events_str}

카테고리별 연간 이벤트 요약:
{category_summaries_str}"""

            try:
                resp = llm.invoke([HumanMessage(content=r4_prompt)])
                parsed = _parse_json_safe(resp.content)
                monthly_results.append(parsed)
                key_count = len(parsed.get("key_events", []))
                print(f"[draft_gen] {name}: Round 4 {month_str} → {key_count} key events")
            except Exception as e:
                print(f"[draft_gen] {name}: Round 4 {month_str} error: {e}")
                monthly_results.append({})

        events_from_monthly = _extract_events_from_monthly(monthly_results)

        # ── Round 5: template_timeline_conflict_resolution ─────────────────────
        print(f"[draft_gen] {name}: Round 5 — timeline_conflict_resolution...")

        timelines_list = [
            {
                "topic": mr.get("month", f"2025-{i+1:02d}") + " 타임라인",
                "month_overview": mr.get("month_overview", ""),
                "events": mr.get("key_events", []),
            }
            for i, mr in enumerate(monthly_results)
            if isinstance(mr, dict) and mr.get("key_events")
        ]

        timelines_json = json.dumps(timelines_list, ensure_ascii=False, indent=2)[:4000]

        r5_prompt = f"""다음 여러 주제 타임라인 간의 충돌을 분석하고 충돌을 해결하세요. 각 타임라인의 내용이 합리적이고 타임라인 간 논리 충돌이 없도록 보장하세요. 타임라인을 합병하지 마세요.

## 인물 프로파일
{summary}

## 타임라인 목록
{timelines_json}

## 과제 요구사항
1. **충돌 식별**: 각 타임라인 간의 모순점·불일치 점 분석 (이벤트 시간·장소·활동 배치 직접 충돌, 이벤트 장기 영향으로 인한 충돌)
2. **충돌 처리**: 충돌 이벤트 삭제 또는 화해 이벤트 추가로 해결. 타임라인 독립성 유지.
3. **연간 월별 요약**: 2025년 전체적인 월별 요약 생성

## 출력 형식 (JSON)
{{"conflict_resolution_result": [{{"topic": "타임라인명", "modification_content": "수정 내용", "modification_reason": "수정 이유"}}], "annual_overview": {{"1월": {{"monthly_description": "...", "core_events": ["..."], "core_impacts": ["..."]}}, "2월": {{}}, "12월": {{}}}}}}

JSON만 출력하세요."""

        annual_overview: dict = {}
        try:
            resp = llm.invoke([HumanMessage(content=r5_prompt)])
            r5_result = _parse_json_safe(resp.content)
            annual_overview = r5_result.get("annual_overview", {})
            print(f"[draft_gen] {name}: Round 5 → conflict resolved, {len(annual_overview)} months in overview")
        except Exception as e:
            print(f"[draft_gen] {name}: Round 5 error: {e}")

        # ── Final events ───────────────────────────────────────────────────────
        # Prefer monthly analysis events (refined); fall back to r1+r2 raw events
        final_events = events_from_monthly if events_from_monthly else all_events
        if not final_events:
            final_events = list(_DEFAULT_EVENTS)

        final_events.sort(key=lambda e: e.get("date", ""))
        print(f"[draft_gen] {name}: done — {len(final_events)} final events")

        daily_drafts[name] = final_events

    return {**state, "daily_drafts": daily_drafts}
