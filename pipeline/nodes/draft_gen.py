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

        prompt = f"""인물 분석을 기반으로 2025년 내 발생 가능한{event_count}개의 생활 이벤트를 생성합니다.

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
- 시간 중복: 합리적 시간 중복 허용 (예: "윈난 여행" 기간 중 "원격 업무 처리").

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
- 장면 부합: 이벤트 발생 시간, 장소, 인물 신분에 부합.
- 간결하게: 각 이벤트 설명은 15-50자.

## 三、출력 형식 요구사항

### 1. 구조화 출력
- 엄격하게 「카테고리 + 이벤트 수」로 그룹화하여 출력, 각 카테고리 독립 단락.
- 각 카테고리 아래 이벤트는 시간 순서로 정렬.
- 카테고리 이름은 위의 9개 카테고리와 완전히 일치.

### 2. 형식 예시
<type>Career（20개）</type>
    업계 고봉 포럼 참가 (2025-03-18 至 2025-03-19): 가족 기업 대표로 업계 트렌드 학습 및 공급업체 자원 확장.
    신제품 출시 기획 (2025-04-01 至 2025-04-15): 여름 신제품 홍보 방안 주도 수립.

### 3. 핵심 제약
- 출력 내용에는 카테고리 제목과 이벤트 목록만 포함, 추가 분석/설명 텍스트 금지.
- 각 이벤트는 반드시 「이벤트명（시간）: 설명」 형식으로 제시.

---

인물 분석: {name}({age}세, {job}, {city} 거주), 취미: {hobbies}

형식: [{{"date":"YYYY-MM-DD","title":"...","category":"work|family|health|social|hobby|finance","description":"..."}}]
JSON 배열만 출력하세요. 마크다운 없이."""

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
