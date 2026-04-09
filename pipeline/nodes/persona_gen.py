"""persona_gen node — generates Korean personas via LLM (Step 0)."""
import json
import re

from langchain.schema import HumanMessage

from pipeline.full_state import FullPipelineState


# ── LLM helper ────────────────────────────────────────────────────────────────

def _get_llm(provider: str, max_tokens: int = 4096):
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
    """Strip markdown fences and parse JSON."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


# ── Fallback default persona ───────────────────────────────────────────────────

def _default_persona(idx: int) -> dict:
    defaults = [
        {
            "name": "김민준",
            "age": 32,
            "gender": "남",
            "job": "소프트웨어 개발자",
            "mbti": "INTP",
            "hobbies": ["게임", "독서", "등산", "요리"],
            "personality_traits": ["분석적", "내향적", "논리적"],
            "economic_level": "중",
            "home_city": "서울",
            "relationships": [
                {"name": "이지연", "relation": "여자친구", "social_circle": "연인"},
                {"name": "박성호", "relation": "친구", "social_circle": "대학 동기"},
                {"name": "김부장", "relation": "상사", "social_circle": "직장"},
                {"name": "최팀장", "relation": "팀장", "social_circle": "직장"},
                {"name": "정대리", "relation": "동료", "social_circle": "직장"},
                {"name": "김엄마", "relation": "어머니", "social_circle": "가족"},
                {"name": "김아빠", "relation": "아버지", "social_circle": "가족"},
                {"name": "김동생", "relation": "동생", "social_circle": "가족"},
                {"name": "이현우", "relation": "친구", "social_circle": "고등학교 동기"},
                {"name": "서준호", "relation": "친구", "social_circle": "동네"},
            ],
        },
        {
            "name": "이수진",
            "age": 28,
            "gender": "여",
            "job": "마케터",
            "mbti": "ENFJ",
            "hobbies": ["요가", "카페 탐방", "여행", "사진 촬영", "쇼핑"],
            "personality_traits": ["외향적", "창의적", "공감 능력"],
            "economic_level": "중",
            "home_city": "부산",
            "relationships": [
                {"name": "오지훈", "relation": "남자친구", "social_circle": "연인"},
                {"name": "박지현", "relation": "친구", "social_circle": "대학 동기"},
                {"name": "강과장", "relation": "상사", "social_circle": "직장"},
                {"name": "한대리", "relation": "동료", "social_circle": "직장"},
                {"name": "이팀장", "relation": "팀장", "social_circle": "직장"},
                {"name": "이엄마", "relation": "어머니", "social_circle": "가족"},
                {"name": "이아빠", "relation": "아버지", "social_circle": "가족"},
                {"name": "이언니", "relation": "언니", "social_circle": "가족"},
                {"name": "최민서", "relation": "친구", "social_circle": "고등학교 동기"},
                {"name": "윤채원", "relation": "친구", "social_circle": "직장 외"},
            ],
        },
    ]
    return defaults[idx % len(defaults)]


# ── Node ──────────────────────────────────────────────────────────────────────

def generate_personas(state: FullPipelineState) -> FullPipelineState:
    """Step 0: Generate `count` Korean personas via LLM."""
    provider = state.get("provider", "claude")
    count = max(1, state.get("count", 1))

    print(f"[persona_gen] Generating {count} persona(s) with {provider}...")

    llm = _get_llm(provider, max_tokens=4096)

    CITIES  = ["서울", "부산", "대구", "인천", "광주", "대전"]
    JOBS    = ["소프트웨어 개발자", "간호사", "교사", "마케터", "요리사", "디자이너",
               "회계사", "영업사원", "경찰관", "의사", "작가", "사진작가", "유튜버",
               "물리치료사", "건축가", "공무원", "약사", "번역가", "인테리어 디자이너"]
    import random
    assigned_cities = random.sample(CITIES * 3, min(count, len(CITIES * 3)))
    assigned_jobs   = random.sample(JOBS,   min(count, len(JOBS)))
    diversity_hint  = "\n".join(
        f"- 페르소나 {i+1}: 도시={assigned_cities[i]}, 직업={assigned_jobs[i]}, 나이={random.randint(22,55)}, 성별={'남' if i%2==0 else '여'}"
        for i in range(count)
    )

    prompt = f"""다음 조건으로 한국인 페르소나 {count}명의 JSON 배열을 생성하세요.

각 페르소나 필드:
- name: 한국 이름 (문자열)
- age: 정수
- gender: "남" 또는 "여"
- job: 직업
- mbti: MBTI 유형
- hobbies: 취미 리스트 (4~8개)
- personality_traits: 성격 특성 리스트 (2~4개)
- economic_level: "하", "중", "상" 중 하나
- home_city: 문자열
- relationships: 10~15명 리스트, 각 항목은 {{"name": "...", "relation": "...", "social_circle": "..."}}

아래 다양성 조건을 반드시 따르세요 (각 번호가 각 페르소나에 해당):
{diversity_hint}

JSON 배열만 출력하세요. 마크다운, 설명 없이."""

    personas = []
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_json(response.content)
        if isinstance(parsed, list):
            personas = parsed
        elif isinstance(parsed, dict):
            personas = [parsed]
    except Exception as e:
        print(f"[persona_gen] LLM parse error: {e}. Using defaults.")

    # Pad with defaults if LLM returned fewer than requested
    while len(personas) < count:
        personas.append(_default_persona(len(personas)))

    personas = personas[:count]
    print(f"[persona_gen] Generated {len(personas)} persona(s): {[p.get('name', '?') for p in personas]}")

    return {**state, "personas": personas}
