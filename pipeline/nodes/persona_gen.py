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
    persons  = state.get("persons", [])   # from person_gen
    count    = max(1, state.get("count", 1))

    # If person_gen ran, use those skeletons; otherwise fall back to count
    if not persons:
        print("[persona_gen] No persons from person_gen — generating from scratch")
        import random
        CITIES = ["서울", "부산", "대구", "인천", "광주", "대전"]
        persons = [
            {"name": f"이일{i+1}", "age": random.randint(25, 50),
             "gender": "남" if i % 2 == 0 else "여", "job": "회사원"}
            for i in range(count)
        ]

    print(f"[persona_gen] Expanding {len(persons)} person(s) into full personas with {provider}...")
    llm = _get_llm(provider, max_tokens=4096)

    import random
    CITIES = ["서울", "부산", "대구", "인천", "광주", "대전"]
    random.shuffle(CITIES)

    # Build a single batch prompt for all persons
    person_list = "\n".join(
        f"{i+1}. {p['name']} ({p['age']}세, {p['gender']}, 직업: {p['job']}, 도시: {CITIES[i % len(CITIES)]})"
        for i, p in enumerate(persons)
    )

    prompt = f"""아래 기본 정보를 기반으로 한국인 페르소나 {len(persons)}명의 JSON 배열을 생성하세요.
기본 정보 (반드시 유지):
{person_list}

각 페르소나는 기본 정보(name/age/gender/job/home_city)를 그대로 유지하고 아래 필드를 추가합니다:
- mbti: MBTI 유형 (나이와 직업에 자연스러운 것으로)
- hobbies: 취미 리스트 (4~8개, 나이와 직업에 어울리는 취미로)
- personality_traits: 성격 특성 (2~4개)
- economic_level: "하"/"중"/"상" (직업과 나이 고려)
- relationships: 10~15명 [{{이름, 관계, 소셜서클}}] (직업/나이대 연령에 맞는 인연)

JSON 배열만 출력하세요. 마크다운 없이."""

    personas = []
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_json(response.content)
        if isinstance(parsed, list):
            personas = parsed
        elif isinstance(parsed, dict):
            personas = [parsed]
        # Ensure basic fields from persons are preserved
        for i, p in enumerate(persons):
            if i < len(personas):
                personas[i]["name"]      = p["name"]
                personas[i]["age"]       = p["age"]
                personas[i]["gender"]    = p["gender"]
                personas[i]["job"]       = p["job"]
                personas[i]["home_city"] = CITIES[i % len(CITIES)]
    except Exception as e:
        print(f"[persona_gen] LLM parse error: {e}. Using defaults.")

    # Pad with defaults if fewer than expected
    while len(personas) < len(persons):
        p = persons[len(personas)]
        d = _default_persona(len(personas))
        d.update({"name": p["name"], "age": p["age"],
                  "gender": p["gender"], "job": p["job"]})
        personas.append(d)

    personas = personas[:len(persons)]

    # Normalize relationship keys (LLM may return Korean or variant key names)
    for p in personas:
        rels = p.get("relationships") or p.get("인간관계") or p.get("relations") or []
        normalized = []
        for r in rels:
            normalized.append({
                "name":          r.get("name") or r.get("이름") or r.get("contact_name") or "이름없음",
                "relation":      r.get("relation") or r.get("관계") or r.get("관계유형") or "",
                "social_circle": r.get("social_circle") or r.get("소셜서클") or r.get("소셜_서클") or "",
            })
        p["relationships"] = normalized

    print(f"[persona_gen] Expanded {len(personas)} persona(s): {[p.get('name', '?') for p in personas]}")

    return {**state, "personas": personas}
