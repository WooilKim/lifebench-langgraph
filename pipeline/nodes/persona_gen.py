"""persona_gen node — expands person skeletons into full personas (Step 0).

Three sub-steps mirroring Original LifeBench:
  gen_profile()   → template_refine equivalent: enrich basic profile fields
  gen_relations() → template_relation_1 equivalent: generate relationship list
  gen_persons()   → template_person equivalent: fill in contact details

Prompts are adapted from Original LifeBench (template_refine, template_relation_1,
template_person) with minimal changes for Korean context.
"""
import json
import re
import random

from langchain_core.messages import HumanMessage
from pipeline.full_state import FullPipelineState

CITIES = ["서울", "부산", "대구", "인천", "광주", "대전"]


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
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


# ── Sub-step 1: gen_profile (template_refine 기반) ────────────────────────────

def gen_profile(llm, person: dict, home_city: str) -> dict:
    """Generate full profile fields from basic person skeleton.

    Adapted from LifeBench template_refine:
    - Kept: logic validation, field enrichment approach
    - Changed: Chinese → Korean context, fields adapted to our schema
    """
    skeleton_json = json.dumps({
        "name":      person["name"],
        "age":       person["age"],
        "gender":    person["gender"],
        "job":       person["job"],
        "home_city": home_city,
    }, ensure_ascii=False)

    prompt = f"""당신은 인류학자, 작가, 논리 전문가입니다. 아래 기본 인물 정보를 바탕으로 풍부하고 현실적인 한국인 프로파일을 생성하세요.

기본 정보 (반드시 유지):
{skeleton_json}

생성할 필드:
- mbti: MBTI 유형 (나이·직업·성별에 자연스러운 것)
- hobbies: 취미 리스트 4~8개 (나이·직업에 어울리는 현실적 취미, 독특한 취미 1개 포함 가능)
- personality_traits: 성격 특성 2~4개
- economic_level: "하"/"중"/"상" (직업·나이 고려)
- education: 최종 학력 (나이·직업에 맞게)
- marital_status: 혼인 상태 (미혼/기혼/이혼/사별)
- health_desc: 건강 상태 간단 설명 (1~2문장)
- lifestyle_desc: 일상 생활 패턴 간단 설명 (1~2문장)
- work_desc: 직장 관련 설명 (1~2문장)
- description: 인물 종합 설명 (3~5문장, 성격·생활·특징 포함)
- relation: [] (빈 배열, 다음 단계에서 채움)

검증 사항:
- 나이/학력/직업/수입 수준 상호 일치
- 혼인 상태는 나이·직업에 자연스럽게
- 취미/생활습관/건강 상태와 나이·직업 조화

JSON만 출력하세요. 기본 정보 필드(name/age/gender/job/home_city)도 포함하세요."""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        profile = _parse_json(resp.content)
        # 기본 정보 강제 유지
        profile["name"]      = person["name"]
        profile["age"]       = person["age"]
        profile["gender"]    = person["gender"]
        profile["job"]       = person["job"]
        profile["home_city"] = home_city
        profile.setdefault("relation", [])
        return profile
    except Exception as e:
        print(f"[persona_gen] gen_profile error for {person['name']}: {e}")
        return {
            "name": person["name"], "age": person["age"],
            "gender": person["gender"], "job": person["job"],
            "home_city": home_city,
            "mbti": "INFP", "hobbies": ["독서", "산책"],
            "personality_traits": ["성실한", "내향적"],
            "economic_level": "중", "education": "대학교 졸업",
            "marital_status": "미혼", "health_desc": "건강한 편",
            "lifestyle_desc": "규칙적인 생활을 선호",
            "work_desc": f"{person['job']} 업무에 충실",
            "description": f"{person['name']}은 {person['age']}세 {person['job']}입니다.",
            "relation": [],
        }


# ── Sub-step 2: gen_relations (template_relation_1 기반) ──────────────────────

def gen_relations(llm, profile: dict) -> list:
    """Generate relationship skeleton list.

    Adapted from LifeBench template_relation_1:
    - Kept: social circle inference, 10~15 contacts
    - Changed: Chinese → Korean social circles
    """
    profile_summary = json.dumps({
        k: profile.get(k) for k in
        ["name", "age", "gender", "job", "home_city", "mbti",
         "education", "marital_status", "hobbies", "work_desc"]
    }, ensure_ascii=False)

    prompt = f"""당신은 인류학자, 스토리 작가, 소셜 네트워크 전문가입니다.
아래 인물 프로파일을 분석하여 이 인물이 연락을 유지하는 사람 10~15명의 관계망을 생성하세요.

인물 프로파일:
{profile_summary}

소셜 서클 예시: 가족, 직장, 대학 동기, 고등학교 동기, 동네, 취미 모임, 연인

각 관계는 다음 형식의 JSON 배열로 출력:
[{{"name": "이름", "relation": "관계", "social_circle": "소셜서클"}}]

조건:
- 이름은 한국인 이름
- 직업·나이·생활에 어울리는 관계 구성
- 가족(부모·형제 등), 직장 동료, 친구 등 다양하게 포함
- 같은 서클의 사람들은 서로 연결될 수 있음

JSON 배열만 출력하세요."""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        relations = _parse_json(resp.content)
        if isinstance(relations, list):
            return relations
    except Exception as e:
        print(f"[persona_gen] gen_relations error for {profile['name']}: {e}")

    # fallback
    return [
        {"name": "부모님", "relation": "부모", "social_circle": "가족"},
        {"name": "직장동료", "relation": "동료", "social_circle": "직장"},
        {"name": "대학친구", "relation": "친구", "social_circle": "대학 동기"},
    ]


# ── Sub-step 3: gen_persons (template_person 기반) ────────────────────────────

def gen_persons(llm, profile: dict, relations: list) -> list:
    """Fill in contact details for each relationship.

    Adapted from LifeBench template_person:
    - Kept: per-contact detail generation, social circle coherence
    - Changed: fields adapted to Korean context
    """
    relations_json = json.dumps(relations, ensure_ascii=False)
    profile_summary = json.dumps({
        k: profile.get(k) for k in
        ["name", "age", "gender", "job", "home_city", "mbti", "marital_status"]
    }, ensure_ascii=False)

    prompt = f"""당신은 인류학자, 스토리 작가, 데이터 전문가입니다.
아래 인물의 관계 목록에 있는 각 연락처에 대해 상세 정보를 생성하세요.

주인공 프로파일:
{profile_summary}

관계 목록:
{relations_json}

각 연락처에 대해 다음 필드를 포함한 JSON 배열 생성:
- name: 이름 (관계 목록의 이름 유지)
- relation: 관계 (유지)
- social_circle: 소셜서클 (유지)
- age: 나이 (주인공과의 관계에 맞는 나이)
- gender: 성별 (이름·직업에 일치)
- job: 직업
- home_city: 거주 도시 (서울/부산/대구/인천/광주/대전 중 하나, 일부는 다른 도시 가능)
- description: 이 사람과의 관계·만남 방식·연락 빈도 (1~2문장)

조건:
- 같은 소셜서클 사람들 간 정보 일관성 유지
- 나이는 주인공과 관계에 따라 자연스럽게
- 온라인 친구나 오래 못 만난 친구는 다른 도시에 있을 수 있음

JSON 배열만 출력하세요."""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        persons = _parse_json(resp.content)
        if isinstance(persons, list):
            return persons
    except Exception as e:
        print(f"[persona_gen] gen_persons error for {profile['name']}: {e}")

    # fallback: return relations as-is with minimal fields
    return [
        {**r, "age": 30, "gender": "남", "job": "회사원",
         "home_city": profile.get("home_city", "서울"),
         "description": f"{profile['name']}의 {r['relation']}"}
        for r in relations
    ]


# ── Node ──────────────────────────────────────────────────────────────────────

def generate_personas(state: FullPipelineState) -> FullPipelineState:
    """Step 0: Expand person skeletons into full personas.

    Three sub-steps (mirrors Original LifeBench):
      1. gen_profile()   — template_refine: enrich profile fields
      2. gen_relations() — template_relation_1: generate relationship list
      3. gen_persons()   — template_person: fill contact details
    """
    provider = state.get("provider", "claude")
    persons  = state.get("persons", [])
    count    = max(1, state.get("count", 1))

    if not persons:
        print("[persona_gen] No persons from person_gen — using fallback skeletons")
        persons = [
            {"name": f"김민준{i+1}", "age": 30 + i, "gender": "남", "job": "회사원"}
            for i in range(count)
        ]

    llm = _get_llm(provider, max_tokens=4096)
    city_pool = CITIES * ((len(persons) // len(CITIES)) + 1)
    random.shuffle(city_pool)

    personas = []
    for i, person in enumerate(persons):
        home_city = city_pool[i]
        name = person["name"]
        print(f"[persona_gen] {i+1}/{len(persons)}: {name} — gen_profile...")

        # Sub-step 1: gen_profile
        profile = gen_profile(llm, person, home_city)

        # Sub-step 2: gen_relations
        print(f"[persona_gen] {i+1}/{len(persons)}: {name} — gen_relations...")
        relations = gen_relations(llm, profile)

        # Sub-step 3: gen_persons
        print(f"[persona_gen] {i+1}/{len(persons)}: {name} — gen_persons...")
        contacts = gen_persons(llm, profile, relations)

        # Merge contacts into profile.relationships
        profile["relationships"] = []
        for c in contacts:
            profile["relationships"].append({
                "name":          c.get("name") or c.get("이름") or "이름없음",
                "relation":      c.get("relation") or c.get("관계") or "",
                "social_circle": c.get("social_circle") or c.get("소셜서클") or "",
                "age":           c.get("age"),
                "gender":        c.get("gender"),
                "job":           c.get("job"),
                "home_city":     c.get("home_city"),
                "description":   c.get("description", ""),
            })

        personas.append(profile)
        print(f"[persona_gen] {i+1}/{len(persons)}: {name} — done ({len(profile['relationships'])} contacts)")

    return {**state, "personas": personas}
