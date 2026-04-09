"""persona_gen node — expands person skeletons into full personas (Step 0).

Three sub-steps mirroring Original LifeBench:
  gen_profile()   → template_refine: enriches profile using refer_kr.json random sampling
  gen_relations() → template_relation_1: generates relationship skeleton list
  gen_persons()   → template_person: fills in contact details

refer_kr.json is the Korean equivalent of LifeBench's refer.json.
Same logic: random_select from each category pool, inject into prompt as hints.
"""
import json
import re
import random
from pathlib import Path

from langchain_core.messages import HumanMessage
from pipeline.full_state import FullPipelineState

CITIES = ["서울", "부산", "대구", "인천", "광주", "대전"]


# ── refer_kr.json 로더 (지연 로드) ───────────────────────────────────
_REFER: dict | None = None
_REFER_PATH = Path(__file__).parent.parent.parent / "data" / "refer_kr.json"


def _load_refer() -> dict:
    global _REFER
    if _REFER is not None:
        return _REFER
    if _REFER_PATH.exists():
        with open(_REFER_PATH, encoding="utf-8") as f:
            _REFER = json.load(f)
    else:
        print(f"[persona_gen] refer_kr.json not found at {_REFER_PATH}")
        _REFER = {}
    return _REFER


def _random_select(category: str, n: int) -> list:
    """Randomly sample n items from refer_kr.json category (same as LifeBench random_select)."""
    pool = _load_refer().get(category, [])
    if not pool:
        return []
    return random.sample(pool, min(n, len(pool)))


def _refer_const() -> str:
    """Build the Ref string injected into gen_profile prompt.

    Mirrors LifeBench PersonaGenerator.refer_const():
      - 취미 12개 샘플 (원본: 兴趣 12개)
      - 목표 후보 6개 (원본: 目标规划 6개)
      - 가치관 후보 6개 (원본: 价値观 6개)
    """
    hobbies = _random_select("취미", 12)
    aims    = _random_select("목표", 6)
    values  = _random_select("가치관", 6)

    ref  = f'"hobbies": {hobbies} — 4~8개를 인물 특성에 맞는 자연스러운 취미로 선택, 다른 취미 1개 추가 가능;\n'
    ref += f'"aim": {aims} — 1~2개 선택 후 구체화(합리적 목표 없으면 생략 가능);\n'
    ref += f'"traits": {values} — 2~4개 선택, 인물에 자연스러운 가치관;\n'
    return ref


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


# ── Sub-step 1: gen_profile (template_refine 기반 + refer_kr.json) ─────────────

def gen_profile(llm, person: dict, home_city: str) -> dict:
    """Generate full profile from basic skeleton.

    Mirrors LifeBench PersonaGenerator.gen_profile():
    - _refer_const()로 취미 12개 / 목표 6개 / 가치관 6개 랜덤 추출 후 프롬프트에 주입
    - template_refine 프롬프트 구조 거의 그대로, 한국어 컨텍스트로만 변경
    """
    ref = _refer_const()

    skeleton_json = json.dumps({
        "name": person["name"], "age": person["age"],
        "gender": person["gender"], "job": person["job"],
        "home_city": home_city,
    }, ensure_ascii=False, indent=2)

    prompt = f"""당신은 인류학자, 작가, 논리 전문가입니다. 당신의 과제는 모델이 생성한 개인 정보 JSON 데이터를 처리하는 것입니다.
목표: 해당 생성 데이터에 비현실적이거나 불합리하거나 논리 오류, 환각, 인물이 풍부하지 않거나, 특징이 너무 대중적인 등의 문제가 있는지 판단하고 수정하여, 진실하고 합리적이며 자기일관적이고 다양한 개인 정보를 생성하는 것입니다.

수정 요구사항:

1. **엄격한 필드 요구사항**:
   - 필드 추가 또는 삭제 불가, 반드시 입력 JSON의 key에 따라 생성.
   - 각 필드에 대해서만 보충 또는 수정하고 원래 JSON 구조 완전성 유지.
   - 모든 key는 영어 유지, value는 한국어 텍스트 또는 표준 숫자 사용.

2. **정보 수정 전략**:
   - 필드 간 논리 불일치 여부 확인.
   - 설명 텍스트에 중대한 논리 오류, 불합리한 부분 확인.
   - 연령/교육배경/직업/직위명/수입 일치, 가족 상황과 연령/직업/수입 부합, 거주 도시/수입/직업/교육 수준 합리적 대응, 취미/생활방식/건강 상태와 연령/직업/가족 조화, 과거 경력과 출생지 및 현 거주지 조율.
   - 현실에서 발생 가능성이 매우 낮거나 비현실적인 상황 확인.
   - 원본 정보를 참조로 삼을 것. 
   - 비설명 필드는 매우 불합리한 경우가 아니면 최대한 수정하지 않을 것.
   - 성격 특성(가치관)/취미 필드는 매우 불합리하지 않으면 삭제하지 않을 것.
   - 취미에는 독특하고 비주류적인 항목 하나 추가 고려 가능.
   - 성격 특성(가치관)은 추가하지 말 것.
   - 수정 시 취미/목표/MBTI/가치관/설명 등 재점검. 
   - desc 필드의 정보/주제/내용은 최대한 보존, 세부사항 수정만 가능, 정보 삭제 불가, 추가는 가능. 경력 부분은 확장하여 보충 가능.

3. **필드 의미 설명**:
   - name: 한국 이름 / age: 나이 / gender: 성별
   - home_city: 거주 도시 / job: 직위명 및 직무
   - mbti: MBTI 유형
   - hobbies: 취미 리스트 (4~8개)
   - personality_traits: 성격 특성 및 가치관
   - economic_level: 소득 수준 ("하"/"중"/"상")
   - education: 학력 / marital_status: 혼인 상태
   - health_desc: 건강 상태/만성질환/운동 습관
   - lifestyle_desc: 여가/미디어/운동/일상 생활 습관
   - work_desc: 직장 관련 설명
   - description: 종합 설명
   - relation: 사교 관계 (다음 단계에서 채움)

4. **형식 정리**:
   - 숫자 필드: 표준 타입(int 또는 float). 날짜: YYYY-MM-DD.
   - 텍스트: 자연스럽고 공식적이며 읽기 쉬어야 함.

5. **출력 요구사항**:
   - JSON 형식만 출력. ```json``` 태그 없이 직접 JSON 괄호로 시작.
   - 필드 완전성 유지, 필드 추가/삭제 불가.
   - 필드 간 논리 일치, 전체 자기일관성 보장.
   - 출력의 "relation" 필드 값은 빈 []. "note" 필드 추가하여 수정 내용/이유 기록.

참조 리스트(Ref 힌트 — 이 중에서 자연스럽게 선택):
{ref}
처리할 원본 개인 정보 JSON 데이터:
{skeleton_json}

요구사항에 따라 완전히 보완·정제된 JSON을 생성하세요."""

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
    - Changed: Chinese to Korean social circles
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
      1. gen_profile()   - template_refine: enrich profile fields
      2. gen_relations() - template_relation_1: generate relationship list
      3. gen_persons()   - template_person: fill contact details
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
