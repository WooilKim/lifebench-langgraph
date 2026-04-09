"""person_gen node — generates basic person records (name, age, gender, job).

Two-phase approach:
  Phase 1 (rule-based, no LLM):
    - Gender: 50/50 random
    - Surname: 2020 census distribution (top 30)
    - Given name: koreanname.me ranking data, gender-matched, count-weighted
                  → Male names only for 남, Female names only for 여
    - Age: weighted random (25–55 dominant)

  Phase 2 (LLM):
    - Job: LLM selects from age × gender-appropriate candidate list
"""
import json
import random
from pathlib import Path
from typing import Optional
from langchain_core.messages import HumanMessage
from pipeline.full_state import FullPipelineState


# ── 2020 통계청 성씨 분포 (상위 30개, 누적 ~90%) ──────────────────────────────
_SURNAME_DATA = [
    ("김", 19.28), ("이", 14.14), ("박", 8.44), ("최", 4.70), ("정", 4.38),
    ("강", 2.31), ("조", 2.12), ("윤", 2.08), ("장", 2.00), ("임", 1.72),
    ("한", 1.38), ("오", 1.26), ("서", 1.25), ("신", 1.24), ("권", 1.21),
    ("황", 1.12), ("안", 1.04), ("송", 1.02), ("류", 0.87), ("전", 0.85),
    ("홍", 0.76), ("고", 0.75), ("문", 0.73), ("양", 0.72), ("손", 0.71),
    ("배", 0.63), ("백", 0.60), ("허", 0.55), ("유", 0.54), ("남", 0.52),
]
_SURNAMES        = [s for s, _ in _SURNAME_DATA]
_SURNAME_WEIGHTS = [w for _, w in _SURNAME_DATA]


# ── koreanname.me 이름 데이터 (지연 로드) ─────────────────────────────────────
_NAME_DATA: Optional[dict] = None
_NAME_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "korean_names.json"

_FALLBACK_MALE   = ["민준", "서준", "도윤", "예준", "시우", "주원", "지호", "건우", "성윤", "태윤"]
_FALLBACK_FEMALE = ["서윤", "서연", "지우", "하윤", "서현", "지아", "하은", "수아", "지유", "예린"]


def _load_name_data() -> dict:
    global _NAME_DATA
    if _NAME_DATA is not None:
        return _NAME_DATA
    if _NAME_DATA_PATH.exists():
        with open(_NAME_DATA_PATH, encoding="utf-8") as f:
            _NAME_DATA = json.load(f)
        print(f"[person_gen] Loaded koreanname.me data: "
              f"{len(_NAME_DATA.get('male', []))}M / {len(_NAME_DATA.get('female', []))}F names")
    else:
        print(f"[person_gen] koreanname.me data not found ({_NAME_DATA_PATH}). "
              "Run scripts/fetch_korean_names.py first. Using fallback names.")
        _NAME_DATA = {"male": [], "female": []}
    return _NAME_DATA


def _sample_name(gender: str) -> str:
    """Sample a full Korean name (no LLM).

    Algorithm:
    - Surname: weighted sampling from 2020 census (김 19.3%, 이 14.1%, 박 8.4%...)
    - Given name: gender-matched pool from koreanname.me
                  (male names → 남, female names → 여, never crossed)
                  top 500 sampled by count weight
    """
    surname = random.choices(_SURNAMES, weights=_SURNAME_WEIGHTS, k=1)[0]

    data = _load_name_data()
    pool = data.get("male" if gender == "남" else "female", [])

    if pool:
        top     = pool[:500]
        names   = [n["name"] for n in top]
        weights = [n["count"] for n in top]
        given   = random.choices(names, weights=weights, k=1)[0]
    else:
        given = random.choice(_FALLBACK_MALE if gender == "남" else _FALLBACK_FEMALE)

    return surname + given


# ── 나이 × 성별 직업 후보 테이블 ──────────────────────────────────────────────
# (job, male_weight, female_weight)
# weight: 해당 성별·나이대에서 이 직업이 선택될 상대적 확률
# 0 = 해당 성별에 자연스럽지 않은 직업 (완전 제외는 아니지만 거의 안 뽑힘)
_JOB_TABLE = {
    (18, 22): [
        ("대학생",         10, 10),
        ("아르바이트생",   5,  5),
        ("카페 바리스타",  2,  4),
        ("편의점 직원",    3,  2),
    ],
    (23, 27): [
        ("신입 개발자",    5,  3),
        ("마케터",         3,  5),
        ("영업 사원",      5,  3),
        ("은행 창구 직원", 3,  5),
        ("간호사",         1,  7),
        ("초등학교 교사",  2,  6),
        ("물류 직원",      5,  2),
        ("회사원",         5,  5),
    ],
    (28, 35): [
        ("소프트웨어 개발자", 7, 3),
        ("UI/UX 디자이너",    4, 5),
        ("마케터",             3, 6),
        ("회계사",             4, 5),
        ("간호사",             1, 8),
        ("중학교 교사",        3, 6),
        ("공무원",             5, 5),
        ("영업 관리자",        5, 3),
        ("HR 담당자",          2, 6),
        ("의사",               4, 3),
        ("약사",               3, 5),
        ("변호사",             4, 4),
        ("스타트업 창업자",    5, 2),
        ("프리랜서 작가",      3, 4),
    ],
    (36, 45): [
        ("팀장",               6, 3),
        ("부장",               6, 2),
        ("IT 프로젝트 매니저", 5, 3),
        ("병원 과장",          4, 3),
        ("법무팀장",           4, 3),
        ("고등학교 교사",      3, 5),
        ("대학 교수",          4, 4),
        ("자영업자",           5, 5),
        ("중소기업 대표",      6, 2),
        ("부동산 중개사",      4, 5),
        ("세무사",             4, 4),
        ("건축사",             5, 2),
    ],
    (46, 55): [
        ("이사",               6, 2),
        ("임원",               6, 2),
        ("교장",               4, 4),
        ("대학 교수",          4, 4),
        ("의원급 의사",        5, 3),
        ("변호사",             5, 3),
        ("자영업자",           5, 5),
        ("건설 현장 소장",     7, 1),
        ("공기업 부장",        6, 2),
    ],
    (56, 70): [
        ("명예 교수",   4, 3),
        ("자영업자",    5, 5),
        ("컨설턴트",    5, 3),
        ("농업인",      5, 4),
        ("은퇴 준비 중", 4, 4),
        ("공장 관리자", 6, 2),
        ("사회복지사",  3, 6),
    ],
}


def _candidate_jobs(age: int, gender: str) -> list:
    """Return ordered job candidates for (age, gender), weighted list for display."""
    for (lo, hi), entries in _JOB_TABLE.items():
        if lo <= age <= hi:
            weights = [(e[0], e[1] if gender == "남" else e[2]) for e in entries]
            # Sort by weight descending, return names only
            return [e[0] for e in sorted(weights, key=lambda x: -x[1])]
    return ["자영업자", "프리랜서"]


def _get_llm(provider: str):
    p = (provider or "claude").lower()
    if p == "claude":
        from langchain_anthropic import ChatAnthropic
        from llm.config import CLAUDE_MODEL, ANTHROPIC_API_KEY
        return ChatAnthropic(model=CLAUDE_MODEL, api_key=ANTHROPIC_API_KEY, max_tokens=128)
    if p == "gpt":
        from langchain_openai import ChatOpenAI
        from llm.config import GPT_MODEL, OPENAI_API_KEY
        return ChatOpenAI(model=GPT_MODEL, api_key=OPENAI_API_KEY, max_tokens=128)
    if p == "glm":
        from langchain_openai import ChatOpenAI
        from llm.config import GLM_MODEL, GLM_API_KEY, GLM_BASE_URL
        return ChatOpenAI(model=GLM_MODEL, api_key=GLM_API_KEY, base_url=GLM_BASE_URL, max_tokens=128)
    raise ValueError(f"Unknown provider: {p!r}")


# ── Age pool ──────────────────────────────────────────────────────────────────
_AGE_POOL = (
    list(range(18, 24)) * 1 +   # 대학생 (6개)
    list(range(24, 40)) * 3 +   # 주니어~미드 (48개)
    list(range(40, 56)) * 2 +   # 시니어 (32개)
    list(range(56, 71)) * 1     # 노년 (15개)
)


def generate_persons(state: FullPipelineState) -> FullPipelineState:
    """Step -1: Generate basic person skeletons (name, age, gender, job).

    Phase 1 — Rule-based (no LLM):
      1. Gender assigned 50/50, shuffled
      2. Surname sampled from 2020 census distribution
      3. Given name from gender-specific koreanname.me pool (count-weighted)
         → Male names only → 남, Female names only → 여 (never crossed)
      4. Age from working-age weighted pool

    Phase 2 — LLM:
      5. Job: LLM selects from age×gender-appropriate candidate list
    """
    provider = state.get("provider", "claude")
    count    = max(1, state.get("count", 1))

    # ── Phase 1: rule-based name/age/gender ─────────────────────────────────────
    genders = (["남"] * ((count + 1) // 2) + ["여"] * (count // 2))
    random.shuffle(genders)

    skeletons = []
    used_names: set = set()
    for i in range(count):
        gender = genders[i]
        age    = random.choice(_AGE_POOL)
        for _ in range(10):
            name = _sample_name(gender)
            if name not in used_names:
                break
        used_names.add(name)
        skeletons.append({"name": name, "age": age, "gender": gender})

    print(f"[person_gen] Phase 1 done: {[(s['name'], s['age'], s['gender']) for s in skeletons]}")

    # ── Phase 2: LLM assigns job ─────────────────────────────────────────
    llm = _get_llm(provider)
    persons = []
    for i, s in enumerate(skeletons):
        candidates = _candidate_jobs(s["age"], s["gender"])
        prompt = (
            f"{s['name']}({s['age']}세, {s['gender']}성)에게 가장 자연스러운 직업 하나를 고르세요.\n"
            f"후보: {', '.join(candidates[:8])}\n"
            f"직업 이름만 출력하세요. 설명 없이."
        )
        job = candidates[0]  # fallback
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            raw  = resp.content.strip().strip('"').strip("'")
            for c in candidates:
                if c in raw or raw in c:
                    job = c
                    break
            else:
                job = raw if raw else candidates[0]
        except Exception as e:
            print(f"[person_gen] LLM error for {s['name']}: {e}. Using fallback.")

        persons.append({"name": s["name"], "age": s["age"], "gender": s["gender"], "job": job})
        print(f"[person_gen] {i+1}/{count}: {s['name']} ({s['age']}세 {s['gender']}, {job})")

    return {**state, "persons": persons}
