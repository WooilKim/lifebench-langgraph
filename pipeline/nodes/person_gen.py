"""person_gen node — generates basic person records (name, age, gender, job).

Equivalent to the original LifeBench's manually collected person.json.
  - Surname: 2020 census distribution (top 30 surnames)
  - Given name: sampled from koreanname.me ranking data (weighted by count)
    → Run scripts/fetch_korean_names.py once to download data/korean_names.json
  - Age: weighted random (25–55 dominant working age)
  - Job: LLM picks from age-appropriate candidate list
"""
import json
import random
from pathlib import Path
from typing import Optional
from langchain_core.messages import HumanMessage
from pipeline.full_state import FullPipelineState


# ── 2020 통계청 성씨 분포 (상위 30개, 누적 ~90%) ──────────────────────────────
# Source: 통계청 인구총조사 2020
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

_FALLBACK_MALE   = ["민준", "서준", "도윤", "예준", "시우", "주원", "지호", "예산", "건우", "성윤"]
_FALLBACK_FEMALE = ["서윤", "서연", "지우", "하윤", "서현", "지아", "예조", "아인", "하아", "나은"]


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
    """Sample a full Korean name: surname (census-weighted) + given name (koreanname.me-weighted)."""
    surname = random.choices(_SURNAMES, weights=_SURNAME_WEIGHTS, k=1)[0]

    data = _load_name_data()
    pool = data.get("male" if gender == "남" else "female", [])
    if pool:
        top     = pool[:500]  # top 500으로 현실적이면서 다양하게
        names   = [n["name"] for n in top]
        weights = [n["count"] for n in top]
        given   = random.choices(names, weights=weights, k=1)[0]
    else:
        given = random.choice(_FALLBACK_MALE if gender == "남" else _FALLBACK_FEMALE)

    return surname + given


# ── 나이대별 가능한 직업 목록 ──────────────────────────────────────────────────
_AGE_JOB_MAP = {
    (18, 22): ["대학생", "아르바이트생", "편의점 직원", "카페 바리스타"],
    (23, 27): ["신입 개발자", "마케터", "영업 사원", "회사원", "은행 창구 직원",
               "물류 직원", "콜센터 상담원", "간호사", "초등학교 교사"],
    (28, 35): ["소프트웨어 개발자", "UI/UX 디자이너", "마케터", "회계사", "간호사",
               "중학교 교사", "공무원", "영업 관리자", "HR 담당자", "의사",
               "약사", "변호사", "스타트업 창업자", "프리랜서 작가"],
    (36, 45): ["팀장", "부장", "IT 프로젝트 매니저", "병원 과장", "법무팀장",
               "고등학교 교사", "대학 교수", "자영업자", "중소기업 대표",
               "부동산 중개사", "세무사", "건축사"],
    (46, 55): ["이사", "임원", "교장", "대학 교수", "의원급 의사", "변호사",
               "자영업자", "건설 현장 소장", "공기업 부장"],
    (56, 70): ["명예 교수", "자영업자", "컨설턴트", "농업인", "은퇴 준비 중",
               "공장 관리자", "사회복지사"],
}


def _candidate_jobs(age: int) -> list:
    for (lo, hi), jobs in _AGE_JOB_MAP.items():
        if lo <= age <= hi:
            return jobs
    return ["자영업자", "프리랜서"]


def _get_llm(provider: str):
    p = (provider or "claude").lower()
    if p == "claude":
        from langchain_anthropic import ChatAnthropic
        from llm.config import CLAUDE_MODEL, ANTHROPIC_API_KEY
        return ChatAnthropic(model=CLAUDE_MODEL, api_key=ANTHROPIC_API_KEY, max_tokens=512)
    if p == "gpt":
        from langchain_openai import ChatOpenAI
        from llm.config import GPT_MODEL, OPENAI_API_KEY
        return ChatOpenAI(model=GPT_MODEL, api_key=OPENAI_API_KEY, max_tokens=512)
    if p == "glm":
        from langchain_openai import ChatOpenAI
        from llm.config import GLM_MODEL, GLM_API_KEY, GLM_BASE_URL
        return ChatOpenAI(model=GLM_MODEL, api_key=GLM_API_KEY, base_url=GLM_BASE_URL, max_tokens=512)
    raise ValueError(f"Unknown provider: {p!r}")


def generate_persons(state: FullPipelineState) -> FullPipelineState:
    """Step -1: Generate basic person skeletons (name, age, gender, job).

    Names combine:
    - Surname: weighted sampling from 2020 census surname distribution
    - Given name: weighted sampling from koreanname.me ranking data (count-weighted)
    Jobs are selected by LLM from age-appropriate candidate pools.
    """
    provider = state.get("provider", "claude")
    count    = max(1, state.get("count", 1))
    llm      = _get_llm(provider)

    # Distribute genders roughly 50/50
    genders = (["남"] * ((count + 1) // 2) + ["여"] * (count // 2))
    random.shuffle(genders)

    # Age distribution: bias towards working age (25–55)
    age_pool = (
        list(range(18, 24)) * 1 +   # 대학생 (6개)
        list(range(24, 40)) * 3 +   # 주니어~미드 (48개, 가중)
        list(range(40, 56)) * 2 +   # 시니어 (32개)
        list(range(56, 71)) * 1     # 노년 (15개)
    )

    persons = []
    for i in range(count):
        gender     = genders[i]
        age        = random.choice(age_pool)
        name       = _sample_name(gender)
        candidates = _candidate_jobs(age)

        # LLM picks the most natural job given age + gender from candidates
        prompt = (
            f"다음 후보 직업 중에서 {age}세 {gender}성에게 가장 자연스러운 직업 하나만 골라주세요.\n"
            f"후보: {', '.join(candidates)}\n"
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
            print(f"[person_gen] LLM error for {name}: {e}. Using fallback.")

        persons.append({
            "name":   name,
            "age":    age,
            "gender": gender,
            "job":    job,
        })
        print(f"[person_gen] {i+1}/{count}: {name} ({age}세 {gender}, {job})")

    return {**state, "persons": persons}
