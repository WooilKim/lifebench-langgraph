"""person_gen node — generates basic person records (name, age, gender, job).

This is the first step of the pipeline, equivalent to the original LifeBench's
manually collected person.json. Instead of manual collection, we:
  - Sample Korean names probabilistically from 2020 census surname distribution
  - Generate age randomly across working-age range (18–70)
  - Use LLM to assign a realistic job given the age and gender

No persona details yet — just the skeleton that persona_gen will expand.
"""
import random
from langchain.schema import HumanMessage
from pipeline.full_state import FullPipelineState


# ── 2020 통계청 성씨 분포 (상위 30개, 누적 ~90%) ─────────────────────────────
# Source: 통계청 인구총조사 2020
SURNAME_WEIGHTS = [
    ("김", 19.28), ("이", 14.14), ("박", 8.44), ("최", 4.70), ("정", 4.38),
    ("강", 2.31), ("조", 2.12), ("윤", 2.08), ("장", 2.00), ("임", 1.72),
    ("한", 1.38), ("오", 1.26), ("서", 1.25), ("신", 1.24), ("권", 1.21),
    ("황", 1.12), ("안", 1.04), ("송", 1.02), ("류", 0.87), ("전", 0.85),
    ("홍", 0.76), ("고", 0.75), ("문", 0.73), ("양", 0.72), ("손", 0.71),
    ("배", 0.63), ("백", 0.60), ("허", 0.55), ("유", 0.54), ("남", 0.52),
]
_SURNAMES = [s for s, _ in SURNAME_WEIGHTS]
_WEIGHTS  = [w for _, w in SURNAME_WEIGHTS]

# ── 자주 쓰이는 남녀 이름 음절 (현대 한국 이름) ──────────────────────────────
_GIVEN_MALE = [
    "준혁", "민준", "서준", "지호", "현우", "태양", "도윤", "주원", "시우", "예준",
    "건우", "민재", "지훈", "성민", "재원", "우진", "동현", "승준", "태민", "재민",
    "진우", "성준", "현준", "민호", "재혁", "영재", "태준", "지민", "상훈", "동훈",
]
_GIVEN_FEMALE = [
    "서연", "지우", "서현", "민지", "지아", "채원", "수빈", "하은", "지유", "예린",
    "수아", "지원", "소현", "나현", "민서", "하늘", "지영", "수연", "예지", "은지",
    "혜진", "소연", "유진", "은서", "다은", "미래", "채빈", "지선", "아름", "나연",
]

# ── 나이대별 가능한 직업 목록 ─────────────────────────────────────────────────
AGE_JOB_MAP = {
    (18, 22): [  # 대학생/초기 취업
        "대학생", "아르바이트생", "편의점 직원", "카페 바리스타",
    ],
    (23, 27): [  # 신입/주니어
        "신입 개발자", "마케터", "영업 사원", "회사원", "은행 창구 직원",
        "물류 직원", "콜센터 상담원", "간호사", "초등학교 교사",
    ],
    (28, 35): [  # 주니어~미드레벨
        "소프트웨어 개발자", "UI/UX 디자이너", "마케터", "회계사", "간호사",
        "중학교 교사", "공무원", "영업 관리자", "HR 담당자", "의사",
        "약사", "변호사", "스타트업 창업자", "프리랜서 작가",
    ],
    (36, 45): [  # 시니어/관리직
        "팀장", "부장", "IT 프로젝트 매니저", "병원 과장", "법무팀장",
        "고등학교 교사", "대학 교수", "자영업자", "중소기업 대표",
        "부동산 중개사", "세무사", "건축사",
    ],
    (46, 55): [  # 베테랑
        "이사", "임원", "교장", "대학 교수", "의원급 의사", "변호사",
        "자영업자", "건설 현장 소장", "공기업 부장",
    ],
    (56, 70): [  # 시니어
        "명예 교수", "자영업자", "컨설턴트", "농업인", "은퇴 준비 중",
        "공장 관리자", "사회복지사",
    ],
}


def _sample_name(gender: str) -> str:
    """Sample a Korean name: surname (weighted) + given name (uniform)."""
    surname = random.choices(_SURNAMES, weights=_WEIGHTS, k=1)[0]
    given = random.choice(_GIVEN_MALE if gender == "남" else _GIVEN_FEMALE)
    return surname + given


def _candidate_jobs(age: int) -> list:
    for (lo, hi), jobs in AGE_JOB_MAP.items():
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

    Equivalent to the original LifeBench's manually collected person.json.
    - Names sampled from 2020 census surname distribution
    - Ages distributed across 18–70
    - Jobs assigned by LLM, constrained to age-appropriate candidates
    """
    provider = state.get("provider", "claude")
    count    = max(1, state.get("count", 1))
    llm      = _get_llm(provider)

    # Distribute genders roughly 50/50, shuffle for variety
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
        gender = genders[i]
        age    = random.choice(age_pool)
        name   = _sample_name(gender)
        candidates = _candidate_jobs(age)

        # LLM picks the most natural job given age+gender from candidates
        prompt = (
            f"다음 후보 직업 중에서 {age}세 {gender}성에게 가장 자연스러운 직업 하나만 골라주세요.\n"
            f"후보: {', '.join(candidates)}\n"
            f"직업 이름만 출력하세요. 설명 없이."
        )
        job = candidates[0]  # fallback
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            raw  = resp.content.strip().strip('"').strip("'")
            # Validate: must be one of candidates (fuzzy)
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
