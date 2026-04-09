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
    - Candidates sourced from:
        · LifeBench person.json survey occupations (Korean-translated)
        · LifeBench refer.json job pool 137개 (Korean-mapped)
        · Original candidates
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

    - Surname: weighted from 2020 census
    - Given name: gender-specific pool from koreanname.me, count-weighted top 500
      Male names → 남 only / Female names → 여 only (never crossed)
    """
    surname = random.choices(_SURNAMES, weights=_SURNAME_WEIGHTS, k=1)[0]
    data    = _load_name_data()
    pool    = data.get("male" if gender == "남" else "female", [])
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
# Sources:
#   - LifeBench person.json 설문 직업 (한국어 번역)
#   - LifeBench refer.json 직업 풀 137개 (한국어 매핑)
#   - 기존 직업 후보
_JOB_TABLE = {
    (18, 22): [
        ("대학생",                  10, 10),
        ("아르바이트생",             5,  5),
        ("카페 바리스타",             2,  4),
        ("편의점 직원",              3,  2),
        ("소셜 마케터 인턴",          1,  3),  # refer.json
        ("유튜버·크리에이터",         2,  3),  # person.json (학생 겸 자유기고가)
        ("배달 라이더",              3,  1),
    ],
    (23, 27): [
        ("신입 개발자",              5,  3),
        ("마케터",                  3,  5),
        ("영업 사원",                5,  3),
        ("은행 창구 직원",            3,  5),
        ("간호사",                  1,  7),
        ("조산사",                  0,  3),  # refer.json 助产士
        ("초등학교 교사",             2,  6),
        ("보험 에이전트",             3,  4),  # person.json 保险代理人
        ("통신 엔지니어",             4,  2),  # refer.json 通信工程技术人员
        ("광고 디자이너",             2,  4),  # refer.json 广告设计人员
        ("번역가",                  3,  4),  # refer.json 翻译
        ("물류 직원",                5,  2),
        ("회사원",                  5,  5),
    ],
    (28, 35): [
        ("소프트웨어 개발자",          7,  3),
        ("컴퓨터 소프트웨어 기술자",    5,  3),  # refer.json 计算机软件技术人员
        ("UI/UX 디자이너",           4,  5),
        ("실내 인테리어 디자이너",      3,  5),  # refer.json 室内装饰设计人员
        ("마케터",                  3,  6),
        ("회계사",                  4,  5),  # refer.json 会计人员
        ("감사 담당자",               3,  4),  # refer.json 审计人员
        ("통계 연구원",               3,  4),  # refer.json 统计人员
        ("경제 계획 담당자",           4,  3),  # refer.json 经济计划人员
        ("간호사",                  1,  8),
        ("중학교 교사",               3,  6),
        ("공무원",                  5,  5),
        ("영업 관리자",               5,  3),
        ("HR 담당자",                2,  6),
        ("의사",                    4,  3),
        ("주치의",                  4,  3),  # person.json 住院医师
        ("수의사",                  3,  2),  # refer.json 兽医
        ("약사",                    3,  5),
        ("변호사",                  4,  4),
        ("공증인",                  3,  3),  # refer.json 公证员
        ("기계 엔지니어",             5,  2),  # person.json 机械工程师
        ("수치제어 조작원",            5,  1),  # person.json 数控机床操作员
        ("스타트업 창업자",            5,  2),
        ("유튜버·스트리머",            3,  4),  # person.json
        ("프리랜서 작가",             3,  4),  # refer.json 文学作家
        ("연구원",                  4,  4),
        ("의류 매장 운영자",           1,  5),  # person.json 精品服饰店店主
        ("동영상 편집자",             2,  4),
    ],
    (36, 45): [
        ("팀장",                    6,  3),
        ("부장",                    6,  2),
        ("IT 프로젝트 매니저",         5,  3),
        ("병원 과장",                 4,  3),
        ("법무팀장",                  4,  3),
        ("고등학교 교사",              3,  5),
        ("대학 교수",                 4,  4),
        ("고등 교육 교사",             4,  4),  # refer.json 高等教育教师
        ("자영업자",                  5,  5),
        ("중소기업 대표",              6,  2),
        ("부동산 중개사",              4,  5),
        ("세무사",                   4,  4),
        ("건축사",                   5,  2),
        ("라디오 아나운서",            2,  4),  # refer.json 播音员
        ("방송 진행자",               2,  4),  # refer.json 节目主持人
        ("항공 승무원",               2,  6),  # refer.json 民用航空维修
        ("운동선수·코치",              3,  3),  # refer.json 运动员/教练员
        ("경제 통계 연구원",           4,  4),  # refer.json 经济学研究人员
        ("언론인 기자",               3,  4),  # refer.json 文字记者
        ("도서관 사서",               3,  5),  # refer.json 图书资料业务人员
        ("사회복지사",                3,  6),
    ],
    (46, 55): [
        ("이사",                    6,  2),
        ("임원",                    6,  2),
        ("회사 직능부서 임원",          6,  3),  # refer.json 企业职能部门经理或主管
        ("교장",                    4,  4),
        ("대학 교수",                 4,  4),
        ("의원급 의사",                5,  3),
        ("변호사",                   5,  3),
        ("법관",                    4,  2),  # refer.json 法官
        ("검사",                    4,  2),  # refer.json 检察官
        ("자영업자",                  5,  5),
        ("건설 현장 소장",             7,  1),
        ("공기업 부장",               6,  2),
        ("입법기관 후보원",             4,  2),  # refer.json 法学研究人员
        ("위생단위 책임자",             4,  4),  # refer.json 卫生单位负责人
        ("도서관장",                  3,  5),
        ("항공기 조종사",              5,  1),  # refer.json 飞行驾驶员
    ],
    (56, 70): [
        ("명예 교수",                 4,  3),
        ("자영업자",                  5,  5),
        ("컨설턴트",                  5,  3),
        ("농업인",                   5,  4),  # person.json 家庭主妇/兼职零工
        ("은퇴 준비 중",               4,  4),
        ("퇴직자",                   4,  4),  # refer.json 退休人员
        ("공장 관리자",               6,  2),
        ("사회복지사",                3,  6),
        ("환경 미화원",               4,  5),  # refer.json 环卫工
        ("문화재 관리사",              3,  4),  # refer.json 文物鉴定和保管人员
        ("전통 악기 연주자",           2,  3),  # refer.json 民族乐器演奏员
        ("스포츠 심판원",              3,  2),  # refer.json 裁判员
        ("시장 상인",                 4,  4),  # refer.json 超市老板
        ("약국 운영자",               3,  5),  # refer.json 药店老板
    ],
}


def _candidate_jobs(age: int, gender: str) -> list:
    """Return jobs sorted by gender-appropriate weight (descending)."""
    for (lo, hi), entries in _JOB_TABLE.items():
        if lo <= age <= hi:
            weighted = [(e[0], e[1] if gender == "남" else e[2]) for e in entries]
            return [e[0] for e in sorted(weighted, key=lambda x: -x[1])]
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
    list(range(18, 24)) * 1 +
    list(range(24, 40)) * 3 +
    list(range(40, 56)) * 2 +
    list(range(56, 71)) * 1
)


def generate_persons(state: FullPipelineState) -> FullPipelineState:
    """Step -1: Generate basic person skeletons (name, age, gender, job).

    Phase 1 — Rule-based (no LLM):
      1. Gender assigned 50/50, shuffled
      2. Surname from 2020 census distribution
      3. Given name from gender-specific koreanname.me pool (count-weighted top 500)
      4. Age from working-age weighted pool

    Phase 2 — LLM:
      5. Job selected from age×gender-appropriate candidate list
         (candidates from person.json survey + refer.json 137 jobs, Korean-mapped)
    """
    provider = state.get("provider", "claude")
    count    = max(1, state.get("count", 1))

    # ── Phase 1 ──────────────────────────────────────────────────────────────
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

    print(f"[person_gen] Phase 1: {[(s['name'], s['age'], s['gender']) for s in skeletons]}")

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    llm     = _get_llm(provider)
    persons = []
    for i, s in enumerate(skeletons):
        candidates = _candidate_jobs(s["age"], s["gender"])
        prompt = (
            f"{s['name']}({s['age']}세, {s['gender']}성)에게 가장 자연스러운 직업 하나를 고르세요.\n"
            f"후보: {', '.join(candidates[:10])}\n"
            f"직업 이름만 출력하세요. 설명 없이."
        )
        job = candidates[0]
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
