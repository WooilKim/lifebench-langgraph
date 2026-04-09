"""phone_data_gen node — generates synthetic phone data from daily events (Step 3).

Rule-based: no LLM calls. Generates call/sms/push records from event context.
Event IDs use an integer scheme so sms_gen/noti_gen can parse them without hashing:
  - calls:  persona_idx * 100_000 + call_seq        (0–9 999)
  - sms:    persona_idx * 100_000 + 10_000 + sms_seq (10_000–19_999)
  - push:   persona_idx * 100_000 + 20_000 + noti_seq(20_000–29_999)
This lets the multi_formatter reconstruct per-persona identifiers deterministically.
"""
import hashlib
import random
from datetime import datetime, timedelta

from pipeline.full_state import FullPipelineState


# ── Phone number helper ────────────────────────────────────────────────────────

def _gen_phone(name: str) -> str:
    """Deterministic fake Korean mobile number from contact name."""
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16)
    part1 = h % 10000
    part2 = (h >> 16) % 10000
    return f"010-{part1:04d}-{part2:04d}"


# ── Korean app definitions ─────────────────────────────────────────────────────

# (app_name, category_hint)
_KOREAN_APPS = [
    ("카카오톡", "social"),
    ("네이버", "news"),
    ("유튜브", "entertainment"),
    ("인스타그램", "social"),
    ("토스", "finance"),
    ("카카오페이", "finance"),
    ("배달의민족", "food"),
    ("쿠팡", "shopping"),
    ("네이버페이", "finance"),
    ("카카오맵", "navigation"),
    ("네이버지도", "navigation"),
    ("멜론", "music"),
    ("스포티파이", "music"),
    ("왓챠", "entertainment"),
    ("넷플릭스", "entertainment"),
    ("당근마켓", "shopping"),
    ("무신사", "shopping"),
    ("클래스101", "education"),
    ("삼성페이", "finance"),
    ("국민은행", "finance"),
    ("신한은행", "finance"),
    ("하나은행", "finance"),
    ("롯데ON", "shopping"),
    ("11번가", "shopping"),
    ("네이버 블로그", "social"),
    ("트위터", "social"),
    ("페이스북", "social"),
]

# Hobby → preferred apps mapping
_HOBBY_APP_MAP: dict = {
    "독서": ["밀리의서재", "리디북스", "예스24"],
    "게임": ["카카오게임즈", "넥슨", "스팀"],
    "요리": ["만개의레시피", "배달의민족", "마켓컬리"],
    "등산": ["네이버지도", "트랭글", "카카오맵"],
    "여행": ["에어비앤비", "야놀자", "여기어때"],
    "사진": ["인스타그램", "VSCO", "포토북"],
    "음악": ["멜론", "스포티파이", "유튜브"],
    "영화": ["왓챠", "넷플릭스", "씨네21"],
    "운동": ["나이키런클럽", "카카오헬스케어", "스포타임"],
    "쇼핑": ["쿠팡", "무신사", "당근마켓"],
    "카페": ["네이버지도", "망고플레이트", "카카오맵"],
    "요가": ["클래스101", "네이버 블로그", "유튜브"],
}

# Category → message templates (Korean)
_SOCIAL_MSG_TEMPLATES = [
    "오늘 시간 있어? 밥 한번 먹자!",
    "저번에 얘기한 거 어떻게 됐어?",
    "주말에 뭐 해? 같이 나가자",
    "오늘 회식이래. 올 수 있어?",
    "생일 축하해! 🎂",
    "오랜만이야~ 잘 지내지?",
    "그때 빌려준 거 다음에 줄게!",
    "지금 어디야? 나 근처야",
    "주말에 모임 있는데 참석 가능해?",
    "사진 보냈어. 확인해봐!",
]

_WORK_MSG_TEMPLATES = [
    "내일 오전 회의 준비됐어요?",
    "보고서 언제까지 될 것 같아요?",
    "방금 메일 확인했어요?",
    "오늘 오후에 클라이언트 미팅이에요",
    "업무 관련해서 잠깐 통화 가능해요?",
    "퇴근 후 저녁 어때요? 팀 회식해요",
    "자료 공유해 줄 수 있어요?",
    "프로젝트 일정 다시 확인 부탁해요",
]

_FAMILY_MSG_TEMPLATES = [
    "밥은 먹었어?",
    "이번 주말 집에 와?",
    "오늘 늦어?",
    "엄마가 보고 싶다",
    "아빠 생신 선물 샀어?",
    "설날에 다들 오는 거지?",
    "건강은 좀 어때?",
    "추석 때 고향 내려올 거야?",
]

# SMS receive templates (counterpart replies)
_RECEIVE_MSG_TEMPLATES = [
    "응, 가능해!",
    "알겠어, 확인해볼게",
    "오늘은 좀 어렵고 다음에 하자",
    "알려줘서 고마워",
    "그래, 거기서 만나자",
    "잠깐만, 지금 확인할게",
    "좋아! 기대된다",
    "ㅋㅋ 맞아맞아",
    "오케이 알겠어!",
]


# ── Push notification content templates ───────────────────────────────────────

_PUSH_TEMPLATES: dict = {
    "카카오톡":    [("새 메시지", "{contact}님이 메시지를 보냈습니다."),
                   ("단톡방 알림", "새 메시지 {n}개"),
                   ("카카오페이 송금", "송금 알림이 도착했습니다.")],
    "네이버":     [("오늘의 뉴스", "주요 뉴스를 확인하세요."),
                   ("실시간 검색어", "지금 인기 검색어를 확인하세요."),
                   ("네이버 포인트", "포인트가 적립되었습니다.")],
    "유튜브":     [("추천 영상", "오늘의 추천 영상을 확인하세요."),
                   ("구독 채널 업로드", "새 영상이 업로드되었습니다.")],
    "토스":       [("거래 내역", "결제 {amount}원"),
                   ("오늘의 금융 정보", "내 신용점수를 확인하세요."),
                   ("혜택 알림", "오늘 사용할 수 있는 혜택이 있어요.")],
    "배달의민족": [("주문 완료", "주문이 접수되었습니다."),
                   ("배달 출발", "음식이 출발했습니다."),
                   ("오늘의 쿠폰", "오늘만 사용 가능한 쿠폰을 확인하세요.")],
    "쿠팡":       [("배송 알림", "상품이 발송되었습니다."),
                   ("오늘의 딜", "오늘의 특가 상품을 확인하세요."),
                   ("배달 완료", "상품이 배달 완료되었습니다.")],
    "카카오페이": [("결제 완료", "{amount}원 결제 완료"),
                   ("머니 입금", "카카오페이 머니 입금 완료")],
    "멜론":       [("추천 음악", "오늘의 추천 플레이리스트"),
                   ("차트 업데이트", "최신 멜론 차트 확인하세요.")],
    "인스타그램": [("좋아요", "회원님 게시물에 좋아요가 달렸습니다."),
                   ("새 팔로워", "새로운 팔로워가 생겼습니다."),
                   ("댓글 알림", "회원님 게시물에 댓글이 달렸습니다.")],
}

_DEFAULT_PUSH = [
    ("알림", "새로운 알림이 있습니다."),
    ("업데이트", "앱이 업데이트되었습니다."),
]


# ── Datetime helpers ───────────────────────────────────────────────────────────

def _to_unix_ms(dt_str: str) -> int:
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp() * 1000)


def _offset_dt(dt_str: str, delta_minutes: int) -> str:
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    dt2 = dt + timedelta(minutes=delta_minutes)
    return dt2.strftime("%Y-%m-%d %H:%M:%S")


# ── Core generation logic ──────────────────────────────────────────────────────

def _pick_contact(relationships: list, prefer_circles: list = None):
    if not relationships:
        return None
    if prefer_circles:
        preferred = [r for r in relationships if r.get("social_circle") in prefer_circles]
        if preferred:
            return random.choice(preferred)
    return random.choice(relationships)


def _gen_calls_for_persona(persona: dict, events: list, base_id: int, rng: random.Random) -> list:
    name = persona.get("name", "")
    relationships = persona.get("relationships", [])
    calls = []
    seq = 0

    for evt in events:
        event_type = evt.get("event_type", "")
        category = evt.get("category", "")  # draft events have category; detailed don't
        desc = evt.get("description", "")
        dt = evt.get("datetime", "")

        if not dt:
            continue

        # Determine if this event should trigger a call
        trigger = False
        circles = []
        if any(kw in event_type for kw in ["회의", "미팅", "업무", "클라이언트"]):
            trigger = rng.random() < 0.35
            circles = ["직장"]
        elif any(kw in event_type for kw in ["모임", "약속", "외출", "친구"]):
            trigger = rng.random() < 0.45
            circles = ["대학 동기", "고등학교 동기", "동네"]
        elif any(kw in event_type for kw in ["가족", "명절", "부모", "생일"]):
            trigger = rng.random() < 0.55
            circles = ["가족"]
        elif any(kw in desc for kw in ["전화", "통화", "연락"]):
            trigger = rng.random() < 0.6
        elif rng.random() < 0.05:  # small baseline
            trigger = True

        if not trigger:
            continue

        contact = _pick_contact(relationships, prefer_circles=circles if circles else None)
        if not contact:
            continue

        direction = rng.choice([0, 1])  # 0=outgoing, 1=incoming
        missed = rng.random() < 0.1
        call_result = "missed" if missed else "connected"
        duration = 0 if missed else rng.randint(30, 900)

        # call starts a few minutes after the event
        call_dt = _offset_dt(dt, rng.randint(-5, 15))
        call_dt_end = _offset_dt(call_dt, duration // 60 + 1) if not missed else call_dt

        event_id = base_id + seq
        calls.append({
            "event_id": event_id,
            "datetime": call_dt,
            "datetime_end": call_dt_end,
            "contactName": contact["name"],
            "phoneNumber": _gen_phone(contact["name"]),
            "direction": direction,
            "call_result": call_result,
            "duration_seconds": duration,
            "persona_name": name,
        })
        seq += 1
        if seq >= 9999:  # guard per-persona range
            break

    return calls


def _gen_sms_for_persona(persona: dict, events: list, base_id: int, rng: random.Random) -> list:
    name = persona.get("name", "")
    relationships = persona.get("relationships", [])
    sms_list = []
    seq = 0

    for evt in events:
        event_type = evt.get("event_type", "")
        desc = evt.get("description", "")
        dt = evt.get("datetime", "")

        if not dt:
            continue

        # SMS trigger conditions
        trigger = False
        circles = []
        templates = _SOCIAL_MSG_TEMPLATES

        if any(kw in event_type for kw in ["회의", "미팅", "업무"]):
            trigger = rng.random() < 0.25
            circles = ["직장"]
            templates = _WORK_MSG_TEMPLATES
        elif any(kw in event_type for kw in ["모임", "약속", "친구", "외출"]):
            trigger = rng.random() < 0.4
            circles = ["대학 동기", "고등학교 동기", "동네"]
            templates = _SOCIAL_MSG_TEMPLATES
        elif any(kw in event_type for kw in ["가족", "명절", "부모"]):
            trigger = rng.random() < 0.45
            circles = ["가족"]
            templates = _FAMILY_MSG_TEMPLATES
        elif rng.random() < 0.04:
            trigger = True

        if not trigger:
            continue

        contact = _pick_contact(relationships, prefer_circles=circles if circles else None)
        if not contact:
            continue

        sms_dt = _offset_dt(dt, rng.randint(-10, 10))
        msg = rng.choice(templates)
        msg_type = rng.choice(["Send", "Receive"])
        body = msg if msg_type == "Send" else rng.choice(_RECEIVE_MSG_TEMPLATES)

        event_id = base_id + seq
        sms_list.append({
            "event_id": event_id,
            "datetime": sms_dt,
            "contactName": contact["name"],
            "phoneNumber": _gen_phone(contact["name"]),
            "message_content": body,
            "message_type": msg_type,
            "persona_name": name,
        })
        seq += 1

        # Generate a reply ~50% of the time
        if rng.random() < 0.5 and seq < 9999:
            reply_dt = _offset_dt(sms_dt, rng.randint(1, 30))
            reply_type = "Receive" if msg_type == "Send" else "Send"
            reply_body = rng.choice(_RECEIVE_MSG_TEMPLATES) if reply_type == "Receive" else rng.choice(templates)
            event_id2 = base_id + seq
            sms_list.append({
                "event_id": event_id2,
                "datetime": reply_dt,
                "contactName": contact["name"],
                "phoneNumber": _gen_phone(contact["name"]),
                "message_content": reply_body,
                "message_type": reply_type,
                "persona_name": name,
            })
            seq += 1

        if seq >= 9999:
            break

    return sms_list


def _get_persona_apps(persona: dict) -> list:
    """Return relevant app names for this persona based on hobbies and job."""
    apps = set()
    # Always include core Korean apps
    for app_name, _ in _KOREAN_APPS[:6]:
        apps.add(app_name)

    for hobby in persona.get("hobbies", []):
        for key, app_list in _HOBBY_APP_MAP.items():
            if key in hobby or hobby in key:
                apps.update(app_list)

    job = persona.get("job", "").lower()
    if any(kw in job for kw in ["금융", "회계", "은행", "투자"]):
        apps.update(["토스", "카카오페이", "삼성페이", "국민은행"])
    if any(kw in job for kw in ["마케팅", "디자인", "광고"]):
        apps.update(["인스타그램", "페이스북", "트위터"])

    return list(apps)[:12]  # keep at most 12 apps


def _gen_push_for_persona(persona: dict, events: list, base_id: int, rng: random.Random) -> list:
    """Generate push notifications — 5-15 per calendar day (not per event)."""
    name = persona.get("name", "")
    relationships = persona.get("relationships", [])
    contact_names = [r["name"] for r in relationships] if relationships else ["친구"]

    apps = _get_persona_apps(persona)
    push_list = []
    seq = 0

    # Group events by date to generate daily push batches
    by_date: dict = {}
    for evt in events:
        dt = evt.get("datetime", "")
        if not dt:
            continue
        day = dt[:10]
        by_date.setdefault(day, []).append(evt)

    for day, day_events in sorted(by_date.items()):
        num_push = rng.randint(5, 15)
        for _ in range(num_push):
            app_name = rng.choice(apps)
            templates = _PUSH_TEMPLATES.get(app_name, _DEFAULT_PUSH)
            title_tmpl, text_tmpl = rng.choice(templates)

            contact = rng.choice(contact_names) if contact_names else "친구"
            amount = rng.randint(1, 100) * 1000

            title = title_tmpl
            text = text_tmpl.format(
                contact=contact,
                n=rng.randint(1, 20),
                amount=f"{amount:,}",
            )

            # Spread across the day (7:00–23:00)
            base_dt = f"{day} 07:00:00"
            push_dt = _offset_dt(base_dt, rng.randint(0, 960))
            push_status = rng.choice(["Read", "Unread"])

            event_id = base_id + seq
            push_list.append({
                "event_id": event_id,
                "datetime": push_dt,
                "source": app_name,
                "title": title,
                "content": text,
                "push_status": push_status,
                "persona_name": name,
            })
            seq += 1
            if seq >= 9999:
                break
        if seq >= 9999:
            break

    return push_list


# ── Pre-compute identifiers for splitting ─────────────────────────────────────

def _compute_identifiers(calls: list, sms_list: list, push_list: list) -> dict:
    """Compute the identifiers that call_gen/sms_gen/noti_gen will produce."""

    def ts(dt_str: str) -> int:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp() * 1000)

    call_ids = set()
    for c in calls:
        t = ts(c["datetime"])
        call_ids.add(f"call_{c['event_id']}_{t}")

    sms_ids = set()
    for s in sms_list:
        mid = int(s["event_id"])  # always int in our scheme
        t = ts(s["datetime"])
        sms_ids.add(f"sms_{mid}_{t}")

    noti_ids = set()
    for n in push_list:
        eid = int(n["event_id"])
        t = ts(n["datetime"])
        noti_ids.add(f"noti_{eid}_{t}")

    return {"call_ids": call_ids, "sms_ids": sms_ids, "noti_ids": noti_ids}


# ── Node ──────────────────────────────────────────────────────────────────────

def generate_phone_data(state: FullPipelineState) -> FullPipelineState:
    """Step 3: Generate synthetic phone data from daily events."""
    personas = state.get("personas", [])
    daily_events_map = state.get("daily_events_map", {})

    all_calls: list = []
    all_sms: list = []
    all_push: list = []
    persona_event_id_map: dict = {}

    for i, persona in enumerate(personas):
        name = persona.get("name", f"persona_{i}")
        events = daily_events_map.get(name, [])

        # Deterministic RNG per persona for reproducibility
        rng = random.Random(hash(name) & 0xFFFFFFFF)

        base_call = i * 100_000
        base_sms  = i * 100_000 + 10_000
        base_push = i * 100_000 + 20_000

        print(f"[phone_data_gen] Generating phone data for {name} ({len(events)} events)...")

        calls    = _gen_calls_for_persona(persona, events, base_call, rng)
        sms_list = _gen_sms_for_persona(persona, events, base_sms, rng)
        push     = _gen_push_for_persona(persona, events, base_push, rng)

        persona_event_id_map[name] = _compute_identifiers(calls, sms_list, push)

        all_calls.extend(calls)
        all_sms.extend(sms_list)
        all_push.extend(push)

        print(f"[phone_data_gen] {name}: {len(calls)} calls, {len(sms_list)} sms, {len(push)} push")

    return {
        **state,
        "raw_calls": all_calls,
        "raw_sms": all_sms,
        "raw_push": all_push,
        "persona_event_id_map": persona_event_id_map,
    }
