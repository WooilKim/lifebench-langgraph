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


# ── Email templates ──────────────────────────────────────────────────────────

_EMAIL_DOMAINS = ["gmail.com", "naver.com", "kakao.com", "daum.net", "company.co.kr"]

_EMAIL_TEMPLATES: dict = {
    "직장": [
        ("[업무] 프로젝트 진행 상황 공유", "안녕하세요. 이번 주 진행 상황을 공유드립니다."),
        ("[공지] 전사 회의 일정 안내", "다음 주 전사 회의가 예정되어 있습니다."),
        ("[결재] 출장 신청서 승인 요청", "출장 신청서를 첨부하오니 검토 부탁드립니다."),
        ("[업무] 보고서 피드백 요청", "보고서 검토 후 피드백 주시면 감사하겠습니다."),
        ("RE: 미팅 일정 조율", "말씀하신 시간 괜찮습니다. 확인 부탁드립니다."),
    ],
    "가족": [
        ("주말 일정 공유", "이번 주말 가족 모임 일정이에요."),
        ("사진 공유", "지난번 가족 여행 사진 보내드려요."),
        ("안부 인사", "잘 지내고 있나요? 건강 챙기세요."),
    ],
    "social": [
        ("[이벤트] 동창회 모임 안내", "다음 달 동창회가 예정되어 있습니다."),
        ("생일 축하해!", "생일 축하한다! 건강하고 행복하게 지내길 바라."),
        ("여행 계획 공유", "다음 여행지 후보 몇 가지 정리해봤어."),
    ],
    "service": [
        ("[알림] 결제 완료 안내", "결제가 정상 처리되었습니다."),
        ("[안내] 배송 현황 업데이트", "주문하신 상품이 발송되었습니다."),
        ("[공지] 서비스 이용 약관 변경 안내", "서비스 이용 약관이 변경되었습니다."),
        ("[쿠폰] 특별 할인 혜택 안내", "회원님께만 드리는 특별 할인 쿠폰입니다."),
        ("뉴스레터", "이번 주 주요 소식을 전달해드립니다."),
    ],
}


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

def _normalize_rel(r: dict) -> dict:
    """Normalize relationship dict keys (LLM may use Korean keys)."""
    return {
        "name":         r.get("name") or r.get("이름") or r.get("contact_name") or "이름없음",
        "relation":     r.get("relation") or r.get("관계") or "",
        "social_circle": r.get("social_circle") or r.get("소셜서클") or r.get("소셜_서클") or "",
    }


def _pick_contact(relationships: list, prefer_circles: list = None):
    if not relationships:
        return None
    relationships = [_normalize_rel(r) for r in relationships]
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


def _gen_emails_for_persona(persona: dict, events: list, base_id: int, rng: random.Random) -> list:
    """Generate emails — 1~4 per calendar day."""
    name = persona.get("name", "")
    job = persona.get("job", "")
    relationships = persona.get("relationships", [])

    # Build contact email map deterministically
    def _contact_email(contact_name: str) -> str:
        import hashlib
        h = int(hashlib.md5(contact_name.encode("utf-8")).hexdigest(), 16)
        domain = _EMAIL_DOMAINS[h % len(_EMAIL_DOMAINS)]
        local = contact_name.replace(" ", "").lower()[:8]
        return f"{local}@{domain}"

    work_contacts = [r["name"] for r in relationships
                     if any(kw in r.get("social_circle", "") for kw in ["직장", "회사", "업무"])]
    family_contacts = [r["name"] for r in relationships
                       if "가족" in r.get("social_circle", "")]
    social_contacts = [r["name"] for r in relationships
                       if r["name"] not in work_contacts + family_contacts]

    emails = []
    seq = 0

    # Group events by date
    by_date: dict = {}
    for evt in events:
        dt = evt.get("datetime", "")
        if not dt:
            continue
        by_date.setdefault(dt[:10], []).append(evt)

    for day in sorted(by_date.keys()):
        num_emails = rng.randint(1, 4)
        for _ in range(num_emails):
            # Choose email category
            roll = rng.random()
            if roll < 0.4 and work_contacts:
                category = "직장"
                from_name = rng.choice(work_contacts)
            elif roll < 0.55 and family_contacts:
                category = "가족"
                from_name = rng.choice(family_contacts)
            elif roll < 0.7 and social_contacts:
                category = "social"
                from_name = rng.choice(social_contacts)
            else:
                category = "service"
                from_name = rng.choice(["no-reply", "noreply", "support", "info"])
                from_name = f"{from_name}@{rng.choice(_EMAIL_DOMAINS[1:])}"

            templates = _EMAIL_TEMPLATES.get(category, _EMAIL_TEMPLATES["service"])
            subject, snippet = rng.choice(templates)

            if category != "service":
                from_addr = _contact_email(from_name)
            else:
                from_addr = from_name

            send_dt = f"{day} {rng.randint(7, 22):02d}:{rng.randint(0, 59):02d}:00"
            event_id = base_id + seq

            emails.append({
                "event_id": event_id,
                "datetime": send_dt,
                "from_address": from_addr,
                "subject": subject,
                "snippet": snippet,
                "persona_name": name,
            })
            seq += 1
            if seq >= 9999:
                break
        if seq >= 9999:
            break

    return emails


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


# ── Node ──────────────────────────────────────────────────────────────────────

def _compute_identifiers(calls: list, sms_list: list, push_list: list, emails: list) -> dict:
    """Compute the identifiers that call_gen/sms_gen/noti_gen/gmail_gen will produce."""

    def ts(dt_str: str) -> int:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp() * 1000)

    call_ids = {f"call_{c['event_id']}_{ts(c['datetime'])}" for c in calls}
    sms_ids  = {f"sms_{s['event_id']}_{ts(s['datetime'])}" for s in sms_list}
    noti_ids = {f"noti_{n['event_id']}_{ts(n['datetime'])}" for n in push_list}
    gmail_ids = {f"gmail_{e['event_id']}_{ts(e['datetime'])}" for e in emails}

    return {
        "call_ids":  call_ids,
        "sms_ids":   sms_ids,
        "noti_ids":  noti_ids,
        "gmail_ids": gmail_ids,
    }


def generate_phone_data(state: FullPipelineState) -> FullPipelineState:
    """Step 3: Generate synthetic phone data from daily events."""
    personas = state.get("personas", [])
    daily_events_map = state.get("daily_events_map", {})

    all_calls: list = []
    all_sms: list = []
    all_push: list = []
    all_emails: list = []
    persona_event_id_map: dict = {}

    for i, persona in enumerate(personas):
        name = persona.get("name", f"persona_{i}")
        events = daily_events_map.get(name, [])

        rng = random.Random(hash(name) & 0xFFFFFFFF)

        base_call  = i * 100_000
        base_sms   = i * 100_000 + 10_000
        base_push  = i * 100_000 + 20_000
        base_email = i * 100_000 + 30_000

        print(f"[phone_data_gen] Generating phone data for {name} ({len(events)} events)...")

        calls    = _gen_calls_for_persona(persona, events, base_call, rng)
        sms_list = _gen_sms_for_persona(persona, events, base_sms, rng)
        push     = _gen_push_for_persona(persona, events, base_push, rng)
        emails   = _gen_emails_for_persona(persona, events, base_email, rng)

        persona_event_id_map[name] = _compute_identifiers(calls, sms_list, push, emails)

        all_calls.extend(calls)
        all_sms.extend(sms_list)
        all_push.extend(push)
        all_emails.extend(emails)

        print(f"[phone_data_gen] {name}: {len(calls)} calls, {len(sms_list)} sms, {len(push)} push, {len(emails)} emails")

    return {
        **state,
        "raw_calls":  all_calls,
        "raw_sms":    all_sms,
        "raw_push":   all_push,
        "raw_emails": all_emails,
        "persona_event_id_map": persona_event_id_map,
    }
